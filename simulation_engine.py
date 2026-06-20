
import numpy as np
from typing import Dict, List, Optional, Tuple
from river_channel import RiverChannel, CrossSection, Tributary
from hydrodynamics import Hydrodynamics
from water_quality import WaterQualityModel, WaterQualityParams
from pollution_sources import SourceManager, PointSource, NonPointSource, AccidentalSource, ReleaseType
from scipy.linalg import solve_banded


class RiverSimulation:
    """河流模拟核心引擎"""

    def __init__(self):
        self.channel = RiverChannel()
        self.hydro = None
        self.wq_model = None
        self.source_manager = SourceManager()

        self._initialized = False

    def setup_default_channel(self):
        """设置默认河道"""
        section1 = CrossSection(
            x=0.0, shape='rectangular',
            bottom_width=20.0, slope=0.0005,
            manning_n=0.03, length=5000.0
        )
        section2 = CrossSection(
            x=5000.0, shape='trapezoidal',
            bottom_width=25.0, side_slope=2.0,
            slope=0.0003, manning_n=0.035,
            length=5000.0
        )
        self.channel.add_section(section1)
        self.channel.add_section(section2)

        self.hydro = Hydrodynamics(self.channel)
        self.wq_model = WaterQualityModel()

        self._initialized = True

    def set_water_quality_params(self, **kwargs):
        """设置水质参数"""
        params = WaterQualityParams()
        for key, value in kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)
        self.wq_model = WaterQualityModel(params)

    def add_point_source(self, name: str, x: float, flow_rate: float,
                         bod_conc: float, do_conc: float = 8.0,
                         nh3n_conc: float = 0.0, cod_conc: float = 0.0):
        """添加点源"""
        ps = PointSource(name=name, x=x, flow_rate=flow_rate,
                         bod_conc=bod_conc, do_conc=do_conc,
                         nh3n_conc=nh3n_conc, cod_conc=cod_conc)
        self.source_manager.add_source(ps)

    def add_nonpoint_source(self, name: str, start_x: float, end_x: float,
                            area: float = 1000000.0, bod_load: float = 0.0,
                            nh3n_load: float = 0.0, cod_load: float = 0.0,
                            runoff_coeff: float = 0.3):
        """添加面源"""
        nps = NonPointSource(name=name, start_x=start_x, end_x=end_x,
                             area=area, bod_load=bod_load,
                             nh3n_load=nh3n_load, cod_load=cod_load,
                             runoff_coeff=runoff_coeff)
        self.source_manager.add_source(nps)

    def add_accidental_source(self, name: str, x: float,
                               release_type: str = 'instantaneous',
                               total_mass_bod: float = 1000.0,
                               total_mass_nh3n: float = 100.0,
                               total_mass_cod: float = 500.0,
                               release_duration: float = 3600.0,
                               start_time: float = 0.0,
                               flow_rate: float = 0.5):
        """添加突发泄漏源"""
        rt = ReleaseType.INSTANTANEOUS if release_type == 'instantaneous' else ReleaseType.CONTINUOUS
        acc = AccidentalSource(name=name, x=x, release_type=rt,
                               total_mass_bod=total_mass_bod,
                               total_mass_nh3n=total_mass_nh3n,
                               total_mass_cod=total_mass_cod,
                               release_duration=release_duration,
                               start_time=start_time,
                               flow_rate=flow_rate)
        self.source_manager.add_source(acc)

    def run_steady_simulation(self, Q_upstream: float,
                              initial_bod: float = 2.0,
                              initial_do: float = 8.5,
                              initial_nh3n: float = 0.5,
                              initial_cod: float = 5.0,
                              flow_mode: str = 'uniform',
                              n_grid: int = 100,
                              wq_scheme: str = 'upwind') -> Dict:
        """运行稳态模拟"""
        if not self._initialized:
            self.setup_default_channel()

        if flow_mode == 'uniform':
            hydro_result = self.hydro.uniform_flow(Q_upstream, n_grid)
        else:
            section = self.channel.sections[-1]
            h_control = self.hydro.normal_depth(section, Q_upstream)
            hydro_result = self.hydro.gradually_varied_flow(
                Q_upstream, h_control, control_x='downstream', n_grid=n_grid
            )

        x = hydro_result['x']
        V = hydro_result['V']
        A = hydro_result['A']
        h = hydro_result['h']
        Q = hydro_result['Q']

        sources_list = []

        for ps in self.source_manager.get_point_sources():
            sources_list.append({
                'type': 'point',
                'x': ps.x,
                'flow_rate': ps.flow_rate,
                'concentration': {
                    'bod': ps.bod_conc,
                    'do': ps.do_conc,
                    'nh3n': ps.nh3n_conc,
                    'cod': ps.cod_conc,
                }
            })

        for nps in self.source_manager.get_nonpoint_sources():
            dist_loads = nps.get_distributed_load(x)
            sources_list.append({
                'type': 'nonpoint',
                'start_x': nps.start_x,
                'end_x': nps.end_x,
                'distributed_load': dist_loads,
                'flow_rate': nps.area * nps.runoff_coeff * 0.001,
            })

        for acc in self.source_manager.get_accidental_sources():
            if acc.release_type == ReleaseType.CONTINUOUS:
                acc_flow = acc.flow_rate
                duration = acc.release_duration if acc.release_duration > 0 else 1.0
                sources_list.append({
                    'type': 'point',
                    'x': acc.x,
                    'flow_rate': acc_flow,
                    'concentration': {
                        'bod': acc.total_mass_bod / (acc_flow * duration) if acc_flow > 0 else acc.total_mass_bod / duration,
                        'do': 2.0,
                        'nh3n': acc.total_mass_nh3n / (acc_flow * duration) if acc_flow > 0 else acc.total_mass_nh3n / duration,
                        'cod': acc.total_mass_cod / (acc_flow * duration) if acc_flow > 0 else acc.total_mass_cod / duration,
                    }
                })

        initial_conditions = {
            'bod': initial_bod,
            'do': initial_do,
            'nh3n': initial_nh3n,
            'cod': initial_cod,
        }

        Dx = self.wq_model.params.Dx

        wq_result = self._solve_wq_steady(
            x, V, A, h, Q, Dx, sources_list,
            initial_conditions, wq_scheme
        )

        do_min_idx = np.argmin(wq_result['do'])
        critical_x = x[do_min_idx]
        critical_do = wq_result['do'][do_min_idx]

        result = {
            'x': x,
            'h': h,
            'V': V,
            'Q': Q,
            'A': A,
            'water_level': hydro_result['water_level'],
            'z': hydro_result['z'],
            'bod': wq_result['bod'],
            'do': wq_result['do'],
            'nh3n': wq_result['nh3n'],
            'cod': wq_result['cod'],
            'critical_x': critical_x,
            'critical_do': critical_do,
            'wq_scheme': wq_scheme,
        }

        return result

    def _solve_wq_steady(self, x, V, A, h, Q, Dx, sources,
                          initial_conditions, scheme='upwind') -> Dict:
        """求解稳态水质方程"""
        n = len(x)
        dx = x[1] - x[0]

        K1_per_sec = self.wq_model.params.K1 / 86400.0
        K2_per_sec = self.wq_model.params.K2 / 86400.0
        K_nh3n_per_sec = self.wq_model.params.K_nh3n / 86400.0
        K_cod_per_sec = self.wq_model.params.K_cod / 86400.0
        D_sat = self.wq_model.params.D_O_sat

        nonpoint_bod = np.zeros(n)
        nonpoint_nh3n = np.zeros(n)
        nonpoint_cod = np.zeros(n)

        for src in sources:
            if src['type'] == 'nonpoint':
                loads = src['distributed_load']
                for j in range(n):
                    if src['start_x'] <= x[j] <= src['end_x']:
                        nonpoint_bod[j] += loads.get('bod', np.zeros(n))[j]
                        nonpoint_nh3n[j] += loads.get('nh3n', np.zeros(n))[j]
                        nonpoint_cod[j] += loads.get('cod', np.zeros(n))[j]

        if scheme == 'upwind':
            return self._solve_wq_steady_upwind(
                x, V, A, Q, K1_per_sec, K2_per_sec, K_nh3n_per_sec, K_cod_per_sec,
                D_sat, Dx, sources, nonpoint_bod, nonpoint_nh3n, nonpoint_cod,
                initial_conditions
            )
        else:
            return self._solve_wq_steady_crank_nicolson(
                x, V, A, Q, K1_per_sec, K2_per_sec, K_nh3n_per_sec, K_cod_per_sec,
                D_sat, Dx, sources, nonpoint_bod, nonpoint_nh3n, nonpoint_cod,
                initial_conditions
            )

    def _find_grid_index(self, x_grid: np.ndarray, x_src: float) -> int:
        """找到源位置对应的网格索引（最近的下游网格点）"""
        if x_src <= x_grid[0]:
            return 0
        if x_src >= x_grid[-1]:
            return len(x_grid) - 1
        idx = np.searchsorted(x_grid, x_src)
        return min(idx, len(x_grid) - 1)

    def _solve_wq_steady_upwind(self, x, V, A, Q, K1, K2, K_nh3n, K_cod,
                                 D_sat, Dx, sources,
                                 nonpoint_bod, nonpoint_nh3n, nonpoint_cod,
                                 initial_conditions):
        """上风格式求解"""
        n = len(x)

        bod = np.zeros(n)
        do = np.zeros(n)
        nh3n = np.zeros(n)
        cod = np.zeros(n)

        bod[0] = initial_conditions['bod']
        do[0] = initial_conditions['do']
        nh3n[0] = initial_conditions['nh3n']
        cod[0] = initial_conditions['cod']

        point_source_idx = {}
        for src in sources:
            if src['type'] == 'point':
                idx = self._find_grid_index(x, src['x'])
                if idx not in point_source_idx:
                    point_source_idx[idx] = []
                point_source_idx[idx].append(src)

        for i in range(1, n):
            dx_i = x[i] - x[i - 1]
            u_i = V[i]

            point_src_bod = 0
            point_src_do = 0
            point_src_nh3n = 0
            point_src_cod = 0
            point_src_flow = 0

            if i in point_source_idx:
                for src in point_source_idx[i]:
                    point_src_bod += src['flow_rate'] * src['concentration']['bod']
                    point_src_do += src['flow_rate'] * src['concentration']['do']
                    point_src_nh3n += src['flow_rate'] * src['concentration']['nh3n']
                    point_src_cod += src['flow_rate'] * src['concentration']['cod']
                    point_src_flow += src['flow_rate']

            Q_prev = Q[i - 1] + point_src_flow if point_src_flow > 0 else Q[i - 1]

            if Q_prev > 0:
                bod[i] = (Q[i - 1] * bod[i - 1] + point_src_bod) / Q_prev
                do[i] = (Q[i - 1] * do[i - 1] + point_src_do) / Q_prev
                nh3n[i] = (Q[i - 1] * nh3n[i - 1] + point_src_nh3n) / Q_prev
                cod[i] = (Q[i - 1] * cod[i - 1] + point_src_cod) / Q_prev
            else:
                bod[i] = bod[i - 1]
                do[i] = do[i - 1]
                nh3n[i] = nh3n[i - 1]
                cod[i] = cod[i - 1]

            A_i = A[i] if A[i] > 0 else 1.0
            bod[i] += nonpoint_bod[i] / (A_i * u_i) * dx_i if u_i > 0 else 0
            nh3n[i] += nonpoint_nh3n[i] / (A_i * u_i) * dx_i if u_i > 0 else 0
            cod[i] += nonpoint_cod[i] / (A_i * u_i) * dx_i if u_i > 0 else 0

            if u_i > 0:
                travel_time = dx_i / u_i

                bod[i] *= np.exp(-K1 * travel_time)
                nh3n[i] *= np.exp(-K_nh3n * travel_time)
                cod[i] *= np.exp(-K_cod * travel_time)

                deficit = D_sat - do[i]
                do_change = (K2 * deficit - K1 * bod[i]) * travel_time
                do[i] += do_change
                do[i] = min(do[i], D_sat)
                do[i] = max(do[i], 0.0)

            if Dx > 0 and u_i > 0 and i > 1:
                Pe = u_i * dx_i / Dx if Dx > 0 else 1000
                if Pe < 2:
                    diff_bod = Dx * (bod[i - 1] - 2 * bod[i] + bod[i - 2]) / dx_i ** 2
                    bod[i] += diff_bod * dx_i / u_i
                    bod[i] = max(bod[i], 0.0)

                    diff_do = Dx * (do[i - 1] - 2 * do[i] + do[i - 2]) / dx_i ** 2
                    do[i] += diff_do * dx_i / u_i
                    do[i] = min(do[i], D_sat)
                    do[i] = max(do[i], 0.0)

                    diff_nh3n = Dx * (nh3n[i - 1] - 2 * nh3n[i] + nh3n[i - 2]) / dx_i ** 2
                    nh3n[i] += diff_nh3n * dx_i / u_i
                    nh3n[i] = max(nh3n[i], 0.0)

                    diff_cod = Dx * (cod[i - 1] - 2 * cod[i] + cod[i - 2]) / dx_i ** 2
                    cod[i] += diff_cod * dx_i / u_i
                    cod[i] = max(cod[i], 0.0)

            bod[i] = max(bod[i], 0.0)
            nh3n[i] = max(nh3n[i], 0.0)
            cod[i] = max(cod[i], 0.0)

        return {
            'bod': bod,
            'do': do,
            'nh3n': nh3n,
            'cod': cod,
        }

    def _solve_wq_steady_crank_nicolson(self, x, V, A, Q, K1, K2, K_nh3n, K_cod,
                                         D_sat, Dx, sources,
                                         nonpoint_bod, nonpoint_nh3n, nonpoint_cod,
                                         initial_conditions):
        """Crank-Nicolson隐格式求解稳态对流扩散方程

        对流扩散方程: u * dC/dx = Dx * d²C/dx² - K*C + S(x)
        采用Crank-Nicolson半隐格式逐点递推:
          - 对流项: 上风差分 (稳定)
          - 扩散项: C-N半隐格式 (中心差分, i-1用已知值, i+1用上一迭代值)
          - 衰减项: 半隐式处理 (C^{n+1} 用新值)
        上游: Dirichlet BC  C[0] = C_in
        下游: 自然衰减 (零梯度外推)
        点源通过流量混合法处理
        """
        n = len(x)
        dx = x[1] - x[0]

        bod = np.zeros(n)
        do_arr = np.zeros(n)
        nh3n = np.zeros(n)
        cod = np.zeros(n)

        point_source_idx = {}
        for src in sources:
            if src['type'] == 'point':
                idx = self._find_grid_index(x, src['x'])
                if idx > 0 and idx not in point_source_idx:
                    point_source_idx[idx] = []
                if idx > 0:
                    point_source_idx[idx].append(src)

        components = [
            ('bod', bod, K1, initial_conditions['bod'], nonpoint_bod),
            ('nh3n', nh3n, K_nh3n, initial_conditions['nh3n'], nonpoint_nh3n),
            ('cod', cod, K_cod, initial_conditions['cod'], nonpoint_cod),
        ]

        for comp_name, comp_arr, K_decay, init_val, np_load in components:
            comp_arr[0] = init_val

            for i in range(1, n):
                dx_i = x[i] - x[i - 1]
                u_i = V[i] if V[i] > 1e-10 else 1e-10
                A_i = A[i] if A[i] > 0 else 1.0

                Pe = u_i * dx_i / Dx if Dx > 0 else 1000.0

                if Dx > 0 and Pe < 2.0 and i < n - 1:
                    c_left = Dx / dx_i ** 2
                    c_right = Dx / dx_i ** 2
                    c_adv = u_i / dx_i
                    c_decay = K_decay / 2

                    denom = c_adv + c_left + c_decay
                    numer = (c_adv + c_left) * comp_arr[i - 1] + c_right * comp_arr[i + 1] if i + 1 < n else (c_adv + c_left) * comp_arr[i - 1]

                    if np_load[i] > 0:
                        numer += np_load[i] / A_i

                    comp_arr[i] = numer / denom if denom > 0 else comp_arr[i - 1]
                else:
                    travel = dx_i / u_i
                    comp_arr[i] = comp_arr[i - 1] * np.exp(-K_decay * travel)
                    if np_load[i] > 0:
                        comp_arr[i] += np_load[i] * dx_i / (A_i * u_i)

                if i in point_source_idx:
                    Q_prev = Q[i - 1]
                    ps_mass = 0
                    ps_flow = 0
                    for src in point_source_idx[i]:
                        ps_mass += src['flow_rate'] * src['concentration'][comp_name]
                        ps_flow += src['flow_rate']
                    if Q_prev + ps_flow > 0:
                        comp_arr[i] = (Q_prev * comp_arr[i] + ps_mass) / (Q_prev + ps_flow)

                comp_arr[i] = max(comp_arr[i], 0.0)

            if Dx > 0:
                for iteration in range(5):
                    comp_arr_new = comp_arr.copy()
                    for i in range(1, n - 1):
                        dx_i = x[i] - x[i - 1]
                        u_i = V[i] if V[i] > 1e-10 else 1e-10
                        A_i = A[i] if A[i] > 0 else 1.0

                        Pe = u_i * dx_i / Dx if Dx > 0 else 1000.0
                        if Pe >= 5.0:
                            continue

                        c_diff = Dx / dx_i ** 2
                        c_decay = K_decay
                        weight = min(1.0, 2.0 / (Pe + 1))

                        cn_val = (c_diff * (comp_arr[i - 1] + comp_arr[i + 1])
                                  + (1.0 / dx_i - c_decay) * comp_arr[i]
                                  + np_load[i] / A_i) / (2 * c_diff + 1.0 / dx_i + c_decay)

                        comp_arr_new[i] = (1 - weight) * comp_arr[i] + weight * cn_val
                        comp_arr_new[i] = max(comp_arr_new[i], 0.0)

                    comp_arr_new[0] = init_val
                    if n > 1:
                        comp_arr_new[-1] = comp_arr_new[-2]
                    comp_arr[:] = comp_arr_new

        do_arr[0] = initial_conditions['do']
        for i in range(1, n):
            dx_i = x[i] - x[i - 1]
            u_i = V[i] if V[i] > 1e-10 else 1e-10
            travel = dx_i / u_i

            deficit = D_sat - do_arr[i - 1]
            do_arr[i] = do_arr[i - 1] + (K2 * deficit - K1 * bod[i]) * travel

            if i in point_source_idx:
                Q_prev = Q[i - 1]
                ps_do = 0
                ps_flow = 0
                for src in point_source_idx[i]:
                    ps_do += src['flow_rate'] * src['concentration']['do']
                    ps_flow += src['flow_rate']
                if Q_prev + ps_flow > 0:
                    do_arr[i] = (Q_prev * do_arr[i] + ps_do) / (Q_prev + ps_flow)

            do_arr[i] = max(do_arr[i], 0.0)
            do_arr[i] = min(do_arr[i], D_sat)

        return {
            'bod': bod,
            'do': do_arr,
            'nh3n': nh3n,
            'cod': cod,
        }

    def run_unsteady_simulation(self, Q_upstream: np.ndarray,
                                 t_total: float, dt: float,
                                 initial_bod: float = 2.0,
                                 initial_do: float = 8.5,
                                 initial_nh3n: float = 0.5,
                                 initial_cod: float = 5.0,
                                 n_grid: int = 100,
                                 wq_scheme: str = 'upwind') -> Dict:
        """运行非稳态模拟"""
        if not self._initialized:
            self.setup_default_channel()

        hydro_result = self.hydro.kinematic_wave(Q_upstream, t_total, dt, n_grid)

        x = hydro_result['x']
        t = hydro_result['t']
        Q = hydro_result['Q']
        h = hydro_result['h']
        V = hydro_result['V']
        A = hydro_result['A']

        initial_conditions = {
            'bod': initial_bod,
            'do': initial_do,
            'nh3n': initial_nh3n,
            'cod': initial_cod,
        }

        sources_list = []
        for ps in self.source_manager.get_point_sources():
            sources_list.append({
                'type': 'point',
                'x': ps.x,
                'flow_rate': ps.flow_rate,
                'concentration': {
                    'bod': ps.bod_conc,
                    'do': ps.do_conc,
                    'nh3n': ps.nh3n_conc,
                    'cod': ps.cod_conc,
                }
            })

        for acc in self.source_manager.get_accidental_sources():
            sources_list.append({
                'type': 'accidental',
                'x': acc.x,
                'total_mass_bod': acc.total_mass_bod,
                'total_mass_nh3n': acc.total_mass_nh3n,
                'total_mass_cod': acc.total_mass_cod,
                'start_time': acc.start_time,
                'release_duration': acc.release_duration,
                'is_instantaneous': acc.release_type == ReleaseType.INSTANTANEOUS,
            })

        Dx = self.wq_model.params.Dx
        wq_result = self.wq_model.solve_unsteady_advection_diffusion(
            x, t, V, A, Dx, sources_list, initial_conditions, wq_scheme
        )

        result = {
            'x': x,
            't': t,
            'Q': Q,
            'h': h,
            'V': V,
            'A': A,
            'bod': wq_result['bod'],
            'do': wq_result['do'],
            'nh3n': wq_result['nh3n'],
            'cod': wq_result['cod'],
            'courant_warning': hydro_result.get('courant_warning', False),
        }

        return result

    def check_stability(self, V_max: float, dx: float, dt: float) -> Tuple[bool, float, float]:
        """检查数值稳定性"""
        Dx = self.wq_model.params.Dx
        Co_adv = V_max * dt / dx
        Co_diff = 2 * Dx * dt / dx ** 2

        stable = (Co_adv <= 1.0) and (Co_diff <= 1.0)

        return stable, Co_adv, Co_diff

    def suggest_dt(self, V_max: float, dx: float, safety_factor: float = 0.8) -> float:
        """建议时间步长"""
        Dx = self.wq_model.params.Dx
        dt_adv = dx / V_max if V_max > 0 else float('inf')
        dt_diff = dx ** 2 / (2 * Dx) if Dx > 0 else float('inf')
        dt_min = min(dt_adv, dt_diff)
        return safety_factor * dt_min if dt_min != float('inf') else 3600.0
