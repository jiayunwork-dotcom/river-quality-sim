
import numpy as np
from typing import Dict, List, Optional, Tuple
from river_channel import RiverChannel, CrossSection, Tributary
from hydrodynamics import Hydrodynamics
from water_quality import WaterQualityModel, WaterQualityParams
from pollution_sources import SourceManager, PointSource, NonPointSource, AccidentalSource, ReleaseType


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
                               start_time: float = 0.0):
        """添加突发泄漏源"""
        rt = ReleaseType.INSTANTANEOUS if release_type == 'instantaneous' else ReleaseType.CONTINUOUS
        acc = AccidentalSource(name=name, x=x, release_type=rt,
                               total_mass_bod=total_mass_bod,
                               total_mass_nh3n=total_mass_nh3n,
                               total_mass_cod=total_mass_cod,
                               release_duration=release_duration,
                               start_time=start_time)
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
            inputs = ps.get_mass_input()
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
        }

        return result

    def _solve_wq_steady(self, x, V, A, h, Q, Dx, sources,
                          initial_conditions, scheme='upwind') -> Dict:
        """求解稳态水质方程"""
        n = len(x)

        K1_per_sec = self.wq_model.params.K1 / 86400.0
        K2_per_sec = self.wq_model.params.K2 / 86400.0
        K_nh3n_per_sec = self.wq_model.params.K_nh3n / 86400.0
        K_cod_per_sec = self.wq_model.params.K_cod / 86400.0
        D_sat = self.wq_model.params.D_O_sat

        bod = np.zeros(n)
        do = np.zeros(n)
        nh3n = np.zeros(n)
        cod = np.zeros(n)

        bod[0] = initial_conditions['bod']
        do[0] = initial_conditions['do']
        nh3n[0] = initial_conditions['nh3n']
        cod[0] = initial_conditions['cod']

        for i in range(1, n):
            dx_i = x[i] - x[i - 1]
            u_i = V[i]

            point_src_bod = 0
            point_src_do = 0
            point_src_nh3n = 0
            point_src_cod = 0
            point_src_flow = 0

            for src in sources:
                if src['type'] == 'point':
                    if x[i - 1] < src['x'] <= x[i]:
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

            if u_i > 0:
                travel_time = dx_i / u_i

                bod[i] *= np.exp(-K1_per_sec * travel_time)
                nh3n[i] *= np.exp(-K_nh3n_per_sec * travel_time)
                cod[i] *= np.exp(-K_cod_per_sec * travel_time)

                deficit = D_sat - do[i]
                do_change = (K2_per_sec * deficit - K1_per_sec * bod[i]) * travel_time
                do[i] += do_change
                do[i] = min(do[i], D_sat)
                do[i] = max(do[i], 0.0)

            if Dx > 0 and u_i > 0 and i > 1:
                diff_bod = Dx * (bod[i - 1] - 2 * bod[i] + bod[i - 2]) / dx_i ** 2
                bod[i] += diff_bod * dx_i / u_i
                bod[i] = max(bod[i], 0.0)

                diff_do = Dx * (do[i - 1] - 2 * do[i] + do[i - 2]) / dx_i ** 2
                do[i] += diff_do * dx_i / u_i
                do[i] = min(do[i], D_sat)
                do[i] = max(do[i], 0.0)

        return {
            'bod': bod,
            'do': do,
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
