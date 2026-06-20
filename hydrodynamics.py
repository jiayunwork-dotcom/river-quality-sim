
import numpy as np
from typing import Tuple, Dict, List
from river_channel import RiverChannel, CrossSection


class Hydrodynamics:
    """水动力计算模块"""

    def __init__(self, channel: RiverChannel):
        self.channel = channel
        self.g = 9.81

    def manning_velocity(self, section: CrossSection, h: float) -> float:
        """曼宁公式计算流速"""
        R = section.hydraulic_radius(h)
        if R <= 0:
            return 0.0
        return (1.0 / section.manning_n) * (R ** (2.0 / 3.0)) * np.sqrt(section.slope)

    def manning_flow(self, section: CrossSection, h: float) -> float:
        """曼宁公式计算流量"""
        A = section.area(h)
        V = self.manning_velocity(section, h)
        return A * V

    def normal_depth(self, section: CrossSection, Q: float,
                     h_tol: float = 1e-6, max_iter: int = 100) -> float:
        """计算正常水深（牛顿迭代法）"""
        if Q <= 0:
            return 0.0

        h = 1.0
        for _ in range(max_iter):
            Q_calc = self.manning_flow(section, h)
            if abs(Q_calc - Q) < h_tol:
                break

            dh = 0.001 * h
            dQ = self.manning_flow(section, h + dh) - Q_calc
            if dQ == 0:
                h += dh
            else:
                h += (Q - Q_calc) * dh / dQ

            if h < 0.01:
                h = 0.01

        return h

    def critical_depth(self, section: CrossSection, Q: float,
                       h_tol: float = 1e-6, max_iter: int = 100) -> float:
        """计算临界水深"""
        if Q <= 0:
            return 0.0

        h = 1.0
        for _ in range(max_iter):
            A = section.area(h)
            B = section.top_width(h)
            if A == 0 or B == 0:
                h += 0.1
                continue

            Froude_sq = (Q ** 2 * B) / (self.g * A ** 3)
            if abs(Froude_sq - 1.0) < h_tol:
                break

            dh = 0.001 * h
            A_new = section.area(h + dh)
            B_new = section.top_width(h + dh)
            dF = (Q ** 2 * B_new) / (self.g * A_new ** 3) - Froude_sq

            if dF == 0:
                h += dh
            else:
                h += (1.0 - Froude_sq) * dh / dF

            if h < 0.01:
                h = 0.01

        return h

    def froude_number(self, section: CrossSection, Q: float, h: float) -> float:
        """计算弗劳德数"""
        A = section.area(h)
        B = section.top_width(h)
        if A == 0 or B == 0:
            return 0.0
        V = Q / A
        c = np.sqrt(self.g * A / B)
        return V / c

    def specific_energy(self, section: CrossSection, Q: float, h: float) -> float:
        """计算断面比能"""
        A = section.area(h)
        if A == 0:
            return h
        V = Q / A
        return h + V ** 2 / (2 * self.g)

    def uniform_flow(self, Q_upstream: float, n_grid: int = 100) -> Dict:
        """恒定均匀流计算"""
        total_len = self.channel.total_length()
        x = np.linspace(self.channel.sections[0].x,
                        self.channel.sections[-1].x, n_grid)

        h = np.zeros(n_grid)
        V = np.zeros(n_grid)
        A = np.zeros(n_grid)
        Q_arr = np.full(n_grid, Q_upstream)
        z = np.zeros(n_grid)
        water_level = np.zeros(n_grid)

        current_Q = Q_upstream
        trib_idx = 0

        for i in range(n_grid):
            section = self.channel.get_section_at(x[i])

            while (trib_idx < len(self.channel.tributaries) and
                   self.channel.tributaries[trib_idx].x <= x[i]):
                trib = self.channel.tributaries[trib_idx]
                conc = {'bod': 0, 'do': 8, 'nh3n': 0, 'cod': 0}
                current_Q, _ = self.channel.mix_tributary(current_Q, conc, trib)
                trib_idx += 1

            Q_arr[i] = current_Q
            h[i] = self.normal_depth(section, current_Q)
            A[i] = section.area(h[i])
            V[i] = current_Q / A[i] if A[i] > 0 else 0
            z[i] = section.slope * (x[-1] - x[i])
            water_level[i] = z[i] + h[i]

        return {
            'x': x,
            'h': h,
            'V': V,
            'Q': Q_arr,
            'A': A,
            'z': z,
            'water_level': water_level,
        }

    def gradually_varied_flow(self, Q_upstream: float,
                              control_h: float, control_x: str = 'downstream',
                              n_grid: int = 100, dx: float = 10.0) -> Dict:
        """恒定非均匀流 - 标准步长法推算水面线"""
        total_len = self.channel.total_length()

        if control_x == 'downstream':
            x = np.linspace(self.channel.sections[-1].x,
                            self.channel.sections[0].x, n_grid)
            direction = -1
        else:
            x = np.linspace(self.channel.sections[0].x,
                            self.channel.sections[-1].x, n_grid)
            direction = 1

        h = np.zeros(n_grid)
        V = np.zeros(n_grid)
        A = np.zeros(n_grid)
        Q_arr = np.full(n_grid, Q_upstream)
        z = np.zeros(n_grid)
        water_level = np.zeros(n_grid)

        section = self.channel.get_section_at(x[0])
        h[0] = control_h
        A[0] = section.area(h[0])
        V[0] = Q_upstream / A[0] if A[0] > 0 else 0

        for i in range(1, n_grid):
            section = self.channel.get_section_at(x[i])

            dx_step = abs(x[i] - x[i - 1])
            h_prev = h[i - 1]
            section_prev = self.channel.get_section_at(x[i - 1])

            Sf_prev = (section_prev.manning_n ** 2 * V[i - 1] ** 2 /
                       (section_prev.hydraulic_radius(h_prev) ** (4.0 / 3.0)))

            h_guess = h_prev
            for _ in range(30):
                A_guess = section.area(h_guess)
                V_guess = Q_upstream / A_guess if A_guess > 0 else 0
                R_guess = section.hydraulic_radius(h_guess)
                Sf_guess = (section.manning_n ** 2 * V_guess ** 2 /
                           (R_guess ** (4.0 / 3.0)) if R_guess > 0 else 0)
                Sf_avg = (Sf_prev + Sf_guess) / 2.0

                E_prev = h_prev + V[i - 1] ** 2 / (2 * self.g)
                E_guess = h_guess + V_guess ** 2 / (2 * self.g)

                delta_E = direction * (section.slope - Sf_avg) * dx_step
                h_new = h_guess + (E_prev + delta_E - E_guess)

                if abs(h_new - h_guess) < 1e-5:
                    break
                h_guess = h_new

            h[i] = max(h_guess, 0.01)
            A[i] = section.area(h[i])
            V[i] = Q_upstream / A[i] if A[i] > 0 else 0

        if direction == -1:
            x = x[::-1]
            h = h[::-1]
            V = V[::-1]
            A = A[::-1]
            Q_arr = Q_arr[::-1]

        for i in range(n_grid):
            section = self.channel.get_section_at(x[i])
            z[i] = section.slope * (x[-1] - x[i])
            water_level[i] = z[i] + h[i]

        return {
            'x': x,
            'h': h,
            'V': V,
            'Q': Q_arr,
            'A': A,
            'z': z,
            'water_level': water_level,
        }

    def kinematic_wave(self, Q_upstream: np.ndarray, t_total: float,
                       dt: float, n_grid: int = 100) -> Dict:
        """非恒定流 - 运动波近似"""
        total_len = self.channel.total_length()
        x = np.linspace(self.channel.sections[0].x,
                        self.channel.sections[-1].x, n_grid)
        dx = x[1] - x[0]

        n_steps = int(t_total / dt)
        t = np.arange(n_steps + 1) * dt

        Q = np.zeros((n_steps + 1, n_grid))
        h = np.zeros((n_steps + 1, n_grid))
        V = np.zeros((n_steps + 1, n_grid))
        A = np.zeros((n_steps + 1, n_grid))

        section = self.channel.get_section_at(x[0])
        Q[0, :] = Q_upstream[0] if hasattr(Q_upstream, '__len__') else Q_upstream
        for j in range(n_grid):
            sec = self.channel.get_section_at(x[j])
            h[0, j] = self.normal_depth(sec, Q[0, j])
            A[0, j] = sec.area(h[0, j])
            V[0, j] = Q[0, j] / A[0, j] if A[0, j] > 0 else 0

        courant_warning = False
        for i in range(n_steps):
            Q[i + 1, 0] = Q_upstream[i + 1] if hasattr(Q_upstream, '__len__') and len(Q_upstream) > i + 1 else Q_upstream[-1] if hasattr(Q_upstream, '__len__') else Q_upstream

            for j in range(1, n_grid):
                sec = self.channel.get_section_at(x[j])
                sec_prev = self.channel.get_section_at(x[j - 1])

                alpha = 5 / 3

                Q_prev = Q[i, j]
                h_prev = h[i, j]
                V_prev = V[i, j]
                c_k = alpha * V_prev

                if c_k * dt / dx > 1.0:
                    courant_warning = True

                dQdx = (Q[i, j] - Q[i, j - 1]) / dx
                Q_new = Q[i, j] - V_prev * dt * dQdx

                Q[i + 1, j] = max(Q_new, 0)
                h[i + 1, j] = self.normal_depth(sec, Q[i + 1, j])
                A[i + 1, j] = sec.area(h[i + 1, j])
                V[i + 1, j] = Q[i + 1, j] / A[i + 1, j] if A[i + 1, j] > 0 else 0

        result = {
            'x': x,
            't': t,
            'Q': Q,
            'h': h,
            'V': V,
            'A': A,
            'courant_warning': courant_warning,
        }
        return result

    def check_courant(self, V: float, dx: float, dt: float) -> Tuple[bool, float]:
        """检查Courant数"""
        if dx == 0:
            return True, 0.0
        Co = V * dt / dx
        return Co <= 1.0, Co

    def suggest_dt(self, V_max: float, dx: float, safety_factor: float = 0.8) -> float:
        """建议时间步长"""
        if V_max <= 0:
            return 3600.0
        return safety_factor * dx / V_max
