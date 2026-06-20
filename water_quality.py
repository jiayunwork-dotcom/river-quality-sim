
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import solve_banded


@dataclass
class WaterQualityParams:
    """水质模型参数"""
    K1: float = 0.25
    K2: float = 0.5
    K_nh3n: float = 0.1
    K_cod: float = 0.15
    D_O_sat: float = 9.5
    D_sed: float = 0.5
    P_photo: float = 1.0
    Dx: float = 10.0


class WaterQualityModel:
    """一维对流扩散水质模型，耦合BOD-DO"""

    def __init__(self, params: WaterQualityParams = None):
        self.params = params or WaterQualityParams()
        self.components = ['bod', 'do', 'nh3n', 'cod']

    def streeter_phelps(self, L0: float, D0: float, u: float, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """Streeter-Phelps 解析解"""
        K1 = self.params.K1
        K2 = self.params.K2
        D_sat = self.params.D_O_sat

        if u <= 0:
            return np.full_like(x, L0), np.full_like(x, D_sat - D0)

        t = x / u
        L = L0 * np.exp(-K1 * t)

        D = (K1 * L0 / (K2 - K1)) * (np.exp(-K1 * t) - np.exp(-K2 * t)) + (D_sat - D0) * np.exp(-K2 * t)

        return L, D

    def critical_point(self, L0: float, D0: float, u: float) -> Tuple[float, float, float]:
        """计算氧垂曲线临界点"""
        K1 = self.params.K1
        K2 = self.params.K2
        D_sat = self.params.D_O_sat

        if u <= 0 or K1 == K2:
            return 0.0, D_sat - D0, 0.0

        D0_deficit = D_sat - D0

        tc = (1 / (K2 - K1)) * np.log((K2 / K1) * (1 - D0_deficit * (K2 - K1) / (K1 * L0)))
        tc = max(0, tc)

        x_c = u * tc

        D_c = (K1 / K2) * L0 * np.exp(-K1 * tc)
        DO_c = D_sat - D_c

        return x_c, DO_c, tc

    def solve_steady_advection_diffusion(self, x: np.ndarray, u: np.ndarray,
                                         A: np.ndarray, Dx: float,
                                         sources: List[Dict],
                                         initial_conditions: Dict,
                                         scheme: str = 'upwind') -> Dict:
        """稳态对流扩散方程求解"""
        n = len(x)
        dx = np.diff(x)

        results = {}

        for comp in self.components:
            C = np.zeros(n)
            C[0] = initial_conditions.get(comp, 0.0)

            if comp == 'do':
                C[0] = initial_conditions.get(comp, self.params.D_O_sat)

            if scheme == 'upwind':
                for i in range(1, n):
                    dx_i = x[i] - x[i - 1]
                    u_i = u[i]
                    A_i = A[i]

                    if u_i == 0:
                        C[i] = C[i - 1]
                        continue

                    if comp == 'bod':
                        decay = -self.params.K1 * C[i - 1]
                    elif comp == 'do':
                        deficit = self.params.D_O_sat - C[i - 1]
                        decay = self.params.K2 * deficit - self.params.K1 * initial_conditions.get('bod', 5.0)
                    elif comp == 'nh3n':
                        decay = -self.params.K_nh3n * C[i - 1]
                    elif comp == 'cod':
                        decay = -self.params.K_cod * C[i - 1]
                    else:
                        decay = 0

                    src = 0
                    for s in sources:
                        if abs(s.get('x', 0) - x[i]) < dx_i:
                            if comp in s.get('concentration', {}):
                                Q_in = s.get('flow_rate', 0)
                                Q_main = u_i * A_i
                                if Q_main + Q_in > 0:
                                    src = (Q_in * s['concentration'][comp]) / (Q_main + Q_in)

                    C[i] = C[i - 1] + (decay + src / dx_i if dx_i > 0 else 0) * dx_i / u_i

                    if comp == 'do':
                        deficit = self.params.D_O_sat - C[i - 1]
                        bod_val = results.get('bod', np.zeros(n))[i - 1] if 'bod' in results else initial_conditions.get('bod', 5.0)
                        dCdx = (-u_i * (C[i - 1] - (C[i - 2] if i > 1 else C[i - 1])) / dx_i +
                                self.params.K2 * deficit - self.params.K1 * bod_val)
                        C[i] = C[i - 1] + dCdx * dx_i / u_i if u_i > 0 else C[i - 1]

            elif scheme == 'implicit':
                a = np.zeros(n)
                b = np.zeros(n)
                c = np.zeros(n)
                d = np.zeros(n)

                b[0] = 1.0
                d[0] = C[0]

                for i in range(1, n):
                    dx_i = x[i] - x[i - 1]
                    u_i = u[i]

                    if comp == 'bod':
                        decay = self.params.K1
                    elif comp == 'nh3n':
                        decay = self.params.K_nh3n
                    elif comp == 'cod':
                        decay = self.params.K_cod
                    else:
                        decay = 0

                    if comp == 'do':
                        a[i] = -Dx / dx_i ** 2 - u_i / (2 * dx_i)
                        b[i] = 2 * Dx / dx_i ** 2 + u_i / dx_i + self.params.K2
                        c[i] = -Dx / dx_i ** 2 + u_i / (2 * dx_i)
                        d[i] = self.params.K2 * self.params.D_O_sat
                    else:
                        a[i] = -Dx / dx_i ** 2 - u_i / (2 * dx_i)
                        b[i] = 2 * Dx / dx_i ** 2 + u_i / dx_i + decay
                        c[i] = -Dx / dx_i ** 2 + u_i / (2 * dx_i)
                        d[i] = 0

                ab = np.zeros((3, n))
                ab[0, 1:] = c[:-1]
                ab[1, :] = b
                ab[2, :-1] = a[1:]

                C = solve_banded((1, 1), ab, d)

            results[comp] = C

        return results

    def solve_unsteady_advection_diffusion(self, x: np.ndarray, t: np.ndarray,
                                           u: np.ndarray, A: np.ndarray,
                                           Dx: float,
                                           sources: List[Dict],
                                           initial_conditions: Dict,
                                           scheme: str = 'upwind') -> Dict:
        """非稳态对流扩散方程求解"""
        nx = len(x)
        nt = len(t)
        dx = x[1] - x[0]
        dt = t[1] - t[0]

        results = {comp: np.zeros((nt, nx)) for comp in self.components}

        for comp in self.components:
            results[comp][0, 0] = initial_conditions.get(comp, 0.0)
            if comp == 'do':
                results[comp][0, 0] = initial_conditions.get(comp, self.params.D_O_sat)
            results[comp][0, 1:] = results[comp][0, 0]

        for n in range(nt - 1):
            for comp in self.components:
                C_prev = results[comp][n, :].copy()
                C_new = C_prev.copy()

                if scheme == 'upwind':
                    for i in range(1, nx):
                        u_i = u[i] if len(u.shape) == 1 else u[n, i]
                        dt_i = dt

                        adv = -u_i * (C_prev[i] - C_prev[i - 1]) / dx
                        diff = Dx * (C_prev[i + 1] - 2 * C_prev[i] + C_prev[i - 1]) / dx ** 2 if i < nx - 1 else 0

                        if comp == 'bod':
                            decay = -self.params.K1 * C_prev[i]
                        elif comp == 'do':
                            deficit = self.params.D_O_sat - C_prev[i]
                            bod_val = results['bod'][n, i]
                            decay = self.params.K2 * deficit - self.params.K1 * bod_val
                        elif comp == 'nh3n':
                            decay = -self.params.K_nh3n * C_prev[i]
                        elif comp == 'cod':
                            decay = -self.params.K_cod * C_prev[i]
                        else:
                            decay = 0

                        src = 0
                        for s in sources:
                            if s.get('type') == 'point':
                                src_x = s.get('x', 0)
                                if abs(src_x - x[i]) < dx:
                                    Q_in = s.get('flow_rate', 0)
                                    A_i = A[i] if len(A.shape) == 1 else A[n, i]
                                    Q_main = u_i * A_i
                                    conc = s.get('concentration', {}).get(comp, 0)
                                    src = (Q_in * conc) / (A_i * dx)

                        C_new[i] = C_prev[i] + dt_i * (adv + diff + decay + src)

                    C_new[0] = initial_conditions.get(comp, 0.0)
                    if comp == 'do':
                        C_new[0] = initial_conditions.get(comp, self.params.D_O_sat)
                    C_new[-1] = C_new[-2]

                elif scheme == 'crank_nicolson':
                    a_coeff = np.zeros(nx)
                    b_coeff = np.zeros(nx)
                    c_coeff = np.zeros(nx)
                    d = np.zeros(nx)

                    b_coeff[0] = 1.0
                    d[0] = initial_conditions.get(comp, 0.0)
                    if comp == 'do':
                        d[0] = initial_conditions.get(comp, self.params.D_O_sat)

                    b_coeff[-1] = 1.0
                    d[-1] = C_prev[-1]

                    r = Dx * dt / (2 * dx ** 2)

                    for i in range(1, nx - 1):
                        u_i = u[i] if len(u.shape) == 1 else u[n, i]
                        c_adv = u_i * dt / (4 * dx)

                        if comp == 'bod':
                            decay = self.params.K1
                        elif comp == 'nh3n':
                            decay = self.params.K_nh3n
                        elif comp == 'cod':
                            decay = self.params.K_cod
                        else:
                            decay = 0

                        a_coeff[i] = -r + c_adv
                        b_coeff[i] = 1 + 2 * r + decay * dt / 2
                        c_coeff[i] = -r - c_adv

                        if comp == 'do':
                            deficit = self.params.D_O_sat - C_prev[i]
                            bod_val = results['bod'][n, i]
                            source_term = (self.params.K2 * deficit - self.params.K1 * bod_val) * dt / 2
                            b_coeff[i] = 1 + 2 * r + self.params.K2 * dt / 2
                            d[i] = C_prev[i] + r * (C_prev[i + 1] - 2 * C_prev[i] + C_prev[i - 1]) + \
                                   self.params.K2 * self.params.D_O_sat * dt / 2 - self.params.K1 * bod_val * dt / 2
                        else:
                            d[i] = C_prev[i] + r * (C_prev[i + 1] - 2 * C_prev[i] + C_prev[i - 1]) - \
                                   decay * C_prev[i] * dt / 2

                    ab = np.zeros((3, nx))
                    ab[0, 1:] = c_coeff[:-1]
                    ab[1, :] = b_coeff
                    ab[2, :-1] = a_coeff[1:]

                    C_new = solve_banded((1, 1), ab, d)

                results[comp][n + 1, :] = C_new

        return results

    def gaussian_pulse(self, x: np.ndarray, t: float, x0: float, M: float,
                        u: float, Dx: float, A: float = 1.0) -> np.ndarray:
        """瞬时源高斯解析解"""
        if t <= 0:
            return np.zeros_like(x)

        sigma = np.sqrt(2 * Dx * t)
        if sigma == 0:
            return np.zeros_like(x)

        C = (M / (A * sigma * np.sqrt(np.pi))) * np.exp(-(x - x0 - u * t) ** 2 / (2 * sigma ** 2))
        return C
