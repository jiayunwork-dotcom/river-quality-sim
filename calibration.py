
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.stats import t


@dataclass
class CalibrationData:
    """率定数据"""
    x: List[float]
    bod: Optional[List[float]] = None
    do: Optional[List[float]] = None
    nh3n: Optional[List[float]] = None
    cod: Optional[List[float]] = None


@dataclass
class CalibrationResult:
    """率定结果"""
    K1: float
    K2: float
    K1_std: float
    K2_std: float
    K1_ci: Tuple[float, float]
    K2_ci: Tuple[float, float]
    r_squared_bod: float
    r_squared_do: float
    covariance: np.ndarray


class ParameterCalibration:
    """参数率定模块"""

    def __init__(self):
        pass

    def streeter_phelps_bod(self, x, L0, K1, u):
        """BOD衰减模型"""
        if u <= 0:
            return np.full_like(x, L0)
        t = x / u
        return L0 * np.exp(-K1 * t)

    def streeter_phelps_do(self, x, L0, D0, K1, K2, u, D_sat):
        """DO模型"""
        if u <= 0:
            return np.full_like(x, D_sat - D0)
        t = x / u
        D = (K1 * L0 / (K2 - K1)) * (np.exp(-K1 * t) - np.exp(-K2 * t)) + D0 * np.exp(-K2 * t)
        return D_sat - D

    def calibrate_k1(self, x_data: np.ndarray, bod_data: np.ndarray,
                     u: float, L0: float) -> Tuple[float, float, float]:
        """率定K1参数（BOD衰减系数）"""
        x_data = np.array(x_data)
        bod_data = np.array(bod_data)

        valid = ~np.isnan(bod_data) & (bod_data > 0)
        x_data = x_data[valid]
        bod_data = bod_data[valid]

        if len(x_data) < 3:
            return 0.2, 0.0, 0.0

        def model(x, K1):
            return self.streeter_phelps_bod(x, L0, K1, u)

        try:
            popt, pcov = curve_fit(model, x_data, bod_data, p0=[0.2],
                                   bounds=([0.01], [2.0]))
            K1 = popt[0]
            K1_std = np.sqrt(np.diag(pcov))[0]

            y_pred = model(x_data, K1)
            ss_res = np.sum((bod_data - y_pred) ** 2)
            ss_tot = np.sum((bod_data - np.mean(bod_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return K1, K1_std, r_squared
        except Exception as e:
            print(f"K1率定失败: {e}")
            return 0.2, 0.0, 0.0

    def calibrate_k2(self, x_data: np.ndarray, do_data: np.ndarray,
                     u: float, L0: float, D0: float, K1: float,
                     D_sat: float = 9.5) -> Tuple[float, float, float]:
        """率定K2参数（复氧系数）"""
        x_data = np.array(x_data)
        do_data = np.array(do_data)

        valid = ~np.isnan(do_data)
        x_data = x_data[valid]
        do_data = do_data[valid]

        if len(x_data) < 4:
            return 0.5, 0.0, 0.0

        def model(x, K2):
            return self.streeter_phelps_do(x, L0, D0, K1, K2, u, D_sat)

        try:
            popt, pcov = curve_fit(model, x_data, do_data, p0=[0.5],
                                   bounds=([0.01], [5.0]))
            K2 = popt[0]
            K2_std = np.sqrt(np.diag(pcov))[0]

            y_pred = model(x_data, K2)
            ss_res = np.sum((do_data - y_pred) ** 2)
            ss_tot = np.sum((do_data - np.mean(do_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return K2, K2_std, r_squared
        except Exception as e:
            print(f"K2率定失败: {e}")
            return 0.5, 0.0, 0.0

    def calibrate_joint(self, x_data: np.ndarray, bod_data: np.ndarray,
                        do_data: np.ndarray, u: float, L0: float,
                        D0: float, D_sat: float = 9.5) -> CalibrationResult:
        """联合率定K1和K2"""
        x_data = np.array(x_data)
        bod_data = np.array(bod_data)
        do_data = np.array(do_data)

        valid_bod = ~np.isnan(bod_data) & (bod_data > 0)
        valid_do = ~np.isnan(do_data)

        def combined_model(x_all, K1, K2):
            n_bod = np.sum(valid_bod)
            x_bod = x_all[:n_bod]
            x_do = x_all[n_bod:]

            bod_pred = self.streeter_phelps_bod(x_bod, L0, K1, u)
            do_pred = self.streeter_phelps_do(x_do, L0, D0, K1, K2, u, D_sat)

            return np.concatenate([bod_pred, do_pred])

        x_combined = np.concatenate([x_data[valid_bod], x_data[valid_do]])
        y_combined = np.concatenate([bod_data[valid_bod], do_data[valid_do]])

        if len(x_combined) < 5:
            return CalibrationResult(
                K1=0.25, K2=0.5,
                K1_std=0.0, K2_std=0.0,
                K1_ci=(0.2, 0.3), K2_ci=(0.4, 0.6),
                r_squared_bod=0.0, r_squared_do=0.0,
                covariance=np.eye(2)
            )

        try:
            popt, pcov = curve_fit(
                combined_model, x_combined, y_combined,
                p0=[0.25, 0.5],
                bounds=([0.01, 0.01], [2.0, 5.0]),
                maxfev=10000
            )

            K1, K2 = popt
            perr = np.sqrt(np.diag(pcov))
            K1_std, K2_std = perr

            dof = len(x_combined) - 2
            t_val = t.ppf(0.975, dof) if dof > 0 else 1.96

            K1_ci = (K1 - t_val * K1_std, K1 + t_val * K1_std)
            K2_ci = (K2 - t_val * K2_std, K2 + t_val * K2_std)

            bod_pred = self.streeter_phelps_bod(x_data[valid_bod], L0, K1, u)
            ss_res_bod = np.sum((bod_data[valid_bod] - bod_pred) ** 2)
            ss_tot_bod = np.sum((bod_data[valid_bod] - np.mean(bod_data[valid_bod])) ** 2)
            r2_bod = 1 - (ss_res_bod / ss_tot_bod) if ss_tot_bod > 0 else 0

            do_pred = self.streeter_phelps_do(x_data[valid_do], L0, D0, K1, K2, u, D_sat)
            ss_res_do = np.sum((do_data[valid_do] - do_pred) ** 2)
            ss_tot_do = np.sum((do_data[valid_do] - np.mean(do_data[valid_do])) ** 2)
            r2_do = 1 - (ss_res_do / ss_tot_do) if ss_tot_do > 0 else 0

            return CalibrationResult(
                K1=K1, K2=K2,
                K1_std=K1_std, K2_std=K2_std,
                K1_ci=K1_ci, K2_ci=K2_ci,
                r_squared_bod=r2_bod, r_squared_do=r2_do,
                covariance=pcov
            )
        except Exception as e:
            print(f"联合率定失败: {e}")
            return CalibrationResult(
                K1=0.25, K2=0.5,
                K1_std=0.0, K2_std=0.0,
                K1_ci=(0.2, 0.3), K2_ci=(0.4, 0.6),
                r_squared_bod=0.0, r_squared_do=0.0,
                covariance=np.eye(2)
            )

    def calculate_r_squared(self, y_true, y_pred):
        """计算R平方"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
