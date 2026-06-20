
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats


@dataclass
class SimulationRecord:
    """单次模拟记录"""
    record_id: int
    timestamp: datetime
    params: Dict
    x: np.ndarray
    bod: np.ndarray
    do: np.ndarray
    nh3n: np.ndarray

    def param_summary(self) -> str:
        p = self.params
        parts = []
        if 'Q_upstream' in p:
            parts.append(f"Q={p['Q_upstream']:.1f}")
        if 'K1' in p:
            parts.append(f"K1={p['K1']:.2f}")
        if 'K2' in p:
            parts.append(f"K2={p['K2']:.2f}")
        if 'initial_bod' in p:
            parts.append(f"BOD0={p['initial_bod']:.1f}")
        if 'initial_do' in p:
            parts.append(f"DO0={p['initial_do']:.1f}")
        if 'initial_nh3n' in p:
            parts.append(f"NH3N0={p['initial_nh3n']:.1f}")
        ps_count = p.get('point_source_count', 0)
        if ps_count > 0:
            parts.append(f"{ps_count}PS")
        return ", ".join(parts)

    def get_compliance_rate(self, standards: Dict) -> float:
        bod_ok = self.bod <= standards['bod']
        do_ok = self.do >= standards['do']
        nh3n_ok = self.nh3n <= standards['nh3n']
        all_ok = bod_ok & do_ok & nh3n_ok
        return np.mean(all_ok) * 100


@dataclass
class SimulationHistory:
    """模拟历史记录队列（最多50条）"""
    max_records: int = 50
    records: List[SimulationRecord] = field(default_factory=list)
    _counter: int = 0

    def add_record(self, params: Dict, x: np.ndarray, bod: np.ndarray,
                   do: np.ndarray, nh3n: np.ndarray) -> SimulationRecord:
        self._counter += 1
        record = SimulationRecord(
            record_id=self._counter,
            timestamp=datetime.now(),
            params=params.copy(),
            x=x.copy(),
            bod=bod.copy(),
            do=do.copy(),
            nh3n=nh3n.copy()
        )
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
        return record

    def __len__(self):
        return len(self.records)

    def is_empty(self):
        return len(self.records) == 0

    def clear(self):
        self.records = []


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """计算移动平均值"""
    if window <= 1 or len(data) < window:
        return np.full_like(data, np.nan)
    kernel = np.ones(window) / window
    ma = np.convolve(data, kernel, mode='same')
    pad_left = window // 2
    pad_right = window - pad_left - 1
    for i in range(pad_left):
        ma[i] = np.mean(data[:i + pad_right + 1])
    for i in range(len(data) - pad_right, len(data)):
        ma[i] = np.mean(data[i - pad_left:])
    return ma


def modified_z_score(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """计算改进Z-score（基于中位数和MAD）"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        mad = 1e-10
    mod_z = 0.6745 * (data - median) / mad
    return mod_z, median, mad


def detect_anomalies(records: List[SimulationRecord]) -> Dict:
    """逐点异常检测 - 改进Z-score方法"""
    if len(records) == 0:
        return {}

    n_records = len(records)
    ref_x = records[0].x
    n_points = len(ref_x)

    anomaly_matrix = {}
    anomaly_details = []

    for comp_idx, comp_name in enumerate(['bod', 'nh3n', 'do']):
        matrix = np.zeros((n_records, n_points))
        comp_data = np.zeros((n_records, n_points))

        for r_idx, rec in enumerate(records):
            if comp_name == 'bod':
                comp_data[r_idx] = rec.bod
            elif comp_name == 'nh3n':
                comp_data[r_idx] = rec.nh3n
            else:
                comp_data[r_idx] = rec.do

        for p_idx in range(n_points):
            point_data = comp_data[:, p_idx]
            mod_z, median, mad = modified_z_score(point_data)

            for r_idx in range(n_records):
                z_val = abs(mod_z[r_idx])
                if z_val > 3:
                    matrix[r_idx, p_idx] = 2
                elif z_val > 2:
                    matrix[r_idx, p_idx] = 1
                else:
                    matrix[r_idx, p_idx] = 0

                if z_val > 2:
                    severity = "严重异常" if z_val > 3 else "轻度异常"
                    comp_label = {'bod': 'BOD', 'nh3n': 'NH3-N', 'do': 'DO'}[comp_name]
                    anomaly_details.append({
                        'record_id': records[r_idx].record_id,
                        'record_time': records[r_idx].timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        'position': f"{ref_x[p_idx]:.0f}m",
                        'component': comp_label,
                        'value': comp_data[r_idx, p_idx],
                        'median': median,
                        'mad': mad,
                        'z_score': mod_z[r_idx],
                        'severity': severity,
                        'severity_score': z_val,
                    })

        anomaly_matrix[comp_name] = matrix

    anomaly_details.sort(key=lambda x: x['severity_score'], reverse=True)

    return {
        'anomaly_matrix': anomaly_matrix,
        'anomaly_details': anomaly_details,
        'positions': ref_x,
    }


def compute_spearman_correlation(records: List[SimulationRecord],
                                  standards: Dict) -> Dict:
    """计算Spearman秩相关系数矩阵"""
    if len(records) < 3:
        return {}

    param_vars = ['K1', 'K2', 'Q_upstream', 'initial_bod', 'initial_do', 'initial_nh3n']
    actual_params = []
    for pv in param_vars:
        if all(pv in r.params for r in records):
            actual_params.append(pv)

    result_vars = ['avg_bod', 'min_do', 'max_nh3n', 'compliance_rate']
    param_data = {pv: np.array([r.params[pv] for r in records]) for pv in actual_params}

    result_data = {
        'avg_bod': np.array([np.mean(r.bod) for r in records]),
        'min_do': np.array([np.min(r.do) for r in records]),
        'max_nh3n': np.array([np.max(r.nh3n) for r in records]),
        'compliance_rate': np.array([r.get_compliance_rate(standards) for r in records]),
    }

    all_vars = actual_params + result_vars
    n_vars = len(all_vars)

    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.ones((n_vars, n_vars))
    ci_low = np.zeros((n_vars, n_vars))
    ci_high = np.zeros((n_vars, n_vars))

    all_data = {}
    all_data.update(param_data)
    all_data.update(result_data)

    n_bootstrap = 1000
    n_records = len(records)

    for i in range(n_vars):
        for j in range(i, n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                ci_low[i, j] = 1.0
                ci_high[i, j] = 1.0
            else:
                x = all_data[all_vars[i]]
                y = all_data[all_vars[j]]

                res = stats.spearmanr(x, y)
                corr = res.statistic if hasattr(res, 'statistic') else res.correlation
                p_val = res.pvalue if hasattr(res, 'pvalue') else 1.0

                if np.isnan(corr):
                    corr = 0.0
                if np.isnan(p_val):
                    p_val = 1.0

                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val

                boot_corrs = []
                for _ in range(n_bootstrap):
                    idx = np.random.choice(n_records, size=n_records, replace=True)
                    x_boot = x[idx]
                    y_boot = y[idx]
                    if len(np.unique(x_boot)) > 1 and len(np.unique(y_boot)) > 1:
                        boot_res = stats.spearmanr(x_boot, y_boot)
                        boot_corr = boot_res.statistic if hasattr(boot_res, 'statistic') else boot_res.correlation
                        if not np.isnan(boot_corr):
                            boot_corrs.append(boot_corr)

                if len(boot_corrs) > 10:
                    boot_corrs = np.array(boot_corrs)
                    ci_low[i, j] = np.percentile(boot_corrs, 2.5)
                    ci_high[i, j] = np.percentile(boot_corrs, 97.5)
                else:
                    ci_low[i, j] = corr
                    ci_high[i, j] = corr

                ci_low[j, i] = ci_low[i, j]
                ci_high[j, i] = ci_high[i, j]

    return {
        'all_vars': all_vars,
        'param_vars': actual_params,
        'result_vars': result_vars,
        'corr_matrix': corr_matrix,
        'p_matrix': p_matrix,
        'ci_low': ci_low,
        'ci_high': ci_high,
    }


def compute_summary_stats(records: List[SimulationRecord], standards: Dict) -> Dict:
    """计算趋势统计摘要"""
    if len(records) < 3:
        return {}

    n = len(records)

    if n >= 5:
        half = n // 2
        first_half = records[:half]
        second_half = records[half:]
    else:
        third = max(1, n // 3)
        first_half = records[:third]
        second_half = records[-third:]

    first_rate = np.mean([r.get_compliance_rate(standards) for r in first_half])
    second_rate = np.mean([r.get_compliance_rate(standards) for r in second_half])
    rate_change = second_rate - first_rate
    if rate_change > 0.5:
        trend = "上升 📈"
    elif rate_change < -0.5:
        trend = "下降 📉"
    else:
        trend = "稳定 ➡️"

    anomaly_result = detect_anomalies(records)
    anomaly_details = anomaly_result.get('anomaly_details', [])
    severe_anomalies = [a for a in anomaly_details if a['severity'] == '严重异常']
    anomaly_ratio = len(severe_anomalies) / (n * len(records[0].x) * 3) * 100 if n > 0 else 0

    param_vars = ['K1', 'K2', 'Q_upstream', 'initial_bod', 'initial_do', 'initial_nh3n']
    max_cv = 0
    max_cv_var = "无"
    for pv in param_vars:
        if all(pv in r.params for r in records):
            vals = np.array([r.params[pv] for r in records])
            if np.mean(vals) != 0:
                cv = np.std(vals) / np.abs(np.mean(vals)) * 100
                if cv > max_cv:
                    max_cv = cv
                    label_map = {
                        'K1': 'BOD衰减系数K1',
                        'K2': '复氧系数K2',
                        'Q_upstream': '上游流量Q',
                        'initial_bod': '上游BOD初始值',
                        'initial_do': '上游DO初始值',
                        'initial_nh3n': '上游NH3-N初始值',
                    }
                    max_cv_var = f"{label_map.get(pv, pv)} (CV={cv:.1f}%)"

    avg_do_per_position = np.mean([r.do for r in records], axis=0)
    worst_idx = np.argmin(avg_do_per_position)
    worst_position = records[0].x[worst_idx]
    worst_do = avg_do_per_position[worst_idx]

    return {
        'compliance_trend': trend,
        'first_rate': first_rate,
        'second_rate': second_rate,
        'rate_change': rate_change,
        'anomaly_ratio': anomaly_ratio,
        'anomaly_count': len(severe_anomalies),
        'max_cv_var': max_cv_var,
        'worst_position': f"{worst_position:.0f}m",
        'worst_do': worst_do,
    }
