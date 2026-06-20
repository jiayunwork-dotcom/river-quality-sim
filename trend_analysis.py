
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
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
    """计算移动平均值（中心窗口，边界处自动扩展/收缩，保证至少2个数据点）"""
    n = len(data)
    if window <= 1 or n < 2:
        return np.full_like(data, np.nan)

    w = min(window, n)
    ma = np.zeros(n)

    for i in range(n):
        half_left = w // 2
        half_right = (w - 1) // 2
        left = max(0, i - half_left)
        right = min(n - 1, i + half_right)

        if right - left + 1 < 2:
            if left == 0:
                right = min(n - 1, 1)
            else:
                left = max(0, n - 2)

        ma[i] = np.mean(data[left:right + 1])

    return ma


def modified_z_score(data: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """计算改进Z-score（基于中位数和MAD），对MAD=0的情况做稳健处理"""
    median = np.median(data)
    abs_dev = np.abs(data - median)
    mad = np.median(abs_dev)

    eps = 1e-12 * max(abs(median), 1.0)
    if mad < eps:
        std = np.std(data, ddof=0)
        if std < eps:
            return np.zeros_like(data, dtype=float), median, mad
        else:
            mad = 0.6745 * std

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


PARAM_LABEL_MAP = {
    'K1': 'BOD衰减系数K1',
    'K2': '复氧系数K2',
    'Q_upstream': '上游流量Q',
    'initial_bod': '上游BOD初始值',
    'initial_do': '上游DO初始值',
    'initial_nh3n': '上游NH3-N初始值',
    'Dx': '扩散系数Dx',
    'ps1_flow': '排污口1排放流量',
    'ps1_bod': '排污口1 BOD浓度',
    'ps1_nh3n': '排污口1 NH3-N浓度',
}


def analyze_anomaly_cause(anomaly_record: SimulationRecord,
                          all_records: List[SimulationRecord],
                          position_idx: int,
                          component: str,
                          point_sources: Optional[List] = None) -> Dict:
    """
    分析异常的可能成因：
    1. 将异常记录的参数与其他正常记录的参数对比，找出偏差最大的参数
    2. 查找异常断面上游最近的污染源位置

    注意：对比参数偏差时，排除点源数量为0的记录
    """
    if len(all_records) < 2:
        return {}

    anomaly_record_id = anomaly_record.record_id

    normal_records = [r for r in all_records
                      if r.record_id != anomaly_record_id
                      and r.params.get('point_source_count', 0) > 0]

    if len(normal_records) == 0:
        normal_records = [r for r in all_records if r.record_id != anomaly_record_id]

    if len(normal_records) == 0:
        return {}

    anomaly_params = anomaly_record.params
    param_names = [k for k in anomaly_params.keys()
                   if k in PARAM_LABEL_MAP and k not in ['point_source_count', 'nonpoint_source_count']]

    param_deviations = []
    for pname in param_names:
        if not all(pname in r.params for r in normal_records):
            continue

        normal_vals = np.array([r.params[pname] for r in normal_records])
        anomaly_val = anomaly_params[pname]

        normal_median = np.median(normal_vals)
        normal_mad = np.median(np.abs(normal_vals - normal_median))

        if normal_median == 0:
            continue

        eps = 1e-12 * max(abs(normal_median), 1.0)
        if normal_mad < eps:
            normal_mad = np.std(normal_vals, ddof=0) * 0.6745
            if normal_mad < eps:
                normal_mad = eps

        mod_z = 0.6745 * (anomaly_val - normal_median) / normal_mad
        deviation_pct = (anomaly_val - normal_median) / abs(normal_median) * 100

        param_deviations.append({
            'param_name': pname,
            'param_label': PARAM_LABEL_MAP.get(pname, pname),
            'anomaly_value': anomaly_val,
            'normal_median': normal_median,
            'normal_mad': normal_mad,
            'standardized_deviation': abs(mod_z),
            'deviation_percent': deviation_pct,
            'mod_z': mod_z,
            'direction': '偏高' if mod_z > 0 else '偏低',
        })

    param_deviations.sort(key=lambda x: x['standardized_deviation'], reverse=True)

    ref_x = anomaly_record.x
    position_x = ref_x[position_idx]

    nearest_upstream = None
    in_impact_range = False
    impact_source_name = None

    if point_sources and len(point_sources) > 0:
        upstream_sources = [ps for ps in point_sources if ps.x <= position_x]
        if upstream_sources:
            nearest_upstream = min(upstream_sources, key=lambda ps: position_x - ps.x)
            distance = position_x - nearest_upstream.x
            if distance <= 1000:
                in_impact_range = True
                impact_source_name = nearest_upstream.name

    comp_label = {'bod': 'BOD', 'nh3n': 'NH3-N', 'do': 'DO'}[component]

    cause_explanations = []
    for pd in param_deviations[:3]:
        direction_word = "升高" if pd['mod_z'] > 0 else "降低"
        cause_explanations.append(
            f"该记录的{pd['param_label']}{direction_word}{abs(pd['deviation_percent']):.1f}%，"
            f"可能是导致断面{position_x:.0f}m处{comp_label}异常的原因"
        )

    return {
        'param_deviations': param_deviations,
        'top_causes': cause_explanations,
        'nearest_upstream_source': nearest_upstream,
        'in_impact_range': in_impact_range,
        'impact_source_name': impact_source_name,
        'position_x': position_x,
        'component': comp_label,
        'anomaly_record_id': anomaly_record_id,
    }


from typing import List as TList


@dataclass(kw_only=True)
class TrendWarningRule:
    """趋势预警规则基类"""
    rule_type: Literal['monotonic', 'slope', 'volatility']
    section_idx: int
    component: Literal['bod', 'nh3n', 'do']
    rule_id: int = field(default_factory=lambda: 0)
    enabled: bool = True
    rule_name: str = ""

    def get_section_label(self, positions: np.ndarray) -> str:
        return f"x = {positions[self.section_idx]:.0f}m"

    def get_component_label(self) -> str:
        return {'bod': 'BOD', 'nh3n': 'NH3-N', 'do': 'DO'}[self.component]


@dataclass(kw_only=True)
class MonotonicTrendRule(TrendWarningRule):
    """单调趋势预警规则"""
    consecutive_count: int = 5
    rule_type: Literal['monotonic'] = 'monotonic'

    def check(self, records: List[SimulationRecord]) -> Dict:
        if len(records) < self.consecutive_count:
            return {'triggered': False, 'reason': '记录数量不足'}

        values = []
        for r in records:
            if self.component == 'bod':
                values.append(r.bod[self.section_idx])
            elif self.component == 'nh3n':
                values.append(r.nh3n[self.section_idx])
            else:
                values.append(r.do[self.section_idx])

        values = np.array(values)
        val_median = np.median(values)
        tolerance = abs(val_median) * 0.01 if val_median != 0 else 0.001

        n = len(values)
        for start in range(n - self.consecutive_count + 1):
            window = values[start:start + self.consecutive_count]
            rising = True
            falling = True
            has_rising = False
            has_falling = False
            for i in range(1, len(window)):
                diff = window[i] - window[i - 1]
                if abs(diff) <= tolerance:
                    continue
                if diff < 0:
                    rising = False
                    has_falling = True
                if diff > 0:
                    falling = False
                    has_rising = True
                if not rising and not falling:
                    break

            if (rising and has_rising) or (falling and has_falling):
                direction = "单调上升" if rising else "单调下降"
                trigger_values = [f"{v:.3f}" for v in window]
                return {
                    'triggered': True,
                    'direction': direction,
                    'start_record': records[start].record_id,
                    'end_record': records[start + self.consecutive_count - 1].record_id,
                    'values': trigger_values,
                    'message': f"断面{self.get_section_label(records[0].x)}的{self.get_component_label()}连续{self.consecutive_count}条记录呈{direction}",
                }

        return {'triggered': False, 'reason': '未检测到连续单调趋势'}


@dataclass(kw_only=True)
class SlopeWarningRule(TrendWarningRule):
    """斜率预警规则"""
    window_size: int = 5
    slope_threshold: float = 0.1
    rule_type: Literal['slope'] = 'slope'

    def check(self, records: List[SimulationRecord]) -> Dict:
        if len(records) < self.window_size:
            return {'triggered': False, 'reason': '记录数量不足'}

        values = []
        for r in records:
            if self.component == 'bod':
                values.append(r.bod[self.section_idx])
            elif self.component == 'nh3n':
                values.append(r.nh3n[self.section_idx])
            else:
                values.append(r.do[self.section_idx])

        values = np.array(values)
        n = len(values)

        for start in range(n - self.window_size + 1):
            window = values[start:start + self.window_size]
            ma = moving_average(window, self.window_size)

            valid_mask = ~np.isnan(ma)
            if np.sum(valid_mask) < 2:
                continue

            x = np.arange(len(ma))[valid_mask]
            y = ma[valid_mask]

            if len(x) < 2:
                continue

            slope, intercept = np.polyfit(x, y, 1)

            if abs(slope) >= self.slope_threshold:
                direction = "上升" if slope > 0 else "下降"
                trigger_values = [f"{v:.3f}" for v in window]
                return {
                    'triggered': True,
                    'slope': slope,
                    'direction': direction,
                    'start_record': records[start].record_id,
                    'end_record': records[start + self.window_size - 1].record_id,
                    'values': trigger_values,
                    'message': f"断面{self.get_section_label(records[0].x)}的{self.get_component_label()}移动平均斜率为{slope:.4f}，{direction}趋势超过阈值{self.slope_threshold}",
                }

        return {'triggered': False, 'reason': '斜率未超过阈值'}


@dataclass(kw_only=True)
class VolatilityWarningRule(TrendWarningRule):
    """波动预警规则（变异系数CV）"""
    window_size: int = 5
    cv_threshold: float = 0.1
    rule_type: Literal['volatility'] = 'volatility'

    def check(self, records: List[SimulationRecord]) -> Dict:
        if len(records) < self.window_size:
            return {'triggered': False, 'reason': '记录数量不足'}

        values = []
        for r in records:
            if self.component == 'bod':
                values.append(r.bod[self.section_idx])
            elif self.component == 'nh3n':
                values.append(r.nh3n[self.section_idx])
            else:
                values.append(r.do[self.section_idx])

        values = np.array(values)
        n = len(values)

        for start in range(n - self.window_size + 1):
            window = values[start:start + self.window_size]
            mean_val = np.mean(window)
            std_val = np.std(window, ddof=0)

            if abs(mean_val) < 1e-10:
                continue

            cv = std_val / abs(mean_val)

            if cv >= self.cv_threshold:
                trigger_values = [f"{v:.3f}" for v in window]
                return {
                    'triggered': True,
                    'cv': cv,
                    'mean': mean_val,
                    'std': std_val,
                    'start_record': records[start].record_id,
                    'end_record': records[start + self.window_size - 1].record_id,
                    'values': trigger_values,
                    'message': f"断面{self.get_section_label(records[0].x)}的{self.get_component_label()}最近{self.window_size}条记录变异系数CV={cv:.3f}，超过阈值{self.cv_threshold}",
                }

        return {'triggered': False, 'reason': '变异系数未超过阈值'}


@dataclass
class TrendWarningRules:
    """趋势预警规则管理器"""
    rules: TList[TrendWarningRule] = field(default_factory=list)
    _next_id: int = 1

    def add_rule(self, rule: TrendWarningRule) -> int:
        rule.rule_id = self._next_id
        self._next_id += 1
        self.rules.append(rule)
        return rule.rule_id

    def remove_rule(self, rule_id: int):
        self.rules = [r for r in self.rules if r.rule_id != rule_id]

    def check_all(self, records: List[SimulationRecord]) -> List[Dict]:
        triggered = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            if len(records) < 2:
                continue
            result = rule.check(records)
            if result['triggered']:
                result['rule_id'] = rule.rule_id
                result['rule_type'] = rule.rule_type
                result['rule_name'] = rule.rule_name or f"规则{rule.rule_id}"
                result['component'] = rule.component
                result['section_idx'] = rule.section_idx
                triggered.append(result)
        return triggered

    def clear(self):
        self.rules = []
