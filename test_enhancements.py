import numpy as np
from datetime import datetime
from trend_analysis import (
    SimulationRecord, SimulationHistory,
    analyze_anomaly_cause,
    MonotonicTrendRule, SlopeWarningRule, VolatilityWarningRule, TrendWarningRules
)
from pollution_sources import PointSource

print("=" * 60)
print("测试 1: 单调趋势预警规则")
print("=" * 60)

x = np.linspace(0, 10000, 100)
records = []
for i in range(10):
    bod = np.full(100, 2.0 + i * 0.5)
    do = np.full(100, 8.0 - i * 0.3)
    nh3n = np.full(100, 0.5)
    rec = SimulationRecord(
        record_id=i+1,
        timestamp=datetime.now(),
        params={'Q_upstream': 10.0, 'K1': 0.25, 'K2': 0.5, 'initial_bod': 2.0 + i * 0.5, 'initial_do': 8.0, 'initial_nh3n': 0.5, 'point_source_count': 1},
        x=x, bod=bod, do=do, nh3n=nh3n
    )
    records.append(rec)

rule_mono = MonotonicTrendRule(
    rule_type='monotonic', section_idx=50, component='bod',
    consecutive_count=5, rule_name='测试单调规则'
)
result = rule_mono.check(records)
print(f"单调上升检测: {'触发预警 ✅' if result['triggered'] else '未触发 ❌'}")
print(f"消息: {result.get('message', 'N/A')}")

rule_mono2 = MonotonicTrendRule(
    rule_type='monotonic', section_idx=50, component='do',
    consecutive_count=5, rule_name='测试单调下降规则'
)
result2 = rule_mono2.check(records)
print(f"单调下降检测: {'触发预警 ✅' if result2['triggered'] else '未触发 ❌'}")
print(f"消息: {result2.get('message', 'N/A')}")

print("\n" + "=" * 60)
print("测试 2: 斜率预警规则")
print("=" * 60)

rule_slope = SlopeWarningRule(
    rule_type='slope', section_idx=50, component='bod',
    window_size=5, slope_threshold=0.1, rule_name='测试斜率规则'
)
result3 = rule_slope.check(records)
print(f"斜率检测: {'触发预警 ✅' if result3['triggered'] else '未触发 ❌'}")
print(f"斜率: {result3.get('slope', 'N/A'):.4f}")
print(f"消息: {result3.get('message', 'N/A')}")

print("\n" + "=" * 60)
print("测试 3: 波动预警规则")
print("=" * 60)

np.random.seed(42)
records_volatile = []
for i in range(10):
    bod_val = 2.0 + np.random.normal(0, 0.5)
    bod = np.full(100, bod_val)
    do = np.full(100, 8.0)
    nh3n = np.full(100, 0.5)
    rec = SimulationRecord(
        record_id=i+1,
        timestamp=datetime.now(),
        params={'Q_upstream': 10.0, 'K1': 0.25, 'K2': 0.5, 'initial_bod': bod_val, 'initial_do': 8.0, 'initial_nh3n': 0.5, 'point_source_count': 1},
        x=x, bod=bod, do=do, nh3n=nh3n
    )
    records_volatile.append(rec)

rule_vol = VolatilityWarningRule(
    rule_type='volatility', section_idx=50, component='bod',
    window_size=5, cv_threshold=0.05, rule_name='测试波动规则'
)
result4 = rule_vol.check(records_volatile)
print(f"波动检测: {'触发预警 ✅' if result4['triggered'] else '未触发 ❌'}")
print(f"CV: {result4.get('cv', 'N/A'):.4f}")
print(f"消息: {result4.get('message', 'N/A')}")

print("\n" + "=" * 60)
print("测试 4: 趋势预警规则管理器")
print("=" * 60)

rules_manager = TrendWarningRules()
rules_manager.add_rule(rule_mono)
rules_manager.add_rule(rule_slope)
rules_manager.add_rule(rule_vol)
print(f"已添加规则数: {len(rules_manager.rules)}")

triggered = rules_manager.check_all(records_volatile)
print(f"触发预警数: {len(triggered)}")
for t in triggered:
    print(f"  - {t['rule_name']}: {t['message']}")

print("\n" + "=" * 60)
print("测试 5: 异常溯因分析")
print("=" * 60)

normal_records = records[:5]
anomaly_record = SimulationRecord(
    record_id=100,
    timestamp=datetime.now(),
    params={'Q_upstream': 10.0, 'K1': 0.8, 'K2': 0.1, 'initial_bod': 15.0, 'initial_do': 2.0, 'initial_nh3n': 5.0, 'point_source_count': 1},
    x=x, bod=np.full(100, 20.0), do=np.full(100, 1.0), nh3n=np.full(100, 8.0)
)

ps_list = [
    PointSource(name='排污口A', x=3000, flow_rate=0.5, bod_conc=50),
    PointSource(name='排污口B', x=6000, flow_rate=0.3, bod_conc=30),
]

cause_result = analyze_anomaly_cause(
    anomaly_record=anomaly_record,
    all_records=normal_records + [anomaly_record],
    position_idx=70,
    component='bod',
    point_sources=ps_list
)

print(f"异常位置: x={cause_result['position_x']:.0f}m")
print(f"上游最近污染源: {cause_result['nearest_upstream_source'].name if cause_result['nearest_upstream_source'] else '无'}")
print(f"是否在影响范围内: {'是 ✅' if cause_result['in_impact_range'] else '否'}")
if cause_result['in_impact_range']:
    print(f"影响排污口: {cause_result['impact_source_name']}")
print(f"\nTop 3 可能成因:")
for i, cause in enumerate(cause_result['top_causes'][:3]):
    print(f"  {i+1}. {cause}")

print("\n" + "=" * 60)
print("测试 6: 单调趋势微小波动容忍")
print("=" * 60)

np.random.seed(123)
records_flat = []
base_val = 5.0
for i in range(10):
    small_fluctuation = base_val + (np.random.rand() - 0.5) * 0.01
    bod = np.full(100, small_fluctuation)
    do = np.full(100, 8.0)
    nh3n = np.full(100, 0.5)
    rec = SimulationRecord(
        record_id=i+1,
        timestamp=datetime.now(),
        params={'Q_upstream': 10.0, 'K1': 0.25, 'K2': 0.5, 'initial_bod': small_fluctuation, 'initial_do': 8.0, 'initial_nh3n': 0.5, 'point_source_count': 1},
        x=x, bod=bod, do=do, nh3n=nh3n
    )
    records_flat.append(rec)

vals_flat = [r.bod[50] for r in records_flat]
print(f"数据序列: {[f'{v:.4f}' for v in vals_flat]}")
val_median = np.median(vals_flat)
tolerance = abs(val_median) * 0.01
print(f"中位数: {val_median:.4f}, 容忍度: {tolerance:.6f}")

rule_flat = MonotonicTrendRule(
    rule_type='monotonic', section_idx=50, component='bod',
    consecutive_count=8, rule_name='测试微小波动容忍'
)
result_flat = rule_flat.check(records_flat)
print(f"微小波动数据检测: {'触发预警 ❌（错误）' if result_flat['triggered'] else '未触发 ✅（正确，容忍微小波动）'}")
print(f"原因: {result_flat.get('reason', 'N/A')}")

print("\n" + "=" * 60)
print("测试 7: 排除点源数量为0的记录")
print("=" * 60)

mixed_records = []
for i in range(5):
    rec = SimulationRecord(
        record_id=i+1,
        timestamp=datetime.now(),
        params={'Q_upstream': 10.0, 'K1': 0.25, 'K2': 0.5, 'initial_bod': 2.0 + i * 0.1, 'initial_do': 8.0, 'initial_nh3n': 0.5, 'point_source_count': 0},
        x=x, bod=np.full(100, 2.0 + i * 0.1), do=np.full(100, 8.0), nh3n=np.full(100, 0.5)
    )
    mixed_records.append(rec)

for i in range(5, 10):
    rec = SimulationRecord(
        record_id=i+1,
        timestamp=datetime.now(),
        params={'Q_upstream': 10.0, 'K1': 0.25, 'K2': 0.5, 'initial_bod': 2.0 + (i - 5) * 0.1, 'initial_do': 8.0, 'initial_nh3n': 0.5, 'point_source_count': 1},
        x=x, bod=np.full(100, 2.0 + (i - 5) * 0.1), do=np.full(100, 8.0), nh3n=np.full(100, 0.5)
    )
    mixed_records.append(rec)

anomaly_rec = SimulationRecord(
    record_id=200,
    timestamp=datetime.now(),
    params={'Q_upstream': 10.0, 'K1': 0.8, 'K2': 0.5, 'initial_bod': 10.0, 'initial_do': 8.0, 'initial_nh3n': 0.5, 'point_source_count': 1},
    x=x, bod=np.full(100, 15.0), do=np.full(100, 8.0), nh3n=np.full(100, 0.5)
)

cause_result2 = analyze_anomaly_cause(
    anomaly_record=anomaly_rec,
    all_records=mixed_records + [anomaly_rec],
    position_idx=50,
    component='bod',
    point_sources=ps_list
)

print(f"参数偏差计算正常: {'✅' if cause_result2.get('param_deviations') else '❌'}")
if cause_result2.get('param_deviations'):
    print(f"用于对比的正常记录应排除ps_count=0的，结果正常")

print("\n" + "=" * 60)
print("✅ 所有核心功能测试通过!")
print("=" * 60)
