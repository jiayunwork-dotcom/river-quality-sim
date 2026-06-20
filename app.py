
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from river_channel import RiverChannel, CrossSection, Tributary
from hydrodynamics import Hydrodynamics
from water_quality import WaterQualityModel, WaterQualityParams
from pollution_sources import SourceManager, PointSource, NonPointSource, AccidentalSource, ReleaseType
from simulation_engine import RiverSimulation
from calibration import ParameterCalibration, CalibrationData
from scenario_analysis import Scenario, ScenarioManager
from report_generator import ReportGenerator
from warning_module import (
    WarningThreshold, WarningRules, EmergencyMeasure,
    WarningHistory, WARNING_NORMAL, WARNING_BLUE, WARNING_ORANGE, WARNING_RED,
    WARNING_NAMES, WARNING_COLORS, WARNING_COLORS_LIGHT,
    evaluate_warnings, plot_river_warning_schematic,
    plot_dashboard_ring_chart, build_stats_cards_data,
    run_emergency_simulation,
    EmergencyPlan, PlanSimulationResult, RecommendationResult,
    run_batch_emergency_simulations, recommend_optimal_plan,
    plot_river_warning_schematic_multi_plan, PLAN_LINE_COLORS,
)

rcParams['font.sans-serif'] = [
    'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei',
    'Arial Unicode MS', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'DejaVu Sans'
]
rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')

st.set_page_config(
    page_title="河流水质动态模拟系统",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

WATER_QUALITY_STANDARDS = {
    'bod': 4.0,
    'do': 5.0,
    'nh3n': 1.0,
    'cod': 20.0,
}

STANDARD_NAMES = {
    'bod': 'BOD',
    'do': 'DO',
    'nh3n': 'NH3-N',
    'cod': 'COD',
}


def compute_compliance_at_point(bod_val, do_val, nh3n_val):
    """计算单个点的三项指标是否全部达标"""
    bod_ok = bod_val <= WATER_QUALITY_STANDARDS['bod']
    do_ok = do_val >= WATER_QUALITY_STANDARDS['do']
    nh3n_ok = nh3n_val <= WATER_QUALITY_STANDARDS['nh3n']
    return bod_ok and do_ok and nh3n_ok


def compute_compliance_array(result):
    """计算所有网格点的达标状态数组"""
    x = result['x']
    n = len(x)
    compliance = np.zeros(n, dtype=bool)
    for i in range(n):
        compliance[i] = compute_compliance_at_point(
            result['bod'][i], result['do'][i], result['nh3n'][i]
        )
    return compliance


def compute_sliding_compliance_rate(result, window=500.0):
    """计算滑动窗口达标率曲线

    以当前断面为中心，前后各取window米范围内的所有网格点，
    统计这些点中三项指标全部达标的比例。
    """
    x = result['x']
    n = len(x)
    compliance = compute_compliance_array(result)
    rates = np.zeros(n)

    for i in range(n):
        x_center = x[i]
        mask = (x >= x_center - window) & (x <= x_center + window)
        window_points = compliance[mask]
        if len(window_points) > 0:
            rates[i] = np.sum(window_points) / len(window_points) * 100
        else:
            rates[i] = 100.0

    return x, rates


def find_threshold_crossings(x, rates, down_threshold=60.0, up_threshold=80.0):
    """找到达标率首次跌破阈值和首次回升到阈值的位置"""
    first_below = None
    first_above = None

    below_started = False
    for i in range(1, len(rates)):
        if rates[i - 1] >= down_threshold and rates[i] < down_threshold and first_below is None:
            first_below = (x[i], rates[i])
            below_started = True
        if below_started and rates[i - 1] < up_threshold and rates[i] >= up_threshold and first_above is None:
            first_above = (x[i], rates[i])
            break

    return first_below, first_above


def plot_compliance_curve(x, rates, first_below=None, first_above=None):
    """绘制达标率沿程变化曲线"""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, rates, color='#27ae60', linewidth=2.5, label='达标率')
    ax.fill_between(x, rates, 0, color='#27ae60', alpha=0.15)

    ax.axhline(y=60, color='#e67e22', linestyle='--', alpha=0.8, linewidth=1.5, label='60%警戒线')
    ax.axhline(y=80, color='#2980b9', linestyle='--', alpha=0.8, linewidth=1.5, label='80%达标线')

    if first_below is not None:
        ax.scatter([first_below[0]], [first_below[1]], color='#e74c3c', s=120, zorder=5,
                   edgecolor='black', linewidth=1.5)
        ax.annotate(f"首次跌破60%\nx={first_below[0]:.0f}m, {first_below[1]:.1f}%",
                    xy=(first_below[0], first_below[1]),
                    xytext=(first_below[0] + 300, first_below[1] - 15),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdecea', edgecolor='#e74c3c'))

    if first_above is not None:
        ax.scatter([first_above[0]], [first_above[1]], color='#16a085', s=120, zorder=5,
                   edgecolor='black', linewidth=1.5)
        ax.annotate(f"首次回升到80%\nx={first_above[0]:.0f}m, {first_above[1]:.1f}%",
                    xy=(first_above[0], first_above[1]),
                    xytext=(first_above[0] + 300, first_above[1] + 8),
                    fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='#138d75', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f8f5', edgecolor='#16a085'))

    ax.set_xlabel('河流距离 (m)', fontsize=11)
    ax.set_ylabel('达标率 (%)', fontsize=11)
    ax.set_title('达标率沿程变化曲线 (±500m滑动窗口)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')

    return fig


def generate_compliance_summary(result):
    """生成水质达标评估摘要数据"""
    x = result['x']
    total_length = x[-1] - x[0]

    x_rates, rates = compute_sliding_compliance_rate(result)
    avg_rate = np.mean(rates)

    compliance = compute_compliance_array(result)
    non_compliance_mask = ~compliance

    worst_idx = None
    worst_score = float('inf')
    for i in range(len(x)):
        score = 0
        if result['bod'][i] > WATER_QUALITY_STANDARDS['bod']:
            score += (result['bod'][i] / WATER_QUALITY_STANDARDS['bod'] - 1) * 100
        if result['do'][i] < WATER_QUALITY_STANDARDS['do']:
            score += (WATER_QUALITY_STANDARDS['do'] / result['do'][i] - 1) * 100 if result['do'][i] > 0 else 1000
        if result['nh3n'][i] > WATER_QUALITY_STANDARDS['nh3n']:
            score += (result['nh3n'][i] / WATER_QUALITY_STANDARDS['nh3n'] - 1) * 100
        if score > 0 and score < worst_score:
            worst_score = score
            worst_idx = i

    worst_x = x[worst_idx] if worst_idx is not None else x[0]
    worst_items = []
    if worst_idx is not None:
        if result['bod'][worst_idx] > WATER_QUALITY_STANDARDS['bod']:
            worst_items.append('BOD')
        if result['do'][worst_idx] < WATER_QUALITY_STANDARDS['do']:
            worst_items.append('DO')
        if result['nh3n'][worst_idx] > WATER_QUALITY_STANDARDS['nh3n']:
            worst_items.append('NH3-N')

    if not worst_items:
        worst_items = ['无']

    exceed_segments_length = 0.0
    if np.any(non_compliance_mask):
        dx = x[1] - x[0]
        exceed_segments_length = np.sum(non_compliance_mask) * dx

    exceed_ratio = exceed_segments_length / total_length * 100 if total_length > 0 else 0

    bod_exceed_total = np.sum(result['bod'] > WATER_QUALITY_STANDARDS['bod'])
    do_below_total = np.sum(result['do'] < WATER_QUALITY_STANDARDS['do'])
    nh3n_exceed_total = np.sum(result['nh3n'] > WATER_QUALITY_STANDARDS['nh3n'])

    most_severe_comp = 'BOD'
    most_severe_pct = 0
    if bod_exceed_total > 0:
        max_excess = (np.max(result['bod']) / WATER_QUALITY_STANDARDS['bod'] - 1) * 100
        most_severe_pct = max_excess
    if do_below_total > 0:
        min_do = np.min(result['do'])
        excess = (WATER_QUALITY_STANDARDS['do'] / min_do - 1) * 100 if min_do > 0 else 999
        if excess > most_severe_pct:
            most_severe_pct = excess
            most_severe_comp = 'DO'
    if nh3n_exceed_total > 0:
        max_excess = (np.max(result['nh3n']) / WATER_QUALITY_STANDARDS['nh3n'] - 1) * 100
        if max_excess > most_severe_pct:
            most_severe_pct = max_excess
            most_severe_comp = 'NH3-N'

    non_comp_indices = np.where(non_compliance_mask)[0]
    start_seg = end_seg = '-'
    if len(non_comp_indices) > 0:
        start_seg = f"{x[non_comp_indices[0]]:.0f}m"
        end_seg = f"{x[non_comp_indices[-1]]:.0f}m"

    if most_severe_pct == 0:
        conclusion = "该河段水质整体满足地表水III类标准，水环境质量良好。"
    else:
        conclusion = (f"该河段{most_severe_comp}超标严重（最大超标{most_severe_pct:.1f}%），"
                      f"建议重点治理{start_seg}至{end_seg}段。")

    return {
        'avg_compliance_rate': avg_rate,
        'worst_position': worst_x,
        'worst_items': worst_items,
        'exceed_length_ratio': exceed_ratio,
        'exceed_length': exceed_segments_length,
        'conclusion': conclusion,
        'most_severe_comp': most_severe_comp,
        'most_severe_pct': most_severe_pct,
        'start_seg': start_seg,
        'end_seg': end_seg,
    }


def init_session_state():
    """初始化会话状态"""
    if 'simulation' not in st.session_state:
        sim = RiverSimulation()
        sim.setup_default_channel()
        st.session_state.simulation = sim

    if 'result' not in st.session_state:
        st.session_state.result = None

    if 'unsteady_result' not in st.session_state:
        st.session_state.unsteady_result = None

    if 'scenario_manager' not in st.session_state:
        st.session_state.scenario_manager = ScenarioManager()

    if 'calibration_data' not in st.session_state:
        st.session_state.calibration_data = CalibrationData(x=[])

    if 'auto_run' not in st.session_state:
        st.session_state.auto_run = True

    if 'warning_rules' not in st.session_state:
        st.session_state.warning_rules = WarningRules()

    if 'warning_data' not in st.session_state:
        st.session_state.warning_data = None

    if 'warning_history' not in st.session_state:
        st.session_state.warning_history = WarningHistory(max_records=20)

    if 'emergency_measures' not in st.session_state:
        st.session_state.emergency_measures = []

    if 'emergency_result' not in st.session_state:
        st.session_state.emergency_result = None

    if 'emergency_warning_data' not in st.session_state:
        st.session_state.emergency_warning_data = None

    if 'emergency_sim' not in st.session_state:
        st.session_state.emergency_sim = None

    if 'warning_hover_idx' not in st.session_state:
        st.session_state.warning_hover_idx = None

    if 'steady_sim_params' not in st.session_state:
        st.session_state.steady_sim_params = {}

    if 'batch_plan_configs' not in st.session_state:
        st.session_state.batch_plan_configs = []

    if 'batch_plan_results' not in st.session_state:
        st.session_state.batch_plan_results = None

    if 'batch_plan_batch_id' not in st.session_state:
        st.session_state.batch_plan_batch_id = None

    if 'batch_recommend_mode' not in st.session_state:
        st.session_state.batch_recommend_mode = 'cost_min'


def plot_water_quality_profile(result, components=None):
    """绘制沿程水质剖面图"""
    if components is None:
        components = ['bod', 'do', 'nh3n', 'cod']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    color_map = {
        'bod': '#e74c3c',
        'do': '#3498db',
        'nh3n': '#f39c12',
        'cod': '#9b59b6',
    }

    name_map = {
        'bod': 'BOD (mg/L)',
        'do': 'DO (mg/L)',
        'nh3n': 'NH3-N (mg/L)',
        'cod': 'COD (mg/L)',
    }

    x = result['x']

    for idx, comp in enumerate(components):
        ax = axes[idx]
        data = result[comp]

        scheme = result.get('wq_scheme', 'upwind')
        label_suffix = f" ({scheme.upper()}方案)" if idx == 0 else ""

        ax.plot(x, data, color=color_map[comp], linewidth=2, label=name_map[comp] + label_suffix)

        if comp in WATER_QUALITY_STANDARDS:
            ax.axhline(y=WATER_QUALITY_STANDARDS[comp], color='green',
                       linestyle='--', alpha=0.7, label=f'III类标准')

        if comp == 'do':
            ax.fill_between(x, 0, 2, where=data < 2, color='red', alpha=0.2, label='严重缺氧')
            ax.fill_between(x, 2, 5, where=(data >= 2) & (data < 5), color='orange', alpha=0.2, label='轻度缺氧')

        ax.set_xlabel('河流距离 (m)')
        ax.set_ylabel(name_map[comp])
        ax.set_title(f'沿程{name_map[comp].split(" ")[0]}浓度分布')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_water_level_profile(result):
    """绘制水面线图"""
    fig, ax = plt.subplots(figsize=(10, 4))

    x = result['x']
    water_level = result['water_level']
    z = result['z']
    h = result['h']

    ax.plot(x, water_level, color='#3498db', linewidth=2, label='水面线')
    ax.plot(x, z, color='#7f8c8d', linewidth=1.5, linestyle='--', label='河床高程')
    ax.fill_between(x, z, water_level, color='#3498db', alpha=0.3)

    ax.set_xlabel('河流距离 (m)')
    ax.set_ylabel('高程 (m)')
    ax.set_title('河道水面线图')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_oxygen_sag_curve(result):
    """绘制氧垂曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = result['x']
    do = result['do']
    bod = result['bod']

    ax.plot(x, do, color='#3498db', linewidth=2, label='DO浓度')
    ax.plot(x, bod, color='#e74c3c', linewidth=2, label='BOD浓度', linestyle='--')

    ax.axhline(y=5.0, color='green', linestyle='--', alpha=0.7, label='III类标准 (5mg/L)')
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='严重缺氧线 (2mg/L)')

    if result.get('critical_x') is not None and result.get('critical_do') is not None:
        ax.scatter([result['critical_x']], [result['critical_do']],
                   color='red', s=100, zorder=5, label=f'临界点')
        ax.annotate(f"临界点\nx={result['critical_x']:.1f}m\nDO={result['critical_do']:.2f}mg/L",
                    xy=(result['critical_x'], result['critical_do']),
                    xytext=(result['critical_x'] + 500, result['critical_do'] + 1),
                    arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('河流距离 (m)')
    ax.set_ylabel('浓度 (mg/L)')
    ax.set_title('氧垂曲线 (BOD-DO耦合)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_concentration_heatmap(result, component='bod'):
    """绘制时间-距离浓度等值线图"""
    x = result['x']
    t = result['t']
    data = result[component]

    fig, ax = plt.subplots(figsize=(12, 6))

    T, X = np.meshgrid(t / 3600, x)
    im = ax.pcolormesh(T, X, data.T, cmap='YlOrRd', shading='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{component.upper()} 浓度 (mg/L)')

    ax.set_xlabel('时间 (小时)')
    ax.set_ylabel('河流距离 (m)')
    ax.set_title(f'{component.upper()} 时间-距离浓度分布')

    return fig


def plot_scenario_comparison(results, component='bod'):
    """绘制多情景对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        ax.plot(result['x'], result[component], color=color,
                linewidth=2, label=result['scenario_name'])

    if component in WATER_QUALITY_STANDARDS:
        ax.axhline(y=WATER_QUALITY_STANDARDS[component], color='green',
                   linestyle='--', alpha=0.7, label='III类标准')

    name_map = {
        'bod': 'BOD', 'do': 'DO', 'nh3n': 'NH3-N', 'cod': 'COD'
    }

    ax.set_xlabel('河流距离 (m)')
    ax.set_ylabel(f'{name_map.get(component, component)} 浓度 (mg/L)')
    ax.set_title(f'多情景对比 - {name_map.get(component, component)}沿程分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_scenario_compliance_comparison(results, scenario_manager=None):
    """绘制多情景达标率对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    solid_linestyles = ['-', '--', ':', '-.']

    baseline_names = set()
    if scenario_manager is not None:
        for s in scenario_manager.scenarios:
            if s.is_baseline:
                baseline_names.add(s.name)

    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        x_rates, rates = compute_sliding_compliance_rate(result)

        scenario_name = result.get('scenario_name', f'情景{i+1}')
        is_baseline = scenario_name in baseline_names

        if is_baseline:
            linestyle = '-'
            linewidth = 3
            label = f"{scenario_name} (基准) — 实线"
        else:
            linestyle = solid_linestyles[(i % (len(solid_linestyles) - 1)) + 1]
            linewidth = 2.2
            style_tag = {'--': '虚线', ':': '点线', '-.': '点划线'}
            label = f"{scenario_name} — {style_tag.get(linestyle, '对比线')}"

        ax.plot(x_rates, rates, color=color, linestyle=linestyle,
                linewidth=linewidth, label=label, alpha=0.9)

    ax.axhline(y=60, color='#e67e22', linestyle='--', alpha=0.6, linewidth=1.2, label='60%警戒线')
    ax.axhline(y=80, color='#2980b9', linestyle='--', alpha=0.6, linewidth=1.2, label='80%达标线')
    ax.fill_between(x_rates, 0, 60, color='#e74c3c', alpha=0.05)
    ax.fill_between(x_rates, 60, 80, color='#f39c12', alpha=0.05)
    ax.fill_between(x_rates, 80, 100, color='#27ae60', alpha=0.05)

    ax.set_xlabel('河流距离 (m)', fontsize=11)
    ax.set_ylabel('达标率 (%)', fontsize=11)
    ax.set_title('多情景达标率沿程对比 (±500m滑动窗口)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='best', fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')

    return fig


def sidebar_river_setup():
    """侧边栏：河道设置"""
    st.sidebar.header("🌊 河道设置")

    sim = st.session_state.simulation

    n_sections = st.sidebar.number_input("断面数量", min_value=1, max_value=10, value=2)

    channel = RiverChannel()

    for i in range(n_sections):
        st.sidebar.subheader(f"断面 {i+1}")
        x = st.sidebar.number_input(f"位置 (m) - 断面{i+1}", value=float(i * 5000), key=f"x_{i}")

        shape = st.sidebar.selectbox(
            f"断面形状 - 断面{i+1}",
            ['rectangular', 'trapezoidal'],
            key=f"shape_{i}",
            format_func=lambda x: '矩形' if x == 'rectangular' else '梯形'
        )

        bottom_width = st.sidebar.slider(f"底宽 (m) - 断面{i+1}",
                                         5.0, 100.0, 20.0, key=f"bw_{i}")

        side_slope = 1.5
        if shape == 'trapezoidal':
            side_slope = st.sidebar.slider(f"边坡系数 - 断面{i+1}",
                                           0.5, 5.0, 2.0, key=f"ss_{i}")

        slope = st.sidebar.slider(f"底坡 - 断面{i+1}",
                                  0.0001, 0.01, 0.0005,
                                  format="%.4f", key=f"slope_{i}")

        manning_n = st.sidebar.slider(f"曼宁系数 - 断面{i+1}",
                                      0.01, 0.1, 0.03,
                                      format="%.3f", key=f"n_{i}")

        section = CrossSection(
            x=x, shape=shape, bottom_width=bottom_width,
            side_slope=side_slope, slope=slope, manning_n=manning_n
        )
        channel.add_section(section)

    sim.channel = channel
    sim.hydro = Hydrodynamics(channel)

    st.sidebar.subheader("支流设置")
    n_tribs = st.sidebar.number_input("支流数量", min_value=0, max_value=5, value=0)

    for i in range(n_tribs):
        st.sidebar.caption(f"支流 {i+1}")
        trib_x = st.sidebar.number_input(f"位置 (m) - 支流{i+1}", value=3000.0, key=f"trib_x_{i}")
        trib_flow = st.sidebar.slider(f"流量 (m³/s) - 支流{i+1}",
                                      0.0, 20.0, 2.0, key=f"trib_q_{i}")
        trib_bod = st.sidebar.slider(f"BOD (mg/L) - 支流{i+1}",
                                     0.0, 50.0, 5.0, key=f"trib_bod_{i}")

        trib = Tributary(x=trib_x, flow_rate=trib_flow, bod_conc=trib_bod)
        sim.channel.add_tributary(trib)


def sidebar_wq_params():
    """侧边栏：水质参数"""
    st.sidebar.header("🧪 水质参数")

    K1 = st.sidebar.slider("BOD衰减系数 K1 (1/d)", 0.01, 2.0, 0.25, 0.01, key="K1_slider")
    K2 = st.sidebar.slider("复氧系数 K2 (1/d)", 0.01, 5.0, 0.5, 0.01, key="K2_slider")
    Dx = st.sidebar.slider("扩散系数 Dx (m²/s)", 0.0, 100.0, 10.0, 0.5, key="Dx_slider")
    D_sat = st.sidebar.slider("饱和溶解氧 (mg/L)", 5.0, 15.0, 9.5, 0.1, key="Dsat_slider")

    K_nh3n = st.sidebar.slider("氨氮衰减系数 (1/d)", 0.01, 1.0, 0.1, 0.01, key="Knh3n_slider")
    K_cod = st.sidebar.slider("COD衰减系数 (1/d)", 0.01, 1.0, 0.15, 0.01, key="Kcod_slider")

    sim = st.session_state.simulation
    sim.set_water_quality_params(
        K1=K1, K2=K2, Dx=Dx, D_O_sat=D_sat,
        K_nh3n=K_nh3n, K_cod=K_cod
    )

    st.session_state.current_wq_params = {
        'K1': K1, 'K2': K2, 'Dx': Dx, 'D_O_sat': D_sat,
        'K_nh3n': K_nh3n, 'K_cod': K_cod
    }


def sidebar_sources():
    """侧边栏：污染源设置"""
    st.sidebar.header("🏭 污染源设置")

    sim = st.session_state.simulation
    sim.source_manager = SourceManager()

    source_type = st.sidebar.selectbox(
        "污染源类型",
        ['point', 'nonpoint', 'accidental'],
        format_func=lambda x: {'point': '点源', 'nonpoint': '面源', 'accidental': '突发泄漏'}[x],
        key="source_type_select"
    )

    if source_type == 'point':
        n_sources = st.sidebar.number_input("点源数量", min_value=0, max_value=10, value=1, key="n_points")

        for i in range(n_sources):
            with st.sidebar.expander(f"点源 {i+1}", expanded=(i == 0)):
                name = st.text_input(f"名称 - 点源{i+1}", f"排污口{i+1}", key=f"ps_name_{i}")
                x = st.number_input(f"位置 (m) - 点源{i+1}", 0.0, 20000.0, 3000.0, key=f"ps_x_{i}")
                q = st.slider(f"排放流量 (m³/s) - 点源{i+1}", 0.0, 5.0, 0.5, key=f"ps_q_{i}")
                bod = st.slider(f"BOD (mg/L) - 点源{i+1}", 0.0, 200.0, 50.0, key=f"ps_bod_{i}")
                do = st.slider(f"DO (mg/L) - 点源{i+1}", 0.0, 10.0, 2.0, key=f"ps_do_{i}")
                nh3n = st.slider(f"NH3-N (mg/L) - 点源{i+1}", 0.0, 50.0, 10.0, key=f"ps_nh3n_{i}")
                cod = st.slider(f"COD (mg/L) - 点源{i+1}", 0.0, 200.0, 80.0, key=f"ps_cod_{i}")

                sim.add_point_source(name, x, q, bod, do, nh3n, cod)

    elif source_type == 'nonpoint':
        n_sources = st.sidebar.number_input("面源数量", min_value=0, max_value=5, value=1, key="n_nonpoints")

        for i in range(n_sources):
            with st.sidebar.expander(f"面源 {i+1}", expanded=(i == 0)):
                name = st.text_input(f"名称 - 面源{i+1}", f"农业面源{i+1}", key=f"nps_name_{i}")
                start_x = st.number_input(f"起始位置 (m) - 面源{i+1}", 0.0, 20000.0, 2000.0, key=f"nps_start_{i}")
                end_x = st.number_input(f"终止位置 (m) - 面源{i+1}", 0.0, 20000.0, 6000.0, key=f"nps_end_{i}")
                area = st.slider(f"汇水面积 (km²) - 面源{i+1}", 0.1, 50.0, 5.0, key=f"nps_area_{i}")
                bod_load = st.slider(f"BOD负荷 (kg/km²·d) - 面源{i+1}", 0.0, 100.0, 10.0, key=f"nps_bod_{i}")
                nh3n_load = st.slider(f"NH3-N负荷 (kg/km²·d) - 面源{i+1}", 0.0, 50.0, 2.0, key=f"nps_nh3n_{i}")
                cod_load = st.slider(f"COD负荷 (kg/km²·d) - 面源{i+1}", 0.0, 100.0, 15.0, key=f"nps_cod_{i}")

                sim.add_nonpoint_source(name, start_x, end_x,
                                        area * 1e6,
                                        bod_load,
                                        nh3n_load,
                                        cod_load)

    else:
        n_sources = st.sidebar.number_input("泄漏源数量", min_value=0, max_value=3, value=0, key="n_accidental")

        for i in range(n_sources):
            with st.sidebar.expander(f"泄漏源 {i+1}"):
                name = st.text_input(f"名称 - 泄漏{i+1}", f"突发泄漏{i+1}", key=f"acc_name_{i}")
                x = st.number_input(f"位置 (m) - 泄漏{i+1}", 0.0, 20000.0, 5000.0, key=f"acc_x_{i}")
                release_type = st.selectbox(
                    f"排放类型 - 泄漏{i+1}",
                    ['continuous', 'instantaneous'],
                    key=f"acc_type_{i}",
                    format_func=lambda x: '持续排放' if x == 'continuous' else '瞬时排放'
                )
                mass_bod = st.slider(f"BOD总量 (kg) - 泄漏{i+1}", 10.0, 10000.0, 1000.0, key=f"acc_mass_{i}")
                mass_nh3n = st.slider(f"NH3-N总量 (kg) - 泄漏{i+1}", 10.0, 5000.0, 200.0, key=f"acc_mass_nh3n_{i}")
                mass_cod = st.slider(f"COD总量 (kg) - 泄漏{i+1}", 10.0, 10000.0, 1500.0, key=f"acc_mass_cod_{i}")
                duration = st.slider(f"排放时长 (小时) - 泄漏{i+1}", 0.1, 24.0, 2.0, key=f"acc_dur_{i}") if release_type == 'continuous' else 0
                flow_rate = st.slider(f"排放流量 (m³/s) - 泄漏{i+1}", 0.0, 2.0, 0.5, key=f"acc_flow_{i}") if release_type == 'continuous' else 0

                sim.add_accidental_source(
                    name, x, release_type,
                    total_mass_bod=mass_bod * 1000,
                    total_mass_nh3n=mass_nh3n * 1000,
                    total_mass_cod=mass_cod * 1000,
                    release_duration=duration * 3600 if release_type == 'continuous' else 1,
                    flow_rate=flow_rate
                )


def main_page():
    """主页面"""
    st.title("🌊 河流水质动态模拟与污染物迁移预测系统")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 稳态模拟", "⏱️ 非稳态模拟", "📈 参数率定",
        "🔄 情景分析", "⚠️ 预警与应急响应", "📋 结果表格", "📄 报告导出"
    ])

    with tab1:
        steady_simulation_tab()

    with tab2:
        unsteady_simulation_tab()

    with tab3:
        calibration_tab()

    with tab4:
        scenario_analysis_tab()

    with tab5:
        warning_emergency_tab()

    with tab6:
        results_table_tab()

    with tab7:
        report_export_tab()


def steady_simulation_tab():
    """稳态模拟标签页"""
    st.header("稳态水质模拟")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        flow_mode = st.selectbox(
            "水流模式",
            ['uniform', 'gradually_varied'],
            format_func=lambda x: {
                'uniform': '恒定均匀流',
                'gradually_varied': '恒定非均匀流'
            }[x]
        )

    with col2:
        wq_scheme = st.selectbox(
            "数值求解方案",
            ['upwind', 'crank_nicolson'],
            format_func=lambda x: {
                'upwind': '上风格式 (显式)',
                'crank_nicolson': 'Crank-Nicolson隐格式'
            }[x]
        )

    with col3:
        auto_run = st.checkbox("参数变化自动运行（推荐）", value=st.session_state.auto_run, key="auto_run_checkbox")
        st.session_state.auto_run = auto_run
        if auto_run:
            st.caption("✅ 调节滑块后图表将自动刷新")

    st.subheader("上游边界条件")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Q_upstream = st.slider("上游流量 (m³/s)", 0.1, 50.0, 10.0, 0.5, key="steady_Q")
    with col2:
        initial_bod = st.slider("上游BOD (mg/L)", 0.0, 20.0, 2.0, 0.5, key="steady_bod")
    with col3:
        initial_do = st.slider("上游DO (mg/L)", 0.0, 15.0, 8.5, 0.1, key="steady_do")
    with col4:
        initial_nh3n = st.slider("上游NH3-N (mg/L)", 0.0, 10.0, 0.5, 0.1, key="steady_nh3n")
    with col5:
        initial_cod = st.slider("上游COD (mg/L)", 0.0, 50.0, 5.0, 0.5, key="steady_cod")

    n_grid = st.slider("空间网格数", 20, 500, 100, 10, key="steady_grid")

    col_run, col_empty = st.columns([1, 3])
    with col_run:
        run_steady = st.button("🚀 运行稳态模拟", type="primary", use_container_width=True)

    sim = st.session_state.simulation
    should_run = run_steady or st.session_state.auto_run
    if should_run:
        with st.spinner("正在计算..."):
            result = sim.run_steady_simulation(
                Q_upstream=Q_upstream,
                initial_bod=initial_bod,
                initial_do=initial_do,
                initial_nh3n=initial_nh3n,
                initial_cod=initial_cod,
                flow_mode=flow_mode,
                n_grid=n_grid,
                wq_scheme=wq_scheme,
            )
            st.session_state.result = result
            st.session_state.steady_sim_params = {
                'Q_upstream': Q_upstream,
                'initial_bod': initial_bod,
                'initial_do': initial_do,
                'initial_nh3n': initial_nh3n,
                'initial_cod': initial_cod,
                'flow_mode': flow_mode,
                'n_grid': n_grid,
                'wq_scheme': wq_scheme,
            }

    if st.session_state.result is not None:
        result = st.session_state.result

        st.success(f"✅ 模拟完成！使用 {result.get('wq_scheme', wq_scheme).upper()} 数值方案")

        st.subheader("📈 沿程水质分布")
        fig_profile = plot_water_quality_profile(result)
        st.pyplot(fig_profile, use_container_width=True)

        with st.expander("✅ 达标分析", expanded=True):
            st.markdown("**地表水III类标准：** DO ≥ 5mg/L | BOD ≤ 4mg/L | NH3-N ≤ 1mg/L")
            st.caption(f"共 {len(result['x'])} 个网格点，超标单元格以红色高亮，括号内为超标倍数")

            x = result['x']
            all_indices = list(range(len(x)))

            table_data = []
            for idx in all_indices:
                xi = x[idx]
                bod_val = result['bod'][idx]
                do_val = result['do'][idx]
                nh3n_val = result['nh3n'][idx]

                bod_ok = bod_val <= WATER_QUALITY_STANDARDS['bod']
                do_ok = do_val >= WATER_QUALITY_STANDARDS['do']
                nh3n_ok = nh3n_val <= WATER_QUALITY_STANDARDS['nh3n']
                all_ok = bod_ok and do_ok and nh3n_ok

                if bod_ok:
                    bod_str = f"{bod_val:.2f}"
                else:
                    excess = (bod_val / WATER_QUALITY_STANDARDS['bod'] - 1) * 100
                    bod_str = f"{bod_val:.2f} 🔴(+{excess:.1f}%)"

                if do_ok:
                    do_str = f"{do_val:.2f}"
                else:
                    excess = (WATER_QUALITY_STANDARDS['do'] / do_val - 1) * 100 if do_val > 0 else 999
                    do_str = f"{do_val:.2f} 🔴(+{excess:.1f}%)"

                if nh3n_ok:
                    nh3n_str = f"{nh3n_val:.2f}"
                else:
                    excess = (nh3n_val / WATER_QUALITY_STANDARDS['nh3n'] - 1) * 100
                    nh3n_str = f"{nh3n_val:.2f} 🔴(+{excess:.1f}%)"

                status_str = "✅ 达标" if all_ok else "❌ 超标"

                table_data.append({
                    '位置 (m)': f"{xi:.0f}",
                    'BOD (mg/L)': bod_str,
                    'DO (mg/L)': do_str,
                    'NH3-N (mg/L)': nh3n_str,
                    '达标判定': status_str,
                })

            df_compliance = pd.DataFrame(table_data)

            def highlight_exceed_cells(val):
                if '🔴' in str(val):
                    return 'background-color: #fdecea; color: #c0392b; font-weight: bold'
                return ''

            styled_df = df_compliance.style.map(highlight_exceed_cells, subset=['BOD (mg/L)', 'DO (mg/L)', 'NH3-N (mg/L)'])
            styled_df = styled_df.map(
                lambda v: 'background-color: #fdecea; color: #c0392b' if '❌' in str(v) else 'background-color: #e8f8f5; color: #138d75',
                subset=['达标判定']
            )
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=420)

        st.subheader("📊 达标率沿程变化曲线")
        x_rates, rates = compute_sliding_compliance_rate(result)
        first_below, first_above = find_threshold_crossings(x_rates, rates)
        fig_compliance = plot_compliance_curve(x_rates, rates, first_below, first_above)
        st.pyplot(fig_compliance, use_container_width=True)

        col_cb1, col_cb2, col_cb3 = st.columns(3)
        with col_cb1:
            if first_below is not None:
                st.warning(f"⚠️ 达标率首次跌破60%: **x={first_below[0]:.0f}m** ({first_below[1]:.1f}%)")
            else:
                st.success("✅ 达标率全程保持在60%以上")
        with col_cb2:
            if first_above is not None:
                st.success(f"💚 达标率首次回升到80%: **x={first_above[0]:.0f}m** ({first_above[1]:.1f}%)")
            else:
                st.info("ℹ️ 达标率尚未回升到80%以上")
        with col_cb3:
            avg_rate = np.mean(rates)
            st.metric("全河段平均达标率", f"{avg_rate:.1f}%")

        st.subheader("🌊 水面线图")
        fig_water = plot_water_level_profile(result)
        st.pyplot(fig_water, use_container_width=True)

        st.subheader("💧 氧垂曲线")
        fig_oxygen = plot_oxygen_sag_curve(result)
        st.pyplot(fig_oxygen, use_container_width=True)

        if result.get('critical_x') is not None:
            st.info(
                f"**临界点信息：** 距离上游 {result['critical_x']:.1f} m 处，"
                f"最低 DO 浓度为 {result['critical_do']:.2f} mg/L"
            )

        with st.expander("📊 水质统计数据"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("最大BOD", f"{np.max(result['bod']):.2f} mg/L")
                st.metric("BOD超标率", f"{np.sum(result['bod'] > 4) / len(result['bod']) * 100:.1f} %")

            with col2:
                st.metric("最低DO", f"{np.min(result['do']):.2f} mg/L")
                st.metric("DO达标率", f"{np.sum(result['do'] >= 5) / len(result['do']) * 100:.1f} %")

            with col3:
                st.metric("最大NH3-N", f"{np.max(result['nh3n']):.2f} mg/L")
                st.metric("NH3-N超标率", f"{np.sum(result['nh3n'] > 1) / len(result['nh3n']) * 100:.1f} %")

            with col4:
                st.metric("最大COD", f"{np.max(result['cod']):.2f} mg/L")

            st.caption(f"当前参数：K1={sim.wq_model.params.K1:.3f} 1/d, K2={sim.wq_model.params.K2:.3f} 1/d, Dx={sim.wq_model.params.Dx:.1f} m²/s")


def unsteady_simulation_tab():
    """非稳态模拟标签页"""
    st.header("非稳态水质模拟")

    sim = st.session_state.simulation

    col1, col2, col3 = st.columns(3)
    with col1:
        t_total_hours = st.slider("总模拟时长 (小时)", 1, 120, 24, key="unsteady_t")
    with col2:
        dt_minutes = st.slider("时间步长 (分钟)", 1, 60, 10, key="unsteady_dt")
    with col3:
        wq_scheme = st.selectbox(
            "数值方案",
            ['upwind', 'crank_nicolson'],
            format_func=lambda x: {'upwind': '上风格式', 'crank_nicolson': 'Crank-Nicolson隐格式'}[x],
            key="unsteady_scheme"
        )

    col1, col2 = st.columns(2)
    with col1:
        Q_base = st.slider("基流流量 (m³/s)", 0.1, 50.0, 10.0, 0.5, key="unsteady_Qbase")
    with col2:
        has_flood = st.checkbox("洪水过程", value=False, key="unsteady_flood")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        initial_bod = st.slider("上游初始BOD (mg/L)", 0.0, 20.0, 2.0, 0.5, key='unsteady_bod')
    with col2:
        initial_do = st.slider("上游初始DO (mg/L)", 0.0, 15.0, 8.5, 0.1, key='unsteady_do')
    with col3:
        initial_nh3n = st.slider("上游初始NH3-N (mg/L)", 0.0, 10.0, 0.5, 0.1, key='unsteady_nh3n')
    with col4:
        initial_cod = st.slider("上游初始COD (mg/L)", 0.0, 50.0, 5.0, 0.5, key='unsteady_cod')

    n_grid = st.slider("空间网格数", 20, 200, 50, 10, key='unsteady_grid')

    t_total = t_total_hours * 3600
    dt = dt_minutes * 60

    n_steps = int(t_total / dt)
    t = np.arange(n_steps + 1) * dt

    if has_flood:
        col1, col2 = st.columns(2)
        with col1:
            flood_peak = st.slider("洪峰流量 (m³/s)", Q_base, 100.0, 30.0, 0.5, key="unsteady_peak")
        with col2:
            flood_time = st.slider("洪峰出现时间 (小时)", 0, t_total_hours, t_total_hours // 2, key="unsteady_peak_t")

        Q_upstream = Q_base + (flood_peak - Q_base) * np.exp(-((t / 3600 - flood_time) ** 2) / 16)
    else:
        Q_upstream = np.full(n_steps + 1, Q_base)

    run_unsteady = st.button("⏱️ 运行非稳态模拟", type="primary")

    if run_unsteady:
        with st.spinner("正在计算非稳态过程..."):
            result = sim.run_unsteady_simulation(
                Q_upstream=Q_upstream,
                t_total=t_total,
                dt=dt,
                initial_bod=initial_bod,
                initial_do=initial_do,
                initial_nh3n=initial_nh3n,
                initial_cod=initial_cod,
                n_grid=n_grid,
                wq_scheme=wq_scheme,
            )
            st.session_state.unsteady_result = result

            if result.get('courant_warning'):
                st.warning("⚠️ 警告：Courant数超过1，数值可能不稳定！建议减小时间步长。")

            dx = result['x'][1] - result['x'][0]
            V_max = np.max(result['V'])
            suggested_dt = sim.suggest_dt(V_max, dx)
            st.info(f"💡 建议时间步长：{suggested_dt / 60:.1f} 分钟（当前：{dt_minutes:.1f} 分钟）")

    if st.session_state.unsteady_result is not None:
        result = st.session_state.unsteady_result

        st.subheader("🔥 时间-距离浓度分布")

        comp_select = st.selectbox(
            "选择组分",
            ['bod', 'do', 'nh3n', 'cod'],
            format_func=lambda x: {'bod': 'BOD', 'do': 'DO', 'nh3n': 'NH3-N', 'cod': 'COD'}[x],
            key="unsteady_comp"
        )

        fig_heatmap = plot_concentration_heatmap(result, comp_select)
        st.pyplot(fig_heatmap, use_container_width=True)

        st.subheader("📈 时间过程线")

        time_idx = st.slider("选择时刻", 0, len(result['t']) - 1, len(result['t']) // 2, key="unsteady_time_idx")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result['x'], result[comp_select][time_idx, :], 'b-', linewidth=2)
        if comp_select in WATER_QUALITY_STANDARDS:
            ax.axhline(y=WATER_QUALITY_STANDARDS[comp_select], color='green', linestyle='--', label='III类标准')
        ax.set_xlabel('河流距离 (m)')
        ax.set_ylabel(f'{comp_select.upper()} 浓度 (mg/L)')
        ax.set_title(f't = {result["t"][time_idx] / 3600:.2f} 小时时的浓度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)


def calibration_tab():
    """参数率定标签页"""
    st.header("参数率定")

    cal = ParameterCalibration()
    sim = st.session_state.simulation

    st.subheader("📝 输入实测数据")

    data_input_format = st.radio("数据输入方式", ["手动输入", "表格编辑"], key="cal_input_format")

    x_data = []
    bod_data = []
    do_data = []

    if data_input_format == "手动输入":
        n_points = st.number_input("监测断面数量", min_value=3, max_value=20, value=5, key="cal_npoints")

        cols = st.columns(3)
        for i in range(n_points):
            with cols[0]:
                x = st.number_input(f"距离 (m) - 点{i+1}", value=float(i * 2000), key=f"cal_x_{i}")
                x_data.append(x)
            with cols[1]:
                bod = st.number_input(f"BOD (mg/L) - 点{i+1}", value=5.0 - i * 0.5, key=f"cal_bod_{i}")
                bod_data.append(bod)
            with cols[2]:
                do = st.number_input(f"DO (mg/L) - 点{i+1}", value=6.0 + i * 0.3, key=f"cal_do_{i}")
                do_data.append(do)

    else:
        df = pd.DataFrame({
            '距离(m)': [0, 2000, 4000, 6000, 8000, 10000],
            'BOD(mg/L)': [5.0, 4.2, 3.5, 3.0, 2.6, 2.3],
            'DO(mg/L)': [7.5, 6.2, 5.5, 5.2, 5.4, 5.8],
        })
        edited_df = st.data_editor(df, num_rows="dynamic", key="cal_editor")

        x_data = edited_df['距离(m)'].tolist()
        bod_data = edited_df['BOD(mg/L)'].tolist()
        do_data = edited_df['DO(mg/L)'].tolist()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        u = st.slider("河段平均流速 (m/s)", 0.1, 5.0, 0.5, 0.1, key="cal_u")
    with col2:
        L0 = st.number_input("初始BOD浓度 L0 (mg/L)", value=5.0, key="cal_L0")
    with col3:
        D0 = st.number_input("初始DO浓度 D0 (mg/L)", value=7.5, key="cal_D0")
    with col4:
        D_sat = st.number_input("饱和溶解氧 (mg/L)", value=9.5, key="cal_Dsat")

    col1, col2, col3 = st.columns(3)
    with col1:
        calibrate_k1 = st.button("🔬 率定K1 (BOD)")
    with col2:
        calibrate_k2 = st.button("🔬 率定K2 (复氧)")
    with col3:
        calibrate_joint = st.button("🔬 联合率定K1&K2", type="primary")

    if calibrate_k1 and len(x_data) >= 3:
        K1, K1_std, r2 = cal.calibrate_k1(
            np.array(x_data), np.array(bod_data), u, L0
        )

        st.success(f"✅ K1率定完成")
        col1, col2, col3 = st.columns(3)
        col1.metric("K1", f"{K1:.4f} 1/d")
        col2.metric("标准差", f"{K1_std:.4f}")
        col3.metric("R²", f"{r2:.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        x_fit = np.linspace(min(x_data), max(x_data), 100)
        bod_fit = cal.streeter_phelps_bod(x_fit, L0, K1, u)
        ax.scatter(x_data, bod_data, color='red', s=50, zorder=5, label='实测数据')
        ax.plot(x_fit, bod_fit, 'b-', linewidth=2, label='拟合曲线')
        ax.set_xlabel('河流距离 (m)')
        ax.set_ylabel('BOD浓度 (mg/L)')
        ax.set_title('BOD衰减曲线拟合')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    if calibrate_k2 and len(x_data) >= 4:
        K1_current = sim.wq_model.params.K1
        K2, K2_std, r2 = cal.calibrate_k2(
            np.array(x_data), np.array(do_data),
            u, L0, D_sat - D0, K1_current, D_sat
        )

        st.success(f"✅ K2率定完成")
        col1, col2, col3 = st.columns(3)
        col1.metric("K2", f"{K2:.4f} 1/d")
        col2.metric("标准差", f"{K2_std:.4f}")
        col3.metric("R²", f"{r2:.4f}")

    if calibrate_joint and len(x_data) >= 5:
        result = cal.calibrate_joint(
            np.array(x_data), np.array(bod_data), np.array(do_data),
            u, L0, D_sat - D0, D_sat
        )

        st.success(f"✅ 联合率定完成")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("K1", f"{result.K1:.4f} 1/d")
            st.metric("K1 95%置信区间", f"[{result.K1_ci[0]:.4f}, {result.K1_ci[1]:.4f}]")
            st.metric("BOD拟合 R²", f"{result.r_squared_bod:.4f}")

        with col2:
            st.metric("K2", f"{result.K2:.4f} 1/d")
            st.metric("K2 95%置信区间", f"[{result.K2_ci[0]:.4f}, {result.K2_ci[1]:.4f}]")
            st.metric("DO拟合 R²", f"{result.r_squared_do:.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x_fit = np.linspace(min(x_data), max(x_data), 100)
        bod_fit = cal.streeter_phelps_bod(x_fit, L0, result.K1, u)
        do_fit = cal.streeter_phelps_do(x_fit, L0, D_sat - D0, result.K1, result.K2, u, D_sat)

        ax1.scatter(x_data, bod_data, color='red', s=50, zorder=5, label='实测数据')
        ax1.plot(x_fit, bod_fit, 'b-', linewidth=2, label='拟合曲线')
        ax1.set_xlabel('河流距离 (m)')
        ax1.set_ylabel('BOD浓度 (mg/L)')
        ax1.set_title('BOD衰减曲线拟合')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(x_data, do_data, color='red', s=50, zorder=5, label='实测数据')
        ax2.plot(x_fit, do_fit, 'b-', linewidth=2, label='拟合曲线')
        ax2.set_xlabel('河流距离 (m)')
        ax2.set_ylabel('DO浓度 (mg/L)')
        ax2.set_title('DO复氧曲线拟合')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        if st.button("✅ 应用率定参数到模型", key="apply_cal_params"):
            sim.set_water_quality_params(K1=result.K1, K2=result.K2)
            st.success(f"参数已应用！K1={result.K1:.4f}, K2={result.K2:.4f}")


def scenario_analysis_tab():
    """情景分析标签页"""
    st.header("情景分析")

    sc_manager = st.session_state.scenario_manager
    sim = st.session_state.simulation

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 情景列表")

        scenario_name = st.text_input("新情景名称", "改进方案1", key="sc_new_name")
        scenario_desc = st.text_area("情景描述", "", key="sc_desc")

        st.subheader("情景参数设置")

        col_a, col_b = st.columns(2)
        with col_a:
            sc_Q = st.slider("上游流量 (m³/s)", 0.1, 50.0, 10.0, 0.5, key="sc_Q")
            sc_K1 = st.slider("K1 (1/d)", 0.01, 2.0, sim.wq_model.params.K1, 0.01, key="sc_K1")
            sc_K2 = st.slider("K2 (1/d)", 0.01, 5.0, sim.wq_model.params.K2, 0.01, key="sc_K2")
        with col_b:
            sc_bod = st.slider("上游BOD (mg/L)", 0.0, 20.0, 2.0, 0.5, key="sc_bod")
            sc_do = st.slider("上游DO (mg/L)", 0.0, 15.0, 8.5, 0.1, key="sc_do")
            sc_nh3n = st.slider("上游NH3-N (mg/L)", 0.0, 10.0, 0.5, 0.1, key="sc_nh3n")
            sc_cod = st.slider("上游COD (mg/L)", 0.0, 50.0, 5.0, 0.5, key="sc_cod")

        st.subheader("点源调整")
        n_sc_sources = st.number_input("该情景点源数量", min_value=0, max_value=10, value=0, key="sc_n_sources")

        sc_sources = []
        for i in range(n_sc_sources):
            with st.expander(f"情景点源 {i+1}"):
                src_name = st.text_input(f"名称", f"排污口{i+1}", key=f"sc_src_name_{i}")
                src_x = st.number_input(f"位置 (m)", 0.0, 20000.0, 3000.0, key=f"sc_src_x_{i}")
                src_q = st.slider(f"排放流量 (m³/s)", 0.0, 5.0, 0.5, key=f"sc_src_q_{i}")
                src_bod = st.slider(f"BOD (mg/L)", 0.0, 200.0, 50.0, key=f"sc_src_bod_{i}")
                src_do = st.slider(f"DO (mg/L)", 0.0, 10.0, 2.0, key=f"sc_src_do_{i}")
                src_nh3n = st.slider(f"NH3-N (mg/L)", 0.0, 50.0, 10.0, key=f"sc_src_nh3n_{i}")
                src_cod = st.slider(f"COD (mg/L)", 0.0, 200.0, 80.0, key=f"sc_src_cod_{i}")

                sc_sources.append({
                    'name': src_name, 'x': src_x, 'flow_rate': src_q,
                    'bod': src_bod, 'do': src_do, 'nh3n': src_nh3n, 'cod': src_cod
                })

        st.subheader("面源调整")
        n_sc_nps = st.number_input("该情景面源数量", min_value=0, max_value=5, value=0, key="sc_n_nps")

        sc_nonpoint_sources = []
        for i in range(n_sc_nps):
            with st.expander(f"情景面源 {i+1}"):
                nps_name = st.text_input(f"名称", f"农业面源{i+1}", key=f"sc_nps_name_{i}")
                nps_start = st.number_input(f"起始位置 (m)", 0.0, 20000.0, 1000.0, key=f"sc_nps_start_{i}")
                nps_end = st.number_input(f"终止位置 (m)", 0.0, 20000.0, 8000.0, key=f"sc_nps_end_{i}")
                nps_area = st.slider(f"汇水面积 (km²)", 0.1, 50.0, 5.0, key=f"sc_nps_area_{i}")
                nps_bod = st.slider(f"BOD负荷 (kg/km²·d)", 0.0, 200.0, 30.0, key=f"sc_nps_bod_{i}")
                nps_nh3n = st.slider(f"NH3-N负荷 (kg/km²·d)", 0.0, 50.0, 5.0, key=f"sc_nps_nh3n_{i}")
                nps_cod = st.slider(f"COD负荷 (kg/km²·d)", 0.0, 200.0, 50.0, key=f"sc_nps_cod_{i}")

                sc_nonpoint_sources.append({
                    'name': nps_name, 'start_x': nps_start, 'end_x': nps_end,
                    'area': nps_area * 1e6,
                    'bod_load': nps_bod, 'nh3n_load': nps_nh3n, 'cod_load': nps_cod
                })

        st.subheader("突发源调整")
        n_sc_acc = st.number_input("该情景突发源数量", min_value=0, max_value=3, value=0, key="sc_n_acc")

        sc_accidental_sources = []
        for i in range(n_sc_acc):
            with st.expander(f"情景突发源 {i+1}"):
                acc_name = st.text_input(f"名称", f"泄漏{i+1}", key=f"sc_acc_name_{i}")
                acc_x = st.number_input(f"位置 (m)", 0.0, 20000.0, 5000.0, key=f"sc_acc_x_{i}")
                acc_type = st.selectbox(
                    f"排放类型",
                    ['continuous', 'instantaneous'],
                    key=f"sc_acc_type_{i}",
                    format_func=lambda x: '持续排放' if x == 'continuous' else '瞬时排放'
                )
                acc_bod = st.slider(f"BOD总量 (kg)", 10.0, 10000.0, 2000.0, key=f"sc_acc_bod_{i}")
                acc_nh3n = st.slider(f"NH3-N总量 (kg)", 10.0, 5000.0, 300.0, key=f"sc_acc_nh3n_{i}")
                acc_cod = st.slider(f"COD总量 (kg)", 10.0, 10000.0, 1500.0, key=f"sc_acc_cod_{i}")
                if acc_type == 'continuous':
                    acc_dur = st.slider(f"排放时长 (小时)", 0.1, 24.0, 4.0, key=f"sc_acc_dur_{i}")
                    acc_flow = st.slider(f"排放流量 (m³/s)", 0.0, 2.0, 0.5, key=f"sc_acc_flow_{i}")
                else:
                    acc_dur = 1.0
                    acc_flow = 0.0

                sc_accidental_sources.append({
                    'name': acc_name, 'x': acc_x, 'release_type': acc_type,
                    'total_mass_bod': acc_bod * 1000,
                    'total_mass_nh3n': acc_nh3n * 1000,
                    'total_mass_cod': acc_cod * 1000,
                    'release_duration': acc_dur * 3600 if acc_type == 'continuous' else 1.0,
                    'flow_rate': acc_flow
                })

        is_baseline = st.checkbox("设为基准情景", value=len(sc_manager.scenarios) == 0, key="sc_baseline")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            add_custom = st.button("➕ 添加自定义情景")
        with col_btn2:
            add_current = st.button("📋 添加当前状态为情景")

        if add_custom or add_current:
            scenario = Scenario(
                name=scenario_name,
                description=scenario_desc,
                is_baseline=is_baseline
            )
            scenario.upstream_flow = sc_Q
            scenario.upstream_bod = sc_bod
            scenario.upstream_do = sc_do
            scenario.upstream_nh3n = sc_nh3n
            scenario.upstream_cod = sc_cod
            scenario.K1 = sc_K1
            scenario.K2 = sc_K2

            if add_current:
                current_ps = []
                for ps in sim.source_manager.get_point_sources():
                    current_ps.append({
                        'name': ps.name, 'x': ps.x, 'flow_rate': ps.flow_rate,
                        'bod': ps.bod_conc, 'do': ps.do_conc,
                        'nh3n': ps.nh3n_conc, 'cod': ps.cod_conc
                    })
                current_nps = []
                for nps in sim.source_manager.get_nonpoint_sources():
                    current_nps.append({
                        'name': nps.name, 'start_x': nps.start_x, 'end_x': nps.end_x,
                        'area': nps.area,
                        'bod_load': nps.bod_load, 'nh3n_load': nps.nh3n_load,
                        'cod_load': nps.cod_load
                    })
                current_acc = []
                for acc in sim.source_manager.get_accidental_sources():
                    current_acc.append({
                        'name': acc.name, 'x': acc.x,
                        'release_type': 'continuous' if acc.release_type == ReleaseType.CONTINUOUS else 'instantaneous',
                        'total_mass_bod': acc.total_mass_bod,
                        'total_mass_nh3n': acc.total_mass_nh3n,
                        'total_mass_cod': acc.total_mass_cod,
                        'release_duration': acc.release_duration,
                        'flow_rate': acc.flow_rate
                    })
                scenario.point_sources = current_ps
                scenario.nonpoint_sources = current_nps
                scenario.accidental_sources = current_acc
                src_info = f"{len(current_ps)}个点源, {len(current_nps)}个面源, {len(current_acc)}个突发源"
                sc_manager.add_scenario(scenario)
                st.success(f"情景 '{scenario_name}' 已添加（从侧边栏同步: {src_info}）")
            else:
                scenario.point_sources = sc_sources
                scenario.nonpoint_sources = sc_nonpoint_sources
                scenario.accidental_sources = sc_accidental_sources
                sc_manager.add_scenario(scenario)
                st.success(f"情景 '{scenario_name}' 已添加 (含{len(sc_sources)}个点源, {len(sc_nonpoint_sources)}个面源, {len(sc_accidental_sources)}个突发源)")

        scenario_names = sc_manager.get_scenario_names()
        if scenario_names:
            for s in sc_manager.scenarios:
                src_count = f"{len(s.point_sources)}点源/{len(s.nonpoint_sources)}面源/{len(s.accidental_sources)}突发源"
                baseline_tag = " [基准]" if s.is_baseline else ""
                st.markdown(f"**{s.name}**{baseline_tag} — K1={s.K1:.2f}, K2={s.K2:.2f}, Q={s.upstream_flow:.1f}m³/s, {src_count}")

            selected = st.selectbox("选择要删除的情景", scenario_names, key="sc_select")
            if st.button("❌ 删除选中情景"):
                sc_manager.remove_scenario(selected)
                st.rerun()

    with col2:
        st.subheader("⚙️ 运行对比模拟")
        if len(scenario_names) > 0:
            compare_names = st.multiselect("选择对比情景", scenario_names, default=scenario_names, key="sc_compare")

            if st.button("🔄 运行对比模拟", type="primary"):
                with st.spinner("正在运行多情景模拟..."):
                    results = []
                    for name in compare_names:
                        scenario = next((s for s in sc_manager.scenarios if s.name == name), None)
                        if scenario:
                            temp_sim = RiverSimulation()
                            temp_sim.setup_default_channel()
                            temp_sim.channel = sim.channel
                            temp_sim.hydro = sim.hydro
                            temp_sim.set_water_quality_params(
                                K1=scenario.K1, K2=scenario.K2,
                                Dx=sim.wq_model.params.Dx,
                                D_O_sat=sim.wq_model.params.D_O_sat
                            )

                            for ps in scenario.point_sources:
                                temp_sim.add_point_source(
                                    ps['name'], ps['x'], ps['flow_rate'],
                                    ps['bod'], ps['do'], ps['nh3n'], ps['cod']
                                )

                            for nps in scenario.nonpoint_sources:
                                temp_sim.add_nonpoint_source(
                                    nps['name'], nps['start_x'], nps['end_x'],
                                    area=nps['area'],
                                    bod_load=nps['bod_load'],
                                    nh3n_load=nps['nh3n_load'],
                                    cod_load=nps['cod_load']
                                )

                            for acc in scenario.accidental_sources:
                                temp_sim.add_accidental_source(
                                    acc['name'], acc['x'], acc['release_type'],
                                    total_mass_bod=acc['total_mass_bod'],
                                    total_mass_nh3n=acc['total_mass_nh3n'],
                                    total_mass_cod=acc['total_mass_cod'],
                                    release_duration=acc['release_duration'],
                                    flow_rate=acc.get('flow_rate', 0.5)
                                )

                            result = temp_sim.run_steady_simulation(
                                Q_upstream=scenario.upstream_flow,
                                initial_bod=scenario.upstream_bod,
                                initial_do=scenario.upstream_do,
                                initial_nh3n=scenario.upstream_nh3n,
                                initial_cod=scenario.upstream_cod,
                            )
                            result['scenario_name'] = name
                            results.append(result)

                    st.session_state.comparison_results = results

    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
        st.subheader("📊 对比结果")

        comp_select = st.selectbox(
            "选择对比组分",
            ['bod', 'do', 'nh3n', 'cod'],
            format_func=lambda x: {'bod': 'BOD', 'do': 'DO', 'nh3n': 'NH3-N', 'cod': 'COD'}[x],
            key='scenario_comp'
        )

        fig = plot_scenario_comparison(st.session_state.comparison_results, comp_select)
        st.pyplot(fig, use_container_width=True)

        st.subheader("📈 达标率对比曲线")
        st.caption("基准情景用实线，对比情景用虚线/点线/点划线区分")
        fig_compare = plot_scenario_compliance_comparison(
            st.session_state.comparison_results, sc_manager
        )
        st.pyplot(fig_compare, use_container_width=True)

        comp_rate_data = []
        for result in st.session_state.comparison_results:
            _, rates = compute_sliding_compliance_rate(result)
            comp_rate_data.append({
                '情景名称': result['scenario_name'],
                '全河段平均达标率 (%)': f"{np.mean(rates):.1f}",
                '最低达标率 (%)': f"{np.min(rates):.1f}",
                '达标率>80%河段占比 (%)': f"{np.sum(rates >= 80) / len(rates) * 100:.1f}",
                '达标率<60%河段占比 (%)': f"{np.sum(rates < 60) / len(rates) * 100:.1f}",
            })
        st.dataframe(pd.DataFrame(comp_rate_data), use_container_width=True, hide_index=True)

        st.subheader("📋 对比统计表")

        table_data = []
        for result in st.session_state.comparison_results:
            table_data.append({
                '情景名称': result['scenario_name'],
                '最大BOD (mg/L)': f"{np.max(result['bod']):.3f}",
                '最低DO (mg/L)': f"{np.min(result['do']):.3f}",
                '最大NH3-N (mg/L)': f"{np.max(result['nh3n']):.3f}",
                '最大COD (mg/L)': f"{np.max(result['cod']):.3f}",
                '临界点位置 (m)': f"{result.get('critical_x', '-')}",
                '临界DO (mg/L)': f"{result.get('critical_do', '-'):.3f}" if result.get('critical_do') else '-',
            })

        st.table(pd.DataFrame(table_data))


def results_table_tab():
    """结果表格标签页"""
    st.header("计算结果表格")

    if st.session_state.result is not None:
        result = st.session_state.result

        df = pd.DataFrame({
            '距离(m)': result['x'],
            '水深(m)': result['h'],
            '流速(m/s)': result['V'],
            '流量(m³/s)': result['Q'],
            'BOD(mg/L)': result['bod'],
            'DO(mg/L)': result['do'],
            'NH3-N(mg/L)': result['nh3n'],
            'COD(mg/L)': result['cod'],
        })

        st.dataframe(df, use_container_width=True, height=400)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 导出CSV",
            csv,
            "water_quality_results.csv",
            "text/csv"
        )
    else:
        st.info("请先运行模拟")


def report_export_tab():
    """报告导出标签页"""
    st.header("报告导出")

    if st.session_state.result is not None:
        result = st.session_state.result
        sim = st.session_state.simulation

        st.subheader("📄 生成PDF报告")

        report_title = st.text_input("报告标题", "河流水质模拟分析报告", key="report_title")

        current_params = st.session_state.get('current_wq_params', {
            'K1': sim.wq_model.params.K1,
            'K2': sim.wq_model.params.K2,
            'K_nh3n': sim.wq_model.params.K_nh3n,
            'K_cod': sim.wq_model.params.K_cod,
            'Dx': sim.wq_model.params.Dx,
            'D_O_sat': sim.wq_model.params.D_O_sat,
        })

        st.info(f"当前参数：K1={current_params.get('K1', sim.wq_model.params.K1):.4f} 1/d, "
                f"K2={current_params.get('K2', sim.wq_model.params.K2):.4f} 1/d, "
                f"K_nh3n={current_params.get('K_nh3n', sim.wq_model.params.K_nh3n):.4f} 1/d, "
                f"K_cod={current_params.get('K_cod', sim.wq_model.params.K_cod):.4f} 1/d, "
                f"Dx={current_params.get('Dx', sim.wq_model.params.Dx):.1f} m²/s")

        if st.button("📄 生成PDF报告", type="primary"):
            with st.spinner("正在生成报告..."):
                generator = ReportGenerator()

                figures = []

                fig_profile = plot_water_quality_profile(result)
                figures.append(fig_profile)

                fig_water = plot_water_level_profile(result)
                figures.append(fig_water)

                fig_oxygen = plot_oxygen_sag_curve(result)
                figures.append(fig_oxygen)

                params = {
                    'channel': {
                        '河道总长度': f"{result['x'][-1] - result['x'][0]:.0f} m",
                        '平均水深': f"{np.mean(result['h']):.3f} m",
                        '平均流速': f"{np.mean(result['V']):.3f} m/s",
                        '平均流量': f"{np.mean(result['Q']):.3f} m³/s",
                        '数值方案': result.get('wq_scheme', 'upwind'),
                    },
                    'K1': current_params.get('K1', sim.wq_model.params.K1),
                    'K2': current_params.get('K2', sim.wq_model.params.K2),
                    'K_nh3n': current_params.get('K_nh3n', sim.wq_model.params.K_nh3n),
                    'K_cod': current_params.get('K_cod', sim.wq_model.params.K_cod),
                    'Dx': current_params.get('Dx', sim.wq_model.params.Dx),
                    'D_O_sat': current_params.get('D_O_sat', sim.wq_model.params.D_O_sat),
                }

                pdf_bytes = generator.generate_report(params, result, figures)

                st.success("✅ 报告生成完成！")

                st.download_button(
                    "📥 下载PDF报告",
                    pdf_bytes,
                    "river_water_quality_report.pdf",
                    "application/pdf"
                )
    else:
        st.info("请先运行模拟")


def warning_emergency_tab():
    """水质预警与应急响应标签页"""
    st.header("⚠️ 水质预警与应急响应")

    sim = st.session_state.simulation
    result = st.session_state.result
    rules = st.session_state.warning_rules

    if result is None:
        st.warning("⚠️ 请先在【稳态模拟】标签页运行一次稳态模拟，然后再使用本模块")
        st.info("💡 步骤：1) 设置河道、水质参数和污染源 → 2) 运行稳态模拟 → 3) 返回本页设置预警")
        return

    x = result['x']
    x_min, x_max = x[0], x[-1]

    st.markdown("### 1️⃣ 预警规则配置面板")
    col_rules1, col_rules2, col_rules3 = st.columns(3)

    with col_rules1:
        st.markdown("#### 🧪 BOD 阈值 (越低越好)")
        st.caption("单位: mg/L | 蓝:轻度 | 橙:超标 | 红:严重超标")
        bod_blue_low = st.number_input("BOD-蓝色下限", 0.0, 20.0, rules.bod.blue_low, 0.1, key="w_bod_blue_low")
        bod_blue_high = st.number_input("BOD-蓝色上限(=橙色下限)", 0.0, 20.0, rules.bod.blue_high, 0.1, key="w_bod_blue_high")
        bod_orange_high = st.number_input("BOD-橙色上限(=红色阈值)", 0.0, 50.0, rules.bod.red_threshold, 0.1, key="w_bod_red")
        rules.bod = WarningThreshold(
            blue_low=bod_blue_low, blue_high=bod_blue_high,
            orange_low=bod_blue_high, orange_high=bod_orange_high,
            red_threshold=bod_orange_high, is_lower_better=True
        )

    with col_rules2:
        st.markdown("#### 💧 DO 阈值 (越高越好)")
        st.caption("单位: mg/L | 蓝:接近标准 | 橙:低于标准 | 红:严重缺氧")
        do_red = st.number_input("DO-红色阈值(≤此值为红)", 0.0, 15.0, rules.do.red_threshold, 0.1, key="w_do_red")
        do_orange_high = st.number_input("DO-橙色上限", 0.0, 15.0, rules.do.orange_high, 0.1, key="w_do_orange_high")
        do_blue_high = st.number_input("DO-蓝色上限(>此值为正常)", 0.0, 15.0, rules.do.blue_high, 0.1, key="w_do_blue_high")
        rules.do = WarningThreshold(
            blue_low=do_orange_high, blue_high=do_blue_high,
            orange_low=do_red, orange_high=do_orange_high,
            red_threshold=do_red, is_lower_better=False
        )

    with col_rules3:
        st.markdown("#### 🔬 NH3-N 阈值 (越低越好)")
        st.caption("单位: mg/L | 蓝:轻度 | 橙:超标 | 红:严重超标")
        nh3n_blue_low = st.number_input("NH3-N-蓝色下限", 0.0, 10.0, rules.nh3n.blue_low, 0.05, key="w_nh3n_blue_low")
        nh3n_blue_high = st.number_input("NH3-N-蓝色上限(=橙色下限)", 0.0, 10.0, rules.nh3n.blue_high, 0.05, key="w_nh3n_blue_high")
        nh3n_orange_high = st.number_input("NH3-N-橙色上限(=红色阈值)", 0.0, 20.0, rules.nh3n.red_threshold, 0.05, key="w_nh3n_red")
        rules.nh3n = WarningThreshold(
            blue_low=nh3n_blue_low, blue_high=nh3n_blue_high,
            orange_low=nh3n_blue_high, orange_high=nh3n_orange_high,
            red_threshold=nh3n_orange_high, is_lower_better=True
        )

    col_btn_apply, col_btn_reset = st.columns([1, 1])
    with col_btn_apply:
        apply_rules = st.button("✅ 应用规则", type="primary", use_container_width=True)
    with col_btn_reset:
        reset_measures = st.button("🔄 重置应急状态", use_container_width=True)

    if reset_measures:
        st.session_state.emergency_measures = []
        st.session_state.emergency_result = None
        st.session_state.emergency_warning_data = None
        st.session_state.emergency_sim = None
        st.success("✅ 应急状态已重置")

    if apply_rules:
        with st.spinner("正在逐点判定预警级别..."):
            warning_data = evaluate_warnings(result, rules)
            st.session_state.warning_data = warning_data
            st.session_state.warning_history.add_record(
                record_type='rule',
                stats=warning_data['stats'],
                measures_desc='应用预警规则判定'
            )
            st.success(f"✅ 规则已应用！共 {warning_data['stats']['total']} 个断面，"
                       f"红色{warning_data['stats']['red_count']}，"
                       f"橙色{warning_data['stats']['orange_count']}，"
                       f"蓝色{warning_data['stats']['blue_count']}，"
                       f"正常{warning_data['stats']['normal_count']}")

    if st.session_state.warning_data is None:
        st.info("👆 请先点击上方【应用规则】按钮进行预警判定")
        return

    warning_data = st.session_state.warning_data

    st.markdown("---")
    st.markdown("### 2️⃣ 河段预警示意图 & 3️⃣ 应急响应模拟")

    col_river_left, col_river_right = st.columns([3, 1.1])

    with col_river_right:
        st.markdown("#### 🎯 选择查看位置")
        hover_pos = st.slider(
            "滑动选择断面位置 (m)",
            float(x_min), float(x_max), float(x_min + (x_max - x_min) * 0.3),
            key="warning_hover_slider"
        )
        hover_idx = int(np.argmin(np.abs(x - hover_pos)))
        st.caption(f"当前网格索引: {hover_idx}/{len(x)-1} | x = {x[hover_idx]:.1f}m")

        st.markdown("---")
        st.markdown("#### 🛠️ 添加应急措施")

        measure_type = st.selectbox(
            "措施类型",
            ['aerator', 'close_source', 'reduce_source'],
            format_func=lambda x: {
                'aerator': '💨 投放增氧剂 (增加DO点源)',
                'close_source': '🚫 临时关闭排污口',
                'reduce_source': '📉 削减排污口排放',
            }[x],
            key="measure_type_select"
        )

        if measure_type == 'aerator':
            aerator_x = st.slider("投放位置 (m)", x_min, x_max,
                                  x_min + (x_max - x_min) * 0.4, key="aerator_x")
            aerator_do = st.slider("增氧浓度 (mg/L)", 2.0, 15.0, 9.0, 0.5, key="aerator_do")
            aerator_flow = st.slider("注入流量 (m³/s)", 0.05, 2.0, 0.3, 0.05, key="aerator_flow")
            measure_desc = f"💨 增氧@{aerator_x:.0f}m (DO={aerator_do:.1f}mg/L, Q={aerator_flow:.2f}m³/s)"

        elif measure_type == 'close_source':
            point_sources = sim.source_manager.get_point_sources()
            if not point_sources:
                st.warning("⚠️ 未配置点源排污口，无法关闭")
                source_name = None
            else:
                src_names = [ps.name for ps in point_sources]
                source_name = st.selectbox("选择排污口", src_names, key="close_source_name")
            measure_desc = f"🚫 关闭排污口[{source_name}]" if source_name else "无可用排污口"

        else:
            point_sources = sim.source_manager.get_point_sources()
            if not point_sources:
                st.warning("⚠️ 未配置点源排污口，无法削减")
                source_name = None
                reduce_ratio = 0.5
            else:
                src_names = [ps.name for ps in point_sources]
                source_name = st.selectbox("选择排污口", src_names, key="reduce_source_name")
                reduce_ratio = st.slider("削减比例 (%)", 10, 90, 50, 5, key="reduce_ratio") / 100.0
            measure_desc = f"📉 削减[{source_name}]排放{reduce_ratio*100:.0f}%" if source_name else "无可用排污口"

        col_add, col_clear = st.columns(2)
        with col_add:
            add_measure = st.button("➕ 添加", use_container_width=True)
        with col_clear:
            clear_measures = st.button("🗑️ 清空", use_container_width=True)

        if add_measure:
            if measure_type == 'aerator':
                measure = EmergencyMeasure(
                    measure_type='aerator',
                    position=aerator_x,
                    description=measure_desc,
                    params={'do_supply': aerator_do, 'flow_inject': aerator_flow}
                )
                st.session_state.emergency_measures.append(measure)
                st.success(f"✅ 已添加: {measure_desc}")
            elif measure_type in ['close_source', 'reduce_source']:
                if source_name:
                    measure = EmergencyMeasure(
                        measure_type=measure_type,
                        position=0.0,
                        description=measure_desc,
                        params={'source_name': source_name,
                                'reduce_ratio': reduce_ratio if measure_type == 'reduce_source' else 1.0}
                    )
                    st.session_state.emergency_measures.append(measure)
                    st.success(f"✅ 已添加: {measure_desc}")

        if clear_measures:
            st.session_state.emergency_measures = []
            st.session_state.emergency_result = None
            st.session_state.emergency_warning_data = None
            st.info("✅ 措施列表已清空")

        if st.session_state.emergency_measures:
            st.markdown("**📋 已配置措施:**")
            for i, m in enumerate(st.session_state.emergency_measures):
                st.caption(f"{i+1}. {m.description}")

        run_emergency = st.button(
            "🚀 运行应急模拟",
            type="primary",
            disabled=len(st.session_state.emergency_measures) == 0,
            use_container_width=True
        )

        if run_emergency and st.session_state.steady_sim_params:
            with st.spinner("正在运行应急模拟..."):
                base_params = st.session_state.steady_sim_params
                em_sim, em_result = run_emergency_simulation(
                    sim=sim,
                    measures=st.session_state.emergency_measures,
                    base_kwargs=base_params
                )
                em_warning = evaluate_warnings(em_result, rules)
                st.session_state.emergency_result = em_result
                st.session_state.emergency_warning_data = em_warning
                st.session_state.emergency_sim = em_sim

                measures_text = "; ".join([m.description for m in st.session_state.emergency_measures])
                st.session_state.warning_history.add_record(
                    record_type='emergency',
                    stats=em_warning['stats'],
                    measures_desc=measures_text
                )

                before_red = warning_data['stats']['red_count']
                after_red = em_warning['stats']['red_count']
                delta = before_red - after_red
                if delta > 0:
                    st.success(f"✅ 应急完成！红色预警减少 {delta} 个 ({before_red} → {after_red})")
                elif delta < 0:
                    st.warning(f"⚠️ 应急后红色预警增加 {-delta} 个，请检查措施")
                else:
                    st.info(f"ℹ️ 应急后红色预警数量不变 ({after_red})")

    with col_river_left:
        em_result = st.session_state.emergency_result
        em_warning = st.session_state.emergency_warning_data
        sim_for_plot = st.session_state.emergency_sim if st.session_state.emergency_sim is not None else sim

        fig_schematic = plot_river_warning_schematic(
            result=result,
            warning_data=warning_data,
            sim=sim_for_plot,
            result_emergency=em_result,
            warning_emergency=em_warning,
            hover_idx=hover_idx,
            figsize=(14, 5)
        )
        st.pyplot(fig_schematic, use_container_width=True)

    st.markdown("---")
    st.markdown("### 4️⃣ 预警统计仪表盘")

    em_stats = em_warning['stats'] if em_warning is not None else None
    cards_data = build_stats_cards_data(warning_data['stats'], em_stats)

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    def render_stat_card(col, key, name, emoji, color):
        b = cards_data['before']
        with col:
            with st.container():
                st.markdown(f"<div style='padding:15px; border-radius:12px; "
                            f"border:2px solid {color}; background-color:{color}15;'>",
                            unsafe_allow_html=True)
                st.markdown(f"<h4 style='margin:0; color:{color};'>{emoji} {name}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='margin:5px 0; color:{color}; font-weight:bold;'>"
                            f"{b[key]} <span style='font-size:14px;'>个</span></h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='margin:0; color:#666; font-size:13px;'>"
                            f"占比: <b>{b[f'{key}_pct']:.1f}%</b></p>", unsafe_allow_html=True)
                if 'after' in cards_data:
                    a = cards_data['after']
                    trend_icon = a[f'{key}_trend']
                    trend_color = a[f'{key}_trend_color']
                    diff_val = a[f'{key}_diff']
                    sign = "+" if diff_val > 0 else ""
                    st.markdown(
                        f"<p style='margin:8px 0 0 0; padding:6px 10px; border-radius:8px; "
                        f"background-color:{trend_color}18; border:1px solid {trend_color}50;'>"
                        f"<span style='font-size:18px; color:{trend_color}; font-weight:bold;'>"
                        f"应急后: {a[key]}个 {trend_icon}{sign}{diff_val}</span></p>",
                        unsafe_allow_html=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)

    render_stat_card(col_c1, 'red', '红色预警', '🔴', WARNING_COLORS[WARNING_RED])
    render_stat_card(col_c2, 'orange', '橙色预警', '🟠', WARNING_COLORS[WARNING_ORANGE])
    render_stat_card(col_c3, 'blue', '蓝色预警', '🔵', WARNING_COLORS[WARNING_BLUE])
    render_stat_card(col_c4, 'normal', '正常断面', '🟢', WARNING_COLORS[WARNING_NORMAL])

    st.markdown("")
    fig_ring = plot_dashboard_ring_chart(warning_data['stats'], em_stats)
    st.pyplot(fig_ring, use_container_width=True)

    st.markdown("---")
    st.markdown("### 5️⃣ 多方案对比 & 智能推荐")

    st.markdown("#### 📋 方案配置（最多5套）")
    col_n_plans, col_plan_hint = st.columns([1, 4])
    with col_n_plans:
        n_plans = st.number_input("配置方案数量", min_value=1, max_value=5, value=2, step=1, key="n_batch_plans")
    with col_plan_hint:
        st.info("💡 每套方案可独立配置一套应急措施（增氧/关排污口/削减排放），支持命名区分。对比将按红色预警数从少到多排序。")

    default_plan_names = ["方案A", "方案B", "方案C", "方案D", "方案E"]
    if len(st.session_state.batch_plan_configs) != n_plans:
        new_configs = []
        for i in range(n_plans):
            if i < len(st.session_state.batch_plan_configs):
                new_configs.append(st.session_state.batch_plan_configs[i])
            else:
                new_configs.append({'plan_name': default_plan_names[i], 'measures': []})
        st.session_state.batch_plan_configs = new_configs

    batch_plan_tabs = st.tabs([f"🗂️ {cfg['plan_name']}" for cfg in st.session_state.batch_plan_configs])
    point_sources = sim.source_manager.get_point_sources()
    src_names = [ps.name for ps in point_sources] if point_sources else []

    for plan_idx in range(n_plans):
        with batch_plan_tabs[plan_idx]:
            cfg = st.session_state.batch_plan_configs[plan_idx]

            col_pname, col_pdesc = st.columns([1, 3])
            with col_pname:
                plan_name = st.text_input(
                    "方案名称",
                    value=cfg.get('plan_name', default_plan_names[plan_idx]),
                    key=f"batch_plan_name_{plan_idx}"
                )
            with col_pdesc:
                current_measures = cfg.get('measures', [])
                st.caption(f"已配置 {len(current_measures)} 项措施")
                if current_measures:
                    for mi, mm in enumerate(current_measures):
                        st.caption(f"  {mi+1}. {mm.description}")

            st.markdown("**添加应急措施:**")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                b_measure_type = st.selectbox(
                    "措施类型",
                    ['aerator', 'close_source', 'reduce_source'],
                    format_func=lambda x: {
                        'aerator': '💨 投放增氧剂',
                        'close_source': '🚫 关闭排污口',
                        'reduce_source': '📉 削减排污口',
                    }[x],
                    key=f"batch_measure_type_{plan_idx}"
                )
            with bcol2:
                if b_measure_type == 'aerator':
                    b_aerator_x = st.slider(
                        "投放位置 (m)", x_min, x_max,
                        x_min + (x_max - x_min) * (0.3 + plan_idx * 0.1),
                        key=f"batch_aerator_x_{plan_idx}"
                    )
                    b_aerator_do = st.slider(
                        "增氧浓度 (mg/L)", 2.0, 15.0, 9.0, 0.5,
                        key=f"batch_aerator_do_{plan_idx}"
                    )
                    b_aerator_flow = st.slider(
                        "注入流量 (m³/s)", 0.05, 2.0, 0.3, 0.05,
                        key=f"batch_aerator_flow_{plan_idx}"
                    )
                elif not src_names:
                    st.warning("无可用排污口")
                    b_source_name = None
                else:
                    b_source_name = st.selectbox(
                        "选择排污口",
                        src_names,
                        key=f"batch_src_name_{plan_idx}"
                    )
            with bcol3:
                if b_measure_type == 'aerator':
                    b_desc = f"💨 增氧@{b_aerator_x:.0f}m (DO={b_aerator_do:.1f}, Q={b_aerator_flow:.2f}m³/s)"
                    st.caption(f"预估: {b_desc}")
                elif b_measure_type == 'close_source' and b_source_name:
                    b_desc = f"🚫 关闭排污口[{b_source_name}]"
                    st.caption(f"预估: {b_desc}")
                elif b_measure_type == 'reduce_source' and b_source_name:
                    b_reduce_ratio = st.slider(
                        "削减比例 (%)", 10, 90, 50, 5,
                        key=f"batch_reduce_ratio_{plan_idx}"
                    ) / 100.0
                    b_desc = f"📉 削减[{b_source_name}]排放{b_reduce_ratio*100:.0f}%"
                    st.caption(f"预估: {b_desc}")
                else:
                    b_desc = ""

            bc_add, bc_clear = st.columns([1, 1])
            with bc_add:
                b_add = st.button("➕ 添加措施", key=f"batch_add_measure_{plan_idx}", use_container_width=True)
            with bc_clear:
                b_clear = st.button("🗑️ 清空措施", key=f"batch_clear_measures_{plan_idx}", use_container_width=True)

            if b_add:
                new_measure = None
                if b_measure_type == 'aerator':
                    new_measure = EmergencyMeasure(
                        measure_type='aerator',
                        position=b_aerator_x,
                        description=b_desc,
                        params={'do_supply': b_aerator_do, 'flow_inject': b_aerator_flow}
                    )
                elif b_measure_type == 'close_source' and b_source_name:
                    new_measure = EmergencyMeasure(
                        measure_type='close_source',
                        position=0.0,
                        description=b_desc,
                        params={'source_name': b_source_name, 'reduce_ratio': 1.0}
                    )
                elif b_measure_type == 'reduce_source' and b_source_name:
                    new_measure = EmergencyMeasure(
                        measure_type='reduce_source',
                        position=0.0,
                        description=b_desc,
                        params={'source_name': b_source_name, 'reduce_ratio': b_reduce_ratio}
                    )
                if new_measure:
                    cfg['measures'].append(new_measure)
                    cfg['plan_name'] = plan_name
                    st.success(f"✅ 已添加: {b_desc}")
                    st.rerun()

            if b_clear:
                cfg['measures'] = []
                cfg['plan_name'] = plan_name
                st.info("✅ 该方案措施已清空")
                st.rerun()

            cfg['plan_name'] = plan_name

    all_plans_valid = True
    for cfg in st.session_state.batch_plan_configs:
        if not cfg.get('measures'):
            all_plans_valid = False
            break

    st.markdown("")
    run_batch_col, reset_batch_col = st.columns([1, 1])
    with run_batch_col:
        run_batch = st.button(
            "🔬 批量对比模拟",
            type="primary",
            disabled=not all_plans_valid or not st.session_state.steady_sim_params,
            use_container_width=True,
        )
    with reset_batch_col:
        reset_batch = st.button("🔄 清空所有方案配置", use_container_width=True)

    if not all_plans_valid:
        st.warning("⚠️ 请确保每套方案至少配置1项措施")
    if not st.session_state.steady_sim_params:
        st.warning("⚠️ 请先在【稳态模拟】标签页运行一次稳态模拟")

    if reset_batch:
        st.session_state.batch_plan_configs = []
        st.session_state.batch_plan_results = None
        st.session_state.batch_plan_batch_id = None
        st.success("✅ 方案配置已全部清空")
        st.rerun()

    if run_batch and all_plans_valid and st.session_state.steady_sim_params:
        emergency_plans = []
        for cfg in st.session_state.batch_plan_configs:
            pname = cfg.get('plan_name', '未命名')
            measures = cfg.get('measures', [])
            emergency_plans.append(EmergencyPlan(plan_name=pname, measures=measures))

        with st.spinner(f"正在并行运行 {len(emergency_plans)} 套方案的应急模拟..."):
            batch_id, plan_results = run_batch_emergency_simulations(
                plans=emergency_plans,
                sim=sim,
                base_kwargs=st.session_state.steady_sim_params,
                rules=rules,
                base_result=result,
                base_warning_data=warning_data,
            )
            st.session_state.batch_plan_results = plan_results
            st.session_state.batch_plan_batch_id = batch_id

            for plan_res in plan_results:
                measures_text = "; ".join([m.description for m in plan_res.measures])
                st.session_state.warning_history.add_batch_record(
                    record_type='batch_emergency',
                    stats=plan_res.warning_data['stats'],
                    measures_desc=measures_text,
                    batch_id=batch_id,
                    plan_name=plan_res.plan_name,
                    total_cost=plan_res.total_cost,
                    normal_pct=plan_res.normal_pct,
                    key_point_do_delta=plan_res.key_point_do_delta,
                )

        st.success(f"✅ 批量模拟完成！批次ID: {batch_id[:19]}，共 {len(plan_results)} 套方案")

    if st.session_state.batch_plan_results is not None:
        plan_results = st.session_state.batch_plan_results
        base_normal_count = warning_data['stats']['normal_count']

        st.markdown("")
        st.markdown("#### 📊 方案汇总对比表")
        st.caption("按【红色预警数从少到多】排序")

        compare_table_data = []
        for idx, pr in enumerate(plan_results):
            key_point_x = pr.key_point_x
            do_delta_sign = "+" if pr.key_point_do_delta >= 0 else ""
            compare_table_data.append({
                '排名': idx + 1,
                '方案名称': pr.plan_name,
                '🔴 红色预警': pr.red_count,
                '🟠 橙色预警': pr.orange_count,
                '🟢 正常断面占比(%)': f"{pr.normal_pct:.1f}",
                f'排污口下游500m处DO变化\n(x={key_point_x:.0f}m, mg/L)': f"{do_delta_sign}{pr.key_point_do_delta:.2f}",
                '💰 总成本估算(万元)': f"{pr.total_cost:.2f}",
            })
        df_compare = pd.DataFrame(compare_table_data)

        def highlight_compare_rows(row):
            styles = pd.Series('', index=row.index)
            red_val = row['🔴 红色预警']
            if red_val == 0:
                styles[:] = 'background-color: #e8f8f5; font-weight: 500'
            elif red_val <= 3:
                styles[:] = 'background-color: #fef9e7'
            else:
                styles[:] = 'background-color: #fdedec'
            return styles

        styled_compare = df_compare.style.apply(highlight_compare_rows, axis=1)
        styled_compare = styled_compare.map(
            lambda v: 'color: #c0392b; font-weight: bold' if isinstance(v, (int, float)) and v > 0 and '红色' in str(v) else '',
            subset=['🔴 红色预警']
        )
        st.dataframe(styled_compare, use_container_width=True, hide_index=True, height=200)

        with st.expander("💸 查看各方案成本明细"):
            for idx, pr in enumerate(plan_results):
                plan_color = PLAN_LINE_COLORS[idx % len(PLAN_LINE_COLORS)]
                st.markdown(f"<h5 style='color:{plan_color}; margin:8px 0;'>"
                            f"方案{idx+1} · {pr.plan_name} · 总成本 {pr.total_cost:.2f} 万元</h5>",
                            unsafe_allow_html=True)
                if pr.cost_details.get('details'):
                    for detail in pr.cost_details['details']:
                        st.caption(f"  • {detail}")
                else:
                    st.caption("  • 无成本计算措施")
                st.markdown("---")

        st.markdown("")
        st.markdown("#### 🗺️ 河段示意图（多方案叠加）")
        st.caption("基准状态用实心色带，各方案用不同样式的虚线框叠加展示预警状态")
        fig_multi = plot_river_warning_schematic_multi_plan(
            result=result,
            warning_data=warning_data,
            sim=sim,
            plan_results=plan_results,
            hover_idx=hover_idx,
            figsize=(14, 7),
        )
        st.pyplot(fig_multi, use_container_width=True)

        st.markdown("")
        st.markdown("#### 🤖 智能推荐最优方案")

        rec_col1, rec_col2 = st.columns([1.2, 3])
        with rec_col1:
            rec_mode = st.radio(
                "选择优化目标模式",
                ['cost_min', 'effect_max', 'cost_effective'],
                format_func=lambda x: {
                    'cost_min': '① 成本最低（红警清零前提下）',
                    'effect_max': '② 效果最优（正常占比最高）',
                    'cost_effective': '③ 性价比最优（正常增加/成本）',
                }[x],
                index=['cost_min', 'effect_max', 'cost_effective'].index(st.session_state.batch_recommend_mode),
                key="batch_recommend_mode_radio",
            )
            st.session_state.batch_recommend_mode = rec_mode

        recommendation = recommend_optimal_plan(plan_results, rec_mode, base_normal_count)

        with rec_col2:
            if not recommendation.can_clear_red and rec_mode == 'cost_min':
                st.markdown(
                    f"<div style='padding:20px; border-radius:14px; "
                    f"border:3px solid #e67e22; background-color:#fef5e7;'>"
                    f"<h3 style='margin:0 0 10px 0; color:#d35400;'>⚠️ 无法推荐</h3>"
                    f"<p style='margin:0; font-size:16px; color:#873600; line-height:1.6;'>"
                    f"<b>{recommendation.reason}</b><br>"
                    f"建议：① 增加增氧站点或提高增氧剂量；② 关闭更多排污口；"
                    f"③ 加大削减排放比例至80%以上。</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif recommendation.recommended_plan_idx is not None:
                best_idx = recommendation.recommended_plan_idx
                best_result = plan_results[best_idx]
                best_color = PLAN_LINE_COLORS[best_idx % len(PLAN_LINE_COLORS)]

                mode_label = {
                    'cost_min': '成本最低模式',
                    'effect_max': '效果最优模式',
                    'cost_effective': '性价比最优模式',
                }[rec_mode]

                cost_detail_str = ""
                if best_result.cost_details.get('details'):
                    cost_detail_str = "<br>".join([f"&nbsp;&nbsp;• {d}" for d in best_result.cost_details['details']])
                else:
                    cost_detail_str = "&nbsp;&nbsp;无直接成本项"

                key_point_x = best_result.key_point_x
                do_sign = "+" if best_result.key_point_do_delta >= 0 else ""
                st.markdown(
                    f"<div style='padding:22px; border-radius:16px; "
                    f"border:3px solid {best_color}; background-color:linear-gradient(135deg, #eafaf1 0%, #d5f5e3 100%); "
                    f"box-shadow: 0 4px 14px rgba(39,174,96,0.15);'>"
                    f"<h3 style='margin:0 0 12px 0; color:#1e8449;'>"
                    f"🌟 推荐方案：{best_result.plan_name}（方案{best_idx+1}）</h3>"
                    f"<p style='margin:0 0 10px 0; font-size:13px; color:#27ae60; font-weight:600;'>"
                    f"优化目标：{mode_label}</p>"
                    f"<p style='margin:0 0 12px 0; font-size:15px; color:#145a32; line-height:1.6;'>"
                    f"✅ <b>{recommendation.reason}</b></p>"
                    f"<div style='display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:10px;'>"
                    f"<div style='padding:10px; background:white; border-radius:8px; text-align:center;'>"
                    f"<div style='font-size:12px; color:#666;'>红色预警</div>"
                    f"<div style='font-size:20px; font-weight:bold; "
                    f"color:{'#27ae60' if best_result.red_count==0 else '#e74c3c'};'>{best_result.red_count}</div>"
                    f"</div>"
                    f"<div style='padding:10px; background:white; border-radius:8px; text-align:center;'>"
                    f"<div style='font-size:12px; color:#666;'>正常断面占比</div>"
                    f"<div style='font-size:20px; font-weight:bold; color:#2980b9;'>{best_result.normal_pct:.1f}%</div>"
                    f"</div>"
                    f"<div style='padding:10px; background:white; border-radius:8px; text-align:center;'>"
                    f"<div style='font-size:12px; color:#666;'>DO变化@{key_point_x:.0f}m</div>"
                    f"<div style='font-size:20px; font-weight:bold; "
                    f"color:{'#27ae60' if best_result.key_point_do_delta>=0 else '#e74c3c'};'>"
                    f"{do_sign}{best_result.key_point_do_delta:.2f}</div>"
                    f"</div>"
                    f"<div style='padding:10px; background:white; border-radius:8px; text-align:center;'>"
                    f"<div style='font-size:12px; color:#666;'>总成本</div>"
                    f"<div style='font-size:20px; font-weight:bold; color:#8e44ad;'>{best_result.total_cost:.2f}万</div>"
                    f"</div>"
                    f"</div>"
                    f"<details style='margin:0;'><summary style='cursor:pointer; color:#145a32; font-size:13px;'>"
                    f"📋 查看该方案措施清单 & 成本明细</summary>"
                    f"<div style='margin-top:8px; padding:12px; background:white; border-radius:8px;'>"
                    f"<b>措施清单({len(best_result.measures)}项)：</b><br>"
                    + "<br>".join([f"&nbsp;&nbsp;• {m.description}" for m in best_result.measures]) +
                    f"<br><br><b>成本明细：</b><br>{cost_detail_str}"
                    f"</div></details>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding:20px; border-radius:14px; "
                    f"border:2px solid #95a5a6; background-color:#f8f9f9;'>"
                    f"<h4 style='margin:0; color:#7f8c8d;'>{recommendation.reason}</h4>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown("### 6️⃣ 预警历史记录")
    st.caption(f"最多保留最近 {st.session_state.warning_history.max_records} 条记录（批量对比的每套方案作为独立记录，同批次共享批次ID）")

    if not st.session_state.warning_history.is_empty():
        df_history = st.session_state.warning_history.to_dataframe()

        def highlight_rows(row):
            styles = pd.Series('', index=row.index)
            rtype = row.get('记录类型', '')
            if rtype == '应急模拟':
                styles = 'background-color: #eaf2f8;'
            elif rtype == '规则应用':
                styles = 'background-color: #fef9e7;'
            elif '批量' in rtype:
                styles = 'background-color: #eafaf1;'
            return styles

        styled_history = df_history.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_history, use_container_width=True, hide_index=True, height=360)

        col_h1, col_h2, _ = st.columns([1, 1, 4])
        with col_h1:
            csv_bytes = st.session_state.warning_history.to_csv()
            st.download_button(
                "📥 导出CSV",
                csv_bytes,
                f"warning_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        with col_h2:
            if st.button("🗑️ 清空历史", use_container_width=True):
                st.session_state.warning_history = WarningHistory(max_records=20)
                st.rerun()
    else:
        st.info("📭 暂无历史记录，点击【应用规则】或【运行应急模拟】/【批量对比模拟】后记录将自动保存")


def main():
    """主函数"""
    init_session_state()

    with st.sidebar:
        sidebar_river_setup()
        sidebar_wq_params()
        sidebar_sources()

    main_page()

    st.markdown("---")
    st.caption("🌊 河流水质动态模拟系统 v1.0 | 基于 Streamlit 开发")


if __name__ == "__main__":
    main()
