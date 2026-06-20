
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

rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 稳态模拟", "⏱️ 非稳态模拟", "📈 参数率定",
        "🔄 情景分析", "📋 结果表格", "📄 报告导出"
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
        results_table_tab()

    with tab6:
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

    if st.session_state.result is not None:
        result = st.session_state.result

        st.success(f"✅ 模拟完成！使用 {result.get('wq_scheme', wq_scheme).upper()} 数值方案")

        st.subheader("📈 沿程水质分布")
        fig_profile = plot_water_quality_profile(result)
        st.pyplot(fig_profile, use_container_width=True)

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

        if st.button("➕ 添加情景"):
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
            scenario.point_sources = sc_sources
            scenario.nonpoint_sources = sc_nonpoint_sources
            scenario.accidental_sources = sc_accidental_sources
            sc_manager.add_scenario(scenario)
            st.success(f"情景 '{scenario_name}' 已添加 (含{len(sc_sources)}个点源, {len(sc_nonpoint_sources)}个面源, {len(sc_accidental_sources)}个突发源)")

        scenario_names = sc_manager.get_scenario_names()
        if scenario_names:
            selected = st.selectbox("选择情景", scenario_names, key="sc_select")
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
