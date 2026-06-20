
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib import rcParams

rcParams['font.sans-serif'] = [
    'PingFang SC', 'Heiti SC', 'Microsoft YaHei', 'SimHei',
    'Arial Unicode MS', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
    'DejaVu Sans'
]
rcParams['axes.unicode_minus'] = False
import io
import copy
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from pollution_sources import SourceManager, PointSource
from simulation_engine import RiverSimulation

WARNING_NORMAL = 0
WARNING_BLUE = 1
WARNING_ORANGE = 2
WARNING_RED = 3

WARNING_NAMES = {
    WARNING_NORMAL: "正常",
    WARNING_BLUE: "蓝色预警",
    WARNING_ORANGE: "橙色预警",
    WARNING_RED: "红色预警",
}

WARNING_COLORS = {
    WARNING_NORMAL: "#27ae60",
    WARNING_BLUE: "#3498db",
    WARNING_ORANGE: "#f39c12",
    WARNING_RED: "#e74c3c",
}

WARNING_COLORS_LIGHT = {
    WARNING_NORMAL: "#a9dfbf",
    WARNING_BLUE: "#aed6f1",
    WARNING_ORANGE: "#fad7a0",
    WARNING_RED: "#f5b7b1",
}


@dataclass
class WarningThreshold:
    blue_low: float = 0.0
    blue_high: float = 0.0
    orange_low: float = 0.0
    orange_high: float = 0.0
    red_threshold: float = 0.0
    is_lower_better: bool = True

    def get_level(self, value: float) -> int:
        if self.is_lower_better:
            if value >= self.red_threshold:
                return WARNING_RED
            elif value >= self.orange_low and value < self.red_threshold:
                return WARNING_ORANGE
            elif value >= self.blue_low and value < self.orange_low:
                return WARNING_BLUE
            else:
                return WARNING_NORMAL
        else:
            if value <= self.red_threshold:
                return WARNING_RED
            elif value > self.red_threshold and value <= self.orange_high:
                return WARNING_ORANGE
            elif value > self.orange_high and value <= self.blue_high:
                return WARNING_BLUE
            else:
                return WARNING_NORMAL


@dataclass
class WarningRules:
    bod: WarningThreshold = field(default_factory=lambda: WarningThreshold(
        blue_low=2.0, blue_high=4.0,
        orange_low=4.0, orange_high=6.0,
        red_threshold=6.0, is_lower_better=True
    ))
    do: WarningThreshold = field(default_factory=lambda: WarningThreshold(
        blue_low=5.0, blue_high=6.0,
        orange_low=3.0, orange_high=5.0,
        red_threshold=3.0, is_lower_better=False
    ))
    nh3n: WarningThreshold = field(default_factory=lambda: WarningThreshold(
        blue_low=0.5, blue_high=1.0,
        orange_low=1.0, orange_high=2.0,
        red_threshold=2.0, is_lower_better=True
    ))

    def get_point_level(self, bod_val: float, do_val: float, nh3n_val: float) -> int:
        l_bod = self.bod.get_level(bod_val)
        l_do = self.do.get_level(do_val)
        l_nh3n = self.nh3n.get_level(nh3n_val)
        return max(l_bod, l_do, l_nh3n)

    def get_point_details(self, bod_val: float, do_val: float, nh3n_val: float) -> Dict:
        return {
            'bod_level': self.bod.get_level(bod_val),
            'do_level': self.do.get_level(do_val),
            'nh3n_level': self.nh3n.get_level(nh3n_val),
            'overall_level': self.get_point_level(bod_val, do_val, nh3n_val),
        }


@dataclass
class EmergencyMeasure:
    measure_type: str
    position: float
    description: str
    params: Dict = field(default_factory=dict)


@dataclass
class WarningHistoryRecord:
    timestamp: str
    record_type: str
    red_count: int
    orange_count: int
    blue_count: int
    normal_count: int
    measures_desc: str
    batch_id: Optional[str] = None
    plan_name: Optional[str] = None
    total_cost: Optional[float] = None
    normal_pct: Optional[float] = None
    key_point_do_delta: Optional[float] = None


def evaluate_warnings(result: Dict, rules: WarningRules) -> Dict:
    x = result['x']
    n = len(x)
    levels = np.zeros(n, dtype=int)
    details = []

    for i in range(n):
        detail = rules.get_point_details(
            result['bod'][i],
            result['do'][i],
            result['nh3n'][i]
        )
        levels[i] = detail['overall_level']
        details.append(detail)

    stats = {
        'red_count': int(np.sum(levels == WARNING_RED)),
        'orange_count': int(np.sum(levels == WARNING_ORANGE)),
        'blue_count': int(np.sum(levels == WARNING_BLUE)),
        'normal_count': int(np.sum(levels == WARNING_NORMAL)),
        'total': n,
    }
    stats['red_pct'] = stats['red_count'] / n * 100 if n > 0 else 0
    stats['orange_pct'] = stats['orange_count'] / n * 100 if n > 0 else 0
    stats['blue_pct'] = stats['blue_count'] / n * 100 if n > 0 else 0
    stats['normal_pct'] = stats['normal_count'] / n * 100 if n > 0 else 0

    return {
        'x': x,
        'levels': levels,
        'details': details,
        'stats': stats,
    }


def get_warning_cmap() -> LinearSegmentedColormap:
    colors = [
        (0.0, WARNING_COLORS[WARNING_NORMAL]),
        (0.33, WARNING_COLORS[WARNING_BLUE]),
        (0.66, WARNING_COLORS[WARNING_ORANGE]),
        (1.0, WARNING_COLORS[WARNING_RED]),
    ]
    return LinearSegmentedColormap.from_list('warning_cmap', colors, N=256)


def level_to_color(level: int) -> str:
    return WARNING_COLORS.get(level, WARNING_COLORS[WARNING_NORMAL])


def plot_river_warning_schematic(
    result: Dict,
    warning_data: Dict,
    sim: RiverSimulation,
    result_emergency: Optional[Dict] = None,
    warning_emergency: Optional[Dict] = None,
    hover_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    x = warning_data['x']
    levels = warning_data['levels']
    n = len(x)

    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=figsize,
                                           gridspec_kw={'width_ratios': [3, 1.2]})

    river_y = 0.0
    river_height = 1.0
    band_height_top = 0.6

    x_min, x_max = x[0], x[-1]
    x_range = x_max - x_min if x_max > x_min else 1.0

    segment_width = (x[1] - x[0]) if n > 1 else x_range / n
    for i in range(n):
        x_start = x[i] - segment_width / 2
        color = level_to_color(int(levels[i]))
        rect = plt.Rectangle(
            (x_start, river_y - band_height_top / 2),
            segment_width, band_height_top,
            facecolor=color, edgecolor='none', alpha=0.85
        )
        ax_main.add_patch(rect)

    if result_emergency is not None and warning_emergency is not None:
        levels_em = warning_emergency['levels']
        band_height_dash = 0.28
        dash_y_bottom = river_y + band_height_top / 2 + 0.08
        for i in range(n):
            x_start = x[i] - segment_width / 2
            color = level_to_color(int(levels_em[i]))
            rect = plt.Rectangle(
                (x_start, dash_y_bottom),
                segment_width, band_height_dash,
                facecolor='none', edgecolor=color, linewidth=2,
                linestyle='--', alpha=0.9
            )
            ax_main.add_patch(rect)

        legend_patches_em = []
        for lv in [WARNING_NORMAL, WARNING_BLUE, WARNING_ORANGE, WARNING_RED]:
            patch = mpatches.Patch(
                facecolor='none', edgecolor=WARNING_COLORS[lv],
                linewidth=2, linestyle='--',
                label=f"应急后-{WARNING_NAMES[lv]}"
            )
            legend_patches_em.append(patch)

    point_sources = sim.source_manager.get_point_sources()
    for ps in point_sources:
        if x_min <= ps.x <= x_max:
            ax_main.scatter(ps.x, river_y - band_height_top / 2 - 0.15,
                            marker='^', s=180, color='#8e44ad', zorder=10,
                            edgecolor='white', linewidth=1.5, label='排污口' if ps == point_sources[0] else "")
            ax_main.annotate(ps.name[:6], xy=(ps.x, river_y - band_height_top / 2 - 0.15),
                             xytext=(ps.x, river_y - band_height_top / 2 - 0.45),
                             ha='center', fontsize=8, color='#8e44ad', fontweight='bold')

    tributaries = sim.channel.tributaries
    for trib in tributaries:
        if x_min <= trib.x <= x_max:
            arrow_x = trib.x
            arrow_y_top = river_y + band_height_top / 2 + 0.55
            arrow_y_bottom = river_y + band_height_top / 2 + 0.15
            ax_main.annotate("",
                             xy=(arrow_x, arrow_y_bottom),
                             xytext=(arrow_x, arrow_y_top),
                             arrowprops=dict(arrowstyle='->', color='#16a085', lw=2.5),
                             zorder=10)
            ax_main.text(arrow_x, arrow_y_top + 0.12,
                         f"支流\n{trib.flow_rate:.1f}m³/s",
                         ha='center', va='bottom', fontsize=8,
                         color='#16a085', fontweight='bold')

    if hover_idx is not None and 0 <= hover_idx < n:
        hover_x = x[hover_idx]
        ax_main.axvline(x=hover_x, color='#2c3e50', linestyle=':', linewidth=1.5, alpha=0.6)
        ax_main.scatter([hover_x], [river_y], marker='o', s=120,
                        color='white', edgecolor='#2c3e50', linewidth=2.5, zorder=15)

    ax_main.set_xlim(x_min - x_range * 0.03, x_max + x_range * 0.03)
    ax_main.set_ylim(-1.3, 1.6)
    ax_main.set_xlabel('河流距离 (m)', fontsize=11, fontweight='bold')
    ax_main.set_title('河段预警状态示意图', fontsize=13, fontweight='bold', pad=12)
    ax_main.set_yticks([])
    ax_main.set_yticklabels([])
    for spine in ['left', 'right', 'top']:
        ax_main.spines[spine].set_visible(False)
    ax_main.spines['bottom'].set_position(('data', river_y - band_height_top / 2 - 0.65))
    ax_main.grid(axis='x', alpha=0.2, linestyle='--')

    legend_patches = []
    for lv in [WARNING_NORMAL, WARNING_BLUE, WARNING_ORANGE, WARNING_RED]:
        patch = mpatches.Patch(
            color=WARNING_COLORS[lv], alpha=0.85,
            label=f"{WARNING_NAMES[lv]}"
        )
        legend_patches.append(patch)

    if result_emergency is not None:
        all_handles = legend_patches + legend_patches_em
    else:
        all_handles = legend_patches

    ax_main.legend(handles=all_handles, loc='upper right',
                   fontsize=8, framealpha=0.95, ncol=2 if result_emergency else 1,
                   bbox_to_anchor=(1.0, 1.0))

    if hover_idx is not None and 0 <= hover_idx < n:
        display_idx = hover_idx
        bod_val = result['bod'][display_idx]
        do_val = result['do'][display_idx]
        nh3n_val = result['nh3n'][display_idx]
        detail = warning_data['details'][display_idx]
        overall_lv = detail['overall_level']

        ax_info.axis('off')
        title_color = WARNING_COLORS.get(overall_lv, '#333333')
        ax_info.text(0.5, 0.97, f"📍 位置: x={x[display_idx]:.0f}m",
                     ha='center', va='top', fontsize=13, fontweight='bold',
                     transform=ax_info.transAxes)
        ax_info.text(0.5, 0.90, f"预警级别: {WARNING_NAMES[overall_lv]}",
                     ha='center', va='top', fontsize=12, fontweight='bold',
                     color=title_color, bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor=WARNING_COLORS_LIGHT.get(overall_lv, '#f0f0f0'),
                                                  edgecolor=title_color),
                     transform=ax_info.transAxes)

        info_items = [
            ('BOD', bod_val, detail['bod_level'], 'mg/L', True),
            ('DO', do_val, detail['do_level'], 'mg/L', False),
            ('NH3-N', nh3n_val, detail['nh3n_level'], 'mg/L', True),
        ]

        y_pos = 0.76
        for name, val, lv, unit, lower_better in info_items:
            lv_color = WARNING_COLORS.get(lv, '#333333')
            rect = mpatches.FancyBboxPatch((0.05, y_pos - 0.01), 0.90, 0.20,
                                           boxstyle='round,pad=0.05',
                                           facecolor=WARNING_COLORS_LIGHT.get(lv, '#f0f0f0'),
                                           edgecolor=lv_color, linewidth=1.5,
                                           transform=ax_info.transAxes)
            ax_info.add_patch(rect)

            ax_info.text(0.10, y_pos + 0.14, f"{name}",
                         ha='left', va='center', fontsize=11, fontweight='bold',
                         transform=ax_info.transAxes)
            ax_info.text(0.55, y_pos + 0.14, f"{val:.3f} {unit}",
                         ha='right', va='center', fontsize=11,
                         color=lv_color, fontweight='bold',
                         transform=ax_info.transAxes)
            ax_info.text(0.10, y_pos + 0.04, f"级别: {WARNING_NAMES[lv]}",
                         ha='left', va='center', fontsize=9, color=lv_color,
                         transform=ax_info.transAxes)
            y_pos -= 0.23

        if result_emergency is not None and warning_emergency is not None:
            em_bod = result_emergency['bod'][display_idx]
            em_do = result_emergency['do'][display_idx]
            em_nh3n = result_emergency['nh3n'][display_idx]
            em_detail = warning_emergency['details'][display_idx]
            em_overall = em_detail['overall_level']

            y_pos = 0.12
            ax_info.text(0.5, y_pos + 0.08, "── 应急后对比 ──",
                         ha='center', va='center', fontsize=10,
                         fontweight='bold', color='#555555',
                         transform=ax_info.transAxes)

            delta_bod = em_bod - bod_val
            delta_do = em_do - do_val
            delta_nh3n = em_nh3n - nh3n_val

            for name, em_val, delta_val in [
                ('BOD', em_bod, delta_bod),
                ('DO', em_do, delta_do),
                ('NH3-N', em_nh3n, delta_nh3n),
            ]:
                arrow = "↓" if (delta_val < 0 and name != 'DO') or (delta_val > 0 and name == 'DO') else \
                        "↑" if (delta_val > 0 and name != 'DO') or (delta_val < 0 and name == 'DO') else "→"
                arrow_color = '#27ae60' if arrow == "↓" and name != 'DO' else \
                              '#27ae60' if arrow == "↑" and name == 'DO' else \
                              '#e74c3c' if arrow != "→" else '#7f8c8d'
                ax_info.text(0.08, y_pos - 0.02,
                             f"{name}: {em_val:.3f} {arrow}{abs(delta_val):.3f}",
                             ha='left', va='center', fontsize=9,
                             color=arrow_color, fontweight='bold',
                             transform=ax_info.transAxes)
                y_pos -= 0.05

            lv_change = em_overall - overall_lv
            if lv_change < 0:
                change_text = f"✅ 级别改善: {WARNING_NAMES[overall_lv]} → {WARNING_NAMES[em_overall]}"
                change_color = '#27ae60'
            elif lv_change > 0:
                change_text = f"⚠️ 级别恶化: {WARNING_NAMES[overall_lv]} → {WARNING_NAMES[em_overall]}"
                change_color = '#e74c3c'
            else:
                change_text = f"➡️ 级别不变: {WARNING_NAMES[overall_lv]}"
                change_color = '#7f8c8d'
            ax_info.text(0.5, 0.02, change_text,
                         ha='center', va='bottom', fontsize=9.5,
                         fontweight='bold', color=change_color,
                         transform=ax_info.transAxes)

    else:
        ax_info.axis('off')
        ax_info.text(0.5, 0.5,
                     "💡 在左侧滑动滑块\n选择位置查看详情",
                     ha='center', va='center', fontsize=11,
                     color='#7f8c8d', style='italic',
                     transform=ax_info.transAxes)
        ax_info.set_title('断面信息面板', fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    return fig


def plot_dashboard_ring_chart(stats: Dict, stats_emergency: Optional[Dict] = None) -> plt.Figure:
    if stats_emergency is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        axes = [ax1, ax2]
        titles = ['应急前预警级别分布', '应急后预警级别分布']
        all_stats = [stats, stats_emergency]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))
        axes = [ax1]
        titles = ['预警级别分布']
        all_stats = [stats]

    for idx, ax in enumerate(axes):
        s = all_stats[idx]
        labels = []
        sizes = []
        colors = []
        explode = []

        for lv, name in [(WARNING_RED, WARNING_NAMES[WARNING_RED]),
                         (WARNING_ORANGE, WARNING_NAMES[WARNING_ORANGE]),
                         (WARNING_BLUE, WARNING_NAMES[WARNING_BLUE]),
                         (WARNING_NORMAL, WARNING_NAMES[WARNING_NORMAL])]:
            count_key = {WARNING_RED: 'red_count', WARNING_ORANGE: 'orange_count',
                         WARNING_BLUE: 'blue_count', WARNING_NORMAL: 'normal_count'}[lv]
            count = s[count_key]
            pct = s[f"{count_key.split('_')[0]}_pct"]
            if count > 0 or True:
                labels.append(f"{name}\n{count}个 ({pct:.1f}%)")
                sizes.append(count if count > 0 else 0.1)
                colors.append(WARNING_COLORS[lv])
                explode.append(0.05 if lv == WARNING_RED else 0.0)

        if all(sz <= 0.1 for sz in sizes):
            sizes = [0.1] * 4

        wedges, texts = ax.pie(
            sizes, labels=None, colors=colors, startangle=90,
            explode=explode, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.5)
        )

        centre_circle = plt.Circle((0, 0), 0.60, fc='white', edgecolor='#cccccc', linewidth=1)
        ax.add_artist(centre_circle)

        total = s['total']
        total_warn = s['red_count'] + s['orange_count'] + s['blue_count']
        ax.text(0, 0.08, f"总计", ha='center', va='center', fontsize=9, color='#666666')
        ax.text(0, -0.05, f"{total}", ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
        ax.text(0, -0.22, f"预警{total_warn}", ha='center', va='center', fontsize=9, color='#e74c3c', fontweight='bold')

        ax.legend(wedges, labels, title="级别统计",
                  loc="center left", bbox_to_anchor=(1.0, 0.5),
                  fontsize=8, title_fontsize=9, framealpha=0.9)

        ax.set_title(titles[idx], fontsize=11, fontweight='bold', pad=10)
        ax.axis('equal')

    plt.tight_layout()
    return fig


def build_stats_cards_data(stats: Dict, stats_emergency: Optional[Dict] = None) -> Dict:
    result = {
        'before': {
            'red': stats['red_count'],
            'orange': stats['orange_count'],
            'blue': stats['blue_count'],
            'normal': stats['normal_count'],
            'red_pct': stats['red_pct'],
            'orange_pct': stats['orange_pct'],
            'blue_pct': stats['blue_pct'],
            'normal_pct': stats['normal_pct'],
        }
    }
    if stats_emergency is not None:
        result['after'] = {
            'red': stats_emergency['red_count'],
            'orange': stats_emergency['orange_count'],
            'blue': stats_emergency['blue_count'],
            'normal': stats_emergency['normal_count'],
            'red_pct': stats_emergency['red_pct'],
            'orange_pct': stats_emergency['orange_pct'],
            'blue_pct': stats_emergency['blue_pct'],
            'normal_pct': stats_emergency['normal_pct'],
        }
        for key in ['red', 'orange', 'blue', 'normal']:
            before = result['before'][key]
            after = result['after'][key]
            diff = after - before
            if diff < 0:
                trend = "↓"
                trend_color = '#27ae60' if key != 'normal' else '#e74c3c'
            elif diff > 0:
                trend = "↑"
                trend_color = '#e74c3c' if key != 'normal' else '#27ae60'
            else:
                trend = "→"
                trend_color = '#7f8c8d'
            result['after'][f'{key}_diff'] = diff
            result['after'][f'{key}_trend'] = trend
            result['after'][f'{key}_trend_color'] = trend_color
    return result


def run_emergency_simulation(
    sim: RiverSimulation,
    measures: List[EmergencyMeasure],
    base_kwargs: Dict,
) -> Tuple[RiverSimulation, Dict]:
    sim_copy = copy.deepcopy(sim)

    for measure in measures:
        if measure.measure_type == 'aerator':
            x_pos = measure.position
            do_supply = measure.params.get('do_supply', 2.0)
            flow_inject = measure.params.get('flow_inject', 0.3)
            sat_do = sim_copy.wq_model.params.D_O_sat
            sim_copy.add_point_source(
                name=f"增氧站@{x_pos:.0f}m",
                x=x_pos,
                flow_rate=flow_inject,
                bod_conc=0.0,
                do_conc=min(do_supply, sat_do),
                nh3n_conc=0.0,
                cod_conc=0.0,
            )
        elif measure.measure_type == 'close_source':
            source_name = measure.params.get('source_name', '')
            new_sources = []
            for s in sim_copy.source_manager.sources:
                if isinstance(s, PointSource) and s.name == source_name:
                    closed_ps = PointSource(
                        name=f"{s.name}(已关闭)",
                        x=s.x,
                        flow_rate=0.0,
                        bod_conc=0.0,
                        do_conc=sim_copy.wq_model.params.D_O_sat,
                        nh3n_conc=0.0,
                        cod_conc=0.0,
                    )
                    new_sources.append(closed_ps)
                else:
                    new_sources.append(s)
            sim_copy.source_manager.sources = new_sources
            sim_copy.source_manager.sources.sort(key=lambda s: s.x)
        elif measure.measure_type == 'reduce_source':
            source_name = measure.params.get('source_name', '')
            reduce_ratio = measure.params.get('reduce_ratio', 0.5)
            for s in sim_copy.source_manager.sources:
                if isinstance(s, PointSource) and s.name == source_name:
                    s.bod_conc *= (1 - reduce_ratio)
                    s.nh3n_conc *= (1 - reduce_ratio)
                    s.cod_conc *= (1 - reduce_ratio)

    with StSpinnerPatch():
        result = sim_copy.run_steady_simulation(**base_kwargs)

    return sim_copy, result


class StSpinnerPatch:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class WarningHistory:
    def __init__(self, max_records: int = 20):
        self.records: List[WarningHistoryRecord] = []
        self.max_records = max_records

    def add_record(self, record_type: str, stats: Dict, measures_desc: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = WarningHistoryRecord(
            timestamp=timestamp,
            record_type=record_type,
            red_count=stats['red_count'],
            orange_count=stats['orange_count'],
            blue_count=stats['blue_count'],
            normal_count=stats['normal_count'],
            measures_desc=measures_desc if measures_desc else "无",
        )
        self.records.insert(0, record)
        if len(self.records) > self.max_records:
            self.records = self.records[:self.max_records]

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for r in self.records:
            total = r.red_count + r.orange_count + r.blue_count + r.normal_count
            data.append({
                '时间戳': r.timestamp,
                '记录类型': {'rule': '规则应用', 'emergency': '应急模拟', 'initial': '初始状态'}.get(r.record_type, r.record_type),
                '红色预警': r.red_count,
                '橙色预警': r.orange_count,
                '蓝色预警': r.blue_count,
                '正常': r.normal_count,
                '总计': total,
                '预警率(%)': f"{(r.red_count + r.orange_count + r.blue_count) / total * 100:.1f}" if total > 0 else "0.0",
                '应急措施描述': r.measures_desc,
            })
        return pd.DataFrame(data)

    def to_csv(self) -> bytes:
        df = self.to_dataframe()
        return df.to_csv(index=False).encode('utf-8-sig')

    def is_empty(self) -> bool:
        return len(self.records) == 0

    def add_batch_record(
        self,
        record_type: str,
        stats: Dict,
        measures_desc: str,
        batch_id: str,
        plan_name: str,
        total_cost: Optional[float] = None,
        normal_pct: Optional[float] = None,
        key_point_do_delta: Optional[float] = None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = WarningHistoryRecord(
            timestamp=timestamp,
            record_type=record_type,
            red_count=stats['red_count'],
            orange_count=stats['orange_count'],
            blue_count=stats['blue_count'],
            normal_count=stats['normal_count'],
            measures_desc=measures_desc if measures_desc else "无",
            batch_id=batch_id,
            plan_name=plan_name,
            total_cost=total_cost,
            normal_pct=normal_pct,
            key_point_do_delta=key_point_do_delta,
        )
        self.records.insert(0, record)
        if len(self.records) > self.max_records:
            self.records = self.records[:self.max_records]

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for r in self.records:
            total = r.red_count + r.orange_count + r.blue_count + r.normal_count
            type_map = {
                'rule': '规则应用',
                'emergency': '应急模拟',
                'initial': '初始状态',
                'batch_emergency': '批量对比应急',
            }
            row = {
                '时间戳': r.timestamp,
                '记录类型': type_map.get(r.record_type, r.record_type),
                '红色预警': r.red_count,
                '橙色预警': r.orange_count,
                '蓝色预警': r.blue_count,
                '正常': r.normal_count,
                '总计': total,
                '预警率(%)': f"{(r.red_count + r.orange_count + r.blue_count) / total * 100:.1f}" if total > 0 else "0.0",
                '应急措施描述': r.measures_desc,
            }
            if r.batch_id:
                row['批次ID'] = r.batch_id[:16] + '...' if len(r.batch_id) > 16 else r.batch_id
            if r.plan_name:
                row['方案名称'] = r.plan_name
            if r.total_cost is not None:
                row['总成本(万元)'] = f"{r.total_cost:.2f}"
            if r.normal_pct is not None:
                row['正常断面占比(%)'] = f"{r.normal_pct:.1f}"
            if r.key_point_do_delta is not None:
                delta_sign = "+" if r.key_point_do_delta >= 0 else ""
                row['关键断面DO变化'] = f"{delta_sign}{r.key_point_do_delta:.2f}mg/L"
            data.append(row)
        return pd.DataFrame(data)


def generate_batch_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


@dataclass
class EmergencyPlan:
    plan_name: str
    measures: List[EmergencyMeasure]


@dataclass
class PlanSimulationResult:
    plan_name: str
    measures: List[EmergencyMeasure]
    result: Dict
    warning_data: Dict
    sim: RiverSimulation
    red_count: int
    orange_count: int
    normal_count: int
    normal_pct: float
    key_point_x: float
    key_point_do_before: float
    key_point_do_after: float
    key_point_do_delta: float
    total_cost: float
    cost_details: Dict


def calculate_plan_cost(
    measures: List[EmergencyMeasure],
    sim: RiverSimulation,
) -> Tuple[float, Dict]:
    total_cost = 0.0
    cost_details = {
        'aerator_cost': 0.0,
        'close_source_cost': 0.0,
        'reduce_cost': 0.0,
        'details': [],
    }

    for measure in measures:
        if measure.measure_type == 'aerator':
            flow_inject = measure.params.get('flow_inject', 0.3)
            aerator_cost = flow_inject * 10.0
            total_cost += aerator_cost
            cost_details['aerator_cost'] += aerator_cost
            cost_details['details'].append(
                f"增氧剂@{measure.position:.0f}m: 流量{flow_inject:.2f}m³/s × 10万 = {aerator_cost:.2f}万元"
            )
        elif measure.measure_type == 'close_source':
            source_name = measure.params.get('source_name', '')
            for s in sim.source_manager.get_point_sources():
                if isinstance(s, PointSource) and s.name == source_name:
                    daily_flow_m3 = s.flow_rate * 86400
                    daily_flow_ton = daily_flow_m3
                    close_cost = daily_flow_ton * 5000 / 10000
                    total_cost += close_cost
                    cost_details['close_source_cost'] += close_cost
                    cost_details['details'].append(
                        f"关闭[{source_name}]: 日排放量{daily_flow_ton:.1f}吨 × 5000元/吨 = {close_cost:.2f}万元"
                    )
        elif measure.measure_type == 'reduce_source':
            source_name = measure.params.get('source_name', '')
            reduce_ratio = measure.params.get('reduce_ratio', 0.5)
            for s in sim.source_manager.get_point_sources():
                if isinstance(s, PointSource) and s.name == source_name:
                    daily_flow_m3 = s.flow_rate * 86400
                    daily_flow_ton = daily_flow_m3
                    reduce_cost = daily_flow_ton * 5000 * reduce_ratio / 10000
                    total_cost += reduce_cost
                    cost_details['reduce_cost'] += reduce_cost
                    cost_details['details'].append(
                        f"削减[{source_name}]{reduce_ratio*100:.0f}%: 日排放量{daily_flow_ton:.1f}吨 × 5000元/吨 × {reduce_ratio:.2f} = {reduce_cost:.2f}万元"
                    )

    return total_cost, cost_details


def find_key_point_do(x: np.ndarray, do: np.ndarray, key_point_x: float) -> float:
    idx = int(np.argmin(np.abs(x - key_point_x)))
    return float(do[idx])


def run_single_plan_simulation(
    plan: EmergencyPlan,
    sim: RiverSimulation,
    base_kwargs: Dict,
    rules: WarningRules,
    base_result: Dict,
    base_warning_data: Dict,
) -> PlanSimulationResult:
    em_sim, em_result = run_emergency_simulation(
        sim=sim,
        measures=plan.measures,
        base_kwargs=base_kwargs,
    )
    em_warning = evaluate_warnings(em_result, rules)
    stats = em_warning['stats']

    point_sources = sim.source_manager.get_point_sources()
    if point_sources:
        key_point_x = point_sources[0].x + 500.0
    else:
        key_point_x = base_result['x'][-1] * 0.3

    do_before = find_key_point_do(base_result['x'], base_result['do'], key_point_x)
    do_after = find_key_point_do(em_result['x'], em_result['do'], key_point_x)
    do_delta = do_after - do_before

    total_cost, cost_details = calculate_plan_cost(plan.measures, sim)

    return PlanSimulationResult(
        plan_name=plan.plan_name,
        measures=plan.measures,
        result=em_result,
        warning_data=em_warning,
        sim=em_sim,
        red_count=stats['red_count'],
        orange_count=stats['orange_count'],
        normal_count=stats['normal_count'],
        normal_pct=stats['normal_pct'],
        key_point_x=key_point_x,
        key_point_do_before=do_before,
        key_point_do_after=do_after,
        key_point_do_delta=do_delta,
        total_cost=total_cost,
        cost_details=cost_details,
    )


def run_batch_emergency_simulations(
    plans: List[EmergencyPlan],
    sim: RiverSimulation,
    base_kwargs: Dict,
    rules: WarningRules,
    base_result: Dict,
    base_warning_data: Dict,
) -> Tuple[str, List[PlanSimulationResult]]:
    batch_id = generate_batch_id()
    results = []

    for plan in plans:
        plan_result = run_single_plan_simulation(
            plan=plan,
            sim=sim,
            base_kwargs=base_kwargs,
            rules=rules,
            base_result=base_result,
            base_warning_data=base_warning_data,
        )
        results.append(plan_result)

    results.sort(key=lambda r: (r.red_count, r.orange_count, -r.normal_pct))

    return batch_id, results


PLAN_LINE_COLORS = [
    '#8e44ad',
    '#16a085',
    '#d35400',
    '#2980b9',
    '#c0392b',
]

PLAN_LINESTYLES = [
    '--',
    ':',
    '-.',
    (0, (3, 5, 1, 5)),
    (0, (5, 10)),
]


def plot_river_warning_schematic_multi_plan(
    result: Dict,
    warning_data: Dict,
    sim: RiverSimulation,
    plan_results: List[PlanSimulationResult],
    hover_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 7),
) -> plt.Figure:
    x = warning_data['x']
    levels = warning_data['levels']
    n = len(x)

    n_plans = len(plan_results)
    height_main = 5 + 1.2 * n_plans
    fig_height = max(figsize[1], height_main)

    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(figsize[0], fig_height),
                                           gridspec_kw={'width_ratios': [3, 1.2]})

    river_y = 0.0
    river_height = 1.0
    band_height_top = 0.6

    x_min, x_max = x[0], x[-1]
    x_range = x_max - x_min if x_max > x_min else 1.0

    segment_width = (x[1] - x[0]) if n > 1 else x_range / n
    for i in range(n):
        x_start = x[i] - segment_width / 2
        color = level_to_color(int(levels[i]))
        rect = plt.Rectangle(
            (x_start, river_y - band_height_top / 2),
            segment_width, band_height_top,
            facecolor=color, edgecolor='none', alpha=0.85
        )
        ax_main.add_patch(rect)

    dash_y_bottom = river_y + band_height_top / 2 + 0.15
    band_height_dash = 0.28
    band_gap = 0.18

    for plan_idx, plan_result in enumerate(plan_results):
        plan_color = PLAN_LINE_COLORS[plan_idx % len(PLAN_LINE_COLORS)]
        plan_linestyle = PLAN_LINESTYLES[plan_idx % len(PLAN_LINESTYLES)]
        levels_plan = plan_result.warning_data['levels']
        current_dash_y = dash_y_bottom + plan_idx * (band_height_dash + band_gap)

        for i in range(n):
            x_start = x[i] - segment_width / 2
            cell_color = level_to_color(int(levels_plan[i]))
            rect = plt.Rectangle(
                (x_start, current_dash_y),
                segment_width, band_height_dash,
                facecolor='none', edgecolor=cell_color, linewidth=2,
                linestyle=plan_linestyle, alpha=0.95
            )
            ax_main.add_patch(rect)

        label_y = current_dash_y + band_height_dash / 2
        ax_main.text(
            x_min - x_range * 0.015, label_y,
            f"方案{plan_idx+1}:{plan_result.plan_name}",
            ha='right', va='center', fontsize=8.5, fontweight='bold',
            color=plan_color,
            transform=ax_main.get_yaxis_transform()
        )

    point_sources = sim.source_manager.get_point_sources()
    for ps in point_sources:
        if x_min <= ps.x <= x_max:
            ax_main.scatter(ps.x, river_y - band_height_top / 2 - 0.15,
                            marker='^', s=180, color='#8e44ad', zorder=10,
                            edgecolor='white', linewidth=1.5, label='排污口' if ps == point_sources[0] else "")
            ax_main.annotate(ps.name[:6], xy=(ps.x, river_y - band_height_top / 2 - 0.15),
                             xytext=(ps.x, river_y - band_height_top / 2 - 0.45),
                             ha='center', fontsize=8, color='#8e44ad', fontweight='bold')

    tributaries = sim.channel.tributaries
    for trib in tributaries:
        if x_min <= trib.x <= x_max:
            arrow_x = trib.x
            arrow_y_top = river_y + band_height_top / 2 + 0.55 + n_plans * (band_height_dash + band_gap)
            arrow_y_bottom = river_y + band_height_top / 2 + 0.15 + n_plans * (band_height_dash + band_gap)
            ax_main.annotate("",
                             xy=(arrow_x, arrow_y_bottom),
                             xytext=(arrow_x, arrow_y_top),
                             arrowprops=dict(arrowstyle='->', color='#16a085', lw=2.5),
                             zorder=10)
            ax_main.text(arrow_x, arrow_y_top + 0.12,
                         f"支流\n{trib.flow_rate:.1f}m³/s",
                         ha='center', va='bottom', fontsize=8,
                         color='#16a085', fontweight='bold')

    if hover_idx is not None and 0 <= hover_idx < n:
        hover_x = x[hover_idx]
        ax_main.axvline(x=hover_x, color='#2c3e50', linestyle=':', linewidth=1.5, alpha=0.6)
        ax_main.scatter([hover_x], [river_y], marker='o', s=120,
                        color='white', edgecolor='#2c3e50', linewidth=2.5, zorder=15)

    y_bottom_limit = -1.3
    y_top_limit = 1.6 + n_plans * (band_height_dash + band_gap)
    ax_main.set_xlim(x_min - x_range * 0.03, x_max + x_range * 0.03)
    ax_main.set_ylim(y_bottom_limit, y_top_limit)
    ax_main.set_xlabel('河流距离 (m)', fontsize=11, fontweight='bold')
    ax_main.set_title('河段预警状态示意图（含多方案对比）', fontsize=13, fontweight='bold', pad=12)
    ax_main.set_yticks([])
    ax_main.set_yticklabels([])
    for spine in ['left', 'right', 'top']:
        ax_main.spines[spine].set_visible(False)
    ax_main.spines['bottom'].set_position(('data', river_y - band_height_top / 2 - 0.65))
    ax_main.grid(axis='x', alpha=0.2, linestyle='--')

    legend_patches = []
    for lv in [WARNING_NORMAL, WARNING_BLUE, WARNING_ORANGE, WARNING_RED]:
        patch = mpatches.Patch(
            color=WARNING_COLORS[lv], alpha=0.85,
            label=f"{WARNING_NAMES[lv]}"
        )
        legend_patches.append(patch)

    legend_patches_plan = []
    for plan_idx, plan_result in enumerate(plan_results):
        plan_color = PLAN_LINE_COLORS[plan_idx % len(PLAN_LINE_COLORS)]
        plan_linestyle = PLAN_LINESTYLES[plan_idx % len(PLAN_LINESTYLES)]
        patch = mpatches.Patch(
            facecolor='none', edgecolor=plan_color,
            linewidth=2.5, linestyle=plan_linestyle,
            label=f"方案{plan_idx+1}: {plan_result.plan_name}"
        )
        legend_patches_plan.append(patch)

    all_handles = legend_patches + legend_patches_plan
    ax_main.legend(handles=all_handles, loc='upper right',
                   fontsize=7.5, framealpha=0.95, ncol=3 if n_plans > 2 else 2,
                   bbox_to_anchor=(1.0, 1.0))

    if hover_idx is not None and 0 <= hover_idx < n:
        display_idx = hover_idx
        bod_val = result['bod'][display_idx]
        do_val = result['do'][display_idx]
        nh3n_val = result['nh3n'][display_idx]
        detail = warning_data['details'][display_idx]
        overall_lv = detail['overall_level']

        ax_info.axis('off')
        title_color = WARNING_COLORS.get(overall_lv, '#333333')
        ax_info.text(0.5, 0.98, f"📍 位置: x={x[display_idx]:.0f}m",
                     ha='center', va='top', fontsize=12, fontweight='bold',
                     transform=ax_info.transAxes)
        ax_info.text(0.5, 0.92, f"基准: {WARNING_NAMES[overall_lv]}",
                     ha='center', va='top', fontsize=11, fontweight='bold',
                     color=title_color,
                     transform=ax_info.transAxes)

        info_items = [
            ('BOD', bod_val, detail['bod_level'], 'mg/L', True),
            ('DO', do_val, detail['do_level'], 'mg/L', False),
            ('NH3-N', nh3n_val, detail['nh3n_level'], 'mg/L', True),
        ]

        y_pos = 0.82
        for name, val, lv, unit, lower_better in info_items:
            lv_color = WARNING_COLORS.get(lv, '#333333')
            ax_info.text(0.10, y_pos,
                         f"{name}: {val:.2f} {unit}",
                         ha='left', va='center', fontsize=9.5,
                         color=lv_color, fontweight='bold',
                         transform=ax_info.transAxes)
            y_pos -= 0.035

        y_pos -= 0.03
        ax_info.text(0.5, y_pos, "── 各方案在该位置对比 ──",
                     ha='center', va='center', fontsize=9.5,
                     fontweight='bold', color='#555555',
                     transform=ax_info.transAxes)
        y_pos -= 0.03

        for plan_idx, plan_result in enumerate(plan_results):
            if plan_idx >= 4:
                ax_info.text(0.5, y_pos, f"... 其余{len(plan_results)-4}个方案省略 ...",
                             ha='center', va='center', fontsize=8.5,
                             color='#7f8c8d', style='italic',
                             transform=ax_info.transAxes)
                break
            plan_color = PLAN_LINE_COLORS[plan_idx % len(PLAN_LINE_COLORS)]
            em_bod = plan_result.result['bod'][display_idx]
            em_do = plan_result.result['do'][display_idx]
            em_nh3n = plan_result.result['nh3n'][display_idx]
            em_overall = plan_result.warning_data['details'][display_idx]['overall_level']
            em_lv_color = WARNING_COLORS.get(em_overall, '#333333')

            ax_info.text(0.05, y_pos,
                         f"方案{plan_idx+1} {plan_result.plan_name[:8]}:",
                         ha='left', va='center', fontsize=8.5,
                         color=plan_color, fontweight='bold',
                         transform=ax_info.transAxes)
            ax_info.text(0.55, y_pos,
                         f"BOD{em_bod:.1f} DO{em_do:.1f} NH3N{em_nh3n:.1f}",
                         ha='left', va='center', fontsize=8,
                         color=em_lv_color,
                         transform=ax_info.transAxes)
            y_pos -= 0.028

    else:
        ax_info.axis('off')
        ax_info.text(0.5, 0.5,
                     "💡 在左侧滑动滑块\n选择位置查看详情",
                     ha='center', va='center', fontsize=11,
                     color='#7f8c8d', style='italic',
                     transform=ax_info.transAxes)
        ax_info.set_title('断面信息面板', fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    return fig


@dataclass
class RecommendationResult:
    mode: str
    recommended_plan_idx: Optional[int]
    recommended_plan_name: Optional[str]
    reason: str
    can_clear_red: bool


def recommend_optimal_plan(
    plan_results: List[PlanSimulationResult],
    mode: str,
    base_normal_count: int,
) -> RecommendationResult:
    if not plan_results:
        return RecommendationResult(
            mode=mode,
            recommended_plan_idx=None,
            recommended_plan_name=None,
            reason="无可对比的方案",
            can_clear_red=False,
        )

    clear_red_plans = [r for r in plan_results if r.red_count == 0]
    can_clear_red = len(clear_red_plans) > 0

    if mode == 'cost_min':
        if not can_clear_red:
            return RecommendationResult(
                mode=mode,
                recommended_plan_idx=None,
                recommended_plan_name=None,
                reason="当前方案均无法消除红色预警，建议增加措施力度",
                can_clear_red=False,
            )
        sorted_plans = sorted(clear_red_plans, key=lambda r: (r.total_cost, r.orange_count))
        best = sorted_plans[0]
        best_idx = plan_results.index(best)
        reason = (
            f"在{len(clear_red_plans)}个能清零红色预警的方案中，"
            f"【{best.plan_name}】总成本最低（{best.total_cost:.2f}万元），"
            f"同时橙色预警{best.orange_count}个，正常断面占比{best.normal_pct:.1f}%。"
        )
        return RecommendationResult(
            mode=mode,
            recommended_plan_idx=best_idx,
            recommended_plan_name=best.plan_name,
            reason=reason,
            can_clear_red=True,
        )

    elif mode == 'effect_max':
        sorted_plans = sorted(plan_results, key=lambda r: (-r.normal_pct, r.red_count, r.orange_count))
        best = sorted_plans[0]
        best_idx = plan_results.index(best)
        if best.red_count == 0:
            red_status = "已清零红色预警"
        else:
            red_status = f"仍有{best.red_count}个红色预警"
        reason = (
            f"在所有{len(plan_results)}个方案中，"
            f"【{best.plan_name}】的正常断面占比最高（{best.normal_pct:.1f}%），"
            f"{red_status}，橙色预警{best.orange_count}个，"
            f"不考虑成本约束的情况下效果最优。"
        )
        return RecommendationResult(
            mode=mode,
            recommended_plan_idx=best_idx,
            recommended_plan_name=best.plan_name,
            reason=reason,
            can_clear_red=best.red_count == 0,
        )

    elif mode == 'cost_effective':
        ranked = []
        for r in plan_results:
            normal_increase = max(r.normal_count - base_normal_count, 0)
            if r.total_cost > 0:
                ratio = normal_increase / r.total_cost
            else:
                ratio = float('inf') if normal_increase > 0 else 0.0
            ranked.append((ratio, r))
        ranked.sort(key=lambda x: (-x[0], x[1].red_count, x[1].orange_count))
        best_ratio, best = ranked[0]
        best_idx = plan_results.index(best)

        if best_ratio == float('inf'):
            ratio_desc = "零成本"
        else:
            ratio_desc = f"{best_ratio:.2f}个正常断面/万元"

        normal_increase = max(best.normal_count - base_normal_count, 0)
        reason = (
            f"以【正常断面增加数÷总成本】为性价比指标，"
            f"【{best.plan_name}】性价比最优（{ratio_desc}）："
            f"投入{best.total_cost:.2f}万元，正常断面较基准增加{normal_increase}个"
            f"（{base_normal_count}→{best.normal_count}），"
            f"当前正常断面占比{best.normal_pct:.1f}%。"
        )
        return RecommendationResult(
            mode=mode,
            recommended_plan_idx=best_idx,
            recommended_plan_name=best.plan_name,
            reason=reason,
            can_clear_red=best.red_count == 0,
        )

    else:
        return RecommendationResult(
            mode=mode,
            recommended_plan_idx=0,
            recommended_plan_name=plan_results[0].plan_name,
            reason="未知优化模式，默认返回第一个方案",
            can_clear_red=plan_results[0].red_count == 0,
        )
