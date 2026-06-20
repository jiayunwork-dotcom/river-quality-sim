
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
