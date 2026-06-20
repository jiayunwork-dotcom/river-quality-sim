
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class CrossSection:
    """河道断面类"""
    x: float
    shape: str = 'rectangular'
    bottom_width: float = 10.0
    side_slope: float = 1.5
    slope: float = 0.001
    manning_n: float = 0.03
    length: float = 1000.0

    def area(self, h: float) -> float:
        """计算过水面积"""
        if self.shape == 'rectangular':
            return self.bottom_width * h
        elif self.shape == 'trapezoidal':
            return (self.bottom_width + self.side_slope * h) * h
        return 0.0

    def wetted_perimeter(self, h: float) -> float:
        """计算湿周"""
        if self.shape == 'rectangular':
            return self.bottom_width + 2 * h
        elif self.shape == 'trapezoidal':
            return self.bottom_width + 2 * h * np.sqrt(1 + self.side_slope ** 2)
        return 0.0

    def hydraulic_radius(self, h: float) -> float:
        """计算水力半径"""
        P = self.wetted_perimeter(h)
        if P == 0:
            return 0.0
        return self.area(h) / P

    def top_width(self, h: float) -> float:
        """计算水面宽度"""
        if self.shape == 'rectangular':
            return self.bottom_width
        elif self.shape == 'trapezoidal':
            return self.bottom_width + 2 * self.side_slope * h
        return 0.0


@dataclass
class Tributary:
    """支流类"""
    x: float
    flow_rate: float = 0.0
    bod_conc: float = 0.0
    do_conc: float = 8.0
    nh3n_conc: float = 0.0
    cod_conc: float = 0.0
    is_inflow: bool = True


@dataclass
class RiverChannel:
    """河道类，由多个断面串联组成"""
    sections: List[CrossSection] = field(default_factory=list)
    tributaries: List[Tributary] = field(default_factory=list)

    def add_section(self, section: CrossSection):
        """添加断面"""
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.x)

    def add_tributary(self, tributary: Tributary):
        """添加支流"""
        self.tributaries.append(tributary)
        self.tributaries.sort(key=lambda t: t.x)

    def total_length(self) -> float:
        """计算河道总长度"""
        if not self.sections:
            return 0.0
        return self.sections[-1].x - self.sections[0].x

    def get_section_at(self, x: float) -> CrossSection:
        """获取指定位置的断面（线性插值）"""
        if not self.sections:
            raise ValueError("No sections defined")

        if x <= self.sections[0].x:
            return self.sections[0]
        if x >= self.sections[-1].x:
            return self.sections[-1]

        for i in range(len(self.sections) - 1):
            if self.sections[i].x <= x <= self.sections[i + 1].x:
                return self.sections[i]

        return self.sections[-1]

    def get_tributaries_between(self, x_start: float, x_end: float) -> List[Tributary]:
        """获取指定区间内的支流"""
        return [t for t in self.tributaries if x_start < t.x <= x_end]

    def mix_tributary(self, main_flow: float, main_conc: dict,
                       trib: Tributary) -> Tuple[float, dict]:
        """支流与主流混合计算"""
        if trib.is_inflow:
            total_flow = main_flow + trib.flow_rate
            if total_flow == 0:
                return 0.0, main_conc

            mixed_conc = {}
            for key in main_conc.keys():
                trib_key = f"{key}_conc"
                trib_val = getattr(trib, trib_key, 0.0)
                mixed_conc[key] = (main_flow * main_conc[key] +
                                   trib.flow_rate * trib_val) / total_flow
            return total_flow, mixed_conc
        else:
            new_flow = main_flow - trib.flow_rate
            if new_flow < 0:
                new_flow = 0.0
            return new_flow, main_conc.copy()
