
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class SourceType(Enum):
    POINT = "point"
    NONPOINT = "nonpoint"
    ACCIDENTAL = "accidental"


class ReleaseType(Enum):
    INSTANTANEOUS = "instantaneous"
    CONTINUOUS = "continuous"


@dataclass
class PollutionSource:
    """污染源基类"""
    name: str
    source_type: SourceType
    x: float

    def get_mass_input(self, t: float = 0.0) -> Dict[str, float]:
        return {}


@dataclass
class PointSource(PollutionSource):
    """点源污染"""
    flow_rate: float = 0.0
    bod_conc: float = 0.0
    do_conc: float = 8.0
    nh3n_conc: float = 0.0
    cod_conc: float = 0.0

    def __init__(self, name: str, x: float, flow_rate: float = 0.0,
                 bod_conc: float = 0.0, do_conc: float = 8.0,
                 nh3n_conc: float = 0.0, cod_conc: float = 0.0):
        super().__init__(name=name, source_type=SourceType.POINT, x=x)
        self.flow_rate = flow_rate
        self.bod_conc = bod_conc
        self.do_conc = do_conc
        self.nh3n_conc = nh3n_conc
        self.cod_conc = cod_conc

    def get_mass_input(self, t: float = 0.0) -> Dict[str, float]:
        return {
            'bod': self.flow_rate * self.bod_conc,
            'do': self.flow_rate * self.do_conc,
            'nh3n': self.flow_rate * self.nh3n_conc,
            'cod': self.flow_rate * self.cod_conc,
            'flow_rate': self.flow_rate,
        }


@dataclass
class NonPointSource(PollutionSource):
    """面源污染"""
    area: float = 1000000.0
    bod_load: float = 0.0
    nh3n_load: float = 0.0
    cod_load: float = 0.0
    runoff_coeff: float = 0.3
    start_x: float = 0.0
    end_x: float = 1000.0

    def __init__(self, name: str, start_x: float, end_x: float,
                 area: float = 1000000.0, bod_load: float = 0.0,
                 nh3n_load: float = 0.0, cod_load: float = 0.0,
                 runoff_coeff: float = 0.3):
        super().__init__(name=name, source_type=SourceType.NONPOINT,
                         x=(start_x + end_x) / 2)
        self.area = area
        self.bod_load = bod_load
        self.nh3n_load = nh3n_load
        self.cod_load = cod_load
        self.runoff_coeff = runoff_coeff
        self.start_x = start_x
        self.end_x = end_x

    def get_mass_input(self, t: float = 0.0) -> Dict[str, float]:
        return {
            'bod': self.area * self.bod_load,
            'nh3n': self.area * self.nh3n_load,
            'cod': self.area * self.cod_load,
            'flow_rate': self.area * self.runoff_coeff * 0.001,
        }

    def get_distributed_load(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """获取分布在空间上的负荷
        单位换算：
        输入: self.area (m²), self.bod_load (kg/km²·d)
        输出: g/(m·s) - 单位河流长度上的质量输入率
        """
        dx = x[1] - x[0] if len(x) > 1 else 1.0

        bod_dist = np.zeros_like(x)
        nh3n_dist = np.zeros_like(x)
        cod_dist = np.zeros_like(x)

        mask = (x >= self.start_x) & (x <= self.end_x)
        total_length = np.sum(mask) * dx if np.sum(mask) > 0 else 1.0

        if total_length > 0:
            area_km2 = self.area / 1e6
            kg_per_day_to_g_per_s = 1000.0 / 86400.0

            bod_total_g_per_s = area_km2 * self.bod_load * kg_per_day_to_g_per_s
            nh3n_total_g_per_s = area_km2 * self.nh3n_load * kg_per_day_to_g_per_s
            cod_total_g_per_s = area_km2 * self.cod_load * kg_per_day_to_g_per_s

            bod_dist[mask] = bod_total_g_per_s / total_length
            nh3n_dist[mask] = nh3n_total_g_per_s / total_length
            cod_dist[mask] = cod_total_g_per_s / total_length

        return {
            'bod': bod_dist,
            'nh3n': nh3n_dist,
            'cod': cod_dist,
        }


@dataclass
class AccidentalSource(PollutionSource):
    """突发泄漏源"""
    release_type: ReleaseType = ReleaseType.INSTANTANEOUS
    total_mass_bod: float = 1000.0
    total_mass_nh3n: float = 100.0
    total_mass_cod: float = 500.0
    release_duration: float = 3600.0
    start_time: float = 0.0
    flow_rate: float = 0.5

    def __init__(self, name: str, x: float,
                 release_type: ReleaseType = ReleaseType.INSTANTANEOUS,
                 total_mass_bod: float = 1000.0,
                 total_mass_nh3n: float = 100.0,
                 total_mass_cod: float = 500.0,
                 release_duration: float = 3600.0,
                 start_time: float = 0.0,
                 flow_rate: float = 0.5):
        super().__init__(name=name, source_type=SourceType.ACCIDENTAL, x=x)
        self.release_type = release_type
        self.total_mass_bod = total_mass_bod
        self.total_mass_nh3n = total_mass_nh3n
        self.total_mass_cod = total_mass_cod
        self.release_duration = release_duration
        self.start_time = start_time
        self.flow_rate = flow_rate

    def get_mass_input(self, t: float = 0.0) -> Dict[str, float]:
        if self.release_type == ReleaseType.INSTANTANEOUS:
            if abs(t - self.start_time) < 1e-6:
                return {
                    'bod': self.total_mass_bod,
                    'nh3n': self.total_mass_nh3n,
                    'cod': self.total_mass_cod,
                    'instantaneous': True,
                }
            return {}
        else:
            if self.start_time <= t <= self.start_time + self.release_duration:
                rate_bod = self.total_mass_bod / self.release_duration
                rate_nh3n = self.total_mass_nh3n / self.release_duration
                rate_cod = self.total_mass_cod / self.release_duration
                return {
                    'bod': rate_bod,
                    'nh3n': rate_nh3n,
                    'cod': rate_cod,
                    'flow_rate': self.flow_rate,
                }
            return {}


@dataclass
class SourceManager:
    """污染源管理器"""
    sources: List[PollutionSource] = field(default_factory=list)

    def add_source(self, source: PollutionSource):
        self.sources.append(source)
        self.sources.sort(key=lambda s: s.x)

    def remove_source(self, name: str):
        self.sources = [s for s in self.sources if s.name != name]

    def get_point_sources(self) -> List[PointSource]:
        return [s for s in self.sources if isinstance(s, PointSource)]

    def get_nonpoint_sources(self) -> List[NonPointSource]:
        return [s for s in self.sources if isinstance(s, NonPointSource)]

    def get_accidental_sources(self) -> List[AccidentalSource]:
        return [s for s in self.sources if isinstance(s, AccidentalSource)]

    def get_total_point_sources_at(self, x: float, dx: float) -> Dict[str, float]:
        """获取指定位置附近的点源总输入"""
        total = {'bod': 0, 'do': 0, 'nh3n': 0, 'cod': 0, 'flow_rate': 0}
        for s in self.get_point_sources():
            if abs(s.x - x) < dx / 2:
                inputs = s.get_mass_input()
                for k, v in inputs.items():
                    if k in total:
                        total[k] += v
        return total

    def get_distributed_loads(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有面源的空间分布负荷"""
        loads = {'bod': np.zeros_like(x), 'nh3n': np.zeros_like(x), 'cod': np.zeros_like(x)}
        for s in self.get_nonpoint_sources():
            dist_loads = s.get_distributed_load(x)
            for k in loads:
                loads[k] += dist_loads.get(k, np.zeros_like(x))
        return loads
