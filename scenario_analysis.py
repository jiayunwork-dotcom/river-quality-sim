
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from copy import deepcopy


@dataclass
class Scenario:
    """情景类"""
    name: str
    description: str = ""
    is_baseline: bool = False

    upstream_flow: float = 10.0
    upstream_bod: float = 2.0
    upstream_do: float = 8.5
    upstream_nh3n: float = 0.5
    upstream_cod: float = 5.0

    point_sources: List[Dict] = field(default_factory=list)
    nonpoint_sources: List[Dict] = field(default_factory=list)
    accidental_sources: List[Dict] = field(default_factory=list)

    K1: float = 0.25
    K2: float = 0.5
    Dx: float = 10.0

    flow_mode: str = 'uniform'

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'is_baseline': self.is_baseline,
            'upstream_flow': self.upstream_flow,
            'upstream_bod': self.upstream_bod,
            'upstream_do': self.upstream_do,
            'upstream_nh3n': self.upstream_nh3n,
            'upstream_cod': self.upstream_cod,
            'point_sources': self.point_sources,
            'nonpoint_sources': self.nonpoint_sources,
            'accidental_sources': self.accidental_sources,
            'K1': self.K1,
            'K2': self.K2,
            'Dx': self.Dx,
            'flow_mode': self.flow_mode,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Scenario':
        scenario = cls(name=data['name'])
        for key, value in data.items():
            if hasattr(scenario, key):
                setattr(scenario, key, value)
        return scenario


@dataclass
class ScenarioResult:
    """情景计算结果"""
    scenario_name: str
    x: np.ndarray
    bod: np.ndarray
    do: np.ndarray
    nh3n: np.ndarray
    cod: np.ndarray
    h: np.ndarray
    V: np.ndarray
    Q: np.ndarray
    water_level: np.ndarray

    critical_x: Optional[float] = None
    critical_do: Optional[float] = None


class ScenarioManager:
    """情景管理器"""

    def __init__(self):
        self.scenarios: List[Scenario] = []
        self.results: Dict[str, ScenarioResult] = {}

    def add_scenario(self, scenario: Scenario):
        self.scenarios.append(scenario)

    def remove_scenario(self, name: str):
        self.scenarios = [s for s in self.scenarios if s.name != name]
        if name in self.results:
            del self.results[name]

    def get_baseline(self) -> Optional[Scenario]:
        for s in self.scenarios:
            if s.is_baseline:
                return s
        return None

    def get_scenario_names(self) -> List[str]:
        return [s.name for s in self.scenarios]

    def compare_scenarios(self, scenario_names: List[str] = None) -> List[ScenarioResult]:
        """对比多个情景的结果"""
        if scenario_names is None:
            scenario_names = self.get_scenario_names()

        results = []
        for name in scenario_names:
            if name in self.results:
                results.append(self.results[name])

        return results

    def calculate_improvement(self, baseline_name: str, scenario_name: str,
                              component: str = 'bod') -> Dict:
        """计算某情景相对于基准情景的改善程度"""
        if baseline_name not in self.results or scenario_name not in self.results:
            return {'improvement_percent': 0, 'max_reduction': 0, 'avg_reduction': 0}

        baseline = self.results[baseline_name]
        scenario = self.results[scenario_name]

        baseline_val = getattr(baseline, component)
        scenario_val = getattr(scenario, component)

        if component == 'do':
            improvement = scenario_val - baseline_val
            improvement_percent = (improvement / baseline_val * 100) if baseline_val.mean() > 0 else 0
        else:
            reduction = baseline_val - scenario_val
            improvement_percent = (reduction / baseline_val * 100) if baseline_val.mean() > 0 else 0

        return {
            'improvement_percent': float(np.mean(improvement_percent)),
            'max_reduction': float(np.max(baseline_val - scenario_val)) if component != 'do' else float(np.max(scenario_val - baseline_val)),
            'avg_reduction': float(np.mean(baseline_val - scenario_val)) if component != 'do' else float(np.mean(scenario_val - baseline_val)),
        }

    def export_comparison_table(self, scenario_names: List[str] = None) -> Dict:
        """导出对比表格数据"""
        if scenario_names is None:
            scenario_names = self.get_scenario_names()

        table_data = {
            'scenarios': scenario_names,
            'max_bod': [],
            'min_do': [],
            'max_nh3n': [],
            'max_cod': [],
            'critical_x': [],
            'critical_do': [],
        }

        for name in scenario_names:
            if name in self.results:
                r = self.results[name]
                table_data['max_bod'].append(float(np.max(r.bod)))
                table_data['min_do'].append(float(np.min(r.do)))
                table_data['max_nh3n'].append(float(np.max(r.nh3n)))
                table_data['max_cod'].append(float(np.max(r.cod)))
                table_data['critical_x'].append(r.critical_x)
                table_data['critical_do'].append(r.critical_do)

        return table_data
