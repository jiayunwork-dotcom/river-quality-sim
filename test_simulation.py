
from simulation_engine import RiverSimulation
import numpy as np

sim = RiverSimulation()
sim.setup_default_channel()

sim.add_point_source('排污口1', 2000.0, 0.5, 100, 2, 20, 150)

result = sim.run_steady_simulation(
    Q_upstream=10.0,
    initial_bod=3.0,
    initial_do=8.5,
    initial_nh3n=0.5,
    initial_cod=5.0,
    flow_mode='uniform',
    n_grid=100,
)

print('模拟成功!')
print(f'河道长度: {result["x"][-1]:.0f} m')
print(f'上游BOD: {result["bod"][0]:.2f} mg/L')
print(f'最大BOD: {np.max(result["bod"]):.2f} mg/L')
print(f'下游BOD: {result["bod"][-1]:.2f} mg/L')
print(f'上游DO: {result["do"][0]:.2f} mg/L')
print(f'最低DO: {np.min(result["do"]):.2f} mg/L')
print(f'下游DO: {result["do"][-1]:.2f} mg/L')
print(f'临界点位置: {result["critical_x"]:.1f} m')
print(f'临界点DO: {result["critical_do"]:.2f} mg/L')
print(f'平均流速: {np.mean(result["V"]):.3f} m/s')
