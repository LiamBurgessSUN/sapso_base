import itertools
import numpy as np
import pandas as pd

from plots import plot_values_line_with_std
from Swarm import Swarm

df = pd.DataFrame(columns=['c1', 'c2', 'w', 'fitness'])

# w_range = np.linspace(0.4, 1.1, 20)
# c1_range = np.linspace(0.8, 2.5, 20)
# c2_range = np.linspace(0.8, 2.5, 20)

w_range = np.array((np.float64(0.729844),), dtype=np.float64)
c1_range = np.array((np.float64(1.496180),), dtype=np.float64)
c2_range = np.array((np.float64(1.496180),), dtype=np.float64)

# Generate search space tuples
grid_space = list(itertools.product(c1_range, c2_range, w_range))
iterations = 5000
for sample in grid_space:
    stop_count = 0
    swarm = Swarm(
        number_particles=30,
        bounds=(-100, 100),
    )
    swarm.set_control_parameters(sample[0], sample[1], sample[2])
    vals = ()
    for l in range(iterations):
        vals = swarm.step(l)
        print(vals[0])

    print(f"\n{swarm.best_fitness}")
    print(f"\n{swarm.global_best_position}")

    df.loc[len(df)] = sample[0], sample[1], sample[2], swarm.best_fitness

plot_values_line_with_std(len(swarm.avg_velocity_log), swarm.avg_velocity_log, swarm.avg_velocity_std_log,
                          "Velocity Log")
plot_values_line_with_std(len(swarm.avg_velocity_log), swarm.diversity_log, swarm.diversity_std_log, "Diversity Log")
plot_values_line_with_std(len(swarm.avg_velocity_log), swarm.percentage_bound_log, swarm.percentage_bound_std_log,
                          "Percentage Bound")
plot_values_line_with_std(len(swarm.fitness_log), swarm.fitness_log, swarm.fitness_std_log, "Best Fitness Log")
