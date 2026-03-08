import itertools
import numpy as np
import pandas as pd

from fitness_function.FitnessFunction import (
    EllipticFunction,
    Bohachevsky1Function,
    BonyadiMichalewiczFunction,
    BrownFunction,
    CosineMixtureFunction,
    CrossLegTableFunction,
    DeflectedCorrugatedSpringFunction,
    DiscussFunction,
    DropWaveFunction,
    EggCrateFunction,
    EggHolderFunction
)

from plots import plot_values_line_with_std
from Swarm import Swarm

# 1. Setup Data Collection
df = pd.DataFrame(columns=['c1', 'c2', 'w', 'fitness', 'iterations_taken'])

# 2. Select Fitness Function
# You can swap this with any of the newly added functions (e.g., EggHolderFunction())
ff = EllipticFunction()

# 3. Define Parameter Search Space (Constant PSO Baseline)
w_range = np.array((np.float64(0.729844),), dtype=np.float64)
c1_range = np.array((np.float64(1.496180),), dtype=np.float64)
c2_range = np.array((np.float64(1.496180),), dtype=np.float64)

grid_space = list(itertools.product(c1_range, c2_range, w_range))

# 4. Simulation Configuration
MAX_ITERATIONS = 5000
PATIENCE = 100  # Number of steps to wait for improvement before stopping

print(f"🚀 Starting Swarm Simulation on {ff.__class__.__name__}")

for sample in grid_space:
    c1, c2, w = sample

    # Initialize Swarm with new items: stagnation threshold and patience
    swarm = Swarm(
        number_particles=30,
        fitness_function=ff,
        expected_iterations=MAX_ITERATIONS,
        stagnation_threshold=1e-12,
        stagnation_patience=PATIENCE
    )

    swarm.set_control_parameters(c1, c2, w)

    # 5. Run Search Loop
    actual_iterations = 0
    for i in range(MAX_ITERATIONS):
        actual_iterations = i + 1
        best_f, best_pos = swarm.step(i)

        # Monitor progress (optional printing)
        if i % 500 == 0:
            print(f"Iteration {i}: Best Fitness = {best_f:.4e}")

        # --- NEW ITEM: Early Termination ---
        if swarm.is_stagnated():
            print(f"🛑 Stagnation detected at iteration {i}. Terminating early.")
            break

    print(f"\n✅ Results for (c1={c1}, c2={c2}, w={w}):")
    print(f"🏆 Final Fitness: {swarm.best_fitness:.6e}")
    print(f"⏱️  Iterations: {actual_iterations}")
    print(f"Best Position: {swarm.global_best_position}")

    # Record data
    df.loc[len(df)] = [c1, c2, w, swarm.best_fitness, actual_iterations]

# 6. Comprehensive Visualization
# Using the standard plot_values_line_with_std for all tracked metrics
print("\n📊 Generating diagnostic plots...")

plot_values_line_with_std(
    len(swarm.avg_velocity_log),
    swarm.avg_velocity_log,
    swarm.avg_velocity_std_log,
    "Particle Velocity Log"
)

plot_values_line_with_std(
    len(swarm.diversity_log),
    swarm.diversity_log,
    swarm.diversity_std_log,
    "Swarm Diversity Log"
)

plot_values_line_with_std(
    len(swarm.percentage_bound_log),
    swarm.percentage_bound_log,
    swarm.percentage_bound_std_log,
    "Search Boundary Adherence"
)

plot_values_line_with_std(
    len(swarm.best_log),
    swarm.best_log,
    title=f"Global Best Fitness Evolution ({ff.__class__.__name__})"
)