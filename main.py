import itertools

import numpy as np
from Swarm import Swarm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_grid_search(df, title: str = ""):
    # 1. Use pivot_table instead of pivot to handle the 3rd dimension (w)
    # This averages the Fitness across all 'w' values for each c1/c2 pair.
    pivot_table = df.pivot_table(
        index='c2',
        columns='c1',
        values='fitness',
        aggfunc='mean'  # Or 'min' if you want to see the best possible potential
    )

    # 2. Visualization
    plt.figure(figsize=(12, 9))
    sns.set_theme(style="white")

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=.5,
        cbar_kws={'label': 'Mean Fitness (across all w)'}
    )

    plt.title("PSO Parameter Sensitivity: $c_1$ vs $c_2$", fontsize=16)
    plt.xlabel("Cognitive Coefficient ($c_1$)", fontsize=12)
    plt.ylabel("Social Coefficient ($c_2$)", fontsize=12)
    plt.show()


def plot_values_line(num_iterations: int, data_points: list, title: str = ""):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plot = sns.lineplot(
        x=range(num_iterations),
        y=data_points,
        linewidth=0.9,
        marker=None,
        color='#1f77b4'
    )

    plot.set_title(f"{title} ({num_iterations} Iterations)", fontsize=16)
    plot.set_xlabel("Iteration Index", fontsize=12)
    plot.set_ylabel("Fitness Value", fontsize=12)

    # --- THE FIX ---
    # Instead of xlim(0, ...), we set a margin or a slight negative start
    # Option A: Manual offset (e.g., -50 iterations of padding)
    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)

    # Option B: Let Matplotlib handle it automatically with margins
    # plt.margins(x=0.05)
    # ----------------

    plt.tight_layout()
    plt.show()


def plot_walk(steps, x, y, title: str = ""):
    df = pd.DataFrame({
        'Time': steps,
        'X-Coordinate': x,
        'Y-Coordinate': y
    })

    # 3. Plotting with Seaborn
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 8))

    # We use relplot or scatterplot.
    # Mapping 'hue' to 'Time' creates the temporal progression.
    plot = sns.scatterplot(
        data=df,
        x='X-Coordinate',
        y='Y-Coordinate',
        hue='Time',
        palette='viridis',  # 'viridis' is perceptually uniform for time
        size='Time',  # Optional: points get larger as time progresses
        legend='brief',
        alpha=0.7
    )

    # 4. Adding a Path (Line) to show trajectory
    plt.plot(x, y, color='gray', alpha=0.3, linestyle='--')

    plt.title("Particle Trajectory Over Time", fontsize=15)
    plt.show()


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
    # swarm.sample_control_parameters_randomly()
    vals = ()
    for l in range(iterations):
        vals = swarm.step(l)
        print(vals[0])

    print(f"\n{swarm.best_fitness}")
    print(f"\n{swarm.global_best_position}")

    df.loc[len(df)] = sample[0], sample[1], sample[2], swarm.best_fitness

plot_values_line(len(swarm.avg_velocity_log), swarm.avg_velocity_log, "Velocity Log")
plot_values_line(len(swarm.avg_velocity_log), swarm.diversity_log, "Diversity Log")
plot_values_line(len(swarm.avg_velocity_log), swarm.percentage_bound_log, "Percentage Bound")
plot_values_line(len(swarm.best_log), swarm.best_log, "Best Fitness Log")
plot_values_line(len(swarm.best_log), swarm.best_log / np.linalg.norm(swarm.best_log), "Best Fitness Log Mean")
# plot_walk(iterations, [x[0][0] for x in swarm.history[:]], [x[0][1] for x in swarm.history[:]])
