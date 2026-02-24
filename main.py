import numpy as np
from Swarm import Swarm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_values_line(num_iterations: int, data_points: list):
    sns.set_theme(style="whitegrid")

    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    plot = sns.lineplot(x=range(num_iterations), y=data_points, marker='o', color='b')

    # 4. Professional Labeling
    plot.set_title("Time Series Analysis: Float Sequence", fontsize=16)
    plot.set_xlabel("Index", fontsize=12)
    plot.set_ylabel("Value", fontsize=12)

    # 5. Display/Save
    plt.show()


def plot_walk(steps, x, y):
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


swarm = Swarm(
    number_particles=20,
    bounds=np.array(((-5, 5), (-5, 5)), int),
)

iterations = 10000
for i in range(iterations):
    vals = swarm.step()
    print(vals)

plot_values_line(iterations, swarm.best_log)
plot_values_line(iterations,
                 (swarm.best_log - np.min(swarm.best_log)) / (np.max(swarm.best_log) - np.min(swarm.best_log)))
plot_values_line(iterations, swarm.diversity_log)
plot_values_line(iterations, swarm.percentage_bound_log)
plot_values_line(iterations, swarm.avg_velocity_log)
plot_walk(iterations, np.transpose(swarm.x_history)[0], np.transpose(swarm.y_history)[0])
plot_walk(iterations, np.transpose(swarm.x_history)[1], np.transpose(swarm.y_history)[1])
plot_walk(iterations, np.transpose(swarm.x_history)[2], np.transpose(swarm.y_history)[2])
