import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Swarm import Swarm


def plot_swarm(swarm: Swarm, func_name: str):
    plot_values_line_with_std(
        num_iterations=len(swarm.local_cognitive_c1_log),
        data_points=swarm.local_cognitive_c1_log,
        title=f"SAC-SAPSO: {func_name} C1"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.global_cognitive_c2_log),
        data_points=swarm.global_cognitive_c2_log,
        title=f"SAC-SAPSO: {func_name} C2"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.inertia_log),
        data_points=swarm.inertia_log,
        title=f"SAC-SAPSO: {func_name} Inertia"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.best_log),
        data_points=swarm.best_log,
        title=f"SAC-SAPSO: {func_name} Fitness"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.fitness_log),
        data_points=swarm.fitness_log,
        std_dev=swarm.fitness_std_log,
        title=f"SAC-SAPSO: {func_name} Fitness"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.avg_velocity_log),
        data_points=swarm.avg_velocity_log,
        std_dev=swarm.avg_velocity_std_log,
        title=f"SAC-SAPSO: {func_name} Avg Velocity",
        use_log=True
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.diversity_log),
        data_points=swarm.diversity_log,
        std_dev=swarm.diversity_std_log,
        title=f"SAC-SAPSO: {func_name} Diversity"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.percentage_bound_log),
        data_points=swarm.percentage_bound_log,
        std_dev=swarm.percentage_bound_std_log,
        title=f"SAC-SAPSO: {func_name} % Bound"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.stability_log),
        data_points=swarm.stability_log,
        title=f"SAC-SAPSO: {func_name} % Stability"
    )


def plot_values_line_with_std(num_iterations: int, data_points: list, std_dev: list = None, title: str = "",
                              use_log: bool = False):
    """
    Plots a professional line graph with an optional standard deviation shadow.
    Includes a flag for Symmetrical Logarithmic Y-axis scaling to handle values around 0.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    x_values = np.arange(num_iterations)
    y_values = np.array(data_points)

    # 1. Plot the main trend line
    plot = sns.lineplot(
        x=x_values,
        y=y_values,
        linewidth=1.2,
        marker=None,
        color='#1f77b4',
        label='Mean/Best Fitness'
    )

    # 2. Add the Shadow Area (Standard Deviation)
    if std_dev is not None:
        std_array = np.array(std_dev)
        plt.fill_between(
            x_values,
            y_values - std_array,
            y_values + std_array,
            color='#1f77b4',
            alpha=0.2,
            label='Std Dev'
        )

    # Professional Labeling & Styling
    plot.set_title(f"{title} ({num_iterations} Iterations)", fontsize=16)
    plot.set_xlabel("Iteration Index", fontsize=12)
    plot.set_ylabel("Value (SymLog Scale)" if use_log else "Value", fontsize=12)

    # Maintain the 'Breathing Room' on the X-Axis
    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)

    # 3. Apply Symmetrical Log Scale if requested
    # symlog is required to "center" the log scale at 0 and handle negative values.
    if use_log:
        # linthresh determines the size of the linear region around 0.
        # We set it to a small value relative to typical fitness ranges.
        plt.yscale('symlog', linthresh=1e-6)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grid_search(df, title: str = ""):
    pivot_table = df.pivot_table(
        index='c2',
        columns='c1',
        values='fitness',
        aggfunc='mean'
    )

    plt.figure(figsize=(12, 9))
    sns.set_theme(style="white")

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        linewidths=.5,
        cbar_kws={'label': 'Mean Fitness'}
    )

    plt.title(f"PSO Parameter Sensitivity: {title}", fontsize=16)
    plt.xlabel("Cognitive Coefficient ($c_1$)", fontsize=12)
    plt.ylabel("Social Coefficient ($c_2$)", fontsize=12)
    plt.show()


def plot_values_line(num_iterations: int, data_points: list, title: str = "", use_log: bool = False):
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
    plot.set_ylabel("Value", fontsize=12)

    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)

    if use_log:
        plt.yscale('symlog', linthresh=1e-6)

    plt.tight_layout()
    plt.show()


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
