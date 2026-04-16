import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Swarm import Swarm


def plot_runs_from_dataframe(
        df: pd.DataFrame,
        metric: str = "best_fitness",
        title: str = "",
        use_log: bool = True
):
    """
    Groups a DataFrame by 'run' and 'step_number', then plots trajectories for a metric.

    This implementation follows the empirical analysis structure of von Eschwege & Engelbrecht (2024),
    where solution quality and swarm characteristics are tracked across independent runs
    to assess the effectiveness of the SAC-SAPSO agent.

    Args:
        df: The raw experiment DataFrame.
        metric: The column name to plot (e.g., 'best_fitness', 'diversity', 'velocity').
        title: Optional plot title.
        use_log: If True, uses a logarithmic scale for the y-axis (recommended for fitness/diversity).
    """
    # 1. Grouping logic
    # We group by 'run' to separate independent trajectories.
    # We don't group by 'step_number' for the line itself, but we sort by it to ensure
    # the temporal progression is correct in the visualization.
    runs = df.groupby("run")

    plt.figure(figsize=(10, 6))

    all_final_values = []

    for run_id, group_df in runs:
        # Ensure correct temporal ordering for the line plot
        sorted_group = group_df.sort_values("step_number")

        # Plot the individual run trajectory
        plt.plot(
            sorted_group["step_number"],
            sorted_group[metric],
            alpha=0.4,  # Lower alpha for individual runs to handle overlap
            linewidth=1,
            label=f"Run {run_id}" if len(runs) <= 10 else None  # Avoid legend clutter
        )

        # Track final values for potential stats reporting
        all_final_values.append(sorted_group[metric].iloc[-1])

    # Styling and Labels
    if use_log:
        plt.yscale("log")
        plt.ylabel(f"{metric} (Log Scale)")
    else:
        plt.ylabel(metric)

    plt.xlabel("Time Step (t)")

    # Paper-style titles
    plot_title = title if title else f"Convergence Behavior: {metric}"
    plt.title(plot_title)

    plt.grid(True, which="both", ls="-", alpha=0.2)

    if len(runs) <= 10:
        plt.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

    # Log summary statistics to console (SwarmProf style)
    print(f"📊 {metric} Stats (across {len(runs)} runs):")
    print(f"   - Mean: {np.mean(all_final_values):.4e}")
    print(f"   - StdDev: {np.std(all_final_values):.4e}")


def plot_multiple_series(series_list: list, labels: list, title: str = "", use_log: bool = False):
    """
    Plots multiple data series on a single graph.
    Expects series_list to be a list of lists/arrays of equal length.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Check if all series have the same length
    lengths = [len(s) for s in series_list]
    num_iterations = max(lengths) if lengths else 0
    x_values = np.arange(num_iterations)

    for i, data in enumerate(series_list):
        label = labels[i] if i < len(labels) else f"Series {i + 1}"
        sns.lineplot(
            x=np.arange(len(data)),
            y=data,
            linewidth=1.5,
            label=label
        )

    # Professional Labeling & Styling
    plt.title(f"{title} ({num_iterations} Iterations)", fontsize=16)
    plt.xlabel("Iteration Index", fontsize=12)
    plt.ylabel("Value (SymLog Scale)" if use_log else "Value", fontsize=12)

    # Maintain the 'Breathing Room' on the X-Axis
    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)

    # Apply Symmetrical Log Scale if requested
    if use_log:
        plt.yscale('symlog', linthresh=1e-6)

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_series_any(series_list: list, labels: list, title: str = "", use_log: bool = False):
    """
    Plots multiple data series on a single graph.
    Expects series_list to be a list of lists/arrays of equal length.
    Applies a color gradient to the lines based on the number of series.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))  # Increased height to space y-axis wider

    # Check if all series have the same length
    lengths = [len(s) for s in series_list]
    num_iterations = max(lengths) if lengths else 0

    num_series = len(series_list)
    # Generate a color gradient palette based on the exact number of series
    palette = sns.color_palette("viridis", n_colors=num_series)

    for i, data in enumerate(series_list):
        label = labels[i] if i < len(labels) else f"Series {i + 1}"
        sns.lineplot(
            x=np.arange(len(data)),
            y=data,
            linewidth=1.5,
            label=label,
            color=palette[i]  # Apply the gradient color here
        )

    # Professional Labeling & Styling
    plt.title(f"{title} ({num_iterations} Iterations)", fontsize=16)
    plt.xlabel("Iteration Index", fontsize=12)
    plt.ylabel("Value (SymLog Scale)" if use_log else "Value", fontsize=12)

    # Maintain the 'Breathing Room' on the X-Axis and Y-Axis
    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)
    plt.margins(y=0.15)  # Adds 15% extra padding to the top and bottom of the graph

    # Apply Symmetrical Log Scale if requested
    if use_log:
        plt.yscale('symlog', linthresh=1e-6)

    # Handle the legend intelligently if there are many lines
    if num_series > 10:
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    else:
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_swarm(swarm: Swarm, func_name: str):
    """
    Standard diagnostic plot suite for a completed Swarm run.
    """
    plot_multiple_series(
        [
            swarm.local_cognitive_c1_log,
            swarm.global_cognitive_c2_log,
            swarm.inertia_log
        ],
        [
            "C1",
            "C2",
            "Inertia"
        ],
        "Control Parameters"
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.best_log),
        data_points=swarm.best_log,
        title=f"SAC-SAPSO: {func_name} Best Fitness",
        use_log=False
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.fitness_log),
        data_points=swarm.fitness_log,
        std_dev=swarm.fitness_std_log,
        title=f"SAC-SAPSO: {func_name} Fitness",
        use_log=False
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
        title=f"SAC-SAPSO: {func_name} Diversity",
        use_log=True
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.percentage_bound_log),
        data_points=swarm.percentage_bound_log,
        title=f"SAC-SAPSO: {func_name} Feasible",
        use_log=False
    )

    plot_values_line_with_std(
        num_iterations=len(swarm.stability_log),
        data_points=swarm.stability_log,
        title=f"SAC-SAPSO: {func_name} Stability Condition",
        use_log=False
    )




def plot_values_line_with_std(num_iterations: int, data_points: list, std_dev: list = None, title: str = "",
                              use_log: bool = False):
    """
    Plots a professional line graph with an optional standard deviation shadow.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    x_values = np.arange(num_iterations)
    y_values = np.array(data_points)

    plot = sns.lineplot(
        x=x_values,
        y=y_values,
        linewidth=1.2,
        color='#1f77b4',
        label='Mean/Best'
    )

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

    plot.set_title(f"{title} ({num_iterations} Iterations)", fontsize=16)
    plot.set_xlabel("Iteration Index", fontsize=12)
    plot.set_ylabel("Value (SymLog Scale)" if use_log else "Value", fontsize=12)

    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)

    if use_log:
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


def plot_walk(steps, x, y, title: str = ""):
    df = pd.DataFrame({
        'Time': steps,
        'X-Coordinate': x,
        'Y-Coordinate': y
    })

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x='X-Coordinate',
        y='Y-Coordinate',
        hue='Time',
        palette='viridis',
        size='Time',
        legend='brief',
        alpha=0.7
    )

    plt.plot(x, y, color='gray', alpha=0.3, linestyle='--')
    plt.title(f"Particle Trajectory: {title}", fontsize=15)
    plt.show()
