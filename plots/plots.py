import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Swarm import Swarm


def plot_runs_from_dataframe(runs_groupby, metric: str = "best_fitness", title: str = "", use_log: bool = True):
    """
    Extracts multiple independent runs from a Pandas GroupBy object and plots them interactively.

    Args:
        runs_groupby: A Pandas DataFrameGroupBy object (e.g., df.groupby(["run"]))
        metric (str): The column name to plot (e.g., "best_fitness", "avg_velocity")
        title (str): Optional title prefix
        use_log (bool): Whether to use a log scale for the Y-axis
    """
    series_list = []
    labels = []

    for run_id, group_df in runs_groupby:
        # If your run_id comes out as a tuple (depending on pandas version), extract the first element
        run_label = run_id[0] if isinstance(run_id, tuple) else run_id

        series_list.append(group_df[metric].tolist())
        labels.append(f"Run {run_label}")

    plot_title = f"{title}: {metric}" if title else f"{len(labels)} Independent Runs: {metric}"

    plot_multiple_series_any(
        series_list=series_list,
        labels=labels,
        title=plot_title,
        use_log=use_log
    )


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
