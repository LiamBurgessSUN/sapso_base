import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

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
    Plots multiple data series on a single interactive graph using Plotly.
    Expects series_list to be a list of lists/arrays of equal length.
    Applies a color gradient to the lines based on the number of series.
    """
    lengths = [len(s) for s in series_list]
    num_iterations = max(lengths) if lengths else 0
    num_series = len(series_list)

    # Generate a color gradient palette scaled to the exact number of series
    if num_series > 1:
        colors = pc.sample_colorscale("viridis", [i / (num_series - 1) for i in range(num_series)])
    else:
        colors = pc.sample_colorscale("viridis", [0.5])

    fig = go.Figure()

    for i, data in enumerate(series_list):
        label = labels[i] if i < len(labels) else f"Series {i + 1}"
        fig.add_trace(go.Scatter(
            x=np.arange(len(data)),
            y=data,
            mode='lines',
            name=label,
            line=dict(width=1.5, color=colors[i])
        ))

    layout_args = dict(
        title=f"{title} ({num_iterations} Iterations)",
        title_font_size=16,
        xaxis_title="Iteration Index",
        yaxis_title="Value (Log Scale)" if use_log else "Value",
        xaxis=dict(range=[-num_iterations * 0.02, num_iterations * 1.02]),
        template="plotly_white",
        height=600,
    )

    if num_series > 10:
        layout_args["legend"] = dict(x=1.02, y=1, xanchor="left", yanchor="top")

    fig.update_layout(**layout_args)

    if use_log:
        fig.update_yaxes(type="log")

    fig.show()


def plot_multiple_series_any(series_list: list, labels: list, title: str = "", use_log: bool = False):
    """
    Plots multiple data series on a single graph.
    (Migrated to Plotly to match standard suite formatting)
    """
    # Simply mapping this to the main Plotly function since the Plotly implementation
    # natively handles the dynamic spacing and color palettes requested previously.
    plot_multiple_series(series_list, labels, title, use_log)


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
    Plots a professional interactive line graph with an optional standard deviation shadow.
    """
    x_values = np.arange(num_iterations)
    y_values = np.array(data_points)

    fig = go.Figure()

    # Add Standard Deviation Shadow
    if std_dev is not None:
        std_array = np.array(std_dev)
        y_upper = y_values + std_array
        y_lower = y_values - std_array

        fig.add_trace(go.Scatter(
            x=np.concatenate([x_values, x_values[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='Std Dev'
        ))

    # Add Mean/Best Line
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='Mean/Best'
    ))

    fig.update_layout(
        title=f"{title} ({num_iterations} Iterations)",
        title_font_size=16,
        xaxis_title="Iteration Index",
        yaxis_title="Value (Log Scale)" if use_log else "Value",
        xaxis=dict(range=[-num_iterations * 0.02, num_iterations * 1.02]),
        template="plotly_white",
        height=600,
    )

    if use_log:
        fig.update_yaxes(type="log")

    fig.show()


def plot_grid_search(df, title: str = ""):
    """
    Generates an interactive Heatmap for Parameter Sensitivity using Plotly.
    """
    pivot_table = df.pivot_table(
        index='c2',
        columns='c1',
        values='fitness',
        aggfunc='mean'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='inferno_r',
        text=np.round(pivot_table.values, 2),
        texttemplate="%{text}",
        colorbar=dict(title='Mean Fitness')
    ))

    fig.update_layout(
        title=f"PSO Parameter Sensitivity: {title}",
        title_font_size=16,
        xaxis_title="Cognitive Coefficient (c1)",
        yaxis_title="Social Coefficient (c2)",
        template="plotly_white",
        height=800,
        width=900
    )

    fig.show()


def plot_walk(steps, x, y, title: str = ""):
    """
    Generates an interactive scatter plot of the particle trajectory.
    """
    fig = go.Figure()

    # Trajectory underlying dashed line
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.4,
        showlegend=False,
        name='Path'
    ))

    # Interactive scatter points
    # Dynamically scaling point sizes between 5 and 15
    max_step = max(steps) if len(steps) > 0 and max(steps) > 0 else 1
    scaled_sizes = 5 + 10 * (np.array(steps) / max_step)

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=scaled_sizes,
            color=steps,
            colorscale='viridis',
            showscale=True,
            colorbar=dict(title="Time (Steps)"),
            opacity=0.8
        ),
        text=[f"Step: {t}<br>X: {xi:.2f}<br>Y: {yi:.2f}" for t, xi, yi in zip(steps, x, y)],
        hoverinfo="text",
        name='Position'
    ))

    fig.update_layout(
        title=f"Particle Trajectory: {title}",
        title_font_size=16,
        xaxis_title="X-Coordinate",
        yaxis_title="Y-Coordinate",
        template="plotly_dark",
        height=800,
        width=900
    )

    fig.show()