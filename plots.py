import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_values_line_with_std(num_iterations: int, data_points: list, std_dev: list = None, title: str = ""):
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
        # fill_between(x, lower_bound, upper_bound)
        plt.fill_between(
            x_values,
            y_values - std_array,
            y_values + std_array,
            color='#1f77b4',
            alpha=0.2,  # Transparency of the shadow
            label='Std Dev'
        )

    # Professional Labeling & Styling
    plot.set_title(f"{title} ({num_iterations} Iterations)", fontsize=16)
    plot.set_xlabel("Iteration Index", fontsize=12)
    plot.set_ylabel("Value", fontsize=12)

    # Maintain the 'Breathing Room' on the Y-Axis
    plt.xlim(-num_iterations * 0.02, num_iterations * 1.02)

    # Optional: Log scale if values span many orders of magnitude
    # plt.yscale('log')

    plt.legend()
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
