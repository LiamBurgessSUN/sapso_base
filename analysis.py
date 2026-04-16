import numpy as np
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plots.plotly_plots import plot_multiple_series, plot_runs_from_dataframe
from plots.plots import plot_values_line_with_std


def scale_range(df: DataFrame, field: str):
    group_cols = ['run', 'function_name']
    func_mins = df.groupby(group_cols)[field].transform('min')
    func_maxs = df.groupby(group_cols)[field].transform('max')
    epsilon = 1e-9
    df[f'normalized_{field}'] = (df[field] - func_mins) / (func_maxs - func_mins + epsilon)


def format_by_run(
        df: pd.DataFrame,
        metric: str = "best_fitness",
):
    series_list = []
    labels = []
    all_final_values = []

    # 1. Grouping and Prep
    runs = df.groupby("run")

    for run_id, group_df in runs:
        sorted_group = group_df.sort_values("step_number")

        series_list.append(sorted_group[metric].values)
        labels.append(f"Run {run_id}")
        all_final_values.append(sorted_group[metric].iloc[-1])

    return series_list, labels, all_final_values


if __name__ == '__main__':
    df = pd.read_json("trial_results/sac_sapso_policy_nt_125.json")
    print(df.columns)
    print(df.shape)

    scale_range(df, 'best_fitness')
    scale_range(df, 'avg_velocity')
    scale_range(df, 'swarm_diversity')
    scale_range(df, 'particles_in_bounds')

    aggregate = (df.groupby(["step_number"])
    .agg(
        vel_mean=("normalized_avg_velocity", np.mean),
        vel_std=("normalized_avg_velocity", np.std),
        swarm_mean=("normalized_swarm_diversity", np.mean),
        swarm_std=("normalized_swarm_diversity", np.std),
        fitness_mean=("normalized_best_fitness", np.mean),
        fitness_std=("normalized_best_fitness", np.std),
        bounded_mean=("normalized_particles_in_bounds", np.mean),
        bounded_std=("normalized_particles_in_bounds", np.std),
        stable_mean=("stable", np.mean),
        stable_std=("stable", np.std),
        c1_mean=("c1", np.mean),
        c1_std=("c1", np.std),
        c2_mean=("c2", np.mean),
        c2_std=("c2", np.std),
        inertia_mean=("inertia", np.mean),
        inertia_std=("inertia", np.std),
    )
    )

    # 1. Average Particle Velocity (Section 4.1, Metric 5)
    plot_values_line_with_std(5000, aggregate["vel_mean"].tolist(), aggregate["vel_std"].tolist(),
                              "Average Particle Velocity", use_log=True)

    # 2. Swarm Diversity (Section 4.1, Metric 2)
    plot_values_line_with_std(5000, aggregate["swarm_mean"].tolist(), aggregate["swarm_std"].tolist(),
                              "Swarm Diversity",
                              use_log=True)

    # 3. Global Best Fitness (Section 4.1, Metric 1)
    plot_values_line_with_std(5000, aggregate["fitness_mean"].tolist(), aggregate["fitness_std"].tolist(),
                              "Global Best Fitness", use_log=True)

    # 4. Particles In Bounds (Section 4.1, Metric 3)
    plot_values_line_with_std(5000, aggregate["bounded_mean"].tolist(), aggregate["bounded_std"].tolist(),
                              "Particles In Bounds", use_log=False)

    plot_values_line_with_std(5000, aggregate["stable_mean"].tolist(), aggregate["stable_std"].tolist(), "Stability",
                              use_log=False)

    plot_multiple_series(
        [
            aggregate['c1_mean'].tolist(),
            aggregate['c2_mean'].tolist(),
            aggregate['inertia_mean'].tolist(),
        ],
        [
            aggregate['c1_std'].tolist(),
            aggregate['c2_std'].tolist(),
            aggregate['inertia_std'].tolist(),
        ],
        [
            "C1",
            "C2",
            "Inertia"
        ],
        "Control Parameters"
    )

    series_list, labels, all_final_values = format_by_run(df, metric="normalized_best_fitness")

    plot_multiple_series(
        mean_list=series_list,
        std_list=None,
        labels=labels,
        title=f"Normalized Best Fitness",
        use_log=False
    )
