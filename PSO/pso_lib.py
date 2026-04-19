import pyswarms as ps
import numpy as np
import pandas as pd
import os
from fitness_function.FitnessFunction import TRAINING_SET


def get_poli_stability_limit(w: float) -> float:
    """Calculates the limit for c1 + c2 based on Poli's stability condition (Eq 4)."""
    return (24 * (1 - w ** 2)) / (7 - 5 * w)


def run_baselines():
    """
    Recreates the four baseline experiments from Section 5.1 of the paper:
    1. baseline_constant
    2. baseline_timevariant
    3. baseline_tvac
    4. baseline_random
    """

    num_runs = 30
    max_iterations = 5000
    n_particles = 30
    dimensions = 30
    results_data = []

    # The paper uses both sets, but we'll focus on EVALUATION_SET for speed.
    # Change to TRAINING_SET to reproduce the full Table 3.
    target_set = TRAINING_SET
    modes = ['constant', 'timevariant', 'tvac', 'random']

    print(f"Starting baseline recreation on {len(target_set)} functions...")

    for mode in modes:
        print(f"\n--- Running Experiment: baseline_{mode} ---")

        for function_class in target_set:
            func_instance = function_class()
            func_name = function_class.__name__
            print(f"Function: {func_name}")

            for run_idx in range(num_runs):
                np.random.seed(17 + run_idx)

                # Initialize particles manually for the custom loop
                # This ensures we match the paper's initialization logic
                min_b, max_b = func_instance.bounds
                pos = np.random.uniform(min_b, max_b, (n_particles, dimensions))
                vel = np.zeros((n_particles, dimensions))

                pbest_pos = pos.copy()
                pbest_fit = func_instance.fitness_function(pos)

                gbest_idx = np.argmin(pbest_fit)
                gbest_pos = pbest_pos[gbest_idx].copy()
                gbest_fit = pbest_fit[gbest_idx]

                for t in range(max_iterations):
                    # 1. Update Control Parameters (CPs) based on baseline mode
                    if mode == 'constant':
                        w, c1, c2 = 0.729844, 1.496180, 1.496180

                    elif mode == 'timevariant':
                        # Eq (3) from the paper
                        w = 0.4 * (((t - max_iterations) / max_iterations) ** 2) + 0.4
                        c1 = -3 * (t / max_iterations) + 3.5
                        c2 = 3 * (t / max_iterations) + 0.5

                    elif mode == 'tvac':
                        # Section 2.2: Fixed w, time-varying c1 and c2
                        w = 0.729844
                        c1 = (2.5 - 0.5) * ((max_iterations - t) / max_iterations) + 0.5
                        c2 = (0.5 - 2.5) * ((max_iterations - t) / max_iterations) + 2.5

                    elif mode == 'random':
                        # Section 5.1: Randomly sample stable CPs adhering to Eq (4)
                        w = np.random.uniform(0.1, 0.9)
                        limit = get_poli_stability_limit(w)
                        # Ensure c1+c2 < limit
                        c1 = np.random.uniform(0.1, limit / 2)
                        c2 = np.random.uniform(0.1, limit / 2)

                    # 2. PSO Velocity Update (Standard Inertia Weight PSO)
                    r1 = np.random.uniform(0, 1, (n_particles, dimensions))
                    r2 = np.random.uniform(0, 1, (n_particles, dimensions))

                    vel = (w * vel +
                           c1 * r1 * (pbest_pos - pos) +
                           c2 * r2 * (gbest_pos - pos))

                    # 3. Position Update
                    pos = pos + vel

                    # 4. Fitness Evaluation (Feasible space check as per Section 4.2)
                    # Note: Paper says particles in infeasible space are assigned infinite fitness
                    # and do not update best positions.
                    current_fit = func_instance.fitness_function(pos)

                    # Boundary mask: True if particle is inside bounds in all dimensions
                    in_bounds = np.all((pos >= min_b) & (pos <= max_b), axis=1)

                    for i in range(n_particles):
                        if in_bounds[i]:
                            if current_fit[i] < pbest_fit[i]:
                                pbest_fit[i] = current_fit[i]
                                pbest_pos[i] = pos[i].copy()

                                if pbest_fit[i] < gbest_fit:
                                    gbest_fit = pbest_fit[i]
                                    gbest_pos = pbest_pos[i].copy()

                results_data.append({
                    "run": run_idx,
                    "mode": mode,
                    "function_name": func_name,
                    "best_fitness": gbest_fit
                })

    # Summary and Export
    df = pd.DataFrame(results_data)

    # 5. Scale the best fitness values per function across all modes and runs
    # This matches the "Normalised global best solution quality" metric (Section 4.1)
    min_vals = df.groupby('function_name')['best_fitness'].transform('min')
    max_vals = df.groupby('function_name')['best_fitness'].transform('max')
    range_vals = max_vals - min_vals

    # Safe division to prevent NaNs if max == min
    df['scaled_fitness'] = np.where(
        range_vals > 0,
        (df['best_fitness'] - min_vals) / range_vals,
        0.0
    )

    # Compute overall average and std per mode (Matches format of Tables 3 and 6)
    overall_summary = df.groupby('mode')['scaled_fitness'].agg(
        μ='mean',
        σ='std'
    ).reset_index()

    print("\n--- Overall Scaled Baseline Results Summary ---")
    print(overall_summary.to_string(index=False))

    output_dir = "pso_paper_results"
    os.makedirs(output_dir, exist_ok=True)
    df.to_json(f"{output_dir}/baseline_recreation_30runs.json", orient="records", indent=4)
    print(f"\nResults saved to {output_dir}/baseline_recreation_30runs.json")


if __name__ == "__main__":
    run_baselines()
