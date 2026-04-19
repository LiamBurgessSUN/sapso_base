from typing import Tuple
import pandas as pd
import numpy as np
import os

# Assuming these are available in your local environment
from fitness_function.FitnessFunction import TRAINING_SET as TARGET_FUNCTIONS
from PSO.Swarm import Swarm


def time_series_results():
    """
    Implements a static time-variant baseline for PSO parameter adaptation
    as discussed in von Eschwege & Engelbrecht (2024).
    Performs 30 independent runs per function as per Section 4.2.
    """

    def sample_control_by_time(time: int, max_steps: int = 5000) -> Tuple[float, float, float]:
        """
        Calculates w, c1, and c2 based on time-step.
        Based on Equation (3) from the foundational paper.
        """
        return (
            0.4 * (((time - max_steps) / max_steps) ** 2) + 0.4,  # w
            -3 * (time / max_steps) + 3.5,  # c1
            3 * (time / max_steps) + 0.5  # c2
        )

    results_data = []
    scaled_list = []
    num_runs = 30
    max_iterations = 5000

    print(f"Starting evaluation on {len(TARGET_FUNCTIONS)} functions with {num_runs} runs each...")

    for function in TARGET_FUNCTIONS:
        func_name = function.__name__
        print(f"Evaluating function: {func_name}")

        for run_idx in range(num_runs):
            # Using a unique seed per run for independence (methodology Section 4.2)
            # We use (run_idx + global_offset) to keep it deterministic but varied
            current_seed = 17 + run_idx

            swarm = Swarm(
                number_particles=30,
                fitness_function=function(),
                expected_iterations=max_iterations,
                seed=current_seed,
                clamping=False
            )

            for i in range(max_iterations):
                iteration = i + 1

                # Update CPs based on time-variant schedule
                w, c1, c2 = sample_control_by_time(iteration, max_iterations)
                swarm.set_control_parameters(c1, c2, w)

                swarm.step()

                # Log metrics for every step (Note: This creates a large dataset)
                # In a real production environment, you might log every 10th step to save space
                results_data.append({
                    "run": run_idx,
                    "n_t": -1,
                    "step_number": i,
                    "function_name": func_name,
                    "inertia": swarm.inertia,
                    "c1": swarm.local_cognitive_c1,
                    "c2": swarm.global_cognitive_c2,
                    "stable": swarm.sample_stability(),
                    "particles_in_bounds": swarm.sample_boundedness(),
                    "avg_velocity": swarm.avg_velocity_log[-1] if swarm.avg_velocity_log else 0,
                    "swarm_diversity": swarm.diversity_log[-1] if swarm.diversity_log else 0,
                    "best_fitness": swarm.best_fitness,
                })

            # Process fitness logs for the scaled statistics calculation
            data = np.array(swarm.best_log)
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val

            if range_val > 0:
                normalized_fitness = (data - min_val) / range_val
                scaled_list.append(normalized_fitness)
            else:
                scaled_list.append(np.zeros_like(data))

    # Consolidated Statistics across all runs and functions
    if scaled_list:
        scaled_data = np.concatenate(scaled_list, axis=0)
        print(f"\n--- Global Statistics (Scaled across {num_runs} runs) ---")
        print(f"Mean: {np.mean(scaled_data):.6f}")
        print(f"Std Dev: {np.std(scaled_data):.6f}")

    # Export results
    df = pd.DataFrame(results_data)
    output_dir = "pso_paper_results"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/time_based_30runs.json"

    # Using 'orient' to keep the JSON structure manageable
    df.to_json(output_filename, index=False)
    print(f"\nEvaluation complete. Total records: {len(df)}")
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    time_series_results()