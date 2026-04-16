from typing import Tuple
import pandas as pd
import numpy as np

# Assuming these are available in your local environment
from fitness_function.FitnessFunction import EVALUATION_SET
from Swarm import Swarm


def time_series_results():
    """
    Implements a static time-variant baseline for PSO parameter adaptation
    as discussed in von Eschwege & Engelbrecht (2024).
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
    # Using a list to collect arrays is more efficient than repeated np.concatenate
    scaled_list = []

    print(f"Starting evaluation on {len(EVALUATION_SET)} functions...")

    for function in EVALUATION_SET:
        # Initializing the Swarm based on the paper's experimental setup (ns=30, nx=30)
        swarm = Swarm(
            number_particles=30,
            fitness_function=function(),
            expected_iterations=5000,
            seed=17,
            clamping=False  # Baseline comparison often omits clamping initially
        )

        for i in range(5000):
            iteration = i + 1

            # Get parameters from the time-variant schedule
            w, c1, c2 = sample_control_by_time(iteration, 5000)
            swarm.set_control_parameters(c1, c2, w)

            # Perform a PSO step
            swarm.step()

            # Log metrics corresponding to Section 4.1 of the paper
            results_data.append({
                "run": -1,
                "n_t": -1,
                "step_number": i,
                "function_name": function.__name__,
                "inertia": swarm.inertia,
                "c1": swarm.local_cognitive_c1,
                "c2": swarm.global_cognitive_c2,
                "stable": swarm.sample_stability(),
                "particles_in_bounds": swarm.sample_boundedness(),
                "avg_velocity": swarm.avg_velocity_log[-1] if swarm.avg_velocity_log else 0,
                "swarm_diversity": swarm.diversity_log[-1] if swarm.diversity_log else 0,
                "best_fitness": swarm.best_fitness,
            })

        # Process fitness logs for this function
        data = np.array(swarm.best_log)
        min_val = np.min(data)
        max_val = np.max(data)

        # Avoid division by zero if fitness never changes
        range_val = max_val - min_val
        if range_val > 0:
            normalized_fitness = (data - min_val) / range_val
            scaled_list.append(normalized_fitness)
        else:
            scaled_list.append(np.zeros_like(data))

    # Correctly consolidate the scaled data
    if scaled_list:
        scaled_data = np.concatenate(scaled_list, axis=0)
        print(f"\nGlobal Statistics (Scaled):")
        print(f"Mean: {np.mean(scaled_data):.6f}")
        print(f"Std Dev: {np.std(scaled_data):.6f}")
    else:
        print("No data collected.")

    # Export results for visualization
    df = pd.DataFrame(results_data)
    output_filename = "pso_paper_results/time_based.json"

    # Ensure directory exists (basic check)
    import os
    os.makedirs("pso_paper_results", exist_ok=True)

    df.to_json(output_filename, index=False)
    print(f"\nEvaluation complete. Results saved to {output_filename}")


if __name__ == "__main__":
    time_series_results()