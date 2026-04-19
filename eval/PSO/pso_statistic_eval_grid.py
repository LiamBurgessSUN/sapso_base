import numpy as np
import pandas as pd
from PSO.Swarm import Swarm
from fitness_function.FitnessFunction import EVALUATION_SET

SET_SIZE = len(EVALUATION_SET)


def is_stable(omega: float, c1: float, c2: float) -> bool:
    """
    Validates if a given CP configuration is stable according to Poli.
    """
    if not (-1 < omega < 1):
        return False

    stability_limit = (24 * (1 - omega ** 2)) / (7 - 5 * omega)
    return (c1 + c2) < stability_limit


if __name__ == "__main__":
    # Simulation Configuration based on pso_statistic_eval_random.py
    MAX_ITERATIONS = 5000
    N_T = 10  # Interval for parameter re-sampling
    SEED = 17
    NUM_RUNS = 30
    results_data = []

    print(f"--- Starting Statistical Evaluation (Runs: {NUM_RUNS}, Steps: {MAX_ITERATIONS}) ---")

    w_values = np.round(np.linspace(0.2, 1.2, NUM_RUNS), 8)  # [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    c1_values = np.round(np.linspace(0.5, 3.2, NUM_RUNS), 8)  # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    c2_values = np.round(np.linspace(0.5, 3.2, NUM_RUNS), 8)  # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    for run_idx in range(NUM_RUNS):
        for func_index, func_class in enumerate(EVALUATION_SET):
            # Instantiate the fitness function and get its name
            fitness_fn = func_class()
            func_name = fitness_fn.__class__.__name__

            # Initialize the swarm
            swarm = Swarm(
                number_particles=30,
                fitness_function=fitness_fn,
                seed=SEED,
                stagnation_patience=400
            )

            # Initial stable parameter sampling
            inertia, c1, c2 = w_values[run_idx], c1_values[run_idx], c2_values[run_idx]
            print("Parameters: ", inertia, c1, c2)
            swarm.set_control_parameters(inertia, c1, c2)

            for step in range(MAX_ITERATIONS):
                # Execute one step of PSO
                swarm.step()

                # Collect metrics for analysis
                results_data.append({
                    "run": run_idx,
                    "n_t": N_T,
                    "step_number": step,
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

            print(f"Run {run_idx} | Function: {func_name} | Best Fitness: {swarm.best_fitness:.4e}")
        assert len(results_data) == (run_idx + 1) * MAX_ITERATIONS * SET_SIZE

    # Export results to JSON for further statistical analysis
    df = pd.DataFrame(results_data)
    output_filename = f"pso_trial_results/pso_statistic_eval_grid.json"
    df.to_json(output_filename, index=False)
    print(f"\nEvaluation complete. Results saved to {output_filename}")
