import numpy as np
import pandas as pd
from typing import Tuple, Optional
from Swarm import Swarm
from fitness_function.FitnessFunction import EVALUATION_SET

SET_SIZE = len(EVALUATION_SET)

def sample_stable_pso_params(
        omega_range: Tuple[float, float] = (0.0, 1.0),
        c_max_bound: float = 4.0,
        seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Samples PSO control parameters (omega, c1, c2) that satisfy Poli's stability criterion.

    Ref: von Eschwege & Engelbrecht (2024), Equation (4).
    Criterion: c1 + c2 < (24 * (1 - omega**2)) / (7 - 5 * omega) AND omega in (-1, 1).

    Args:
        omega_range: The bounds for sampling inertia weight (default [0, 1]).
        c_max_bound: The maximum value to consider for c1 or c2 before checking stability.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (omega, c1, c2) satisfying the stability condition.
    """
    if seed is not None:
        np.random.seed(seed)

    while True:
        # 1. Sample omega within the specified range
        omega = np.random.uniform(omega_range[0], omega_range[1])

        # 2. Calculate the upper bound for (c1 + c2) based on Poli's condition
        # Limit omega to slightly less than 1.0 to avoid division by zero or instability at the limit
        safe_omega = min(omega, 0.999)
        stability_limit = (24 * (1 - safe_omega ** 2)) / (7 - 5 * safe_omega)

        # 3. Sample c1 and c2
        # We sample c1 and c2 independently up to a reasonable bound, then check sum
        c1 = np.random.uniform(0, c_max_bound)
        c2 = np.random.uniform(0, c_max_bound)

        # 4. Check the stability condition
        if (c1 + c2) < stability_limit:
            return float(omega), float(c1), float(c2)


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

    for run_idx in range(NUM_RUNS):
        for func_index, func_class in enumerate(EVALUATION_SET):
            # Instantiate the fitness function and get its name
            fitness_fn = func_class()
            func_name = fitness_fn.__class__.__name__

            # Initialize the swarm
            swarm = Swarm(
                number_particles=30,
                fitness_function=fitness_fn,
                seed=SEED + run_idx,
                stagnation_patience=200
            )

            # Initial stable parameter sampling
            inertia, c1, c2 = sample_stable_pso_params(
                omega_range=(0.5, 1.0),
                c_max_bound=4.0,
                seed=SEED + run_idx
            )
            swarm.set_control_parameters(inertia, c1, c2)

            for step in range(MAX_ITERATIONS):
                # Periodically re-sample control parameters if N_T is reached
                if step > 0 and (step + 1) % N_T == 0:
                    inertia, c1, c2 = sample_stable_pso_params(
                        omega_range=(0.5, 1.0),
                        c_max_bound=4.0,
                        seed=SEED + run_idx + func_index
                    )
                    print(f"Resample Intertia: {inertia}, C1: {c1}, C2: {c2}")
                    swarm.set_control_parameters(inertia, c1, c2)

                # Execute one step of PSO
                swarm.step(step)

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
    output_filename = f"pso_trial_results/pso_statistic_eval_{N_T}.json"
    df.to_json(output_filename, index=False)
    print(f"\nEvaluation complete. Results saved to {output_filename}")
