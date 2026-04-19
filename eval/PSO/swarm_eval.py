import numpy as np
import pandas as pd
from typing import Type

from fitness_function.FitnessFunction import (
    FitnessFunction,
    EVALUATION_SET
)

from Swarm import Swarm
from results.plots import plot_swarm

# ======================================================================
# CONFIGURATION BLOCK - BASELINE BENCHMARKING
# ======================================================================
# 1. Select Baseline Mode (Ref: Paper Section 5.1)
# 'constant'      -> Fixed w, c1, c2 (Clerc & Kennedy standard)
# 'time-variant'  -> Parameters evolve over time (Eq. 3)
# 'random'        -> Random sampling adhering to Poli's condition
MODE = 'time-variant'

# 2. Search Parameters
MAX_ITERATIONS = 5000
STAGNATION_PATIENCE = 150
NUM_PARTICLES = 30


# ======================================================================

def run_single_baseline(func_class: Type[FitnessFunction], mode: str):
    """
    Executes a single baseline run for a specific function class and mode.
    """
    ff = func_class()

    # Standard "Constant PSO" parameters (Eq. 16 Baseline)
    w_const, c1_const, c2_const = 0.729844, 1.496180, 1.496180

    swarm = Swarm(
        number_particles=NUM_PARTICLES,
        fitness_function=ff,
        expected_iterations=MAX_ITERATIONS,
        stagnation_patience=STAGNATION_PATIENCE
    )

    # Initialize with constant values
    swarm.set_control_parameters(c1_const, c2_const, w_const)

    actual_iterations = 0
    for i in range(MAX_ITERATIONS):
        actual_iterations = i + 1

        if mode == 'time-variant':
            swarm.sample_control_parameters_with_time(i)
        elif mode == 'random':
            swarm.sample_control_parameters_randomly()

        # Execute physics step
        swarm.step(i)

        if swarm.is_stagnated():
            break

    return swarm, actual_iterations


def benchmark_baselines():
    """
    Iterates through the Evaluation Set using the selected baseline mode.
    Generates a summary table for direct comparison with SAC-SAPSO results.
    """
    print(f"🚀 Benchmarking Baseline PSO")
    print(f"🛠️  Mode: {MODE.upper()} | Particles: {NUM_PARTICLES} | Max Iter: {MAX_ITERATIONS}")
    print("-" * 60)

    results = []

    fitness_log = []
    for func_class in EVALUATION_SET:
        func_name = func_class().__class__.__name__
        print(f"Testing: {func_name}...", end=" ", flush=True)

        swarm, steps = run_single_baseline(func_class, MODE)

        results.append({
            "Function": func_name,
            "Best Fitness": f"{swarm.best_fitness:.4e}",
            "Iterations": steps,
            "Stagnated": "Yes" if swarm.is_stagnated() else "No"
        })
        print(f"Done. (Best: {swarm.best_fitness:.2e})")

        # Optional: Generate plots for each test function
        plot_swarm(swarm, func_name)

        fitness_log.append(swarm.best_fitness)

    # Display Summary Table (Mirroring RL rl_eval.py output)
    df = pd.DataFrame(results)
    print(f"\n📊 BASELINE SUMMARY TABLE ({MODE.upper()})")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    print(f"Mean Fitness: {np.mean(fitness_log):.4e}")
    print(f"Std Fitness: {np.std(fitness_log):.4e}")


if __name__ == "__main__":
    benchmark_baselines()