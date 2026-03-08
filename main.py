import itertools
import numpy as np
import pandas as pd
from typing import Type

from fitness_function.FitnessFunction import (
    FitnessFunction,
    EllipticFunction,
    Bohachevsky1Function,
    BonyadiMichalewiczFunction,
    BrownFunction,
    CosineMixtureFunction,
    CrossLegTableFunction,
    DeflectedCorrugatedSpringFunction,
    DiscussFunction,
    DropWaveFunction,
    EggCrateFunction,
    EggHolderFunction
)

from plots import plot_values_line_with_std
from Swarm import Swarm

# ======================================================================
# CONFIGURATION BLOCK - THE RESEARCHER'S DASHBOARD
# ======================================================================
# 1. Select Fitness Function from the Paper
# Options: EllipticFunction, Bohachevsky1Function, EggHolderFunction, etc.
FUNCTION_CLASS: Type[FitnessFunction] = EllipticFunction

# 2. Select Baseline Mode (Ref: Paper Section 5.1)
# 'constant'      -> Fixed w, c1, c2 (Clerc & Kennedy standard)
# 'time-variant'  -> Parameters evolve over time (Eq. 3)
# 'random'        -> Random sampling adhering to Poli's condition
MODE = 'constant'

# 3. Swarm Parameters
MAX_ITERATIONS = 5000
STAGNATION_PATIENCE = 150  # Steps to wait for delta < 1e-12
NUM_PARTICLES = 30


# ======================================================================

def run_pso_baseline():
    """
    Runs the swarm using the latest adaptive features and stagnation logic.
    This serves as the primary comparison point for your SAC-SAPSO model.
    """
    ff = FUNCTION_CLASS()

    print(f"🚀 Initializing Swarm Intelligence: {ff.__class__.__name__}")
    print(f"🛠️  Mode: {MODE.upper()} | Particles: {NUM_PARTICLES} | Max Iter: {MAX_ITERATIONS}")

    # Standard "Constant PSO" parameters (Eq. 16 Baseline)
    w_const, c1_const, c2_const = 0.729844, 1.496180, 1.496180

    swarm = Swarm(
        number_particles=NUM_PARTICLES,
        fitness_function=ff,
        expected_iterations=MAX_ITERATIONS,
        stagnation_patience=STAGNATION_PATIENCE
    )

    # Initialize with constant values; will be updated in loop if mode isn't 'constant'
    swarm.set_control_parameters(c1_const, c2_const, w_const)

    # Main Search Loop
    actual_iterations = 0
    for i in range(MAX_ITERATIONS):
        actual_iterations = i + 1

        # --- Handle Adaptive Baselines (New Item) ---
        if MODE == 'time-variant':
            swarm.sample_control_parameters_with_time(i)
        elif MODE == 'random':
            swarm.sample_control_parameters_randomly()

        # Execute physics step
        swarm.step(i)

        # Periodic Progress Report
        if i % 1000 == 0:
            stability_status = "STABLE" if swarm.stability_log[-1] else "DIVERGENT"
            print(f"Iteration {i:4d}: Best Fitness = {swarm.best_fitness:.4e} | {stability_status}")

        # --- Early Termination (New Item) ---
        if swarm.is_stagnated():
            print(f"🛑 Early Termination: Stagnation patience ({STAGNATION_PATIENCE}) exhausted at iteration {i}.")
            break

    print(f"\n✅ Simulation Complete!")
    print(f"🏆 Final Global Best: {swarm.best_fitness:.6e}")
    print(f"⏱️  Actual Iterations: {actual_iterations}")

    # Comprehensive Post-Search Analytics
    print("\n📊 Generating Diagnostic Analytics...")

    # 1. Best Fitness Log (Convergence Curve)
    plot_values_line_with_std(
        len(swarm.best_log),
        swarm.best_log,
        title=f"Convergence: {ff.__class__.__name__}"
    )

    # 2. Velocity Log (Detecting explosion/equilibrium - Ref: Figure 5/10)
    plot_values_line_with_std(
        len(swarm.avg_velocity_log),
        swarm.avg_velocity_log,
        swarm.avg_velocity_std_log,
        f"Mean Particle Velocity ({MODE})"
    )

    # 3. Diversity Log (Exploration vs Exploitation - Ref: Figure 6/11)
    plot_values_line_with_std(
        len(swarm.diversity_log),
        swarm.diversity_log,
        swarm.diversity_std_log,
        "Swarm Diversity (Spatial Spread)"
    )

    # 4. Search Space Adherence (Percentage of particles in bounds)
    plot_values_line_with_std(
        len(swarm.percentage_bound_log),
        swarm.percentage_bound_log,
        swarm.percentage_bound_std_log,
        "Feasible Search Bound Adherence"
    )


if __name__ == "__main__":
    # Ensure you are using the latest Swarm.py with is_stagnated() support!
    run_pso_baseline()