import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from GymWrapper import SAPSOEnv
from plots import plot_values_line_with_std


def evaluate_model(model_path="sac_sapso_policy.zip", num_particles=30, dim=30, patience=100):
    """
    Runs a deterministic simulation of the trained SAC-SAPSO policy.
    Termination is handled natively by the Swarm's internal stagnation check.
    """
    # 1. Initialize the Environment
    # We pass the stagnation patience here; it will be handled by the internal Swarm
    env = SAPSOEnv(
        num_particles=num_particles,
        dim=dim,
        max_steps=5000,
        n_t=10,
        stagnation_patience=patience
    )

    # 2. Load the trained SAC model
    print(f"📂 Loading model from {model_path}...")
    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{model_path}' not found. Please ensure training is complete.")
        return

    # 3. Reset environment
    obs, _ = env.reset()
    done = False

    print(f"🐝 Swarm simulation started (Stagnation Patience: {patience})...")

    # 4. Evaluation Loop
    while not done:
        # Use deterministic=True to bypass the entropy-based exploration used in training
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            print(f"🛑 Early termination: Swarm stagnated at iteration {env.current_step}.")

        done = terminated or truncated

    # 5. Extract logs from the internal swarm instance for visualization
    swarm = env.swarm
    print(f"\n✅ Simulation Complete!")
    print(f"🏆 Final Global Best Fitness: {swarm.best_fitness:.6e}")
    print(f"⏱️  Total Iterations: {env.current_step}")

    # 6. Visualize the performance
    print("📊 Generating plots...")

    # Plot Fitness progression (Global Best)
    plot_values_line_with_std(
        num_iterations=len(swarm.best_log),
        data_points=swarm.best_log,
        title="RL-Adapted Best Fitness"
    )

    # Plot Diversity (Exploration vs Exploitation balance)
    plot_values_line_with_std(
        num_iterations=len(swarm.diversity_log),
        data_points=swarm.diversity_log,
        std_dev=swarm.diversity_std_log,
        title="Swarm Diversity (RL-Guided)"
    )

    # Plot Velocity (Convergence behavior)
    # Note: If the plot looks like a flat line, uncomment 'plt.yscale(log)' in plots.py
    plot_values_line_with_std(
        num_iterations=len(swarm.avg_velocity_log),
        data_points=swarm.avg_velocity_log,
        std_dev=swarm.avg_velocity_std_log,
        title="Particle Velocity - Trained Policy"
    )


if __name__ == "__main__":
    evaluate_model()