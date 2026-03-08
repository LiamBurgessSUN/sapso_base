import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from GymWrapper import SAPSOEnv
from plots import plot_values_line_with_std


def evaluate_model(model_path="sac_sapso_policy.zip", num_particles=30, dim=30):
    # 1. Recreate the environment used during training
    # Note: We use n_t=10 to match the training setup
    env = SAPSOEnv(num_particles=num_particles, dim=dim, max_steps=5000, n_t=10)

    # 2. Load the trained SAC model
    print(f"📂 Loading model from {model_path}...")
    model = SAC.load(model_path)

    # 3. Reset environment and storage for logs
    obs, _ = env.reset()
    done = False

    print("🐝 Swarm simulation started using RL policy...")

    # We loop until the environment is truncated (reaches max_steps)
    while not done:
        # Use deterministic=True for evaluation to see the learned "best" policy
        action, _states = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if episode is finished
        done = terminated or truncated

    # 4. Extract logs from the internal swarm instance for visualization
    swarm = env.swarm
    print(f"\n✅ Simulation Complete!")
    print(f"🏆 Final Global Best Fitness: {swarm.best_fitness:.6e}")

    # 5. Visualize the performance
    print("📊 Generating plots...")

    # Plot Fitness progression
    plot_values_line_with_std(
        num_iterations=len(swarm.fitness_log),
        data_points=swarm.best_log,
        title="RL-Adapted Best Fitness"
    )

    # Plot Diversity (Exploration vs Exploitation)
    plot_values_line_with_std(
        num_iterations=len(swarm.diversity_log),
        data_points=swarm.diversity_log,
        std_dev=swarm.diversity_std_log,
        title="Swarm Diversity (RL-Guided)"
    )

    # Plot Velocity (Convergence behavior)
    plot_values_line_with_std(
        num_iterations=len(swarm.avg_velocity_log),
        data_points=swarm.avg_velocity_log,
        std_dev=swarm.avg_velocity_std_log,
        title="Particle Velocity (Log Scale) - Trained Policy"
    )


if __name__ == "__main__":
    evaluate_model()