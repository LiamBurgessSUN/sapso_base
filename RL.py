# ======================================================================
# SB3 Training Setup (Strict matching to Section 4.3 & Table 2)
# ======================================================================
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from GymWrapper import SAPSOEnv

if __name__ == "__main__":
    # 1. Initialize custom Environment
    env = SAPSOEnv(num_particles=30, dim=30, max_steps=5000, bounds=(-100.0, 100.0), n_t=10)

    # Optional: Verify the environment complies with Gymnasium specs
    check_env(env, warn=True)

    # 2. Define Network Architecture (Section 4.3: Actor layer size = 256, Critic layer size = 256)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    # 3. Instantiate Soft Actor-Critic (SAC) Model
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=0.0001,  # Table 2: Learning rate a
        buffer_size=1_000_000,  # Table 2: Replay buffer size
        tau=0.005,  # Table 2: Target smoothing coefficient T
        gamma=1.0,  # Table 2: Discount factor gamma
        batch_size=256,  # Default robust batch size
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42
    )

    print("🐝 Starting SAC-SAPSO Training... Hold on to your particles!")

    # 4. Train the Model (Table 2: Training steps = 2 * 10^5)
    TOTAL_TIMESTEPS = 200_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4)

    # 5. Save the optimal policy
    model.save("sac_sapso_policy")
    print("🎓 Training Complete. Model saved as 'sac_sapso_policy.zip'")