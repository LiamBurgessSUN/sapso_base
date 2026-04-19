import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from rl_env_wrapper import ParameterLoggingCallback, SAPSOEnv

# Global Seed for Reproducibility (Section 4.2 / SwarmProf Standards)
SEED = 17


def train_sapso():
    n_t = 125
    auto = False
    # 1. Initialize custom Environment with randomized landscape cycling
    env = SAPSOEnv(
        num_particles=30,
        dim=30,
        max_steps=5000,
        n_t=n_t,
        seed=SEED,
        auto=auto,
    )

    check_env(env, warn=True)

    # 2. Network Architecture (Section 4.3)
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )

    # 3. Instantiate SAC Model
    # Note: entropy coefficient (ent_coef) 'auto' allows the agent to
    # find the best balance of exploration (Section 2.6)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        buffer_size=1_000_000,
        tau=0.005,
        gamma=1.0,
        batch_size=256,
        ent_coef='auto',
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=SEED,
        tensorboard_log="./sac_sapso_tensorboard/"
    )

    print(f"🐝 Training on 45 functions (Shuffled Seed: {SEED})...")

    # 4. Total Training Steps (2 * 10^5)
    TOTAL_TIMESTEPS = 200_000

    callback = ParameterLoggingCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=4, callback=callback)

    if auto:
        ntype = "auto"
    else:
        ntype = str(n_t)

    path = f"sac_sapso_policy_nt_{ntype}"

    model.save("policies/"+path)
    print(f"🎓 Training Complete. Model saved as policies/{path}")

    env.df.to_json(f"./train_results/{path}.json")


if __name__ == "__main__":
    train_sapso()