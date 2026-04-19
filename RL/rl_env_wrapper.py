import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
import random
import pandas as pd

from PSO.Swarm import Swarm
from fitness_function.FitnessFunction import TRAINING_SET


class ParameterLoggingCallback(BaseCallback):
    """
    Custom callback for logging chosen Control Parameters and swarm metrics to TensorBoard.
    """

    def __init__(self, verbose=0):
        super(ParameterLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.logger.record("swarm_params/inertia_w", info.get("current_w", 0.0))
        self.logger.record("swarm_params/cognitive_c1", info.get("current_c1", 0.0))
        self.logger.record("swarm_params/social_c2", info.get("current_c2", 0.0))
        self.logger.record("swarm_metrics/global_best_fitness", info.get("global_best_fitness", 0.0))
        self.logger.record("swarm_metrics/diversity", info.get("swarm_diversity", 0.0))
        self.logger.record("swarm_metrics/average_velocity", info.get("avg_velocity", 0.0))
        self.logger.record("swarm_metrics/stability", info.get("is_stable", 0.0))
        return True


class SAPSOEnv(gym.Env):
    """
    Gymnasium Environment for SAC-SAPSO.
    Implements randomized landscape switching for robust training.
    """

    def __init__(self, num_particles=30, dim=30, max_steps=5000, n_t=125,
                 stagnation_patience=100, seed=42, fitness_function_class=None, auto: bool = False):
        super(SAPSOEnv, self).__init__()
        self.n_s = num_particles
        self.dim = dim
        self.t_max = max_steps
        self.n_t = n_t
        self.stagnation_patience = stagnation_patience
        self.eval_mode_func = fitness_function_class

        self.rng = random.Random(seed)
        self.training_queue = list(TRAINING_SET)
        self.rng.shuffle(self.training_queue)
        self.current_func_idx = 0

        self.auto = auto

        if auto:
            print("Using auto-tuning")
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            print(f"Using nt = {self.n_t}")
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_particles + 3,), dtype=np.float32)

        self.swarm = None
        self.current_step = 0

        # We store records in a list for O(1) append performance.
        # Constant DataFrame reallocation with .loc is O(N^2) and extremely slow.
        self.results_data = []

    @property
    def df(self) -> pd.DataFrame:
        """Returns the collected data as a Pandas DataFrame."""
        return pd.DataFrame(self.results_data)

    def reset(self, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self.rng = random.Random(seed)
            self.rng.shuffle(self.training_queue)

        if self.eval_mode_func is not None:
            func_class = self.eval_mode_func
        else:
            func_class = self.training_queue[self.current_func_idx]
            self.current_func_idx = (self.current_func_idx + 1) % len(self.training_queue)

        self.ff = func_class()
        self.swarm = Swarm(
            number_particles=self.n_s,
            fitness_function=self.ff,
            expected_iterations=self.t_max,
            stagnation_patience=self.stagnation_patience,
            seed=seed
        )
        self.current_step = 0
        return self._get_observation(), {"function": self.ff.__class__.__name__}

    def _get_observation(self):
        v_norm = self.swarm.sample_velocities_normalized()
        stability = 1.0 if self.swarm.sample_stability() else -1.0
        infeasibility = 0.0
        if self.swarm.percentage_bound_log:
            infeasibility = (1.0 - self.swarm.percentage_bound_log[-1]) * 2.0 - 1.0
        completion = (self.current_step / self.t_max) * 2.0 - 1.0
        obs = np.concatenate([v_norm, [stability, infeasibility, completion]])
        return np.nan_to_num(obs.astype(np.float32))

    def step(self, action):
        w_scaled = action[0]
        c1_scaled = (action[1] + 1.0) * 2.0
        c2_scaled = (action[2] + 1.0) * 2.0
        if self.auto:
            self.n_t = int(25 + ((action[3] + 1.0) * (125 - 25)) / 2)
            print(f"Chosen n_t={self.n_t}")

        self.swarm.set_control_parameters(c1_scaled, c2_scaled, w_scaled)
        y_old = self.swarm.best_fitness

        for _ in range(self.n_t):
            if self.current_step < self.t_max:
                self.swarm.step(self.current_step)
                self.current_step += 1

                # Capture per-swarm-step metrics
                self.results_data.append({
                    "nt": self.n_t,
                    "step_number": self.current_step,
                    "function_name": self.ff.__class__.__name__,
                    "inertia": self.swarm.inertia,
                    "c1": self.swarm.local_cognitive_c1,
                    "c2": self.swarm.global_cognitive_c2,
                    "stable": self.swarm.sample_stability(),
                    "particles_in_bounds": self.swarm.sample_boundedness(),
                    "avg_velocity": self.swarm.avg_velocity_log[-1],
                    "swarm_diversity": self.swarm.diversity_log[-1],
                    "best_fitness": self.swarm.best_fitness,
                })

                if self.swarm.is_stagnated():
                    print("Stagnated... stopping early")
                    break

        y_new = self.swarm.best_fitness
        obs = self._get_observation()
        reward = self._calculate_reward(y_old, y_new, self.swarm.sample_stability())

        terminated = self.swarm.is_stagnated()
        truncated = self.current_step >= self.t_max

        info = {
            "global_best_fitness": self.swarm.best_fitness,
            "current_w": w_scaled,
            "current_c1": c1_scaled,
            "current_c2": c2_scaled,
            "swarm_diversity": self.swarm.diversity_log[-1] if self.swarm.diversity_log else 0.0,
            "avg_velocity": self.swarm.avg_velocity_log[-1] if self.swarm.avg_velocity_log else 0.0,
            "is_stable": 1.0 if self.swarm.sample_stability() else 0.0,
            "function_name": self.ff.__class__.__name__
        }

        return obs, float(reward), terminated, truncated, info

    def _calculate_reward(self, y_old, y_new, stable):
        scale = 1.0
        # scale = 1.0 if stable else 0.5
        if np.isinf(y_old) or np.isinf(y_new) or y_old == y_new:
            return 0.0
        if y_old > 0 > y_new:
            return 1.0 * scale
        beta = abs(y_old) + abs(y_new)
        if y_new > 0 and y_old > 0:
            return scale * 2.0 * ((y_old + beta) - (y_new + beta)) / (y_old + beta)
        elif y_new < 0 and y_old < 0:
            return scale * 2.0 * ((y_old + 2 * beta) - (y_new + 2 * beta)) / (y_old + 2 * beta)
        return 0.0
