import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

from Swarm import Swarm
import gymnasium as gym

from fitness_function.FitnessFunction import EllipticFunction


class ParameterLoggingCallback(BaseCallback):
    """
    Custom callback for logging the SAC agent's chosen Control Parameters
    (w, c1, c2) and the swarm's metrics (fitness, diversity, velocity, stability) to TensorBoard.
    """

    def __init__(self, verbose=0):
        super(ParameterLoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Access the first environment's info dict (SB3 uses vectorized envs by default)
        info = self.locals["infos"][0]

        # Log Control Parameters (SAC Actions)
        self.logger.record("swarm_params/inertia_w", info.get("current_w", 0.0))
        self.logger.record("swarm_params/cognitive_c1", info.get("current_c1", 0.0))
        self.logger.record("swarm_params/social_c2", info.get("current_c2", 0.0))

        # Log Swarm Performance Metrics
        self.logger.record("swarm_metrics/global_best_fitness", info.get("global_best_fitness", 0.0))
        self.logger.record("swarm_metrics/diversity", info.get("swarm_diversity", 0.0))

        # Log Average Velocity to TensorBoard
        self.logger.record("swarm_metrics/average_velocity", info.get("avg_velocity", 0.0))

        # New: Log Swarm Stability (Poli's Condition)
        self.logger.record("swarm_metrics/stability", info.get("is_stable", 0.0))

        return True


class SAPSOEnv(gym.Env):
    """
    Gymnasium Environment Wrapper for SAC-SAPSO.
    """

    def __init__(self, num_particles=30, dim=30, max_steps=5000, n_t=10):
        super(SAPSOEnv, self).__init__()
        self.n_s = num_particles
        self.dim = dim
        self.t_max = max_steps
        self.n_t = n_t
        self.ff = EllipticFunction()

        # Action: [w, c1, c2] in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: squashed velocities + global metrics
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_particles + 3,), dtype=np.float32)

        self.swarm = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.swarm = Swarm(number_particles=self.n_s, fitness_function=self.ff, expected_iterations=self.t_max)
        self.current_step = 0
        return self._get_observation(), {}

    def _get_observation(self):
        v_norm = self.swarm.sample_velocities_normalized()

        # Metric 1: Stability (Poli's Condition)
        stability = 1.0 if self.swarm.sample_stability() else -1.0

        # Metric 2: Percentage of Infeasible Particles
        infeasibility = 0.0
        if self.swarm.percentage_bound_log:
            infeasibility = (1.0 - self.swarm.percentage_bound_log[-1]) * 2.0 - 1.0

        # Metric 3: Search Completion
        completion = (self.current_step / self.t_max) * 2.0 - 1.0

        obs = np.concatenate([v_norm, [stability, infeasibility, completion]])
        return np.nan_to_num(obs.astype(np.float32))

    def step(self, action):
        # Map SAC actions to PSO ranges
        w_scaled = action[0]
        c1_scaled = (action[1] + 1.0) * 2.0
        c2_scaled = (action[2] + 1.0) * 2.0

        self.swarm.set_control_parameters(c1_scaled, c2_scaled, w_scaled)
        y_old = self.swarm.best_fitness

        # Simulation Loop (n_t frequency)
        for _ in range(self.n_t):
            if self.current_step < self.t_max:
                self.swarm.step(self.current_step)
                self.current_step += 1

        y_new = self.swarm.best_fitness
        obs = self._get_observation()
        reward = self._calculate_reward(y_old, y_new)

        terminated = False
        truncated = self.current_step >= self.t_max

        # info dictionary passed to Callback
        info = {
            "global_best_fitness": self.swarm.best_fitness,
            "current_w": w_scaled,
            "current_c1": c1_scaled,
            "current_c2": c2_scaled,
            "swarm_diversity": self.swarm.diversity_log[-1] if self.swarm.diversity_log else 0.0,
            "avg_velocity": self.swarm.avg_velocity_log[-1] if self.swarm.avg_velocity_log else 0.0,
            "is_stable": 1.0 if self.swarm.sample_stability() else 0.0
        }

        return obs, float(reward), terminated, truncated, info

    def _calculate_reward(self, y_old, y_new):
        if np.isinf(y_old) or np.isinf(y_new) or y_old == y_new:
            return 0.0
        if y_old > 0 > y_new: return 1.0
        beta = abs(y_old) + abs(y_new)
        if y_new > 0 and y_old > 0:
            return 2.0 * ((y_old + beta) - (y_new + beta)) / (y_old + beta)
        elif y_new < 0 and y_old < 0:
            return 2.0 * ((y_old + 2 * beta) - (y_new + 2 * beta)) / (y_old + 2 * beta)
        return 0.0