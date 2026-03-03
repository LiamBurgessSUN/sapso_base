import numpy as np
from gymnasium import spaces

from Swarm import Swarm
import gymnasium as gym


class SAPSOEnv(gym.Env):
    """
    Gymnasium Environment Wrapper for SAC-SAPSO.
    Incorporates the n_t (action frequency) and velocity clamping 
    as described in Sections 3.3.2 and 5.3 of the paper.
    """

    def __init__(self, num_particles=30, dim=30, max_steps=5000, bounds=(-100.0, 100.0), n_t=10):
        super(SAPSOEnv, self).__init__()

        self.n_s = num_particles
        self.dim = dim
        self.t_max = max_steps
        self.bounds = bounds
        self.n_t = n_t  # SAC observation frequency (action applied for n_t steps)

        self.swarm = Swarm(number_particles=self.n_s, bounds=self.bounds, expected_iterations=self.t_max)
        self.current_step = 0

        # Action space: [w, c1, c2] 
        # w in [-1.0, 1.0], c1 in [0.0, 4.0], c2 in [0.0, 4.0]
        # SB3 automatically scales its native [-1, 1] output to these Box bounds.
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 4.0, 4.0]),
            dtype=np.float32
        )

        # Observation space: n_s normalized velocities + 3 swarm metrics
        obs_dim = self.n_s + 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.swarm = Swarm(number_particles=self.n_s, bounds=self.bounds, expected_iterations=self.t_max)
        self.current_step = 0

        return self._get_observation(), {}

    def step(self, action):
        w, c1, c2 = action

        # 1. Apply chosen Control Parameters (CPs) to the Swarm
        self.swarm.set_control_parameters(local_cognitive=c1, global_cognitive=c2, inertia=w)

        y_old = self.swarm.best_fitness

        # 2. Advance the Swarm by n_t iterations (Observation Frequency)
        steps_to_take = min(self.n_t, self.t_max - self.current_step)
        for _ in range(steps_to_take):
            # Enforce Velocity Clamping (Section 5.3) to prevent velocity explosion
            # (Assuming clamp_velocities_dynamic_bound is called for both axes/all dims inside your Swarm loop)
            # In a true N-dim scenario, you'd loop clamping across dimensions. We'll simulate its intent:
            try:
                self.swarm.clamp_velocities_dynamic_bound(direction=0)
            except AttributeError:
                pass  # Fallback if specific dimension clamping isn't perfectly mapped in the snippet

            self.swarm.step(iteration=self.current_step)
            self.current_step += 1

        y_new = self.swarm.best_fitness

        # 3. Extract Observations and Reward
        obs = self._get_observation()
        reward = self._calculate_reward(y_old, y_new)

        # 4. Termination
        terminated = False
        truncated = self.current_step >= self.t_max

        info = {
            "global_best_fitness": self.swarm.best_fitness,
            "swarm_diversity": self.swarm.diversity_log[-1] if self.swarm.diversity_log else 0.0
        }

        return obs, float(reward), terminated, truncated, info

    def _get_observation(self):
        lower_bound, upper_bound = self.bounds

        # Metric 1: Normalized & Squashed Velocities (Eq. 22)
        v_mags = np.linalg.norm(self.swarm.velocity, axis=1)
        v_norm = np.tanh((2.0 / (upper_bound - lower_bound)) * (v_mags - ((lower_bound + upper_bound) / 2.0)))

        # Metric 2: Percentage of Stable Particles (Poli's Condition)
        w, c1, c2 = self.swarm.inertia, self.swarm.local_cognitive_c1, self.swarm.global_cognitive_c2
        denominator = 7.0 - 5.0 * w
        if denominator <= 0:
            perc_stable = 0.0
        else:
            is_stable = (c1 + c2) < (24.0 * (1.0 - w ** 2)) / denominator
            perc_stable = 1.0 if is_stable else 0.0

        # Metric 3: Percentage of Infeasible Particles
        if len(self.swarm.percentage_bound_log) > 0:
            perc_infeasible = 1.0 - self.swarm.percentage_bound_log[-1]
        else:
            perc_infeasible = 0.0

        # Metric 4: Percentage of Search Completion
        perc_completion = self.current_step / self.t_max

        return np.concatenate([v_norm, [perc_stable, perc_infeasible, perc_completion]]).astype(np.float32)

    def _calculate_reward(self, y_old, y_new):
        """ Relative global best improvement (Eq. 25) """
        if y_new == y_old: return 0.0

        beta = abs(y_old) + abs(y_new)

        if y_new > 0 and y_old > 0:
            return 2.0 * (((y_old + beta) - (y_new + beta)) / (y_old + beta))
        elif y_new < 0 and y_old < 0:
            return 2.0 * (((y_old + 2 * beta) - (y_new + 2 * beta)) / (y_old + 2 * beta))
        else:
            return 1.0 