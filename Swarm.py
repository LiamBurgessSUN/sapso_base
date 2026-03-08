import numpy as np
from fontTools.misc.arrayTools import vectorLength

from fitness_function.FitnessFunction import FitnessFunction


###             axis_1 ->
###  axis_0 || [ sample_1_x , sample_1_y...
###         \/   sample_2_x , sample_2_y ]
class Swarm:
    def __init__(self, number_particles: int,
                 fitness_function: FitnessFunction,
                 expected_iterations: int = 5000,
                 stagnation_threshold: float = 1e-12,
                 stagnation_patience: int = 100):
        self.number_particles = number_particles

        self.bounds = fitness_function.bounds
        self.ff = fitness_function

        self.expected_iterations = expected_iterations

        # Stagnation Parameters
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_patience = stagnation_patience
        self.patience_counter = 0

        self.seed = 42
        np.random.seed(seed=self.seed)

        self.fitness = np.full((number_particles,), np.inf)
        self.banned_particles = np.zeros(number_particles)

        self.velocity = np.random.uniform(low=0.5, high=0.8, size=(number_particles, 30))
        self.position = np.random.uniform(low=self.bounds[0] + 0.1, high=self.bounds[1] - 0.1,
                                          size=(number_particles, 30))
        self.local_best_position = self.position.copy()

        # Initialize Best Fitness tracking
        self.best_fitness = np.inf
        self.prior_best = np.inf
        self.global_best_position = self.position[0].copy()

        self.inertia = np.float64(0.729844)
        self.inertia_log = []
        self.local_cognitive_c1 = np.float64(1.496180)
        self.local_cognitive_c1_log = []
        self.global_cognitive_c2 = np.float64(1.496180)
        self.global_cognitive_c2_log = []

        self.best_log = []
        self.fitness_log = []
        self.fitness_std_log = []
        self.diversity_log = []
        self.diversity_std_log = []
        self.percentage_bound_log = []
        self.percentage_bound_std_log = []
        self.avg_velocity_log = []
        self.avg_velocity_std_log = []
        self.stability_log = []
        self.history = []

        self.delta_vector = np.random.uniform(0, 1, size=30)

    def is_stagnated(self) -> bool:
        """Returns True if the swarm has failed to improve significantly for a set period."""
        return self.patience_counter >= self.stagnation_patience

    def check_stagnation(self):
        """Internal logic to update the patience counter based on fitness delta."""
        improvement = self.prior_best - self.best_fitness

        if improvement > self.stagnation_threshold:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def set_control_parameters(self, local_cognitive, global_cognitive, inertia):
        self.local_cognitive_c1 = np.float64(local_cognitive)
        self.global_cognitive_c2 = np.float64(global_cognitive)
        self.inertia = np.float64(inertia)

    def sample_diversity(self):
        centers = np.mean(self.position, axis=0)
        norm = np.linalg.norm(self.position - centers, axis=1)
        self.diversity_log.append(np.mean(norm))
        self.diversity_std_log.append(np.std(norm))

    def sample_boundedness(self) -> np.ndarray:
        in_bounds = (self.position > self.bounds[0]) & (self.position < self.bounds[1])
        particle_mask = np.all(in_bounds, axis=1)
        self.percentage_bound_log.append(np.mean(particle_mask))
        self.percentage_bound_std_log.append(np.std(particle_mask))
        return particle_mask

    def sample_average_velocity(self, prior_position):
        step_sizes = np.linalg.norm(self.position - prior_position, axis=1, ord=2)
        self.avg_velocity_log.append(np.mean(step_sizes))
        self.avg_velocity_std_log.append(np.std(step_sizes))

    def sample_stability(self) -> bool:
        stable = (self.local_cognitive_c1 + self.global_cognitive_c2) < (24 * (1 - self.inertia ** 2)) / (
                7 - 5 * self.inertia)
        self.stability_log.append(stable)
        return stable

    def sample_velocities_normalized(self) -> np.array:
        b_min, b_max = self.bounds
        b_range = b_max - b_min
        v_magnitudes = np.linalg.norm(self.velocity, axis=1)
        v_normed = (2.0 / b_range) * v_magnitudes
        return np.tanh(v_normed)

    def sample_control_parameters_randomly(self):
        """Baseline Mode: Randomly samples parameters adhering to Poli's stability (Eq. 4)."""
        while True:
            w = np.random.uniform(-1, 1)
            c1 = np.random.uniform(0, 4)
            c2 = np.random.uniform(0, 4)
            denom = (7 - 5 * w)
            if denom > 0 and (c1 + c2) < (24 * (1 - w ** 2)) / denom:
                self.set_control_parameters(c1, c2, w)
                break

    def sample_control_parameters_with_time(self, time: int):
        """Baseline Mode: Time-variant adaptation following Equation 3."""
        t_max = self.expected_iterations
        self.inertia = 0.4 * (((time - t_max) / t_max) ** 2) + 0.4
        self.local_cognitive_c1 = -3 * (time / t_max) + 3.5
        self.global_cognitive_c2 = 3 * (time / t_max) + 0.5

    def step(self, iteration: int = 0) -> tuple:
        self.sample_diversity()
        in_bound_particles = self.sample_boundedness()
        self.sample_stability()

        costs = self.fitness_function()

        # Boundary Penalty
        in_bounds_mask = (self.position >= self.bounds[0]) & (self.position <= self.bounds[1])
        particle_in_bounds = np.all(in_bounds_mask, axis=1)
        costs[~particle_in_bounds] = np.inf

        self.fitness_log.append(np.mean(costs[particle_in_bounds]) if any(particle_in_bounds) else np.inf)
        self.fitness_std_log.append(np.std(costs[particle_in_bounds]) if any(particle_in_bounds) else 0.0)

        min_cost = np.min(costs)
        min_particle = np.argmin(costs)

        if min_cost < self.best_fitness:
            self.prior_best = self.best_fitness
            self.best_fitness = min_cost
            self.global_best_position = self.position[min_particle].copy()

        # Update stagnation counter
        self.check_stagnation()

        self.best_log.append(self.best_fitness)

        better_mask = costs < self.fitness
        self.fitness[better_mask] = costs[better_mask]
        self.local_best_position[better_mask] = self.position[better_mask]

        r1 = np.random.uniform(0, 1, size=self.position.shape)
        r2 = np.random.uniform(0, 1, size=self.position.shape)

        self.velocity = (self.inertia * self.velocity +
                         self.local_cognitive_c1 * r1 * (self.local_best_position - self.position) +
                         self.global_cognitive_c2 * r2 * (self.global_best_position - self.position))

        self.clamp_velocities_simple_bound()

        prior_pos = self.position.copy()
        self.position += self.velocity
        self.history.append(self.position.copy())

        self.sample_average_velocity(prior_pos)

        return min_cost, self.global_best_position

    def clamp_velocities_simple_bound(self):
        lower, upper = self.bounds
        v_max = self.delta_vector * (upper - lower)
        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def fitness_function(self):
        return self.ff.fitness_function(self.position)
