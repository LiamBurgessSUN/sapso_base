import numpy as np
from fitness_function.FitnessFunction import FitnessFunction


class Swarm:
    def __init__(self, number_particles: int,
                 fitness_function: FitnessFunction,
                 expected_iterations: int = 5000,
                 stagnation_threshold: float = 1e-12,
                 stagnation_patience: int = 200):
        self.number_particles = number_particles
        self.bounds = fitness_function.bounds
        self.ff = fitness_function
        self.expected_iterations = expected_iterations
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_patience = stagnation_patience
        self.patience_counter = 0

        self.seed = 42
        np.random.seed(seed=self.seed)

        self.fitness = np.full((number_particles,), np.inf)
        self.velocity = np.random.uniform(low=0.5, high=0.8, size=(number_particles, 30))
        self.position = np.random.uniform(low=self.bounds[0] + 0.1, high=self.bounds[1] - 0.1,
                                          size=(number_particles, 30))
        self.local_best_position = self.position.copy()

        self.best_fitness = np.inf
        self.prior_best = np.inf
        self.global_best_position = self.position[0].copy()

        self.inertia = 0.729844
        self.inertia_log = []
        self.local_cognitive_c1 = 1.496180
        self.local_cognitive_c1_log = []
        self.global_cognitive_c2 = 1.496180
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

        self.delta_vector = np.random.uniform(0, 1)

    def is_stagnated(self) -> bool:
        return self.patience_counter >= self.stagnation_patience

    def check_stagnation(self):
        improvement = self.prior_best - self.best_fitness
        if improvement > self.stagnation_threshold:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def set_control_parameters(self, local_cognitive, global_cognitive, inertia):
        self.local_cognitive_c1 = local_cognitive
        self.global_cognitive_c2 = global_cognitive
        self.inertia = inertia

    def sample_diversity(self):
        centers = np.mean(self.position, axis=0)
        norm = np.linalg.norm(self.position - centers, axis=1)
        self.diversity_log.append(np.mean(norm))
        self.diversity_std_log.append(np.std(norm))

    def sample_boundedness(self) -> int:
        """Counts how many particles are currently within the feasible search space."""
        in_bounds = (self.position >= self.bounds[0]) & (self.position <= self.bounds[1])
        particle_mask = np.all(in_bounds, axis=1)
        count_in_bounds = np.sum(particle_mask)
        self.percentage_bound_log.append(count_in_bounds / self.number_particles)
        return int(count_in_bounds)

    def sample_average_velocity(self, prior_position):
        step_sizes = np.linalg.norm(self.position - prior_position, axis=1, ord=2)
        self.avg_velocity_log.append(np.mean(step_sizes))
        self.avg_velocity_std_log.append(np.std(step_sizes))

    def sample_stability(self) -> bool:
        """Poli's stability condition (Section 2.3, Eq. 4)"""
        denom = (7 - 5 * self.inertia)
        if denom <= 0: return False
        return (self.local_cognitive_c1 + self.global_cognitive_c2) < (24 * (1 - self.inertia ** 2)) / denom

    def sample_velocities_normalized(self) -> np.ndarray:
        b_min, b_max = self.bounds
        b_range = b_max - b_min
        v_magnitudes = np.linalg.norm(self.velocity, axis=1)
        v_normed = (2.0 / b_range) * v_magnitudes
        return np.tanh(v_normed)

    def sample_control_values(self):
        self.inertia_log.append(self.inertia)
        self.local_cognitive_c1_log.append(self.local_cognitive_c1)
        self.global_cognitive_c2_log.append(self.global_cognitive_c2)

    def step(self, iteration: int = 0):
        self.sample_diversity()
        self.stability_log.append(self.sample_stability())
        self.sample_control_values()

        costs = self.ff.fitness_function(self.position)

        # Boundary Penalty: disqualify out-of-bounds particles
        in_bounds = (self.position >= self.bounds[0]) & (self.position <= self.bounds[1])
        particle_in_bounds = np.all(in_bounds, axis=1)
        costs[~particle_in_bounds] = np.inf

        self.fitness_log.append(np.mean(costs[particle_in_bounds]) if any(particle_in_bounds) else np.inf)

        min_cost = np.min(costs)
        if min_cost < self.best_fitness:
            self.prior_best = self.best_fitness
            self.best_fitness = min_cost
            self.global_best_position = self.position[np.argmin(costs)].copy()

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
        self.sample_average_velocity(prior_pos)

    def clamp_velocities_simple_bound(self):
        lower, upper = self.bounds
        v_max = self.delta_vector * (upper - lower)
        self.velocity = np.clip(self.velocity, -v_max, v_max)