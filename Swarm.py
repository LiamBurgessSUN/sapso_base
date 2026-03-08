import numpy as np
from fontTools.misc.arrayTools import vectorLength

from fitness_function.FitnessFunction import FitnessFunction


###             axis_1 ->
###  axis_0 || [ sample_1_x , sample_1_y...
###         \/   sample_2_x , sample_2_y ]
class Swarm:
    def __init__(self, number_particles: int, fitness_function: FitnessFunction, expected_iterations: int = 5000):
        self.number_particles = number_particles

        self.bounds = fitness_function.bounds
        self.ff = fitness_function

        self.expected_iterations = expected_iterations

        self.seed = 42
        np.random.seed(seed=self.seed)

        self.fitness = np.full((number_particles,), np.inf)

        self.banned_particles = np.zeros(number_particles)

        self.velocity = np.random.uniform(low=0.5, high=0.8, size=(number_particles, 30))
        self.position = np.random.uniform(low=self.bounds[0] + 0.1, high=self.bounds[1] - 0.1,
                                          size=(number_particles, 30))
        self.local_best_position = self.position.copy()
        self.global_best_position = np.random.uniform(low=self.bounds[0], high=self.bounds[1], size=(number_particles,))

        self.inertia = np.float64(0.729844)
        self.inertia_log = []

        self.local_cognitive_c1 = np.float64(1.496180)
        self.local_cognitive_c1_log = []

        self.global_cognitive_c2 = np.float64(1.496180)
        self.global_cognitive_c2_log = []

        self.converged = False

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

        self.fitness_stagnation = False

        self.prior_best = np.min(self.fitness_function())
        self.best_fitness = np.min(self.fitness_function())

        self.delta_vector = np.random.uniform(0, 1, size=30)

    def _log_control(self):
        self.local_cognitive_c1_log.append(self.local_cognitive_c1)
        self.global_cognitive_c2_log.append(self.global_cognitive_c2)
        self.inertia_log.append(self.inertia)

    def set_control_parameters(self, local_cognitive, global_cognitive, inertia):
        self.local_cognitive_c1 = np.float64(local_cognitive)
        self.global_cognitive_c2 = np.float64(global_cognitive)
        self.inertia = np.float64(inertia)
        print(
            f"Parameters Changed -- C1: {self.local_cognitive_c1}, C2: {self.global_cognitive_c2}, Inertia: {self.inertia}")

    def sample_stagnation(self):
        self.fitness_stagnation = np.abs(self.best_fitness - self.prior_best) < 0.001

    def sample_control_parameters_randomly(self):
        self.inertia = np.float64(np.random.uniform(-1, 1))
        self.local_cognitive_c1 = np.float64(np.random.uniform(0, 4))
        self.global_cognitive_c2 = np.float64(np.random.uniform(0, 4))

    def sample_control_parameters_with_time(self, time: int = 1):
        self.inertia = 0.4 * (((time - self.expected_iterations) / self.expected_iterations) ** 2) + 0.4
        self.local_cognitive_c1 = -3 * (time / self.expected_iterations) + 3.5
        self.global_cognitive_c2 = 3 * (time / self.expected_iterations) + 0.5

    def sample_diversity(self):
        centers = np.mean(self.position, axis=0)

        # axis 1 = dimension
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
        mean_step_size = np.mean(step_sizes)
        self.avg_velocity_log.append(mean_step_size)
        self.avg_velocity_std_log.append(np.std(step_sizes))

    def sample_stability(self) -> bool:
        stable = (self.local_cognitive_c1 + self.global_cognitive_c2) < (24 * (1 - self.inertia ** 2)) / (
                7 - 5 * self.inertia)
        self.stability_log.append(
            stable
        )

        return stable

    def sample_velocities_normalized(self) -> np.array:
        b_min, b_max = self.bounds

        b_range = b_max - b_min
        midpoint = (b_min + b_max) / 2.0

        return np.tanh((2.0 / b_range) * (self.velocity - midpoint))

    def step(self, iteration: int = 0) -> tuple:
        # self.sample_control_parameters_with_time(iteration)
        self.sample_diversity()

        in_bound_particles = self.sample_boundedness()

        self.sample_stagnation()

        self._log_control()
        self.sample_stability()

        costs = self.fitness_function()

        self.fitness_log.append(np.mean(costs))
        self.fitness_std_log.append(np.std(costs))

        in_bounds = (self.position >= self.bounds[0]) & (self.position <= self.bounds[1])
        particle_in_bounds = np.all(in_bounds, axis=1)
        costs[~particle_in_bounds] = np.inf

        min_cost = np.min(costs)
        min_particle = np.argmin(costs)

        if min_cost < self.best_fitness and in_bound_particles[min_particle]:
            self.prior_best = self.best_fitness
            self.best_fitness = min_cost

            self.global_best_position = self.position[min_particle].copy()

        self.best_log.append(self.best_fitness.copy())

        better_mask = costs < self.fitness

        self.fitness[better_mask] = costs[better_mask]
        self.local_best_position[better_mask] = self.position[better_mask]

        r1 = np.random.uniform(0, 1, size=self.position.shape)
        r2 = np.random.uniform(0, 1, size=self.position.shape)

        self.velocity = (self.inertia * self.velocity +
                         self.local_cognitive_c1 * r1 * (self.local_best_position - self.position) +
                         self.global_cognitive_c2 * r2 * (self.global_best_position - self.position))

        self.clamp_velocities_simple_bound()

        self.history.append(self.position.copy())

        self.position += self.velocity

        self.sample_average_velocity(self.history[-1])

        return min_cost, self.global_best_position

    def clamp_velocities_simple_bound(self):
        lower, upper = self.bounds

        v_max = self.delta_vector * (upper - lower)

        self.velocity = np.clip(self.velocity, -v_max, v_max)

    def clamp_velocities_dynamic_bound(self):
        lower, upper = self.bounds

        delta = np.random.uniform(0, 1)

        space_diagonal = np.sqrt(np.sum((upper - lower) ** 2))
        v_max = delta * space_diagonal

        current_norms = np.linalg.norm(self.velocity, axis=1, ord=2)

        mask = current_norms > v_max

        if np.any(mask):
            scaling_factors = v_max / current_norms[mask]
            self.velocity[mask] *= scaling_factors[:, np.newaxis]

    def compute_average_velocity(self) -> np.float64:
        pass

    def fitness_function(self):
        return self.ff.fitness_function(self.position)
