import numpy as np
from fontTools.misc.arrayTools import vectorLength


class Swarm:
    def __init__(self, number_particles: int, bounds: np.ndarray):
        self.number_particles = number_particles

        self.best_fitness = 10
        self.fitness = np.random.default_rng().uniform(low=8, high=10, size=number_particles)

        self.x_velocity = np.random.default_rng().uniform(low=0.5, high=0.8, size=number_particles)
        self.x_position = np.random.default_rng().uniform(low=bounds[0][0], high=bounds[0][1], size=number_particles)
        self.x_local_best = np.random.default_rng().uniform(low=8, high=10, size=number_particles)
        self.x_global_best = np.random.default_rng().uniform(low=8, high=10)

        self.y_velocity = np.random.default_rng().uniform(low=0.5, high=0.8, size=number_particles)
        self.y_position = np.random.default_rng().uniform(low=bounds[1][0], high=bounds[1][1], size=number_particles)
        self.y_local_best = np.random.default_rng().uniform(low=8, high=10, size=number_particles)
        self.y_global_best = np.random.default_rng().uniform(low=8, high=10)

        self.inertia = 0.729844
        self.local_cognitive_c1 = 1.496180
        self.global_cognitive_c2 = 1.496180

        self.seed = 42
        self.bounds = bounds
        self.converged = False

        self.best_log = []
        self.diversity_log = []
        self.percentage_bound_log = []
        self.avg_velocity_log = []

        self.x_history = []
        self.y_history = []

    def sample_control_parameters_randomly(self) -> tuple:
        self.inertia = np.random.uniform(-1, 1)
        self.local_cognitive_c1 = np.random.uniform(0, 2)
        self.global_cognitive_c2 = np.random.uniform(0, 2)

        return self.inertia, self.local_cognitive_c1, self.global_cognitive_c2

    def sample_control_parameters_with_time(self, time: int = 1) -> tuple:
        self.inertia = 0.4 * (((time - self.number_particles) / self.number_particles) ** 2) + 0.4
        self.local_cognitive_c1 = -3 * (time / self.number_particles) + 3.5
        self.global_cognitive_c2 = 3 * (time / self.number_particles) + 0.5

        return self.inertia, self.local_cognitive_c1, self.global_cognitive_c2

    def sample_diversity(self):
        x_center: float = np.sum(self.x_position) / self.number_particles
        y_center: float = np.sum(self.y_position) / self.number_particles

        np.linalg.norm(self.x_position - x_center) + np.linalg.norm(self.y_position - y_center)

        self.diversity_log.append(x_center / self.number_particles)

    def sample_boundedness(self):
        in_x = (self.x_position > self.bounds[0][0]) & (self.x_position < self.bounds[0][1])
        in_y = (self.y_position > self.bounds[1][0]) & (self.y_position < self.bounds[1][1])

        mask = in_x | in_y

        self.percentage_bound_log.append(np.mean(mask) * 100)

    def sample_average_velocity(self, prior_x, prior_y):
        self.avg_velocity_log.append((np.linalg.norm(self.x_position - prior_x) + np.linalg.norm(
            self.y_position - prior_y)) / self.number_particles)

    def step(self):
        costs = self.fitness_function()

        if np.min(costs) < self.best_fitness:
            self.best_fitness = np.min(costs)

            self.x_global_best = self.x_position[np.argmin(costs)]
            self.y_global_best = self.y_position[np.argmin(costs)]

        self.best_log.append(self.best_fitness)

        for index, cost in enumerate(costs):
            if cost < self.fitness[index]:
                self.fitness[index]: float
                self.fitness[index] = cost

                self.x_local_best[index]: float
                self.x_local_best[index] = self.x_position[index]

                self.y_local_best[index]: float
                self.y_local_best[index] = self.y_position[index]

        # self.normalize_velocity(0)
        # self.normalize_velocity(1)

        self.sample_diversity()
        self.sample_boundedness()

        self.x_velocity = (self.inertia * self.x_velocity
                           + self.local_cognitive_c1 * np.random.uniform(0, 1) * (self.x_local_best - self.x_position)
                           + self.global_cognitive_c2 * np.random.uniform(0, 1) * (
                                   self.x_global_best - self.x_position))

        self.y_velocity = (self.inertia * self.y_velocity
                           + self.local_cognitive_c1 * np.random.uniform(0, 1) * (self.y_local_best - self.y_position)
                           + self.global_cognitive_c2 * np.random.uniform(0, 1) * (
                                   self.y_global_best - self.y_position))

        prior_x = self.x_position
        prior_y = self.y_position

        self.x_history.append(prior_x)
        self.y_history.append(prior_y)

        self.x_position = self.x_position + self.x_velocity
        self.y_position = self.y_position + self.y_velocity

        self.sample_average_velocity(prior_x, prior_y)

        self.converged = self.has_converged()

        return np.min(costs), self.x_global_best, self.y_global_best

    def has_converged(self) -> bool:
        return (self.local_cognitive_c1 + self.global_cognitive_c2) < (24 * (1 - self.inertia ** 2)) / (
                7 - 5 * self.inertia)

    def normalize_velocity(self, direction: int = 0):
        def apply(velocities):
            return np.tanh((2 / (self.bounds[direction][0] - self.bounds[direction][1])) * (
                    velocities - ((self.bounds[direction][0] + self.bounds[direction][1]) / 2)))

        if direction == 0:
            self.x_velocity = apply(self.x_velocity)
        else:
            self.y_velocity = apply(self.y_velocity)

    def clamp_velocities_simple_bound(self, direction: int = 0):
        def apply(velocities):
            self.bounds[direction][1]: float
            self.bounds[direction][0]: float
            velocity_max: float
            velocity_max = np.random.uniform(0, 1) * (self.bounds[direction][0] - self.bounds[direction][1])

            for index, velocity in enumerate(velocities):
                if velocity_max < velocity:
                    velocities[index] = velocity_max
                elif velocity < -velocity_max:
                    velocities[index] = -velocity_max

            return velocities

        if direction == 0:
            self.x_velocity = apply(self.x_velocity)
        else:
            self.y_velocity = apply(self.y_velocity)

    def clamp_velocities_dynamic_bound(self, direction: int = 0):
        velocity_max = np.random.uniform(0, 1) * np.linalg.norm(self.bounds)

        def apply(velocities):
            for index, velocity in enumerate(velocities):
                if velocity > velocity_max:
                    velocities[index] = (velocity_max / np.abs(velocity)) * velocity

            return velocities

        if direction == 0:
            self.x_velocity = apply(self.x_velocity)
        else:
            self.y_velocity = apply(self.y_velocity)

    def compute_average_velocity(self) -> float:
        return (1 / (self.number_particles * self.bounds.ndim)) * (
                np.sum(self.x_velocity) + np.sum(self.y_velocity.sum()))

    def fitness_function(self):
        return (1 - self.x_position) ** 2 + 100 * (self.y_position - self.x_position ** 2) ** 2
