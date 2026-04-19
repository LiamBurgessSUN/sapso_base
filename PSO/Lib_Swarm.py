import random

import numpy as np
import pyswarms.backend as ps
from pyswarms.backend.topology import Star
from pyswarms.backend.handlers import VelocityHandler, BoundaryHandler

from fitness_function.FitnessFunction import FitnessFunction


class Swarm:
    random.seed(17)
    np.random.seed(17)

    def __init__(self, fitness_function: FitnessFunction):
        self.n_particles = 30
        self.bounds = fitness_function.bounds
        self.ff = fitness_function
        self.c1 = 0
        self.c2 = 0
        self.w = 1

        self.options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}

        self.topology = Star()

        self.bounds = (np.full(self.n_particles, self.bounds[0]), np.full(self.n_particles, self.bounds[1]))

        self.vh = None
        # self.vh = VelocityHandler(
        #     strategy='zero'
        # )

        # self.bh = BoundaryHandler(
        #     strategy='shrink'
        # )

        self.bh = None

        self.swarm = ps.create_swarm(
            n_particles=self.n_particles,
            bounds=self.bounds,
            dimensions=30,
            options=self.options
        )

    def set_control(self, c1, c2, w):
        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}
        self.swarm.options = self.options

    def step(self, step_no: int):
        self.swarm.current_cost = self.ff.fitness_function(self.swarm.position)  # Compute current cost
        self.swarm.pbest_cost = self.ff.fitness_function(self.swarm.pbest_pos)  # Compute personal best pos
        self.swarm.pbest_pos, self.swarm.pbest_cost = ps.compute_pbest(self.swarm)  # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        if np.min(self.swarm.pbest_cost) < self.swarm.best_cost:
            self.swarm.best_pos, self.swarm.best_cost = self.topology.compute_gbest(self.swarm)

        # Let's print our output
        if step_no % 20 == 0:
            print('Iteration: {} | self.swarm.best_cost: {:.4f}'.format(step_no + 1, self.swarm.best_cost))

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        if self.vh is not None:
            self.swarm.velocity = self.topology.compute_velocity(self.swarm, vh=self.vh, bounds=self.bounds)
        else:
            self.swarm.velocity = self.topology.compute_velocity(self.swarm, bounds=self.bounds)

        if self.bh is not None:
            self.swarm.position = self.topology.compute_position(self.swarm, bh=self.bh, bounds=self.bounds)
        else:
            self.swarm.position = self.topology.compute_position(self.swarm, bounds=self.bounds)
