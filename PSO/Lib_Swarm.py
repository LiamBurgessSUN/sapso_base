import random
import numpy as np
import pyswarms.backend as ps
from pyswarms.backend.topology import Star

from fitness_function.FitnessFunction import FitnessFunction


class Swarm:
    """
    SwarmProf Validated backend wrapper for PySwarms.
    Implements the SAC-SAPSO boundary handling strategy:
    1. Unconstrained Roaming (bh=None)
    2. Valid Attractor Constraint (PBest Restriction)
    """

    def __init__(self, fitness_function: FitnessFunction, n_particles: int = 30, dimensions: int = 30, seed: int = 17):
        # SwarmProf Fix: Allow dynamic seeding for statistical robustness (50 independent runs).
        # Default is 17 for reproducibility, but can be changed by the evaluation script.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.n_particles = n_particles
        self.dimensions = dimensions
        self.ff = fitness_function

        # Initial control parameters
        self.c1 = 0.0
        self.c2 = 0.0
        self.w = 1.0
        self.options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}

        # Star topology equates to Global Best (gbest)
        self.topology = Star()

        # BUG FIX: Bounds must match `dimensions`, not `n_particles`
        min_bound = np.full(self.dimensions, fitness_function.bounds[0])
        max_bound = np.full(self.dimensions, fitness_function.bounds[1])
        self.bounds = (min_bound, max_bound)

        # STRATEGY 1: Unconstrained Roaming
        # No boundary handlers, allowing particles to leave the search space.
        self.vh = None
        self.bh = None

        # Create the swarm state dictionary via pyswarms backend
        self.swarm = ps.create_swarm(
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            bounds=self.bounds,
            options=self.options
        )

        # SWARMPROF FIX: PySwarms initializes pbest_cost and best_cost as empty arrays.
        # We initialize them to infinity so step 0 correctly registers the initial positions.
        self.swarm.pbest_cost = np.full(self.n_particles, np.inf)
        self.swarm.best_cost = np.inf

    def set_control(self, c1: float, c2: float, w: float):
        """Injects new hyperparameters for the current timestep."""
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.options = {'c1': self.c1, 'c2': self.c2, 'w': self.w}
        self.swarm.options = self.options

    def step(self, step_no: int):
        """Executes a single step of the PSO algorithm."""

        # Compute current cost
        self.swarm.current_cost = self.ff.fitness_function(self.swarm.position)

        # STRATEGY 2: Valid Attractor Constraint (PBest Restriction)
        # 1. Find particles that are strictly within bounds
        in_bounds = np.all((self.swarm.position >= self.bounds[0]) &
                           (self.swarm.position <= self.bounds[1]), axis=1)

        # 2. Find particles that improved their cost (Assuming minimization)
        better_cost = self.swarm.current_cost < self.swarm.pbest_cost

        # 3. Create a boolean mask for particles that are BOTH in bounds AND have a better cost
        update_mask = in_bounds & better_cost

        # 4. Manually update pbest positions and costs using the mask
        self.swarm.pbest_pos[update_mask] = self.swarm.position[update_mask]
        self.swarm.pbest_cost[update_mask] = self.swarm.current_cost[update_mask]

        # Update global best using the restricted pbest pool
        if np.min(self.swarm.pbest_cost) < self.swarm.best_cost:
            self.swarm.best_pos, self.swarm.best_cost = self.topology.compute_gbest(self.swarm)

        # Let's print our output
        if step_no % 20 == 0:
            print('Iteration: {} | self.swarm.best_cost: {:.4f}'.format(step_no + 1, self.swarm.best_cost))

        # Update velocity matrix
        if self.vh is not None:
            self.swarm.velocity = self.topology.compute_velocity(self.swarm, vh=self.vh, bounds=self.bounds)
        else:
            self.swarm.velocity = self.topology.compute_velocity(self.swarm, bounds=self.bounds)

        # Update position matrix (Roaming is allowed because bh is None)
        if self.bh is not None:
            self.swarm.position = self.topology.compute_position(self.swarm, bh=self.bh, bounds=self.bounds)
        else:
            self.swarm.position = self.topology.compute_position(self.swarm, bounds=self.bounds)