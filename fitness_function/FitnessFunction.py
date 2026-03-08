from typing import Tuple
import numpy as np


class FitnessFunction:
    bounds: tuple[float, float]

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        pass

    def fitness_function(self, positions: np.ndarray) -> np.ndarray:
        fitness = self._fitness_function_invoke(positions)

        assert fitness is not None
        assert not np.all(np.isnan(fitness))
        assert not np.all(np.isinf(fitness))

        return fitness


class EllipticFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-100, 100)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n_particles, n_dims = (positions.shape[0], positions.shape[1])

        coefficients = (10 ** 6) ** (np.arange(n_dims) / (n_dims - 1))

        squared_pos = positions ** 2

        fitness = np.sum(coefficients * squared_pos, axis=1)

        return fitness
