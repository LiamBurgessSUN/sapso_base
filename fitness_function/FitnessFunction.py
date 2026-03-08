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
        n_dims = positions.shape[1]
        coefficients = (10 ** 6) ** (np.arange(n_dims) / (n_dims - 1))
        fitness = np.sum(coefficients * (positions ** 2), axis=1)
        return fitness


class Bohachevsky1Function(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-15, 15)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j = positions[:, :-1]
        x_next = positions[:, 1:]

        term = (x_j ** 2 + 2 * x_next ** 2
                - 0.3 * np.cos(3 * np.pi * x_j)
                - 0.4 * np.cos(4 * np.pi * x_next) + 0.7)

        return np.sum(term, axis=1)


class BonyadiMichalewiczFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5, 5)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        num = np.prod(positions + 1, axis=1)
        den = np.prod((positions + 1) ** 2 + 1, axis=1)
        return num / den


class BrownFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-1, 1)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j = positions[:, :-1]
        x_next = positions[:, 1:]

        term = (x_j ** 2) ** (x_next ** 2 + 1) + (x_next ** 2) ** (x_j ** 2 + 1)
        return np.sum(term, axis=1)


class CosineMixtureFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-1, 1)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return 0.1 * np.sum(np.cos(5 * np.pi * positions), axis=1) + np.sum(positions ** 2, axis=1)


class CrossLegTableFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-10, 10)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(positions, axis=1)
        prod_sin = np.prod(np.sin(positions), axis=1)

        inner = np.abs(np.exp(np.abs(100 - norm / np.pi)) * prod_sin) + 1
        return -1.0 / (inner ** 0.1)


class DeflectedCorrugatedSpringFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.alpha = 5.0
        self.K = 5.0
        self.bounds = (0, 2 * self.alpha)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n_dims = positions.shape[1]
        diff = positions - self.alpha
        sq_sum = np.sum(diff ** 2, axis=1)

        # As per Eq A8, the cosine term uses the global sum of squares
        cos_term = np.cos(self.K * np.sqrt(sq_sum))

        # Vectorized subtraction within the summation
        return 0.1 * (sq_sum - n_dims * cos_term)


class DiscussFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-100, 100)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return 1e6 * (positions[:, 0] ** 2) + np.sum(positions[:, 1:] ** 2, axis=1)


class DropWaveFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5.12, 5.12)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        sq_norm = np.sum(positions ** 2, axis=1)
        return -(1 + np.cos(12 * np.sqrt(sq_norm))) / (2 + 0.5 * sq_norm)


class EggCrateFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-5, 5)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(positions ** 2, axis=1) + 24 * np.sum(np.sin(positions) ** 2, axis=1)


class EggHolderFunction(FitnessFunction):
    def __init__(self):
        super().__init__()
        self.bounds = (-512, 512)

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j = positions[:, :-1]
        x_next = positions[:, 1:]

        term1 = -(x_next + 47) * np.sin(np.sqrt(np.abs(x_next + x_j / 2 + 47)))
        term2 = -x_j * np.sin(np.sqrt(np.abs(x_j - (x_next + 47))))

        return np.sum(term1 + term2, axis=1)