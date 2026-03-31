from typing import List, Type
import numpy as np

class FitnessFunction:
    """Base class for all optimization benchmarks as defined in Appendix A."""
    bounds: tuple[float, float]

    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fitness_function(self, positions: np.ndarray) -> np.ndarray:
        return self._fitness_function_invoke(positions)

# ======================================================================
# TRAINING SET (45 Functions)
# ======================================================================

class Ackley1Function(FitnessFunction):
    def __init__(self): self.bounds = (-32, 32)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        sum1 = np.sum(positions**2, axis=1)
        sum2 = np.sum(np.cos(2 * np.pi * positions), axis=1)
        return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e

class Alpine1Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(positions * np.sin(positions) + 0.1 * positions), axis=1)

class Bohachevsky1Function(FitnessFunction):
    def __init__(self): self.bounds = (-15, 15)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        return np.sum(x_j**2 + 2*x_next**2 - 0.3*np.cos(3*np.pi*x_j) - 0.4*np.cos(4*np.pi*x_next) + 0.7, axis=1)

class BonyadiMichalewiczFunction(FitnessFunction):
    def __init__(self): self.bounds = (-5, 5)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.prod(positions + 1, axis=1) / np.prod((positions + 1)**2 + 1, axis=1)

class BrownFunction(FitnessFunction):
    def __init__(self): self.bounds = (-1, 1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        return np.sum((x_j**2)**(x_next**2 + 1) + (x_next**2)**(x_j**2 + 1), axis=1)

class CosineMixtureFunction(FitnessFunction):
    def __init__(self): self.bounds = (-1, 1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return 0.1 * np.sum(np.cos(5 * np.pi * positions), axis=1) + np.sum(positions**2, axis=1)

class DeflectedCorrugatedSpringFunction(FitnessFunction):
    def __init__(self):
        self.alpha, self.K = 5.0, 5.0
        self.bounds = (0, 2 * self.alpha)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n, diff = positions.shape[1], positions - self.alpha
        sq_sum = np.sum(diff**2, axis=1)
        return 0.1 * (sq_sum - n * np.cos(self.K * np.sqrt(sq_sum)))

class DiscussFunction(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return 1e6 * (positions[:, 0]**2) + np.sum(positions[:, 1:]**2, axis=1)

class DropWaveFunction(FitnessFunction):
    def __init__(self): self.bounds = (-5.12, 5.12)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        sq_norm = np.sum(positions**2, axis=1)
        return -(1 + np.cos(12 * np.sqrt(sq_norm))) / (2 + 0.5 * sq_norm)

class EggCrateFunction(FitnessFunction):
    def __init__(self): self.bounds = (-5, 5)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(positions**2, axis=1) + 24 * np.sum(np.sin(positions)**2, axis=1)

class EggHolderFunction(FitnessFunction):
    def __init__(self): self.bounds = (-512, 512)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        return np.sum(-(x_next + 47) * np.sin(np.sqrt(np.abs(x_next + x_j/2 + 47))) - x_j * np.sin(np.sqrt(np.abs(x_j - (x_next + 47)))), axis=1)

class EllipticFunction(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        coeffs = (10**6)**(np.arange(n) / (n - 1))
        return np.sum(coeffs * (positions**2), axis=1)

class ExponentialFunction(FitnessFunction):
    def __init__(self): self.bounds = (-1, 1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return -np.exp(-0.5 * np.sum(positions**2, axis=1))

class GiuntaFunction(FitnessFunction):
    def __init__(self): self.bounds = (-1, 1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        arg = (16/15) * positions - 1
        return 0.6 + np.sum(np.sin(arg) + np.sin(arg)**2 + (1/50)*np.sin(4*arg), axis=1)

class HolderTable1Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return -np.abs(np.prod(np.cos(positions), axis=1) * np.exp(np.abs(1 - np.sqrt(np.sum(positions**2, axis=1))/np.pi)))

class Levy3Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        y = 1 + (positions - 1) / 4
        y_j, y_next = y[:, :-1], y[:, 1:]
        term = np.sum((y_j - 1)**2 * (1 + 10 * np.sin(np.pi * y_next)**2), axis=1)
        return np.sin(np.pi * y[:, 0])**2 + term + (y[:, -1] - 1)**2

class LevyMontalvo2Function(FitnessFunction):
    def __init__(self): self.bounds = (-5, 5)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        term = np.sum((x_j - 1)**2 * (np.sin(3 * np.pi * x_next)**2 + 1), axis=1)
        return 0.1 * (np.sin(3 * np.pi * positions[:, 0])**2 + term + (positions[:, -1] - 1)**2 * (np.sin(2 * np.pi * positions[:, -1])**2 + 1))

class Mishra1Function(FitnessFunction):
    def __init__(self): self.bounds = (0, 1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        val = (1 + n - np.sum(positions[:, :-1], axis=1))
        # Numerical safety clip for power operation during particle explosion
        safe_val = np.maximum(val, 1e-12)
        return safe_val**val

class Mishra4Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sqrt(np.abs(np.sin(np.sqrt(np.abs(np.sum(positions**2, axis=1)))))) + 0.01 * np.sum(positions, axis=1)

class NeedleEyeFunction(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10); self.eye = 0.0001
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        mask = np.all(np.abs(positions) < self.eye, axis=1)
        res = np.sum(100 + np.abs(positions), axis=1)
        res[mask] = 1.0
        return res

class NorwegianFunction(FitnessFunction):
    def __init__(self): self.bounds = (-1.1, 1.1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.prod(np.cos(np.pi * positions**3) * (99 + positions) / 100, axis=1)

class PathologicalFunction(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        num = np.sin(np.sqrt(100 * x_j**2 + x_next**2))**2 - 0.5
        den = 0.5 + 0.001 * (x_j - x_next)**4
        return np.sum(0.5 + num / den, axis=1)

class Penalty1Function(FitnessFunction):
    def __init__(self): self.bounds = (-50, 50)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        y = 1 + (positions + 1) / 4
        y_j, y_next = y[:, :-1], y[:, 1:]
        u = lambda x: 100 * (x - 10)**4 * (x > 10) + 100 * (-x - 10)**4 * (x < -10)
        term = np.sum((y_j - 1)**2 * (1 + 10 * np.sin(np.pi * y_next)**2), axis=1)
        return (np.pi/30) * (10 * np.sin(np.pi * y[:, 0])**2 + term + (y[:, -1] - 1)**2) + np.sum(u(positions), axis=1)

class Penalty2Function(FitnessFunction):
    def __init__(self): self.bounds = (-50, 50)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        u = lambda x: 100 * (x - 5)**4 * (x > 5) + 100 * (-x - 5)**4 * (x < -5)
        term = np.sum((x_j - 1)**2 * (1 + np.sin(3 * np.pi * x_next)**2), axis=1)
        return 0.1 * (np.sin(3 * np.pi * positions[:, 0])**2 + term + (positions[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * positions[:, -1])**2)) + np.sum(u(positions), axis=1)

class PeriodicFunction(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return 1 + np.sum(np.sin(positions)**2, axis=1) - 0.1 * np.exp(-np.sum(positions**2, axis=1))

class Pinter2Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        idx = np.arange(1, n + 1)
        # Simplified handle for recursive indices A and B
        x_prev = np.roll(positions, 1, axis=1)
        x_next = np.roll(positions, -1, axis=1)
        A = x_prev * np.sin(positions) + np.sin(x_next)
        B = x_prev**2 - 2*positions + 3*x_next - np.cos(positions) + 1
        return np.sum(idx * positions**2, axis=1) + np.sum(20 * idx * np.sin(A)**2, axis=1) + np.sum(idx * np.log10(1 + idx * B**2), axis=1)

class Price2Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return 1 + np.sum(np.sin(positions)**2, axis=1) - 0.1 * np.exp(-np.sum(positions**2, axis=1))

class QingsFunction(FitnessFunction):
    def __init__(self): self.bounds = (-500, 500)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        idx = np.arange(1, positions.shape[1] + 1)
        return np.sum((positions**2 - idx)**2, axis=1)

class QuadricFunction(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(np.cumsum(positions, axis=1)**2, axis=1)

class QuinticFunction(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(positions**5 - 3*positions**4 + 4*positions**3 + 2*positions**2 - 10*positions - 4), axis=1)

class RanaFunction(FitnessFunction):
    def __init__(self): self.bounds = (-500, 500)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        t1 = np.sqrt(np.abs(x_next + x_j + 1))
        t2 = np.sqrt(np.abs(x_next - x_j + 1))
        return np.sum((x_next + 1) * np.cos(t2) * np.sin(t1) + x_j * np.cos(t1) * np.sin(t2), axis=1)

class RastriginFunction(FitnessFunction):
    def __init__(self): self.bounds = (-5.12, 5.12)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        return 10 * n + np.sum(positions**2 - 10 * np.cos(2 * np.pi * positions), axis=1)

class Ripple25Function(FitnessFunction):
    def __init__(self): self.bounds = (0, 1)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(-np.exp(-2 * np.log(2) * ((positions - 0.1)/0.8)**2) * (np.sin(5 * np.pi * positions)**6), axis=1)

class RosenbrockFunction(FitnessFunction):
    def __init__(self): self.bounds = (-30, 30)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        return np.sum(100 * (x_next - x_j**2)**2 + (x_j - 1)**2, axis=1)

class SalomonFunction(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.sum(positions**2, axis=1))
        return -np.cos(2 * np.pi * norm) + 0.1 * norm + 1

class Schubert4Function(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        res = np.zeros(positions.shape[0])
        for i in range(1, 6):
            res += np.sum(i * np.cos((i + 1) * positions + i), axis=1)
        return res

class Schwefel1Function(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return (np.sum(positions**2, axis=1))**np.sqrt(np.pi)

class SinusoidalFunction(FitnessFunction):
    def __init__(self): self.bounds = (0, 180)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        # A, B, z are assumed constants from context if not specified; using defaults
        A, B, z = 2.5, 5.0, 30.0
        return -(A * np.prod(np.sin(positions - z), axis=1) + np.prod(np.sin(B * (positions - z)), axis=1))

class StepFunction3Function(FitnessFunction):
    def __init__(self): self.bounds = (-5.12, 5.12)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(np.floor(positions**2), axis=1)

class TridFunction(FitnessFunction):
    def __init__(self): self.bounds = (-20, 20)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum((positions - 1)**2, axis=1) - np.sum(positions[:, 1:] * positions[:, :-1], axis=1)

class TrigonometricFunction(FitnessFunction):
    def __init__(self): self.bounds = (0, np.pi)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        sum_cos = np.sum(np.cos(positions), axis=1)[:, np.newaxis]
        idx = np.arange(1, n + 1)
        res = n - sum_cos + idx * (1 - np.cos(positions) - np.sin(positions))
        return np.sum(res**2, axis=1)

class VincentFunction(FitnessFunction):
    def __init__(self): self.bounds = (0.25, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        # Safety clip: ensure positions are positive before log operation
        # Particles outside bounds will be set to np.inf in Swarm.py anyway.
        safe_pos = np.maximum(positions, 1e-12)
        return -np.sum(np.sin(10 * np.log(safe_pos)), axis=1)

class WeierstrassFunction(FitnessFunction):
    def __init__(self):
        self.bounds = (-0.5, 0.5); self.a, self.b, self.k_max = 0.5, 3.0, 20
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        res = np.zeros(positions.shape[0])
        for k in range(self.k_max + 1):
            res += np.sum(self.a**k * np.cos(2 * np.pi * self.b**k * (positions + 0.5)), axis=1)
        val_off = n * sum(self.a**k * np.cos(np.pi * self.b**k) for k in range(self.k_max + 1))
        return res - val_off

class XinSheYang1Function(FitnessFunction):
    def __init__(self): self.bounds = (-5, 5)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        idx = np.arange(1, positions.shape[1] + 1)
        epsilon = np.random.uniform(0, 1, positions.shape)
        return np.sum(epsilon * np.abs(positions)**idx, axis=1)

class XinSheYang2Function(FitnessFunction):
    def __init__(self): self.bounds = (-2*np.pi, 2*np.pi)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(positions), axis=1) * np.exp(-np.sum(np.sin(positions**2), axis=1))

# ======================================================================
# EVALUATION SET (The 7 BOLDED functions)
# ======================================================================

class CrossLegTableFunction(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(positions, axis=1)
        prod_sin = np.prod(np.sin(positions), axis=1)
        return -1.0 / (np.abs(np.exp(np.abs(100 - norm/np.pi)) * prod_sin) + 1)**0.1

class Lanczos3Function(FitnessFunction):
    def __init__(self): self.bounds = (-20, 20); self.k = 3
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        # np.sinc is sin(pi*x)/(pi*x). Paper (A17) is sin(x)/x.
        return np.prod(np.sinc(positions/np.pi) * np.sinc(positions/(self.k * np.pi)), axis=1)

class MichalewiczTestFunction(FitnessFunction):
    def __init__(self): self.bounds = (0, np.pi); self.m = 10
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        idx = np.arange(1, positions.shape[1] + 1)
        return -np.sum(np.sin(positions) * np.sin(idx * positions**2 / np.pi)**(2 * self.m), axis=1)

class Schaffer4Function(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        sq_sum = x_j**2 + x_next**2
        num = np.cos(np.sin(np.abs(x_j**2 - x_next**2)))**2 - 0.5
        den = (1 + 0.001 * sq_sum)**2
        return np.sum(0.5 + num / den, axis=1)

class SineEnvelopeFunction(FitnessFunction):
    def __init__(self): self.bounds = (-100, 100)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        sq_sum = x_j**2 + x_next**2
        num = np.sin(np.sqrt(sq_sum))**2 - 0.5
        den = (1 + 0.001 * sq_sum)**2
        return np.sum(0.5 + num / den, axis=1)

class StretchedVSineWaveFunction(FitnessFunction):
    def __init__(self): self.bounds = (-10, 10)
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        x_j, x_next = positions[:, :-1], positions[:, 1:]
        sq_sum = x_j**2 + x_next**2
        return np.sum(sq_sum**0.25 * (np.sin(50 * sq_sum**0.1)**2 + 0.1), axis=1)

class WavyFunction(FitnessFunction):
    def __init__(self): self.bounds = (-np.pi, np.pi); self.k = 10
    def _fitness_function_invoke(self, positions: np.ndarray) -> np.ndarray:
        n = positions.shape[1]
        return 1 - (1/n) * np.sum(np.cos(self.k * positions) * np.exp(-positions**2/2), axis=1)

# ======================================================================
# REGISTRIES
# ======================================================================

TRAINING_SET: List[Type[FitnessFunction]] = [
    Ackley1Function, Alpine1Function, Bohachevsky1Function, BonyadiMichalewiczFunction,
    BrownFunction, CosineMixtureFunction, DeflectedCorrugatedSpringFunction, DiscussFunction,
    DropWaveFunction, EggCrateFunction, EggHolderFunction, EllipticFunction, ExponentialFunction,
    GiuntaFunction, HolderTable1Function, Levy3Function, LevyMontalvo2Function, MichalewiczTestFunction,
    Mishra1Function, Mishra4Function, NeedleEyeFunction, NorwegianFunction, PathologicalFunction,
    Penalty1Function, Penalty2Function, PeriodicFunction, Pinter2Function, Price2Function, QingsFunction,
    QuadricFunction, QuinticFunction, RanaFunction, RastriginFunction, Ripple25Function, RosenbrockFunction,
    SalomonFunction, Schubert4Function, Schwefel1Function, SinusoidalFunction, StepFunction3Function,
    TridFunction, TrigonometricFunction, VincentFunction, WeierstrassFunction, XinSheYang1Function,
    XinSheYang2Function
]

EVALUATION_SET: List[Type[FitnessFunction]] = [
    CrossLegTableFunction, Lanczos3Function, MichalewiczTestFunction, Schaffer4Function,
    SineEnvelopeFunction, StretchedVSineWaveFunction, WavyFunction
]