from typing import Tuple

import numpy as np


def min_max_array(data) -> []:
    data = np.array(data)
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = abs(max_val - min_val)

    if range_val > 0:
        normalized = (data - min_val) / range_val
    else:
        normalized = np.zeros_like(data)

    assert len(normalized) == len(data)

    return normalized


def sample_control_by_time(time: int, max_steps: int = 5000) -> Tuple[float, float, float]:
    """
    Calculates w, c1, and c2 based on time-step.
    Based on Equation (3) from the foundational paper.
    """
    return (
        0.4 * (((time - max_steps) / max_steps) ** 2) + 0.4,  # w
        -3 * (time / max_steps) + 3.5,  # c1
        3 * (time / max_steps) + 0.5  # c2
    )


def sample_diversity(positions: np.ndarray) -> tuple[float, float]:
    centers = np.mean(positions, axis=0)
    norm = np.linalg.norm(positions - centers, axis=1)
    return np.mean(norm), np.std(norm)


def sample_stability(w: float, c1: float, c2: float) -> bool:
    """Poli's stability condition (Section 2.3, Eq. 4)"""
    denom = (7 - 5 * w)
    if denom <= 0: return False
    return (c1 + c2) < (24 * (1 - w ** 2)) / denom


def sample_number_in_bounds(positions, lower, upper) -> int:
    """Counts how many particles are currently within the feasible search space."""
    in_bounds = (positions >= lower) & (positions <= upper)
    particle_mask = np.all(in_bounds, axis=1)
    count_in_bounds = np.sum(particle_mask)
    return int(count_in_bounds)
