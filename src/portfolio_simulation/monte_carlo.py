"""Monte Carlo estimators computed from simulated samples."""

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def mc_mean(samples: Array) -> tuple[float, float]:
    """Estimate E[X] and its standard error sigma_hat / sqrt(N)."""
    return float(samples.mean()), float(samples.std(ddof=1) / np.sqrt(len(samples)))


def value_at_risk(final_values: Array, initial_value: float, alpha: float = 0.05) -> float:
    """Loss that is exceeded with probability alpha."""
    return float(initial_value - np.quantile(final_values, alpha))