"""Shared fixtures for the ChronoScope test suite.

Kept dependency-free (numpy/pandas only) so the suite runs with the same
environment the application itself needs -- no extra test dependencies.

Replicates get *independent* noise draws. Sharing one noise realisation across
replicates makes pseudo-replicates that look like a consistent signal to any
method that pools them, which inflates significance under the null.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

SEED = 20260921
VARIABLE = 'signal'
CONDITION = 'ctrl'


def _grid(n_days, sampling_interval):
    return np.arange(0.0, n_days * 24.0, sampling_interval)


def synthetic_series(n_days=3, sampling_interval=4.0, period=24.0,
                     amplitude=2.0, mesor=10.0, noise_sd=0.3, seed=SEED):
    """A single clean cosine rhythm sampled every `sampling_interval` hours."""
    rng = np.random.default_rng(seed)
    times = _grid(n_days, sampling_interval)
    values = mesor + amplitude * np.cos(2 * np.pi * times / period)
    return times, values + rng.normal(0.0, noise_sd, times.size)


def synthetic_noise(n_days=3, sampling_interval=4.0, mesor=10.0,
                    noise_sd=2.0, seed=SEED + 1):
    """A single arrhythmic series on the same sampling grid."""
    rng = np.random.default_rng(seed)
    times = _grid(n_days, sampling_interval)
    return times, mesor + rng.normal(0.0, noise_sd, times.size)


def rhythmic_dataframe(n_days=4, sampling_interval=2.0, n_replicates=3,
                       period=24.0, amplitude=2.0, mesor=10.0,
                       noise_sd=0.5, seed=SEED, variable=VARIABLE,
                       condition=CONDITION):
    """Long-format DataFrame: a 24 h rhythm with independent replicate noise."""
    rng = np.random.default_rng(seed)
    times = _grid(n_days, sampling_interval)
    signal = mesor + amplitude * np.cos(2 * np.pi * times / period)
    rows = []
    for _ in range(n_replicates):
        values = signal + rng.normal(0.0, noise_sd, times.size)
        rows += [{'time': float(t), 'condition': condition, variable: float(v)}
                 for t, v in zip(times, values)]
    return pd.DataFrame(rows)


def noise_dataframe(n_days=4, sampling_interval=2.0, n_replicates=3,
                    mesor=10.0, noise_sd=2.0, seed=SEED + 1,
                    variable=VARIABLE, condition=CONDITION):
    """Long-format DataFrame of pure noise, same grid and replicate count."""
    rng = np.random.default_rng(seed)
    times = _grid(n_days, sampling_interval)
    rows = []
    for _ in range(n_replicates):
        values = mesor + rng.normal(0.0, noise_sd, times.size)
        rows += [{'time': float(t), 'condition': condition, variable: float(v)}
                 for t, v in zip(times, values)]
    return pd.DataFrame(rows)
