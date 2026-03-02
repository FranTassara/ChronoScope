"""
Feature Extraction Module for Consensus Rhythmicity Score
==========================================================

Shared module that runs multiple circadian analysis methods on a given
time series and extracts a standardized feature vector (~20 features).

Used by both the training script and the runtime meta-classifier.

Author: Francisco Tassara
"""

from typing import Dict, Optional, Any, List
import warnings
import numpy as np
import pandas as pd


# Ordered list of feature names - this defines the model's expected input
FEATURE_NAMES: List[str] = [
    'jtk_p_value',
    'jtk_tau',
    'jtk_period',
    'cosinor_p_value',
    'cosinor_r_squared',
    'cosinor_amplitude',
    'cosinor_period',
    'ls_p_value',
    'ls_power',
    'ls_dominant_period',
    'f24_score',
    'harmonic_p_value',
    'harmonic_r_squared',
    'method_agreement',
    'period_concordance',
    'amplitude_relative',
    'log_min_p_value',
    'period_dev_24h',
]


def _compute_r_squared(times: np.ndarray, values: np.ndarray,
                        mesor: float, amplitude: float,
                        acrophase_rad: float, period: float) -> float:
    """Compute R-squared from cosinor parameters."""
    y_fit = mesor + amplitude * np.cos(2 * np.pi * times / period - acrophase_rad)
    ss_res = np.sum((values - y_fit) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def extract_features(
    times: np.ndarray,
    values: np.ndarray,
    data: Optional[pd.DataFrame] = None,
    variable: Optional[str] = None,
    condition: Optional[str] = None,
    time_col: str = 'time',
    condition_col: str = 'condition',
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Run all sub-methods on a time series and extract the feature vector.

    Each sub-method is wrapped in try/except. If a method fails,
    its features are set to NaN (handled by the model's imputer).

    Args:
        times: Array of time points
        values: Array of measurement values
        data: Full DataFrame (needed for F24 which handles its own filtering)
        variable: Variable column name
        condition: Condition value
        time_col: Time column name
        condition_col: Condition column name
        parameters: Optional parameters dict

    Returns:
        Dict mapping feature names to float values
    """
    features: Dict[str, float] = {name: np.nan for name in FEATURE_NAMES}

    parameters = parameters or {}
    default_params = {
        'period': None,
        'period_range': (18.0, 32.0),
    }
    for k, v in default_params.items():
        if k not in parameters:
            parameters[k] = v

    p_values_for_agreement = []
    periods_for_concordance = []

    # --- JTK Cycle ---
    try:
        from .rhythm_analysis import _run_discrete_jtk

        series = pd.Series(values, index=times)
        period_range = list(range(20, 29))
        jtk_result = _run_discrete_jtk(series, period_range=period_range, asymmetries=[0.5])

        if jtk_result is not None:
            features['jtk_p_value'] = jtk_result.adj_p_value
            features['jtk_tau'] = jtk_result.tau
            features['jtk_period'] = jtk_result.period
            p_values_for_agreement.append(jtk_result.adj_p_value)
            periods_for_concordance.append(jtk_result.period)
    except Exception:
        pass

    # --- Cosinor OLS ---
    try:
        from .rhythm_analysis import _fit_cosinor, DefaultPeriodRanges

        cosinor_result = _fit_cosinor(times, values, period_range=DefaultPeriodRanges.CIRCADIAN)

        if cosinor_result is not None:
            features['cosinor_p_value'] = cosinor_result.adj_p_value
            features['cosinor_amplitude'] = cosinor_result.amplitude
            features['cosinor_period'] = cosinor_result.period
            p_values_for_agreement.append(cosinor_result.adj_p_value)
            periods_for_concordance.append(cosinor_result.period)

            # Compute R-squared from parameters
            r_sq = _compute_r_squared(
                times, values,
                cosinor_result.mesor,
                cosinor_result.amplitude,
                cosinor_result.acrophase_rad,
                cosinor_result.period
            )
            features['cosinor_r_squared'] = r_sq

            # For amplitude_relative
            if cosinor_result.mesor and cosinor_result.mesor != 0:
                features['amplitude_relative'] = cosinor_result.amplitude / abs(cosinor_result.mesor)
    except Exception:
        pass

    # --- Lomb-Scargle ---
    try:
        from .rhythm_analysis import _compute_lomb_scargle

        ls_result = _compute_lomb_scargle(
            times, values,
            period_range=(18.0, 32.0),
            n_periods=1000
        )

        if ls_result is not None:
            features['ls_p_value'] = ls_result.false_alarm_probability
            features['ls_power'] = ls_result.dominant_power
            features['ls_dominant_period'] = ls_result.dominant_period
            p_values_for_agreement.append(ls_result.false_alarm_probability)
            periods_for_concordance.append(ls_result.dominant_period)
    except Exception:
        pass

    # --- Fourier F24 ---
    try:
        from .rhythm_analysis import _compute_fourier_f24

        f24_times = times
        f24_values = values

        # If we have full data, use it for better F24 (handles replicates)
        if data is not None and variable is not None and condition is not None:
            cond_data = data[data[condition_col] == condition]
            f24_times = cond_data[time_col].values.astype(float)
            f24_values = cond_data[variable].values.astype(float)
            mask = ~(np.isnan(f24_times) | np.isnan(f24_values))
            f24_times = f24_times[mask]
            f24_values = f24_values[mask]

        f24_result = _compute_fourier_f24(f24_times, f24_values, target_period=24.0, n_permutations=100)

        if f24_result is not None:
            features['f24_score'] = f24_result.f24_score
    except Exception:
        pass

    # --- Harmonic Cosinor ---
    try:
        from .rhythm_analysis import _fit_harmonic_cosinor, DefaultPeriodRanges

        harm_result = _fit_harmonic_cosinor(
            times, values,
            period_range=DefaultPeriodRanges.CIRCADIAN,
            n_harmonics=2
        )

        if harm_result is not None:
            features['harmonic_p_value'] = harm_result.adj_p_value
            p_values_for_agreement.append(harm_result.adj_p_value)

            # Get R-squared from fit_model if available
            if harm_result.fit_model and 'r_squared' in harm_result.fit_model:
                features['harmonic_r_squared'] = harm_result.fit_model['r_squared']
    except Exception:
        pass

    # --- Derived features ---

    # Method agreement: fraction of methods with p < 0.05
    if p_values_for_agreement:
        valid_p = [p for p in p_values_for_agreement if p is not None and not np.isnan(p)]
        if valid_p:
            features['method_agreement'] = sum(1 for p in valid_p if p < 0.05) / len(valid_p)

    # Period concordance: std of detected periods (lower = more consistent)
    if len(periods_for_concordance) >= 2:
        valid_periods = [p for p in periods_for_concordance if p is not None and not np.isnan(p)]
        if len(valid_periods) >= 2:
            features['period_concordance'] = float(np.std(valid_periods))

    # Log of minimum p-value across methods (compresses dynamic range)
    if p_values_for_agreement:
        valid_p = [p for p in p_values_for_agreement if p is not None and not np.isnan(p) and p > 0]
        if valid_p:
            features['log_min_p_value'] = float(np.log10(min(valid_p)))

    # Period deviation from 24h: min |period - 24| across methods
    if periods_for_concordance:
        valid_periods = [p for p in periods_for_concordance if p is not None and not np.isnan(p)]
        if valid_periods:
            features['period_dev_24h'] = float(min(abs(p - 24.0) for p in valid_periods))

    return features
