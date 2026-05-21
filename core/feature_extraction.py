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


# Ordered list of feature names — defines the model's input vector.
#
# v2: trimmed from 18 → 11 features. Removed jtk_tau, jtk_period,
# harmonic_p_value, harmonic_r_squared, method_agreement,
# period_concordance, log_min_p_value — all had permutation importance
# ≤ 0.002 on the v1 holdout (essentially noise for generalization),
# while their MDI was inflated by training-set correlations.
#
# extract_features() still computes harmonic_p_value, method_agreement,
# period_concordance, jtk_tau, jtk_period etc. and returns them in the
# features dict — they are consumed by the UI (radar chart, sub-method
# details panel) and by analysis_engine result objects. They are simply
# not fed to the Random Forest.
FEATURE_NAMES: List[str] = [
    'jtk_p_value',
    'cosinor_p_value',
    'cosinor_r_squared',
    'cosinor_amplitude',
    'cosinor_period',
    'ls_p_value',
    'ls_power',
    'ls_dominant_period',
    'f24_score',
    'amplitude_relative',
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


# Window the model was trained on. Used by _resolve_params to decide whether
# a user-supplied period_range can be honored.
_TRAINED_PERIOD_LOW = 18.0
_TRAINED_PERIOD_HIGH = 32.0


def _resolve_params(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Filter user parameters for the AI consensus path.

    Policy:
      - period_range: honored iff it intersects the training window
        [18, 32]h. Clipped to that window if it spills out, falls back
        to training defaults if it's entirely outside. Affects features
        that are fed to the model (JTK, Cosinor, LS).
      - n_harmonics: honored freely. Only affects harmonic_p_value /
        harmonic_r_squared, which are NO LONGER fed to the model — they
        only drive the UI's harmonic sub-method panel. Safe to expose.
      - f24 target_period and jtk asymmetries: locked. Changing them
        silently shifts feature semantics the model relies on.

    Returns a dict with keys: jtk_periods (list[int]),
    cosinor_periods (list[float]), ls_range (tuple), harmonic_periods
    (list[float]), n_harmonics (int).
    """
    from .rhythm_analysis import DefaultPeriodRanges

    parameters = parameters or {}
    n_harmonics = int(parameters.get('n_harmonics', 2))

    defaults = {
        'jtk_periods':      DefaultPeriodRanges.CIRCADIAN_INT,
        'cosinor_periods':  DefaultPeriodRanges.CIRCADIAN,
        'ls_range':         (18.0, 32.0),
        'harmonic_periods': DefaultPeriodRanges.CIRCADIAN,
        'n_harmonics':      n_harmonics,
    }

    user_range = parameters.get('period_range')
    if user_range is None:
        return defaults

    lo, hi = float(user_range[0]), float(user_range[1])
    safe_lo = max(_TRAINED_PERIOD_LOW, lo)
    safe_hi = min(_TRAINED_PERIOD_HIGH, hi)

    if safe_lo >= safe_hi - 1.0:
        warnings.warn(
            f"AI consensus model was trained on period range "
            f"[{_TRAINED_PERIOD_LOW}, {_TRAINED_PERIOD_HIGH}]h; "
            f"requested [{lo}, {hi}]h falls outside. Using training "
            f"defaults — predictions for this range are not supported.",
            stacklevel=2,
        )
        return defaults

    if (lo, hi) != (safe_lo, safe_hi):
        warnings.warn(
            f"AI consensus: clipping requested period range "
            f"[{lo}, {hi}]h to [{safe_lo}, {safe_hi}]h "
            f"(model's training window).",
            stacklevel=2,
        )

    jtk_periods = list(range(int(np.ceil(safe_lo)),
                              int(np.floor(safe_hi)) + 1))
    cosinor_periods = list(np.arange(safe_lo, safe_hi + 1e-9, 0.5))

    return {
        'jtk_periods':      jtk_periods,
        'cosinor_periods':  cosinor_periods,
        'ls_range':         (safe_lo, safe_hi),
        'harmonic_periods': cosinor_periods,
        'n_harmonics':      n_harmonics,
    }


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

    p = _resolve_params(parameters)

    p_values_for_agreement = []
    periods_for_concordance = []

    # --- JTK Cycle ---
    # asymmetries locked at [0.5]: the model was trained with the symmetric
    # waveform assumption; varying asymmetries shifts jtk_p_value semantics.
    try:
        from .rhythm_analysis import _run_discrete_jtk

        series = pd.Series(values, index=times)
        jtk_result = _run_discrete_jtk(
            series, period_range=p['jtk_periods'], asymmetries=[0.5]
        )

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
        from .rhythm_analysis import _fit_cosinor

        cosinor_result = _fit_cosinor(times, values, period_range=p['cosinor_periods'])

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
            period_range=p['ls_range'],
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
    # Uses the same (times, values) input as every other sub-method: the
    # caller is responsible for whatever replicate aggregation it wants
    # (training averages replicates per timepoint upstream). Earlier
    # versions had F24 reach into the full DataFrame and re-extract per-
    # replicate values, which gave it access to within-timepoint variance
    # that other methods didn't see. The asymmetric data view inflated
    # f24_score's MDI relative to its permutation importance (~5-7x gap),
    # not because F24 is fundamentally that informative.
    try:
        from .rhythm_analysis import _compute_fourier_f24

        f24_result = _compute_fourier_f24(times, values, target_period=24.0, n_permutations=100)

        if f24_result is not None:
            features['f24_score'] = f24_result.f24_score
    except Exception:
        pass

    # --- Harmonic Cosinor ---
    # harmonic_p_value / harmonic_r_squared are no longer model features,
    # so honoring user n_harmonics here only affects the UI sub-method
    # panel — it cannot move the model's prediction.
    try:
        from .rhythm_analysis import _fit_harmonic_cosinor

        harm_result = _fit_harmonic_cosinor(
            times, values,
            period_range=p['harmonic_periods'],
            n_harmonics=p['n_harmonics'],
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

    # Log of minimum p-value across methods (compresses dynamic range).
    # Floor at machine epsilon so that p==0.0 (numerical underflow for very
    # strong rhythms) maps to log10(~5e-324) ≈ -323 rather than becoming NaN.
    if p_values_for_agreement:
        valid_p = [p for p in p_values_for_agreement if p is not None and not np.isnan(p)]
        if valid_p:
            min_p = max(min(valid_p), np.finfo(float).tiny)
            features['log_min_p_value'] = float(np.log10(min_p))

    # Period deviation from 24h: min |period - 24| across methods
    if periods_for_concordance:
        valid_periods = [p for p in periods_for_concordance if p is not None and not np.isnan(p)]
        if valid_periods:
            features['period_dev_24h'] = float(min(abs(p - 24.0) for p in valid_periods))

    return features
