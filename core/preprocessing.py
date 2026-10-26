"""
Preprocessing Module
====================

Time series preprocessing pipeline for circadian rhythm analysis.

Provides three sequential filter stages:
  1. Outlier removal  — marks bad data points as NaN
  2. Detrending       — removes non-rhythmic slow trends
  3. Smoothing        — reduces high-frequency noise

Each stage is individually optional and controlled via PreprocessingConfig.

Pipeline execution order: outlier removal → detrending → smoothing
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import signal as sp_signal


# =============================================================================
# ENUMS
# =============================================================================

class DetrendMethod(Enum):
    LINEAR = "linear"
    MOVING_AVERAGE = "moving_average"
    POLYNOMIAL = "polynomial"


class OutlierMethod(Enum):
    IQR = "iqr"
    ZSCORE = "zscore"


class SmoothMethod(Enum):
    MOVING_AVERAGE = "moving_average"
    SAVITZKY_GOLAY = "savitzky_golay"
    BUTTERWORTH = "butterworth"


# =============================================================================
# CONFIG AND REPORT
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline.

    All stages are disabled by default. Enable each stage with its flag
    and adjust parameters as needed.
    """
    # --- Outlier removal ---
    remove_outliers: bool = False
    outlier_method: OutlierMethod = OutlierMethod.IQR
    # IQR: multiplier (1.5 = mild, 3.0 = extreme). ZSCORE: number of SDs.
    outlier_threshold: float = 1.5

    # --- Detrending ---
    detrend: bool = False
    detrend_method: DetrendMethod = DetrendMethod.MOVING_AVERAGE
    # Window size in hours for moving-average detrending (default = one period).
    # A 24 h centered window removes trends while preserving the circadian oscillation.
    detrend_window: float = 24.0
    detrend_poly_degree: int = 2  # polynomial degree for DetrendMethod.POLYNOMIAL

    # --- Smoothing ---
    smooth: bool = False
    smooth_method: SmoothMethod = SmoothMethod.MOVING_AVERAGE
    smooth_window: int = 3  # samples; must be odd (enforced internally for Savitzky-Golay)
    # Butterworth-specific (ignored for other methods)
    butterworth_order: int = 4          # filter order (1–8); higher = steeper rolloff
    butterworth_cutoff: float = 12.0    # low-pass cutoff expressed as a period in hours;
                                        # oscillations shorter than this value are attenuated


class StageSkippedWarning(RuntimeWarning):
    """A filter stage could not be applied and returned the series unchanged."""


@dataclass
class PreprocessingReport:
    """Summary of what the preprocessing pipeline applied."""
    series_processed: int = 0
    outliers_removed: int = 0
    detrend_applied: bool = False
    smooth_applied: bool = False
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# STAGE 1 — OUTLIER REMOVAL
# =============================================================================

def _remove_outliers_array(
    values: np.ndarray,
    method: OutlierMethod,
    threshold: float,
) -> np.ndarray:
    """Return a copy of *values* with outliers replaced by NaN."""
    result = values.astype(float).copy()
    valid = ~np.isnan(result)

    if valid.sum() < 4:
        warnings.warn("Outlier removal skipped: fewer than 4 valid values.", StageSkippedWarning)
        return result

    if method == OutlierMethod.IQR:
        q1, q3 = np.percentile(result[valid], [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
        result[(result < lo) | (result > hi)] = np.nan

    else:  # ZSCORE
        mu = np.nanmean(result)
        sigma = np.nanstd(result)
        if sigma == 0:
            return result
        z = np.abs((result - mu) / sigma)
        result[z > threshold] = np.nan

    return result


# =============================================================================
# STAGE 2 — DETRENDING
# =============================================================================

def _detrend_array(
    times: np.ndarray,
    values: np.ndarray,
    method: DetrendMethod,
    window_hours: float = 24.0,
    poly_degree: int = 2,
) -> np.ndarray:
    """Return a detrended copy of *values*.

    NaN values are ignored during trend estimation.  The trend is still
    evaluated at their positions and subtracted, so detrended NaN positions
    remain NaN.
    """
    result = values.astype(float).copy()
    valid = ~np.isnan(result)

    if valid.sum() < 4:
        warnings.warn("Detrending skipped: fewer than 4 valid values.", StageSkippedWarning)
        return result

    t_valid = times[valid]
    v_valid = result[valid]

    if method == DetrendMethod.LINEAR:
        coeffs = np.polyfit(t_valid, v_valid, 1)
        trend = np.polyval(coeffs, times)
        result -= trend

    elif method == DetrendMethod.MOVING_AVERAGE:
        dt = float(np.median(np.diff(np.sort(times))))
        if dt <= 0:
            warnings.warn(
                "Moving-average detrending skipped: time points are repeated "
                "(replicates or subjects), so the sampling interval is 0.",
                StageSkippedWarning,
            )
            return result
        window_samples = max(3, int(round(window_hours / dt)))
        s = pd.Series(result, dtype=float)
        trend = s.rolling(window=window_samples, center=True, min_periods=1).mean()
        result -= trend.to_numpy()

    else:  # POLYNOMIAL
        degree = max(1, min(poly_degree, len(t_valid) - 2))
        coeffs = np.polyfit(t_valid, v_valid, degree)
        trend = np.polyval(coeffs, times)
        result -= trend

    return result


# =============================================================================
# STAGE 3 — SMOOTHING
# =============================================================================

def _smooth_array(
    values: np.ndarray,
    method: SmoothMethod,
    window: int,
    times: Optional[np.ndarray] = None,
    butterworth_order: int = 4,
    butterworth_cutoff: float = 12.0,
) -> np.ndarray:
    """Return a smoothed copy of *values*.

    NaN positions are linearly interpolated before smoothing and restored
    as NaN in the output.

    Args:
        values:             1-D measurement array (may contain NaN).
        method:             Smoothing algorithm.
        window:             Window size in samples (Moving Average / Savitzky-Golay).
        times:              Time axis in hours — required for Butterworth.
        butterworth_order:  Filter order (1–8).
        butterworth_cutoff: Low-pass cutoff expressed as a period in hours.
                            Oscillations shorter than this are attenuated.
    """
    result = values.astype(float).copy()
    nan_mask = np.isnan(result)

    if nan_mask.all():
        warnings.warn("Smoothing skipped: no valid values.", StageSkippedWarning)
        return result

    # Fill NaN so the smoother doesn't see gaps
    if nan_mask.any():
        filled = pd.Series(result).interpolate(method='linear', limit_direction='both').to_numpy()
    else:
        filled = result.copy()

    if method == SmoothMethod.MOVING_AVERAGE:
        smoothed = (
            pd.Series(filled)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy()
            .copy()
        )

    elif method == SmoothMethod.SAVITZKY_GOLAY:
        w = window if window % 2 == 1 else window + 1
        w = max(w, 5)
        polyorder = min(3, w - 2)
        smoothed = sp_signal.savgol_filter(filled, window_length=w, polyorder=polyorder).copy()

    else:  # BUTTERWORTH
        if times is None or len(filled) < 6:
            # Cannot compute cutoff without a time axis; return unfiltered
            warnings.warn(
                "Butterworth smoothing skipped: needs a time axis and at least 6 values.",
                StageSkippedWarning,
            )
            return result

        dt = float(np.median(np.diff(np.sort(times))))
        if dt <= 0:
            warnings.warn(
                "Butterworth smoothing skipped: time points are repeated "
                "(replicates or subjects), so the sampling interval is 0.",
                StageSkippedWarning,
            )
            return result

        nyquist = 1.0 / (2.0 * dt)          # cycles / hour
        fc = 1.0 / butterworth_cutoff        # cutoff frequency in cycles / hour
        Wn = fc / nyquist                    # normalised [0, 1]

        if not (0.0 < Wn < 1.0):
            warnings.warn(
                f"Butterworth cutoff {butterworth_cutoff} h is outside the valid range "
                f"for the sampling interval {dt} h. Smoothing skipped.",
                StageSkippedWarning,
            )
            return result

        # filtfilt requires len(signal) > 3*(order+1); reduce order if needed
        max_safe_order = max(1, int((len(filled) - 2) // 3))
        order = min(butterworth_order, max_safe_order)

        b, a = sp_signal.butter(order, Wn, btype='low', analog=False)
        smoothed = sp_signal.filtfilt(b, a, filled).copy()

    smoothed[nan_mask] = np.nan
    return smoothed


# =============================================================================
# PUBLIC API
# =============================================================================

def _run_stage(report: PreprocessingReport, where: str, stage, *args, **kwargs):
    """Run one filter stage; a skipped stage is recorded in *report*.

    Returns (values, applied).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', StageSkippedWarning)
        values = stage(*args, **kwargs)
    applied = True
    for w in caught:
        if issubclass(w.category, StageSkippedWarning):
            report.warnings.append(f"{where}: {w.message}")
            applied = False
        else:
            warnings.warn(w.message, w.category)
    return values, applied


def apply_preprocessing(
    df: pd.DataFrame,
    variables: List[str],
    config: PreprocessingConfig,
    time_col: str = 'time',
    condition_col: Optional[str] = 'condition',
    subject_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """Apply the preprocessing pipeline to *variables* in *df*.

    Processes each (variable, condition) group independently, or each
    (variable, condition, subject) series when *subject_col* is given.  The
    input DataFrame is never modified; a transformed copy is returned.

    Outlier removal acts on all values of the series.  Detrending and
    smoothing act on the mean at each time point, sorted by time; the
    correction is added to every value of that time point, so replicate
    scatter is kept.

    Pipeline order: outlier removal → detrending → smoothing.

    Args:
        df:            Input DataFrame.
        variables:     Column names to preprocess.
        config:        Pipeline parameters.
        time_col:      Name of the time column (values in hours).
        condition_col: Name of the condition column.  Pass None if absent.
        subject_col:   Name of the subject column (repeated-measures data).
                       Pass None if absent.

    Returns:
        (preprocessed_df, report)
    """
    report = PreprocessingReport()

    if not (config.remove_outliers or config.detrend or config.smooth):
        return df.copy(), report

    result_df = df.copy()
    vars_present = [v for v in variables if v in df.columns]

    if not vars_present:
        report.warnings.append("None of the requested variables were found in the DataFrame.")
        return result_df, report

    if time_col not in df.columns:
        report.warnings.append(f"Time column '{time_col}' not found; preprocessing skipped.")
        return result_df, report

    # Build list of (group_label, row_index) pairs: one series per condition,
    # or per condition x subject when a subject column is given
    keys = [c for c in (condition_col, subject_col) if c and c in df.columns]
    if keys:
        groups = [(key if isinstance(key, tuple) else (key,), grp.index)
                  for key, grp in df.groupby(keys, sort=False, dropna=False)]
    else:
        groups = [((), df.index)]

    for label, idx in groups:
        times = df.loc[idx, time_col].to_numpy(dtype=float)
        timed = np.isfinite(times)
        # Unique sorted time points; inv maps each timed row to its time point
        t_unique, inv = np.unique(times[timed], return_inverse=True)

        for var in vars_present:
            values = df.loc[idx, var].to_numpy(dtype=float, copy=True)
            where = " / ".join([var] + [str(k) for k in label])

            if config.remove_outliers:
                n_nan_before = int(np.isnan(values).sum())
                values, _ = _run_stage(
                    report, where, _remove_outliers_array,
                    values, config.outlier_method, config.outlier_threshold,
                )
                report.outliers_removed += int(np.isnan(values).sum()) - n_nan_before

            if config.detrend or config.smooth:
                # Detrending and smoothing act on the time course: they filter the
                # mean at each time point (sorted by time) and add the resulting
                # correction to every value of that time point. With one value per
                # time point this is the series itself.
                mean = (pd.Series(values[timed]).groupby(inv).mean()
                        .reindex(range(len(t_unique))).to_numpy(dtype=float))
                filtered = mean

                if config.detrend:
                    filtered, applied = _run_stage(
                        report, where, _detrend_array,
                        t_unique, filtered,
                        config.detrend_method,
                        config.detrend_window,
                        config.detrend_poly_degree,
                    )
                    report.detrend_applied |= applied

                if config.smooth:
                    filtered, applied = _run_stage(
                        report, where, _smooth_array,
                        filtered, config.smooth_method, config.smooth_window,
                        times=t_unique,
                        butterworth_order=config.butterworth_order,
                        butterworth_cutoff=config.butterworth_cutoff,
                    )
                    report.smooth_applied |= applied

                values[timed] += (filtered - mean)[inv]

            result_df.loc[idx, var] = values
            report.series_processed += 1

    return result_df, report
