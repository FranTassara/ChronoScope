"""
Circadian metrics for the Visualization module.

Provides three analysis functions used when DAM or AWD data is loaded:
  - chi_square_periodogram : Sokolove-Bushell (1978) period estimation
  - compute_is_iv          : Interdaily Stability and Intradaily Variability
  - compute_alpha_rho      : Activity/rest duration and onset/offset variability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chi-square periodogram (Sokolove & Bushell, 1978)
# ---------------------------------------------------------------------------

def chi_square_periodogram(
    times: np.ndarray,
    activity: np.ndarray,
    period_min: float = 18.0,
    period_max: float = 32.0,
    period_step: float = 0.1,
    alpha: float = 0.05,
) -> Dict:
    """Sokolove-Bushell chi-square periodogram.

    For each test period P the time series is folded into P/bin_size slots.
    The between-slot variance relative to the total variance gives Qp.
    τ is the period that maximises Qp.

    Parameters
    ----------
    times, activity : uniform-interval 1-D arrays (hours, counts)
    period_min / max / step : search range for τ (hours)
    alpha : significance level for the threshold line

    Returns
    -------
    dict with keys:
        periods      – array of tested periods
        qp           – array of Qp statistics (one per period)
        tau          – period at max Qp (hours)
        tau_qp       – Qp value at τ
        significance – Qp threshold at the requested alpha level
        p_values     – approximate p-values for each period
        bin_size_h   – inferred bin size (hours)
        n_points     – number of data points used
    """
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)

    # Infer bin size from the median spacing between consecutive time points
    diffs = np.diff(times)
    bin_size_h = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 1.0

    N = len(activity)
    grand_mean = np.mean(activity)
    grand_var = np.var(activity, ddof=1)

    if grand_var == 0 or N < 3:
        return {
            'periods': np.array([]),
            'qp': np.array([]),
            'tau': None,
            'tau_qp': None,
            'significance': None,
            'p_values': np.array([]),
            'bin_size_h': bin_size_h,
            'n_points': N,
        }

    periods = np.arange(period_min, period_max + period_step * 0.5, period_step)
    qp_values = np.zeros(len(periods))

    for idx, P in enumerate(periods):
        n_slots = max(int(round(P / bin_size_h)), 2)
        # Assign each measurement to a slot: slot = floor(t / bin_size) % n_slots
        slot_indices = (np.floor(times / bin_size_h).astype(int)) % n_slots

        slot_sums = np.zeros(n_slots)
        slot_counts = np.zeros(n_slots, dtype=int)
        for i, s in enumerate(slot_indices):
            slot_sums[s] += activity[i]
            slot_counts[s] += 1

        # Only use slots that have at least one observation
        occupied = slot_counts > 0
        if occupied.sum() < 2:
            continue
        slot_means = slot_sums[occupied] / slot_counts[occupied]

        # Qp = N * between-slot variance / (n_slots * grand_variance)
        between_var = np.sum((slot_means - grand_mean) ** 2 * slot_counts[occupied]) / N
        qp_values[idx] = N * between_var / grand_var

    # Significance threshold: Qp ~ chi²(n_slots - 1) / (n_slots - 1) * N
    # Practical approach: use a fixed chi²-based threshold averaged over the
    # period range; this matches the original Sokolove-Bushell 0.05 line.
    from scipy import stats as _stats
    # Use median n_slots across the search range for the threshold
    median_n_slots = max(int(round(np.median(periods) / bin_size_h)), 2)
    df = median_n_slots - 1
    chi2_crit = _stats.chi2.ppf(1 - alpha, df=df)
    # Normalised to Qp scale: threshold Qp ≈ chi2_crit / N
    significance = chi2_crit / N * (N / median_n_slots) if N > 0 else None

    # Approximate p-values
    n_slots_per_period = np.array([max(int(round(P / bin_size_h)), 2) for P in periods])
    # Use chi2 CDF per period
    p_values = np.array([
        1 - _stats.chi2.cdf(qp * (n_sl - 1), df=n_sl - 1)
        for qp, n_sl in zip(qp_values, n_slots_per_period)
    ])

    best_idx = int(np.argmax(qp_values))
    tau = float(periods[best_idx])
    tau_qp = float(qp_values[best_idx])

    return {
        'periods': periods,
        'qp': qp_values,
        'tau': tau,
        'tau_qp': tau_qp,
        'significance': significance,
        'p_values': p_values,
        'bin_size_h': bin_size_h,
        'n_points': N,
    }


# ---------------------------------------------------------------------------
# Interdaily Stability (IS) and Intradaily Variability (IV)
# ---------------------------------------------------------------------------

def compute_is_iv(
    times: np.ndarray,
    activity: np.ndarray,
    target_bin_h: float = 1.0,
) -> Dict:
    """Compute IS and IV after resampling to hourly bins.

    IS (Interdaily Stability): how consistent the 24-h profile is across days.
        IS = 1 means perfect day-to-day consistency, 0 = random.
    IV (Intradaily Variability): how fragmented the activity bouts are.
        IV = 0 means perfectly smooth; high values = many transitions.

    Reference: Van Someren et al. (1999) Chronobiol Int.

    Parameters
    ----------
    times    : time in hours (ZT, continuous, not modulo 24)
    activity : activity counts
    target_bin_h : resampling resolution in hours (default 1 h)

    Returns
    -------
    dict with keys: IS, IV, n_days, n_bins_per_day, resampled (bool)
    """
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)

    # Build a pandas Series with a datetime-like index for easy resampling
    # We create a synthetic DatetimeIndex starting at epoch 0
    origin = pd.Timestamp('2000-01-01')
    dt_index = origin + pd.to_timedelta(times * 3600, unit='s')
    series = pd.Series(activity.astype(float), index=dt_index)
    series = series.sort_index()

    # Infer native bin size
    native_bin_h = float(np.median(np.diff(times)[np.diff(times) > 0])) if len(times) > 1 else 1.0
    resampled = native_bin_h < target_bin_h * 0.99

    # Resample to target_bin_h if needed
    rule = f'{int(target_bin_h * 60)}min'
    rs = series.resample(rule).mean()
    rs = rs.dropna()

    if len(rs) < 2:
        return {'IS': None, 'IV': None, 'n_days': 0, 'n_bins_per_day': 0, 'resampled': resampled}

    x = rs.values
    N = len(x)
    p = max(1, int(round(24.0 / target_bin_h)))  # bins per day

    if N < p:
        return {'IS': None, 'IV': None, 'n_days': N / p, 'n_bins_per_day': p, 'resampled': resampled}

    x_mean = np.mean(x)
    total_var = np.sum((x - x_mean) ** 2)

    if total_var == 0:
        return {'IS': 0.0, 'IV': 0.0, 'n_days': N / p, 'n_bins_per_day': p, 'resampled': resampled}

    # IS: reshape into days × bins, compute hourly means across days
    n_complete_days = N // p
    if n_complete_days < 1:
        return {'IS': None, 'IV': None, 'n_days': N / p, 'n_bins_per_day': p, 'resampled': resampled}

    x_trimmed = x[:n_complete_days * p]
    day_matrix = x_trimmed.reshape(n_complete_days, p)
    hourly_means = day_matrix.mean(axis=0)

    is_numerator = n_complete_days * p * np.sum((hourly_means - x_mean) ** 2)
    is_denominator = p * np.sum((x_trimmed - x_mean) ** 2)
    IS = float(is_numerator / is_denominator) if is_denominator > 0 else None

    # IV: consecutive differences
    iv_numerator = N * np.sum(np.diff(x) ** 2)
    iv_denominator = (N - 1) * total_var
    IV = float(iv_numerator / iv_denominator) if iv_denominator > 0 else None

    return {
        'IS': IS,
        'IV': IV,
        'n_days': n_complete_days,
        'n_bins_per_day': p,
        'resampled': resampled,
    }


# ---------------------------------------------------------------------------
# Activity duration (α), rest duration (ρ), and onset/offset variability
# ---------------------------------------------------------------------------

def compute_alpha_rho(
    times: np.ndarray,
    activity: np.ndarray,
    threshold_method: str = 'mean',
) -> Dict:
    """Compute α (active phase duration), ρ (rest duration), and variability.

    For each day:
      1. Average activity profile across subjects (already averaged if input is
         per-condition mean series).
      2. Threshold = mean (or 25th-percentile) of that day's profile.
      3. α = fraction of the day above threshold × 24 h.
      4. ρ = 24 - α.
      5. onset / offset = first / last ZT hour above threshold.

    Parameters
    ----------
    times            : ZT hours (continuous, not modulo 24)
    activity         : activity values
    threshold_method : 'mean' | 'percentile25'

    Returns
    -------
    dict with keys:
        days          – list of day numbers (1-based)
        alpha_h       – α per day (hours)
        rho_h         – ρ per day (hours)
        onset_zt      – activity onset per day (ZT hours)
        offset_zt     – activity offset per day (ZT hours)
        onset_sd      – SD of onset times (hours)
        offset_sd     – SD of offset times (hours)
        alpha_mean    – mean α across days (hours)
        alpha_sd      – SD of α across days (hours)
        rho_mean      – mean ρ across days (hours)
        rho_sd        – SD of ρ across days (hours)
    """
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)

    df = pd.DataFrame({'time': times, 'activity': activity})
    df['zt'] = df['time'] % 24.0
    df['day'] = (df['time'] // 24).astype(int) + 1

    days_list: List[int] = []
    alpha_h: List[Optional[float]] = []
    rho_h: List[Optional[float]] = []
    onset_zt: List[Optional[float]] = []
    offset_zt: List[Optional[float]] = []

    for day, day_df in df.groupby('day'):
        profile = day_df.groupby('zt')['activity'].mean().sort_index()
        if len(profile) == 0:
            continue

        if threshold_method == 'percentile25':
            thresh = float(np.percentile(profile.values, 25))
        else:
            thresh = float(profile.mean())

        above = profile[profile > thresh]

        if len(above) > 0:
            on = float(above.index[0])
            off = float(above.index[-1])
            # α = fraction of unique ZT bins above threshold × 24 h
            n_total_bins = len(profile)
            n_active_bins = len(above)
            alpha = 24.0 * n_active_bins / n_total_bins
            rho = 24.0 - alpha
        else:
            on = None
            off = None
            alpha = 0.0
            rho = 24.0

        days_list.append(int(day))
        alpha_h.append(alpha)
        rho_h.append(rho)
        onset_zt.append(on)
        offset_zt.append(off)

    valid_onsets = [v for v in onset_zt if v is not None]
    valid_offsets = [v for v in offset_zt if v is not None]
    valid_alpha = [v for v in alpha_h if v is not None]
    valid_rho = [v for v in rho_h if v is not None]

    return {
        'days': days_list,
        'alpha_h': alpha_h,
        'rho_h': rho_h,
        'onset_zt': onset_zt,
        'offset_zt': offset_zt,
        'onset_sd': float(np.std(valid_onsets)) if len(valid_onsets) > 1 else 0.0,
        'offset_sd': float(np.std(valid_offsets)) if len(valid_offsets) > 1 else 0.0,
        'alpha_mean': float(np.mean(valid_alpha)) if valid_alpha else None,
        'alpha_sd': float(np.std(valid_alpha)) if len(valid_alpha) > 1 else 0.0,
        'rho_mean': float(np.mean(valid_rho)) if valid_rho else None,
        'rho_sd': float(np.std(valid_rho)) if len(valid_rho) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Helper: select data for a lighting phase
# ---------------------------------------------------------------------------

def filter_days(
    times: np.ndarray,
    activity: np.ndarray,
    day_start: Optional[int],
    day_end: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, activity) restricted to [day_start, day_end] (inclusive, 1-based)."""
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)
    days = np.floor(times / 24).astype(int) + 1
    mask = np.ones(len(times), dtype=bool)
    if day_start is not None:
        mask &= days >= day_start
    if day_end is not None:
        mask &= days <= day_end
    return times[mask], activity[mask]
