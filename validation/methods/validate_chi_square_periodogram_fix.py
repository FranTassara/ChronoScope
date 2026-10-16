#!/usr/bin/env python3
"""
Validation: chi-square periodogram significance and period estimation.

Compares three versions of the chi-square periodogram used by the Locomotor
Activity Analysis module (core.visualization_circadian_metrics):

  OLD      Original implementation (before 2026-09-17), kept verbatim below.
           Its Qp normalisation and its threshold (chi2_crit / n_slots) were
           never derived from the same formula, so the threshold sat roughly
           n_slots-fold too low.
  INTERIM  First fix (2026-09-17): Qp replaced by the "greedy" formula of
           Tackenberg & Hughey (2021, PLOS Comput Biol 17:e1008567), checked
           against spectr::cspgram(). Significance was still decided by
           comparing the largest Qp over the scan with ONE flat chi² line at
           df = median(n_slots) - 1, and tau was the argmax of raw Qp.
  NEW      Current code. Same greedy Qp. tau is the period with the smallest
           per-period chi² p-value, and rhythmicity is decided by a scan-wide
           max-statistic test: the series is permuted in contiguous 2-h
           blocks (1000 permutations, fixed seed), the minimum per-period
           p-value over the whole scan is recorded for each permutation, and
           the recording is called rhythmic when its own minimum p-value falls
           in the lower alpha tail of that null distribution.

Why INTERIM was still wrong (quantified in [1]):
  * Qp at period P follows chi²(n_slots(P) - 1) under the null, and n_slots
    grows with P (36 -> 64 slots over 18-32 h at 30-min bins). A flat line at
    the median df is far too low for long periods: at P = 32 h alone, it is
    exceeded by chance 36% of the time.
  * Even with a df-matched line, taking the best of the ~29 distinct fold
    widths scanned inflates the error rate (look-elsewhere effect).
  * argmax of raw Qp favours long periods, because the null mean of Qp grows
    with P (see [3]).
Block (rather than single-bin) permutation keeps short-range autocorrelation
in the null, which single-bin shuffling destroys; real activity recordings are
autocorrelated (bouts), and single-bin shuffling is anti-conservative on them.

Usage (from project root):
    python validation/methods/validate_chi_square_periodogram_fix.py
Outputs:
    validation/methods/chi_square_periodogram_fix_validation.txt
    validation/methods/chi_square_periodogram_fix_validation.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as _stats

from core.visualization_circadian_metrics import chi_square_periodogram as chi_square_periodogram_new

OUT_DIR = Path(__file__).resolve().parent
TXT_OUT = OUT_DIR / 'chi_square_periodogram_fix_validation.txt'
PNG_OUT = OUT_DIR / 'chi_square_periodogram_fix_validation.png'

# GUI defaults for the Activity Profile analysis
BIN_H = 0.5
P_MIN, P_MAX, P_STEP = 18.0, 32.0, 0.1
ALPHA = 0.05


# ---------------------------------------------------------------------------
# OLD (pre-2026-09-17) implementation, kept here verbatim for comparison only.
# ---------------------------------------------------------------------------
def chi_square_periodogram_old(times, activity, period_min=18.0, period_max=32.0,
                                period_step=0.1, alpha=0.05):
    times = np.asarray(times, dtype=float)
    activity = np.asarray(activity, dtype=float)
    diffs = np.diff(times)
    bin_size_h = float(np.median(diffs[diffs > 0])) if len(diffs) > 0 else 1.0
    N = len(activity)
    grand_mean = np.mean(activity)
    grand_var = np.var(activity, ddof=1)
    if grand_var == 0 or N < 3:
        return {'periods': np.array([]), 'qp': np.array([]), 'tau': None,
                'tau_qp': None, 'significance': None, 'p_values': np.array([]),
                'bin_size_h': bin_size_h, 'n_points': N}

    periods = np.arange(period_min, period_max + period_step * 0.5, period_step)
    qp_values = np.zeros(len(periods))
    for idx, P in enumerate(periods):
        n_slots = max(int(round(P / bin_size_h)), 2)
        slot_indices = (np.floor(times / bin_size_h).astype(int)) % n_slots
        slot_sums = np.zeros(n_slots)
        slot_counts = np.zeros(n_slots, dtype=int)
        for i, s in enumerate(slot_indices):
            slot_sums[s] += activity[i]
            slot_counts[s] += 1
        occupied = slot_counts > 0
        if occupied.sum() < 2:
            continue
        slot_means = slot_sums[occupied] / slot_counts[occupied]
        between_var = np.sum((slot_means - grand_mean) ** 2 * slot_counts[occupied]) / N
        qp_values[idx] = N * between_var / grand_var

    median_n_slots = max(int(round(np.median(periods) / bin_size_h)), 2)
    df = median_n_slots - 1
    chi2_crit = _stats.chi2.ppf(1 - alpha, df=df)
    significance = chi2_crit / N * (N / median_n_slots) if N > 0 else None

    best_idx = int(np.argmax(qp_values))
    tau = float(periods[best_idx])
    tau_qp = float(qp_values[best_idx])
    return {'periods': periods, 'qp': qp_values, 'tau': tau, 'tau_qp': tau_qp,
            'significance': significance, 'bin_size_h': bin_size_h, 'n_points': N}


# ---------------------------------------------------------------------------
# The three decision rules, applied to one recording.
# ---------------------------------------------------------------------------
def evaluate(times, activity, with_old=True):
    new = chi_square_periodogram_new(times, activity, P_MIN, P_MAX, P_STEP, ALPHA)
    # INTERIM = greedy Qp (identical to NEW's qp) + flat median-df line + argmax.
    periods = new['periods']
    median_slots = max(int(round(np.median(periods) / BIN_H)), 2)
    interim_thr = _stats.chi2.ppf(1 - ALPHA, median_slots - 1)
    i_max = int(np.argmax(new['qp']))
    out = {
        'interim_sig': bool(new['qp'][i_max] > interim_thr),
        'interim_tau': float(periods[i_max]),
        'new_sig': bool(new['rhythmic']),
        'new_tau': new['tau'],
    }
    if with_old:
        old = chi_square_periodogram_old(times, activity, P_MIN, P_MAX, P_STEP, ALPHA)
        out['old_sig'] = bool(old['tau_qp'] > old['significance'])
    return out


# ---------------------------------------------------------------------------
# Synthetic data (30-min bins, as produced by the GUI's resampling step)
# ---------------------------------------------------------------------------
def _times(n_days):
    return np.arange(int(round(n_days * 24 / BIN_H))) * BIN_H


def _ar1(n, phi, rng):
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi ** 2) * e[i]
    return x


def white_noise(n_days, rng):
    t = _times(n_days)
    return t, rng.normal(10, 2.0, len(t))


def ar1_noise(n_days, rng, phi):
    t = _times(n_days)
    return t, 10 + 2.0 * _ar1(len(t), phi, rng)


def dam_counts(n_days, rng, amp=0.0, period=24.0):
    """DAM-like beam-crossing counts per 30-min bin: zero-heavy negative
    binomial counts whose log-rate is an AR(1) process (activity bouts),
    optionally modulated by a rectified rhythm."""
    t = _times(n_days)
    log_rate = np.log(3.0) + 1.0 * _ar1(len(t), 0.6, rng)
    if amp:
        log_rate = log_rate + amp * (np.maximum(0, np.sin(2 * np.pi * t / period)) - 0.32)
    mu = np.exp(log_rate)
    size = 1.0  # NB dispersion: many zero bins
    return t, rng.negative_binomial(size, size / (size + mu)).astype(float)


def rhythmic(n_days, rng, amp, period=24.0):
    t = _times(n_days)
    a = 10 + amp * np.maximum(0, np.sin(2 * np.pi * t / period)) + rng.normal(0, 2.0, len(t))
    return t, a


def monte_carlo(gen, n_trials, seed, with_old=False, true_period=None):
    rng = np.random.default_rng(seed)
    keys = ['interim_sig', 'new_sig'] + (['old_sig'] if with_old else [])
    hits = dict.fromkeys(keys, 0)
    taus_i, taus_n = [], []
    for _ in range(n_trials):
        r = evaluate(*gen(rng), with_old=with_old)
        for k in keys:
            hits[k] += r[k]
        taus_i.append(r['interim_tau'])
        taus_n.append(r['new_tau'])
    rates = {k: 100.0 * v / n_trials for k, v in hits.items()}
    return rates, np.array(taus_i), np.array(taus_n)


def binom_ci(p_pct, n):
    """Wilson 95% interval, in percent."""
    p, z = p_pct / 100.0, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def main():
    lines = []
    w = lines.append
    w("Chi-square periodogram: significance calibration and period estimation")
    w("Greedy Qp: Tackenberg & Hughey (2021) PLOS Comput Biol 17:e1008567")
    w("Scan-wide threshold: block-permutation max-statistic test")
    w("=" * 78)
    w(f"Settings (GUI defaults): periods {P_MIN:g}-{P_MAX:g} h, step {P_STEP:g} h, "
      f"{BIN_H*60:.0f}-min bins, alpha = {ALPHA:g}")
    n_slots = np.unique([max(int(round(P / BIN_H)), 2) for P in
                         np.arange(P_MIN, P_MAX + P_STEP / 2, P_STEP)])
    w(f"  {len(np.arange(P_MIN, P_MAX + P_STEP / 2, P_STEP))} periods tested, "
      f"{len(n_slots)} distinct fold widths (n_slots {n_slots.min()}-{n_slots.max()}, "
      f"df {n_slots.min()-1}-{n_slots.max()-1})")
    w("  NEW: 1000 permutations of contiguous 2-h blocks, seed 0")
    w("  OLD     = original code (before 2026-09-17)")
    w("  INTERIM = greedy Qp + one flat chi² line at median df + tau = argmax Qp")
    w("  NEW     = greedy Qp + scan-wide block-permutation threshold + tau = argmin p")

    # -----------------------------------------------------------------
    # [1] False-positive rate on arrhythmic recordings
    # -----------------------------------------------------------------
    w("\n[1] False-positive rate on arrhythmic recordings (nominal alpha = 5%)")
    w("-" * 78)
    w("  Rate of recordings called rhythmic; [ ] = Wilson 95% CI for NEW.")
    nt = 1000
    for n_days in [3, 5, 7, 10]:
        rates, ti, tn = monte_carlo(lambda g, d=n_days: white_noise(d, g), nt,
                                    seed=n_days, with_old=True)
        lo, hi = binom_ci(rates['new_sig'], nt)
        w(f"  white noise, {n_days:>2} d (n={nt}): OLD {rates['old_sig']:5.1f}%  "
          f"INTERIM {rates['interim_sig']:5.1f}%  NEW {rates['new_sig']:4.1f}% [{lo:.1f}-{hi:.1f}]")
    nt = 500
    for label, gen, seed in [
        ("AR(1) phi=0.5, 10 d", lambda g: ar1_noise(10, g, 0.5), 21),
        ("AR(1) phi=0.8, 10 d", lambda g: ar1_noise(10, g, 0.8), 22),
        ("DAM-like counts, 5 d", lambda g: dam_counts(5, g), 23),
        ("DAM-like counts, 10 d", lambda g: dam_counts(10, g), 24),
    ]:
        rates, _, _ = monte_carlo(gen, nt, seed=seed)
        lo, hi = binom_ci(rates['new_sig'], nt)
        w(f"  {label:<22} (n={nt}): INTERIM {rates['interim_sig']:5.1f}%  "
          f"NEW {rates['new_sig']:4.1f}% [{lo:.1f}-{hi:.1f}]")
    w("  DAM-like counts = zero-heavy negative-binomial counts with an AR(1)")
    w("  log-rate (activity bouts), no rhythm.")

    # -----------------------------------------------------------------
    # [2] Power and period accuracy on rhythmic recordings
    # -----------------------------------------------------------------
    w("\n[2] Power and period accuracy on rhythmic recordings (true tau = 23.6 h)")
    w("-" * 78)
    w("  Gaussian: 10 + A*max(0, sin) + N(0, 2^2). Power = % called rhythmic.")
    w("  |tau err| = median absolute error of the reported tau over all trials.")
    nt = 300
    for amp in [1.0, 1.5, 2.0, 3.0]:
        for n_days in [5, 10]:
            rates, ti, tn = monte_carlo(lambda g, a=amp, d=n_days: rhythmic(d, g, a, 23.6),
                                        nt, seed=int(amp * 100) + n_days)
            w(f"  A={amp:<3g} {n_days:>2} d (n={nt}): power INTERIM {rates['interim_sig']:5.1f}%"
              f"  NEW {rates['new_sig']:5.1f}%   |tau err| INTERIM "
              f"{np.median(np.abs(ti - 23.6)):4.2f} h  NEW {np.median(np.abs(tn - 23.6)):4.2f} h")
    for amp in [0.5, 1.0, 2.0, 3.0]:
        for n_days in [5, 10]:
            rates, ti, tn = monte_carlo(lambda g, a=amp, d=n_days: dam_counts(d, g, a, 23.6),
                                        nt, seed=500 + int(amp * 10) + n_days)
            w(f"  DAM-like, rhythm {amp:g}, {n_days:>2} d (n={nt}): power INTERIM "
              f"{rates['interim_sig']:5.1f}%  NEW {rates['new_sig']:5.1f}%   |tau err| "
              f"INTERIM {np.median(np.abs(ti - 23.6)):4.2f} h  NEW {np.median(np.abs(tn - 23.6)):4.2f} h")
    w("  INTERIM's power figures are not comparable: it also calls ~97% of")
    w("  arrhythmic recordings rhythmic (see [1]).")

    # -----------------------------------------------------------------
    # [3] Period bias on noise
    # -----------------------------------------------------------------
    w("\n[3] Where tau lands on pure noise (white noise, 10 d, n=1000)")
    w("-" * 78)
    _, ti, tn = monte_carlo(lambda g: white_noise(10, g), 1000, seed=77)
    w(f"  INTERIM (argmax Qp):  median tau {np.median(ti):.1f} h, "
      f"{np.mean(ti >= 30) * 100:.0f}% of trials at tau >= 30 h")
    w(f"  NEW (argmin p):       median tau {np.median(tn):.1f} h, "
      f"{np.mean(tn >= 30) * 100:.0f}% of trials at tau >= 30 h")
    w(f"  (A uniform spread over {P_MIN:g}-{P_MAX:g} h would put "
      f"{(P_MAX - 30) / (P_MAX - P_MIN) * 100:.0f}% at >= 30 h.)")
    w("  The null mean of Qp is its df = n_slots - 1, which grows with P, so")
    w("  the raw-Qp argmax drifts to the longest periods scanned whenever the")
    w("  rhythm is weak. Comparing periods on the p-value scale removes this.")

    # -----------------------------------------------------------------
    # [4] Periodogram shape: OLD vs NEW Qp(P) on clean rhythmic series
    # -----------------------------------------------------------------
    w("\n[4] Periodogram shape and period recovery on a clean 24-h rhythm")
    w("-" * 78)
    record_lengths_days = [5, 6.3, 7, 9.7, 12]
    fig, axes = plt.subplots(len(record_lengths_days), 1,
                             figsize=(7, 2.4 * len(record_lengths_days)), sharex=True)
    for ax, n_days in zip(axes, record_lengths_days):
        rng = np.random.default_rng(42)
        t = _times(n_days)
        a = np.clip(10 + 8 * np.maximum(0, np.sin(2 * np.pi * t / 24.0))
                    + rng.normal(0, 1.5, len(t)), 0, None)
        old = chi_square_periodogram_old(t, a, P_MIN, P_MAX, P_STEP)
        new = chi_square_periodogram_new(t, a, P_MIN, P_MAX, P_STEP, ALPHA)
        w(f"  {n_days:>4} d (N={len(t)}): OLD tau={old['tau']:.2f} h  NEW tau={new['tau']:.2f} h"
          f"  NEW rhythmic={new['rhythmic']}  global p={new['global_p']:.4f}")
        ax.plot(old['periods'], old['qp'], color='#D55E00', lw=1.0, label='OLD Qp')
        ax.plot(new['periods'], new['qp'], color='#0072B2', lw=1.2, label='NEW Qp')
        ax.axhline(old['significance'], color='#D55E00', lw=0.6, ls='--', alpha=0.7,
                   label='OLD threshold')
        ax.plot(new['periods'], new['threshold_curve'], color='#0072B2', lw=0.8, ls=':',
                label='NEW scan-wide threshold')
        ax.axvline(24.0, color='#888888', lw=0.6, ls=':')
        ax.set_ylabel('Qp')
        ax.set_title(f'{n_days} days (N={len(t)})', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('Test period (h)')
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=200, bbox_inches='tight')
    w("  OLD and NEW Qp(P) curves are close in shape: ChronoScope's original")
    w("  modulo-based slot assignment already used every data point rather than")
    w("  discarding a partial final cycle, so the discontinuities Tackenberg &")
    w("  Hughey describe for the classic formula were not prominent here. Both")
    w("  recover tau = 24 h; NEW reports the period of the best fold width")
    w("  (n_slots * bin), i.e. tau is resolved to the 30-min bin size.")

    w("\n" + "=" * 78)
    w("Summary")
    w("  * OLD called 100% of pure-noise recordings rhythmic (threshold ~n_slots-")
    w("    fold too low).")
    w("  * INTERIM (the 2026-09-17 fix) still called ~95-97% of them rhythmic.")
    w("    The earlier version of this report described that rate as 'close to")
    w("    the nominal alpha'; that was wrong. Most of the excess came from")
    w("    comparing every period with one flat line at the median df; the rest")
    w("    from taking the best of many periods (look-elsewhere effect).")
    w("  * NEW holds the false-positive rate at the nominal 5% on white noise")
    w("    and on DAM-like count noise (see [1]). It is NOT fully calibrated on")
    w("    strongly autocorrelated Gaussian noise: ~10% at AR(1) phi = 0.5 and")
    w("    ~12% at phi = 0.8 (30-min bins), because dependence that outlasts the")
    w("    2-h blocks is partly broken by the permutation. This is still an")
    w("    8-10-fold reduction from INTERIM on the same series.")
    w("  * NEW reports tau without the long-period bias of the raw-Qp argmax")
    w("    (see [3]) and with smaller period error on rhythmic series (see [2]).")

    report = "\n".join(lines)
    print(report)
    TXT_OUT.write_text(report, encoding='utf-8')
    print(f"\nSaved -> {TXT_OUT}")
    print(f"Saved -> {PNG_OUT}")


if __name__ == '__main__':
    main()
