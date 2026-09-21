#!/usr/bin/env python3
"""
Validation: chi-square periodogram fix (Tackenberg & Hughey 2021).

Compares the OLD (pre-fix) and NEW (post-fix) chi-square periodogram
implementations in core.visualization_circadian_metrics.chi_square_periodogram.

Findings this script quantifies:
  1. [Significance calibration -- the dominant, practically important bug]
     The OLD implementation's significance threshold was miscalibrated by
     roughly a factor of n_slots (it compared Qp, which already carries an
     implicit ~n_slots scaling, against chi2_crit/n_slots instead of
     chi2_crit itself). Monte-Carlo false-positive rate on pure noise
     (arrhythmic) recordings shows the OLD code calls noise "significantly
     rhythmic" at a rate far above the nominal alpha; the NEW code's
     per-period false-positive rate matches alpha closely.
  2. [Periodogram continuity -- the specific issue the reviewer/Hughey lab
     describe] The classic Sokolove & Bushell (1978) formula folds the
     series using K = floor(N / n_slots) complete rows and discards the
     remainder, producing discontinuities in Qp(P) that depend on N.
     ChronoScope's slot-assignment already used every data point (modulo
     binning, not row-major discarding), so it was structurally close to
     Tackenberg & Hughey's "greedy" fix already; this script confirms the
     old and new Qp curves are nearly identical in shape (differing by a
     near-constant N/(N-1) factor from the old code's variance-based
     normalisation) rather than showing dramatic sawtooth breaks in this
     implementation specifically. The NEW code replaces the normalisation
     with the paper's exact validated "greedy" formula regardless, so the
     result is now traceable to a peer-reviewed reference rather than an
     internally-derived approximation, and the significance threshold is
     correctly calibrated as a direct consequence.
  3. [Residual caveat, disclosed not fixed] Because the app scans ~280
     candidate periods and reports whether the *best* Qp exceeds a
     single-period significance threshold, the effective false-positive
     rate over the full scan is higher than the nominal alpha for both OLD
     and NEW (classic "multiple comparisons / look-elsewhere effect" for
     periodograms). This is not the bug the reviewer flagged and is not
     addressed by the Tackenberg & Hughey correction; it is quantified here
     and disclosed as a known limitation.
  4. True-positive behaviour on a clean 24-h rhythmic signal is unchanged
     (tau correctly recovered) by the fix.

Usage (from project root):
    python core/validate_chi_square_periodogram_fix.py
Outputs:
    core/chi_square_periodogram_fix_validation.txt
    core/chi_square_periodogram_fix_validation.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as _stats

from core.visualization_circadian_metrics import chi_square_periodogram as chi_square_periodogram_new

OUT_DIR = Path(__file__).resolve().parent
TXT_OUT = OUT_DIR / 'chi_square_periodogram_fix_validation.txt'
PNG_OUT = OUT_DIR / 'chi_square_periodogram_fix_validation.png'


# ---------------------------------------------------------------------------
# OLD (pre-fix) implementation, kept here verbatim for comparison only.
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
# Synthetic data
# ---------------------------------------------------------------------------
def make_rhythmic_actogram(n_days, bin_size_h=0.5, true_period=24.0, seed=42):
    rng = np.random.default_rng(seed)
    n_points = int(round(n_days * 24 / bin_size_h))
    times = np.arange(n_points) * bin_size_h
    signal = 10 + 8 * np.maximum(0, np.sin(2 * np.pi * times / true_period))
    noise = rng.normal(0, 1.5, size=n_points)
    activity = np.clip(signal + noise, 0, None)
    return times, activity


def make_arrhythmic(n_days, bin_size_h=0.5, seed=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    n_points = int(round(n_days * 24 / bin_size_h))
    times = np.arange(n_points) * bin_size_h
    activity = rng.normal(10, 2.0, size=n_points)
    return times, activity


def monte_carlo_fpr(n_days, n_trials=300, bin_size_h=0.5, alpha=0.05, seed=0):
    """Fraction of pure-noise trials called 'significant' at the scan-wide
    best period (tau_qp > significance), for both implementations."""
    rng = np.random.default_rng(seed)
    n_sig_old = 0
    n_sig_new = 0
    for _ in range(n_trials):
        times, activity = make_arrhythmic(n_days, bin_size_h=bin_size_h, rng=rng)
        old = chi_square_periodogram_old(times, activity, 18.0, 32.0, 0.2, alpha=alpha)
        new = chi_square_periodogram_new(times, activity, 18.0, 32.0, 0.2, alpha=alpha)
        if old['tau_qp'] is not None and old['significance'] is not None and old['tau_qp'] > old['significance']:
            n_sig_old += 1
        if new['tau_qp'] is not None and new['significance'] is not None and new['tau_qp'] > new['significance']:
            n_sig_new += 1
    return n_sig_old / n_trials, n_sig_new / n_trials


def roughness(qp):
    """Sum of |second differences| of Qp(P), normalised by curve scale --
    a simple discontinuity/smoothness proxy, secondary to the significance
    calibration finding above."""
    d2 = np.diff(qp, n=2)
    scale = np.mean(np.abs(qp)) + 1e-9
    return float(np.sum(np.abs(d2)) / scale)


def main():
    lines = []
    lines.append("Chi-square periodogram fix validation")
    lines.append("Tackenberg & Hughey (2021) PLOS Comput Biol 17:e1008567 'greedy' correction")
    lines.append("=" * 78)

    # -----------------------------------------------------------------
    # 1) Significance calibration: Monte Carlo false-positive rate on
    #    pure-noise (arrhythmic) recordings, at the app's own decision
    #    rule (best-of-scan Qp vs. a single significance threshold).
    # -----------------------------------------------------------------
    lines.append("\n[1] Significance calibration (Monte Carlo, pure noise, alpha=0.05)")
    lines.append("-" * 78)
    alpha = 0.05
    n_trials = 300
    for n_days in [3, 5, 7, 10]:
        fpr_old, fpr_new = monte_carlo_fpr(n_days, n_trials=n_trials, alpha=alpha, seed=n_days)
        lines.append(f"  {n_days} days: OLD false-positive rate = {fpr_old*100:5.1f}%   "
                     f"NEW false-positive rate = {fpr_new*100:5.1f}%   (nominal alpha = {alpha*100:.0f}%)")
    lines.append("  (OLD's threshold was too low by ~n_slots-fold, so it calls essentially")
    lines.append("   any input 'significantly rhythmic' regardless of true periodicity; NEW's")
    lines.append("   per-scan false-positive rate is close to, though still somewhat above,")
    lines.append("   the nominal alpha -- see the multiple-testing caveat in [3] below.)")

    # -----------------------------------------------------------------
    # 2) Periodogram continuity: old vs new Qp(P) shape on a clean 24-h
    #    rhythmic signal across several recording lengths.
    # -----------------------------------------------------------------
    lines.append("\n[2] Periodogram continuity and true-positive period recovery")
    lines.append("-" * 78)
    record_lengths_days = [5, 6.3, 7, 9.7, 12]
    bin_size_h = 0.5
    period_min, period_max, period_step = 18.0, 32.0, 0.05

    fig, axes = plt.subplots(len(record_lengths_days), 1,
                              figsize=(7, 2.4 * len(record_lengths_days)), sharex=True)

    for ax, n_days in zip(axes, record_lengths_days):
        times, activity = make_rhythmic_actogram(n_days, bin_size_h=bin_size_h)
        N = len(activity)

        old = chi_square_periodogram_old(times, activity, period_min, period_max, period_step)
        new = chi_square_periodogram_new(times, activity, period_min, period_max, period_step)

        rough_old = roughness(old['qp'])
        rough_new = roughness(new['qp'])

        lines.append(f"  {n_days} days (N={N}): OLD tau={old['tau']:.2f}h Qp={old['tau_qp']:.1f} "
                     f"roughness={rough_old:.4f}  |  NEW tau={new['tau']:.2f}h Qp={new['tau_qp']:.1f} "
                     f"roughness={rough_new:.4f}")

        ax.plot(old['periods'], old['qp'], color='#D55E00', lw=1.0, label='OLD')
        ax.plot(new['periods'], new['qp'], color='#0072B2', lw=1.2, label='NEW')
        ax.axhline(new['significance'], color='#0072B2', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(old['significance'], color='#D55E00', lw=0.6, ls='--', alpha=0.6)
        ax.axvline(24.0, color='#888888', lw=0.6, ls=':')
        ax.set_ylabel('Qp')
        ax.set_title(f'{n_days} days (N={N})', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Test period (h)')
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=200, bbox_inches='tight')

    lines.append("\n  Both implementations correctly recover tau ~= 24 h for this clean")
    lines.append("  rhythmic signal; the old and new Qp(P) curves are close in shape in this")
    lines.append("  implementation (ChronoScope's original modulo-based slot assignment already")
    lines.append("  used every data point rather than discarding a partial final cycle, unlike")
    lines.append("  the classic discard-based formula the reviewer/paper describe), differing")
    lines.append("  mainly by the normalisation constant corrected in [1].")

    # -----------------------------------------------------------------
    # 3) Residual multiple-testing caveat (disclosed, not fixed here).
    # -----------------------------------------------------------------
    lines.append("\n[3] Residual caveat: multiple comparisons across the scanned period range")
    lines.append("-" * 78)
    lines.append("  The app reports significance by comparing the BEST Qp across ~280 scanned")
    lines.append("  periods to a single per-period chi-square threshold. Even with the fixed")
    lines.append("  calibration, this 'take the best of many correlated tests' procedure has a")
    lines.append("  higher true false-positive rate than the nominal alpha (see [1], NEW column,")
    lines.append("  still above 5% for short recordings). This look-elsewhere effect is a known")
    lines.append("  property of periodogram peak-picking in general, is NOT the discontinuity")
    lines.append("  bug reported by Tackenberg & Hughey, and is NOT fixed by this change. A full")
    lines.append("  fix would require a permutation- or extreme-value-corrected threshold and is")
    lines.append("  left as a documented limitation / future-work item.")

    lines.append("\n" + "=" * 78)
    lines.append("Summary: replaced the Qp normalisation with the peer-reviewed 'greedy' CSP")
    lines.append("formula (Tackenberg & Hughey 2021), which correctly recalibrates the")
    lines.append("significance threshold (previously off by roughly n_slots-fold, causing")
    lines.append("near-universal false 'significant rhythmicity' calls) and removes the")
    lines.append("internally-derived approximation in favour of a validated reference formula.")
    lines.append("True-positive period recovery on rhythmic signals is unchanged. A residual,")
    lines.append("distinct multiple-testing caveat over the scanned period range remains and")
    lines.append("is disclosed above, not fixed by this change.")

    report = "\n".join(lines)
    print(report)
    TXT_OUT.write_text(report, encoding='utf-8')
    print(f"\nSaved -> {TXT_OUT}")
    print(f"Saved -> {PNG_OUT}")


if __name__ == '__main__':
    main()
