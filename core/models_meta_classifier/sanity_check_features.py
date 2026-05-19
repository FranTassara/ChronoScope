"""
Feature Extraction Sanity Check
================================
10 rhythmic (clean cosine, 24h, high SNR) + 10 arrhythmic (white noise).
Runs extract_features() on each and prints per-feature statistics.

Flags:
  - CONST   : std == 0 (identically constant)
  - NEAR-0  : std < 1e-6 (almost constant)
  - ALL-NaN : every value is NaN (method always fails)
  - WARN    : suspicious range or near-constant after excluding NaN

Run from project root:
    python sanity_check_features.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — must mirror train_consensus_model.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extraction import extract_features, FEATURE_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(seed=0)

N_TIMEPOINTS = 24          # 24 points × 2h = 48h span (2 full circadian cycles)
SAMPLING_H   = 2.0
TIME         = np.arange(0, N_TIMEPOINTS * SAMPLING_H, SAMPLING_H, dtype=float)
PERIOD       = 24.0
AMPLITUDE    = 2.0
MESOR        = 10.0
NOISE_STD    = 0.3         # SNR ≈ 6.7  (clearly rhythmic)


def make_rhythmic(i: int) -> tuple:
    phase = RNG.uniform(0, 2 * np.pi)
    values = MESOR + AMPLITUDE * np.cos(2 * np.pi * TIME / PERIOD - phase)
    values += RNG.normal(0, NOISE_STD, size=len(TIME))
    return TIME.copy(), values, f"rhythmic_{i:02d}"


def make_arrhythmic(i: int) -> tuple:
    values = MESOR + RNG.normal(0, 1.5, size=len(TIME))   # pure white noise
    return TIME.copy(), values, f"arrhythmic_{i:02d}"


# ---------------------------------------------------------------------------
# Collect features
# ---------------------------------------------------------------------------
records = []       # list of dicts: {label, name, **features}
print("Running feature extraction on 20 series...")

for i in range(10):
    t, v, name = make_rhythmic(i)
    feat = extract_features(t, v)
    feat["_label"] = 1
    feat["_name"]  = name
    records.append(feat)
    print(f"  {name} done")

for i in range(10):
    t, v, name = make_arrhythmic(i)
    feat = extract_features(t, v)
    feat["_label"] = 0
    feat["_name"]  = name
    records.append(feat)
    print(f"  {name} done")

print()

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
rhythmic_records    = [r for r in records if r["_label"] == 1]
arrhythmic_records  = [r for r in records if r["_label"] == 0]

BANNER = "=" * 72
SEP    = "-" * 72

print(BANNER)
print("FEATURE DISTRIBUTION SANITY CHECK")
print(BANNER)
print(f"{'Feature':<25} {'Group':<12} {'n':>3} {'mean':>10} {'std':>10} "
      f"{'min':>10} {'max':>10}  {'Status'}")
print(SEP)

issues = []

for feat_name in FEATURE_NAMES:
    for group_name, grp in [("rhythmic", rhythmic_records),
                            ("arrhythmic", arrhythmic_records)]:
        vals = np.array([r[feat_name] for r in grp], dtype=float)
        valid = vals[~np.isnan(vals)]
        n_valid = len(valid)

        if n_valid == 0:
            status = ">>> ALL-NaN <<<  BUG CANDIDATE"
            issues.append(f"{feat_name} [{group_name}]: ALL values are NaN")
            print(f"  {feat_name:<23} {group_name:<12} {n_valid:>3} {'NaN':>10} {'NaN':>10} "
                  f"{'NaN':>10} {'NaN':>10}  {status}")
            continue

        m, s = float(np.mean(valid)), float(np.std(valid))
        mn, mx = float(np.min(valid)), float(np.max(valid))

        if s == 0.0:
            status = ">>> CONST <<<    BUG CANDIDATE"
            issues.append(f"{feat_name} [{group_name}]: std=0 (constant = {m:.6g})")
        elif s < 1e-6:
            status = ">>> NEAR-0 <<<   BUG CANDIDATE"
            issues.append(f"{feat_name} [{group_name}]: std={s:.2e} (almost constant)")
        elif n_valid < len(grp) // 2:
            status = f"WARN: {len(grp)-n_valid}/{len(grp)} NaN"
            issues.append(f"{feat_name} [{group_name}]: {len(grp)-n_valid}/{len(grp)} values are NaN")
        else:
            status = "OK"

        print(f"  {feat_name:<23} {group_name:<12} {n_valid:>3} {m:>10.4f} {s:>10.4f} "
              f"{mn:>10.4f} {mx:>10.4f}  {status}")

print(SEP)

# ---------------------------------------------------------------------------
# Highlighted audit of the two suspect features
# ---------------------------------------------------------------------------
print()
print(BANNER)
print("DETAILED AUDIT: f24_score  &  harmonic_p_value")
print(BANNER)

for feat_name in ("f24_score", "harmonic_p_value"):
    print(f"\n  {feat_name}:")
    for group_name, grp in [("rhythmic", rhythmic_records),
                            ("arrhythmic", arrhythmic_records)]:
        vals = np.array([r[feat_name] for r in grp], dtype=float)
        print(f"    {group_name}:  {[round(v, 4) if not np.isnan(v) else 'NaN' for v in vals]}")

print()

# ---------------------------------------------------------------------------
# Summary verdict
# ---------------------------------------------------------------------------
print(BANNER)
if issues:
    print(f"ISSUES FOUND ({len(issues)}) — fix before retraining:")
    for issue in issues:
        print(f"  • {issue}")
    print()
    print("  Do NOT retrain until all BUG CANDIDATEs are resolved.")
else:
    print("ALL 18 FEATURES LOOK HEALTHY across both groups.")
    print("No constant, near-constant, or all-NaN features detected.")
    print()
    print("Verdict: SAFE TO RETRAIN.")
print(BANNER)
