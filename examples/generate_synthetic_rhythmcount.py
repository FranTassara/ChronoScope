"""
generate_synthetic_rhythmcount.py
==================================

Generates a synthetic COUNT dataset CSV for testing ChronoScope's RhythmCount module.

RhythmCount fits cosinor models using GLMs for count-distributed data (non-negative
integers): Poisson, Negative Binomial, Zero-Inflated Poisson, and Zero-Inflated NB.
This dataset provides variables designed to exercise each of those distributions,
plus period-detection and negative-control tests.

Output: synthetic_rhythmcount_test_data.csv (same folder as this script)

Variables generated
-------------------
  poisson_circadian        24 h, equidispersed Poisson counts → best fit: Poisson/GP
  negbinom_circadian       24 h, overdispersed NB counts      → best fit: NegBin or ZINB
  zero_inflated_circadian  24 h, sparse with excess zeros      → best fit: ZIP or ZINB
  ultradian_12h_counts     12 h, Poisson counts               → period-detection test
  arrhythmic_counts        no rhythm, Poisson noise            → negative control

Generation model (log-link GLM, same link used by RhythmCount internally)
--------------------------------------------------------------------------
  log(lambda(t)) = log(mesor) + amp_log * cos(2π/period * t − acrophase)
  Y ~ Poisson(lambda(t))   or   Y ~ NB(mu=lambda(t), dispersion)

Conditions
----------
  control    baseline phase (acrophase at ZT6, 2π/24 * 6 = π/4)
  treatment  phase-shifted +6 h (acrophase at ZT12) and +30 % amplitude

Usage
-----
  python examples/generate_synthetic_rhythmcount.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=123)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TIMEPOINTS   = np.arange(0, 48, 2, dtype=float)   # 0–46 h, step 2 h (24 points)
N_REPLICATES = 6                                   # observations per (timepoint, condition)
CONDITIONS   = {
    "control":   {"phase_shift": 0.0,           "amp_factor": 1.0},
    "treatment": {"phase_shift": 2 * np.pi / 4, "amp_factor": 1.3},   # +6 h, +30 %
}

PERIOD_24 = 2 * np.pi / 24
PERIOD_12 = 2 * np.pi / 12


# ---------------------------------------------------------------------------
# Count generators (all use log-link so the scale matches RhythmCount's GLMs)
# ---------------------------------------------------------------------------

def _log_lambda(t, mesor, amp_log, period_rad, acrophase):
    return np.log(mesor) + amp_log * np.cos(period_rad * t - acrophase)


def poisson_counts(t, mesor, amp_log, period_rad, acrophase):
    """Equidispersed Poisson counts (variance == mean)."""
    lam = np.exp(_log_lambda(t, mesor, amp_log, period_rad, acrophase))
    return RNG.poisson(lam).astype(int)


def negbinom_counts(t, mesor, amp_log, period_rad, acrophase, dispersion=4.0):
    """Overdispersed Negative Binomial counts (variance > mean).

    Parameterisation: NB(n=dispersion, p=dispersion/(dispersion+mu)).
    Lower dispersion → more overdispersion relative to Poisson.
    """
    mu = np.exp(_log_lambda(t, mesor, amp_log, period_rad, acrophase))
    p  = dispersion / (dispersion + mu)
    return RNG.negative_binomial(dispersion, p).astype(int)


def zip_counts(t, mesor, amp_log, period_rad, acrophase, zero_prob=0.40):
    """Zero-Inflated Poisson counts (excess structural zeros + Poisson signal)."""
    lam    = np.exp(_log_lambda(t, mesor, amp_log, period_rad, acrophase))
    counts = RNG.poisson(lam).astype(int)
    mask   = RNG.random(size=len(t)) < zero_prob   # structural zeros
    return np.where(mask, 0, counts).astype(int)


# ---------------------------------------------------------------------------
# Variable builder
# ---------------------------------------------------------------------------

def build_count_variables(t, phase_shift, amp_factor):
    """Return dict of variable → integer array for one replicate series."""
    phi = phase_shift
    a   = amp_factor
    return {
        # 24 h, Poisson — equidispersed, high signal (e^0.6 ≈ 1.8x fold)
        "poisson_circadian": poisson_counts(
            t, mesor=20, amp_log=0.6 * a, period_rad=PERIOD_24,
            acrophase=np.pi / 4 + phi,
        ),
        # 24 h, Negative Binomial — overdispersed relative to Poisson
        "negbinom_circadian": negbinom_counts(
            t, mesor=15, amp_log=0.5 * a, period_rad=PERIOD_24,
            acrophase=np.pi / 4 + phi, dispersion=4.0,
        ),
        # 24 h, Zero-Inflated Poisson — 40 % structural zeros (sparse RNA-seq-like)
        "zero_inflated_circadian": zip_counts(
            t, mesor=10, amp_log=0.7 * a, period_rad=PERIOD_24,
            acrophase=np.pi / 3 + phi, zero_prob=0.40,
        ),
        # 12 h ultradian, Poisson — period-detection test
        "ultradian_12h_counts": poisson_counts(
            t, mesor=12, amp_log=0.5 * a, period_rad=PERIOD_12,
            acrophase=np.pi / 2 + phi,
        ),
        # RNA-seq-like NB counts: mesor ~200 CPM, dispersion=8 (moderate overdispersion)
        # Fold change at peak vs trough ≈ e^(2*0.5) ≈ 2.7x, typical of circadian genes
        "rnaseq_like": negbinom_counts(
            t, mesor=200, amp_log=0.5 * a, period_rad=PERIOD_24,
            acrophase=np.pi / 4 + phi, dispersion=8.0,
        ),
        # No rhythm — Poisson noise only (negative control, mean ≈ 14)
        "arrhythmic_counts": RNG.poisson(14, size=len(t)).astype(int),
    }


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------
rows = []
for condition, params in CONDITIONS.items():
    for _rep in range(N_REPLICATES):
        variables = build_count_variables(
            TIMEPOINTS, params["phase_shift"], params["amp_factor"]
        )
        for i, t_val in enumerate(TIMEPOINTS):
            row = {"time": t_val, "condition": condition}
            for var_name, values in variables.items():
                row[var_name] = int(values[i])
            rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(["time", "condition"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).parent / "synthetic_rhythmcount_test_data.csv"
df.to_csv(out_path, index=False)

print(f"Saved:  {out_path}")
print(f"Shape:  {df.shape}  "
      f"({len(TIMEPOINTS)} timepoints × {len(CONDITIONS)} conditions × {N_REPLICATES} replicates)")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst rows:")
print(df.head(8).to_string(index=False))
print(f"\nRows per (time, condition): "
      f"{df.groupby(['time','condition']).size().unique().tolist()} "
      f"(should be [{N_REPLICATES}])")
print("\nDescriptive stats (count variables):")
count_cols = [c for c in df.columns if c not in ("time", "condition")]
print(df[count_cols].describe().round(1).to_string())
print("\nDone. Load synthetic_rhythmcount_test_data.csv in ChronoScope.")
print("Expected: Analysis Type = RHYTHMCOUNT (integer counts, no subject column).")
