"""
generate_synthetic_independent.py
==================================

Generates a synthetic INDEPENDENT dataset CSV for testing ChronoScope.

INDEPENDENT means: different subjects at each timepoint (biological replicates).
The CSV has NO subject column — multiple rows with the same (time, condition)
are automatically recognised as replicates by the analysis engine.

Output: synthetic_independent_test_data.csv (same folder as this script)

Variables generated
-------------------
  circadian_pure    24 h period, low noise   → clean positive control
  circadian_noisy   24 h period, high noise  → robustness test
  ultradian_12h     12 h period              → period-detection test
  multi_harmonic    24 h + 12 h harmonics    → multi-component test
  arrhythmic        pure noise               → negative control

  --- Preprocessing test variables ---
  trend_drift       24 h rhythm + linear drift (+7 units over 48 h)
                    → enable Detrend to recover the clean rhythm
  with_outliers     24 h rhythm + 3 extreme spikes (rep 0 only)
                    → enable Outlier removal to eliminate them
  high_freq_noise   24 h rhythm + 4 h oscillation superimposed
                    → enable Smoothing (Butterworth cutoff 6 h) to suppress it

Conditions
----------
  control    baseline phase (acrophase at ZT6)
  treatment  phase-shifted +6 h (acrophase at ZT12) and +20 % amplitude

Usage
-----
  python examples/generate_synthetic_independent.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TIMEPOINTS   = np.arange(0, 48, 2, dtype=float)   # 0–46 h, step 2 h
N_REPLICATES = 4                                   # subjects per timepoint
CONDITIONS   = {
    "control":   {"phase_shift": 0.0,           "amp_factor": 1.0},
    "treatment": {"phase_shift": 2 * np.pi / 4, "amp_factor": 1.2},   # +6 h, +20 %
}

PERIOD_24 = 2 * np.pi / 24
PERIOD_12 = 2 * np.pi / 12
PERIOD_4  = 2 * np.pi / 4   # 4 h — used for high-frequency noise component


def cosinor(t, mesor, amp, period_rad, acrophase, noise_sd):
    """Single-component cosinor + Gaussian noise."""
    signal = mesor + amp * np.cos(period_rad * t - acrophase)
    return signal + RNG.normal(0, noise_sd, size=len(t))


def build_variables(t, phase_shift, amp_factor, rep=0):
    """Return dict of variable_name → array of values for one replicate series.

    rep is used only for with_outliers: extreme spikes are injected exclusively
    in rep 0 so that outlier detection operates on a realistic single-replicate
    artefact rather than a systematic shift visible in every replicate.
    """
    phi = phase_shift
    a   = amp_factor

    # ----------------------------------------------------------------
    # Existing variables — keep these first to preserve the RNG sequence
    # ----------------------------------------------------------------
    result = {
        "circadian_pure":  cosinor(t, 10.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.3),
        "circadian_noisy": cosinor(t, 8.0,  1.5 * a, PERIOD_24, np.pi / 4 + phi, 1.2),
        "ultradian_12h":   cosinor(t, 15.0, 1.8 * a, PERIOD_12, np.pi / 2 + phi, 0.5),
        "multi_harmonic":  (
            cosinor(t, 18.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.0)
            + cosinor(t, 0.0, 1.0 * a, PERIOD_12, np.pi / 2 + phi, 0.0)
            + RNG.normal(0, 0.6, size=len(t))
        ),
        "arrhythmic":      RNG.normal(14.0, 1.0, size=len(t)),
    }

    # ----------------------------------------------------------------
    # Preprocessing test variables (appended after existing ones)
    # ----------------------------------------------------------------

    # trend_drift: 24 h rhythm starting at low mesor + linear drift of +0.15/h
    # → after 48 h the baseline rises ~7 units; Detrend (linear or MA 24 h) removes it
    trend_base = cosinor(t, 5.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.3)
    result["trend_drift"] = trend_base + 0.15 * t

    # with_outliers: clean 24 h rhythm; rep 0 carries 3 extreme spikes
    # Spikes: +12 at t=6 h (index 3), −10 at t=22 h (index 11), +15 at t=40 h (index 20)
    outlier_sig = cosinor(t, 10.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.3)
    if rep == 0:
        outlier_sig[ 3] += 12.0   # t= 6 h — spike up
        outlier_sig[11] -= 10.0   # t=22 h — spike down
        outlier_sig[20] += 15.0   # t=40 h — large spike up
    result["with_outliers"] = outlier_sig

    # high_freq_noise: 24 h rhythm + 4 h sinusoidal oscillation (amplitude 1.2)
    # → rapid fluctuations clearly visible; Butterworth (cutoff 6 h) or MA (3 pts) suppresses them
    hf_base = cosinor(t, 10.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.2)
    result["high_freq_noise"] = hf_base + 1.2 * np.sin(PERIOD_4 * t)

    return result


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------
rows = []
for condition, params in CONDITIONS.items():
    for rep in range(N_REPLICATES):
        variables = build_variables(TIMEPOINTS, params["phase_shift"], params["amp_factor"], rep=rep)
        for i, t in enumerate(TIMEPOINTS):
            row = {"time": t, "condition": condition}
            for var_name, values in variables.items():
                row[var_name] = round(values[i], 6)
            rows.append(row)

df = pd.DataFrame(rows)
# Sort by time then condition for readability
df = df.sort_values(["time", "condition"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).parent / "synthetic_independent_test_data.csv"
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Shape: {df.shape}  ({len(TIMEPOINTS)} timepoints × {len(CONDITIONS)} conditions × {N_REPLICATES} replicates)")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst rows:")
print(df.head(8).to_string(index=False))
print(f"\nRows per (time, condition): {df.groupby(['time','condition']).size().unique().tolist()} "
      f"(should be [{N_REPLICATES}])")
print("\nDone. Load synthetic_independent_test_data.csv in ChronoScope.")
print("Expected: Analysis Type = INDEPENDENT (no subject column).")
