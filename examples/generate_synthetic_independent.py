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


def cosinor(t, mesor, amp, period_rad, acrophase, noise_sd):
    """Single-component cosinor + Gaussian noise."""
    signal = mesor + amp * np.cos(period_rad * t - acrophase)
    return signal + RNG.normal(0, noise_sd, size=len(t))


def build_variables(t, phase_shift, amp_factor):
    """Return dict of variable_name → array of values for one replicate series."""
    phi = phase_shift
    a   = amp_factor
    return {
        "circadian_pure": cosinor(t, 10.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.3),
        "circadian_noisy": cosinor(t, 8.0, 1.5 * a, PERIOD_24, np.pi / 4 + phi, 1.2),
        "ultradian_12h": cosinor(t, 15.0, 1.8 * a, PERIOD_12, np.pi / 2 + phi, 0.5),
        "multi_harmonic": (
            cosinor(t, 18.0, 2.0 * a, PERIOD_24, np.pi / 4 + phi, 0.0)
            + cosinor(t, 0.0,  1.0 * a, PERIOD_12, np.pi / 2 + phi, 0.0)
            + RNG.normal(0, 0.6, size=len(t))
        ),
        "arrhythmic": RNG.normal(14.0, 1.0, size=len(t)),
    }


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------
rows = []
for condition, params in CONDITIONS.items():
    for rep in range(N_REPLICATES):
        variables = build_variables(TIMEPOINTS, params["phase_shift"], params["amp_factor"])
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
