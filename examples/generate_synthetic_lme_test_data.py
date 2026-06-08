"""
generate_synthetic_lme_test_data.py
=====================================

Generates a dataset that covers ALL functionalities of the LME method in
ChronoScope:

  1. Random intercept (core LME strength)
  2. Fixed effects / covariates (e.g., genotype)
  3. Rhythm detection vs. arrhythmic rejection

Variables
---------
  Variable                    | Rhythm   | Between-subj var | Notes
  ----------------------------|----------|------------------|----------------------------
  rhythm_high_between_var     | YES (24h)| HIGH (SD~1.75)   | Core LME demo
  rhythm_low_between_var      | YES (24h)| LOW  (SD~0.58)   | LME vs OLS comparison
  arrhythmic_high_between_var | NO       | HIGH (SD~1.75)   | No false positives
  arrhythmic_low_between_var  | NO       | LOW  (SD~0.58)   | Negative control
  rhythm_genotype_effect      | YES (24h)| driven by geno   | Fixed effects demo

Column `genotype` (WT / KO)
---------------------------
  subject1-3 = WT (baseline 10)
  subject4-6 = KO (baseline 16, +6 units above WT)

  `rhythm_genotype_effect` is designed so that most between-subject variance
  comes from genotype, not from individual random variation. This means:

  - Analyzed WITHOUT genotype as fixed effect:
      LME still detects the rhythm (random intercepts absorb the WT/KO gap),
      but random_effect_var is large (~9) and AIC is higher.

  - Analyzed WITH genotype as fixed effect:
      The genotype term accounts for the 6-unit mean difference. Random
      intercepts shrink to near zero. AIC drops, and the rhythm (p < 0.05)
      is estimated with the same amplitude as before.

  This directly demonstrates the "Fixed Effects:" control in the LME panel.

Structure
---------
  Columns    : time, condition, subject, genotype,
               rhythm_high_between_var, rhythm_low_between_var,
               arrhythmic_high_between_var, arrhythmic_low_between_var,
               rhythm_genotype_effect
  Timepoints : 0-46 h, step 2 h (2 full circadian cycles)
  Subjects   : 6 per condition (subject1-subject6)
  Conditions : control  (acrophase at ZT3)
               treatment (acrophase +6 h, amplitude +25 %)

How to test in ChronoScope
--------------------------
  Module  : Classical Rhythm Analysis
  Method  : Linear Mixed Effects

  Test 1 -- Random intercept (high variability)
    Variable      : rhythm_high_between_var
    Random Effect : subject
    Fixed Effects : (none)
    Expected      : p < 0.05, amplitude ~ 2.5

  Test 2 -- Arrhythmic rejection
    Variable      : arrhythmic_high_between_var
    Random Effect : subject
    Fixed Effects : (none)
    Expected      : p > 0.05

  Test 3 -- Fixed effects (without covariate)
    Variable      : rhythm_genotype_effect
    Random Effect : subject
    Fixed Effects : (none)
    Expected      : p < 0.05, but random_effect_var large (~9)

  Test 4 -- Fixed effects (with covariate)
    Variable      : rhythm_genotype_effect
    Random Effect : subject
    Fixed Effects : genotype  (select in the Fixed Effects list)
    Expected      : p < 0.05, random_effect_var small (<1), AIC lower than Test 3

Usage
-----
  python examples/generate_synthetic_lme_test_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TIMEPOINTS = np.arange(0, 48, 2, dtype=float)   # 0..46 h, step 2 h
OMEGA      = 2 * np.pi / 24.0
ACROPHASE  = np.pi / 4                           # peak at ZT3

SUBJECTS = [f"subject{i}" for i in range(1, 7)]

# subject1-3 = WT, subject4-6 = KO
GENOTYPE = {s: ("WT" if int(s[-1]) <= 3 else "KO") for s in SUBJECTS}

CONDITIONS = {
    "control":   {"phase_shift": 0.0,           "amp_factor": 1.0},
    "treatment": {"phase_shift": 2 * np.pi / 4, "amp_factor": 1.25},  # +6 h, +25 %
}

# Between-subject random intercepts
# HIGH: SD ~ 1.75  LOW: SD ~ 0.58  (scaled from the same raw values)
RAW = [-1.5, +2.8, -0.4, +1.9, -2.1, +0.7]
HIGH_INTERCEPTS = {s: v       for s, v in zip(SUBJECTS, RAW)}
LOW_INTERCEPTS  = {s: v / 3.0 for s, v in zip(SUBJECTS, RAW)}

# Genotype baselines for rhythm_genotype_effect
GENO_BASELINE = {"WT": 10.0, "KO": 16.0}   # 6-unit genotype effect
# Small within-genotype random intercepts (so genotype dominates between-subj var)
GENO_INTERCEPTS = {s: v / 6.0 for s, v in zip(SUBJECTS, RAW)}  # SD ~ 0.29

RHYTHM_AMP   = 2.5
WITHIN_NOISE = 0.5

# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------
all_rows = []

for condition, params in CONDITIONS.items():
    ps = params["phase_shift"]
    af = params["amp_factor"]

    for subject in SUBJECTS:
        hi  = HIGH_INTERCEPTS[subject]
        lo  = LOW_INTERCEPTS[subject]
        gi  = GENO_INTERCEPTS[subject]
        gbl = GENO_BASELINE[GENOTYPE[subject]]
        cosinor_base = lambda t: np.cos(OMEGA * t - (ACROPHASE + ps))

        for t in TIMEPOINTS:
            cv = cosinor_base(t)
            row = {
                "time":      t,
                "condition": condition,
                "subject":   subject,
                "genotype":  GENOTYPE[subject],

                # 1. Core random-intercept demo (high between-subject variability)
                "rhythm_high_between_var": round(
                    10.0 + hi + RHYTHM_AMP * af * cv + RNG.normal(0, WITHIN_NOISE), 4
                ),

                # 2. Low between-subject variability (LME vs OLS comparison)
                "rhythm_low_between_var": round(
                    10.0 + lo + RHYTHM_AMP * af * cv + RNG.normal(0, WITHIN_NOISE), 4
                ),

                # 3. No rhythm + high variability (no false positives)
                "arrhythmic_high_between_var": round(
                    10.0 + hi + RNG.normal(0, WITHIN_NOISE), 4
                ),

                # 4. No rhythm + low variability (negative control)
                "arrhythmic_low_between_var": round(
                    10.0 + lo + RNG.normal(0, WITHIN_NOISE), 4
                ),

                # 5. Fixed-effects demo: genotype drives most between-subject variance.
                #    Analyze with and without `genotype` in Fixed Effects.
                "rhythm_genotype_effect": round(
                    gbl + gi + RHYTHM_AMP * af * cv + RNG.normal(0, WITHIN_NOISE), 4
                ),
            }
            all_rows.append(row)

df = pd.DataFrame(all_rows)
df = df.sort_values(["condition", "subject", "time"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).parent / "synthetic_lme_test_data.csv"
df.to_csv(out_path, index=False)

hi_sd = np.std(list(HIGH_INTERCEPTS.values()))
lo_sd = np.std(list(LOW_INTERCEPTS.values()))
gi_sd = np.std(list(GENO_INTERCEPTS.values()))

print(f"Saved  : {out_path}")
print(f"Shape  : {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSubjects per condition : {len(SUBJECTS)}")
print(f"Timepoints             : {len(TIMEPOINTS)}  ({int(TIMEPOINTS[0])}-{int(TIMEPOINTS[-1])} h, step 2 h)")
print(f"Conditions             : {list(CONDITIONS.keys())}")
print(f"Genotypes              : {GENOTYPE}")
print(f"\nBetween-subject intercepts:")
print(f"  HIGH (SD={hi_sd:.2f}): {list(HIGH_INTERCEPTS.values())}")
print(f"  LOW  (SD={lo_sd:.2f}): {[round(v,2) for v in LOW_INTERCEPTS.values()]}")
print(f"  Genotype-driven (within-geno SD={gi_sd:.2f}):")
print(f"    WT baseline={GENO_BASELINE['WT']}, KO baseline={GENO_BASELINE['KO']} (+6 units)")

# ---------------------------------------------------------------------------
# Quick numerical verification
# ---------------------------------------------------------------------------
print("\n--- Numerical verification (control condition) ---")
try:
    import sys, warnings
    sys.path.insert(0, str(Path(__file__).parent.parent))
    warnings.filterwarnings('ignore')
    from core.rhythm_analysis import _fit_lme_model

    ctrl = df[df["condition"] == "control"].copy()
    tests = [
        ("rhythm_high_between_var",     None),
        ("rhythm_low_between_var",      None),
        ("arrhythmic_high_between_var", None),
        ("arrhythmic_low_between_var",  None),
        ("rhythm_genotype_effect",      None),
        ("rhythm_genotype_effect",      ctrl[["genotype"]]),
    ]
    print(f"  {'Variable':<33} {'Fixed Effects':<12} {'p':>7}  {'amp':>5}  {'re_var':>8}  {'AIC':>8}")
    print("  " + "-" * 80)
    for var, fe_df in tests:
        t = ctrl["time"].values.astype(float)
        y = ctrl[var].values.astype(float)
        g = ctrl["subject"].values
        r = _fit_lme_model(t, y, g, period=24.0, fixed_effects_data=fe_df)
        fe_label = "genotype" if fe_df is not None else "(none)"
        aic_str = f"{r.aic:.1f}" if r.aic is not None else "n/a"
        print(f"  {var:<33} {fe_label:<12} {r.p_value:>7.4f}  {r.amplitude:>5.3f}  {r.random_effect_var:>8.4f}  {aic_str:>8}")
except Exception as e:
    print(f"  (verification skipped: {e})")

print("\nDone.")
print("Load synthetic_lme_test_data.csv in ChronoScope.")
