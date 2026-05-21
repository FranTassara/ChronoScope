"""
End-to-end verification that user parameters propagate from a UI-style
parameters dict all the way through the engine -> meta-classifier ->
feature extraction -> sub-method calls.

Three scenarios are run against the same synthetic dataframe:
  A. parameters=None                          (engine default path)
  B. parameters with period_range=(22, 26)    (bounded override, honored)
  C. parameters with period_range=(6, 12)     (outside training window,
                                                falls back with warning)

For each scenario we capture:
  - The probability returned by the model
  - Any warnings emitted (the policy is supposed to warn on clip/fallback)
  - A sample of extracted features

Then we run a programmatic sanity probe on _resolve_params itself with
the exact same dicts, to confirm what was actually fed to the sub-
methods.

If the probabilities differ between A and B (same signal, different
period_range honored), the wiring is alive. If A and C produce the same
probability and C emits a fallback warning, the bounded policy is
working as advertised.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.analysis_engine import AnalysisEngine, AnalysisType
from core.feature_extraction import _resolve_params


def _build_ui_params(period_min, period_max, n_harmonics):
    """Emulate the subset of _get_current_parameters() that the AI cares
    about. Other keys are present in real UI usage but ignored by
    _resolve_params."""
    return {
        'period_range': (period_min, period_max),
        'n_harmonics': n_harmonics,
        # Decoy keys the AI must ignore — these are real UI keys.
        'period_step': 1.0,
        'n_permutations': 1000,
        'loss': 'linear',
    }


def _make_dataframe(period_h=24.0, noise_std=0.3, seed=0):
    """A cosine + Gaussian noise, formatted as a one-condition DataFrame."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 48, 2.0, dtype=float)
    y = 10.0 + 2.0 * np.cos(2 * np.pi * t / period_h) + rng.normal(0, noise_std, len(t))
    df = pd.DataFrame({
        'time': t,
        'condition': 'control',
        'replicate': 'rep1',
        'gene_x': y,
    })
    return df


def _run_and_capture(engine, df, parameters, label):
    """Run engine path and capture warnings + result."""
    print()
    print("=" * 70)
    print(f"SCENARIO {label}: parameters = {parameters}")
    print("=" * 70)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter('always')
        result = engine.run_analysis(
            data=df,
            variable='gene_x',
            condition='control',
            analysis_type=AnalysisType.CONSENSUS_AI,
            time_col='time',
            condition_col='condition',
            parameters=parameters,
        )

    print(f"  Result success: {result.success}")
    prob = None
    if result.success:
        # Engine convention (analysis_engine._run_consensus_ai):
        #   result.r_squared  -> rhythmicity probability (0-1)
        #   result.p_value    -> 1 - probability
        # We use r_squared as the canonical probability.
        prob = getattr(result, 'r_squared', None)
        print(f"  Rhythmicity probability:    {prob}")
        if prob is not None:
            cls = 'Rhythmic' if prob >= 0.7 else ('Borderline' if prob >= 0.3 else 'Arrhythmic')
            print(f"  Classification:             {cls}")
        # Detected period from cosinor (the consensus AI carries it through)
        if hasattr(result, 'period'):
            print(f"  Detected cosinor period:    {result.period}")
    else:
        print(f"  Message: {result.message}")

    if wlist:
        print(f"  Warnings raised: {len(wlist)}")
        for w in wlist:
            print(f"    [{w.category.__name__}] {w.message}")
    else:
        print(f"  Warnings raised: 0")

    # Probe _resolve_params directly with the same dict, to show what
    # the sub-methods actually see
    print(f"  _resolve_params output:")
    resolved = _resolve_params(parameters)
    print(f"    jtk_periods     = {resolved['jtk_periods']}")
    print(f"    ls_range        = {resolved['ls_range']}")
    print(f"    cosinor (first 5) = {list(resolved['cosinor_periods'])[:5]} "
          f"... last = {list(resolved['cosinor_periods'])[-1]}")
    print(f"    n_harmonics     = {resolved['n_harmonics']}")

    return result, [str(w.message) for w in wlist], prob


def main():
    print("=" * 70)
    print("Parameter propagation end-to-end check")
    print("=" * 70)

    engine = AnalysisEngine()

    # === Part 1: a clean 24h signal — verify chain runs, warnings fire ===
    print("\n##### Part 1: clean 24h cosine (SNR ~6.7) #####")
    df_24 = _make_dataframe(period_h=24.0, noise_std=0.3, seed=0)

    rA, wA, pA = _run_and_capture(engine, df_24, None, label="A1 (24h, defaults)")
    paramsB = _build_ui_params(22.0, 26.0, n_harmonics=3)
    rB, wB, pB = _run_and_capture(engine, df_24, paramsB, label="B1 (24h, range=22-26h, n_harm=3)")
    paramsC = _build_ui_params(6.0, 12.0, n_harmonics=2)
    rC, wC, pC = _run_and_capture(engine, df_24, paramsC, label="C1 (24h, range=6-12h, n_harm=2)")

    # === Part 2: a 21h signal — narrow range CAN'T see it ===
    # Default window 20-28h still contains 21h. Narrow (22, 26) does NOT.
    # If the override propagates, the model's probability should DROP under
    # the narrow regime (sub-methods forced to report best period inside
    # 22-26h, but the true signal is 21h, so all p-values get noisy and
    # consensus probability should decrease vs default).
    print("\n##### Part 2: borderline 21h cosine — diagnostic for propagation #####")
    df_21 = _make_dataframe(period_h=21.0, noise_std=0.3, seed=1)

    rA2, wA2, pA2 = _run_and_capture(engine, df_21, None, label="A2 (21h, defaults)")
    paramsB2 = _build_ui_params(22.0, 26.0, n_harmonics=2)
    rB2, wB2, pB2 = _run_and_capture(engine, df_21, paramsB2, label="B2 (21h, range=22-26h)")

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("VERDICTS")
    print("=" * 70)

    def _ok(cond):
        return "OK" if cond else "FAIL"

    has_clip_warning_C = any('outside' in w.lower() for w in wC)
    print(f"  Warnings firing:")
    print(f"    A1 (defaults, 24h):     count={len(wA)}   expected 0   "
          f"{_ok(len(wA) == 0)}")
    print(f"    B1 (22-26, 24h):        count={len(wB)}   expected 0   "
          f"{_ok(len(wB) == 0)}")
    print(f"    C1 (6-12, 24h):         fallback warning = {has_clip_warning_C}  "
          f"{_ok(has_clip_warning_C)}")

    print(f"  Probabilities (24h signal):  A1={pA:.4f}  B1={pB:.4f}  C1={pC:.4f}")
    print(f"    A1 vs C1 (fallback should match defaults): "
          f"|delta|={abs(pA-pC):.4f}   {_ok(abs(pA-pC) < 1e-9)}")
    print(f"    A1 vs B1 (24h signal — both should be high): "
          f"both > 0.7? {_ok(pA > 0.7 and pB > 0.7)}")

    print(f"  Probabilities (21h signal): A2={pA2:.4f}  B2={pB2:.4f}")
    # Whichever direction the probability moves, the fact that it MOVES
    # at all proves the override reached the sub-methods. The detected
    # period also shifts because Cosinor/LS are forced inside the
    # narrowed window. Both are independent propagation signals.
    period_A2 = getattr(rA2, 'period', None)
    period_B2 = getattr(rB2, 'period', None)
    print(f"    Detected period:        A2={period_A2}  B2={period_B2}")
    prob_changed = abs(pA2 - pB2) > 1e-6
    period_changed = (period_A2 is not None and period_B2 is not None
                      and abs(period_A2 - period_B2) > 1e-6)
    print(f"    Probability changed?    {prob_changed}    "
          f"|delta-p|={abs(pA2-pB2):.4f}   {_ok(prob_changed)}")
    print(f"    Detected period changed? {period_changed}   "
          f"{_ok(period_changed)}")
    print(f"    (Both must be True for parameter override to be propagating.)")

    print()
    print("=" * 70)
    print("END")
    print("=" * 70)


if __name__ == '__main__':
    main()
