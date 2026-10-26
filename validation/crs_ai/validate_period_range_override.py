"""
Validation: does honoring a user period_range at inference degrade the
released (v6) model on the published holdout?

Strategy:
  1. Regenerate the training instances exactly as train_consensus_model.py
     does (synthetic seed=42 + GSE11923, GSE11516, GSE77451, GSE39445).
  2. Recover the published 1,117-series holdout. Re-running the
     GroupShuffleSplit does NOT reproduce it: the GSE39445 instance cap
     samples non-deterministically (Supplementary Methods S4), so the
     instance list, and therefore the split, changes between runs. Each row
     of holdout_predictions.csv is instead matched to a regenerated instance
     with the same gene and signal type whose default-range feature vector
     equals the cached X_test.npy row bit-for-bit (GSE39445 is regenerated
     without the cap to make every candidate available).
  3. Re-extract the holdout features under four regimes:
        DEFAULT       training settings (JTK/Cosinor 20-28 h, LS 18-32 h)
        GUI 20-28     the range the GUI pre-fills; must equal DEFAULT now
                      that Lomb-Scargle always keeps its training range
        NARROW 22-26  a tight user range (JTK/Cosinor 22-26 h, LS 18-32 h)
        22-26, LS too the policy before 2026-10-22, when the user range was
                      also applied to Lomb-Scargle (kept for reference)
  4. Predict with the released model and report AUROC, Accuracy, F1,
     Brier, specificity and how many classifications change.

Writes parameter_override_validation.txt next to this script.
"""
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT      = Path(__file__).resolve().parent.parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / 'training_data_meta_classifier'
MODEL_DIR         = PROJECT_ROOT / 'core' / 'models'
SCRIPT_DIR        = Path(__file__).resolve().parent
REPORT_PATH       = SCRIPT_DIR / 'parameter_override_validation.txt'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRAINING_DATA_DIR))

warnings.filterwarnings('ignore')

import core.feature_extraction as fe
from core.feature_extraction import extract_features, FEATURE_NAMES
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss, confusion_matrix,
)

REGIMES = [('DEFAULT', None), ('GUI 20-28', (20.0, 28.0)), ('NARROW 22-26', (22.0, 26.0)),
           ('22-26, LS too', (22.0, 26.0))]
LS_FOLLOWS_USER = {'22-26, LS too'}   # previous policy, re-created by patching _resolve_params


# ---------------------------------------------------------------------------
# Step 1: regenerate the same data as the training script
# ---------------------------------------------------------------------------
print("[1/4] Regenerating training instances (as train_consensus_model.py)...")
t0 = time.time()
from generate_synthetic_training_data import generate_training_instances
from generate_real_training_data import (
    generate_from_geo, generate_from_GSE77451, generate_from_GSE39445,
)

metadata, dataframes = generate_training_instances(seed=42)
n_synth = len(metadata)
biocycle_xlsx = str(TRAINING_DATA_DIR / 'rhythmicdb_query_BioCycle_allModels_noFilters.xlsx')
geo_cache_dir = TRAINING_DATA_DIR / 'data' / 'geo'
n_real = 0
m, d = generate_from_geo(geo_accession='GSE11923', platform_id='GPL1261',
                         starting_id=n_synth + 1000, subsample_intervals=[2.0, 4.0])
metadata += m; dataframes += d; n_real += len(m)
m, d, _, _ = generate_from_geo(
    geo_accession='GSE11516', platform_id='GPL6880', biocycle_xlsx=biocycle_xlsx,
    biocycle_dataset_id='E-GEOD-11516', biocycle_q_threshold=0.01,
    biocycle_period_min=20.0, biocycle_period_max=28.0, non_rhythmic_from_absence=True,
    max_rhythmic=800, max_non_rhythmic=800, starting_id=n_synth + 2000 + n_real,
    subsample_intervals=[], return_ambiguous=True)
metadata += m; dataframes += d; n_real += len(m)
m, d = generate_from_GSE77451(abruzzi_xlsx_path=TRAINING_DATA_DIR / 'abruzzi_2017_cycling.xlsx',
                              geo_cache_dir=geo_cache_dir, starting_instance_id=n_synth + 10000,
                              max_positives_per_cell_type=200, hc_only=True, seed=42)
metadata += m; dataframes += d
# GSE39445 without the cap, so that every published holdout instance is a candidate
m, d = generate_from_GSE39445(moller_xlsx_path=TRAINING_DATA_DIR / 'moller_levet_2013_circadian.xlsx',
                              geo_cache_dir=geo_cache_dir, starting_instance_id=n_synth + 20000,
                              max_per_class=10 ** 6, seed=42)
metadata += m; dataframes += d
print(f"    {len(metadata)} candidate instances in {time.time() - t0:.0f}s")


def _prep(meta_i, df_i):
    """(unique_times, avg_values) as in train_consensus_model, or None if skipped."""
    variable = meta_i['variable']
    cond_data = df_i[df_i['condition'] == 'control']
    if variable not in cond_data.columns:
        return None
    times = cond_data['time'].values.astype(float)
    values = cond_data[variable].values.astype(float)
    valid = ~(np.isnan(times) | np.isnan(values))
    times, values = times[valid], values[valid]
    if len(times) < 4:
        return None
    unique_times = np.unique(times)
    return unique_times, np.array([values[times == t].mean() for t in unique_times])


_resolve_params = fe._resolve_params


def _ls_follows_user(parameters):
    """The policy before 2026-10-22: the user range also bounded Lomb-Scargle."""
    p = _resolve_params(parameters)
    if parameters and parameters.get('period_range') and p['jtk_periods'] != _resolve_params(None)['jtk_periods']:
        p['ls_range'] = (float(p['cosinor_periods'][0]), float(p['cosinor_periods'][-1]))
    return p


def _features(series, period_range, ls_follows_user=False):
    params = {'period_range': period_range} if period_range else None
    fe._resolve_params = _ls_follows_user if ls_follows_user else _resolve_params
    try:
        f = extract_features(series[0], series[1], parameters=params)
    finally:
        fe._resolve_params = _resolve_params
    return np.array([f.get(n, np.nan) for n in FEATURE_NAMES], float)


# ---------------------------------------------------------------------------
# Step 2: recover the published holdout
# ---------------------------------------------------------------------------
print("[2/4] Matching holdout_predictions.csv rows to regenerated instances...")
X_cached = np.load(SCRIPT_DIR / 'X_test.npy')
y_cached = np.load(SCRIPT_DIR / 'y_test.npy')
published = pd.read_csv(SCRIPT_DIR / 'holdout_predictions.csv')

by_key = {}
for i, meta_i in enumerate(metadata):
    gene = meta_i['gene'] if 'gene' in meta_i else f"synth_{meta_i['instance_id']}"
    by_key.setdefault((gene, meta_i.get('signal_type')), []).append(i)

series, used, missing = [], set(), []
for r, row in published.iterrows():
    hit = None
    for i in by_key.get((row['gene'], row['signal_type']), []):
        if i in used or metadata[i]['is_rhythmic'] != y_cached[r]:
            continue
        s = _prep(metadata[i], dataframes[i])
        if s is not None and np.array_equal(_features(s, None), X_cached[r], equal_nan=True):
            hit = (i, s)
            break
    if hit is None:
        missing.append(r)
        continue
    used.add(hit[0]); series.append(hit[1])
if missing:
    sys.exit(f"    {len(missing)} holdout rows could not be matched (first: {missing[:5]})")
y_test = y_cached
print(f"    All {len(series)} holdout series recovered; default features equal X_test.npy bit-for-bit.")


# ---------------------------------------------------------------------------
# Step 3: re-extract features under each regime
# ---------------------------------------------------------------------------
print("[3/4] Extracting features under each regime...")
X = {}
for name, rng in REGIMES:
    t1 = time.time()
    X[name] = X_cached if rng is None else np.array([_features(s, rng, name in LS_FOLLOWS_USER) for s in series])
    print(f"    {name}: {time.time() - t1:.0f}s")


# ---------------------------------------------------------------------------
# Step 4: predict and compare
# ---------------------------------------------------------------------------
print("[4/4] Predicting...")
model = joblib.load(str(MODEL_DIR / 'consensus_rf_model.pkl'))
proba = {name: model.predict_proba(X[name])[:, 1] for name, _ in REGIMES}
assert np.allclose(proba['DEFAULT'], published['y_proba'], rtol=0, atol=1e-12)
assert np.array_equal(X['GUI 20-28'], X['DEFAULT'], equal_nan=True), 'GUI default must reproduce training features'


def _metrics(p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {'Accuracy': accuracy_score(y_test, pred), 'F1': f1_score(y_test, pred),
            'ROC-AUC': roc_auc_score(y_test, p), 'Brier loss': brier_score_loss(y_test, p),
            'Specificity': tn / (tn + fp), 'Recall': tp / (tp + fn), 'cm': (tn, fp, fn, tp)}


def _cls(p):
    return np.where(p >= 0.7, 2, np.where(p >= 0.3, 1, 0))


met = {name: _metrics(proba[name]) for name, _ in REGIMES}
n = len(y_test)
W = 80
lines = []
w = lines.append
w("=" * W)
w("ChronoScope - CONSENSUS RHYTHMICITY SCORE (CRS-AI)")
w("RUNTIME PARAMETER OVERRIDE — VALIDATION REPORT")
w("=" * W)
w("")
w(f"  Validation date:   {date.today().isoformat()}")
w("  Model under test:  v6 (core/models/consensus_rf_model.pkl, 11 features)")
w("  Script:            validation/crs_ai/validate_period_range_override.py")
w("")
w("")
w("1. MOTIVATION")
w("-" * W)
w("")
w("The CRS-AI features were extracted at training time with fixed sub-method")
w("search windows: JTK_CYCLE and Cosinor OLS 20-28 h, Lomb-Scargle 18-32 h,")
w("F24 at 24 h, JTK asymmetry 0.5. A user period_range is honored only if it")
w("overlaps the training window [18, 32] h by more than 1 h (clipped to it if")
w("it spills out, training defaults otherwise), and it is then applied to")
w("JTK and Cosinor; Lomb-Scargle always keeps its training range. Until")
w("2026-10-22 the range also bounded Lomb-Scargle, so a default GUI run")
w("(pre-filled 20-28 h) did not reproduce the training features; the")
w("'22-26, LS too' column re-creates that policy for comparison. This file")
w("quantifies the cost of a user range on the published holdout.")
w("")
w("")
w("2. EXPERIMENTAL DESIGN")
w("-" * W)
w("")
w(f"  - The {n} published holdout series (holdout_predictions.csv) are")
w("    recovered from regenerated training instances; their DEFAULT features")
w("    equal the cached X_test.npy bit-for-bit and their DEFAULT")
w("    probabilities reproduce holdout_predictions.csv.")
w("  - Features are re-extracted under GUI 20-28 (period_range = (20, 28) h,")
w("    which must reproduce DEFAULT), NARROW 22-26 (period_range = (22, 26) h;")
w("    JTK and Cosinor change) and '22-26, LS too' (the previous policy: the")
w("    same range also applied to Lomb-Scargle).")
w("  - The same released model scores every feature matrix.")
w("")
w("")
w(f"3. HOLDOUT METRICS (n = {n})")
w("-" * W)
w("")
hdr = "".join(f"{name:>16s}" for name, _ in REGIMES)
w(f"  {'':<24s}{hdr}")
for k in ('Accuracy', 'F1', 'ROC-AUC', 'Brier loss', 'Specificity', 'Recall'):
    w(f"  {k:<24s}" + "".join(f"{met[name][k]:>16.4f}" for name, _ in REGIMES))
w(f"  {'Confusion TN/FP/FN/TP':<24s}" + "".join(f"{'/'.join(map(str, met[name]['cm'])):>16s}" for name, _ in REGIMES))
w("")
w("  Default ROC-AUC 95% bootstrap CI: [0.9348, 0.9604] (training_report.txt 6.2).")
w("")
w("")
w("4. PREDICTION AGREEMENT WITH DEFAULT")
w("-" * W)
for name, _ in REGIMES[1:]:
    p0, p1 = proba['DEFAULT'], proba[name]
    d = np.abs(p0 - p1)
    changed = [FEATURE_NAMES[j] for j in range(len(FEATURE_NAMES)) if not np.array_equal(X['DEFAULT'][:, j], X[name][:, j])]
    w("")
    w(f"  {name}")
    w(f"    Features that change:          {', '.join(changed)}")
    w(f"    Binary (>= 0.5) agreement:     {np.sum((p0 >= 0.5) == (p1 >= 0.5))} / {n}  ({np.mean((p0 >= 0.5) == (p1 >= 0.5)):.1%})")
    w(f"    Three-class agreement:         {np.sum(_cls(p0) == _cls(p1))} / {n}  ({np.mean(_cls(p0) == _cls(p1)):.1%})")
    w(f"    |delta-p| mean / median / max: {d.mean():.4f} / {np.median(d):.4f} / {d.max():.4f}")
    w(f"    |delta-p| > 0.10:              {np.sum(d > 0.10)} / {n}")
    w(f"    |delta-p| > 0.20:              {np.sum(d > 0.20)} / {n}")
    flips = (p0 >= 0.5) != (p1 >= 0.5)
    w(f"    Binary flips, true rhythmic:   {np.sum(flips & (y_test == 1))} / {np.sum(y_test == 1)}")
    w(f"    Binary flips, true non-rhythmic: {np.sum(flips & (y_test == 0))} / {np.sum(y_test == 0)}")
w("")
w("")
w("5. NOTES")
w("-" * W)
w("")
w("  Out-of-window ranges (overlap with [18, 32] h of 1 h or less) fall back to")
w("  DEFAULT and therefore reproduce its numbers exactly; the fallback and any")
w("  clipping are reported to the user (GUI Notes column, end-of-run warning).")
w("  n_harmonics is not validated: it affects no model feature.")
w("")
w("=" * W)
w("END OF VALIDATION REPORT")
w("=" * W)
REPORT_PATH.write_text("\n".join(lines) + "\n", encoding='utf-8')
print("\n".join(lines))
print(f"\nReport written: {REPORT_PATH}")
