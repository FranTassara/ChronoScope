"""
Validation: does honoring a narrowed period_range at inference degrade the
v5 model's performance on the existing holdout?

Strategy:
  1. Regenerate the same training data (seed=42) + the same
     GroupShuffleSplit (random_state=42) used by train_consensus_model.py
     to recover the exact test indices.
  2. Sanity check: re-extracting features at the DEFAULT range over the
     test set reproduces the cached X_test.npy.
  3. Re-extract features over the test set at user_range=(22, 26)h.
  4. Predict with the saved v5 model on both feature matrices.
  5. Report AUROC, Accuracy, F1, Brier and how many holdout classifications
     change.

Expected: small or no degradation. The synthetic rhythmic periods are
23-25h (all inside 22-26), and real circadian genes are also ~24h, so
narrowing should not lose much signal. Most non-rhythmic genes show no
period in either window. The interesting case is borderline noise that
shows a weak peak at, e.g., 20-21h under defaults — narrowing may drop
those weak peaks from view and change calibration slightly.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / 'training_data_meta_classifier'
MODEL_DIR = PROJECT_ROOT / 'core' / 'models_meta_classifier'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRAINING_DATA_DIR))

warnings.filterwarnings('ignore')

from core.feature_extraction import extract_features, FEATURE_NAMES
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
    precision_score, recall_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
# Step 1: regenerate the same data as the training script
# ---------------------------------------------------------------------------
print("=" * 70)
print("VALIDATION: period_range=(22, 26) vs default on v5 holdout")
print("=" * 70)
print()
print("[1/5] Regenerating training data (seed=42)...")
t0 = time.time()

from generate_synthetic_training_data import generate_training_instances
from generate_real_training_data import generate_from_geo

metadata, dataframes = generate_training_instances(seed=42)
n_synth = len(metadata)

# Same dataset 1 as training
real1_meta, real1_dfs = generate_from_geo(
    geo_accession='GSE11923',
    platform_id='GPL1261',
    starting_id=n_synth + 1000,
    subsample_intervals=[2.0, 4.0],
)
metadata = metadata + real1_meta
dataframes = dataframes + real1_dfs
n_real = len(real1_meta)

# Same dataset 2 as training
biocycle_xlsx = str(TRAINING_DATA_DIR / 'rhythmicdb_query_bioCycle.xlsx')
real2_meta, real2_dfs, _, _ = generate_from_geo(
    geo_accession='GSE11516',
    platform_id='GPL6880',
    biocycle_xlsx=biocycle_xlsx,
    biocycle_dataset_id='E-GEOD-11516',
    biocycle_q_threshold=0.01,
    biocycle_q_non_rhythmic=0.2,
    max_rhythmic=800,
    max_non_rhythmic=800,
    starting_id=n_synth + 2000 + n_real,
    subsample_intervals=[],
    return_ambiguous=True,
)
metadata = metadata + real2_meta
dataframes = dataframes + real2_dfs
print(f"    Done in {time.time() - t0:.1f}s. {len(metadata)} instances total.")


# ---------------------------------------------------------------------------
# Helper: replicate train_consensus_model's per-instance preprocessing
# ---------------------------------------------------------------------------
def _prep(meta_i, df_i):
    """Returns (unique_times, avg_values, df, var, cond) or None if skip."""
    variable = meta_i['variable']
    condition = 'control'
    cond_data = df_i[df_i['condition'] == condition]
    if variable not in cond_data.columns:
        return None
    times = cond_data['time'].values.astype(float)
    values = cond_data[variable].values.astype(float)
    valid = ~(np.isnan(times) | np.isnan(values))
    times = times[valid]
    values = values[valid]
    if len(times) < 4:
        return None
    unique_times = np.unique(times)
    avg_values = np.array([values[times == t].mean() for t in unique_times])
    return unique_times, avg_values, df_i, variable, condition


# ---------------------------------------------------------------------------
# Step 2: extract features at default range over ALL instances to find the
# test indices via the same GroupShuffleSplit as training
# ---------------------------------------------------------------------------
print("[2/5] Filtering instances and rebuilding the split...")

kept_metadata = []
kept_dataframes = []
labels = []
for meta_i, df_i in zip(metadata, dataframes):
    prepped = _prep(meta_i, df_i)
    if prepped is None:
        continue
    kept_metadata.append(meta_i)
    kept_dataframes.append(df_i)
    labels.append(meta_i['is_rhythmic'])

n_kept = len(kept_metadata)
y_full = np.array(labels)
groups = np.array([
    m['gene'] if 'gene' in m else f"synth_{m['instance_id']}"
    for m in kept_metadata
])

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(np.zeros((n_kept, 1)), y_full, groups))
y_test = y_full[test_idx]
print(f"    Total kept: {n_kept}. Test set size: {len(test_idx)}.")
print(f"    Test class balance: {y_test.sum()} rhythmic, "
      f"{len(y_test) - y_test.sum()} non-rhythmic.")


# ---------------------------------------------------------------------------
# Step 3: re-extract features on test instances under TWO regimes
# ---------------------------------------------------------------------------
def _extract_for_test(narrowed_range):
    """Extract features for all test instances. Returns (n_test, n_features) array."""
    rows = []
    t_start = time.time()
    params = {'period_range': narrowed_range} if narrowed_range else None
    for i, idx in enumerate(test_idx):
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            remaining = (len(test_idx) - i) / rate if rate > 0 else 0
            print(f"      {i}/{len(test_idx)} ({elapsed:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining)")
        meta_i = kept_metadata[idx]
        df_i = kept_dataframes[idx]
        prepped = _prep(meta_i, df_i)
        if prepped is None:
            rows.append({n: np.nan for n in FEATURE_NAMES})
            continue
        ut, av, df_i, var, cond = prepped
        feats = extract_features(
            ut, av, df_i, var, cond, 'time', 'condition', parameters=params
        )
        rows.append(feats)
    X = np.array([[row.get(n, np.nan) for n in FEATURE_NAMES] for row in rows])
    print(f"    Feature extraction done in {time.time() - t_start:.1f}s")
    return X


print()
print("[3a/5] Re-extracting test features at DEFAULT range (sanity)...")
X_test_default = _extract_for_test(None)

print()
print("[3b/5] Re-extracting test features at (22, 26)h range...")
X_test_narrow = _extract_for_test((22.0, 26.0))


# ---------------------------------------------------------------------------
# Step 4: sanity check vs cached X_test.npy
# ---------------------------------------------------------------------------
print()
print("[4/5] Sanity: comparing re-extracted DEFAULT features to cached X_test.npy")
X_test_cached = np.load(MODEL_DIR / 'X_test.npy')
y_test_cached = np.load(MODEL_DIR / 'y_test.npy')

# Cached file was saved with v5's 11-feature schema (matches FEATURE_NAMES).
if X_test_default.shape == X_test_cached.shape and len(y_test) == len(y_test_cached):
    label_match = np.array_equal(y_test, y_test_cached)
    if label_match:
        # Compare element-wise. NaNs match NaNs as same.
        diff_mask = ~(np.isnan(X_test_default) & np.isnan(X_test_cached)) & (
            X_test_default != X_test_cached
        )
        if not diff_mask.any():
            print("    OK: re-extracted DEFAULT features match cached X_test.npy bit-for-bit.")
        else:
            max_abs_diff = np.nanmax(np.abs(X_test_default - X_test_cached))
            n_diff = diff_mask.sum()
            print(f"    NOTE: {n_diff} cells differ. max abs diff = {max_abs_diff:.2e}")
    else:
        print(f"    WARNING: labels differ from cache "
              f"({np.sum(y_test != y_test_cached)} mismatches).")
else:
    print(f"    WARNING: shapes differ. Re-extracted={X_test_default.shape}, "
          f"cached={X_test_cached.shape}. Probably a real-data regeneration "
          f"discrepancy. Continuing with re-extracted features as DEFAULT.")


# ---------------------------------------------------------------------------
# Step 5: load model, predict, compare
# ---------------------------------------------------------------------------
print()
print("[5/5] Loading v5 model and predicting on both feature sets...")
model = joblib.load(str(MODEL_DIR / 'consensus_rf_model.pkl'))


def _metrics(y_true, y_proba, label):
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  --- {label} ---")
    print(f"    Accuracy:   {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Precision:  {precision_score(y_true, y_pred):.4f}")
    print(f"    Recall:     {recall_score(y_true, y_pred):.4f}")
    print(f"    F1:         {f1_score(y_true, y_pred):.4f}")
    print(f"    ROC-AUC:    {roc_auc_score(y_true, y_proba):.4f}")
    print(f"    Brier:      {brier_score_loss(y_true, y_proba):.4f}")
    print(f"    Conf TN/FP/FN/TP: {tn}/{fp}/{fn}/{tp}")
    return y_pred


proba_default = model.predict_proba(X_test_default)[:, 1]
proba_narrow = model.predict_proba(X_test_narrow)[:, 1]

print()
pred_default = _metrics(y_test, proba_default, "DEFAULT range")
print()
pred_narrow = _metrics(y_test, proba_narrow, "NARROW range (22-26h)")


# ---------------------------------------------------------------------------
# Comparison: how many predictions changed?
# ---------------------------------------------------------------------------
print()
print("  --- Prediction Agreement (DEFAULT vs NARROW) ---")
n = len(y_test)
agree_binary = (pred_default == pred_narrow).sum()
print(f"    Binary 0.5-threshold agreement: {agree_binary}/{n} "
      f"({100*agree_binary/n:.1f}%)")

# Classification (Rhythmic/Borderline/Arrhythmic) per ChronoScope thresholds
def _classify(p):
    if p >= 0.7:
        return 'Rhythmic'
    if p >= 0.3:
        return 'Borderline'
    return 'Arrhythmic'

cls_default = np.array([_classify(p) for p in proba_default])
cls_narrow = np.array([_classify(p) for p in proba_narrow])
agree_3class = (cls_default == cls_narrow).sum()
print(f"    3-class (Rhy/Border/Arr) agreement: {agree_3class}/{n} "
      f"({100*agree_3class/n:.1f}%)")

# Probability shift distribution
abs_diff = np.abs(proba_default - proba_narrow)
print(f"    |delta-probability|: mean={abs_diff.mean():.4f}, "
      f"median={np.median(abs_diff):.4f}, max={abs_diff.max():.4f}")
print(f"    Cases where |delta-p| > 0.10: {(abs_diff > 0.10).sum()}/{n}")
print(f"    Cases where |delta-p| > 0.20: {(abs_diff > 0.20).sum()}/{n}")

# Where do the disagreements concentrate? By true label.
disagree_mask = pred_default != pred_narrow
if disagree_mask.any():
    n_disagree_R = int(((disagree_mask) & (y_test == 1)).sum())
    n_disagree_NR = int(((disagree_mask) & (y_test == 0)).sum())
    print(f"    Disagreements among true rhythmic:     {n_disagree_R}")
    print(f"    Disagreements among true non-rhythmic: {n_disagree_NR}")

print()
print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
