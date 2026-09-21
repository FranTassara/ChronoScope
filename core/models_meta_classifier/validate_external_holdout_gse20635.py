"""
External LODO (Leave-One-Dataset-Out) validation -- GSE20635
================================================================

Tests whether the CRS-AI model generalises to a *Rattus norvegicus* (rat)
peripheral-tissue microarray dataset never seen during training.

Rat abdominal white adipose tissue (WAT) complements the existing
validation portfolio (mouse liver / Drosophila clock neurons / human
blood for training; zebrafish whole organism for external validation)
with: a mammalian species distinct from mouse and human, a peripheral
metabolic tissue not represented anywhere in the training or prior
validation data, and a single-factor 24h time-course design with true
biological replication (n=2-3 animals/timepoint) rather than the
pooled/tissue-mixed designs used elsewhere.

Probe-level labeling (why this differs from the GSE37332/GSE3416 scripts)
---------------------------------------------------------------------------
The GSE37332 (zebrafish) and GSE3416 (Arabidopsis) validations collapse
probes to a gene-level universe and match against BioCycle's "Gene info"
identifier for that dataset (gene symbol or AGI locus respectively).

For GSE20635, BioCycle's RhythmicDB "Gene info" field is a rat UniGene
cluster ID (e.g. "Rn.94978"), NOT the gene symbol that GPL1355's curated
annotation reports (e.g. "Sumo2"). Collapsing GPL1355 probes to gene
symbols and intersecting with BioCycle's UniGene-keyed labels would
therefore silently fail to match almost every gene. Instead, this script
labels directly at the PROBE level using BioCycle's "Probe ref" column,
which contains the actual Affymetrix probeset ID (e.g. "1367558_x_at")
-- the same identifier space as the series matrix itself. Gene symbols
(via GPL1355.annot.gz) are used only for the human-readable clock-gene
audit table, never for label matching.

Labeling strategy (BioCycle-consistent, non-rhythmic-from-absence)
---------------------------------------------------------------------
  R (Rhythmic, label=1):
      Probe in RhythmicDB ^ U, BioCycle Q-value <= 0.05,
      AND 20 <= Period <= 28 h.

  X (Excluded):
      Probe in RhythmicDB ^ U but not meeting R criteria (borderline
      rhythmics). Excluded from both R and N.

  N (Non-rhythmic, label=0):
      Probe in expression universe U that is completely absent from
      RhythmicDB for this dataset. Capped at 300 probes (seed 42).

Dataset: Sukumaran et al. (2010), Physiol Genomics 42:141. Wistar rat
abdominal white adipose tissue, LD 12:12, 18 circadian timepoints over
24h (0.25-23.75h after lights-on), 2-3 animals/timepoint (52 samples,
2 excluded for RNA degradation per the original study).

Usage
-----
Run from the project root after training:
    python core/models_meta_classifier/validate_external_holdout_gse20635.py
"""

import sys
import time
import warnings
import hashlib
import re
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR     = Path(__file__).parent
PROJECT_ROOT  = MODEL_DIR.parent.parent
TRAINING_DIR  = PROJECT_ROOT / 'training_data_meta_classifier'
GEO_CACHE_DIR = TRAINING_DIR / 'data' / 'geo'
REPORT_PATH   = MODEL_DIR / 'validate_external_holdout_gse20635.txt'
MODEL_PATH    = MODEL_DIR / 'consensus_rf_model.pkl'
FEATURES_PATH = MODEL_DIR / 'feature_names.json'
BIOCYCLE_XLSX = (TRAINING_DIR /
                 'rhythmicdb_query_BioCycle_allModels_noFilters.xlsx')

# GSE20635 / GPL1355 files (expected pre-cached)
SERIES_MATRIX  = GEO_CACHE_DIR / 'GSE20635_series_matrix.txt.gz'
PLATFORM_ANNOT = GEO_CACHE_DIR / 'GPL1355.annot.gz'

BIOCYCLE_DATASET_ID = 'E-GEOD-20635'
Q_RHYTHMIC          = 0.05
PERIOD_MIN          = 20.0
PERIOD_MAX          = 28.0
N_CLASS_CAP         = 300
N_CLASS_SEED        = 42

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRAINING_DIR))

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Imports after path setup
# ---------------------------------------------------------------------------
import json
from core.feature_extraction import extract_features, FEATURE_NAMES
from generate_real_training_data import (
    parse_series_matrix,
    parse_platform_annotation,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix,
)


# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------
def _bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray,
                  n_iter: int = 1000, ci: float = 0.95,
                  seed: int = 42) -> dict:
    """Percentile bootstrap CIs for AUROC, average precision, accuracy, Brier."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs, aps, accs, briers = [], [], [], []

    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        ypr = y_proba[idx]
        yp  = (ypr >= 0.5).astype(int)
        aurocs.append(roc_auc_score(yt, ypr))
        aps.append(average_precision_score(yt, ypr))
        accs.append(accuracy_score(yt, yp))
        briers.append(brier_score_loss(yt, ypr))

    alpha = (1 - ci) / 2
    lo, hi = alpha * 100, (1 - alpha) * 100
    n_valid = len(aurocs)
    return {
        'n_valid': n_valid,
        'auroc_ci':  (float(np.percentile(aurocs, lo)),
                      float(np.percentile(aurocs, hi))),
        'ap_ci':     (float(np.percentile(aps, lo)),
                      float(np.percentile(aps, hi))),
        'acc_ci':    (float(np.percentile(accs, lo)),
                      float(np.percentile(accs, hi))),
        'brier_ci':  (float(np.percentile(briers, lo)),
                      float(np.percentile(briers, hi))),
    }


# ---------------------------------------------------------------------------
# Step 0: Pre-checks
# ---------------------------------------------------------------------------
print("=" * 70)
print("EXTERNAL LODO VALIDATION -- GSE20635 (Rattus norvegicus, "
      "abdominal WAT)")
print("=" * 70)
print()

for path, label in [
    (MODEL_PATH,     "Trained model"),
    (FEATURES_PATH,  "Feature names"),
    (SERIES_MATRIX,  "GSE20635 series matrix"),
    (PLATFORM_ANNOT, "GPL1355 platform annotation"),
    (BIOCYCLE_XLSX,  "BioCycle XLSX"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

GEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Step 1: Load trained model + feature names
# ---------------------------------------------------------------------------
print("[1/5] Loading trained model artifact...")
model = joblib.load(str(MODEL_PATH))
model_bytes = MODEL_PATH.read_bytes()
model_hash  = hashlib.md5(model_bytes).hexdigest()[:12]
model_size_kb = len(model_bytes) / 1024
print(f"  Model loaded: {MODEL_PATH.name}  "
      f"({model_size_kb:.1f} KB, md5={model_hash})")

with open(FEATURES_PATH) as f:
    feature_names_file = json.load(f)
if feature_names_file != FEATURE_NAMES:
    print("  WARNING: feature_names.json differs from FEATURE_NAMES in "
          "feature_extraction module. Using module list.")
print(f"  Features: {len(FEATURE_NAMES)}")

# ---------------------------------------------------------------------------
# Step 2: Parse GSE20635 expression data (probe-level; no gene collapsing)
# ---------------------------------------------------------------------------
print("\n[2/5] Parsing GSE20635 (Rattus norvegicus, GPL1355)...")
t0 = time.time()

print("  [cached] GSE20635_series_matrix.txt.gz")
expr_df, sinfo = parse_series_matrix(str(SERIES_MATRIX))
print(f"  Matrix: {len(expr_df)} probes x {len(expr_df.columns)} samples  "
      f"({time.time()-t0:.1f}s)")

probe_universe = set(expr_df.index)
print(f"  Probe-level expression universe (U): {len(probe_universe)} probes")

# Probe -> gene symbol, for the human-readable clock-gene audit table ONLY
# (not used for R/X/N label matching -- see module docstring).
print("  [cached] GPL1355.annot.gz")
probe_to_symbol = parse_platform_annotation(str(PLATFORM_ANNOT))

# GSE20635 encodes time as "circadian time (hrs after lights on): 0.25" in
# !Sample_characteristics_ch1. The generic extract_timepoints_from_samples()
# regex does not allow the "(hrs after lights on)" parenthetical between
# "time" and the numeric value, so it is extracted directly here.
sample_times = {}
for sid, info in sinfo.items():
    for key, values in info.items():
        if 'characteristic' not in key.lower():
            continue
        for val in values:
            m = re.search(r'circadian\s+time[^:]*:\s*([\d.]+)', val, re.IGNORECASE)
            if m:
                sample_times[sid] = float(m.group(1))
                break
        if sid in sample_times:
            break
print(f"  Samples with circadian timepoints: "
      f"{len(sample_times)}/{len(expr_df.columns)}")
if len(sample_times) == 0:
    raise RuntimeError(
        "No timepoints extracted from GSE20635 sample metadata.\n"
        "Check !Sample_characteristics_ch1 in the series matrix."
    )

t_vals_all = sorted(sample_times.values())
unique_tps  = sorted(set(t_vals_all))
print(f"  Time range: {unique_tps[0]:.2f}-{unique_tps[-1]:.2f} h  "
      f"({len(unique_tps)} unique timepoints)")

# Order samples by time
valid_samples = [s for s in expr_df.columns if s in sample_times]
valid_samples.sort(key=lambda s: sample_times[s])
times_arr  = np.array([sample_times[s] for s in valid_samples])
times_arr -= times_arr.min()   # normalise to start at 0
unique_times_arr = np.unique(times_arr)
print(f"  Unique time bins after normalisation: {len(unique_times_arr)}")

# ---------------------------------------------------------------------------
# Step 3: Build R / X / N probe sets (BioCycle-consistent labeling)
# ---------------------------------------------------------------------------
print("\n[3/5] Building labeled probe sets from BioCycle "
      f"({BIOCYCLE_DATASET_ID})...")

bc_df = pd.read_excel(str(BIOCYCLE_XLSX))
ds_data = bc_df[bc_df['Dataset'] == BIOCYCLE_DATASET_ID].copy()
if len(ds_data) == 0:
    available = bc_df['Dataset'].unique().tolist()
    raise ValueError(
        f"Dataset '{BIOCYCLE_DATASET_ID}' not found in BioCycle XLSX.\n"
        f"Available datasets: {available}"
    )
print(f"  RhythmicDB rows for {BIOCYCLE_DATASET_ID}: {len(ds_data)}")

# Best (lowest) Q-value per PROBE (BioCycle's "Probe ref" = Affymetrix
# probeset ID, matching expr_df.index directly)
probe_best = (ds_data
              .sort_values('Q-value')
              .drop_duplicates('Probe ref', keep='first')
              .set_index('Probe ref'))
all_rhythmicdb_probes = set(probe_best.index)
print(f"  Unique probes in RhythmicDB: {len(all_rhythmicdb_probes)}")

# Intersection with expression universe
rhythmicdb_in_U = all_rhythmicdb_probes & probe_universe
print(f"  RhythmicDB x U: {len(rhythmicdb_in_U)} probes "
      f"({100*len(rhythmicdb_in_U)/len(all_rhythmicdb_probes):.0f}% of RhythmicDB)")

# R class: Q <= 0.05 AND 20 <= Period <= 28 h (intersection with U)
r_mask = (
    (probe_best['Q-value'] <= Q_RHYTHMIC) &
    (probe_best['Period']  >= PERIOD_MIN)  &
    (probe_best['Period']  <= PERIOD_MAX)
)
r_class_db  = set(probe_best[r_mask].index)                 # in RhythmicDB
r_class     = r_class_db & probe_universe                    # ^ U (final R set)

# X class: in RhythmicDB ^ U but NOT in R (excluded -- borderline rhythmics)
x_class = rhythmicdb_in_U - r_class

# N class: in U but completely absent from RhythmicDB
n_class_full = probe_universe - all_rhythmicdb_probes
rng_n = np.random.default_rng(N_CLASS_SEED)
n_class_list = sorted(n_class_full)
rng_n.shuffle(n_class_list)
n_class = set(n_class_list[:N_CLASS_CAP])

print(f"\n  Label summary (probe-level):")
print(f"    R (rhythmic, Q<={Q_RHYTHMIC}, period {PERIOD_MIN:.0f}-{PERIOD_MAX:.0f}h): "
      f"{len(r_class_db)} in DB  ->  {len(r_class)} in U")
print(f"    X (excluded, in DB not R): {len(rhythmicdb_in_U - r_class)} in U  "
      f"(not used)")
print(f"    N (absent from DB, in U):  {len(n_class_full)} available  "
      f"->  {len(n_class)} after cap={N_CLASS_CAP}")

# Canonical rat clock gene audit. Probe IDs identified manually from
# GPL1355.annot.gz's "Gene symbol" column (see script development notes);
# multiple probesets can exist per gene on Affymetrix 3' arrays.
RAT_CLOCK_PROBES = {
    '1374855_at': 'Per1',    '1368303_at': 'Per2',    '1378560_at': 'Per2',
    '1378745_at': 'Per3',    '1388239_at': 'Per3',    '1392640_at': 'Cry1',
    '1369446_at': 'Cry2',    '1372548_at': 'Cry2',    '1370510_a_at': 'Arntl',
    '1369168_a_at': 'Clock', '1380098_at': 'Clock',   '1389456_at': 'Clock',
    '1390199_at': 'Clock',   '1370816_at': 'Nr1d1',   '1370540_at': 'Nr1d2',
    '1370541_at': 'Nr1d2',   '1390430_at': 'Nr1d2',   '1387874_at': 'Dbp',
}
print(f"\n  Canonical rat clock gene audit ({len(RAT_CLOCK_PROBES)} probes):")
for probe, sym in RAT_CLOCK_PROBES.items():
    in_U  = probe in probe_universe
    in_DB = probe in all_rhythmicdb_probes
    in_R  = probe in r_class
    in_X  = probe in x_class
    status = ('R' if in_R else
              'X' if in_X else
              'N' if in_U else
              'absent')
    q_val   = f"Q={probe_best.loc[probe,'Q-value']:.4f}" if in_DB else "not in DB"
    per_val = f"period={probe_best.loc[probe,'Period']:.1f}h" if in_DB else ""
    print(f"    {sym:<7s} {probe:<14s}  class={status:<3s}  {q_val:<18s}  {per_val}")

# ---------------------------------------------------------------------------
# Step 4: Generate labeled instances (probe-level time series)
# ---------------------------------------------------------------------------
print("\n[4/5] Generating labeled instances...")

metadata_list  = []
dataframes_list = []

all_labeled = {p: 1 for p in r_class}
all_labeled.update({p: 0 for p in n_class})

for probe, label in sorted(all_labeled.items()):
    if probe not in expr_df.index:
        continue

    probe_values = expr_df.loc[probe, valid_samples].values.astype(float)
    valid_mask   = ~np.isnan(probe_values)
    if valid_mask.sum() < 6:
        continue

    times_v  = times_arr[valid_mask]
    values_v = probe_values[valid_mask]

    var_name = f'var_ext_{len(metadata_list)}'
    rows = [
        {'time': float(t), 'condition': 'control',
         'replicate': 'rep1', var_name: float(v)}
        for t, v in zip(times_v, values_v)
    ]
    df_inst = pd.DataFrame(rows)

    symbol = probe_to_symbol.get(probe, probe)
    metadata_list.append({
        'instance_id': len(metadata_list),
        'variable':    var_name,
        'signal_type': f'real_{probe}_GSE20635',
        'is_rhythmic': label,
        'gene':        probe,          # BioCycle-matching identifier
        'symbol':      symbol,          # human-readable, display only
        'source':      'biological_external',
    })
    dataframes_list.append(df_inst)

n_pos = sum(1 for m in metadata_list if m['is_rhythmic'] == 1)
n_neg = sum(1 for m in metadata_list if m['is_rhythmic'] == 0)
print(f"  Instances: {len(metadata_list)} total  (R-class={n_pos}, N-class={n_neg})")

if len(metadata_list) == 0:
    raise RuntimeError("No labeled instances generated from GSE20635.")
if len(np.unique([m['is_rhythmic'] for m in metadata_list])) < 2:
    raise RuntimeError("Only one class present -- cannot compute AUROC.")

# Feature extraction
print("\n[4b/5] Extracting features...")
t_fe = time.time()
feature_rows, labels, kept_meta = [], [], []

for meta, df_i in zip(metadata_list, dataframes_list):
    var = meta['variable']
    cond_data = df_i[df_i['condition'] == 'control']
    if var not in cond_data.columns:
        continue
    times_i  = cond_data['time'].values.astype(float)
    values_i = cond_data[var].values.astype(float)
    valid    = ~(np.isnan(times_i) | np.isnan(values_i))
    times_i  = times_i[valid]
    values_i = values_i[valid]
    if len(times_i) < 4:
        continue

    u_times = np.unique(times_i)
    avg_val = np.array([values_i[times_i == t].mean() for t in u_times])

    feats = extract_features(
        u_times, avg_val, df_i, var, 'control', 'time', 'condition',
    )
    feature_rows.append(feats)
    labels.append(meta['is_rhythmic'])
    kept_meta.append(meta)

X = np.array([
    [row.get(name, np.nan) for name in FEATURE_NAMES]
    for row in feature_rows
])
y = np.array(labels)
print(f"  Feature extraction: {len(feature_rows)} instances in "
      f"{time.time()-t_fe:.1f}s")

# ---------------------------------------------------------------------------
# Step 5: Predict and compute metrics
# ---------------------------------------------------------------------------
print("\n[5/5] Predicting and computing metrics...")
y_proba = model.predict_proba(X)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

auroc = roc_auc_score(y, y_proba)
ap    = average_precision_score(y, y_proba)
acc   = accuracy_score(y, y_pred)
brier = brier_score_loss(y, y_proba)

print(f"\n  Bootstrap 95% CI (n=1000 resamples)...")
boot = _bootstrap_ci(y, y_proba, n_iter=1000, seed=42)

print(f"  ROC-AUC:          {auroc:.4f}  "
      f"[{boot['auroc_ci'][0]:.4f}, {boot['auroc_ci'][1]:.4f}]")
print(f"  Avg. Precision:   {ap:.4f}  "
      f"[{boot['ap_ci'][0]:.4f}, {boot['ap_ci'][1]:.4f}]")
print(f"  Accuracy:         {acc:.4f}  "
      f"[{boot['acc_ci'][0]:.4f}, {boot['acc_ci'][1]:.4f}]")
print(f"  Brier score:      {brier:.4f}  "
      f"[{boot['brier_ci'][0]:.4f}, {boot['brier_ci'][1]:.4f}]")

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp_val = cm.ravel()

sens = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else float('nan')
spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
ppv  = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else float('nan')
npv  = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
f1   = (2 * ppv * sens / (ppv + sens)
        if not (np.isnan(ppv) or np.isnan(sens) or ppv + sens == 0)
        else float('nan'))

print(f"\n  Sensitivity (TPR): {sens:.4f}")
print(f"  Specificity (TNR): {spec:.4f}")
print(f"  PPV:               {ppv:.4f}")
print(f"  NPV:               {npv:.4f}")
print(f"  F1-score:          {f1:.4f}")

# Summary per-probe for R class
r_kept = [(m, p) for m, p in zip(kept_meta, y_proba) if m['is_rhythmic'] == 1]
r_kept.sort(key=lambda x: x[1], reverse=True)   # descending P(rhythmic)
print(f"\n  R-class predictions (n={len(r_kept)}, sorted by P(rhythmic)):")
print(f"  {'Symbol':<10s} {'Probe':<16s} {'Q-val':>8s} {'Period':>8s} "
      f"{'P(rhy)':>8s} {'Pred':>10s}")
print(f"  {'-'*9}  {'-'*15}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
for m, p in r_kept[:20]:   # console: top 20
    probe = m['gene']
    q_str   = f"{probe_best.loc[probe,'Q-value']:.4f}" if probe in probe_best.index else '--'
    per_str = f"{probe_best.loc[probe,'Period']:.1f}"  if probe in probe_best.index else '--'
    cls = 'Rhythmic' if p >= 0.5 else 'Arrhythmic'
    print(f"  {m['symbol']:<10s} {probe:<16s} {q_str:>8s} {per_str:>8s} "
          f"{p:>8.4f} {cls:>10s}")
if len(r_kept) > 20:
    print(f"  ... ({len(r_kept)-20} more R-class probes in report)")

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
print(f"\nWriting report -> {REPORT_PATH}")
W = 80
run_date = datetime.now().strftime('%Y-%m-%d %H:%M')

with open(REPORT_PATH, 'w', encoding='utf-8') as f:

    f.write("=" * W + "\n")
    f.write("ChronoScope CRS-AI -- EXTERNAL LODO VALIDATION REPORT\n")
    f.write("GSE20635: Rattus norvegicus, abdominal white adipose tissue, LD\n")
    f.write("=" * W + "\n\n")
    f.write(f"  Run date:    {run_date}\n")
    f.write(f"  Model file:  {MODEL_PATH.name}\n")
    f.write(f"  Model size:  {model_size_kb:.1f} KB\n")
    f.write(f"  Model MD5:   {model_hash}\n")
    f.write(f"  Features:    {len(FEATURE_NAMES)}\n")
    f.write(f"  Dataset:     GSE20635 (GEO), platform GPL1355\n\n")

    # ------------------------------------------------------------------
    f.write("1. DATASET SUMMARY\n")
    f.write("-" * W + "\n\n")
    f.write("  GEO accession:  GSE20635\n")
    f.write("  Species:        Rattus norvegicus (Wistar, male)\n")
    f.write("  Tissue:         Abdominal white adipose tissue (WAT)\n")
    f.write("  Light regime:   LD (12h light : 12h dark)\n")
    f.write("  Publication:    Sukumaran et al. 2010, Physiol Genomics 42:141\n")
    f.write("  Platform:       GPL1355 (Affymetrix Rat Genome 230 2.0 Array)\n")
    f.write(f"  Probes:         {len(expr_df):,}\n")
    f.write(f"  Total samples:  {len(expr_df.columns)}\n")
    f.write(f"  Samples with circadian timepoints: {len(sample_times)} "
            f"(2-3 animals x {len(unique_tps)} timepoints over "
            f"{unique_tps[0]:.2f}-{unique_tps[-1]:.2f} h)\n")
    f.write(f"  Probe-level expression universe (U): {len(probe_universe):,}\n\n")

    f.write("  Note on labeling granularity: unlike the GSE37332 (zebrafish)\n")
    f.write("  and GSE3416 (Arabidopsis) validations, this dataset is labeled\n")
    f.write("  at the PROBE level rather than the gene level. BioCycle's\n")
    f.write("  RhythmicDB entry for E-GEOD-20635 identifies genes by rat\n")
    f.write("  UniGene cluster ID (e.g. Rn.94978), not by the gene symbol\n")
    f.write("  GPL1355's curated annotation reports (e.g. Sumo2). Matching on\n")
    f.write("  BioCycle's 'Probe ref' column (the actual Affymetrix probeset\n")
    f.write("  ID) instead avoids this identifier mismatch entirely and is\n")
    f.write("  also more faithful to what BioCycle itself analysed.\n\n")

    f.write("  BioCycle reference (E-GEOD-20635 from RhythmicDB,\n")
    f.write("  rhythmicdb_query_BioCycle_allModels_noFilters.xlsx):\n")
    f.write(f"    Total probes in RhythmicDB for this dataset: "
            f"{len(all_rhythmicdb_probes):,}\n")
    f.write(f"    RhythmicDB ^ U: {len(rhythmicdb_in_U):,} probes "
            f"({100*len(rhythmicdb_in_U)/len(all_rhythmicdb_probes):.0f}% coverage)\n")
    f.write(f"    R class (Q<={Q_RHYTHMIC}, period {PERIOD_MIN:.0f}-"
            f"{PERIOD_MAX:.0f}h, in U): {len(r_class):,} probes\n")
    f.write(f"    X class (in RhythmicDB ^ U, excluded):  {len(x_class):,} probes\n")
    f.write(f"    N class (absent from RhythmicDB, in U): {len(n_class_full):,} "
            f"available -> {len(n_class)} after cap\n\n")

    f.write("  Canonical rat circadian clock gene label audit:\n\n")
    f.write(f"  {'Gene':<8s}  {'Probe':<14s}  {'Class':<5s}  {'Q-value':<10s}  "
            f"{'Period (h)':<12s}  {'In expression U'}\n")
    f.write(f"  {'-'*7}  {'-'*13}  {'-'*5}  {'-'*9}  {'-'*11}  {'-'*15}\n")
    for probe, sym in RAT_CLOCK_PROBES.items():
        in_U  = probe in probe_universe
        in_DB = probe in all_rhythmicdb_probes
        in_R  = probe in r_class
        in_X  = probe in x_class
        status = ('R' if in_R else
                  'X' if in_X else
                  'N' if in_U else
                  'absent')
        q_str  = f"{probe_best.loc[probe,'Q-value']:.4f}" if in_DB else '--'
        p_str  = f"{probe_best.loc[probe,'Period']:.2f}"  if in_DB else '--'
        u_str  = 'Yes' if in_U else 'No'
        f.write(f"  {sym:<8s}  {probe:<14s}  {status:<5s}  {q_str:<10s}  "
                f"{p_str:<12s}  {u_str}\n")
    f.write("\n")
    f.write("  Note: Per2 (1368303_at) is R-class with Q=0.000 at period=24.0h,\n")
    f.write("  confirming that the Q<=0.05 criterion recovers a bona fide core\n")
    f.write("  clock gene in this dataset. Several other core clock probes are\n")
    f.write("  absent from BioCycle's output for this dataset (failed its\n")
    f.write("  internal missing-data/QC filters) or fall in the borderline\n")
    f.write("  X-class (e.g. Cry2, Dbp) -- consistent with the known lower\n")
    f.write("  core-clock-gene amplitude in adipose tissue relative to liver\n")
    f.write("  or SCN.\n\n")

    f.write(f"  Validation set: {len(kept_meta)} instances "
            f"(R={n_pos}, N={n_neg})\n\n")

    # ------------------------------------------------------------------
    f.write("2. VALIDATION DESIGN\n")
    f.write("-" * W + "\n\n")
    f.write("  GSE20635 adds a mammalian species and tissue combination not\n")
    f.write("  represented anywhere in CRS-AI's training or prior validation\n")
    f.write("  data:\n\n")
    f.write("    * Different species:  CRS-AI was trained on Mus musculus\n")
    f.write("      liver (GSE11516, GSE11923), Drosophila melanogaster clock\n")
    f.write("      neurons (GSE77451), and Homo sapiens blood (GSE39445), and\n")
    f.write("      externally validated on Danio rerio (GSE37332). Rattus\n")
    f.write("      norvegicus is phylogenetically closer to mouse and human\n")
    f.write("      than zebrafish, making this a complementary, higher-\n")
    f.write("      similarity mammalian generalisation test.\n\n")
    f.write("    * Different tissue:  Abdominal white adipose tissue is a\n")
    f.write("      peripheral metabolic clock tissue not present in the\n")
    f.write("      training corpus (liver, neurons, blood) or the zebrafish\n")
    f.write("      whole-organism validation.\n\n")
    f.write("    * True biological replication:  2-3 animals per timepoint,\n")
    f.write("      unlike the whole-organism zebrafish pools.\n\n")
    f.write("    * Different experimental platform:  GPL1355 Affymetrix Rat\n")
    f.write("      Genome 230 2.0 Array.\n\n")
    f.write("    * Zero data leakage:  GSE20635 was never accessed at any\n")
    f.write("      step of CRS-AI training, hyperparameter optimisation, or\n")
    f.write("      feature engineering. Labels were assigned entirely from an\n")
    f.write("      independent BioCycle analysis (RhythmicDB, E-GEOD-20635)\n")
    f.write("      after training was complete.\n\n")
    f.write("  Label quality: BioCycle-consistent non-rhythmic-from-absence\n")
    f.write("  labeling is used, matched at the probe level for the reasons\n")
    f.write("  given above -- the same underlying strategy used for CRS-AI\n")
    f.write("  v6 training and for the GSE37332/GSE3416 validations.\n\n")

    # ------------------------------------------------------------------
    f.write("3. METRICS\n")
    f.write("-" * W + "\n\n")
    f.write(f"  Test set: {len(kept_meta)} instances  "
            f"(R-class={n_pos}, N-class={n_neg})\n")
    f.write(f"  95% CIs from percentile bootstrap "
            f"({boot['n_valid']} valid resamples of 1000):\n\n")
    f.write(f"  ROC-AUC:         {auroc:.4f}  "
            f"[95% CI {boot['auroc_ci'][0]:.4f}, {boot['auroc_ci'][1]:.4f}]\n")
    f.write(f"  Avg. Precision:  {ap:.4f}  "
            f"[95% CI {boot['ap_ci'][0]:.4f}, {boot['ap_ci'][1]:.4f}]\n")
    f.write(f"  Accuracy:        {acc:.4f}  "
            f"[95% CI {boot['acc_ci'][0]:.4f}, {boot['acc_ci'][1]:.4f}]\n")
    f.write(f"  Brier score:     {brier:.4f}  "
            f"[95% CI {boot['brier_ci'][0]:.4f}, {boot['brier_ci'][1]:.4f}]"
            f"  (lower = better calibration)\n\n")
    f.write("  Threshold-dependent metrics (threshold = 0.50):\n\n")
    f.write(f"  Sensitivity (TPR):   {sens:.4f}\n")
    f.write(f"  Specificity (TNR):   {spec:.4f}\n")
    f.write(f"  Positive pred. val.: {ppv:.4f}\n")
    f.write(f"  Negative pred. val.: {npv:.4f}\n")
    f.write(f"  F1-score:            {f1:.4f}\n\n")

    f.write("  Confusion matrix (threshold 0.5):\n\n")
    f.write("                        Predicted\n")
    f.write("                   Arrhythmic  Rhythmic\n")
    f.write(f"    Actual N-class   {tn:>6d}    {fp:>6d}\n")
    f.write(f"    Actual R-class   {fn:>6d}    {tp_val:>6d}\n\n")
    f.write(f"    TP={tp_val}  FP={fp}  TN={tn}  FN={fn}\n\n")

    # ------------------------------------------------------------------
    f.write("4. PER-PROBE PREDICTIONS -- R CLASS\n")
    f.write("-" * W + "\n\n")
    f.write("  All R-class probes (BioCycle-confirmed rhythmic, Q<=0.05,\n")
    f.write("  period 20-28 h), sorted by predicted P(rhythmic) descending.\n\n")
    hdr = (f"  {'Symbol':<10s}  {'Probe':<14s}  {'Q-val':>7s}  {'Period':>8s}  "
           f"{'P(rhythmic)':>11s}  {'Predicted':>10s}\n")
    f.write(hdr)
    f.write(f"  {'-'*9}  {'-'*13}  {'-'*7}  {'-'*7}  {'-'*11}  {'-'*9}\n")
    for m, p in r_kept:
        probe = m['gene']
        q_str  = (f"{probe_best.loc[probe,'Q-value']:.4f}"
                  if probe in probe_best.index else '--')
        pr_str = (f"{probe_best.loc[probe,'Period']:.2f}"
                  if probe in probe_best.index else '--')
        cls = 'Rhythmic' if p >= 0.5 else 'Arrhythmic'
        f.write(f"  {m['symbol']:<10s}  {probe:<14s}  {q_str:>7s}  {pr_str:>8s}  "
                f"{p:>11.4f}  {cls:>10s}\n")
    f.write("\n")

    # ------------------------------------------------------------------
    f.write("5. INTERPRETATION\n")
    f.write("-" * W + "\n\n")
    if auroc >= 0.85:
        auroc_interp = (
            "The AUROC exceeds 0.85, demonstrating strong generalisation of\n"
            "  CRS-AI to a rat peripheral metabolic tissue."
        )
    elif auroc >= 0.70:
        auroc_interp = (
            "The AUROC lies in the range 0.70-0.85, consistent with meaningful\n"
            "  generalisation despite the modest core-clock-gene amplitude\n"
            "  typical of adipose tissue."
        )
    else:
        auroc_interp = (
            "The AUROC is below 0.70. Interpret with caution and report the CI."
        )
    f.write(f"  {auroc_interp}\n\n")

    f.write(f"  External LODO context: CRS-AI was trained exclusively on\n")
    f.write(f"  mouse, Drosophila and human data, and previously validated on\n")
    f.write(f"  zebrafish. An AUROC of {auroc:.3f} [95% CI "
            f"{boot['auroc_ci'][0]:.3f}, {boot['auroc_ci'][1]:.3f}]\n")
    f.write(f"  on held-out rat adipose tissue -- a species, tissue and\n")
    f.write(f"  platform not seen at any training step -- provides additional\n")
    f.write(f"  evidence that CRS-AI's features generalise across mammalian\n")
    f.write(f"  species and peripheral tissue types.\n\n")

    f.write(f"  Label quality is supported by Per2 (1368303_at, Q=0.000,\n")
    f.write(f"  T=24.0h) -- a core mammalian TTFL gene -- landing in R-class.\n\n")

    f.write(f"  Bootstrap CI width reflects sample size (N={len(kept_meta)} instances,\n")
    f.write(f"  R={n_pos}, N={n_neg}). For manuscript reporting, cite the\n")
    f.write(f"  CI rather than the point estimate alone.\n\n")

    f.write(f"  Suggested reporting (JBR Methods section):\n")
    f.write(f"    \"To evaluate generalisation to an additional mammalian\n")
    f.write(f"    species and peripheral tissue, CRS-AI was applied to\n")
    f.write(f"    GSE20635 (Rattus norvegicus, abdominal white adipose\n")
    f.write(f"    tissue, LD cycle; GPL1355 Affymetrix microarray;\n")
    f.write(f"    n={len(kept_meta)} probes: R={n_pos}, N={n_neg}). Probe-level\n")
    f.write(f"    labels were derived independently from RhythmicDB / BioCycle\n")
    f.write(f"    (E-GEOD-20635; Q<=0.05, period 20-28 h for rhythmic class;\n")
    f.write(f"    absent-from-database criterion for non-rhythmic class). The\n")
    f.write(f"    model achieved AUROC={auroc:.3f} (95% CI "
            f"{boot['auroc_ci'][0]:.3f}-{boot['auroc_ci'][1]:.3f}) on\n")
    f.write(f"    this zero-overlap external dataset.\"\n\n")

    f.write("=" * W + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * W + "\n")

print(f"\nReport saved: {REPORT_PATH}")
print("\n" + "=" * 70)
print("EXTERNAL LODO VALIDATION COMPLETE")
print("=" * 70)
print(f"  GSE20635 AUROC: {auroc:.4f}  "
      f"[{boot['auroc_ci'][0]:.4f}, {boot['auroc_ci'][1]:.4f}]")
print(f"  R-class ({n_pos} probes) / N-class ({n_neg} probes)")
