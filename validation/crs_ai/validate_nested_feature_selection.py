"""
Nested cross-validation check -- feature-selection / holdout leakage
======================================================================

Addresses JBR reviewer comment (Reviewer 2, Major Point 3; Plan de
Revision section 5): CRS-AI's 11 model features were chosen by
permutation importance computed on the SAME GroupShuffleSplit holdout
that is later used to report the model's test performance (0.9474
ROC-AUC in training_report.txt). That is textbook feature-selection
leakage -- the holdout is not fully independent of the modeling
decisions made upstream of it.

This script does NOT retrain or replace the production model
(consensus_rf_model.pkl / feature_names.json are left untouched). It
instead runs a proper nested cross-validation as an additional
robustness check:

  Outer loop:  5-fold StratifiedGroupKFold (grouped by gene, same
               grouping convention as train_consensus_model.py) over
               the full training corpus. Each outer fold's test split
               is held out from EVERYTHING in that fold's inner loop.

  Inner step:  Within each outer-train split, a GroupShuffleSplit
               (80/20, grouped by gene) carves out an inner
               validation set. Permutation importance (n_repeats=10,
               scoring=ROC-AUC) is computed on that inner validation
               set only, over the full 18-feature candidate pool
               (11 production features + 7 candidates dropped in the
               v1->v2 cut). A candidate feature is retained if its
               mean permutation drop in ROC-AUC exceeds 1 SD of its
               own shuffle noise -- the exact criterion documented in
               Supplementary Table S1.

  Final fit:   A fresh pipeline (SimpleImputer -> CalibratedClassifierCV
               (RandomForest), identical hyperparameters to
               train_consensus_model.py) is trained on the FULL outer-
               train split using only the features selected by the
               inner step, then evaluated on the outer-test split,
               which never participated in feature selection.

  Aggregate:   Out-of-fold predictions from all 5 outer folds are
               pooled (each of the 5649 training instances gets
               exactly one out-of-fold prediction) into a single
               ROC-AUC/F1/Brier/Accuracy estimate with bootstrap CI,
               directly comparable to the single-partition 0.9474
               reported in the paper. Feature-selection stability
               across the 5 outer folds is also reported.

  External check: each outer fold's fitted model is additionally
               evaluated on GSE37332 (zebrafish), which never
               participated in training or feature engineering at any
               step, mirroring validate_external_holdout_gse37332.py.

Usage
-----
Run from the project root after training (uses the same training
corpus generators as train_consensus_model.py):
    python validation/crs_ai/validate_nested_feature_selection.py
"""

import sys
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR           = Path(__file__).resolve().parent
PROJECT_ROOT         = SCRIPT_DIR.parent.parent
MODEL_DIR            = PROJECT_ROOT / 'core' / 'models'
TRAINING_DIR         = PROJECT_ROOT / 'training_data_meta_classifier'
GEO_CACHE_DIR        = TRAINING_DIR / 'data' / 'geo'
REPORT_PATH          = SCRIPT_DIR / 'validate_nested_feature_selection.txt'
TRAINING_REPORT_PATH = SCRIPT_DIR / 'training_report.txt'
BIOCYCLE_XLSX        = (TRAINING_DIR /
                  'rhythmicdb_query_BioCycle_allModels_noFilters.xlsx')

# GSE37332 / GPL14664 files (external check, cached -- see
# validate_external_holdout_gse37332.py)
ZFISH_SERIES_MATRIX = GEO_CACHE_DIR / 'GSE37332_series_matrix.txt.gz'
ZFISH_PLATFORM_SOFT = GEO_CACHE_DIR / 'GPL14664_family.soft.gz'
ZFISH_DATASET_ID     = 'E-GEOD-37332_LD'
ZFISH_Q_RHYTHMIC     = 0.05
ZFISH_PERIOD_MIN     = 20.0
ZFISH_PERIOD_MAX     = 28.0
ZFISH_N_CAP          = 300
ZFISH_N_SEED         = 42

N_OUTER_FOLDS  = 5
OUTER_SEED     = 42
INNER_SEED     = 42
PERM_N_REPEATS = 10
BOOT_N_ITER    = 1000
BOOT_SEED      = 42

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRAINING_DIR))

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Imports after path setup
# ---------------------------------------------------------------------------
from core.feature_extraction import extract_features, FEATURE_NAMES
from generate_synthetic_training_data import generate_training_instances
from generate_real_training_data import (
    generate_from_geo,
    generate_from_GSE77451,
    generate_from_GSE39445,
    parse_series_matrix,
    parse_platform_annotation,
    map_expression_to_genes,
    extract_timepoints_from_samples,
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedGroupKFold, GroupShuffleSplit,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
)

# ---------------------------------------------------------------------------
# Candidate feature pool (18 = 11 production features + 7 dropped in v1->v2)
# extract_features() still computes all 18 and returns them in its dict --
# only FEATURE_NAMES (the model's input vector) was trimmed. See the v1->v2
# note at the top of core/feature_extraction.py.
# ---------------------------------------------------------------------------
DROPPED_V1_FEATURES = [
    'jtk_tau', 'jtk_period', 'harmonic_p_value', 'harmonic_r_squared',
    'method_agreement', 'period_concordance', 'log_min_p_value',
]
CANDIDATE_FEATURE_NAMES = list(FEATURE_NAMES) + DROPPED_V1_FEATURES

RF_KWARGS = dict(
    n_estimators=200, max_depth=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42, n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Bootstrap CI helper (percentile method, same convention as the other
# validation scripts in this directory)
# ---------------------------------------------------------------------------
def _bootstrap_ci(y_true, y_pred, y_proba, n_iter=BOOT_N_ITER, ci=0.95,
                   seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, aurocs, f1s, briers = [], [], [], []

    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yp, ypr = y_pred[idx], y_proba[idx]
        accs.append(accuracy_score(yt, yp))
        aurocs.append(roc_auc_score(yt, ypr))
        f1s.append(f1_score(yt, yp))
        briers.append(brier_score_loss(yt, ypr))

    alpha = (1 - ci) / 2
    lo, hi = alpha * 100, (1 - alpha) * 100
    return {
        'n_valid': len(accs),
        'accuracy':  (float(np.mean(accs)), float(np.percentile(accs, lo)), float(np.percentile(accs, hi))),
        'auroc':     (float(np.mean(aurocs)), float(np.percentile(aurocs, lo)), float(np.percentile(aurocs, hi))),
        'f1':        (float(np.mean(f1s)), float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))),
        'brier':     (float(np.mean(briers)), float(np.percentile(briers, lo)), float(np.percentile(briers, hi))),
    }


def _parse_reported_holdout(training_report_path):
    """Pull the already-reported single-partition holdout metrics (section
    6.2) out of training_report.txt, so this script always compares against
    whatever the production model currently reports rather than a stale
    hardcoded number."""
    text = training_report_path.read_text(encoding='utf-8')
    section = text.split('6.2 Holdout test set performance')[1]
    section = section.split('6.3 Confusion matrix')[0]

    def _grab(label):
        m = re.search(
            re.escape(label) + r'\s*:\s*([\d.]+)\s*\[95% CI ([\d.]+), ([\d.]+)\]',
            section,
        )
        return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else None

    return {
        'accuracy': _grab('Accuracy'),
        'f1':       _grab('F1 Score'),
        'auroc':    _grab('ROC-AUC'),
        'brier':    _grab('Brier loss'),
    }


# ---------------------------------------------------------------------------
# Production-identical pipeline factory (duplicated from
# train_consensus_model.py -- that module wraps everything inside main(),
# so it is re-implemented here rather than imported, to avoid re-running
# the full production training pipeline as a side effect of an import).
# ---------------------------------------------------------------------------
def _build_pipeline(X_fit, y_fit, groups_fit):
    base_rf = RandomForestClassifier(**RF_KWARGS)
    calibration_splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        .split(X_fit, y_fit, groups=groups_fit)
    )
    calibrated_rf = CalibratedClassifierCV(
        estimator=base_rf, method='isotonic', cv=calibration_splits,
    )
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
        ('classifier', calibrated_rf),
    ])


def _select_features_by_permutation(X_inner_train, y_inner_train,
                                     X_inner_val, y_inner_val,
                                     candidate_names):
    """Table S1 criterion: retain a candidate feature iff its mean
    permutation drop in holdout ROC-AUC exceeds 1 SD of its own
    per-shuffle noise (n_repeats=10). Uses a plain RF (no calibration --
    calibration is irrelevant to feature ranking and just adds cost)."""
    selection_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
        ('classifier', RandomForestClassifier(**RF_KWARGS)),
    ])
    selection_pipeline.fit(X_inner_train, y_inner_train)

    perm = permutation_importance(
        selection_pipeline, X_inner_val, y_inner_val,
        n_repeats=PERM_N_REPEATS, random_state=42, n_jobs=-1,
        scoring='roc_auc',
    )

    rows = []
    selected = []
    for name, mean, std in zip(candidate_names, perm.importances_mean, perm.importances_std):
        retained = mean > std
        rows.append({'feature': name, 'mean_delta_auroc': float(mean),
                      'std': float(std), 'retained': retained})
        if retained:
            selected.append(name)
    return selected, rows


# ---------------------------------------------------------------------------
# Step 1: Build the training corpus (identical to train_consensus_model.py
# Steps 1a/1b -- same seeds, same caps, same starting_ids, so this is the
# exact same 5649-instance corpus the production model was trained on).
# ---------------------------------------------------------------------------
def load_training_corpus():
    print("[1/6] Building training corpus (same generators/seeds as "
          "train_consensus_model.py)...")

    metadata, dataframes = generate_training_instances(seed=42)
    n_synth = len(metadata)
    print(f"  Synthetic instances: {n_synth}")

    n_real = 0

    try:
        real1_meta, real1_dfs = generate_from_geo(
            geo_accession='GSE11923', platform_id='GPL1261',
            starting_id=n_synth + 1000, subsample_intervals=[2.0, 4.0],
        )
        metadata += real1_meta
        dataframes += real1_dfs
        n_real += len(real1_meta)
        print(f"  GSE11923: {len(real1_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE11923 failed: {e}")

    try:
        real2_meta, real2_dfs, _amb_meta, _amb_dfs = generate_from_geo(
            geo_accession='GSE11516', platform_id='GPL6880',
            biocycle_xlsx=str(BIOCYCLE_XLSX),
            biocycle_dataset_id='E-GEOD-11516',
            biocycle_q_threshold=0.01, biocycle_period_min=20.0,
            biocycle_period_max=28.0, non_rhythmic_from_absence=True,
            max_rhythmic=800, max_non_rhythmic=800,
            starting_id=n_synth + 2000 + n_real,
            subsample_intervals=[], return_ambiguous=True,
        )
        metadata += real2_meta
        dataframes += real2_dfs
        n_real += len(real2_meta)
        print(f"  GSE11516: {len(real2_meta)} instances "
              f"(ambiguous borderline-q genes excluded, as in production)")
    except Exception as e:
        print(f"  WARNING: GSE11516 failed: {e}")

    try:
        gse77451_meta, gse77451_dfs = generate_from_GSE77451(
            abruzzi_xlsx_path=TRAINING_DIR / 'abruzzi_2017_cycling.xlsx',
            geo_cache_dir=GEO_CACHE_DIR,
            starting_instance_id=n_synth + 10000,
            max_positives_per_cell_type=200, hc_only=True, seed=42,
        )
        metadata += gse77451_meta
        dataframes += gse77451_dfs
        n_real += len(gse77451_meta)
        print(f"  GSE77451: {len(gse77451_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE77451 failed: {e}")

    try:
        gse39445_meta, gse39445_dfs = generate_from_GSE39445(
            moller_xlsx_path=TRAINING_DIR / 'moller_levet_2013_circadian.xlsx',
            geo_cache_dir=GEO_CACHE_DIR,
            starting_instance_id=n_synth + 20000,
            max_per_class=800, seed=42,
        )
        metadata += gse39445_meta
        dataframes += gse39445_dfs
        n_real += len(gse39445_meta)
        print(f"  GSE39445: {len(gse39445_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE39445 failed: {e}")

    print(f"  Total: {len(metadata)} instances "
          f"({n_synth} synthetic + {n_real} real)")
    return metadata, dataframes


def extract_candidate_features(metadata, dataframes, label="instances"):
    """Extract the full 18-feature candidate pool (not just the 11
    production features) for every instance."""
    n_total = len(metadata)
    feature_rows, labels, kept_metadata = [], [], []
    t0 = time.time()

    for i, (meta, df) in enumerate(zip(metadata, dataframes)):
        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  Extracting {label} {i + 1}/{n_total} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        variable, condition = meta['variable'], 'control'
        cond_data = df[df['condition'] == condition]
        if variable not in cond_data.columns:
            continue
        times = cond_data['time'].values.astype(float)
        values = cond_data[variable].values.astype(float)
        valid = ~(np.isnan(times) | np.isnan(values))
        times, values = times[valid], values[valid]
        if len(times) < 4:
            continue

        unique_times = np.unique(times)
        avg_values = np.array([values[times == t].mean() for t in unique_times])

        feats = extract_features(
            unique_times, avg_values, df, variable, condition,
            'time', 'condition',
        )
        feature_rows.append(feats)
        labels.append(meta['is_rhythmic'])
        kept_metadata.append(meta)

    X = np.array([[row.get(name, np.nan) for name in CANDIDATE_FEATURE_NAMES]
                  for row in feature_rows])
    y = np.array(labels)
    print(f"  Feature extraction complete: {len(feature_rows)} {label} "
          f"in {time.time() - t0:.1f}s")
    return X, y, kept_metadata


# ---------------------------------------------------------------------------
# Step 2: Load GSE37332 (zebrafish) external check -- identical labeling
# logic to validate_external_holdout_gse37332.py, extended to the full
# 18-feature candidate pool.
# ---------------------------------------------------------------------------
def load_zebrafish_external_check():
    print("\n[2/6] Loading GSE37332 (zebrafish) external check dataset...")

    for path, label in [(ZFISH_SERIES_MATRIX, "GSE37332 series matrix"),
                         (ZFISH_PLATFORM_SOFT, "GPL14664 platform annotation")]:
        if not path.exists():
            print(f"  WARNING: {label} not found at {path} -- skipping "
                  f"external check.")
            return None, None, None

    expr_df, sinfo = parse_series_matrix(str(ZFISH_SERIES_MATRIX))
    probe_to_gene = parse_platform_annotation(str(ZFISH_PLATFORM_SOFT))
    gene_expr = map_expression_to_genes(expr_df, probe_to_gene)
    gene_universe = set(gene_expr.index)

    sample_times = extract_timepoints_from_samples(sinfo)
    valid_samples = [s for s in gene_expr.columns if s in sample_times]
    valid_samples.sort(key=lambda s: sample_times[s])
    times_arr = np.array([sample_times[s] for s in valid_samples])
    times_arr -= times_arr.min()

    bc_df = pd.read_excel(str(BIOCYCLE_XLSX))
    ds_data = bc_df[bc_df['Dataset'] == ZFISH_DATASET_ID].copy()
    gene_best = (ds_data.sort_values('Q-value')
                 .drop_duplicates('Gene info', keep='first')
                 .set_index('Gene info'))
    all_rhythmicdb_genes = set(gene_best.index)

    r_mask = ((gene_best['Q-value'] <= ZFISH_Q_RHYTHMIC) &
              (gene_best['Period'] >= ZFISH_PERIOD_MIN) &
              (gene_best['Period'] <= ZFISH_PERIOD_MAX))
    r_class = set(gene_best[r_mask].index) & gene_universe

    n_class_full = gene_universe - all_rhythmicdb_genes
    rng_n = np.random.default_rng(ZFISH_N_SEED)
    n_class_list = sorted(n_class_full)
    rng_n.shuffle(n_class_list)
    n_class = set(n_class_list[:ZFISH_N_CAP])

    gene_index_lower = {str(g).lower(): str(g) for g in gene_expr.index}
    all_labeled = {g: 1 for g in r_class}
    all_labeled.update({g: 0 for g in n_class})

    metadata_list, dataframes_list = [], []
    for gene, label in sorted(all_labeled.items()):
        gkey = gene if gene in gene_expr.index else gene_index_lower.get(gene.lower())
        if gkey is None:
            continue
        gene_values = gene_expr.loc[gkey, valid_samples].values.astype(float)
        valid_mask = ~np.isnan(gene_values)
        if valid_mask.sum() < 6:
            continue
        times_v, values_v = times_arr[valid_mask], gene_values[valid_mask]
        var_name = f'var_zfish_{len(metadata_list)}'
        df_inst = pd.DataFrame([
            {'time': float(t), 'condition': 'control', var_name: float(v)}
            for t, v in zip(times_v, values_v)
        ])
        metadata_list.append({
            'instance_id': len(metadata_list), 'variable': var_name,
            'is_rhythmic': label, 'gene': gene, 'source': 'biological_external',
        })
        dataframes_list.append(df_inst)

    n_pos = sum(1 for m in metadata_list if m['is_rhythmic'] == 1)
    n_neg = sum(1 for m in metadata_list if m['is_rhythmic'] == 0)
    print(f"  Instances: {len(metadata_list)} (R={n_pos}, N={n_neg})")

    X_zfish, y_zfish, kept_zfish = extract_candidate_features(
        metadata_list, dataframes_list, label="zebrafish instances"
    )
    return X_zfish, y_zfish, kept_zfish


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("NESTED CROSS-VALIDATION -- FEATURE-SELECTION / HOLDOUT LEAKAGE CHECK")
    print("=" * 70)

    metadata, dataframes = load_training_corpus()

    print("\n[1b/6] Extracting 18-feature candidate pool "
          f"({', '.join(CANDIDATE_FEATURE_NAMES)})...")
    X, y, kept_metadata = extract_candidate_features(
        metadata, dataframes, label="training instances"
    )
    groups = np.array([
        m['gene'] if 'gene' in m else f"synth_{m['instance_id']}"
        for m in kept_metadata
    ])
    print(f"  X shape: {X.shape}  |  groups: {len(np.unique(groups))} unique")

    X_zfish, y_zfish, kept_zfish = load_zebrafish_external_check()

    reported = _parse_reported_holdout(TRAINING_REPORT_PATH)
    print(f"\n  Already-reported single-partition holdout (training_report.txt):")
    print(f"    ROC-AUC: {reported['auroc']}")

    # -----------------------------------------------------------------
    # Step 3: Outer 5-fold StratifiedGroupKFold, grouped by gene
    # -----------------------------------------------------------------
    print(f"\n[3/6] Outer {N_OUTER_FOLDS}-fold StratifiedGroupKFold "
          f"(grouped by gene, random_state={OUTER_SEED})...")
    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    outer_folds = list(outer_cv.split(X, y, groups))

    fold_results = []
    oof_y_true = np.full(len(y), np.nan)
    oof_y_proba = np.full(len(y), np.nan)

    for fold_idx, (train_ext_idx, test_ext_idx) in enumerate(outer_folds):
        print(f"\n  --- Outer fold {fold_idx + 1}/{N_OUTER_FOLDS} ---")
        X_train_ext, X_test_ext = X[train_ext_idx], X[test_ext_idx]
        y_train_ext, y_test_ext = y[train_ext_idx], y[test_ext_idx]
        groups_train_ext = groups[train_ext_idx]

        overlap = set(groups[train_ext_idx]) & set(groups[test_ext_idx])
        assert len(overlap) == 0, f"Fold {fold_idx}: group leakage {overlap}"

        print(f"    Outer-train: {len(train_ext_idx)}  "
              f"Outer-test: {len(test_ext_idx)}  (no group overlap)")

        # --- Inner split: feature selection NEVER touches outer-test ---
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                       random_state=INNER_SEED)
        inner_train_idx, inner_val_idx = next(
            gss_inner.split(X_train_ext, y_train_ext, groups_train_ext)
        )
        inner_overlap = (set(groups_train_ext[inner_train_idx]) &
                          set(groups_train_ext[inner_val_idx]))
        assert len(inner_overlap) == 0, \
            f"Fold {fold_idx}: inner group leakage {inner_overlap}"

        selected, perm_rows = _select_features_by_permutation(
            X_train_ext[inner_train_idx], y_train_ext[inner_train_idx],
            X_train_ext[inner_val_idx], y_train_ext[inner_val_idx],
            CANDIDATE_FEATURE_NAMES,
        )
        print(f"    Inner selection: {len(inner_train_idx)} inner-train / "
              f"{len(inner_val_idx)} inner-val -> {len(selected)} features "
              f"retained: {sorted(selected)}")

        selected_idx = [CANDIDATE_FEATURE_NAMES.index(f) for f in selected]

        # --- Final fold model: full outer-train, selected features only ---
        pipeline = _build_pipeline(
            X_train_ext[:, selected_idx], y_train_ext, groups_train_ext
        )
        pipeline.fit(X_train_ext[:, selected_idx], y_train_ext)

        y_proba = pipeline.predict_proba(X_test_ext[:, selected_idx])[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        boot = _bootstrap_ci(y_test_ext, y_pred, y_proba)
        print(f"    Outer-test ROC-AUC: {boot['auroc'][0]:.4f}  "
              f"[{boot['auroc'][1]:.4f}, {boot['auroc'][2]:.4f}]")

        oof_y_true[test_ext_idx] = y_test_ext
        oof_y_proba[test_ext_idx] = y_proba

        # --- Zebrafish external check with this fold's model ---
        zfish_auroc = None
        if X_zfish is not None:
            zfish_proba = pipeline.predict_proba(X_zfish[:, selected_idx])[:, 1]
            zfish_auroc = roc_auc_score(y_zfish, zfish_proba)
            print(f"    GSE37332 (zebrafish) AUROC with this fold's model: "
                  f"{zfish_auroc:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'n_train_ext': len(train_ext_idx),
            'n_test_ext': len(test_ext_idx),
            'n_inner_train': len(inner_train_idx),
            'n_inner_val': len(inner_val_idx),
            'selected_features': selected,
            'perm_table': perm_rows,
            'boot': boot,
            'zfish_auroc': zfish_auroc,
        })

    # -----------------------------------------------------------------
    # Step 4: Pooled out-of-fold metrics (directly comparable to the
    # 0.9474 single-partition holdout)
    # -----------------------------------------------------------------
    print(f"\n[4/6] Pooling out-of-fold predictions "
          f"(n={len(oof_y_true)}, one prediction per instance)...")
    assert not np.isnan(oof_y_true).any(), "Some instances got no OOF prediction"
    oof_y_true = oof_y_true.astype(int)
    oof_y_pred = (oof_y_proba >= 0.5).astype(int)
    pooled_boot = _bootstrap_ci(oof_y_true, oof_y_pred, oof_y_proba)
    print(f"  Pooled nested-CV ROC-AUC: {pooled_boot['auroc'][0]:.4f}  "
          f"[{pooled_boot['auroc'][1]:.4f}, {pooled_boot['auroc'][2]:.4f}]")
    print(f"  Reported single-partition ROC-AUC: {reported['auroc'][0]:.4f}  "
          f"[{reported['auroc'][1]:.4f}, {reported['auroc'][2]:.4f}]")

    # -----------------------------------------------------------------
    # Step 5: Feature-selection stability across outer folds
    # -----------------------------------------------------------------
    print("\n[5/6] Feature-selection stability across outer folds...")
    stability = {}
    for name in CANDIDATE_FEATURE_NAMES:
        n_folds_selected = sum(
            1 for fr in fold_results if name in fr['selected_features']
        )
        stability[name] = n_folds_selected
        flag = " <-- production feature" if name in FEATURE_NAMES else ""
        print(f"    {name:<22s} selected in {n_folds_selected}/{N_OUTER_FOLDS} "
              f"folds{flag}")

    # -----------------------------------------------------------------
    # Step 6: Zebrafish aggregate
    # -----------------------------------------------------------------
    zfish_aurocs = [fr['zfish_auroc'] for fr in fold_results
                    if fr['zfish_auroc'] is not None]
    if zfish_aurocs:
        zfish_mean = float(np.mean(zfish_aurocs))
        zfish_std = float(np.std(zfish_aurocs))
        print(f"\n[6/6] GSE37332 across the 5 fold models: "
              f"{zfish_mean:.4f} +/- {zfish_std:.4f}")
    else:
        zfish_mean = zfish_std = None
        print("\n[6/6] GSE37332 external check unavailable (cache files missing).")

    # -----------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------
    _write_report(
        fold_results, pooled_boot, reported, stability,
        zfish_aurocs, zfish_mean, zfish_std, X, kept_metadata, groups,
    )

    print(f"\nReport saved: {REPORT_PATH}")
    print("\n" + "=" * 70)
    print("NESTED CV CHECK COMPLETE")
    print("=" * 70)
    print(f"  Pooled nested-CV ROC-AUC: {pooled_boot['auroc'][0]:.4f} "
          f"[{pooled_boot['auroc'][1]:.4f}, {pooled_boot['auroc'][2]:.4f}]")
    print(f"  Reported (single partition): {reported['auroc'][0]:.4f} "
          f"[{reported['auroc'][1]:.4f}, {reported['auroc'][2]:.4f}]")
    print("  core/models/consensus_rf_model.pkl was NOT "
          "modified by this script.")


def _write_report(fold_results, pooled_boot, reported, stability,
                   zfish_aurocs, zfish_mean, zfish_std, X, kept_metadata, groups):
    W = 80
    run_date = datetime.now().strftime('%Y-%m-%d %H:%M')

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * W + "\n")
        f.write("ChronoScope CRS-AI -- NESTED CROSS-VALIDATION CHECK\n")
        f.write("Feature-selection / holdout leakage (JBR Reviewer 2, Major "
                "Point 3)\n")
        f.write("=" * W + "\n\n")
        f.write(f"  Run date: {run_date}\n")
        f.write(f"  This is a supplementary robustness analysis. It does NOT "
                f"retrain or\n")
        f.write(f"  replace the production model -- consensus_rf_model.pkl "
                f"and\n")
        f.write(f"  feature_names.json are unmodified (still v6, 11 "
                f"features).\n\n")

        # --- 1. Motivation ---
        f.write("1. MOTIVATION\n")
        f.write("-" * W + "\n\n")
        f.write("  CRS-AI's 11 input features were originally selected by\n")
        f.write("  permutation importance computed on the same GroupShuffleSplit\n")
        f.write("  holdout later used to report the model's test performance\n")
        f.write("  (0.9474 ROC-AUC). The holdout is therefore not fully\n")
        f.write("  independent of the feature-selection decision -- a form of\n")
        f.write("  data leakage flagged by JBR Reviewer 2 (Major Point 3).\n\n")
        f.write("  This script re-runs feature selection under proper nested\n")
        f.write("  cross-validation: in each of 5 outer folds, permutation\n")
        f.write("  importance is computed on an inner validation split carved\n")
        f.write("  out of that fold's training data only, and the outer-test\n")
        f.write("  split never participates in the selection decision.\n\n")

        # --- 2. Design ---
        f.write("2. DESIGN\n")
        f.write("-" * W + "\n\n")
        f.write(f"  Candidate feature pool ({len(CANDIDATE_FEATURE_NAMES)} "
                f"features): the 11 production\n")
        f.write(f"  features plus the 7 candidates dropped in the v1->v2 cut\n")
        f.write(f"  (still computed by extract_features(), just not fed to\n")
        f.write(f"  the model): {', '.join(DROPPED_V1_FEATURES)}.\n\n")
        f.write(f"  Outer loop:   {N_OUTER_FOLDS}-fold StratifiedGroupKFold, "
                f"grouped by gene\n")
        f.write(f"                (random_state={OUTER_SEED}), same grouping "
                f"convention as\n")
        f.write(f"                train_consensus_model.py.\n")
        f.write(f"  Inner step:   80/20 GroupShuffleSplit "
                f"(random_state={INNER_SEED}) within\n")
        f.write(f"                each outer-train split. Permutation "
                f"importance\n")
        f.write(f"                (n_repeats={PERM_N_REPEATS}, scoring=ROC-AUC) "
                f"computed on the\n")
        f.write(f"                inner validation split only.\n")
        f.write(f"  Selection rule (Supplementary Table S1 criterion): retain "
                f"a\n")
        f.write(f"                candidate feature iff its mean permutation "
                f"drop in\n")
        f.write(f"                ROC-AUC exceeds 1 SD of its own shuffle "
                f"noise.\n")
        f.write(f"  Final fit:    Pipeline identical to "
                f"train_consensus_model.py\n")
        f.write(f"                (SimpleImputer(constant=-999) -> "
                f"CalibratedClassifierCV\n")
        f.write(f"                (RandomForest, isotonic, 5-fold internal "
                f"StratifiedGroupKFold\n")
        f.write(f"                calibration)), trained on the FULL "
                f"outer-train split using\n")
        f.write(f"                only the features selected by the inner "
                f"step, then\n")
        f.write(f"                evaluated on the outer-test split.\n")
        f.write(f"  Aggregate:    Out-of-fold predictions from all "
                f"{N_OUTER_FOLDS} outer folds are\n")
        f.write(f"                pooled (each instance gets exactly one "
                f"out-of-fold\n")
        f.write(f"                prediction) into a single ROC-AUC / F1 / "
                f"Brier / Accuracy\n")
        f.write(f"                estimate with {BOOT_N_ITER}-resample "
                f"percentile bootstrap CI,\n")
        f.write(f"                directly comparable to the single-partition "
                f"holdout\n")
        f.write(f"                reported in training_report.txt.\n")
        f.write(f"  External check: each outer fold's fitted model is also "
                f"evaluated\n")
        f.write(f"                on GSE37332 (zebrafish), which never "
                f"participated in\n")
        f.write(f"                training or feature engineering at any step.\n\n")

        f.write(f"  Training corpus: {X.shape[0]} instances, "
                f"{len(np.unique(groups))} gene/synthetic groups\n")
        f.write(f"  (identical generators and seeds as "
                f"train_consensus_model.py).\n\n")

        # --- 3. Per-fold results ---
        f.write("3. PER-FOLD RESULTS\n")
        f.write("-" * W + "\n\n")
        for fr in fold_results:
            f.write(f"  Fold {fr['fold']}/{N_OUTER_FOLDS}\n")
            f.write(f"    Outer-train: {fr['n_train_ext']}   "
                    f"Outer-test: {fr['n_test_ext']}\n")
            f.write(f"    Inner-train: {fr['n_inner_train']}   "
                    f"Inner-val: {fr['n_inner_val']}\n")
            f.write(f"    Selected features ({len(fr['selected_features'])}): "
                    f"{', '.join(sorted(fr['selected_features']))}\n")
            b = fr['boot']
            f.write(f"    Outer-test  Accuracy: {b['accuracy'][0]:.4f}  "
                    f"[{b['accuracy'][1]:.4f}, {b['accuracy'][2]:.4f}]\n")
            f.write(f"                F1:       {b['f1'][0]:.4f}  "
                    f"[{b['f1'][1]:.4f}, {b['f1'][2]:.4f}]\n")
            f.write(f"                ROC-AUC:  {b['auroc'][0]:.4f}  "
                    f"[{b['auroc'][1]:.4f}, {b['auroc'][2]:.4f}]\n")
            f.write(f"                Brier:    {b['brier'][0]:.4f}  "
                    f"[{b['brier'][1]:.4f}, {b['brier'][2]:.4f}]\n")
            if fr['zfish_auroc'] is not None:
                f.write(f"    GSE37332 (zebrafish) AUROC with this fold's "
                        f"model: {fr['zfish_auroc']:.4f}\n")
            f.write("\n")

        # --- 4. Pooled nested-CV estimate vs. reported single partition ---
        f.write("4. POOLED NESTED-CV ESTIMATE vs. REPORTED SINGLE PARTITION\n")
        f.write("-" * W + "\n\n")
        f.write("  Pooled out-of-fold predictions (every training instance\n")
        f.write("  contributes exactly one out-of-fold prediction, from the\n")
        f.write("  fold where it was in the outer-test split):\n\n")
        f.write(f"  {'Metric':<12s} {'Nested CV (pooled)':<28s} "
                f"{'Reported (training_report.txt)':<32s}\n")
        f.write(f"  {'-'*11}  {'-'*26}  {'-'*30}\n")
        for label, key in [('Accuracy', 'accuracy'), ('F1', 'f1'),
                            ('ROC-AUC', 'auroc'), ('Brier', 'brier')]:
            nested = pooled_boot[key]
            rep = reported.get(key)
            nested_str = f"{nested[0]:.4f} [{nested[1]:.4f}, {nested[2]:.4f}]"
            rep_str = (f"{rep[0]:.4f} [{rep[1]:.4f}, {rep[2]:.4f}]"
                        if rep else "n/a")
            f.write(f"  {label:<12s} {nested_str:<28s} {rep_str:<32s}\n")
        f.write("\n")

        auroc_nested = pooled_boot['auroc'][0]
        auroc_reported = reported['auroc'][0] if reported.get('auroc') else None
        if auroc_reported is not None:
            delta = auroc_nested - auroc_reported
            ci_overlap = (pooled_boot['auroc'][1] <= reported['auroc'][2] and
                          reported['auroc'][1] <= pooled_boot['auroc'][2])
            f.write(f"  Delta (nested - reported): {delta:+.4f} ROC-AUC "
                    f"points.\n")
            f.write(f"  95% CIs {'overlap' if ci_overlap else 'do NOT overlap'} "
                    f"between the nested-CV pooled\n")
            f.write(f"  estimate and the reported single-partition holdout.\n\n")

        # --- 5. Feature-selection stability ---
        f.write("5. FEATURE-SELECTION STABILITY ACROSS OUTER FOLDS\n")
        f.write("-" * W + "\n\n")
        f.write(f"  {'Feature':<24s} {'Folds selected':>14s}  "
                f"{'Production feature?'}\n")
        f.write(f"  {'-'*23}  {'-'*14}  {'-'*19}\n")
        for name in CANDIDATE_FEATURE_NAMES:
            is_prod = 'Yes' if name in FEATURE_NAMES else 'No'
            f.write(f"  {name:<24s} {stability[name]:>10d}/{N_OUTER_FOLDS}     "
                    f"{is_prod}\n")
        f.write("\n")
        always_selected = [n for n in CANDIDATE_FEATURE_NAMES
                           if stability[n] == N_OUTER_FOLDS]
        never_selected = [n for n in CANDIDATE_FEATURE_NAMES
                          if stability[n] == 0]
        f.write(f"  Selected in all {N_OUTER_FOLDS} folds "
                f"({len(always_selected)}): "
                f"{', '.join(sorted(always_selected)) or '(none)'}\n")
        f.write(f"  Never selected ({len(never_selected)}): "
                f"{', '.join(sorted(never_selected)) or '(none)'}\n")
        prod_set = set(FEATURE_NAMES)
        always_set = set(always_selected)
        f.write(f"\n  Overlap between the {N_OUTER_FOLDS}/{N_OUTER_FOLDS}-fold "
                f"stable set and the current 11\n")
        f.write(f"  production features: {len(prod_set & always_set)}/"
                f"{len(prod_set)} production features are in the "
                f"always-selected set;\n")
        f.write(f"  {len(always_set - prod_set)} always-selected feature(s) "
                f"are not currently in production.\n\n")

        # --- 6. External check: GSE37332 (zebrafish) ---
        f.write("6. EXTERNAL CHECK: GSE37332 (ZEBRAFISH)\n")
        f.write("-" * W + "\n\n")
        if zfish_aurocs:
            f.write(f"  AUROC across the {len(zfish_aurocs)} fold-specific "
                    f"models (never seen at any\n")
            f.write(f"  training or feature-engineering step): "
                    f"{zfish_mean:.4f} +/- {zfish_std:.4f}\n")
            f.write(f"  Per-fold: "
                    f"{', '.join(f'{a:.4f}' for a in zfish_aurocs)}\n\n")
            f.write(f"  For reference, the single production model (v6, "
                    f"11 fixed features)\n")
            f.write(f"  reported AUROC=0.9305 [0.9069, 0.9528] on this same "
                    f"dataset\n")
            f.write(f"  (validate_external_holdout_gse37332.txt).\n\n")
        else:
            f.write("  Not available (GSE37332 cache files missing).\n\n")

        # --- 7. Interpretation ---
        f.write("7. INTERPRETATION\n")
        f.write("-" * W + "\n\n")
        if auroc_reported is not None:
            if abs(delta) < 0.01:
                f.write("  The pooled nested-CV ROC-AUC is essentially "
                        "unchanged from the\n")
                f.write("  single-partition estimate reported in the paper "
                        "(delta < 0.01).\n")
                f.write("  Removing the feature-selection/holdout leakage "
                        "flagged by the\n")
                f.write("  reviewer does not materially change CRS-AI's "
                        "reported performance.\n\n")
            elif delta < 0:
                f.write("  The pooled nested-CV ROC-AUC is lower than the "
                        "single-partition\n")
                f.write("  estimate reported in the paper, consistent with "
                        "the reviewer's\n")
                f.write("  concern that the original holdout number was "
                        "somewhat optimistic\n")
                f.write("  due to shared feature-selection/evaluation data.\n\n")
            else:
                f.write("  The pooled nested-CV ROC-AUC is not lower than "
                        "the single-partition\n")
                f.write("  estimate reported in the paper.\n\n")
        f.write("  Feature-selection stability across outer folds indicates "
                "whether the\n")
        f.write("  11 production features represent a robust choice or an "
                "artifact of\n")
        f.write("  the single original train/test partition (Section 5 "
                "above).\n\n")

        f.write("=" * W + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * W + "\n")


if __name__ == '__main__':
    main()
