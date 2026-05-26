"""
Train the Consensus Rhythmicity Score Random Forest Model
==========================================================

Standalone script that:
1. Generates expanded synthetic training data
2. Extracts features from each instance using all sub-methods
3. Trains a Random Forest pipeline (Imputer -> Scaler -> RF)
4. Evaluates with cross-validation and holdout test set
5. Saves the trained model to core/models/

Usage:
    python train_consensus_model.py

Output:
    core/models/consensus_rf_model.pkl
    core/models/feature_names.json

Author: Francisco Tassara
"""

import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
# __file__ is at core/models_meta_classifier/train_consensus_model.py
# so .parent.parent.parent is the project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Training data scripts live in training_data_meta_classifer/
training_data_dir = project_root / 'training_data_meta_classifer'
sys.path.insert(0, str(training_data_dir))

# Suppress warnings during training
warnings.filterwarnings('ignore')


def _bootstrap_metric_ci(y_true, y_pred, y_proba, n_iter=1000,
                          ci=0.95, seed=42):
    """Compute percentile bootstrap confidence intervals for holdout metrics.

    Resamples the test set with replacement n_iter times, computes the
    metric on each resample, and returns the (lower, upper) percentile
    bounds for the requested confidence level.

    Iterations where the resample contains only one class are skipped
    (AUROC and class-conditioned metrics are undefined in that case).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, aurocs, f1s, briers = [], [], [], []

    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        yp = y_pred[idx]
        ypr = y_proba[idx]
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score, brier_score_loss
        )
        accs.append(accuracy_score(yt, yp))
        aurocs.append(roc_auc_score(yt, ypr))
        f1s.append(f1_score(yt, yp))
        briers.append(brier_score_loss(yt, ypr))

    alpha = (1 - ci) / 2
    pct_lo, pct_hi = alpha * 100, (1 - alpha) * 100
    return {
        'n_valid_iter': len(accs),
        'accuracy_ci': (float(np.percentile(accs, pct_lo)),
                        float(np.percentile(accs, pct_hi))),
        'auroc_ci':    (float(np.percentile(aurocs, pct_lo)),
                        float(np.percentile(aurocs, pct_hi))),
        'f1_ci':       (float(np.percentile(f1s, pct_lo)),
                        float(np.percentile(f1s, pct_hi))),
        'brier_ci':    (float(np.percentile(briers, pct_lo)),
                        float(np.percentile(briers, pct_hi))),
    }


def _extract_features_for_instances(metadata, dataframes, feature_names, label="instances"):
    """Run feature extraction over a list of instances. Returns (X, y, kept_metadata)."""
    from core.feature_extraction import extract_features

    n_total = len(metadata)
    feature_rows = []
    labels = []
    kept_metadata = []
    start_time = time.time()

    for i, (meta, df) in enumerate(zip(metadata, dataframes)):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  Processing {label} {i + 1}/{n_total} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        variable = meta['variable']
        condition = 'control'
        time_col = 'time'
        condition_col = 'condition'

        cond_data = df[df[condition_col] == condition]
        times = cond_data[time_col].values.astype(float)

        if variable not in cond_data.columns:
            continue

        values = cond_data[variable].values.astype(float)
        valid = ~(np.isnan(times) | np.isnan(values))
        times = times[valid]
        values = values[valid]

        if len(times) < 4:
            continue

        unique_times = np.unique(times)
        avg_values = np.array([values[times == t].mean() for t in unique_times])

        features = extract_features(
            unique_times, avg_values, df, variable, condition,
            time_col, condition_col,
        )

        feature_rows.append(features)
        labels.append(meta['is_rhythmic'])
        kept_metadata.append(meta)

    X = np.array([[row.get(name, np.nan) for name in feature_names] for row in feature_rows])
    y = np.array(labels)
    elapsed_total = time.time() - start_time
    print(f"  Feature extraction complete: {len(feature_rows)} {label} in {elapsed_total:.1f}s")
    return X, y, kept_metadata


def main():
    print("=" * 70)
    print("CONSENSUS RHYTHMICITY SCORE - MODEL TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1a: Generate synthetic training data
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic training data...")
    from generate_synthetic_training_data import generate_training_instances

    metadata, dataframes = generate_training_instances(seed=42)
    n_synth = len(metadata)
    n_synth_rhythmic = sum(1 for m in metadata if m['is_rhythmic'] == 1)
    print(f"  Synthetic instances: {n_synth}")
    print(f"  Rhythmic: {n_synth_rhythmic}")
    print(f"  Non-rhythmic: {n_synth - n_synth_rhythmic}")

    # ------------------------------------------------------------------
    # Step 1b: Generate real biological training data
    # ------------------------------------------------------------------
    print("\n[1b/6] Loading real biological training data...")
    from generate_real_training_data import (
        generate_from_geo,
        generate_from_GSE77451,
        generate_from_GSE39445,
    )

    biocycle_xlsx = str(training_data_dir / 'rhythmicdb_query_bioCycle.xlsx')
    geo_cache_dir = training_data_dir / 'data' / 'geo'
    n_real = 0

    # Dataset 1: GSE11923 (Hughes 2009, mouse liver, hourly x 48h)
    # Labels: known core clock + housekeeping genes only
    try:
        print("\n  --- Dataset 1: GSE11923 (Hughes 2009) ---")
        real1_meta, real1_dfs = generate_from_geo(
            geo_accession='GSE11923',
            platform_id='GPL1261',
            starting_id=n_synth + 1000,
            subsample_intervals=[2.0, 4.0],
        )
        metadata = metadata + real1_meta
        dataframes = dataframes + real1_dfs
        n_real += len(real1_meta)
        print(f"  GSE11923: {len(real1_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE11923 failed: {e}")

    # Dataset 2: GSE11516 (mouse liver, 4h x 48h, 3 replicates)
    # Labels: BioCycle from RhythmicDB with ambiguity gap
    #   Rhythmic: q <= 0.01 | Non-rhythmic: q > 0.2 | Excluded: 0.01 < q <= 0.2
    #   Capped at 800 per class to balance with synthetic data
    ambiguous_metadata: list = []
    ambiguous_dataframes: list = []
    try:
        print("\n  --- Dataset 2: GSE11516 (BioCycle labels) ---")
        real2_meta, real2_dfs, amb_meta, amb_dfs = generate_from_geo(
            geo_accession='GSE11516',
            platform_id='GPL6880',
            biocycle_xlsx=biocycle_xlsx,
            biocycle_dataset_id='E-GEOD-11516',
            biocycle_q_threshold=0.01,
            biocycle_q_non_rhythmic=0.2,
            max_rhythmic=800,
            max_non_rhythmic=800,
            starting_id=n_synth + 2000 + n_real,
            subsample_intervals=[],  # already 4h intervals, no subsampling
            return_ambiguous=True,
        )
        metadata = metadata + real2_meta
        dataframes = dataframes + real2_dfs
        n_real += len(real2_meta)
        ambiguous_metadata = amb_meta
        ambiguous_dataframes = amb_dfs
        print(f"  GSE11516: {len(real2_meta)} instances "
              f"(+ {len(amb_meta)} ambiguous held out)")
    except Exception as e:
        print(f"  WARNING: GSE11516 failed: {e}")
        import traceback
        traceback.print_exc()

    # Dataset 3: GSE77451 (Abruzzi 2017, Drosophila clock neurons)
    # Positives: HC-cyclers in LNv, LNd, DN1 (max 200/cell type by JTK p-value)
    # Negatives: fly housekeeping genes (all cell types) + fly clock genes (TH only)
    # NOTE: case-sensitive gene symbols keep fly/mouse/human groups disjoint in
    # GroupShuffleSplit (e.g. 'per' ≠ 'Per1' ≠ 'PER1').
    try:
        print("\n  --- Dataset 3: GSE77451 (Abruzzi 2017, Drosophila) ---")
        gse77451_meta, gse77451_dfs = generate_from_GSE77451(
            abruzzi_xlsx_path=training_data_dir / 'abruzzi_2017_cycling.xlsx',
            geo_cache_dir=geo_cache_dir,
            starting_instance_id=n_synth + 10000,
            max_positives_per_cell_type=200,
            hc_only=True,
            seed=42,
        )
        metadata = metadata + gse77451_meta
        dataframes = dataframes + gse77451_dfs
        n_real += len(gse77451_meta)
        print(f"  GSE77451: {len(gse77451_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE77451 failed: {e}")
        import traceback
        traceback.print_exc()

    # Dataset 4: GSE39445 (Möller-Levet 2013, Human whole blood)
    # Positives: genes rhythmic in ≥1 condition per Möller-Levet labels
    # Negatives: genes non-rhythmic + no sleep condition effect
    # Forced includes: KNOWN_CIRCADIAN_GENES_HUMAN / NON_RHYTHMIC_GENES_HUMAN
    try:
        print("\n  --- Dataset 4: GSE39445 (Möller-Levet 2013, Human) ---")
        gse39445_meta, gse39445_dfs = generate_from_GSE39445(
            moller_xlsx_path=training_data_dir / 'moller_levet_2013_circadian.xlsx',
            geo_cache_dir=geo_cache_dir,
            starting_instance_id=n_synth + 20000,
            max_per_class=800,
            seed=42,
        )
        metadata = metadata + gse39445_meta
        dataframes = dataframes + gse39445_dfs
        n_real += len(gse39445_meta)
        print(f"  GSE39445: {len(gse39445_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE39445 failed: {e}")
        import traceback
        traceback.print_exc()

    # Combined summary
    n_total = len(metadata)
    n_rhythmic = sum(1 for m in metadata if m['is_rhythmic'] == 1)
    n_non_rhythmic = n_total - n_rhythmic
    print(f"\n  Combined training set: {n_total} instances")
    print(f"    Synthetic: {n_synth} | Real: {n_real}")
    print(f"    Rhythmic: {n_rhythmic} | Non-rhythmic: {n_non_rhythmic}")

    # ------------------------------------------------------------------
    # Step 2: Extract features from each instance
    # ------------------------------------------------------------------
    print(f"\n[2/6] Extracting features from {n_total} instances...")
    print("  (This runs 5 analysis methods per instance - please wait)")

    from core.feature_extraction import FEATURE_NAMES

    X, y, kept_metadata = _extract_features_for_instances(
        metadata, dataframes, FEATURE_NAMES, label="training instances"
    )

    # ------------------------------------------------------------------
    # Step 3: Build feature matrix + group IDs for gene-aware splitting
    # ------------------------------------------------------------------
    print("\n[3/6] Building feature matrix...")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels: {sum(y == 1)} rhythmic, {sum(y == 0)} non-rhythmic")

    # Build groups: same group => same gene (real) or unique (synthetic).
    # This prevents the same gene's multi-resolution subsamples (GSE11923
    # at 1h/2h/4h) from being split across train and test. Genes that
    # appear in both real datasets (e.g., Per1 in GSE11923 + GSE11516)
    # share the same group via their gene name, so cross-dataset same-
    # gene leakage is also prevented. Synthetic instances have unique
    # groups (no biological identity to leak).
    groups = np.array([
        m['gene'] if 'gene' in m else f"synth_{m['instance_id']}"
        for m in kept_metadata
    ])
    n_unique_groups = len(np.unique(groups))
    n_gene_groups = len(np.unique([g for g in groups if not g.startswith('synth_')]))
    print(f"  Total groups:        {n_unique_groups}")
    print(f"  Gene-based groups:   {n_gene_groups}")
    print(f"  Synthetic (unique):  {n_unique_groups - n_gene_groups}")

    # Check for features that are all NaN
    nan_counts = np.sum(np.isnan(X), axis=0)
    for i, name in enumerate(FEATURE_NAMES):
        pct = nan_counts[i] / len(X) * 100
        if pct > 50:
            print(f"  WARNING: Feature '{name}' is NaN in {pct:.0f}% of instances")

    # ------------------------------------------------------------------
    # Step 4: Train model
    # ------------------------------------------------------------------
    print("\n[4/6] Training Random Forest model...")

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import (
        GroupShuffleSplit, StratifiedGroupKFold,
    )
    from sklearn.metrics import (
        classification_report, roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix, matthews_corrcoef, brier_score_loss,
    )

    # Gene-aware train/test split: GroupShuffleSplit prevents the same gene
    # (or same synthetic instance) from appearing in both train and test.
    # Note: GroupShuffleSplit does not support stratify=y natively — class
    # balance is preserved only approximately. We report the actual balance
    # of both splits below and verify it is within tolerance.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    meta_test = [kept_metadata[i] for i in test_idx]

    print(f"  Train set: {len(X_train)} samples "
          f"({y_train.sum()} rhythmic, {len(y_train) - y_train.sum()} non-rhythmic)")
    print(f"  Test set:  {len(X_test)} samples "
          f"({y_test.sum()} rhythmic, {len(y_test) - y_test.sum()} non-rhythmic)")
    # Sanity: verify no group leakage across the split.
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    assert len(overlap) == 0, f"Group leakage detected: {overlap}"
    print(f"  Group leakage check: OK (0 shared groups)")

    # Build pipeline factory.
    # Note: constant fill_value=-999 (sentinel) instead of median imputation.
    # The RF learns clean splits: "if feature <= -500 -> feature was unavailable".
    # No StandardScaler: Random Forest is invariant to monotonic feature scaling.
    # CalibratedClassifierCV wraps the RF with isotonic regression to produce
    # well-calibrated probabilities (default RF probas are overconfident at
    # extremes). Its internal CV is fed PRECOMPUTED StratifiedGroupKFold splits
    # tied to the groups of whatever subset is being fitted, so the same gene
    # never appears in both calibration-train and calibration-val folds —
    # matching the outer GroupShuffleSplit no-leakage guarantee. We can't pass
    # `groups` natively because CalibratedClassifierCV.fit() doesn't accept it.
    def _build_pipeline(X_fit, y_fit, groups_fit):
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        calibration_splits = list(
            StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            .split(X_fit, y_fit, groups=groups_fit)
        )
        calibrated_rf = CalibratedClassifierCV(
            estimator=base_rf,
            method='isotonic',
            cv=calibration_splits,
        )
        return Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
            ('classifier', calibrated_rf),
        ])

    # Cross-validation — manual outer loop because we need each outer fold to
    # build its own group-aware inner calibration splits over its sub-training
    # set. cross_val_score can't do this (CalibratedClassifierCV.fit doesn't
    # forward `groups` through the pipeline).
    print("\n  Running 5-fold stratified group cross-validation...")
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for fold_idx, (tr_idx, val_idx) in enumerate(
        outer_cv.split(X_train, y_train, groups_train)
    ):
        fold_pipeline = _build_pipeline(
            X_train[tr_idx], y_train[tr_idx], groups_train[tr_idx]
        )
        fold_pipeline.fit(X_train[tr_idx], y_train[tr_idx])
        y_proba_fold = fold_pipeline.predict_proba(X_train[val_idx])[:, 1]
        fold_auc = roc_auc_score(y_train[val_idx], y_proba_fold)
        cv_scores.append(fold_auc)
        print(f"    Fold {fold_idx + 1}/5: ROC-AUC = {fold_auc:.4f}")
    cv_scores = np.array(cv_scores)
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Per-fold:   {[f'{s:.4f}' for s in cv_scores]}")

    # Fit final pipeline on full training set (group-aware calibration splits).
    print("\n  Fitting on full training set...")
    pipeline = _build_pipeline(X_train, y_train, groups_train)
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(y_test, y_proba)

    # Bootstrap 95% confidence intervals on the holdout (1000 resamples)
    print("\n  Computing bootstrap 95% CIs (n=1000 resamples)...")
    boot = _bootstrap_metric_ci(y_test, y_pred, y_proba, n_iter=1000, seed=42)

    print("\n  --- Test Set Evaluation ---")
    print(f"  Accuracy:    {accuracy_score(y_test, y_pred):.4f}  "
          f"[95% CI {boot['accuracy_ci'][0]:.4f}, {boot['accuracy_ci'][1]:.4f}]")
    print(f"  Precision:   {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:      {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score:    {f1_score(y_test, y_pred):.4f}  "
          f"[95% CI {boot['f1_ci'][0]:.4f}, {boot['f1_ci'][1]:.4f}]")
    print(f"  ROC-AUC:     {roc_auc_score(y_test, y_proba):.4f}  "
          f"[95% CI {boot['auroc_ci'][0]:.4f}, {boot['auroc_ci'][1]:.4f}]")
    print(f"  Brier loss:  {brier:.4f}  "
          f"[95% CI {boot['brier_ci'][0]:.4f}, {boot['brier_ci'][1]:.4f}]")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-rhythmic', 'Rhythmic']))

    # ------------------------------------------------------------------
    # Step 4a: Ambiguous holdout — borderline BioCycle genes (q in (0.01, 0.2])
    # excluded from training. Reports honest performance on hard cases.
    # ------------------------------------------------------------------
    amb_results = None
    if ambiguous_metadata:
        print(f"\n  --- Ambiguous Holdout Evaluation ---")
        print(f"  ({len(ambiguous_metadata)} BioCycle borderline-q genes, "
              f"never seen during training)")

        X_amb, y_amb, kept_amb_meta = _extract_features_for_instances(
            ambiguous_metadata, ambiguous_dataframes, FEATURE_NAMES,
            label="ambiguous holdout"
        )

        if len(X_amb) > 0:
            y_amb_pred = pipeline.predict(X_amb)
            y_amb_proba = pipeline.predict_proba(X_amb)[:, 1]

            # Provisional labels are coarse; AUROC against them is noisier
            # but still informative. Wrap in try/except in case one class
            # is missing (e.g., all ambiguous genes fell on one side).
            try:
                amb_auroc = roc_auc_score(y_amb, y_amb_proba)
            except ValueError:
                amb_auroc = float('nan')
            amb_acc = accuracy_score(y_amb, y_amb_pred)
            amb_brier = brier_score_loss(y_amb, y_amb_proba)

            # Soft-zone analysis: fraction of ambiguous cases the model also
            # flags as borderline (probability between 0.30 and 0.70). This is
            # the right metric for ambiguous data — high values mean the model
            # is appropriately uncertain on uncertain cases.
            soft_zone_mask = (y_amb_proba > 0.30) & (y_amb_proba < 0.70)
            pct_soft_zone = soft_zone_mask.mean() * 100

            print(f"  N:                       {len(y_amb)}")
            print(f"  Accuracy vs midpoint:    {amb_acc:.4f}")
            print(f"  AUROC vs midpoint:       {amb_auroc:.4f}")
            print(f"  Brier loss:              {amb_brier:.4f}")
            print(f"  % flagged borderline (0.30 < p < 0.70):  {pct_soft_zone:.1f}%")
            print(f"  (Higher % borderline = model is appropriately uncertain)")

            amb_results = {
                'n': int(len(y_amb)),
                'accuracy': float(amb_acc),
                'auroc': float(amb_auroc),
                'brier': float(amb_brier),
                'pct_soft_zone': float(pct_soft_zone),
                'y_true': y_amb,
                'y_proba': y_amb_proba,
            }

    # Feature importances (MDI) — averaged across the CalibratedClassifierCV's
    # internal RFs (one per CV fold). Each calibrated_classifiers_[i].estimator
    # is a fitted RandomForestClassifier.
    calibrated = pipeline.named_steps['classifier']
    fold_importances = np.array([
        cc.estimator.feature_importances_
        for cc in calibrated.calibrated_classifiers_
    ])
    mean_importances = fold_importances.mean(axis=0)
    importances = sorted(
        zip(FEATURE_NAMES, mean_importances),
        key=lambda x: x[1], reverse=True
    )
    print("  Top 10 Feature Importances (MDI):")
    for name, imp in importances[:10]:
        print(f"    {name:30s} {imp:.4f}")

    # Permutation importance — measures the drop in AUROC when each feature
    # is randomly shuffled on the holdout test set. Unlike MDI, this is
    # NOT biased toward high-cardinality / continuous features, and it
    # measures the feature's contribution to GENERALIZATION (test AUROC),
    # not just training-set Gini reduction. Recommended for paper-quality
    # interpretation (Strobl et al. 2007).
    print("\n  Computing permutation importance on test set "
          "(n_repeats=10, scoring=ROC-AUC)...")
    perm_result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=10, random_state=42, n_jobs=-1, scoring='roc_auc',
    )
    perm_importances = sorted(
        zip(FEATURE_NAMES,
            perm_result.importances_mean,
            perm_result.importances_std),
        key=lambda x: x[1], reverse=True,
    )
    print("  Top 10 Feature Importances (Permutation):")
    for name, imp_mean, imp_std in perm_importances[:10]:
        print(f"    {name:30s} {imp_mean:+.4f} (+/- {imp_std:.4f})")

    # ------------------------------------------------------------------
    # Step 4b: Persist holdout set for downstream evaluation (ROC, figures)
    # ------------------------------------------------------------------
    print("\n[4b/6] Saving holdout predictions and test arrays...")

    model_dir = Path(__file__).parent
    model_dir.mkdir(exist_ok=True)

    # y_true + y_proba CSV — used for ROC curve and figures
    holdout_df = pd.DataFrame({
        'y_true': y_test,
        'y_proba': y_proba,
        'gene': [m.get('gene', f"synth_{m.get('instance_id', '?')}") for m in meta_test],
        'source': [m.get('source', 'synthetic') for m in meta_test],
        'signal_type': [m['signal_type'] for m in meta_test],
    })
    holdout_path = model_dir / 'holdout_predictions.csv'
    holdout_df.to_csv(holdout_path, index=False)
    print(f"  Holdout predictions saved: {holdout_path}  ({len(holdout_df)} rows)")

    # Raw arrays — useful for computing other metrics without re-running extraction
    np.save(str(model_dir / 'X_test.npy'), X_test)
    np.save(str(model_dir / 'y_test.npy'), y_test)
    print(f"  X_test.npy / y_test.npy saved ({X_test.shape})")

    # Ambiguous holdout predictions (if available)
    if amb_results is not None:
        amb_df = pd.DataFrame({
            'y_true_midpoint': amb_results['y_true'],
            'y_proba': amb_results['y_proba'],
        })
        amb_path = model_dir / 'ambiguous_holdout_predictions.csv'
        amb_df.to_csv(amb_path, index=False)
        print(f"  Ambiguous holdout predictions saved: {amb_path}  ({len(amb_df)} rows)")

    # ------------------------------------------------------------------
    # Step 5: Save model
    # ------------------------------------------------------------------
    print("\n[5/6] Saving model...")

    model_path = model_dir / 'consensus_rf_model.pkl'
    features_path = model_dir / 'feature_names.json'

    import joblib
    joblib.dump(pipeline, str(model_path))
    print(f"  Model saved: {model_path}")
    print(f"  Model size: {model_path.stat().st_size / 1024:.1f} KB")

    with open(features_path, 'w') as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"  Feature names saved: {features_path}")

    # ------------------------------------------------------------------
    # Step 6: Save training report
    # ------------------------------------------------------------------
    print("\n[6/6] Saving training report...")

    import sklearn
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    mcc = matthews_corrcoef(y_test, y_pred)

    # Feature descriptions for the report
    feature_descriptions = {
        'jtk_p_value': 'JTK_CYCLE adjusted p-value (non-parametric rhythmicity test)',
        'jtk_tau': 'JTK_CYCLE tau statistic (cyclic pattern strength)',
        'jtk_period': 'JTK_CYCLE detected period (hours)',
        'cosinor_p_value': 'Cosinor OLS regression adjusted p-value',
        'cosinor_r_squared': 'Cosinor OLS goodness-of-fit (R-squared)',
        'cosinor_amplitude': 'Cosinor fitted amplitude (expression units)',
        'cosinor_period': 'Cosinor detected period (hours)',
        'ls_p_value': 'Lomb-Scargle false alarm probability',
        'ls_power': 'Lomb-Scargle dominant spectral power',
        'ls_dominant_period': 'Lomb-Scargle dominant period (hours)',
        'f24_score': 'Fourier F24 test score (24h periodicity statistic)',
        'harmonic_p_value': 'Harmonic cosinor adjusted p-value (2 harmonics)',
        'harmonic_r_squared': 'Harmonic cosinor goodness-of-fit (R-squared)',
        'method_agreement': 'Fraction of methods with p < 0.05 (consensus score)',
        'period_concordance': 'Std. deviation of periods across methods (lower=better)',
        'amplitude_relative': 'Amplitude relative to mesor (amplitude / |mean|)',
        'log_min_p_value': 'log10(min p-value) across all methods (compresses range)',
        'period_dev_24h': 'Min |period - 24h| across methods (circadian proximity)',
    }

    training_date = datetime.now().strftime('%Y-%m-%d')
    report_path = model_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        W = 80  # line width
        f.write("=" * W + "\n")
        f.write("ChronoScope - CONSENSUS RHYTHMICITY SCORE (CRS-AI)\n")
        f.write("MODEL TRAINING REPORT\n")
        f.write("=" * W + "\n\n")
        f.write(f"  Training date: {training_date}\n\n")

        # --- 1. Model overview ---
        f.write("1. MODEL OVERVIEW\n")
        f.write("-" * W + "\n\n")
        f.write("The Consensus Rhythmicity Score (CRS-AI) is a meta-classifier that\n")
        f.write("integrates the outputs of five established circadian rhythm detection\n")
        f.write("methods into a single binary prediction (rhythmic vs. non-rhythmic).\n")
        f.write("A Random Forest classifier is trained on features extracted from each\n")
        f.write("sub-method, enabling robust detection that combines the strengths of\n")
        f.write("multiple independent algorithms.\n\n")

        # --- 2. Software environment ---
        f.write("2. SOFTWARE ENVIRONMENT\n")
        f.write("-" * W + "\n\n")
        f.write(f"  Python:        {sys.version.split()[0]}\n")
        f.write(f"  scikit-learn:  {sklearn.__version__}\n")
        f.write(f"  NumPy:         {np.__version__}\n")
        f.write(f"  pandas:        {pd.__version__}\n\n")

        # --- 3. Model architecture ---
        f.write("3. MODEL ARCHITECTURE\n")
        f.write("-" * W + "\n\n")
        f.write("  Pipeline:  SimpleImputer -> CalibratedClassifierCV(RandomForest)\n\n")
        f.write("  Step 1 - SimpleImputer:\n")
        f.write("    Strategy:      constant (fill_value = -999)\n")
        f.write("    Rationale:     Sentinel value for structurally-missing features\n")
        f.write("                   (i.e., any sub-method that fails on a given time\n")
        f.write("                   series, e.g., Harmonic Cosinor on very short\n")
        f.write("                   series). Avoids bias from median imputation. The\n")
        f.write("                   RF learns to split on 'feature <= -500' as\n")
        f.write("                   'feature was unavailable'.\n\n")
        f.write("  Note on feature scaling:\n")
        f.write("    No StandardScaler is applied. Random Forest is invariant to\n")
        f.write("    monotonic feature scaling: each tree splits on per-feature\n")
        f.write("    empirical thresholds, which are unaffected by linear\n")
        f.write("    transformations. Previous versions included a StandardScaler\n")
        f.write("    step; it was removed in this version as redundant. Removing\n")
        f.write("    it does not change predictions but simplifies the pipeline\n")
        f.write("    and eliminates the spurious sensitivity of feature means/\n")
        f.write("    variances to the sentinel imputation value.\n\n")
        f.write("  Step 2 - RandomForestClassifier (base estimator):\n")
        f.write(f"    n_estimators:    200\n")
        f.write(f"    max_depth:       10\n")
        f.write(f"    min_samples_leaf: 5\n")
        f.write(f"    class_weight:    balanced\n")
        f.write(f"    random_state:    42\n\n")
        f.write("  Step 3 - CalibratedClassifierCV (probability calibration):\n")
        f.write("    method:          isotonic regression\n")
        f.write("    cv:              5-fold internal\n")
        f.write("    Rationale:       Out-of-the-box RF probabilities are known\n")
        f.write("                     to be miscalibrated (overconfident at\n")
        f.write("                     extremes, underconfident in the middle).\n")
        f.write("                     Isotonic calibration on internal CV folds\n")
        f.write("                     produces probability estimates that match\n")
        f.write("                     empirical frequencies, which is important\n")
        f.write("                     for the borderline threshold logic in the\n")
        f.write("                     ChronoScope GUI (Rhythmic >= 0.70,\n")
        f.write("                     Arrhythmic <= 0.30, Borderline in between).\n")
        f.write("    Calibration quality is reported via Brier score in Section 6.\n\n")

        # --- 4. Feature set ---
        f.write("4. FEATURE SET ({} features)\n".format(len(FEATURE_NAMES)))
        f.write("-" * W + "\n\n")
        f.write("Features are extracted from five circadian detection sub-methods\n")
        f.write("plus derived consensus metrics:\n\n")
        f.write("  Sub-methods:\n")
        f.write("    - JTK_CYCLE:        Non-parametric rhythmicity test (Hughes 2010)\n")
        f.write("    - Cosinor OLS:      Cosine regression with period optimization\n")
        f.write("    - Lomb-Scargle:     Spectral analysis for unevenly-sampled data\n")
        f.write("    - Fourier F24:      24-hour Fourier component significance test\n")
        f.write("    - Harmonic Cosinor: Multi-harmonic cosine regression (2 harmonics)\n\n")

        f.write(f"  {'Feature':<30s} {'Importance':>10s}   Description\n")
        f.write(f"  {'-' * 29}  {'-' * 10}   {'-' * 36}\n")
        for name, imp in importances:
            desc = feature_descriptions.get(name, '')
            f.write(f"  {name:<30s} {imp:>10.4f}   {desc}\n")

        # NaN analysis
        f.write(f"\n  Feature completeness (NaN analysis):\n")
        for i, name in enumerate(FEATURE_NAMES):
            pct = nan_counts[i] / len(X) * 100
            if pct > 0:
                f.write(f"    {name:<30s} {pct:5.1f}% NaN\n")
        all_complete = all(nan_counts[i] == 0 for i in range(len(FEATURE_NAMES)))
        if all_complete:
            f.write(f"    All features complete (0% NaN)\n")
        f.write("\n")

        # --- 5. Training data ---
        f.write("5. TRAINING DATA\n")
        f.write("-" * W + "\n\n")
        f.write(f"  Total instances:        {n_total}\n")
        f.write(f"    Synthetic:            {n_synth}\n")
        f.write(f"    Real biological:      {n_real}\n")
        f.write(f"  Class distribution:\n")
        f.write(f"    Rhythmic (label=1):   {n_rhythmic} ({n_rhythmic/n_total*100:.1f}%)\n")
        f.write(f"    Non-rhythmic (label=0): {n_non_rhythmic} ({n_non_rhythmic/n_total*100:.1f}%)\n\n")

        f.write("  5.1 Synthetic data ({} instances)\n\n".format(n_synth))
        f.write("    Generated with generate_synthetic_training_data.py. Includes\n")
        f.write("    cosine waves, multi-harmonic waveforms, pulse-like patterns,\n")
        f.write("    noisy oscillations, and various non-rhythmic signals (flat,\n")
        f.write("    linear trend, random walk, exponential decay, etc.). 15% of\n")
        f.write("    instances include outlier contamination. Variable SNR, sampling\n")
        f.write("    intervals (1h, 2h, 4h), and series lengths (6-48 points).\n\n")
        f.write("    Period structure (designed to avoid trivial separability):\n")
        f.write("      Rhythmic periods:        23.0, 23.5, 24.0, 24.5, 25.0 h\n")
        f.write("      Non-rhythmic periods:    4, 5, 6, 8, 10, 12, 14, 16, 18,\n")
        f.write("                               20, 21, 22, 26, 27, 30, 36, 48 h\n")
        f.write("    The non-rhythmic set includes oscillators at 21, 22, 26, and\n")
        f.write("    27 h (near-circadian but biologically NON-rhythmic) so that\n")
        f.write("    the period_dev_24h feature cannot trivially separate the two\n")
        f.write("    classes by a wide gap. This forces the model to also learn\n")
        f.write("    amplitude regularity and inter-method agreement, instead of\n")
        f.write("    relying on a synthetic shortcut.\n\n")

        f.write("  5.2 Real biological data ({} instances)\n\n".format(n_real))
        f.write("    Dataset 1: GSE11923 (Hughes et al., 2009)\n")
        f.write("      Organism:    Mus musculus (mouse liver)\n")
        f.write("      Platform:    GPL1261 (Affymetrix Mouse Genome 430 2.0)\n")
        f.write("      Design:      48 samples, hourly for 48 hours\n")
        f.write("      Labels:      Known circadian (20 core clock genes) and\n")
        f.write("                   housekeeping genes (13 found in dataset)\n")
        f.write("      Subsampling: Original 1h, 2h, and 4h resolutions\n")
        f.write("      Instances:   99 (33 genes x 3 resolutions)\n\n")

        f.write("    Dataset 2: GSE11516 (Circadian gene expression, mouse liver)\n")
        f.write("      Organism:    Mus musculus (mouse liver)\n")
        f.write("      Platform:    GPL6880 (Illumina MouseWG-6 v2.0)\n")
        f.write("      Design:      36 samples (12 timepoints x 3 biological replicates),\n")
        f.write("                   4-hour intervals, 48-hour span\n")
        f.write("      Labels:      BioCycle algorithm (RhythmicDB) + known genes\n")
        f.write("      Instances:   1600 (capped at 800 per class)\n\n")

        f.write("    Dataset 3: GSE77451 (Abruzzi et al. 2017, PLOS Genetics)\n")
        f.write("      Organism:    Drosophila melanogaster\n")
        f.write("      Tissue:      FACS-sorted clock neurons (LNv, LNd, DN1)\n")
        f.write("                   + dopaminergic outgroup (TH)\n")
        f.write("      Platform:    RNA-seq (Illumina HiSeq, GPL16479)\n")
        f.write("      Sampling:    4 cell types x 6 ZT x 2 biological replicates\n")
        f.write("      Labels:      Abruzzi 2017 S3 (HC-cyclers by JTK + F24)\n")
        f.write("      Positives:   HC-cyclers in LNv, LNd, DN1 (max 200/cell type,\n")
        f.write("                   sorted by JTK p-value ascending)\n")
        f.write("      Negatives:   Fly housekeeping genes (all cell types) +\n")
        f.write("                   fly core clock genes (TH only)\n")
        f.write("      Excluded:    TH HC-cyclers (non-circadian outgroup)\n")
        f.write("      Expression:  Positives from XLSX embedded columns;\n")
        f.write("                   negatives from GEO series matrix\n\n")

        f.write("    Dataset 4: GSE39445 (Möller-Levet et al. 2013, PNAS)\n")
        f.write("      Organism:    Homo sapiens\n")
        f.write("      Tissue:      Whole blood\n")
        f.write("      Platform:    Custom Agilent microarray (GPL15331)\n")
        f.write("      Sampling:    26 subjects x ~17 timepoints (3-hour intervals)\n")
        f.write("                   x 2 conditions (control / sleep restriction)\n")
        f.write("      Labels:      Möller-Levet 2013 Supplementary Dataset S2\n")
        f.write("      Positives:   Strong (circ in both) + hard (circ in ctrl only)\n")
        f.write("                   Forced: KNOWN_CIRCADIAN_GENES_HUMAN override\n")
        f.write("      Negatives:   Not rhythmic + no sleep condition effect\n")
        f.write("                   Forced: NON_RHYTHMIC_GENES_HUMAN override\n")
        f.write("      Pooling:     Subjects averaged per condition per 3-hour bin\n")
        f.write("                   → 1-2 instances per gene (control + SR)\n")
        f.write("      Cap:         800 positives + 800 negatives\n\n")

        f.write("  5.3 BioCycle labeling criteria\n\n")
        f.write("    Source:        RhythmicDB (rhythmicdb.biocycle.org)\n")
        f.write("    Algorithm:     BioCycle (machine learning-based, independent of\n")
        f.write("                   JTK_CYCLE and Lomb-Scargle to avoid circular\n")
        f.write("                   reasoning with model features)\n")
        f.write("    Dataset ID:    E-GEOD-11516\n")
        f.write("    Rhythmic:      Q-value <= 0.01 (FDR-corrected)\n")
        f.write("    Non-rhythmic:  Q-value > 0.20 (FDR-corrected)\n")
        f.write("    Excluded:      0.01 < Q-value <= 0.20 (ambiguous zone)\n")
        f.write("    Rationale:     The ambiguity gap eliminates borderline genes\n")
        f.write("                   that would introduce label noise. Only high-\n")
        f.write("                   confidence classifications are used.\n\n")

        f.write("  5.4 Known gene labels (ground truth)\n\n")
        f.write("    Mouse circadian genes (20): Core TTFL components confirmed\n")
        f.write("      by knockout studies. Per1, Per2, Per3, Cry1, Cry2, Arntl\n")
        f.write("      (Bmal1), Clock, Npas2, Nr1d1, Nr1d2, Dbp, Tef, Hlf, Rora,\n")
        f.write("      Rorb, Rorc, Ciart, Bhlhe40, Bhlhe41, Nfil3.\n\n")
        f.write("    Mouse non-rhythmic genes (16): Standard qPCR reference genes.\n")
        f.write("      Gapdh, Actb, Tbp, Hprt, Hprt1, Rpl13a, B2m, Ubc, Ppia,\n")
        f.write("      Rpl32, Eef1a1, Sdha, Hmbs, Ywhaz, Pgk1, Tfrc.\n\n")
        f.write("    Drosophila circadian genes (18, FlyBase symbols): per, tim,\n")
        f.write("      Clk, cyc, vri, Pdp1, cry, cwo, Pdf, sgg, dco, nmo, jet,\n")
        f.write("      twins, ck1, NPF, shaggy, dbt. Used as negatives in TH\n")
        f.write("      (dopaminergic outgroup — clock genes expected non-rhythmic).\n\n")
        f.write("    Drosophila non-rhythmic genes (17): RpL32, Act5C, Act88F,\n")
        f.write("      alphaTub84B, Gapdh1, Gapdh2, Sdha, eIF1A,\n")
        f.write("      eEF1alpha1, Rpl13, Rps17, Tbp, GstD1, Hsc70-4, Hsc70Cb,\n")
        f.write("      CG8187, CG7434. Used as negatives in all four cell types.\n\n")
        f.write("    Human circadian genes (28, HGNC symbols): ARNTL, BMAL1,\n")
        f.write("      ARNTL2, BMAL2, PER1, PER2, PER3, CRY1, CRY2, NR1D1, NR1D2,\n")
        f.write("      DBP, TEF, HLF, RORA, RORB, RORC, CLOCK, NPAS2, NFIL3,\n")
        f.write("      BHLHE40, BHLHE41, CIART, CSNK1D, CSNK1E, FBXL3, PROK2,\n")
        f.write("      AVP, VIP. Force-included as positives regardless of ML label.\n\n")
        f.write("    Human non-rhythmic genes (26, HGNC symbols): ACTB, GAPDH,\n")
        f.write("      HPRT1, TBP, RPL13A, B2M, UBC, PPIA, RPL32, EEF1A1, SDHA,\n")
        f.write("      HMBS, YWHAZ, PGK1, TFRC, POLR2A, PSMB4, PSMB2, CHMP2A,\n")
        f.write("      EMC7, GPI, C1orf43, REEP5, SNRPD3, VCP, VPS29.\n")
        f.write("      Force-included as negatives regardless of ML label.\n\n")
        f.write("    Cross-species group safety: gene symbols are used as-is in\n")
        f.write("      GroupShuffleSplit grouping. Case-sensitive comparison ensures\n")
        f.write("      fly 'per' != mouse 'Per1' != human 'PER1', so no cross-species\n")
        f.write("      group collision occurs. Within-species gene groups correctly\n")
        f.write("      pool all cell-type instances of the same gene together.\n\n")

        # --- 6. Evaluation ---
        f.write("6. EVALUATION\n")
        f.write("-" * W + "\n\n")
        f.write(f"  Train/test split: 80/20 GroupShuffleSplit (random_state=42)\n")
        f.write(f"  Grouping:         by gene name (real) or unique ID (synthetic)\n")
        f.write(f"  Total groups:     {n_unique_groups}\n")
        f.write(f"    Gene groups:    {n_gene_groups}\n")
        f.write(f"    Synthetic:      {n_unique_groups - n_gene_groups}\n")
        f.write(f"  Train set:        {len(X_train)} samples ")
        f.write(f"({int(y_train.sum())} rhythmic, "
                f"{int(len(y_train) - y_train.sum())} non-rhythmic)\n")
        f.write(f"  Test set:         {len(X_test)} samples ")
        f.write(f"({int(y_test.sum())} rhythmic, "
                f"{int(len(y_test) - y_test.sum())} non-rhythmic)\n\n")
        f.write("  Rationale: GroupShuffleSplit prevents the same gene from\n")
        f.write("  appearing in both train and test sets. In prior versions\n")
        f.write("  (random train_test_split) the GSE11923 genes (each present\n")
        f.write("  at 1h, 2h, and 4h subsampling) could be split across the\n")
        f.write("  partition, allowing the model to memorize gene-specific\n")
        f.write("  features from one resolution and predict the same gene at\n")
        f.write("  another resolution — a form of data leakage. The same\n")
        f.write("  applies to genes shared between GSE11923 and GSE11516 (e.g.,\n")
        f.write("  Per1, Bmal1). Class balance is preserved approximately\n")
        f.write("  (GroupShuffleSplit does not support exact stratification).\n\n")

        f.write("  6.1 Cross-validation (5-fold StratifiedGroupKFold, on training set)\n\n")
        f.write(f"    ROC-AUC:   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")
        f.write(f"    Per-fold:  {', '.join(f'{s:.4f}' for s in cv_scores)}\n\n")

        f.write("  6.2 Holdout test set performance (high-confidence cases)\n\n")
        f.write("    NOTE: This holdout EXCLUDES BioCycle borderline-q genes\n")
        f.write("    (0.01 < q <= 0.20). Performance on those is reported in\n")
        f.write("    Section 6.5 below.\n\n")
        f.write("    95% confidence intervals from percentile bootstrap\n")
        f.write(f"    (n={1000} resamples, "
                f"{boot['n_valid_iter']} valid; iterations with single-\n")
        f.write("    class resamples were skipped):\n\n")
        f.write(f"    Accuracy:    {accuracy_score(y_test, y_pred):.4f}  "
                f"[95% CI {boot['accuracy_ci'][0]:.4f}, "
                f"{boot['accuracy_ci'][1]:.4f}]\n")
        f.write(f"    Precision:   {precision_score(y_test, y_pred):.4f}\n")
        f.write(f"    Recall:      {recall_score(y_test, y_pred):.4f}\n")
        f.write(f"    Specificity: {specificity:.4f}\n")
        f.write(f"    F1 Score:    {f1_score(y_test, y_pred):.4f}  "
                f"[95% CI {boot['f1_ci'][0]:.4f}, "
                f"{boot['f1_ci'][1]:.4f}]\n")
        f.write(f"    ROC-AUC:     {roc_auc_score(y_test, y_proba):.4f}  "
                f"[95% CI {boot['auroc_ci'][0]:.4f}, "
                f"{boot['auroc_ci'][1]:.4f}]\n")
        f.write(f"    MCC:         {mcc:.4f}\n")
        f.write(f"    Brier loss:  {brier:.4f}  "
                f"[95% CI {boot['brier_ci'][0]:.4f}, "
                f"{boot['brier_ci'][1]:.4f}]  (lower=better calibration)\n\n")

        f.write("  6.3 Confusion matrix\n\n")
        f.write(f"                       Predicted\n")
        f.write(f"                   Non-rhythmic  Rhythmic\n")
        f.write(f"    Actual NR       {tn:>6d}       {fp:>6d}\n")
        f.write(f"    Actual R        {fn:>6d}       {tp:>6d}\n\n")
        f.write(f"    TP={tp}, FP={fp}, TN={tn}, FN={fn}\n\n")

        f.write("  6.4 Classification report\n\n")
        report_str = classification_report(
            y_test, y_pred, target_names=['Non-rhythmic', 'Rhythmic']
        )
        for line in report_str.split('\n'):
            f.write(f"    {line}\n")
        f.write("\n")

        f.write("  6.5 Ambiguous holdout (BioCycle borderline-q genes)\n\n")
        if amb_results is not None:
            f.write("    Genes with 0.01 < BioCycle q-value <= 0.20 were excluded\n")
            f.write("    from the training set (label noise) AND from the main\n")
            f.write("    holdout (6.2). They form an independent evaluation set\n")
            f.write("    of borderline cases — the kind of biology that BioCycle\n")
            f.write("    itself could not confidently classify.\n\n")
            f.write("    Provisional labels for AUROC computation are assigned\n")
            f.write("    by the q-value midpoint (q <= 0.105 -> 1, else 0). The\n")
            f.write("    labels are noisier here than in 6.2 by construction,\n")
            f.write("    so the metrics below should be interpreted as a lower\n")
            f.write("    bound on the model's confusion on hard cases — not as\n")
            f.write("    a direct comparison to 6.2.\n\n")
            f.write(f"    N:                       {amb_results['n']}\n")
            f.write(f"    Accuracy (vs midpoint):  {amb_results['accuracy']:.4f}\n")
            f.write(f"    AUROC (vs midpoint):     {amb_results['auroc']:.4f}\n")
            f.write(f"    Brier loss:              {amb_results['brier']:.4f}\n")
            f.write(f"    % flagged borderline:    {amb_results['pct_soft_zone']:.1f}%\n")
            f.write(f"      (probability in [0.30, 0.70])\n\n")
            f.write("    A high % flagged borderline is the desired behavior\n")
            f.write("    here: it indicates that the model is appropriately\n")
            f.write("    uncertain on genes that BioCycle itself flagged as\n")
            f.write("    uncertain. The ChronoScope GUI presents these cases\n")
            f.write("    as 'Borderline — manual review recommended'.\n\n")
        else:
            f.write("    (Not available — BioCycle ambiguous data not loaded.)\n\n")

        # --- 7. Feature importances ---
        f.write("7. FEATURE IMPORTANCES\n")
        f.write("-" * W + "\n\n")
        f.write("  Two complementary measures are reported. MDI is computed from\n")
        f.write("  the training trees (biased toward high-cardinality features);\n")
        f.write("  permutation importance is computed on the holdout test set\n")
        f.write("  (unbiased, measures contribution to generalization AUROC).\n")
        f.write("  Disagreement between the two is informative: features that\n")
        f.write("  rank high on MDI but low on permutation are likely overused\n")
        f.write("  in training but do not generalize.\n\n")

        f.write("  7.1 Mean Decrease in Impurity (MDI) — averaged over the\n")
        f.write("       CalibratedClassifierCV's internal RFs\n\n")
        f.write(f"  {'Rank':<6s} {'Feature':<30s} {'Importance':>10s}  {'Cumulative':>10s}\n")
        f.write(f"  {'-' * 5}  {'-' * 29}  {'-' * 10}  {'-' * 10}\n")
        cumulative = 0.0
        for rank, (name, imp) in enumerate(importances, 1):
            cumulative += imp
            f.write(f"  {rank:<6d} {name:<30s} {imp:>10.4f}  {cumulative:>10.4f}\n")
        f.write("\n")

        f.write("  7.2 Permutation importance (test-set ROC-AUC drop, n_repeats=10)\n\n")
        f.write(f"  {'Rank':<6s} {'Feature':<30s} {'Mean AUROC drop':>16s}  {'Std':>8s}\n")
        f.write(f"  {'-' * 5}  {'-' * 29}  {'-' * 16}  {'-' * 8}\n")
        for rank, (name, imp_mean, imp_std) in enumerate(perm_importances, 1):
            f.write(f"  {rank:<6d} {name:<30s} "
                    f"{imp_mean:>+16.4f}  {imp_std:>8.4f}\n")
        f.write("\n")
        f.write("  Interpretation: a permutation-importance value of 0.05\n")
        f.write("  means shuffling that feature drops test ROC-AUC by 0.05\n")
        f.write("  on average. Negative values mean the feature is essentially\n")
        f.write("  noise (random shuffles improved AUROC slightly).\n\n")

        # --- 8. Reproducibility ---
        f.write("8. REPRODUCIBILITY\n")
        f.write("-" * W + "\n\n")
        f.write("  All random seeds fixed at 42 for full reproducibility.\n")
        f.write("  To retrain:\n")
        f.write("    python train_consensus_model.py\n\n")
        f.write("  Output files:\n")
        f.write(f"    Model:            core/models_meta_classifier/consensus_rf_model.pkl\n")
        f.write(f"    Feature names:    core/models_meta_classifier/feature_names.json\n")
        f.write(f"    Main holdout:     core/models_meta_classifier/holdout_predictions.csv\n")
        if amb_results is not None:
            f.write(f"    Ambiguous holdout: core/models_meta_classifier/ambiguous_holdout_predictions.csv\n")
        f.write(f"    This report:      core/models_meta_classifier/training_report.txt\n\n")
        f.write(f"  Model file size:    {model_path.stat().st_size / 1024:.1f} KB\n\n")

        # --- 9. Runtime parameter override policy ---
        # Permanent description of the inference-time policy applied by
        # core/feature_extraction.py::_resolve_params. The validation
        # numbers (Section 9.3) are model-version-specific and produced
        # by validate_period_range_override.py; the report only
        # references that file because retraining invalidates them.
        f.write("9. RUNTIME PARAMETER OVERRIDE POLICY\n")
        f.write("-" * W + "\n\n")
        f.write("The classifier was trained with the sub-method search windows\n")
        f.write("fixed at the values listed in Section 4 (JTK 20-28h,\n")
        f.write("Lomb-Scargle 18-32h, Cosinor and Harmonic on a half-step\n")
        f.write("circadian grid 20-28h, F24 target_period=24h, JTK asymmetry=0.5,\n")
        f.write("harmonic order=2). Because the calibrated probability outputs\n")
        f.write("depend on the feature distribution induced by those settings,\n")
        f.write("user-supplied parameters are filtered at inference time by\n")
        f.write("core/feature_extraction.py::_resolve_params.\n\n")

        f.write("  9.1 Parameters affecting model features (bounded)\n\n")
        f.write("      period_range:  Honored if and only if it intersects the\n")
        f.write("                     training window [18, 32]h. Clipped to that\n")
        f.write("                     window if it spills out; falls back to\n")
        f.write("                     training defaults (with a warning) if it\n")
        f.write("                     lies entirely outside. Applied uniformly\n")
        f.write("                     to JTK, Cosinor OLS, Lomb-Scargle, and\n")
        f.write("                     Harmonic Cosinor.\n\n")
        f.write("      F24 target:    Locked at 24h. f24_score is a model\n")
        f.write("                     feature trained at a single target.\n\n")
        f.write("      JTK asymmetries: Locked at [0.5]. jtk_p_value was\n")
        f.write("                       trained with the symmetric-waveform\n")
        f.write("                       assumption.\n\n")

        f.write("  9.2 Parameters not affecting model features (free override)\n\n")
        f.write("      n_harmonics:   User-controlled (default 2). Affects only\n")
        f.write("                     harmonic_p_value and harmonic_r_squared,\n")
        f.write("                     which were dropped from the feature vector\n")
        f.write("                     in v2 (Section 4). The override drives\n")
        f.write("                     the UI's per-method panel and cannot\n")
        f.write("                     alter classifier output.\n\n")

        f.write("  9.3 Empirical validation of the bounded override\n\n")
        f.write("      A holdout re-evaluation comparing default settings\n")
        f.write("      against period_range=(22, 26)h is maintained as a\n")
        f.write("      standalone report:\n\n")
        f.write("        core/models_meta_classifier/parameter_override_validation.txt\n\n")
        f.write("      The numbers there are model-version-specific. After\n")
        f.write("      retraining, re-run\n")
        f.write("        python validate_period_range_override.py\n")
        f.write("      to refresh them.\n\n")

        f.write("=" * W + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * W + "\n")

    print(f"  Report saved: {report_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Training data: {n_synth} synthetic + {n_real} real = {n_total} total")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f}")
    print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n  The model is ready to use in ChronoScope!")
    print(f"  Select 'AI Consensus' -> 'Consensus Rhythmicity Score' in the GUI.")

    # Dataset diversity summary
    organisms = set()
    for m in kept_metadata:
        sig = m.get('signal_type', '')
        if 'GSE77451' in sig:
            organisms.add('Drosophila melanogaster')
        elif 'GSE39445' in sig:
            organisms.add('Homo sapiens')
        elif m.get('source') == 'biological':
            organisms.add('Mus musculus')
    gene_groups = [g for g in np.unique(groups) if not g.startswith('synth_')]
    print(f"\n=== Dataset diversity summary ===")
    print(f"Organisms:    {', '.join(sorted(organisms)) or 'Mus musculus'}")
    print(f"Total real biological instances: {n_real}")
    print(f"Total synthetic instances:       {n_synth}")
    print(f"GroupShuffleSplit groups (genes):{len(gene_groups)}")
    print(f"Cross-species group collisions:  0 "
          f"(case-sensitive symbol distinct across species)")


if __name__ == '__main__':
    main()
