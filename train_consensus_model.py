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

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings during training
warnings.filterwarnings('ignore')


def main():
    print("=" * 70)
    print("CONSENSUS RHYTHMICITY SCORE - MODEL TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1a: Generate synthetic training data
    # ------------------------------------------------------------------
    print("\n[1/6] Generating synthetic training data...")
    from generate_training_data import generate_training_instances

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
    from generate_real_training_data import generate_from_geo

    biocycle_xlsx = os.path.join(project_root, 'data', 'rhythmicdb_query_bioCycle.xlsx')
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
    try:
        print("\n  --- Dataset 2: GSE11516 (BioCycle labels) ---")
        real2_meta, real2_dfs = generate_from_geo(
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
        )
        metadata = metadata + real2_meta
        dataframes = dataframes + real2_dfs
        n_real += len(real2_meta)
        print(f"  GSE11516: {len(real2_meta)} instances")
    except Exception as e:
        print(f"  WARNING: GSE11516 failed: {e}")
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

    from core.feature_extraction import extract_features, FEATURE_NAMES

    feature_rows = []
    labels = []
    start_time = time.time()

    for i, (meta, df) in enumerate(zip(metadata, dataframes)):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  Processing {i + 1}/{n_total} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        variable = meta['variable']
        condition = 'control'

        # Extract times and values
        time_col = 'time'
        condition_col = 'condition'

        cond_data = df[df[condition_col] == condition]
        times = cond_data[time_col].values.astype(float)

        if variable not in cond_data.columns:
            # Skip if variable column missing
            continue

        values = cond_data[variable].values.astype(float)

        # Remove NaN
        valid = ~(np.isnan(times) | np.isnan(values))
        times = times[valid]
        values = values[valid]

        if len(times) < 4:
            continue

        # Average replicates at each timepoint for feature extraction
        unique_times = np.unique(times)
        avg_values = np.array([values[times == t].mean() for t in unique_times])

        # Extract features
        features = extract_features(
            unique_times, avg_values, df, variable, condition,
            time_col, condition_col
        )

        feature_rows.append(features)
        labels.append(meta['is_rhythmic'])

    elapsed_total = time.time() - start_time
    print(f"  Feature extraction complete: {len(feature_rows)} instances in {elapsed_total:.1f}s")

    # ------------------------------------------------------------------
    # Step 3: Build feature matrix
    # ------------------------------------------------------------------
    print("\n[3/6] Building feature matrix...")

    X = np.array([[row.get(name, np.nan) for name in FEATURE_NAMES] for row in feature_rows])
    y = np.array(labels)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels: {sum(y == 1)} rhythmic, {sum(y == 0)} non-rhythmic")

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        train_test_split, cross_val_score, StratifiedKFold
    )
    from sklearn.metrics import (
        classification_report, roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix, matthews_corrcoef,
    )

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")

    # Build pipeline
    # Note: constant fill_value=-999 (sentinel) instead of median imputation.
    # This prevents bias when structurally-missing features (e.g. CWT on short
    # series) get filled with training-set medians that may indicate rhythmicity.
    # The RF learns clean splits: "if feature <= -500 -> feature was unavailable".
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Cross-validation
    print("\n  Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Per-fold:   {[f'{s:.4f}' for s in cv_scores]}")

    # Fit on full training set
    print("\n  Fitting on full training set...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n  --- Test Set Evaluation ---")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-rhythmic', 'Rhythmic']))

    # Feature importances
    rf = pipeline.named_steps['classifier']
    importances = sorted(
        zip(FEATURE_NAMES, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("  Top 10 Feature Importances:")
    for name, imp in importances[:10]:
        print(f"    {name:30s} {imp:.4f}")

    # ------------------------------------------------------------------
    # Step 5: Save model
    # ------------------------------------------------------------------
    print("\n[5/6] Saving model...")

    model_dir = project_root / 'core' / 'models'
    model_dir.mkdir(exist_ok=True)

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
        f.write("CIRCASCOPE - CONSENSUS RHYTHMICITY SCORE (CRS-AI)\n")
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
        f.write("  Pipeline:  SimpleImputer -> StandardScaler -> RandomForestClassifier\n\n")
        f.write("  Step 1 - SimpleImputer:\n")
        f.write("    Strategy:      constant (fill_value = -999)\n")
        f.write("    Rationale:     Sentinel value for structurally-missing features\n")
        f.write("                   (i.e., any sub-method that fails on a given time\n")
        f.write("                   series, e.g., Harmonic Cosinor on very short\n")
        f.write("                   series). Avoids bias from median imputation. The\n")
        f.write("                   RF learns to split on 'feature <= -500' as\n")
        f.write("                   'feature was unavailable'.\n\n")
        f.write("  Step 2 - StandardScaler:\n")
        f.write("    Method:        Zero mean, unit variance normalization\n\n")
        f.write("  Step 3 - RandomForestClassifier:\n")
        f.write(f"    n_estimators:    200\n")
        f.write(f"    max_depth:       10\n")
        f.write(f"    min_samples_leaf: 5\n")
        f.write(f"    class_weight:    balanced\n")
        f.write(f"    random_state:    42\n\n")

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
        f.write("    Generated with generate_training_data.py. Includes cosine waves,\n")
        f.write("    multi-harmonic waveforms, pulse-like patterns, noisy oscillations,\n")
        f.write("    and various non-rhythmic signals (flat, linear trend, random walk,\n")
        f.write("    exponential decay, etc.). 15% of instances include outlier\n")
        f.write("    contamination. Periods sampled from 20-28 hours. Variable SNR,\n")
        f.write("    sampling intervals (1h, 2h, 4h), and series lengths (6-48 points).\n\n")

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
        f.write("    Circadian genes (20): Core clock Transcription-Translation\n")
        f.write("      Feedback Loop (TTFL) components and robust\n")
        f.write("      clock-controlled output genes confirmed by decades of knockout\n")
        f.write("      studies. Per1, Per2, Per3, Cry1, Cry2, Arntl (Bmal1), Clock,\n")
        f.write("      Npas2, Nr1d1, Nr1d2, Dbp, Tef, Hlf, Rora, Rorb, Rorc,\n")
        f.write("      Ciart, Bhlhe40, Bhlhe41, Nfil3.\n\n")
        f.write("    Non-rhythmic genes (16): Standard qPCR reference/housekeeping\n")
        f.write("      genes. Gapdh, Actb, Tbp, Hprt, Hprt1, Rpl13a, B2m, Ubc,\n")
        f.write("      Ppia, Rpl32, Eef1a1, Sdha, Hmbs, Ywhaz, Pgk1, Tfrc.\n\n")

        # --- 6. Evaluation ---
        f.write("6. EVALUATION\n")
        f.write("-" * W + "\n\n")
        f.write(f"  Train/test split: 80/20 (stratified), random_state=42\n")
        f.write(f"  Train set: {len(X_train)} samples\n")
        f.write(f"  Test set:  {len(X_test)} samples\n\n")

        f.write("  6.1 Cross-validation (5-fold stratified, on training set)\n\n")
        f.write(f"    ROC-AUC:   {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")
        f.write(f"    Per-fold:  {', '.join(f'{s:.4f}' for s in cv_scores)}\n\n")

        f.write("  6.2 Holdout test set performance\n\n")
        f.write(f"    Accuracy:    {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"    Precision:   {precision_score(y_test, y_pred):.4f}\n")
        f.write(f"    Recall:      {recall_score(y_test, y_pred):.4f}\n")
        f.write(f"    Specificity: {specificity:.4f}\n")
        f.write(f"    F1 Score:    {f1_score(y_test, y_pred):.4f}\n")
        f.write(f"    ROC-AUC:     {roc_auc_score(y_test, y_proba):.4f}\n")
        f.write(f"    MCC:         {mcc:.4f}\n\n")

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

        # --- 7. Feature importances ---
        f.write("7. FEATURE IMPORTANCES (Random Forest, MDI)\n")
        f.write("-" * W + "\n\n")
        f.write(f"  {'Rank':<6s} {'Feature':<30s} {'Importance':>10s}  {'Cumulative':>10s}\n")
        f.write(f"  {'-' * 5}  {'-' * 29}  {'-' * 10}  {'-' * 10}\n")
        cumulative = 0.0
        for rank, (name, imp) in enumerate(importances, 1):
            cumulative += imp
            f.write(f"  {rank:<6d} {name:<30s} {imp:>10.4f}  {cumulative:>10.4f}\n")
        f.write("\n")

        # --- 8. Reproducibility ---
        f.write("8. REPRODUCIBILITY\n")
        f.write("-" * W + "\n\n")
        f.write("  All random seeds fixed at 42 for full reproducibility.\n")
        f.write("  To retrain:\n")
        f.write("    python train_consensus_model.py\n\n")
        f.write("  Output files:\n")
        f.write(f"    Model:          core/models/consensus_rf_model.pkl\n")
        f.write(f"    Feature names:  core/models/feature_names.json\n")
        f.write(f"    This report:    core/models/training_report.txt\n\n")
        f.write(f"  Model file size:  {model_path.stat().st_size / 1024:.1f} KB\n\n")

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
    print(f"\n  The model is ready to use in CircaScope!")
    print(f"  Select 'AI Consensus' -> 'Consensus Rhythmicity Score' in the GUI.")


if __name__ == '__main__':
    main()
