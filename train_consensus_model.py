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
    # Step 1: Generate training data
    # ------------------------------------------------------------------
    print("\n[1/5] Generating synthetic training data...")
    from generate_training_data import generate_training_instances

    metadata, dataframes = generate_training_instances(seed=42)
    n_total = len(metadata)
    n_rhythmic = sum(1 for m in metadata if m['is_rhythmic'] == 1)
    n_non_rhythmic = n_total - n_rhythmic
    print(f"  Total instances: {n_total}")
    print(f"  Rhythmic: {n_rhythmic}")
    print(f"  Non-rhythmic: {n_non_rhythmic}")

    # ------------------------------------------------------------------
    # Step 2: Extract features from each instance
    # ------------------------------------------------------------------
    print(f"\n[2/5] Extracting features from {n_total} instances...")
    print("  (This runs 6 analysis methods per instance - please wait)")

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
    print("\n[3/5] Building feature matrix...")

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
    print("\n[4/5] Training Random Forest model...")

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (
        train_test_split, cross_val_score, StratifiedKFold
    )
    from sklearn.metrics import (
        classification_report, roc_auc_score, accuracy_score,
        precision_score, recall_score, f1_score
    )

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train set: {len(X_train)} samples")
    print(f"  Test set:  {len(X_test)} samples")

    # Build pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
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
    print("\n[5/5] Saving model...")

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
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f}")
    print(f"  Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n  The model is ready to use in CircaScope!")
    print(f"  Select 'AI Consensus' -> 'Consensus Rhythmicity Score' in the GUI.")


if __name__ == '__main__':
    main()
