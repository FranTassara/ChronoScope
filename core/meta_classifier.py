"""
Consensus Rhythmicity Score (AI) Meta-Classifier
==================================================

Uses a pre-trained Random Forest to combine evidence from multiple
circadian analysis methods into a single rhythmicity probability (0-1).

The classifier runs JTK, Cosinor OLS, Lomb-Scargle, and F24 internally,
extracts an 11-feature vector from their results, and feeds
them to the model for prediction.

Author: Francisco Tassara
"""

import sys
import json
import warnings
from typing import Dict, Optional, Any, List
from pathlib import Path

import numpy as np
import pandas as pd

# Optional scikit-learn import
try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    try:
        import pickle
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False


class ConsensusClassifier:
    """
    AI-based consensus rhythmicity classifier.

    Runs multiple circadian analysis methods, extracts features,
    and feeds them to a Random Forest model to produce a unified
    rhythmicity probability score (0-1).
    """

    # Classification thresholds
    RHYTHMIC_THRESHOLD = 0.7
    BORDERLINE_THRESHOLD = 0.3

    def __init__(self):
        self._model = None
        self._feature_names = None
        self._model_loaded = False

    @staticmethod
    def get_model_dir() -> Path:
        """Get the model directory, handling PyInstaller bundled apps."""
        if getattr(sys, 'frozen', False):
            # PyInstaller: sys._MEIPASS is the root of the extracted bundle.
            # Models are bundled under core/models/ matching the source layout.
            base_dir = Path(sys._MEIPASS) / 'core'
        else:
            base_dir = Path(__file__).parent
        return base_dir / 'models_meta_classifier'

    def load_model(self) -> bool:
        """Load the pre-trained model. Returns True if successful."""
        model_dir = self.get_model_dir()
        model_path = model_dir / 'consensus_rf_model.pkl'
        features_path = model_dir / 'feature_names.json'

        if not model_path.exists():
            warnings.warn(f"Model file not found: {model_path}")
            return False

        try:
            self._model = joblib.load(str(model_path))

            # The pickled RandomForest was trained with n_jobs=-1 (see
            # train_consensus_model.py), which is baked into the estimator.
            # At inference we only ever score one sample at a time, so
            # parallel tree evaluation is pure overhead — and in a frozen
            # (PyInstaller) app it can trip multiprocessing's 'spawn' start
            # method, which re-launches a second copy of the whole GUI.
            # Force sequential prediction on every fitted fold.
            for calibrated_classifier in getattr(self._model, 'calibrated_classifiers_', []):
                estimator = getattr(calibrated_classifier, 'estimator', None)
                if estimator is not None and hasattr(estimator, 'n_jobs'):
                    estimator.n_jobs = 1
            if hasattr(self._model, 'n_jobs'):
                self._model.n_jobs = 1

            if features_path.exists():
                with open(features_path, 'r') as f:
                    self._feature_names = json.load(f)
            else:
                # Fallback to default feature names
                from .feature_extraction import FEATURE_NAMES
                self._feature_names = FEATURE_NAMES

            self._model_loaded = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    def predict(
        self,
        times: np.ndarray,
        values: np.ndarray,
        data: Optional[pd.DataFrame] = None,
        variable: Optional[str] = None,
        condition: Optional[str] = None,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run all methods and predict rhythmicity probability.

        Args:
            times: Time points array
            values: Measurement values array
            data: Full DataFrame (optional, for F24)
            variable: Variable column name
            condition: Condition value
            time_col: Time column name
            condition_col: Condition column name
            parameters: Optional analysis parameters

        Returns:
            Dict with keys:
            - 'probability': float 0-1
            - 'classification': str ('Rhythmic', 'Borderline', 'Arrhythmic')
            - 'features': Dict of all extracted features
            - 'method_results': Dict for radar chart (method -> score 0-1)
            - 'feature_importances': Dict (feature name -> importance)
            - 'sub_method_details': List of per-method summaries
        """
        if not self._model_loaded:
            if not self.load_model():
                return {
                    'probability': None,
                    'classification': 'Error',
                    'features': {},
                    'method_results': {},
                    'feature_importances': {},
                    'sub_method_details': [],
                    'error': 'Model not loaded. Run train_consensus_model.py first.'
                }

        from .feature_extraction import extract_features

        # Extract features by running all sub-methods
        features = extract_features(
            times, values, data, variable, condition,
            time_col, condition_col, parameters
        )

        # Build feature vector in the correct order
        X = np.array([[features.get(name, np.nan) for name in self._feature_names]])

        # Predict probability
        proba = self._model.predict_proba(X)[0]
        rhythmic_prob = float(proba[1])  # probability of class 1 (rhythmic)

        # Classification
        if rhythmic_prob >= self.RHYTHMIC_THRESHOLD:
            classification = 'Rhythmic'
        elif rhythmic_prob >= self.BORDERLINE_THRESHOLD:
            classification = 'Borderline'
        else:
            classification = 'Arrhythmic'

        # Feature importances from the RF
        importances = self._get_feature_importances()

        # Build method contribution scores for radar chart
        method_results = self._build_method_scores(features)

        # Build per-method detail summary
        sub_method_details = self._build_sub_method_details(features)

        return {
            'probability': rhythmic_prob,
            'classification': classification,
            'features': features,
            'method_results': method_results,
            'feature_importances': importances,
            'sub_method_details': sub_method_details,
        }

    def _get_feature_importances(self) -> Dict[str, float]:
        """Extract feature importances from the trained model.

        Handles three model layouts:
          1. Bare RandomForestClassifier (legacy).
          2. Pipeline(...).named_steps['classifier'] -> RandomForestClassifier.
          3. Pipeline(...).named_steps['classifier'] -> CalibratedClassifierCV
             wrapping a RandomForestClassifier (current). In this case
             importances are averaged across the per-fold base RFs.
        """
        if self._model is None or self._feature_names is None:
            return {}

        # Unwrap Pipeline
        classifier = self._model
        if hasattr(classifier, 'named_steps'):
            classifier = classifier.named_steps.get('classifier', classifier)

        # Direct feature_importances_ (bare RF case)
        if hasattr(classifier, 'feature_importances_'):
            return dict(zip(self._feature_names, classifier.feature_importances_))

        # CalibratedClassifierCV case: average over internal base estimators
        if hasattr(classifier, 'calibrated_classifiers_'):
            try:
                fold_importances = np.array([
                    cc.estimator.feature_importances_
                    for cc in classifier.calibrated_classifiers_
                    if hasattr(cc.estimator, 'feature_importances_')
                ])
                if len(fold_importances) > 0:
                    mean_importances = fold_importances.mean(axis=0)
                    return dict(zip(self._feature_names, mean_importances))
            except (AttributeError, TypeError):
                pass

        return {}

    def _build_method_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Build normalized method contribution scores (0-1) for radar chart.

        Each method's "contribution" is a normalized measure of how much
        evidence it provides for rhythmicity.
        """
        def _safe_get(key, default=np.nan):
            v = features.get(key, default)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return v

        scores = {}

        # JTK: 1 - p_value (higher = more rhythmic)
        jtk_p = _safe_get('jtk_p_value', 1.0)
        scores['JTK'] = max(0.0, 1.0 - jtk_p)

        # Cosinor: R-squared (0-1 naturally)
        scores['Cosinor OLS'] = max(0.0, min(1.0, _safe_get('cosinor_r_squared', 0.0)))

        # Lomb-Scargle: 1 - FAP
        ls_p = _safe_get('ls_p_value', 1.0)
        scores['Lomb-Scargle'] = max(0.0, 1.0 - ls_p)

        # F24: normalize to 0-1 (F24 > 2 is rhythmic, cap at ~5)
        f24 = _safe_get('f24_score', 0.0)
        scores['F24'] = max(0.0, min(1.0, f24 / 5.0))

        # Harmonic: 1 - p_value
        harm_p = _safe_get('harmonic_p_value', 1.0)
        scores['Harmonic'] = max(0.0, 1.0 - harm_p)

        # Method agreement: already 0-1
        scores['4-Method Agreement'] = max(0.0, min(1.0, _safe_get('method_agreement', 0.0)))

        return scores

    def _build_sub_method_details(self, features: Dict[str, float]) -> List[Dict]:
        """Build summary of each sub-method's result."""
        methods = [
            ('JTK Cycle', 'jtk_p_value', 'jtk_period'),
            ('Cosinor OLS', 'cosinor_p_value', 'cosinor_period'),
            ('Lomb-Scargle', 'ls_p_value', 'ls_dominant_period'),
            ('Fourier F24', None, None),
            ('Harmonic Cosinor', 'harmonic_p_value', None),
        ]

        details = []
        for name, p_key, period_key in methods:
            p = features.get(p_key) if p_key else None
            period = features.get(period_key) if period_key else None
            success = p is not None and not (isinstance(p, float) and np.isnan(p))

            entry = {'method': name, 'p_value': p, 'period': period, 'success': success}

            # Special case: F24 uses score instead of p-value
            if name == 'Fourier F24':
                f24 = features.get('f24_score')
                entry['f24_score'] = f24
                entry['success'] = f24 is not None and not (isinstance(f24, float) and np.isnan(f24))

            details.append(entry)

        return details
