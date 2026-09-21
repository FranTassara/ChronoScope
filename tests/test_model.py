"""The shipped CRS-AI model and its metadata stay consistent with the code."""
import json
import unittest

from tests.helpers import PROJECT_ROOT, synthetic_series, synthetic_noise


class TestModelArtifacts(unittest.TestCase):

    def test_model_dir_is_core_models(self):
        from core.meta_classifier import ConsensusClassifier
        model_dir = ConsensusClassifier.get_model_dir()
        self.assertEqual(model_dir, PROJECT_ROOT / 'core' / 'models')
        self.assertTrue(model_dir.is_dir(), f'{model_dir} does not exist')

    def test_model_loads(self):
        from core.meta_classifier import ConsensusClassifier
        clf = ConsensusClassifier()
        self.assertTrue(clf.load_model(), 'the packaged model failed to load')

    def test_feature_names_match_feature_extraction(self):
        """If the model was retrained with a different feature set but
        feature_extraction.py was not updated, every prediction is silently
        computed on misaligned columns."""
        from core.feature_extraction import FEATURE_NAMES
        from core.meta_classifier import ConsensusClassifier
        stored = json.loads(
            (ConsensusClassifier.get_model_dir() / 'feature_names.json').read_text(encoding='utf-8'))
        self.assertEqual(list(stored), list(FEATURE_NAMES))

    def test_model_expects_the_same_number_of_features(self):
        from core.feature_extraction import FEATURE_NAMES
        from core.meta_classifier import ConsensusClassifier
        clf = ConsensusClassifier()
        clf.load_model()
        model = clf._model
        n_expected = getattr(model, 'n_features_in_', None)
        if n_expected is None:  # pipeline: ask the first step
            n_expected = getattr(model[0], 'n_features_in_', None)
        self.assertIsNotNone(n_expected, 'could not read n_features_in_ from the model')
        self.assertEqual(n_expected, len(FEATURE_NAMES))


class TestFeatureExtraction(unittest.TestCase):

    def test_extract_features_covers_every_model_feature(self):
        """extract_features() returns a superset of what the model consumes;
        every name the model was trained on must be present and finite."""
        import numpy as np
        from core.feature_extraction import extract_features, FEATURE_NAMES
        times, values = synthetic_series()
        features = extract_features(times, values)
        self.assertIsInstance(features, dict)

        missing = [name for name in FEATURE_NAMES if name not in features]
        self.assertEqual(missing, [], f'features missing from extraction: {missing}')

        vector = np.array([float(features[name]) for name in FEATURE_NAMES])
        self.assertFalse(np.any(np.isnan(vector)),
                         'model feature vector contains NaN for a clean synthetic rhythm')

    def test_rhythmic_series_scores_above_noise(self):
        """End-to-end behavioural check: the packaged model must rank a clean
        24 h rhythm above pure noise sampled on the same grid."""
        from core.meta_classifier import ConsensusClassifier
        clf = ConsensusClassifier()
        self.assertTrue(clf.load_model())

        t_r, v_r = synthetic_series()
        t_n, v_n = synthetic_noise()
        rhythmic = clf.predict(t_r, v_r)
        noise = clf.predict(t_n, v_n)

        self.assertGreater(rhythmic['probability'], noise['probability'])
        self.assertEqual(rhythmic['classification'], 'Rhythmic')
        self.assertGreaterEqual(rhythmic['probability'],
                                ConsensusClassifier.RHYTHMIC_THRESHOLD)


class TestApplicabilityWindow(unittest.TestCase):
    """The length gating added for the JBR revision must stay wired up."""

    def test_timepoint_applicability_boundaries(self):
        from core.meta_classifier import (
            timepoint_applicability, TRAINING_MIN_TIMEPOINTS,
            TRAINING_MAX_TIMEPOINTS, APPLICABLE_MAX_TIMEPOINTS)
        self.assertEqual(timepoint_applicability(None), 'unknown')
        self.assertEqual(timepoint_applicability(TRAINING_MIN_TIMEPOINTS - 1), 'too_short')
        self.assertEqual(timepoint_applicability(TRAINING_MIN_TIMEPOINTS), 'ok')
        self.assertEqual(timepoint_applicability(TRAINING_MAX_TIMEPOINTS), 'ok')
        self.assertEqual(timepoint_applicability(TRAINING_MAX_TIMEPOINTS + 1), 'marginal')
        self.assertEqual(timepoint_applicability(APPLICABLE_MAX_TIMEPOINTS + 1), 'too_long')


if __name__ == '__main__':
    unittest.main()
