"""CRS-AI (Module V): what the user is told about a prediction.

Regression guards for the 2026-10-19 manual audit, group G7 (no retraining,
the 11 features are computed exactly as before):
  M23 a period range clipped to, or replaced by, the training window was only
      reported through warnings.warn, which never reaches the GUI (the app has
      no console); it is now returned with the prediction, shown in a Notes
      column and listed in the end-of-run warning;
  A13 a feature that cannot be computed is imputed with a -999 sentinel that
      no tree of the released model splits on; user decision 2026-10-22: such
      a series is reported as 'Not applicable', without a probability;
  M26 the probability was exported as p_value = 1 - probability; user
      decision 2026-10-22: p_value is left empty;
  the released model must keep reproducing the published holdout
      probabilities (validation/crs_ai/holdout_predictions.csv).
"""
import io
import json
import contextlib
import unittest
import warnings

import numpy as np
import pandas as pd

from tests.helpers import PROJECT_ROOT, synthetic_series, rhythmic_dataframe

warnings.filterwarnings('ignore')


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _classifier():
    from core.meta_classifier import ConsensusClassifier
    clf = ConsensusClassifier()
    assert clf.load_model()
    return clf


class TestPeriodRangeNotice(unittest.TestCase):

    def test_no_notice_inside_the_window(self):
        from core.feature_extraction import period_range_notice
        for p in (None, {}, {'period_range': (20.0, 28.0)}, {'period_range': (18.0, 32.0)}):
            self.assertIsNone(period_range_notice(p))

    def test_clipped_range(self):
        from core.feature_extraction import period_range_notice
        msg = period_range_notice({'period_range': (15.0, 30.0)})
        self.assertIn('clipped to [18, 30] h', msg)

    def test_fallback_without_and_with_a_small_overlap(self):
        from core.feature_extraction import period_range_notice
        self.assertIn('does not overlap', period_range_notice({'period_range': (8.0, 16.0)}))
        # overlapping by <= 1 h also falls back: the old message said "falls outside"
        msg = period_range_notice({'period_range': (31.5, 40.0)})
        self.assertIn('by only 0.5 h', msg)
        self.assertIn('training defaults', msg)

    def test_resolution_policy_is_unchanged(self):
        from core.feature_extraction import _resolve_params
        default = _resolve_params(None)
        self.assertEqual(_resolve_params({'period_range': (8.0, 16.0)}), default)
        self.assertEqual(_resolve_params({'period_range': (31.5, 40.0)}), default)
        clipped = _resolve_params({'period_range': (15.0, 30.0)})
        self.assertEqual(clipped['jtk_periods'], list(range(18, 31)))
        # user decision 2026-10-22: Lomb-Scargle always keeps its training range
        self.assertEqual(clipped['ls_range'], default['ls_range'])
        self.assertEqual(_resolve_params({'period_range': (22.0, 26.0)})['ls_range'], (18.0, 32.0))

    def test_prefilled_gui_range_reproduces_training_features(self):
        from core.feature_extraction import extract_features, FEATURE_NAMES
        t, v = synthetic_series()
        a = extract_features(t, v)
        b = extract_features(t, v, parameters={'period_range': (20.0, 28.0)})
        self.assertEqual([a[n] for n in FEATURE_NAMES], [b[n] for n in FEATURE_NAMES])

    def test_prediction_carries_the_notice(self):
        clf = _classifier()
        t, v = synthetic_series()
        self.assertEqual(clf.predict(t, v)['notes'], [])
        self.assertEqual(clf.predict(t, v, parameters={'period_range': (20.0, 28.0)})['notes'], [])
        clipped = clf.predict(t, v, parameters={'period_range': (15.0, 30.0)})
        explicit = clf.predict(t, v, parameters={'period_range': (18.0, 30.0)})
        self.assertEqual(len(clipped['notes']), 1)
        self.assertEqual(clipped['probability'], explicit['probability'])   # same features as before

    def test_engine_puts_notes_in_the_result_json(self):
        from core.analysis_engine import AnalysisEngine, AnalysisType
        df = rhythmic_dataframe()
        r = _quiet(AnalysisEngine().run_analysis, df, 'signal', 'ctrl', AnalysisType.CONSENSUS_AI,
                   parameters={'period_range': (8.0, 16.0)})
        self.assertTrue(r.success)
        self.assertIn('does not overlap', json.loads(r.message)['notes'][0])


class TestNoticesReachTheGUI(unittest.TestCase):

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])

    def _run_worker(self, period_range):
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod
        df = rhythmic_dataframe()
        df['other'] = df['signal'][::-1].to_numpy()

        class Loader:
            _condition_col, _subject_col = 'condition', None
            def get_data(self): return df
            def get_time_column(self): return 'time'

        cfg = AnalysisConfig(method=AnalysisMethod.CONSENSUS_AI, variables=['signal', 'other'],
                             conditions=['ctrl'], parameters={'period_range': period_range})
        w = AnalysisWorker(cfg, Loader(), 'csv')
        out = []
        w.finished.connect(lambda ok, res, msg: out.append((ok, res, msg)))
        _quiet(w.run)
        return out[-1]

    def test_worker_reports_the_notice_once(self):
        ok, res, msg = self._run_worker((15.0, 30.0))
        self.assertTrue(ok)
        self.assertEqual(len(res), 2)
        self.assertIn('1 warning(s)', msg)
        self.assertIn('clipped to [18, 30] h', msg)

    def test_no_warning_for_the_prefilled_range(self):
        ok, res, msg = self._run_worker((20.0, 28.0))
        self.assertEqual(msg, 'Analysis completed successfully')

    def test_partial_notes_are_counted(self):
        from ui.analysis_panel import AnalysisWorker
        rows = [{'method': 'consensus_ai', 'message': json.dumps({'notes': n})} for n in (['x'], [], ['x'])]
        self.assertEqual(AnalysisWorker._crs_ai_warnings(rows), ['x (2 of 3 series)'])

    def test_results_table_has_a_notes_column(self):
        from ui.results_panel import ResultsPanel
        ok, res, msg = self._run_worker((15.0, 30.0))
        panel = ResultsPanel()
        _quiet(panel.set_results, res)
        table = panel._results_table
        headers = [table.horizontalHeaderItem(j).text() for j in range(table.columnCount())]
        self.assertIn('Notes', headers)
        self.assertIn('clipped', table.item(0, headers.index('Notes')).text())


class TestMissingFeaturesAreNotApplicable(unittest.TestCase):
    """A13 / user decision 2026-10-22: no probability when a model feature is missing."""

    t = np.arange(0, 48, 4.0)

    def test_flat_series(self):
        r = _classifier().predict(self.t, np.full(self.t.size, 5.0))
        self.assertIsNone(r['probability'])
        self.assertEqual(r['classification'], 'Not applicable')
        self.assertEqual(r['missing_features'], ['jtk_p_value', 'ls_power'])
        self.assertIsInstance(r['probability_with_sentinel'], float)
        self.assertIn('Not applicable: jtk_p_value, ls_power', r['notes'][0])

    def test_centred_series_explains_relative_amplitude(self):
        r = _classifier().predict(self.t, np.cos(2 * np.pi * self.t / 24))   # MESOR exactly 0
        self.assertEqual(r['missing_features'], ['amplitude_relative'])
        self.assertIn('not z-scores', r['notes'][0])

    def test_complete_series_keep_their_probability(self):
        r = _classifier().predict(*synthetic_series())
        self.assertEqual(r['missing_features'], [])
        self.assertIsNone(r['probability_with_sentinel'])
        self.assertGreater(r['probability'], 0.7)

    def test_engine_result(self):
        from core.analysis_engine import AnalysisEngine, AnalysisType
        df = rhythmic_dataframe()
        df['flat'] = 5.0
        eng = AnalysisEngine()
        na = _quiet(eng.run_analysis, df, 'flat', 'ctrl', AnalysisType.CONSENSUS_AI, parameters={})
        self.assertTrue(na.success)
        self.assertIsNone(na.r_squared)
        details = json.loads(na.message)
        self.assertEqual(details['classification'], 'Not applicable')
        self.assertIsNotNone(details['probability_with_sentinel'])
        ok = _quiet(eng.run_analysis, df, 'signal', 'ctrl', AnalysisType.CONSENSUS_AI, parameters={})
        self.assertGreater(ok.r_squared, 0.7)
        # M26: the probability is not exported as a p-value any more
        self.assertIsNone(ok.p_value)
        self.assertIsNone(na.p_value)


class TestNotApplicableInTheGUI(unittest.TestCase):

    def setUp(self):
        from PySide6.QtWidgets import QApplication
        self.app = QApplication.instance() or QApplication([])

    def test_worker_table_and_gauge(self):
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod
        from ui.results_panel import ResultsPanel
        df = rhythmic_dataframe()
        df['flat'] = 5.0

        class Loader:
            _condition_col, _subject_col = 'condition', None
            def get_data(self): return df
            def get_time_column(self): return 'time'

        cfg = AnalysisConfig(method=AnalysisMethod.CONSENSUS_AI, variables=['signal', 'flat'],
                             conditions=['ctrl'], parameters={'period_range': (20.0, 28.0)})
        w = AnalysisWorker(cfg, Loader(), 'csv')
        out = []
        w.finished.connect(lambda ok, res, msg: out.append((ok, res, msg)))
        _quiet(w.run)
        ok, res, msg = out[-1]
        self.assertTrue(ok)
        self.assertIn('Not applicable: jtk_p_value, ls_power', msg)
        self.assertIn('(1 of 2 series)', msg)

        panel = ResultsPanel()
        _quiet(panel.set_results, res)
        table = panel._results_table
        headers = [table.horizontalHeaderItem(j).text() for j in range(table.columnCount())]
        row = [table.item(1, j).text() for j in range(table.columnCount())]
        self.assertEqual(row[headers.index('Probability')], 'N/A')
        self.assertEqual(row[headers.index('Classification')], 'Not applicable')
        self.assertIn('not applicable', panel._build_results_summary(res))
        _quiet(panel._plot_probability_gauge, None, 'Not applicable', res[1])   # no needle, no crash


class TestPublishedModelIsUntouched(unittest.TestCase):

    def test_holdout_probabilities_reproduce(self):
        clf = _classifier()
        vd = PROJECT_ROOT / 'validation' / 'crs_ai'
        x = np.load(vd / 'X_test.npy')
        published = pd.read_csv(vd / 'holdout_predictions.csv')
        np.testing.assert_allclose(clf._model.predict_proba(x)[:, 1], published['y_proba'], rtol=0, atol=1e-12)

    def test_classification_thresholds(self):
        from core.meta_classifier import ConsensusClassifier as C
        self.assertEqual((C.RHYTHMIC_THRESHOLD, C.BORDERLINE_THRESHOLD), (0.7, 0.3))


class TestDocumentation(unittest.TestCase):

    def test_training_report_matches_the_code(self):
        text = (PROJECT_ROOT / 'validation' / 'crs_ai' / 'training_report.txt').read_text(encoding='utf-8')
        self.assertNotIn('Arrhythmic <= 0.30', text)          # the code uses < 0.30
        self.assertNotIn('five established', text)            # four sub-methods feed the model
        self.assertNotIn("learns to split on 'feature <= -500'", text)

    def test_gui_description_states_the_fitted_partition(self):
        src = (PROJECT_ROOT / 'ui' / 'analysis_panel.py').read_text(encoding='utf-8')
        self.assertNotIn('trained on "\n                 "5,649', src)
        self.assertIn('4,532 of 5,649', src)
        self.assertNotIn('LS log-spaced', src)


if __name__ == '__main__':
    unittest.main()
