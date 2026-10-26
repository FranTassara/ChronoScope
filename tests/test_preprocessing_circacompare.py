"""Preprocessing pipeline and CircaCompare (Module 0 and Module III).

Regression guards for the 2026-10-19 manual audit, group G6:
  B-10 moving-average detrending and Butterworth smoothing returned the series
       unchanged when time points repeat (replicates or subjects: median
       sampling interval 0), yet the report said applied=True and the GUI
       discarded the report; all stages ran over the rows of a condition in
       file order, mixing replicates and subjects. User decision (2026-10-22):
       with a subject column each subject is its own series; otherwise
       detrending and smoothing filter the per-time mean (sorted by time) and
       add the correction to every value of that time point;
  B-21 CircaCompare Compare Groups failed unless the columns were literally
       named time/condition; a failed optimisation was reported as
       success=True with NaN parameters; the standard errors used n - 2
       residual degrees of freedom for both the 3- and the 6-parameter model.
"""
import io
import os
import contextlib
import unittest
import warnings

import numpy as np
import pandas as pd

from tests.helpers import PROJECT_ROOT  # noqa: F401  (puts the project on sys.path)

warnings.filterwarnings('ignore')

INDEPENDENT_CSV = os.path.join(PROJECT_ROOT, 'examples', 'synthetic_independent_test_data.csv')


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _series(times):
    return pd.DataFrame({'time': times, 'condition': 'A',
                         'y': 10 + 0.1 * np.asarray(times) + np.cos(2 * np.pi * np.asarray(times) / 24)})


def _two_groups(rng, n=12, reps=3):
    t = np.tile(np.repeat(np.arange(0, 24, 24 / n), reps), 2)
    g = np.repeat(['A', 'B'], n * reps)
    y = (10 + np.where(g == 'B', 1.0, 0.0)
         + 2 * np.cos(2 * np.pi * t / 24 - np.where(g == 'B', 1.5, 1.0))
         + rng.normal(0, 0.5, t.size))
    return pd.DataFrame({'time': t, 'condition': g, 'y': y})


def _run_pre(df, subject_col=None, **cfg):
    from core.preprocessing import apply_preprocessing, PreprocessingConfig
    return apply_preprocessing(df, ['y'], PreprocessingConfig(**cfg), subject_col=subject_col)


def _ma_cfg():
    from core.preprocessing import DetrendMethod, SmoothMethod
    return dict(detrend=True, detrend_method=DetrendMethod.MOVING_AVERAGE,
                smooth=True, smooth_method=SmoothMethod.BUTTERWORTH)


class TestPreprocessingReplicates(unittest.TestCase):
    """Without a subject column the time-dependent stages filter the per-time mean (decision (b))."""

    def test_replicated_times_are_filtered_via_the_mean(self):
        from core import preprocessing as pp
        rng = np.random.default_rng(3)
        df = _series(np.repeat(np.arange(0, 48, 2.0), 3))
        df['y'] += rng.normal(0, 0.3, len(df))
        out, rep = _run_pre(df, **_ma_cfg())
        self.assertTrue(rep.detrend_applied and rep.smooth_applied)
        self.assertEqual(rep.warnings, [])
        t = np.arange(0, 48, 2.0)
        mean = df.groupby('time')['y'].mean().to_numpy()
        expected = pp._smooth_array(pp._detrend_array(t, mean, pp.DetrendMethod.MOVING_AVERAGE),
                                    pp.SmoothMethod.BUTTERWORTH, 3, times=t)
        np.testing.assert_allclose(out.groupby('time')['y'].mean().to_numpy(), expected, atol=1e-10)
        # replicate scatter around the mean is kept
        dev = lambda d: (d['y'] - d.groupby('time')['y'].transform('mean')).to_numpy()
        np.testing.assert_allclose(dev(out), dev(df), atol=1e-10)

    def test_file_order_does_not_matter(self):
        df = _series(np.repeat(np.arange(0, 48, 2.0), 2))
        df['y'] += np.random.default_rng(4).normal(0, 0.3, len(df))
        shuffled = df.sample(frac=1, random_state=0)
        a, _ = _run_pre(df, **_ma_cfg())
        b, _ = _run_pre(shuffled, **_ma_cfg())
        np.testing.assert_allclose(b.loc[a.index, 'y'].to_numpy(), a['y'].to_numpy(), atol=1e-10)

    def test_examples_csv_is_applied(self):
        from core.preprocessing import DetrendMethod
        df = pd.read_csv(INDEPENDENT_CSV).rename(columns={'circadian_pure': 'y'})
        out, rep = _run_pre(df, detrend=True, detrend_method=DetrendMethod.MOVING_AVERAGE)
        self.assertTrue(rep.detrend_applied)
        self.assertEqual(rep.warnings, [])
        self.assertFalse(np.allclose(out['y'].values, df['y'].values))


class TestPreprocessingSubjects(unittest.TestCase):
    """With a subject column every subject is its own series (decision (a))."""

    def _subjects(self, n=3):
        t = np.arange(0, 48, 2.0)
        parts = [pd.DataFrame({'time': t, 'condition': 'A', 'subject': f's{k + 1}',
                               'y': 10 + k * 0.2 * t + np.cos(2 * np.pi * t / 24)})
                 for k in range(n)]
        return pd.concat(parts, ignore_index=True)

    def test_each_subject_equals_processing_it_alone(self):
        df = self._subjects()
        out, rep = _run_pre(df, subject_col='subject', **_ma_cfg())
        self.assertEqual(rep.series_processed, 3)
        for s, part in df.groupby('subject'):
            alone, _ = _run_pre(part, **_ma_cfg())
            np.testing.assert_allclose(out.loc[part.index, 'y'].to_numpy(), alone['y'].to_numpy(), atol=1e-10)

    def test_outliers_are_flagged_per_subject(self):
        df = self._subjects(n=6)
        df.loc[(df.subject == 's1'), 'y'] += 100      # an offset subject, not outliers
        _, pooled = _run_pre(df, remove_outliers=True)
        _, per_subject = _run_pre(df, subject_col='subject', remove_outliers=True)
        self.assertGreater(pooled.outliers_removed, 0)
        self.assertEqual(per_subject.outliers_removed, 0)

    def test_worker_passes_the_subject_column(self):
        from unittest import mock
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod
        from core import preprocessing as pp
        data = self._subjects()

        class Loader:
            _condition_col, _subject_col = 'condition', 'subject'
            def get_data(self): return data
            def get_time_column(self): return 'time'

        cfg = AnalysisConfig(method=AnalysisMethod.CIRCACOMPARE_SINGLE, variables=['y'], conditions=['A'],
                             parameters={'period': 24.0, 'loss': 'linear'},
                             preprocessing=pp.PreprocessingConfig(smooth=True))
        with mock.patch.object(pp, 'apply_preprocessing', wraps=pp.apply_preprocessing) as spy:
            _quiet(AnalysisWorker(cfg, Loader(), 'csv').run)
        self.assertEqual(spy.call_args.kwargs['subject_col'], 'subject')


class TestPreprocessingSkipsAreReported(unittest.TestCase):
    """A stage that cannot run is reported as a warning, not as applied."""

    def test_too_few_time_points_for_butterworth(self):
        from core.preprocessing import SmoothMethod
        df = _series(np.repeat(np.arange(0, 10, 2.0), 3))          # 5 time points
        out, rep = _run_pre(df, smooth=True, smooth_method=SmoothMethod.BUTTERWORTH)
        np.testing.assert_array_equal(out['y'].values, df['y'].values)
        self.assertFalse(rep.smooth_applied)
        self.assertEqual(len(rep.warnings), 1)
        self.assertIn('y / A', rep.warnings[0])
        self.assertIn('at least 6', rep.warnings[0])

    def test_unique_times_are_applied_without_warning(self):
        df = _series(np.arange(0, 96, 2.0))
        out, rep = _run_pre(df, **_ma_cfg())
        self.assertTrue(rep.detrend_applied and rep.smooth_applied)
        self.assertEqual(rep.warnings, [])
        self.assertFalse(np.allclose(out['y'].values, df['y'].values))

    def test_worker_shows_preprocessing_warnings(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod
        from core.preprocessing import PreprocessingConfig, SmoothMethod

        class Loader:
            _condition_col, _subject_col = 'condition', None
            def get_data(self): return _series(np.repeat(np.arange(0, 10, 2.0), 3))
            def get_time_column(self): return 'time'

        cfg = AnalysisConfig(method=AnalysisMethod.CIRCACOMPARE_SINGLE, variables=['y'], conditions=['A'],
                             parameters={'period': 24.0, 'loss': 'linear'},
                             preprocessing=PreprocessingConfig(smooth=True, smooth_method=SmoothMethod.BUTTERWORTH))
        w = AnalysisWorker(cfg, Loader(), 'csv')
        out = []
        w.finished.connect(lambda ok, res, msg: out.append((ok, res, msg)))
        _quiet(w.run)
        ok, res, msg = out[-1]
        self.assertTrue(ok)
        self.assertEqual(len(res), 1)
        self.assertIn('1 warning(s)', msg)
        self.assertIn('Butterworth smoothing skipped', msg)

    def test_rosbash_data_report_preprocessing_not_applied(self):
        # G8: Rosbash expression is extracted per gene without the pipeline; this was silent
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod
        from core.preprocessing import PreprocessingConfig

        class RosbashLoader:
            def gene_exists(self, gene): return True
            def get_gene_expression_df(self, gene, condition, cluster, use_log1p):
                return _series(np.arange(0, 48, 4.0)).assign(condition=condition)

        cfg = AnalysisConfig(method=AnalysisMethod.CIRCACOMPARE_SINGLE, variables=['y'], conditions=['A'],
                             parameters={'period': 24.0, 'loss': 'linear'},
                             preprocessing=PreprocessingConfig(smooth=True))
        w = AnalysisWorker(cfg, RosbashLoader(), 'rosbash')
        out = []
        w.finished.connect(lambda ok, res, msg: out.append((ok, res, msg)))
        _quiet(w.run)
        ok, res, msg = out[-1]
        self.assertTrue(ok, msg)
        self.assertEqual(len(res), 1)
        self.assertIn('not applied to single-cell RNA-seq (Rosbash) data', msg)


class TestCircaCompare(unittest.TestCase):

    def _compare(self, df, **params):
        from core.analysis_engine import AnalysisEngine, AnalysisType
        p = {'period': 24.0, 'loss': 'linear', **params}
        tc = 'time' if 'time' in df.columns else 'ZT'
        cc = 'condition' if 'condition' in df.columns else 'group'
        return _quiet(AnalysisEngine().run_comparison, df, 'y', 'A', 'B', AnalysisType.CIRCACOMPARE_COMPARE,
                      time_col=tc, condition_col=cc, parameters=p)

    def test_custom_column_names(self):
        df = _two_groups(np.random.default_rng(0))
        np.random.seed(0)
        ref = self._compare(df)
        np.random.seed(0)
        ren = self._compare(df.rename(columns={'time': 'ZT', 'condition': 'group'}))
        self.assertTrue(ren.success, ren.message)
        self.assertAlmostEqual(ren.amplitude_diff, ref.amplitude_diff, places=10)

    def test_failed_fit_is_not_success(self):
        df = _two_groups(np.random.default_rng(0))
        r = self._compare(df, max_iterations=0)
        self.assertFalse(r.success)
        self.assertIn('Comparison failed', r.message)
        rows = self._compare(df, max_iterations=0, period=[22.0, 24.0])
        self.assertEqual([(r.success, r.period) for r in rows], [(False, 22.0), (False, 24.0)])
        self.assertIn('period 22 h', rows[0].message)

    def test_failed_single_fit_has_message(self):
        from core.analysis_engine import AnalysisEngine, AnalysisType
        df = _two_groups(np.random.default_rng(0))
        r = _quiet(AnalysisEngine().run_analysis, df, 'y', 'A', AnalysisType.CIRCACOMPARE_SINGLE,
                   parameters={'period': 24.0, 'loss': 'linear', 'max_iterations': 0})
        self.assertFalse(r.success)
        self.assertIn('Fit failed', r.message)
        rows = _quiet(AnalysisEngine().run_analysis, df, 'y', 'A', AnalysisType.CIRCACOMPARE_SINGLE,
                      parameters={'period': [22.0, 24.0], 'loss': 'linear', 'max_iterations': 0})
        self.assertEqual([(r.success, r.period) for r in rows], [(False, 22.0), (False, 24.0)])

    def test_worker_reports_failed_comparison(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod

        class Loader:
            _condition_col, _subject_col = 'condition', None
            def get_data(self): return _two_groups(np.random.default_rng(0))
            def get_time_column(self): return 'time'

        cfg = AnalysisConfig(method=AnalysisMethod.CIRCACOMPARE_COMPARE, variables=['y'], conditions=['A', 'B'],
                             parameters={'period': 24.0, 'loss': 'linear', 'max_iterations': 0})
        w = AnalysisWorker(cfg, Loader(), 'csv')
        out = []
        w.finished.connect(lambda ok, res, msg: out.append((ok, res, msg)))
        _quiet(w.run)
        ok, res, msg = out[-1]
        self.assertFalse(ok)                           # no NaN row in the table
        self.assertIn('1 error(s)', msg)
        self.assertIn('Comparison failed', msg)

    def test_standard_errors_use_n_minus_p(self):
        from core import circacompare_analysis as cc
        df = _two_groups(np.random.default_rng(1))
        t = 2 * np.pi * df['time'].values / 24
        y = df['y'].values
        g = (df['condition'].values == 'B').astype(float)
        for res, n_par in ((cc._circacompare_core(t, y, g), 6),
                           (cc._circa_single_core(t[g == 0], y[g == 0]), 3)):
            n = res.fun.size
            expected = (np.sqrt(np.diag(np.linalg.inv(res.jac.T @ res.jac)))
                        * np.sqrt(np.sum(res.fun ** 2) / (n - n_par)))
            np.testing.assert_allclose(res.standard_errors, expected, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
