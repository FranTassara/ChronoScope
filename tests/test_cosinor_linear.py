"""CosinorPy linear methods, Module II 2.1-2.5.

Regression guards for the 2026-10-19 manual audit, group G4:
  B-5  LimoRhyde (Compare Conditions (Independent), multi-component) always
       failed with pandas 3 and the engine returned 0 rows without a message;
  B-13 "Save CosinorPy plots" saved nothing for Independent Models, the
       multi-component Independent method and the dependent multi-component
       comparison, and cosinorpy_plots/ was created with the box unchecked;
  B-14 acrophase_hours was negative (or None) in the multi-component paths;
  B-15 dependent multi-component comparison: p1/p2 were overwritten by the
       n = 1 enrichment re-fit (q < p in the table);
  B-18 periodogram with replicated time points: sampling frequency 1/0;
  plus the subject column name, the "Best model" label, period2 in the
  enrichment, the p_reject colour and the errors of the compare methods.
"""
import io
import os
import shutil
import tempfile
import contextlib
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd

from tests.helpers import PROJECT_ROOT  # noqa: F401  (puts the project on sys.path)

warnings.filterwarnings('ignore')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _cosine(t, period, peak, mesor=10.0, amp=3.0):
    return mesor + amp * np.cos(2 * np.pi * (t - peak) / period)


def _independent(spec, step=2.0, hours=48, reps=3, noise=0.4, seed=7):
    """spec: {condition: (period, peak_time, mesor)}; independent noise per replicate."""
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(0.0, hours, step), reps)
    return pd.concat([pd.DataFrame({'time': t, 'condition': c,
                                    'y': _cosine(t, per, pk, mesor) + rng.normal(0, noise, t.size)})
                      for c, (per, pk, mesor) in spec.items()], ignore_index=True)


def _dependent(spec, subjects=4, step=2.0, hours=48, noise=0.4, seed=5, subject_col='subject'):
    """spec: {condition: (period, peak_time, mesor)}; one series per subject and condition."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, hours, step)
    return pd.concat([pd.DataFrame({'time': t, 'condition': c, subject_col: f'{c}{s}',
                                    'y': _cosine(t, per, pk + rng.normal(0, 0.3), mesor) + rng.normal(0, noise, t.size)})
                      for c, (per, pk, mesor) in spec.items() for s in range(subjects)], ignore_index=True)


def _run(data, atype, params, path=None):
    from core.analysis_engine import AnalysisEngine, AnalysisType
    res = _quiet(AnalysisEngine().run_analysis, data, 'y', data.condition.iloc[0],
                 getattr(AnalysisType, atype), parameters=params, data_file_path=path)
    return res if isinstance(res, list) else [res]


class TestLimoRhyde(unittest.TestCase):
    """B-5: LimoRhyde runs under pandas 3 and its joint tests get their own columns."""

    @staticmethod
    def _limo(d, analysis):
        return _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
                    {'period': 24.0, 'n_components': [2], 'comparison_method': 'LimoRhyde',
                     'analysis_method': analysis, 'bootstrap_size': 50})

    def test_joint_tests_without_analysis(self):
        same = _independent({'A': (24, 4, 10), 'B': (24, 4, 10)}, seed=3)
        diff = _independent({'A': (24, 4, 10), 'B': (24, 10, 12)}, seed=3)
        r_same, r_diff = self._limo(same, 'None'), self._limo(diff, 'None')
        for rows in (r_same, r_diff):
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertTrue(r.success, r.message)
            self.assertEqual(r.method, 'cosinorpy_compare_limorhyde')
            for v in (r.model_p, r.p_params, r.p_ftest):
                self.assertTrue(0 <= v <= 1)
            # no per-parameter test without an analysis: p params is not copied into p-Amp/p-Acro/p-MESOR
            self.assertIsNone(r.p_amplitude)
            self.assertIsNone(r.p_mesor)
        self.assertGreater(r_same[0].p_ftest, 0.05)        # identical rhythms
        self.assertLess(r_diff[0].p_ftest, 1e-6)           # phase and MESOR differ

    def test_p_params_uses_condition_terms_only(self):
        # H-G4-2: 'p params' = smallest p of the condition x rhythm terms; CosinorPy 3.1 included the
        # last common-rhythm coefficient, so it was ~1e-42 for two identical rhythmic conditions (n = 1)
        import statsmodels.api as sm
        from core.analysis_engine import AnalysisEngine
        from core.vendor.cosinorpy.limo import compare_pair_df_extended
        d = _independent({'A': (24, 4, 10), 'B': (24, 4, 10)}, seed=3)
        df = _quiet(AnalysisEngine()._convert_to_cosinorpy_format_all_conditions, d, 'y', 'time', 'condition')
        for n in (1, 2):
            _, p_params, p_f, _, _, _ = _quiet(compare_pair_df_extended, df, 'y-A', 'y-B',
                                               n_components=n, period=24, plot=False)
            x, h = df.x.values, (df.test == 'y-B').values.astype(float)
            harm = np.column_stack([f(2 * np.pi * x * k / 24) for k in range(1, n + 1) for f in (np.sin, np.cos)])
            fit = sm.OLS(df.y.values, sm.add_constant(np.column_stack([harm, harm * h[:, None], h]))).fit()
            self.assertAlmostEqual(p_params, fit.pvalues[1 + 2 * n:1 + 4 * n].min(), places=10)
            self.assertGreater(p_params, 0.01, n)

    def test_bootstrap2(self):
        # bootstrap2 stored its CI/p under keys the library loop does not read
        r = self._limo(_independent({'A': (24, 4, 10), 'B': (24, 10, 12)}, seed=3), 'Bootstrap2')[0]
        self.assertTrue(r.success, r.message)
        self.assertIsNotNone(r.amplitude_diff_ci)
        self.assertTrue(0 <= r.p_mesor <= 1)
        self.assertGreater(r.mesor_diff_ci[0], 0)          # MESOR B - A = +2


class TestSingleComponentPeriodRange(unittest.TestCase):
    """Pooled Model / Independent Models test every period of a range (was: only the first)."""

    def test_every_period(self):
        d = _independent({'A': (24, 4, 10), 'B': (24, 8, 12)})
        for ctype in ('Pooled Model', 'Independent Models'):
            rows = _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
                        {'period': [22.0, 24.0, 26.0], 'n_components': [1], 'comparison_type': ctype})
            self.assertTrue(all(r.success for r in rows), ctype)
            self.assertEqual(sorted(r.period for r in rows), [22.0, 24.0, 26.0], ctype)

    def test_per_condition_periods_still_one_row(self):
        d = _independent({'A': (20, 4, 10), 'B': (28, 4, 12)})
        rows = _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
                    {'period': [22.0, 24.0], 'n_components': [1], 'comparison_type': 'Independent Models',
                     'period1': 20.0, 'period2': 28.0})
        self.assertEqual([(r.period, r.period2) for r in rows], [(20.0, 28.0)])


class TestLinearComponentDifference(unittest.TestCase):
    """Include Linear Component (2.4, multi-component Independent): CosinorPy's d_lin_comp is reported."""

    def test_reported(self):
        from CosinorPy import cosinor
        from core.analysis_engine import AnalysisEngine
        d = _independent({'A': (24, 4, 10), 'B': (24, 8, 12)})
        d.loc[d.condition == 'B', 'y'] += 0.05 * d.loc[d.condition == 'B', 'time']
        r = _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
                 {'period': 24.0, 'n_components': [2], 'comparison_method': 'Independent',
                  'analysis_method': 'CI', 'include_lin_comp': True})[0]
        df = _quiet(AnalysisEngine()._convert_to_cosinorpy_format_all_conditions, d, 'y', 'time', 'condition')
        lib = _quiet(cosinor.compare_pairs, df, [('y-A', 'y-B')], n_components=[2], period=[24.0],
                     analysis='CI', lin_comp=True).iloc[0]
        self.assertAlmostEqual(r.lin_comp_diff, lib['d_lin_comp'], places=10)
        self.assertAlmostEqual(r.lin_comp_diff, 0.05, delta=0.02)
        self.assertAlmostEqual(r.p_lin_comp, lib['p(d_lin_comp)'], places=10)
        self.assertIsNotNone(r.lin_comp_diff_ci)


class TestAcrophaseHours(unittest.TestCase):
    """B-14: acrophase_hours is wrapped to [0, period) in every path."""

    def test_independent_multicomponent_late_peak(self):
        d = _independent({'A': (24, 20.37, 10)})
        for nc in ([2], [1, 2]):
            for r in _run(d, 'COSINORPY_INDEPENDENT', {'period_range': (24.0, 24.0), 'n_components': nc}):
                self.assertTrue(0 <= r.acrophase_hours < 24, (nc, r.acrophase_hours))
                self.assertAlmostEqual(r.acrophase_hours, 20.37, delta=0.5)

    def test_dependent_multicomponent(self):
        d = _dependent({'A': (24, 20.37, 10)})
        for nc in ([2], [1, 2]):
            rows = _run(d, 'COSINORPY_DEPENDENT', {'period_range': (24.0, 24.0), 'n_components': nc})
            for r in rows:
                self.assertIsNotNone(r.acrophase_hours, nc)         # was None with a single row
                self.assertTrue(0 <= r.acrophase_hours < 24, (nc, r.acrophase_hours))
                self.assertAlmostEqual(r.acrophase_hours, 20.37, delta=0.5)


class TestBestModelLabel(unittest.TestCase):
    """The selected multi-component model is labelled by how it was selected."""

    def test_independent_ftest(self):
        d = _independent({'A': (24, 4, 10)})
        rows = _run(d, 'COSINORPY_INDEPENDENT', {'period_range': (24.0, 24.0), 'n_components': [1, 2, 3]})
        self.assertEqual(sum(r.best_model == 'Best model (F-test)' for r in rows), 1)
        self.assertFalse(any('min p-value' in (r.best_model or '') for r in rows))

    def test_dependent_single_n_period_range(self):
        d = _dependent({'A': (24, 4, 10)})
        rows = _run(d, 'COSINORPY_DEPENDENT', {'period_range': (20.0, 28.0), 'period_step': 4.0, 'n_components': [2]})
        best = [r for r in rows if (r.best_model or '').startswith('Best model')]
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].best_model, 'Best model (min RSS)')
        self.assertEqual(best[0].rss, min(r.rss for r in rows))


class TestDependentMultiP1P2(unittest.TestCase):
    """B-15: p1/p2 of the multi-component comparison are the library's n-component p-values."""

    def test_p1_p2_kept(self):
        from CosinorPy import cosinor
        from core.analysis_engine import AnalysisEngine
        d = _dependent({'A': (24, 4, 10), 'B': (24, 8, 10)}, noise=2.0)
        rows = _run(d, 'COSINORPY_COMPARE_DEPENDENT', {'period': 24.0, 'n_components': [2]})
        df = _quiet(AnalysisEngine()._convert_to_cosinorpy_format_population_all_conditions,
                    d, 'y', 'time', 'condition', 'subject')
        lib = _quiet(cosinor.compare_pairs_population, df, [('y-A', 'y-B')], n_components=[2], period=[24.0])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].p1, lib['p1'].iloc[0], places=12)
        self.assertAlmostEqual(rows[0].p2, lib['p2'].iloc[0], places=12)


class TestSavePlots(unittest.TestCase):
    """B-13."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='g4_plots_')
        self.path = os.path.join(self.dir, 'data.csv')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _files(self):
        folder = os.path.join(self.dir, 'cosinorpy_plots')
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else None

    def test_independent_models(self):
        d = _independent({'A': (20, 4, 10), 'B': (28, 4, 12)})
        _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
             {'period': 24.0, 'n_components': [1], 'comparison_type': 'Independent Models',
              'period1': 20.0, 'period2': 28.0, 'save_cosinorpy_plots': True}, self.path)
        files = self._files()
        self.assertIn('y-A_20.0.png', files)       # each condition at its own period
        self.assertIn('y-B_28.0.png', files)

    def test_multicomponent_independent(self):
        d = _independent({'A': (24, 4, 10), 'B': (24, 8, 12)})
        _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
             {'period': 24.0, 'n_components': [2], 'comparison_method': 'Independent',
              'analysis_method': 'CI', 'save_cosinorpy_plots': True}, self.path)
        files = self._files()   # fit_me also saves a phase plot per condition
        for name in ('y-A_compnts=2_per=24.0.png', 'y-B_compnts=2_per=24.0.png', 'y-A_phase.png'):
            self.assertIn(name, files)

    def test_dependent_multicomponent(self):
        d = _dependent({'A': (24, 4, 10), 'B': (24, 8, 12)})
        _run(d, 'COSINORPY_COMPARE_DEPENDENT',
             {'period': 24.0, 'n_components': [2], 'save_cosinorpy_plots': True}, self.path)
        self.assertEqual(len([f for f in self._files() if f.endswith('.png')]), 2)

    def test_no_folder_when_unchecked(self):
        d = _independent({'A': (24, 4, 10)})
        _run(d, 'COSINORPY_INDEPENDENT', {'period_range': (24.0, 24.0), 'n_components': [1]}, self.path)
        dd = _dependent({'A': (24, 4, 10)})
        _run(dd, 'COSINORPY_DEPENDENT', {'period_range': (24.0, 24.0), 'n_components': [1]}, self.path)
        self.assertIsNone(self._files())


class TestPeriodogram(unittest.TestCase):
    """B-18: the sampling interval comes from the unique time points."""

    def _sampling_f(self, d):
        from core.cosinor_analysis import CosinorAnalyzer, DataType
        a = CosinorAnalyzer()
        _quiet(a.load_data, pd.DataFrame({'x': d.time.values, 'y': d.y.values, 'test': 'y_A'}), DataType.INDEPENDENT)
        with mock.patch('core.cosinor_analysis.cosinor.periodogram') as per:
            res = _quiet(a.periodogram, save_folder=None)
        return per.call_args.kwargs['sampling_f'], res

    def test_replicates(self):
        d = _independent({'A': (24, 4, 10)}, step=2.0, reps=3)
        self.assertEqual(d.time.iloc[0], d.time.iloc[1])   # X[1] - X[0] = 0 in CosinorPy
        sf, res = self._sampling_f(d)
        self.assertAlmostEqual(sf, 0.5)
        self.assertTrue(res['success'])
        self.assertEqual(res['warnings'], [])

    def test_uneven_sampling_is_reported(self):
        d = _independent({'A': (24, 4, 10)}, reps=1)
        d = d[d.time != 10.0]
        _, res = self._sampling_f(d)
        self.assertEqual(len(res['warnings']), 1)


class TestSubjectColumn(unittest.TestCase):
    """Dependent methods use the subject column detected at load time, whatever its name."""

    def test_animal_column(self):
        d = _dependent({'A': (24, 4, 10), 'B': (24, 8, 12)}, subject_col='animal')
        r = _run(d, 'COSINORPY_DEPENDENT', {'period_range': (24.0, 24.0), 'n_components': [1],
                                            'subject_col': 'animal'})[0]
        self.assertTrue(r.success, r.message)
        rows = _run(d, 'COSINORPY_COMPARE_DEPENDENT', {'period': 24.0, 'n_components': [1], 'subject_col': 'animal'})
        self.assertTrue(rows and all(x.success for x in rows))

    def test_missing_subject_column_is_reported(self):
        d = _independent({'A': (24, 4, 10), 'B': (24, 8, 12)})
        rows = _run(d, 'COSINORPY_COMPARE_DEPENDENT', {'period': 24.0, 'n_components': [1]})
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0].success)
        self.assertIn('subject column', rows[0].message)


class TestIndependentModelsPeriod2(unittest.TestCase):
    """With period2 != period1, condition 2 is reported and re-fitted at period2."""

    def test_enrichment_uses_period2(self):
        from CosinorPy import cosinor1
        d = _independent({'A': (20, 4, 10), 'B': (28, 4, 15)}, hours=96)
        r = _run(d, 'COSINORPY_COMPARE_INDEPENDENT',
                 {'period': 24.0, 'n_components': [1], 'comparison_type': 'Independent Models',
                  'period1': 20.0, 'period2': 28.0})[0]
        self.assertEqual((r.period, r.period2), (20.0, 28.0))
        B = d[d.condition == 'B']
        fit28 = _quiet(cosinor1.fit_cosinor, B.time.values, B.y.values, period=28.0, plot_on=False)[0]
        self.assertAlmostEqual(r.mesor_g1, float(fit28.params.iloc[0]), places=10)


class TestCompareErrorsReported(unittest.TestCase):
    """Compare methods return a failed result with the error instead of 0 rows."""

    def test_single_condition(self):
        d = _independent({'A': (24, 4, 10)})
        for atype in ('COSINORPY_COMPARE_INDEPENDENT', 'COSINORPY_NONLINEAR_COMPARE_INDEPENDENT'):
            rows = _run(d, atype, {'period': 24.0, 'n_components': [1]})
            self.assertEqual(len(rows), 1, atype)
            self.assertFalse(rows[0].success)
            self.assertIn('At least 2 conditions', rows[0].message)

    def test_worker_reports_errors(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisWorker, AnalysisConfig, AnalysisMethod

        class Loader:
            _condition_col, _subject_col = 'condition', None
            def get_data(self): return _independent({'A': (24, 4, 10), 'B': (24, 8, 12)})
            def get_time_column(self): return 'time'

        cfg = AnalysisConfig(method=AnalysisMethod.COSINORPY_COMPARE_DEPENDENT, variables=['y'],
                             conditions=['A', 'B'], parameters={'period': 24.0, 'n_components': [1]})
        w = AnalysisWorker(cfg, Loader(), 'csv')
        out = []
        w.finished.connect(lambda ok, res, msg: out.append((ok, res, msg)))
        _quiet(w.run)
        ok, res, msg = out[-1]
        self.assertFalse(ok)                           # nothing succeeded -> error dialog
        self.assertIn('1 error(s)', msg)
        self.assertIn('subject column', msg)


class TestPRejectColour(unittest.TestCase):
    """p_reject < 0.05 is a lack of fit: never highlighted as significant (green)."""

    def test_not_green(self):
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.results_panel import ResultsPanel
        rp = ResultsPanel()
        row = dict(variable='v', condition='c', method='cosinorpy_independent', n_components=2, period=24.0,
                   p_value=1e-6, q_value=1e-6, p_reject=0.01, q_reject=0.01, amplitude=1.0, mesor=1.0,
                   acrophase=-1.0, acrophase_hours=3.8, success=True)
        _quiet(rp.add_results, [row])
        t = rp._results_table
        hdr = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
        self.assertEqual(t.item(0, hdr.index('p')).background().color().name(), '#00ff00')
        for col in ('p_reject', 'q_reject'):
            self.assertNotEqual(t.item(0, hdr.index(col)).background().color().name(), '#00ff00', col)


if __name__ == '__main__':
    unittest.main()
