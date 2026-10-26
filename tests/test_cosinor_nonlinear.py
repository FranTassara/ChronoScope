"""CosinorPy nonlinear (generalized cosinor) methods, Module II 2.6-2.9.

Regression guards for the 2026-10-19 manual audit, group G3:
  B-3  Nonlinear Compare (Independent), Independent Models: CosinorPy 3.1's
       fit_generalized_cosinor_compare_pairs_independent fits every pair at
       24 h (period1=24, period2=24 hard-coded), whatever periods are requested;
  B-4  best-period selection used `p or inf`, so the periods whose F-test p
       underflowed to 0.0 ranked last instead of first;
  B-12 the joint-model F-test p of the Pooled Model (model vs. constant mean)
       was stored in p1/q1 and shown as p-Cond1/q-Cond1;
  plus the GUI Bootstrap Size minimum (5 gave NaN intervals).
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
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _two_conditions(spec, step=1.0, hours=88, reps=2, noise=0.6, seed=11):
    """spec: {condition: (period, peak_time)}; amplitude 3, MESOR 10, independent noise."""
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(0.0, hours, step), reps)
    return pd.concat([
        pd.DataFrame({'time': t, 'condition': c,
                      'y': 10 + 3 * np.cos(2 * np.pi * (t - pk) / per) + rng.normal(0, noise, t.size)})
        for c, (per, pk) in spec.items()], ignore_index=True)


def _engine():
    from core.analysis_engine import AnalysisEngine
    return AnalysisEngine()


def _compare(d, **params):
    from core.analysis_engine import AnalysisType
    params = {'n_components': [1], **params}
    return _quiet(_engine().run_analysis, d, 'y', d.condition.iloc[0],
                  AnalysisType.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT, parameters=params)


class TestCompareIndependentPeriods(unittest.TestCase):
    """B-3: the requested per-condition periods reach the fits."""

    def _pairs(self, d, p1, p2):
        from CosinorPy import cosinor_nonlin
        from core.cosinor_analysis import CosinorAnalyzer
        df = _quiet(_engine()._convert_to_cosinorpy_format_all_conditions, d, 'y', 'time', 'condition')
        pairs = [('y-A', 'y-B')]
        lib = _quiet(cosinor_nonlin.fit_generalized_cosinor_compare_pairs_independent, df, pairs,
                     period1=p1, period2=p2)
        ours = _quiet(CosinorAnalyzer._fit_generalized_cosinor_compare_pairs_independent, df, pairs,
                      period1=p1, period2=p2)
        return lib.iloc[0], ours.iloc[0]

    def test_same_as_library_at_24h(self):
        lib, ours = self._pairs(_two_conditions({'A': (24, 4), 'B': (24, 7)}), 24, 24)
        for col in ('d_amplitude', 'p(d_amplitude)', 'd_acrophase', 'd_amplification', 'd_lin_comp',
                    'q(d_amplitude)', 'q(d_acrophase)'):
            self.assertAlmostEqual(lib[col], ours[col], places=10, msg=col)

    def test_period_22_phase_delay(self):
        # B peaks 3 h later; model cos(2*pi*t/P + phi), so a delay gives d_acrophase < 0
        _, ours = self._pairs(_two_conditions({'A': (22, 4), 'B': (22, 7)}), 22, 22)
        self.assertAlmostEqual(ours['d_acrophase'], -2 * np.pi * 3 / 22, delta=0.1)

    def test_different_periods(self):
        d = _two_conditions({'A': (20, 4), 'B': (28, 4)})
        lib, ours = self._pairs(d, 20, 28)
        self.assertLess(abs(ours['d_amplitude']), 0.5)        # equal amplitudes
        self.assertGreater(ours['p(d_amplitude)'], 0.05)
        self.assertGreater(abs(lib['d_amplitude']), 2.0)       # library: both fitted at 24 h
        _, at24 = self._pairs(d, 24, 24)
        self.assertAlmostEqual(lib['d_amplitude'], at24['d_amplitude'], places=10)

    def test_engine_reports_both_periods(self):
        d = _two_conditions({'A': (20, 4), 'B': (28, 4)})
        r = _compare(d, period_range=(20.0, 20.0), period1=20.0, period2=28.0, use_dependent_model=False)[0]
        self.assertEqual((r.period, r.period2), (20.0, 28.0))
        self.assertGreater(r.p_amplitude, 0.05)
        same = _compare(d, period_range=(24.0, 24.0), period1=24.0, period2=24.0, use_dependent_model=False)[0]
        self.assertIsNone(same.period2)


class TestBestPeriodTies(unittest.TestCase):
    """B-4: periods whose p underflows to 0 are the best, ties broken by the F-test in log space."""

    def _mark(self, ps, rss, n_comp=None):
        from core.cosinor_analysis import CosinorAnalyzer
        res = [{'period': 20 + i, 'p_value': p, 'rss': s, 'n_components': (n_comp or [1] * len(ps))[i]}
               for i, (p, s) in enumerate(zip(ps, rss))]
        y = pd.DataFrame({'y': np.random.default_rng(0).normal(10, 3, 100)})
        CosinorAnalyzer()._mark_best_period(res, y)
        return [r['period'] for r in res if r['best_model'].startswith('Yes')]

    def test_zero_p_wins(self):
        self.assertEqual(self._mark([1e-12, 0.0, 3e-9], [9.0, 8.0, 7.0]), [21])

    def test_ties_by_rss(self):
        self.assertEqual(self._mark([0.0, 0.0, 0.0, 1e-5], [50.0, 20.0, 30.0, 10.0]), [21])

    def test_ties_account_for_parameters(self):
        # same RSS, but 4 components spend 6 more parameters than 1 -> lower F, larger p
        self.assertEqual(self._mark([0.0, 0.0], [20.0, 20.0], n_comp=[4, 1]), [21])

    def test_missing_p(self):
        self.assertEqual(self._mark([None, 0.2, np.nan], [1.0, 2.0, 3.0]), [21])

    def test_independent_scan_period_26(self):
        from core.analysis_engine import AnalysisType
        d = _two_conditions({'A': (26, 5)}, hours=120, noise=0.8, seed=7)
        res = _quiet(_engine().run_analysis, d, 'y', 'A', AnalysisType.COSINORPY_NONLINEAR_INDEPENDENT,
                     parameters={'period_range': (22.0, 30.0), 'period_step': 1.0, 'n_components': [1]})
        self.assertGreater(sum(r.p_value == 0 for r in res), 1)   # the tie this guards against
        self.assertEqual([r.period for r in res if r.best_model.startswith('Yes')], [26.0])

    def test_dependent_example_24h(self):
        from core.analysis_engine import AnalysisType
        d = pd.read_csv(PROJECT_ROOT / 'examples' / 'synthetic_rhythms_dependent_test_data.csv')
        res = _quiet(_engine().run_analysis, d, 'circadian_pure', 'control', AnalysisType.COSINORPY_NONLINEAR_DEPENDENT,
                     parameters={'period_range': (20.0, 28.0), 'period_step': 1.0, 'n_components': [1]})
        self.assertEqual([r.period for r in res if r.best_model.startswith('Yes')], [24.0])


class TestJointModelP(unittest.TestCase):
    """B-12: the joint-model F-test p is not a per-condition p."""

    def test_not_in_p_cond1(self):
        d = _two_conditions({'A': (24, 6)}, seed=3)
        d = pd.concat([d, d.assign(condition='B')], ignore_index=True)   # identical conditions
        r = _compare(d, period_range=(24.0, 24.0), use_dependent_model=True)[0]
        self.assertIsNone(r.p1)
        self.assertIsNone(r.q1)
        self.assertLess(r.model_p, 1e-6)          # the model fits ...
        self.assertGreater(r.p_amplitude, 0.5)    # ... but nothing differs

    def test_results_table_columns(self):
        from PySide6.QtWidgets import QApplication
        from ui.results_panel import ResultsPanel
        app = QApplication.instance() or QApplication([])  # noqa: F841
        d = _two_conditions({'A': (20, 4), 'B': (28, 4)})
        joint = _compare(d, period_range=(24.0, 24.0), use_dependent_model=True)[0]
        indep = _compare(d, period_range=(20.0, 20.0), period1=20.0, period2=28.0, use_dependent_model=False)[0]
        p = ResultsPanel()
        for res, present, absent in ((joint, 'p-Model (joint F)', 'Period-2 (h)'),
                                     (indep, 'Period-2 (h)', 'p-Model (joint F)')):
            p.set_results([res.to_dict()])
            t = p._results_table
            headers = [t.horizontalHeaderItem(j).text() for j in range(t.columnCount())]
            self.assertIn(present, headers)
            self.assertNotIn(absent, headers)


class TestBootstrapSizeMinimum(unittest.TestCase):
    """GUI Bootstrap Size: the nonlinear bootstrap needs B > 2N + 3, and is unstable below ~50."""

    def test_spinbox_minimum(self):
        from PySide6.QtWidgets import QApplication
        from ui.analysis_panel import AnalysisPanel
        app = QApplication.instance() or QApplication([])  # noqa: F841
        self.assertEqual(AnalysisPanel()._bootstrap_size_spin.minimum(), 50)

    def test_minimum_gives_finite_intervals(self):
        from CosinorPy import cosinor_nonlin
        d = _two_conditions({'A': (24, 4)})
        np.random.seed(0)
        r = _quiet(cosinor_nonlin.eval_params_n_comp_bootstrap, d.time.values, d.y.values,
                   n_components=4, period=24, bootstrap_size=50)
        self.assertTrue(np.all(np.isfinite(r['CI(amplitude)'])))


if __name__ == '__main__':
    unittest.main()
