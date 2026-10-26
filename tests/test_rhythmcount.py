"""RhythmCount (Module IV).

Regression guards for the 2026-10-19 manual audit, group G2:
  B-6  amplitude, MESOR and peaks were always evaluated over 0-24 h, whatever
       the period;
  B-7  the default Nelder-Mead optimizer stopped at a worse optimum for N >= 2
       while reporting convergence;
  B-16 Selection Test = "F" always failed with a division by zero;
  B-19 the ZIP/ZINB likelihood-ratio p-value used 2N degrees of freedom while
       the null model differs by 4N parameters;
  plus Clean Data skipping non-integer time points, Parameter CIs using a fixed
  reference peak (Period/2) and dropping the peak intervals, and Compare Groups
  ignoring the conditions selected in the GUI.
"""
import json
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from tests.helpers import PROJECT_ROOT

warnings.filterwarnings('ignore')


def _poisson_series(period, acrophase, mesor=20.0, amplitude=12.0, n_cycles=3, reps=3, seed=1):
    rng = np.random.default_rng(seed)
    X = np.repeat(np.arange(0.0, period * n_cycles, 1.0), reps)
    mu = mesor + amplitude * np.cos(2 * np.pi * (X - acrophase) / period)
    return pd.DataFrame({'X': X, 'Y': rng.poisson(mu)})


def _example(variable='poisson_circadian', condition='control'):
    d = pd.read_csv(PROJECT_ROOT / 'examples' / 'synthetic_rhythmcount_test_data.csv')
    d = d[d.condition == condition][['time', variable]]
    return d.rename(columns={'time': 'X', variable: 'Y'})


def _fit(df, n, model, period=24.0):
    from core.rhythmcount_analysis import RhythmCountParameters
    from core.vendor.rhythmcount import data_processing as dp
    p = RhythmCountParameters()
    return dp.fit_to_model(df, n, model, period, p.maxiter, p.maxfun, p.method, 0)


def _curve_over_one_period(results, period):
    """Amplitude, MESOR and peak of the fitted N = 1 Poisson curve on a fine grid."""
    b = results.params
    x = np.linspace(0, period, 4001)[:-1]
    y = np.exp(b[0] + b[1] * np.sin(2 * np.pi * x / period) + b[2] * np.cos(2 * np.pi * x / period))
    return (y.max() - y.min()) / 2, (y.max() + y.min()) / 2, x[np.argmax(y)]


class TestRhythmParamsUsePeriod(unittest.TestCase):
    """B-6: rhythm parameters are evaluated over one full period."""

    def test_period_36(self):
        df = _poisson_series(36, acrophase=30.0)
        res, row, _ = _fit(df, 1, 'poisson', period=36)
        amp, mesor, peak = _curve_over_one_period(res, 36)
        self.assertAlmostEqual(row['amplitude'], amp, delta=0.05)
        self.assertAlmostEqual(row['mesor'], mesor, delta=0.05)
        self.assertEqual(len(row['peaks']), 1)
        self.assertAlmostEqual(row['peaks'][0], peak, delta=0.2)

    def test_short_period_has_one_peak(self):
        for period, acro in ((12, 3.0), (8, 2.0)):
            df = _poisson_series(period, acrophase=acro, n_cycles=6)
            res, row, _ = _fit(df, 1, 'poisson', period=period)
            self.assertEqual(len(row['peaks']), 1, f'period {period}: {row["peaks"]}')
            self.assertAlmostEqual(row['peaks'][0], _curve_over_one_period(res, period)[2], delta=0.2)


class TestOptimizer(unittest.TestCase):
    """B-7: the default fit reaches the maximum likelihood for N >= 2."""

    def test_poisson_matches_newton(self):
        # the Poisson log-likelihood is concave, so Newton's optimum is the global one
        from core.vendor.rhythmcount import data_processing as dp
        df = _example()
        for n in (2, 3):
            res, _, _ = _fit(df, n, 'poisson')
            newton, _, _ = dp.fit_to_model(df, n, 'poisson', 24, 5000, 5000, 'newton', 0)
            self.assertAlmostEqual(res.llf, newton.llf, delta=1e-3)

    def test_not_worse_than_any_single_optimizer(self):
        from core.vendor.rhythmcount import data_processing as dp
        df = _example('zero_inflated_circadian')
        for model in ('zero_poisson', 'nb', 'zero_nb'):
            res, _, _ = _fit(df, 3, model)
            for method in ('nm', 'bfgs'):
                single, _, _ = dp.fit_to_model(df, 3, model, 24, 5000, 5000, method, 0)
                if np.isfinite(single.llf):
                    self.assertGreaterEqual(res.llf, single.llf - 1e-3, f'{model} vs {method}')


class TestSelectionTestF(unittest.TestCase):
    """B-16: 'F' no longer crashes with a division by zero."""

    def test_f_selects_n_within_one_distribution(self):
        from core.rhythmcount_analysis import (RhythmCountAnalyzer, RhythmCountParameters,
                                               SelectionTest, CountModel)
        a = RhythmCountAnalyzer()
        a.load_data(_example())
        r = a.fit_best_model(RhythmCountParameters(selection_test=SelectionTest.F, n_components=[1, 2],
                                                   count_models=[CountModel.POISSON]))
        self.assertTrue(r.success, r.error)
        self.assertEqual(r.count_model, 'poisson')

    def test_f_between_distributions_gives_a_clear_error(self):
        from core.rhythmcount_analysis import (RhythmCountAnalyzer, RhythmCountParameters,
                                               SelectionTest, CountModel)
        a = RhythmCountAnalyzer()
        a.load_data(_example())
        r = a.fit_best_model(RhythmCountParameters(selection_test=SelectionTest.F, n_components=[1],
                                                   count_models=[CountModel.POISSON,
                                                                 CountModel.NEGATIVE_BINOMIAL]))
        self.assertFalse(r.success)
        self.assertIn('F test', r.error)
        self.assertNotIn('division by zero', r.error)


class TestZeroInflatedLRT(unittest.TestCase):
    """B-19: ZIP/ZINB p (LRT) uses the 4N parameters the null model drops."""

    def test_degrees_of_freedom(self):
        df = _example('zero_inflated_circadian')
        for model, k_null in (('zero_poisson', 2), ('zero_nb', 3)):
            for n in (1, 2):
                res, row, _ = _fit(df, n, model)
                self.assertEqual(len(res.params) - k_null, 4 * n)
                expected = stats.chi2.sf(res.llr, 4 * n)
                self.assertTrue(np.isclose(row['llr_pvalue'], expected, rtol=1e-9, atol=0),
                                f'{model} N={n}: {row["llr_pvalue"]} vs {expected}')

    def test_non_inflated_models_unchanged(self):
        df = _example('negbinom_circadian')
        for model in ('poisson', 'nb', 'gen_poisson'):
            res, row, _ = _fit(df, 2, model)
            self.assertEqual(row['llr_pvalue'], res.llr_pvalue)


class TestCleanData(unittest.TestCase):
    """Clean Data filters every sampled time point, not only integer hours."""

    def test_half_hour_time_points_are_filtered(self):
        from core.vendor.rhythmcount import data_processing as dp
        base = _example()
        mixed = pd.concat([base, base.assign(X=base.X + 0.5)], ignore_index=True)
        out = dp.clean_data(mixed.copy())
        n_int = int((out.X % 1 == 0).sum())
        n_half = int((out.X % 1 != 0).sum())
        self.assertLess(n_int, int((mixed.X % 1 == 0).sum()))
        self.assertEqual(n_half, n_int)


class TestDesignWarning(unittest.TestCase):
    """2-h sampling cannot resolve the 4-h harmonic (period 12 h, N = 3)."""

    def test_unresolved_harmonic_is_flagged(self):
        from core.rhythmcount_analysis import RhythmCountAnalyzer
        a = RhythmCountAnalyzer()
        a.load_data(_example())
        self.assertEqual(a.design_warning(2, 12.0), '')
        self.assertEqual(a.design_warning(3, 24.0), '')
        self.assertIn('not resolved', a.design_warning(3, 12.0))


class TestCompareGroupsAPI(unittest.TestCase):
    """compare_groups(plot_comparison=True): both evaluation orders and 2 groups."""

    def test_both_evaluation_orders(self):
        import os
        import tempfile
        import matplotlib
        matplotlib.use('Agg')
        from core.rhythmcount_analysis import RhythmCountAnalyzer, RhythmCountParameters, CountModel
        d = pd.read_csv(PROJECT_ROOT / 'examples' / 'synthetic_rhythmcount_test_data.csv')
        d = d[['time', 'condition', 'poisson_circadian']].rename(columns={'time': 'X', 'poisson_circadian': 'Y'})
        a = RhythmCountAnalyzer()
        a.load_data(d)
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:   # the vendored plot is saved to ./results
            os.chdir(tmp)
            try:
                for order in (True, False):
                    r = a.compare_groups('condition', plot_comparison=True, params=RhythmCountParameters(
                        eval_order=order, n_components=[1], repetitions=3,
                        count_models=[CountModel.POISSON, CountModel.NEGATIVE_BINOMIAL]))
                    self.assertTrue(r.success, f'eval_order={order}: {r.error}')
                    self.assertEqual(sorted(r.results_table['condition']), ['control', 'treatment'])
            finally:
                import matplotlib.pyplot as plt
                plt.close('all')
                os.chdir(cwd)

    def test_confidence_band_sampling_stays_in_range(self):
        # G8: the vendored plot drew random.randint(0, size), which includes size, so entry_at(size)
        # raised an empty IndexError in about 1 of 6 compare_groups(plot_comparison=True) calls
        from unittest import mock
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from core.vendor.rhythmcount import plot
        fig, ax = plt.subplots()
        try:
            with mock.patch.object(plot.random, 'randint', side_effect=lambda a, b: b):
                plot.subplot_confidence_intervals(_example(), 1, 'poisson', ax, repetitions=3)
        finally:
            plt.close(fig)


class TestEngine(unittest.TestCase):
    """Parameter CIs report peak intervals; Compare Groups honours the selection."""

    @classmethod
    def setUpClass(cls):
        from core.analysis_engine import AnalysisEngine
        cls.engine = AnalysisEngine()
        cls.data = pd.read_csv(PROJECT_ROOT / 'examples' / 'synthetic_rhythmcount_test_data.csv')

    def test_parameter_cis_use_the_fitted_peak(self):
        from core.analysis_engine import AnalysisType
        np.random.seed(0)
        r = self.engine.run_analysis(
            self.data, 'poisson_circadian', 'control', AnalysisType.RHYTHMCOUNT_PARAMETER_CIS,
            time_col='time', condition_col='condition',
            parameters={'period': 24.0, 'rc_single_count_model': 'poisson',
                        'rc_single_n_components_list': [1], 'rc_repetitions': 10})
        self.assertTrue(r.success, r.message)
        _, full, _ = _fit(_example(), 1, 'poisson')
        self.assertEqual(r.peak_times, list(full['peaks']))
        self.assertEqual(len(r.peak_cis), 1)
        lo, hi = r.peak_cis[0]
        self.assertLessEqual(lo, full['peaks'][0])
        self.assertGreaterEqual(hi, full['peaks'][0])
        self.assertEqual(len(r.peak_height_cis), 1)

    def test_compare_groups_uses_selected_conditions(self):
        from core.analysis_engine import AnalysisType
        three = pd.concat([self.data, self.data[self.data.condition == 'control'].assign(condition='extra')])
        params = {'period': 24.0, 'rc_count_models': ['poisson'], 'rc_n_components': [1],
                  'rc_repetitions': 3}
        r = self.engine.run_analysis(
            three, 'poisson_circadian', 'control', AnalysisType.RHYTHMCOUNT_COMPARE_GROUPS,
            time_col='time', condition_col='condition',
            parameters={**params, 'selected_conditions': ['control', 'treatment']})
        groups = [row['condition'] for row in json.loads(r.results_table_json)]
        self.assertEqual(sorted(groups), ['control', 'treatment'])
        r = self.engine.run_analysis(
            three, 'poisson_circadian', 'control', AnalysisType.RHYTHMCOUNT_COMPARE_GROUPS,
            time_col='time', condition_col='condition', parameters=params)
        self.assertEqual(len(json.loads(r.results_table_json)), 3)


if __name__ == '__main__':
    unittest.main()
