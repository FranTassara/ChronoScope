"""Period-scan corrections and how significance is decided.

Cosinor OLS and Harmonic Cosinor choose their period by scanning a range and
keeping the best fit, so the F-test p-value at the winning period ignores how
many periods were tried. Both must expose the raw value *and* its Bonferroni
correction, and the significance call must be made on the corrected one.
"""
import unittest

import numpy as np

from tests.helpers import PROJECT_ROOT, VARIABLE, CONDITION, rhythmic_dataframe


def _run(analysis_type, df, parameters=None):
    from core.analysis_engine import AnalysisEngine, AnalysisType
    engine = AnalysisEngine()
    result = engine.run_analysis(
        data=df, variable=VARIABLE, condition=CONDITION,
        analysis_type=getattr(AnalysisType, analysis_type),
        time_col='time', condition_col='condition', parameters=parameters or {},
    )
    return result[0] if isinstance(result, list) else result


class TestPeriodScanCorrection(unittest.TestCase):
    """Both scanning cosinor variants report raw and scan-corrected p-values."""

    @classmethod
    def setUpClass(cls):
        from core.rhythm_analysis import DefaultPeriodRanges
        cls.n_periods = len(DefaultPeriodRanges.CIRCADIAN)
        cls.df = rhythmic_dataframe(amplitude=0.6, noise_sd=2.0, seed=31)

    def _assert_bonferroni(self, result, method):
        self.assertTrue(result.success, f'{method} failed: {result.message}')
        self.assertIsNotNone(result.p_value, f'{method} has no raw p-value')
        self.assertIsNotNone(result.bonf_p_value,
                             f'{method} does not report a scan-corrected p-value')
        expected = min(1.0, result.p_value * self.n_periods)
        # The correction is applied before the raw value is rounded for display,
        # so allow the rounding slack of that last digit.
        self.assertAlmostEqual(result.bonf_p_value, expected,
                               delta=self.n_periods * 1e-6,
                               msg=f'{method}: bonf_p_value is not raw p x n_periods')
        self.assertGreaterEqual(result.bonf_p_value, result.p_value)

    def test_cosinor_ols_reports_both(self):
        self._assert_bonferroni(_run('COSINOR_OLS', self.df), 'cosinor_ols')

    def test_harmonic_cosinor_reports_both(self):
        self._assert_bonferroni(_run('HARMONIC_COSINOR', self.df), 'harmonic_cosinor')

    def test_harmonic_adj_p_value_is_actually_corrected(self):
        """Regression guard: HarmonicCosinorResult.adj_p_value once held the
        uncorrected value despite its name and docstring."""
        from core.rhythm_analysis import _fit_harmonic_cosinor, DefaultPeriodRanges
        times = self.df['time'].to_numpy()[: len(self.df) // 3]
        values = self.df[VARIABLE].to_numpy()[: len(self.df) // 3]
        result = _fit_harmonic_cosinor(times, values,
                                       period_range=DefaultPeriodRanges.CIRCADIAN,
                                       n_harmonics=2)
        self.assertIsNotNone(result)
        if 0.0 < result.p_value < 1.0 / self.n_periods:
            self.assertGreater(result.adj_p_value, result.p_value,
                               'adj_p_value is not corrected for the period scan')


class TestSignificanceSelection(unittest.TestCase):
    """The UI decides significance on the corrected p for scanning methods."""

    @staticmethod
    def _pick(result):
        from ui.results_panel import ResultsPanel
        return ResultsPanel._significance_p(result)

    def test_scanning_methods_use_the_corrected_value(self):
        for method in ('cosinor_ols', 'harmonic_cosinor'):
            with self.subTest(method=method):
                picked = self._pick({'method': method, 'p_value': 0.01,
                                     'bonf_p_value': 0.17})
                self.assertEqual(picked, 0.17)

    def test_other_methods_keep_their_own_p_value(self):
        """JTK reports p_value already BH-adjusted and is well calibrated;
        forcing its Bonferroni column would only cost power."""
        picked = self._pick({'method': 'jtk', 'p_value': 0.01,
                             'bonf_p_value': 0.17})
        self.assertEqual(picked, 0.01)

    def test_falls_back_when_no_correction_is_available(self):
        picked = self._pick({'method': 'cosinor_ols', 'p_value': 0.02,
                             'bonf_p_value': None})
        self.assertEqual(picked, 0.02)

    def test_missing_p_value_is_not_counted(self):
        self.assertIsNone(self._pick({'method': 'cosinor_ols'}))


class TestChiSquarePeriodogramThreshold(unittest.TestCase):
    """The chi-square periodogram reports τ as its most significant period and
    calls rhythmicity with a scan-wide (block-permutation) threshold, not a
    flat single-period chi² line."""

    BIN = 0.5

    def _series(self, rng, n_days=7, amp=0.0, period=24.0):
        t = np.arange(int(n_days * 24 / self.BIN)) * self.BIN
        a = 10 + amp * np.maximum(0, np.sin(2 * np.pi * t / period)) + rng.normal(0, 2.0, len(t))
        return t, a

    def test_rhythmic_series_is_called_rhythmic_at_the_right_period(self):
        from core.visualization_circadian_metrics import chi_square_periodogram
        t, a = self._series(np.random.default_rng(5), amp=4.0, period=23.6)
        r = chi_square_periodogram(t, a, n_perm=199)
        self.assertTrue(r['rhythmic'])
        self.assertAlmostEqual(r['tau'], 23.6, delta=self.BIN)
        self.assertGreater(r['tau_qp'], r['significance'])

    def test_threshold_curve_rises_with_period(self):
        """Qp ~ chi²(n_slots - 1) and n_slots grows with P, so a flat line
        (the pre-fix behaviour) is anti-conservative at long periods."""
        from core.visualization_circadian_metrics import chi_square_periodogram
        t, a = self._series(np.random.default_rng(6))
        r = chi_square_periodogram(t, a, n_perm=199)
        curve = r['threshold_curve']
        self.assertTrue(np.all(np.diff(curve) >= 0))
        self.assertGreater(curve[-1], curve[0])

    def test_fixed_seed_is_reproducible(self):
        from core.visualization_circadian_metrics import chi_square_periodogram
        t, a = self._series(np.random.default_rng(7), amp=1.0)
        r1 = chi_square_periodogram(t, a, n_perm=199)
        r2 = chi_square_periodogram(t, a, n_perm=199)
        self.assertEqual(r1['global_p'], r2['global_p'])
        self.assertEqual(r1['p_crit'], r2['p_crit'])

    def test_n_perm_zero_skips_the_test(self):
        from core.visualization_circadian_metrics import chi_square_periodogram
        t, a = self._series(np.random.default_rng(8), amp=4.0)
        r = chi_square_periodogram(t, a, n_perm=0)
        self.assertIsNotNone(r['tau'])
        self.assertIsNone(r['significance'])
        self.assertIsNone(r['rhythmic'])

    def test_false_positive_rate_on_noise_is_near_alpha(self):
        """Regression guard: the pre-fix rule called ~97% of pure-noise
        recordings rhythmic. At alpha = 0.05, more than 12/100 has
        probability < 0.001 under correct calibration."""
        from core.visualization_circadian_metrics import chi_square_periodogram
        rng = np.random.default_rng(9)
        hits = sum(bool(chi_square_periodogram(*self._series(rng, n_days=5),
                                               n_perm=199, seed=i)['rhythmic'])
                   for i in range(100))
        self.assertLessEqual(hits, 12)


if __name__ == '__main__':
    unittest.main()
