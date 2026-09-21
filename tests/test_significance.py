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


if __name__ == '__main__':
    unittest.main()
