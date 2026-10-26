"""Module I methods 1.1-1.5: JTK, AR-JTK, Cosine-Kendall, Cosinor (OLS), Harmonic.

Regression guards for the 2026-10-23 audit of the User Manual, sections 1.1-1.5 (G9):
  A7  the Cosinor amplitude CI (delta method) swapped Var(β_cos) and Var(β_sin):
      with unbalanced sampling the 95 % CI covered the true amplitude ~48 % of
      the time; the acrophase CI was not wrapped to [0, 2π) like the acrophase;
  A4  JTK reported lag + asym·P + P/2 as the acrophase when tau < 0; the data
      peak sits at the template trough, lag. Only right for asym = 0.5;
  A5  Harmonic Cosinor read the GUI's 'period_range' = (min, max) tuple as the
      list of periods to test: two periods, and Bonferroni = 2 × raw with one;
  A6  the acrophase of harmonic k was φ·P/2π instead of φ·P/(2πk); a peak at
      the cycle end read 24.0 instead of 0; components were labelled H1, H2 by
      amplitude rank, not by harmonic number;
  M4  with a period range, every period whose p rounded to 0 was "Best Fit".
User decisions (2026-10-23):
  A3  AR-JTK pre-whitens the rank series with the AR fitted to the residuals of
      a regression on the best template, and no longer adds the template back
      (null false-positive rate 17-18 % white noise, 100 % AR(1) 0.7, before);
  A2  with a period range, the AR-JTK rows use that pre-whitened series;
  A1  Empirical (permutation) p-values also with a period range (one null over
      the whole range; permutations vectorised);
  A5  Harmonic takes a period range in the GUI, like Cosinor (OLS);
  A6  harmonics reported in harmonic order, plus the peaks, troughs and
      amplitude of the fitted waveform.
"""
import io
import contextlib
import unittest
import warnings

import numpy as np
import pandas as pd

from tests.helpers import PROJECT_ROOT  # noqa: F401  (puts the project on sys.path)

warnings.filterwarnings('ignore')


def _engine_run(times, values, analysis_type, parameters):
    from core.analysis_engine import AnalysisEngine, AnalysisType
    df = pd.DataFrame({'time': times, 'condition': 'c', 'y': values})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r = AnalysisEngine().run_analysis(df, 'y', 'c', getattr(AnalysisType, analysis_type),
                                          parameters=parameters)
    return r if isinstance(r, list) else [r]


def _unbalanced_times(rng):
    """Samples clustered near 6 h and 18 h: Var(β_cos) >> Var(β_sin)."""
    return np.sort(np.concatenate([rng.uniform(5, 7, 15), rng.uniform(17, 19, 15),
                                   rng.uniform(0, 24, 4)]))


def _sawtooth(times, peak, rise=4.8, period=24.0):
    """Fast rise (rise h) to a peak at `peak`, slow linear decay."""
    tm = (times - (peak - rise)) % period
    return np.where(tm <= rise, tm / rise, (period - tm) / (period - rise))


class TestCosinorConfidenceIntervals(unittest.TestCase):
    """A7: amplitude CI from the right delta-method gradient; wrapped acrophase CI."""

    def test_amplitude_se_matches_delta_method(self):
        import statsmodels.api as sm
        from core.rhythm_analysis import _fit_cosinor
        rng = np.random.default_rng(5)
        t = _unbalanced_times(rng)
        y = 5 + np.cos(2 * np.pi * t / 24) + rng.normal(0, 0.5, t.size)
        r = _fit_cosinor(t, y, period_range=[24.0])
        X = np.column_stack([np.ones_like(t), np.cos(2 * np.pi * t / 24), np.sin(2 * np.pi * t / 24)])
        m = sm.OLS(y, X).fit()
        bc, bs = m.params[1], m.params[2]
        cov = m.cov_params()
        se = np.sqrt((bc**2 * cov[1, 1] + bs**2 * cov[2, 2] + 2 * bc * bs * cov[1, 2]) / (bc**2 + bs**2))
        self.assertAlmostEqual((r.amplitude_ci[1] - r.amplitude_ci[0]) / (2 * 1.96), se, places=3)

    def test_amplitude_ci_coverage_with_unbalanced_sampling(self):
        from core.rhythm_analysis import _fit_cosinor
        rng = np.random.default_rng(20261023)
        t = _unbalanced_times(rng)
        hits = 0
        n = 400
        for _ in range(n):
            y = 5 + np.cos(2 * np.pi * (t - 0.5) / 24) + rng.normal(0, 0.5, t.size)
            lo, hi = _fit_cosinor(t, y, period_range=[24.0]).amplitude_ci
            hits += lo <= 1.0 <= hi
        # swapped variances: coverage ~0.48
        self.assertGreater(hits / n, 0.90)

    def test_acrophase_and_ci_are_wrapped(self):
        from core.rhythm_analysis import _fit_cosinor
        t = np.arange(0, 48, 2.0)
        y = 5 + np.cos(2 * np.pi * t / 24) + np.random.default_rng(0).normal(0, 0.3, t.size)
        r = _fit_cosinor(t, y, period_range=[24.0])             # peak at 0 h, CI half-width ~0.6 h
        lo, hi = r.acrophase_ci
        self.assertTrue(0 <= lo <= 2 * np.pi + 1e-4 and 0 <= hi <= 2 * np.pi + 1e-4, r.acrophase_ci)
        self.assertGreater(lo, hi)                              # the CI crosses 0
        y = 5 + np.cos(2 * np.pi * (t - 23.999) / 24)          # noiseless, peak just before 24 h
        r = _fit_cosinor(t, y, period_range=[24.0])
        self.assertLess(r.acrophase_hours, 24.0)
        self.assertAlmostEqual(r.acrophase_hours % 24.0, 0.0, places=2)


class TestJTKAcrophase(unittest.TestCase):
    """A4: with tau < 0 the acrophase is the lag, whatever the asymmetry."""

    def test_negative_tau_with_asymmetric_template(self):
        from core.rhythm_analysis import _run_discrete_jtk, _run_discrete_jtk_all_periods
        t = np.arange(0, 48, 1.0)
        for peak in (2.0, 9.0):
            y = _sawtooth(t, peak) + np.random.default_rng(int(peak)).normal(0, 0.05, t.size)
            s = pd.Series(y, index=t)
            r = _run_discrete_jtk(s, period_range=[24.0], asymmetries=[0.2, 0.5, 0.8])
            self.assertLess(r.tau, 0)                           # a mirrored asym-0.8 template wins
            self.assertNotEqual(r.asymmetry, 0.5)
            self.assertAlmostEqual(r.acrophase, peak, delta=0.5)
            r24 = [x for x in _run_discrete_jtk_all_periods(s, period_range=[23.0, 24.0, 25.0],
                                                            asymmetries=[0.2, 0.5, 0.8])
                   if x.period == 24.0][0]
            self.assertAlmostEqual(r24.acrophase, peak, delta=0.5)

    def test_ar_jtk_inherits_the_fix(self):
        t = np.arange(0, 48, 1.0)
        y = _sawtooth(t, 9.0) + np.random.default_rng(9).normal(0, 0.05, t.size)
        r = _engine_run(t, y, 'AR_JTK', {'period': 24.0, 'asymmetry_search': True, 'prewhiten': True})[0]
        self.assertAlmostEqual(r.acrophase_hours, 9.0, delta=0.5)

    def test_symmetric_template_unchanged(self):
        """asym = 0.5: lag + P/2 + P/2 = lag, so the old and new formulas agree."""
        from core.rhythm_analysis import _run_discrete_jtk
        t = np.arange(0, 48, 2.0)
        y = np.cos(2 * np.pi * (t - 7) / 24) + np.random.default_rng(1).normal(0, 0.2, t.size)
        r = _run_discrete_jtk(pd.Series(y, index=t), period_range=[24.0])
        self.assertAlmostEqual(r.acrophase, 7.0, delta=1.0)


class TestBestFitSingleRow(unittest.TestCase):
    """M4: one Best Fit row per period sweep, the period a single run picks."""

    def test_one_best_row(self):
        from core.rhythm_analysis import _run_discrete_jtk
        t = np.arange(0, 96, 1.0)
        y = 10 + 3 * np.cos(2 * np.pi * (t - 6) / 24) + np.random.default_rng(11).normal(0, 0.5, t.size)
        periods = [22.0, 23.0, 24.0, 25.0, 26.0]
        single = _run_discrete_jtk(pd.Series(y, index=t), period_range=periods)
        for atype, extra in (('JTK', {}), ('AR_JTK', {'prewhiten': True}),
                             ('COSINE_KENDALL', {'resolution': 1.0})):
            with self.subTest(method=atype):
                rows = _engine_run(t, y, atype, {'period': periods, **extra})
                self.assertTrue(all(r.raw_p_value == 0 for r in rows))   # all tied once rounded
                best = [r.period for r in rows if r.best_model == 'Yes']
                self.assertEqual(len(best), 1, best)
                if atype != 'COSINE_KENDALL':
                    self.assertEqual(best[0], single.period)


def _ar1(rng, phi, n=48, amp=0.0):
    z = rng.normal(size=n + 50)
    e = np.zeros(n + 50)
    for i in range(1, n + 50):
        e[i] = phi * e[i - 1] + z[i]
    t = np.arange(n, dtype=float)
    return t, amp * np.cos(2 * np.pi * (t - 6) / 24) + e[50:]


class TestARJTKCalibration(unittest.TestCase):
    """A3: the pre-whitened series no longer contains the winning template."""

    def _rate(self, phi, amp, n_rep, seed, **kw):
        from core.rhythm_analysis import _run_ar_jtk
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_rep):
            t, y = _ar1(rng, phi, amp=amp)
            with contextlib.redirect_stderr(io.StringIO()):
                hits += _run_ar_jtk(pd.Series(y, index=t), period_range=[24.0], **kw)[0].bh_p_value < 0.05
        return hits / n_rep

    def test_null_white_noise(self):
        # template re-injection: 17-18 %
        self.assertLess(self._rate(0.0, 0.0, 150, 1, force_prewhiten=True), 0.08)

    def test_null_autocorrelated_noise(self):
        # template re-injection: 100 % (plain JTK: 28 %)
        self.assertLess(self._rate(0.7, 0.0, 150, 2, force_prewhiten=True), 0.16)
        self.assertLess(self._rate(0.7, 0.0, 150, 3), 0.18)

    def test_power_is_kept(self):
        # residuals r - rank(template) instead of a regression: power 0 %
        self.assertGreater(self._rate(0.0, 0.6, 100, 4, force_prewhiten=True), 0.25)

    def test_short_series_is_not_prewhitened(self):
        from core.rhythm_analysis import _run_ar_jtk
        t = np.arange(0, 20, 2.0)
        res, applied = _run_ar_jtk(pd.Series(np.cumsum(np.random.default_rng(5).normal(size=t.size)), index=t),
                                   period_range=[24.0])
        self.assertFalse(applied)


class TestARJTKPeriodRange(unittest.TestCase):
    """A2: with a period range the rows use the pre-whitened series too."""

    def test_range_rows_are_prewhitened(self):
        from core.rhythm_analysis import (_ar_prewhitened_ranks, _run_ar_jtk,
                                          _run_discrete_jtk_all_periods)
        t, y = _ar1(np.random.default_rng(6), 0.7, amp=1.0)
        periods = [23.0, 24.0, 25.0]
        rows = _engine_run(t, y, 'AR_JTK', {'period': periods, 'prewhiten': True})
        _, r_pw = _ar_prewhitened_ranks(pd.Series(y, index=t), periods, None, [0.5], force_prewhiten=True)
        expected = _run_discrete_jtk_all_periods(r_pw, period_range=periods)
        self.assertEqual([r.raw_p_value for r in rows], [e.p_value for e in expected])
        self.assertTrue(all(r.message == 'AR correction applied' for r in rows))
        amp = (np.percentile(y, 90) - np.percentile(y, 10)) / 2
        self.assertTrue(all(abs(r.amplitude - amp) < 1e-4 for r in rows))      # was (max - min) / 2
        single, _ = _run_ar_jtk(pd.Series(y, index=t), period_range=periods, force_prewhiten=True)
        self.assertEqual(min(r.raw_p_value for r in rows), single.p_value)
        self.assertEqual([r.period for r in rows if r.best_model == 'Yes'], [single.period])


class TestEmpiricalPWithRange(unittest.TestCase):
    """A1: Empirical (permutation) p-values with a period range."""

    t = np.arange(0, 48, 2.0)
    y = 10 + 0.8 * np.cos(2 * np.pi * (t - 6) / 24) + np.random.default_rng(12).normal(0, 1.0, t.size)

    def test_vectorised_null_matches_scipy(self):
        from scipy.stats import kendalltau
        from core.rhythm_analysis import _generate_triangle_template_time, _permutation_max_abs_tau
        y = pd.Series(self.y).rank().values
        tmpl = np.array([_generate_triangle_template_time(self.t, p, lag, a)
                         for p in (22.0, 24.0) for a in (0.3, 0.5) for lag in np.arange(0, p, 1)])
        null = _permutation_max_abs_tau(y, tmpl, 40, seed=3)
        rng = np.random.default_rng(3)
        perms = [rng.permutation(y) for _ in range(40)]
        ref = [max(abs(kendalltau(py, tm)[0]) for tm in tmpl) for py in perms]
        np.testing.assert_allclose(null, ref, rtol=0, atol=1e-12)

    def test_jtk_rows_are_empirical(self):
        from core.rhythm_analysis import _run_discrete_jtk
        periods = [22.0, 23.0, 24.0, 25.0, 26.0]
        rows = _engine_run(self.t, self.y, 'JTK', {'period': periods, 'pvalue_method': 'empirical',
                                                    'jtk_n_permutations': 199})
        single = _run_discrete_jtk(pd.Series(self.y, index=self.t), period_range=periods, n_permutations=199)
        best = [r for r in rows if r.best_model == 'Yes']
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0].raw_p_value, single.p_value)                 # same null, same template
        for r in rows:
            self.assertEqual(r.raw_p_value, r.p_value)
            self.assertEqual(r.raw_p_value, r.bonf_p_value)
            self.assertGreaterEqual(r.raw_p_value, 1 / 200)
        analytical = _engine_run(self.t, self.y, 'JTK', {'period': periods})
        self.assertNotEqual([r.raw_p_value for r in rows], [r.raw_p_value for r in analytical])

    def test_cosine_kendall_rows_are_empirical(self):
        from core.rhythm_analysis import _run_cosine_kendall
        periods = [23.0, 24.0, 25.0]
        rows = _engine_run(self.t, self.y, 'COSINE_KENDALL',
                           {'period': periods, 'resolution': 1.0, 'pvalue_method': 'empirical',
                            'jtk_n_permutations': 199})
        single = _run_cosine_kendall(pd.Series(self.y, index=self.t), period_range=periods, interval=1.0,
                                     n_permutations=199)
        best = [r for r in rows if r.best_model == 'Yes']
        self.assertEqual(best[0].raw_p_value, single.p_value)
        self.assertTrue(all(r.raw_p_value == r.bonf_p_value for r in rows))


class TestHarmonicCosinor(unittest.TestCase):
    """A5 period grid from the GUI parameters; A6 acrophases and labels."""

    def setUp(self):
        from core import rhythm_analysis as ra
        self.ra = ra
        self.orig = ra._fit_harmonic_cosinor
        self.seen = []

        def spy(times, values, period_range=None, n_harmonics=2):
            self.seen.append(list(period_range))
            return self.orig(times, values, period_range=period_range, n_harmonics=n_harmonics)
        ra._fit_harmonic_cosinor = spy

    def tearDown(self):
        self.ra._fit_harmonic_cosinor = self.orig

    def test_gui_single_period_tests_one_period(self):
        t = np.arange(0, 48, 2.0)
        y = 10 + 0.6 * np.cos(2 * np.pi * (t - 6) / 24) + np.random.default_rng(7).normal(0, 1.0, t.size)
        # what the GUI sends in single-period mode (the max box is hidden)
        r = _engine_run(t, y, 'HARMONIC_COSINOR',
                        {'n_harmonics': 1, 'period': 24.0, 'period_range': (24.0, 28.0)})[0]
        self.assertEqual(self.seen, [[24.0]])
        self.assertEqual(r.period, 24.0)
        self.assertEqual(r.bonf_p_value, r.p_value)

    def test_period_list_is_scanned(self):
        t = np.arange(0, 48, 2.0)
        y = 10 + np.cos(2 * np.pi * t / 24) + np.random.default_rng(8).normal(0, 0.3, t.size)
        _engine_run(t, y, 'HARMONIC_COSINOR', {'n_harmonics': 1, 'period': [22.0, 24.0, 26.0]})
        self.assertEqual(self.seen, [[22.0, 24.0, 26.0]])

    def test_harmonic_acrophases(self):
        t = np.arange(0, 48, 1.0)
        # H1: amplitude 1, peak at 0 h; H2: amplitude 2, peaks at 3 h and 15 h
        y = 5 + np.cos(2 * np.pi * t / 24) + 2 * np.cos(2 * 2 * np.pi * (t - 3) / 24)
        h = self.orig(t, y, period_range=[24.0], n_harmonics=2)
        # harmonic order (user decision 2026-10-23), no longer sorted by amplitude
        self.assertEqual(h.harmonic_numbers, [1, 2])
        self.assertEqual(h.amplitudes, [1.0, 2.0])
        self.assertAlmostEqual(h.acrophases[0], 0.0, places=2)     # was 24.0
        self.assertAlmostEqual(h.acrophases[1], 3.0, places=2)     # was 6.0
        r = _engine_run(t, y, 'HARMONIC_COSINOR', {'n_harmonics': 2, 'period': 24.0})[0]
        self.assertEqual((r.amplitude, r.acrophase_hours), (1.0, 0.0))
        self.assertIn('H1: A=1.0000 φ=0.00h, H2: A=2.0000 φ=3.00h', r.message)

    def test_fitted_waveform_peaks(self):
        t = np.arange(0, 48, 1.0)
        y = 5 + np.cos(2 * np.pi * t / 24) + 2 * np.cos(2 * 2 * np.pi * (t - 3) / 24)
        tg = np.linspace(0, 24, 240001)
        yg = np.cos(2 * np.pi * tg / 24) + 2 * np.cos(2 * 2 * np.pi * (tg - 3) / 24)
        h = self.orig(t, y, period_range=[24.0], n_harmonics=2)
        self.assertEqual(len(h.waveform_peaks), 2)                 # bimodal
        self.assertAlmostEqual(h.waveform_peaks[0], tg[np.argmax(yg)], delta=0.02)   # highest first
        self.assertEqual(len(h.waveform_troughs), 2)
        self.assertAlmostEqual(h.waveform_amplitude, (yg.max() - yg.min()) / 2, places=3)
        r = _engine_run(t, y, 'HARMONIC_COSINOR', {'n_harmonics': 2, 'period': 24.0})[0]
        self.assertEqual(r.peak_times, h.waveform_peaks)
        self.assertEqual(r.harmonic_amplitudes, [1.0, 2.0])
        self.assertEqual(r.waveform_amplitude, h.waveform_amplitude)

    def test_single_harmonic_waveform_peak_is_the_acrophase(self):
        t = np.arange(0, 48, 2.0)
        y = 10 + 2 * np.cos(2 * np.pi * (t - 7) / 24)
        h = self.orig(t, y, period_range=[24.0], n_harmonics=1)
        self.assertEqual(len(h.waveform_peaks), 1)
        self.assertAlmostEqual(h.waveform_peaks[0], h.acrophases[0], delta=0.01)
        self.assertAlmostEqual(h.waveform_amplitude, 2.0, places=3)

    def test_gui_uses_a_period_range(self):
        """A5 decision: Harmonic takes Min / Max / Step like Cosinor (OLS)."""
        import os
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisPanel
        p = AnalysisPanel()
        p._source_type = 'csv'
        p._update_module_combo_for_source('csv')
        p._module_combo.setCurrentText('Classical Rhythm Analysis')
        p._method_combo.setCurrentText('Harmonic Cosinor')
        self.assertEqual(p._period_mode, 'range')
        self.assertFalse(p._period_step_spin.isHidden())
        p._period_min_spin.setValue(22.0)
        p._period_max_spin.setValue(26.0)
        p._period_step_spin.setValue(1.0)
        self.assertEqual(p._get_current_parameters()['period'], [22.0, 23.0, 24.0, 25.0, 26.0])

    def test_fit_statistics_unchanged(self):
        """The CRS-AI reads p_value and r_squared: A6 must not touch them."""
        t = np.arange(0, 48, 2.0)
        y = 5 + np.cos(2 * np.pi * (t - 4) / 24) + 0.5 * np.cos(4 * np.pi * (t - 2) / 24) \
            + np.random.default_rng(4).normal(0, 0.3, t.size)
        h = self.orig(t, y, period_range=[24.0], n_harmonics=2)
        X = np.column_stack([np.ones_like(t)] + [f(k * 2 * np.pi * t / 24) for k in (1, 2)
                                                 for f in (np.cos, np.sin)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        self.assertAlmostEqual(h.fit_model['r_squared'], r2, places=10)


if __name__ == '__main__':
    unittest.main()
