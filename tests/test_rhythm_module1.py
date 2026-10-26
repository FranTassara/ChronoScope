"""Module I methods: LME, Fourier F24, Spectral Analysis, CWT.

Regression guards for the 2026-10-19 manual audit, group G5:
  B-2  the LME likelihood-ratio test compared REML log-likelihoods of models
       with different fixed effects (null FPR 0%, median p = 1);
  B-11 LME "R² (marginal)" used fittedvalues (random effects included), the
       AIC/BIC k counted the random-effect variance twice, and "Best Period"
       compared p-values rounded to 6 decimals (every period tied at 0);
  B-8  F24 with > 2 subjects split them with an unseeded RNG;
  B-17 the Refinetti threshold summed the DC bin (Detrend off) and, for
       Lomb-Scargle, the unnormalised power over the 1000-point grid;
  CWT  the period grid could overshoot the maximum analyzable period.
User decisions (2026-10-22): F24 correlation_r removed (always ±1); F24 adds
f24_norm and pF24 without changing f24_score (a CRS-AI feature); the spectral
threshold uses Fisher's m = (N-1)//2 frequencies, and Lomb-Scargle is always
mean-centred with its period grid starting at the Nyquist period.
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

LME_CSV = os.path.join(PROJECT_ROOT, 'examples', 'synthetic_lme_test_data.csv')


def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def _engine():
    from core.analysis_engine import AnalysisEngine
    return AnalysisEngine()


def _lme_data(rng, amp=0.0, n_subj=6, sd_between=1.5):
    t = np.tile(np.arange(0, 48, 2.0), n_subj)
    g = np.repeat([f's{i}' for i in range(n_subj)], 24)
    u = np.repeat(rng.normal(0, sd_between, n_subj), 24)
    y = 10 + u + amp * np.cos(2 * np.pi * (t - 4) / 24) + rng.normal(0, 1, t.size)
    return t, y, g


def _ml_fits(t, y, g, period=24.0):
    import statsmodels.formula.api as smf
    d = pd.DataFrame({'y': y, 'c': np.cos(2 * np.pi * t / period),
                      's': np.sin(2 * np.pi * t / period), 'g': g})
    full = smf.mixedlm('y ~ c + s', d, groups=d.g)
    null = smf.mixedlm('y ~ 1', d, groups=d.g)
    return full, null


class TestLMELikelihoodRatio(unittest.TestCase):
    """B-2: the LRT uses ML fits and is calibrated under the null."""

    def test_null_calibration(self):
        from core.rhythm_analysis import _fit_lme_model
        rng = np.random.default_rng(2026)
        ps = np.array([_fit_lme_model(*_lme_data(rng), period=24.0).p_value for _ in range(120)])
        # REML LRT: FPR 0 and median p = 1
        self.assertLess(np.mean(ps < 0.05), 0.12)
        self.assertGreater(np.mean(ps < 0.05), 0.0)
        self.assertTrue(0.3 < np.median(ps) < 0.7, np.median(ps))

    def test_lr_matches_ml_refits(self):
        from scipy.stats import chi2
        from core.rhythm_analysis import _fit_lme_model
        t, y, g = _lme_data(np.random.default_rng(3), amp=0.4)
        r = _fit_lme_model(t, y, g, period=24.0)
        full, null = _ml_fits(t, y, g)
        lr = 2 * (full.fit(reml=False).llf - null.fit(reml=False).llf)
        self.assertAlmostEqual(r.lr_stat, lr, places=3)
        self.assertAlmostEqual(r.p_value, round(chi2.sf(lr, 2), 6), places=5)

    def test_estimates_stay_reml(self):
        from core.rhythm_analysis import _fit_lme_model
        t, y, g = _lme_data(np.random.default_rng(4), amp=1.0)
        r = _fit_lme_model(t, y, g, period=24.0)
        full, _ = _ml_fits(t, y, g)
        reml = full.fit()
        self.assertAlmostEqual(r.random_effect_var, float(reml.cov_re.iloc[0, 0]), places=3)
        self.assertAlmostEqual(r.residual_var, float(reml.scale), places=3)


class TestLMEFitStatistics(unittest.TestCase):
    """B-11: marginal R², AIC/BIC k, Best Period."""

    def test_marginal_r2_is_fixed_effects_only(self):
        from core.rhythm_analysis import _fit_lme_model
        d = pd.read_csv(LME_CSV)
        d = d[d.condition == 'control']
        for var, lo, hi in [('arrhythmic_high_between_var', 0.0, 0.05),
                            ('rhythm_high_between_var', 0.3, 0.6)]:
            r = _fit_lme_model(d.time.values, d[var].values, d.subject.values, 24.0)
            full, _ = _ml_fits(d.time.values, d[var].values, d.subject.values)
            m = full.fit()
            vf = np.var(m.model.exog @ m.fe_params.values)
            nakagawa = vf / (vf + float(m.cov_re.iloc[0, 0]) + float(m.scale))
            self.assertAlmostEqual(r.r_squared, nakagawa, places=3)
            self.assertTrue(lo <= r.r_squared <= hi, (var, r.r_squared))  # was 0.93 / 0.96

    def test_aic_bic_count_five_parameters(self):
        from core.rhythm_analysis import _fit_lme_model
        t, y, g = _lme_data(np.random.default_rng(5), amp=0.5)
        r = _fit_lme_model(t, y, g, period=24.0)
        llf = _ml_fits(t, y, g)[0].fit(reml=False).llf
        self.assertAlmostEqual(r.aic, -2 * llf + 2 * 5, places=1)
        self.assertAlmostEqual(r.bic, -2 * llf + np.log(len(t)) * 5, places=1)

    def test_best_period_is_unique(self):
        from core.analysis_engine import AnalysisType
        d = pd.read_csv(LME_CSV)
        res = _quiet(_engine().run_analysis, d, 'rhythm_high_between_var', 'control', AnalysisType.LME,
                     parameters={'period': [20.0, 22.0, 24.0, 26.0, 28.0], 'random_effect': 'subject'})
        best = [r.period for r in res if r.best_model == 'Yes']
        self.assertEqual(best, [24.0])  # every period was "Yes" (p rounded to 0)


class TestF24Reproducible(unittest.TestCase):
    """B-8: the split of > 2 subjects into two replicate groups is seeded."""

    def test_more_than_two_subjects(self):
        from core.analysis_engine import AnalysisType
        d = pd.read_csv(LME_CSV)
        runs = {_quiet(_engine().run_analysis, d, 'arrhythmic_low_between_var', 'control',
                       AnalysisType.FOURIER_F24, parameters={'target_period': 24.0}).power
                for _ in range(4)}
        self.assertEqual(len(runs), 1, runs)


class TestF24NormalisedAndPermutationP(unittest.TestCase):
    """Decisions D-F24-1 (correlation_r removed) and D-F24-2 (f24_norm, pF24; f24_score unchanged)."""

    t = np.arange(0, 48, 2.0)

    def test_pure_sine(self):
        from core.rhythm_analysis import _compute_fourier_f24, _compute_fourier_f24_with_replicates
        y = 10 + 2 * np.cos(2 * np.pi * self.t / 24)
        r = _compute_fourier_f24(self.t, y)
        self.assertAlmostEqual(r.f24_norm, 1.0, places=3)
        self.assertAlmostEqual(r.p_f24, 1 / 1001, places=6)
        self.assertAlmostEqual(r.f24_score, (len(self.t) - 1) / 2, delta=1.0)  # noisy permutation mean
        r2 = _compute_fourier_f24_with_replicates(self.t, y, self.t, 0.5 * y)
        self.assertAlmostEqual(r2.f24_norm, 1.0, places=3)
        self.assertFalse(hasattr(r2, 'correlation_r'))

    def test_pf24_calibrated_on_noise(self):
        from core.rhythm_analysis import _compute_fourier_f24
        rng = np.random.default_rng(8)
        ps = np.array([_compute_fourier_f24(self.t, rng.normal(size=self.t.size), n_permutations=200).p_f24
                       for _ in range(300)])
        self.assertTrue(0.01 < np.mean(ps < 0.05) < 0.10, np.mean(ps < 0.05))

    def test_f24_score_unchanged(self):
        # values before the new outputs were added (CRS-AI consumes f24_score with n_permutations=100)
        from core.analysis_engine import AnalysisType
        from core.rhythm_analysis import _compute_fourier_f24
        d = pd.read_csv(os.path.join(PROJECT_ROOT, 'examples', 'synthetic_independent_test_data.csv'))
        r = _quiet(_engine().run_analysis, d, 'circadian_pure', 'control', AnalysisType.FOURIER_F24,
                   parameters={'target_period': 24.0})
        self.assertEqual(r.power, 11.6815)
        self.assertIn('pF24', r.message)
        sub = d[d.condition == 'treatment']
        self.assertEqual(_compute_fourier_f24(sub.time.values.astype(float), sub.arrhythmic.values,
                                              n_permutations=1000).f24_score, 0.5923)

    def test_alpha_selector_shown_for_f24(self):
        # G8: Notes uses pF24 < alpha, and alpha came from a selector shown only for Lomb-Scargle
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from ui.analysis_panel import AnalysisPanel
        p = AnalysisPanel()
        p._source_type = 'csv'
        p._update_module_combo_for_source('csv')
        p._module_combo.setCurrentText('Classical Rhythm Analysis')
        for method, shown in (('Fourier F24', True), ('Lomb-Scargle', True), ('Cosinor (OLS)', False)):
            p._method_combo.setCurrentText(method)
            self.assertEqual(p._method_combo.currentText(), method)
            self.assertEqual(not p._alpha_combo.isHidden(), shown, method)


class TestSpectralThreshold(unittest.TestCase):
    """B-17: the Refinetti threshold excludes the DC bin and is on the Lomb-Scargle power scale."""

    t = np.arange(0, 96, 1.0)
    y = 10 + 2 * np.cos(2 * np.pi * t / 24)

    def _spec(self, per_type, detrending):
        return _quiet(_engine()._run_spectral_analysis, self.t, self.y, 'v', 'c',
                      {'per_type': per_type, 'max_per': 240.0, 'detrending': detrending, 'prominent': True})

    def test_per_detrend_off_ignores_dc(self):
        on, off = self._spec('per', True), self._spec('per', False)
        self.assertAlmostEqual(off.threshold, on.threshold, places=6)
        self.assertEqual(off.significant_peaks, [24.0])  # was [] (DC bin in the sum)

    def test_lombscargle_finds_pure_sine(self):
        r = self._spec('lombscargle', True)
        self.assertEqual(len(r.significant_peaks), 1)  # was 0
        self.assertAlmostEqual(r.significant_peaks[0], 24.0, delta=0.3)
        # threshold in fraction-of-variance units equals the periodogram's g_crit
        per = self._spec('per', True)
        from scipy import signal
        f, P = signal.periodogram(self.y, 1.0, detrend='constant')
        g_per = per.threshold / P[f > 0].sum()
        g_ls = r.threshold / (np.sum((self.y - self.y.mean()) ** 2) / 2)
        self.assertAlmostEqual(g_per, g_ls, places=6)

    def test_white_noise_false_positive_rate(self):
        # Fisher's m = (N-1)//2 frequencies (decision D-S); with N the rate was 0.55-0.95
        rng = np.random.default_rng(12)
        t = np.arange(0, 48, 2.0)
        for per_type in ('per', 'lombscargle'):
            hits = 0
            for _ in range(200):
                r = _quiet(_engine()._run_spectral_analysis, t, 10 + rng.normal(size=t.size), 'v', 'c',
                           {'per_type': per_type, 'max_per': 240.0, 'detrending': True, 'prominent': True})
                hits += len(r.significant_peaks) > 0
            self.assertLess(hits / 200, 0.15, per_type)

    def test_lombscargle_no_alias_and_always_centred(self):
        t = np.arange(0, 48, 2.0)
        y = 10 + 2 * np.cos(2 * np.pi * t / 24)
        on, off = [_quiet(_engine()._run_spectral_analysis, t, y, 'v', 'c',
                          {'per_type': 'lombscargle', 'max_per': 240.0, 'detrending': dt, 'prominent': True})
                   for dt in (True, False)]
        self.assertGreaterEqual(on.periods.min(), 4.0 - 1e-9)   # Nyquist period of 2-h sampling
        self.assertEqual(len(on.significant_peaks), 1)          # was [2.2, 23.9] (alias)
        self.assertAlmostEqual(on.significant_peaks[0], 24.0, delta=0.3)
        self.assertEqual(off.significant_peaks, on.significant_peaks)

    def test_api_lombscargle_units(self):
        from core.rhythm_analysis import RhythmAnalyzer
        a = RhythmAnalyzer()
        a.load_dataframe(pd.DataFrame({'time': self.t, 'condition': 'c', 'v': self.y}))
        r = a.run_spectral_analysis('v', 'c', per_type='lombscargle')
        self.assertAlmostEqual(r['dominant_period'], 24.0, delta=0.3)  # was 24·2π, outside 240 h
        self.assertEqual(len(r['significant_peaks']), 1)


class TestCWTPeriodGrid(unittest.TestCase):
    """The CWT period grid never exceeds the maximum analyzable period."""

    def test_grid_capped(self):
        from core.rhythm_analysis import _compute_cwt
        for hours, rng in [(61.0, (20.0, 28.0)), (240.0, (20.0, 27.9)), (66.0, (20.0, 28.0))]:
            t = np.arange(0, hours, 0.5)
            r = _compute_cwt(t, np.cos(2 * np.pi * t / 24), sampling_interval=0.5, period_range=rng)
            cap = min(rng[1], (t.max() - t.min()) / 3)
            self.assertLessEqual(r.periods.max(), cap + 1e-9, (hours, rng))


if __name__ == '__main__':
    unittest.main()
