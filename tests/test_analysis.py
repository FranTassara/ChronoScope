"""Behavioural smoke tests for the analysis engine.

These do not check exact numbers -- they check that each method still
distinguishes a clean 24 h rhythm from noise sampled on the same grid, which is
the property that actually breaks when a wrapper or a dependency shifts.
"""
import unittest

import pandas as pd

from tests.helpers import (PROJECT_ROOT, VARIABLE, CONDITION,
                           rhythmic_dataframe, noise_dataframe)

# Methods that report a rhythmicity p-value for a single series.
PVALUE_METHODS = ('JTK', 'COSINOR_OLS', 'LOMB_SCARGLE', 'HARMONIC_COSINOR')


def _run(engine, analysis_type, df):
    from core.analysis_engine import AnalysisType
    result = engine.run_analysis(
        data=df, variable=VARIABLE, condition=CONDITION,
        analysis_type=getattr(AnalysisType, analysis_type),
        time_col='time', condition_col='condition', parameters={},
    )
    return result[0] if isinstance(result, list) else result


class TestRhythmDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from core.analysis_engine import AnalysisEngine
        cls.engine = AnalysisEngine()
        cls.rhythmic = rhythmic_dataframe()
        # Several independent noise draws: a single realisation makes the
        # comparison a coin flip rather than a check on the method.
        cls.noise_sets = [noise_dataframe(seed=7000 + i) for i in range(5)]

    def test_methods_are_available(self):
        from core.analysis_engine import AnalysisType
        for name in PVALUE_METHODS:
            available, message = self.engine.check_method_available(getattr(AnalysisType, name))
            with self.subTest(method=name):
                self.assertTrue(available, f'{name} unavailable: {message}')

    def test_rhythmic_signal_is_detected(self):
        for name in PVALUE_METHODS:
            with self.subTest(method=name):
                result = _run(self.engine, name, self.rhythmic)
                self.assertTrue(result.success, f'{name} failed: {result.message}')
                self.assertIsNotNone(result.p_value, f'{name} returned no p-value')
                self.assertLess(result.p_value, 0.05,
                                f'{name} did not detect a clean 24 h rhythm')

    def test_noise_scores_worse_than_signal(self):
        """The rhythm must beat the median of several independent noise draws.

        Comparing against one noise realisation is flaky by construction: any
        method will occasionally meet a noise series that looks periodic.
        """
        import numpy as np
        for name in PVALUE_METHODS:
            with self.subTest(method=name):
                rhythmic = _run(self.engine, name, self.rhythmic)
                noise_p = []
                for df in self.noise_sets:
                    result = _run(self.engine, name, df)
                    self.assertTrue(result.success,
                                    f'{name} failed on noise: {result.message}')
                    self.assertIsNotNone(result.p_value)
                    noise_p.append(float(result.p_value))
                self.assertLess(rhythmic.p_value, float(np.median(noise_p)),
                                f'{name} ranked noise at least as rhythmic as the signal')

    def test_recovered_period_is_near_24h(self):
        result = _run(self.engine, 'LOMB_SCARGLE', self.rhythmic)
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.period, 24.0, delta=2.0)


class TestCountDataPath(unittest.TestCase):
    """Exercises the vendored RhythmCount package end to end."""

    def test_rhythmcount_runs_on_the_shipped_example(self):
        from core.analysis_engine import AnalysisEngine, AnalysisType
        csv = PROJECT_ROOT / 'examples' / 'synthetic_rhythmcount_test_data.csv'
        self.assertTrue(csv.is_file(), f'missing example dataset: {csv}')
        df = pd.read_csv(csv)

        time_col = 'time' if 'time' in df.columns else df.columns[0]
        condition_col = 'condition' if 'condition' in df.columns else None
        if condition_col is None:
            df = df.assign(condition='all')
            condition_col = 'condition'
        variable = [c for c in df.columns if c not in (time_col, condition_col)][0]
        condition = df[condition_col].iloc[0]

        engine = AnalysisEngine()
        available, message = engine.check_method_available(AnalysisType.RHYTHMCOUNT_SINGLE)
        self.assertTrue(available, f'RhythmCount unavailable: {message}')

        result = engine.run_analysis(
            data=df, variable=variable, condition=condition,
            analysis_type=AnalysisType.RHYTHMCOUNT_SINGLE,
            time_col=time_col, condition_col=condition_col, parameters={},
        )
        result = result[0] if isinstance(result, list) else result
        self.assertTrue(result.success, f'RhythmCount failed: {result.message}')


if __name__ == '__main__':
    unittest.main()
