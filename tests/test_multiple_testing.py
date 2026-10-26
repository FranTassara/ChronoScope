"""CosinorPy q-values corrected across variables (core/multiple_testing.py).

G3 decision (2026-10-21): ChronoScope runs CosinorPy once per variable, so the
library's Benjamini-Hochberg q-values covered a single series (q = p with one
period, or the periods / condition pairs of one series). They are now
recomputed after a run over all of its variables; a selected model of a
period / n_components scan enters with its p-values Bonferroni-adjusted for the
number of models fitted.
"""
import io
import contextlib
import subprocess
import sys
import unittest
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from tests.helpers import PROJECT_ROOT

warnings.filterwarnings('ignore')


def _bh(p):
    return list(multipletests(p, method='fdr_bh')[1])


def _fit(var, p, cond='ctrl', method='cosinorpy_nonlinear_independent', period=24.0, n=1, best=None, **kw):
    return {'variable': var, 'condition': cond, 'method': method, 'period': period, 'n_components': n,
            'best_model': best, 'p_value': p, 'q_value': p, **kw}


def _bh_rows(rows, **params):
    from core.multiple_testing import bh_across_variables
    return bh_across_variables(rows, params)


class TestFits(unittest.TestCase):

    def test_single_period_family_is_variables(self):
        p = [0.01, 0.04, 0.03, 0.5]
        rows = _bh_rows([_fit(f'v{i}', pi, p_amplitude=pi / 2) for i, pi in enumerate(p)])
        self.assertEqual([r['q_value'] for r in rows], _bh(p))
        self.assertEqual([r['q_amplitude'] for r in rows], _bh([x / 2 for x in p]))

    def test_conditions_and_methods_are_separate_families(self):
        rows = _bh_rows([_fit('a', 0.01), _fit('b', 0.02), _fit('a', 0.01, cond='trt'),
                         _fit('a', 0.01, method='cosinorpy_independent')])
        self.assertEqual([r['q_value'] for r in rows], [0.02, 0.02, 0.01, 0.01])

    def test_period_scan_uses_selected_row_bonferroni(self):
        rows = []
        for v, ps, best in (('a', [1e-3, 1e-4, 1e-2], 1), ('b', [0.2, 0.01, 0.3], 1)):
            rows += [_fit(v, pi, period=P, best='Yes (min p-value)' if i == best else 'No')
                     for i, (P, pi) in enumerate(zip((22.0, 24.0, 26.0), ps))]
        _bh_rows(rows, period_range=(22.0, 26.0), period_step=2.0, n_components=[1])
        sel = [r for r in rows if r['best_model'].startswith('Yes')]
        self.assertTrue(np.allclose([r['bonf_p_value'] for r in sel], [3e-4, 0.03]))
        self.assertTrue(np.allclose([r['q_value'] for r in sel], _bh([3e-4, 0.03])))
        self.assertTrue(all(r['q_value'] is None for r in rows if r['best_model'] == 'No'))

    def test_collapsed_scan_row_and_n_components_list(self):
        # one row per series returned for a 3-period x 2-n_components scan (e.g. 2.2 with a fixed N)
        rows = _bh_rows([_fit('a', 0.001, method='cosinorpy_independent'),
                         _fit('b', 0.01, method='cosinorpy_independent')],
                        period_range=(22.0, 26.0), period_step=2.0, n_components=[1, 2])
        self.assertTrue(np.allclose([r['bonf_p_value'] for r in rows], [0.006, 0.06]))
        self.assertTrue(np.allclose([r['q_value'] for r in rows], _bh([0.006, 0.06])))

    def test_lack_of_fit_not_scan_adjusted(self):
        rows = _bh_rows([_fit('a', 0.001, p_reject=0.04), _fit('b', 0.002, p_reject=0.5)],
                        period_range=(20.0, 28.0), n_components=[1])
        self.assertEqual([r['q_reject'] for r in rows], _bh([0.04, 0.5]))

    def test_other_methods_untouched(self):
        rows = _bh_rows([{'variable': 'a', 'condition': 'c', 'method': 'jtk', 'p_value': 0.01, 'q_value': 0.5},
                         _fit('a', 0.01, method='cosinorpy_periodogram')])
        self.assertEqual(rows[0]['q_value'], 0.5)
        self.assertEqual(rows[1]['q_value'], 0.01)


class TestComparisons(unittest.TestCase):

    def test_pairs_and_variables_pooled(self):
        def cmp(v, c1, c2, pa, p1):
            return {'variable': v, 'condition1': c1, 'condition2': c2, 'method': 'cosinorpy_compare_pooled',
                    'period': 24.0, 'n_components': None, 'p_amplitude': pa, 'q_amplitude': pa,
                    'p1': p1, 'q1': None, 'model_p': None}
        rows = _bh_rows([cmp('a', 'x', 'y', 0.01, 0.2), cmp('a', 'x', 'z', 0.03, 0.2),
                         cmp('b', 'x', 'y', 0.02, 0.001)])
        self.assertEqual([r['q_amplitude'] for r in rows], _bh([0.01, 0.03, 0.02]))
        self.assertEqual([r['q1'] for r in rows], _bh([0.2, 0.2, 0.001]))
        self.assertIsNone(rows[0]['model_q'])


class TestEndToEnd(unittest.TestCase):

    def test_nonlinear_examples(self):
        from core.analysis_engine import AnalysisEngine, AnalysisType
        d = pd.read_csv(PROJECT_ROOT / 'examples' / 'synthetic_independent_test_data.csv')
        eng, rows = AnalysisEngine(), []
        params = {'period_range': (24.0, 24.0), 'n_components': [1]}
        for v in ('circadian_noisy', 'ultradian_12h', 'arrhythmic', 'with_outliers'):
            with contextlib.redirect_stdout(io.StringIO()):
                rows.append(eng.run_analysis(d, v, 'control', AnalysisType.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT,
                                             parameters={**params, 'use_dependent_model': False})[0].to_dict())
        p = [r['p_amplitude'] for r in rows]
        self.assertEqual([r['q_amplitude'] for r in rows], p)          # library: one pair per call
        _bh_rows(rows, **params)
        self.assertEqual([r['q_amplitude'] for r in rows], _bh(p))

    def test_cli_runs_comparisons_once_per_variable(self):
        out = PROJECT_ROOT / 'tests' / '_cli_compare_tmp.csv'
        try:
            subprocess.run([sys.executable, str(PROJECT_ROOT / 'cli.py'),
                            str(PROJECT_ROOT / 'examples' / 'synthetic_independent_test_data.csv'),
                            '--method', 'cosinorpy_compare_independent', '--variable', 'circadian_pure',
                            'circadian_noisy', 'arrhythmic', '--output', str(out)],
                           check=True, capture_output=True, cwd=PROJECT_ROOT)
            df = pd.read_csv(out)
            self.assertEqual(len(df), 3)
            self.assertTrue(np.allclose(df['q_amplitude'], _bh(df['p_amplitude'].tolist())))
        finally:
            out.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
