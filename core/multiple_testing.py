"""
Benjamini–Hochberg correction across variables for the CosinorPy methods (Module II).

CosinorPy's *_group and *_compare_pairs functions apply BH to every test passed
in one call, so called once with all genes the family is all genes. ChronoScope
calls them once per variable (and, for nonlinear period scans, once per period),
which left the library's q-values computed over a single series: q = p in most
paths, and a correction over the periods or condition pairs of one series in
the rest. bh_across_variables() recomputes every CosinorPy q-value after a run,
over all the variables of that run:

* Fits (2.2, 2.3, 2.6, 2.7): one family per method x condition x parameter,
  one test per variable. When a series was fitted with a single model, the
  family is further split by period and n_components. When several models were
  fitted (a period range and/or several n_components), the test of the series is
  its selected model (best_model "Yes ..." / "Best model ...", or the only row
  returned), with its p-values Bonferroni-adjusted for the m = periods x
  n_components candidates before BH (m·p_value is reported as bonf_p_value, as
  in Module I); the other rows of that series get no q-value.
* Comparisons (2.4, 2.5, 2.8, 2.9): one family per method x period x
  n_components x parameter, spanning every condition pair of every variable.
"""

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from statsmodels.stats.multitest import multipletests

FIT_PQ = [('p_value', 'q_value'), ('p_amplitude', 'q_amplitude'), ('p_acrophase', 'q_acrophase'),
          ('p_mesor', 'q_mesor'), ('p_amplification', 'q_amplification'), ('p_lin_comp', 'q_lin_comp'),
          ('p_reject', 'q_reject')]
COMPARISON_PQ = [('p_amplitude', 'q_amplitude'), ('p_acrophase', 'q_acrophase'), ('p_mesor', 'q_mesor'),
                 ('p_amplification', 'q_amplification'), ('p_lin_comp', 'q_lin_comp'),
                 ('p1', 'q1'), ('p2', 'q2'), ('model_p', 'model_q'),
                 ('p_params', 'q_params'), ('p_ftest', 'q_ftest')]
# p_reject is a lack-of-fit test (small p = poor fit): not adjusted for the model selection
NOT_SCAN_ADJUSTED = {'p_reject'}


def is_cosinorpy(result: Dict[str, Any]) -> bool:
    method = result.get('method') or ''
    return method.startswith('cosinorpy_') and method != 'cosinorpy_periodogram'


def n_candidate_models(parameters: Dict[str, Any]) -> int:
    """Models fitted per series by a CosinorPy fit: periods scanned x n_components values."""
    lo, hi = parameters.get('period_range', (24.0, 24.0))
    step = parameters.get('period_step', 1.0)
    n_periods = len(np.arange(lo, hi + step / 2, step)) if lo != hi else 1
    n_comp = parameters.get('n_components', [1])
    n_comp = n_comp if isinstance(n_comp, (list, tuple)) else [n_comp]
    return n_periods * max(len(n_comp), 1)


def _is_selected(result: Dict[str, Any]) -> bool:
    best = str(result.get('best_model') or '')
    return best.startswith('Yes') or best.startswith('Best model')


def _p(value) -> bool:
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def bh_across_variables(results: List[Dict[str, Any]], parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recompute the q-values of the CosinorPy rows of one run over all its variables (in place)."""
    m = n_candidate_models(parameters)
    # family key -> list of (row, p-value column, q-value column, multiplier)
    families = defaultdict(list)

    def add(key, row, pq, factor=1):
        for p_col, q_col in pq:
            row[q_col] = None
            if _p(row.get(p_col)):
                f = 1 if p_col in NOT_SCAN_ADJUSTED else factor
                families[key + (p_col,)].append((row, p_col, q_col, f))

    rows = [r for r in results if is_cosinorpy(r)]
    comparisons = [r for r in rows if 'condition1' in r]
    fits = [r for r in rows if 'condition1' not in r]

    for r in comparisons:
        add((r['method'], r.get('period'), r.get('n_components')), r, COMPARISON_PQ)

    units = defaultdict(list)  # one series: method x condition x variable (x Rosbash cluster)
    for r in fits:
        units[(r['method'], r.get('condition'), r.get('cluster'), r.get('variable'))].append(r)
    for (method, condition, cluster, _), unit in units.items():
        selected = [r for r in unit if _is_selected(r)] or (unit if len(unit) == 1 else [])
        if m > 1 and len(selected) == 1:
            sel = selected[0]
            for r in unit:
                if r is not sel:
                    for _, q_col in FIT_PQ:
                        r[q_col] = None
            if _p(sel.get('p_value')):
                sel['bonf_p_value'] = min(1.0, m * float(sel['p_value']))
            add((method, condition, cluster, 'selected'), sel, FIT_PQ, factor=m)
        else:
            for r in unit:
                add((method, condition, cluster, r.get('period'), r.get('n_components')), r, FIT_PQ)

    for members in families.values():
        p = [min(1.0, f * float(row[p_col])) for row, p_col, _, f in members]
        q = multipletests(p, method='fdr_bh')[1]
        for (row, _, q_col, _), qv in zip(members, q):
            row[q_col] = float(qv)

    return results
