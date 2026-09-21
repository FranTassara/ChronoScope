"""Validate Locomotor Activity Analysis on tabular (CSV/Excel) activity data.

Until this revision the Locomotor Activity Analysis module was reachable only
when a DAM or AWD loader was active, because the analysis consumes a canonical
long frame (time / condition / subject / activity) that only those two loaders
emit. Reviewer 3 pointed out that behavioural recordings do not always arrive in
those formats (video tracking, for instance). The module now accepts any tabular
activity table through AnalysisWorker._activity_frame_from_table(), which maps
the user's column choices onto that same canonical frame.

This script checks two things:

  1. the mapping itself, including the cases it is expected to reject; and
  2. that the frame it produces actually drives the real downstream analysis,
     by feeding it to the same chi_square_periodogram / compute_is_iv /
     compute_alpha_rho / filter_days functions the GUI calls.

Run:  python validation/methods/validate_locomotor_tabular_input.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.visualization_circadian_metrics import (  # noqa: E402
    chi_square_periodogram,
    compute_alpha_rho,
    compute_is_iv,
    filter_days,
)
from ui.analysis_panel import AnalysisWorker  # noqa: E402

TRUE_PERIOD = 23.7
N_DAYS = 6
BIN_H = 0.5
SEED = 42


# --------------------------------------------------------------------------
# Stubs: _activity_frame_from_table only touches self.loader, self.config and
# self.source_type, so it can be exercised without constructing a QThread.
# --------------------------------------------------------------------------

class _StubConfig:
    def __init__(self, variables):
        self.variables = variables


class _StubLoader:
    """Mimics CircadianDataLoader's column-role accessors."""

    def __init__(self, df, time_col, cond_col, subject_col):
        self._df = df
        self._time_col = time_col
        self._condition_col = cond_col
        self._subject_col = subject_col

    def get_data(self):
        return self._df

    def get_time_column(self):
        return self._time_col

    def get_condition_column(self):
        return self._condition_col


class _StubWorker:
    def __init__(self, loader, variables, source_type='csv'):
        self.loader = loader
        self.config = _StubConfig(variables)
        self.source_type = source_type

    # bind the real implementation
    _activity_frame_from_table = AnalysisWorker._activity_frame_from_table


def make_table(n_subjects=4, n_days=N_DAYS, with_subject=True,
               with_condition=True, activity_numeric=True):
    """A bimodal-ish circadian activity table in arbitrary wide-ish long form."""
    rng = np.random.default_rng(SEED)
    times = np.arange(0, n_days * 24, BIN_H)
    rows = []
    for s in range(n_subjects):
        phase = rng.uniform(0, 2 * np.pi)
        base = 20 * (1 + np.cos(2 * np.pi * times / TRUE_PERIOD - phase))
        counts = rng.poisson(np.clip(base, 0.1, None))
        for t, c in zip(times, counts):
            row = {'hours_elapsed': float(t),
                   'counts': int(c) if activity_numeric else f"v{c}"}
            if with_subject:
                row['fly_id'] = f"fly{s:02d}"
            if with_condition:
                row['genotype'] = 'wt' if s % 2 == 0 else 'mut'
            rows.append(row)
    return pd.DataFrame(rows)


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return bool(condition)


def main():
    print('=' * 78)
    print('Locomotor Activity Analysis on tabular input -- validation')
    print('=' * 78)
    ok = True

    # ---------------------------------------------------------------- mapping
    print('\n1. COLUMN MAPPING\n' + '-' * 78)

    df = make_table()
    w = _StubWorker(_StubLoader(df, 'hours_elapsed', 'genotype', 'fly_id'), ['counts'])
    frame, err = w._activity_frame_from_table(df)
    ok &= check('full table accepted', err is None, err or '')
    if frame is not None:
        ok &= check('canonical columns present',
                    {'time', 'condition', 'activity', 'subject'} <= set(frame.columns),
                    ', '.join(frame.columns))
        ok &= check('no rows lost', len(frame) == len(df), f'{len(frame)} of {len(df)}')
        ok &= check('sorted by time', frame['time'].is_monotonic_increasing)
        ok &= check('subjects preserved', frame['subject'].nunique() == 4,
                    f"{frame['subject'].nunique()} subjects")
        ok &= check('conditions preserved', set(frame['condition']) == {'wt', 'mut'})

    df_nosub = make_table(with_subject=False, with_condition=False)
    w = _StubWorker(_StubLoader(df_nosub, 'hours_elapsed', None, None), ['counts'])
    frame_ns, err = w._activity_frame_from_table(df_nosub)
    ok &= check('table without subject/condition accepted', err is None, err or '')
    if frame_ns is not None:
        ok &= check('subject column omitted when unknown', 'subject' not in frame_ns.columns)
        ok &= check('condition defaults to a single group',
                    set(frame_ns['condition']) == {'all'})

    # ------------------------------------------------------------- rejections
    print('\n2. REJECTED INPUTS\n' + '-' * 78)

    w = _StubWorker(_StubLoader(df, 'hours_elapsed', 'genotype', 'fly_id'), [])
    _, err = w._activity_frame_from_table(df)
    ok &= check('no variable selected is rejected', err is not None and 'activity measure' in err,
                err or 'no error raised')

    w = _StubWorker(_StubLoader(df, None, 'genotype', 'fly_id'), ['counts'])
    _, err = w._activity_frame_from_table(df)
    ok &= check('missing time column is rejected', err is not None and 'time column' in err,
                err or 'no error raised')

    df_txt = make_table(activity_numeric=False)
    w = _StubWorker(_StubLoader(df_txt, 'hours_elapsed', 'genotype', 'fly_id'), ['counts'])
    _, err = w._activity_frame_from_table(df_txt)
    ok &= check('non-numeric activity is rejected', err is not None and 'numeric values' in err,
                err or 'no error raised')

    df_short = make_table(n_days=1)
    df_short = df_short[df_short['hours_elapsed'] < 12]
    w = _StubWorker(_StubLoader(df_short, 'hours_elapsed', 'genotype', 'fly_id'), ['counts'])
    _, err = w._activity_frame_from_table(df_short)
    ok &= check('recording under 24 h is rejected', err is not None and '24 h' in err,
                err or 'no error raised')

    # ------------------------------------------------- downstream end-to-end
    print('\n3. DOWNSTREAM ANALYSIS ON THE MAPPED FRAME\n' + '-' * 78)

    w = _StubWorker(_StubLoader(df, 'hours_elapsed', 'genotype', 'fly_id'), ['counts'])
    frame, err = w._activity_frame_from_table(df)
    if frame is None:
        print(f'  cannot continue: {err}')
        return 1

    # The GUI derives zt/day exactly this way before plotting.
    frame['zt'] = (frame['time'] % 24).round(2)
    frame['day'] = (frame['time'] // 24).astype(int) + 1
    ok &= check('day index spans the recording', int(frame['day'].max()) == N_DAYS,
                f"{int(frame['day'].max())} days")

    prof = frame.groupby('time', as_index=False)['activity'].mean()
    times = prof['time'].to_numpy(dtype=float)
    act = prof['activity'].to_numpy(dtype=float)

    cs = chi_square_periodogram(times, act, period_min=18.0, period_max=32.0, alpha=0.05)
    tau = cs.get('tau')
    ok &= check('chi-square periodogram returns a period', tau is not None)
    if tau is not None:
        ok &= check('recovered tau is close to the simulated period',
                    abs(tau - TRUE_PERIOD) < 0.5, f'tau={tau:.2f} h vs {TRUE_PERIOD} h')

    isiv = compute_is_iv(times, act)
    ok &= check('IS/IV computed', isiv.get('IS') is not None and isiv.get('IV') is not None,
                f"IS={isiv.get('IS')}, IV={isiv.get('IV')}")
    if isiv.get('IS') is not None:
        ok &= check('IS is high for a strongly rhythmic signal', isiv['IS'] > 0.5,
                    f"IS={isiv['IS']:.3f}")

    ar = compute_alpha_rho(times, act, threshold_method='mean')
    ok &= check('alpha/rho computed',
                ar.get('alpha_mean') is not None and ar.get('rho_mean') is not None,
                f"alpha_mean={ar.get('alpha_mean')}, rho_mean={ar.get('rho_mean')}")
    if ar.get('alpha_mean') is not None and ar.get('rho_mean') is not None:
        ok &= check('alpha + rho is one day by construction',
                    abs((ar['alpha_mean'] + ar['rho_mean']) - 24.0) < 1e-6,
                    f"alpha+rho={ar['alpha_mean'] + ar['rho_mean']:.4f} h")
        ok &= check('one alpha/rho value per recorded day',
                    len(ar['days']) == N_DAYS, f"{len(ar['days'])} days")

    ft, fa = filter_days(times, act, 2, 4)
    ok &= check('day filtering restricts the window',
                len(ft) > 0 and ft.min() >= 24.0 and ft.max() < 96.0,
                f'{len(ft)} points in [{ft.min():.1f}, {ft.max():.1f}] h' if len(ft) else 'empty')

    print('\n' + '=' * 78)
    print('RESULT:', 'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    print('=' * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
