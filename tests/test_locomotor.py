"""DAM/AWD loaders and the Locomotor Activity Analysis module (Module VI).

Regression guards for the 2026-10-19 manual audit, group G1:
  B-1  samples between midnight and lights-on were placed 24 h late, so the
       time axis was not monotonic;
  B-9  the AWD epoch was fixed at 6 min instead of read from the header;
  B-20 the chi-square phase label always said "DD", the fallback to the full
       recording was silent, and LL ran to the end with "LL ends day" = 0;
  B-22 unrecorded (day, ZT) cells were drawn as 0, the scan-wide threshold
       was drawn for the first condition only, preprocessing was ignored
       silently and the "Lights on (ZT0)" row leaked into other modules.
"""
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tests.helpers import PROJECT_ROOT  # noqa: F401  (puts the project on sys.path)

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _write_dam(path, start, n_minutes):
    """Minimal TriKinetics monitor file: one row per minute, 32 channels."""
    rng = np.random.default_rng(0)
    months = 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split()
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(n_minutes):
            dt = start + timedelta(minutes=i)
            date = f"{dt.day} {months[dt.month - 1]} {dt.year % 100:02d}"
            counts = '\t'.join(str(c) for c in rng.poisson(2, 32))
            f.write(f"{i + 1}\t{date}\t{dt:%H:%M:%S}\t1\t0\t51\t0\tCt\t0\t{counts}\n")


def _write_awd(path, interval_code, n_samples, start='23-Jan-2020', time='13:58'):
    rng = np.random.default_rng(1)
    header = ['test_animal', start, time, f' {interval_code} ', '00', 'V000000', 'X']
    with open(path, 'w', encoding='latin-1') as f:
        f.write('\n'.join(header) + '\n')
        f.write('\n'.join(str(v) for v in rng.poisson(20, n_samples)) + '\n')


class TestDAMZeitgeberTime(unittest.TestCase):
    """B-1: ZT is continuous from the first lights-on, never 24 h late."""

    def _loader(self, start, n_minutes, lights_on=8):
        from utils.dam_loader import DAMDataLoader, DAMConfig
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'Monitor51.txt'
        _write_dam(path, start, n_minutes)
        ld = DAMDataLoader()
        ld.load_dam(str(path))
        ld.configure(DAMConfig(lights_on_hour=lights_on, exclude_dead=False))
        return ld

    def test_after_midnight_belongs_to_the_previous_zt_day(self):
        ld = self._loader(datetime(2025, 7, 4, 16, 28), 2 * 24 * 60)
        self.assertEqual(ld._datetime_to_zt(datetime(2025, 7, 5, 0, 30)), 16.5)
        self.assertEqual(ld._datetime_to_zt(datetime(2025, 7, 5, 7, 30)), 23.5)
        self.assertEqual(ld._datetime_to_zt(datetime(2025, 7, 5, 8, 30)), 24.5)

    def test_zt_is_monotonic_in_clock_time(self):
        ld = self._loader(datetime(2025, 7, 4, 16, 28), 3 * 24 * 60)
        zt = ld._raw_data['datetime'].apply(ld._datetime_to_zt).to_numpy()
        self.assertTrue(np.all(np.diff(zt) > 0))
        # Elapsed ZT equals elapsed clock time
        span_h = (ld._raw_data['datetime'].max() - ld._raw_data['datetime'].min()).total_seconds() / 3600
        self.assertAlmostEqual(zt[-1] - zt[0], span_h, places=3)

    def test_recording_that_starts_before_lights_on(self):
        """Day 1 starts at the last lights-on before the first sample, so
        ZT stays non-negative and monotonic."""
        ld = self._loader(datetime(2025, 7, 4, 6, 0), 2 * 24 * 60)
        zt = ld._raw_data['datetime'].apply(ld._datetime_to_zt).to_numpy()
        self.assertEqual(zt[0], 22.0)
        self.assertTrue(np.all(np.diff(zt) > 0))

    def test_binned_output_has_no_gaps(self):
        ld = self._loader(datetime(2025, 7, 4, 16, 28), 3 * 24 * 60)
        t = np.sort(ld.process()['time'].unique())
        self.assertTrue(np.allclose(np.diff(t), 0.5))


class TestAWDEpoch(unittest.TestCase):
    """B-9: the native epoch comes from header line 4 (minutes x 4)."""

    def _load(self, code, n):
        from utils.awd_loader import AWDDataLoader
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'a.AWD'
        _write_awd(path, code, n)
        ld = AWDDataLoader()
        ld.load_awd(str(path))
        return ld

    def test_code_4_is_one_minute(self):
        ld = self._load(4, 2 * 1440)
        s = ld.get_summary()
        self.assertEqual(s.native_bin_size_minutes, 1)
        self.assertAlmostEqual(s.total_days, 2.0, delta=0.01)

    def test_code_24_is_six_minutes(self):
        ld = self._load(24, 2 * 240)
        self.assertEqual(ld.get_summary().native_bin_size_minutes, 6)

    def test_rebinning_to_the_requested_bin(self):
        from utils.awd_loader import AWDConfig
        ld = self._load(4, 2 * 1440)
        ld.configure(AWDConfig(target_bin_size_minutes=6, lights_on_hour=7))
        out = ld.process()
        t = out['time'].to_numpy()
        self.assertTrue(np.allclose(np.diff(t), 0.1))
        # Re-binning sums counts: total activity is preserved
        self.assertEqual(out['activity'].sum(), int(ld._raw_data['raw_activity'].sum()))

    def test_zt_is_monotonic(self):
        ld = self._load(4, 2 * 1440)
        zt = ld._raw_data['datetime'].apply(ld._datetime_to_zt).to_numpy()
        self.assertTrue(np.all(np.diff(zt) > 0))

    def test_optional_light_column_is_ignored(self):
        """Data lines may read "count,light"; the file used not to load at
        all, and the count must not absorb the light digits."""
        from utils.awd_loader import AWDDataLoader
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / 'light.AWD'
        body = ['120,5', '0,250', '37,250', '-1,0'] * 3
        path.write_text('\n'.join(['a', '23-Jan-2020', '13:58', '4', '00', 'V0', 'X'] + body) + '\n',
                        encoding='latin-1')
        ld = AWDDataLoader()
        ld.load_awd(str(path))
        vals = ld._raw_data['raw_activity'].tolist()
        self.assertEqual(vals[:3], [120, 0, 37])
        self.assertTrue(np.isnan(vals[3]))
        self.assertEqual(ld.get_summary().missing_data_bins, 3)


# --------------------------------------------------------------------------
# Module VI, run headless through the real worker method
# --------------------------------------------------------------------------

class _Sig:
    def __init__(self, sink=None):
        self.sink = sink

    def emit(self, *args):
        if self.sink is not None:
            self.sink.append(args)


class _Cfg:
    def __init__(self, params, preprocessing=None):
        self.parameters = params
        self.variables = ['activity']
        self.preprocessing = preprocessing


class _Loader:
    def __init__(self, df):
        self._df = df
        self._config = type('C', (), {'lights_on_hour': 8})()

    def get_data(self):
        return self._df.copy()


def _activity_frame(n_subjects=3, n_days=6, t0=8.0, period=23.7, seed=2):
    """Canonical long frame as the DAM loader emits it (time in ZT hours)."""
    rng = np.random.default_rng(seed)
    t = np.arange(t0, n_days * 24, 0.5)
    rows = []
    for s in range(n_subjects):
        lam = 5 + 15 * (np.cos(2 * np.pi * t / period) > 0.3)
        for ti, c in zip(t, rng.poisson(lam)):
            rows.append({'time': ti, 'condition': 'wt', 'subject': f's{s}', 'activity': int(c)})
    return pd.DataFrame(rows)


class _TableLoader(_Loader):
    """Generic CSV/Excel loader with column roles (tabular path)."""
    _subject_col = 'subject'

    def get_time_column(self):
        return 'time'

    def get_condition_column(self):
        return 'condition'


def _run_module(df, params=None, preprocessing=None, source_type='dam'):
    from ui.analysis_panel import AnalysisWorker

    class W:
        _run_activity_profile_visualization = AnalysisWorker._run_activity_profile_visualization
        _activity_frame_from_table = AnalysisWorker._activity_frame_from_table

    w = W()
    w.source_type = source_type
    w.loader = _Loader(df) if source_type in ('dam', 'awd') else _TableLoader(df)
    w.config = _Cfg({'viz_ld_end_day': 0, 'viz_dd_end_day': 0, 'viz_ll_end_day': 0,
                     'viz_dd_skip_days': 1, **(params or {})}, preprocessing)
    w.results, out = [], []
    w.progress, w.finished = _Sig(), _Sig(out)
    w._run_activity_profile_visualization()
    ok, res, msg = out[-1]
    assert ok, msg
    return res[-1], msg


class TestLocomotorPhases(unittest.TestCase):
    """B-20."""

    @classmethod
    def setUpClass(cls):
        cls.df = _activity_frame()

    def test_dd_phase_reported_when_used(self):
        res, msg = _run_module(self.df)
        self.assertEqual(res['chi_sq_data']['wt']['phase_used'], 'DD')
        self.assertEqual(res['chi_sq_data']['wt']['n_full_recording'], 0)
        self.assertNotIn('Warning', msg)

    def test_fallback_to_full_recording_is_labelled_and_reported(self):
        # DD = day 1 only, and its first day is skipped -> no DD data left
        res, msg = _run_module(self.df, {'viz_dd_end_day': 1})
        chi = res['chi_sq_data']['wt']
        self.assertEqual(chi['phase_used'], 'all')
        self.assertEqual(chi['n_full_recording'], chi['n_total'])
        self.assertIn('full recording', msg)

    def test_ll_absent_when_ll_end_is_zero(self):
        res, _ = _run_module(self.df, {'viz_ld_end_day': 1, 'viz_dd_end_day': 3,
                                       'viz_ll_end_day': 0})
        self.assertIsNone(res['lighting_phases']['LL'])
        res, _ = _run_module(self.df, {'viz_ld_end_day': 1, 'viz_dd_end_day': 3,
                                       'viz_ll_end_day': 5})
        self.assertEqual(res['lighting_phases']['LL'], (4, 5))


class TestLocomotorOutputs(unittest.TestCase):
    """B-22 (data side)."""

    @classmethod
    def setUpClass(cls):
        cls.res, cls.msg = _run_module(_activity_frame(t0=8.0))

    def test_unrecorded_cells_are_missing_not_zero(self):
        p = self.res['profile_data']['wt']
        m = np.array(p['activity_matrix'], dtype=float)
        zt = np.array(p['zt_times'])
        self.assertTrue(np.all(np.isnan(m[0, zt < 8.0])))   # day 1 starts at ZT 8
        self.assertFalse(np.any(np.isnan(m[1])))            # day 2 fully recorded

    def test_preprocessing_is_reported_as_not_applied(self):
        from core.preprocessing import PreprocessingConfig
        _, msg = _run_module(_activity_frame(), preprocessing=PreprocessingConfig(smooth=True))
        self.assertIn('preprocessing settings are not applied', msg)
        self.assertNotIn('preprocessing', self.msg)


class TestAlphaRhoDays(unittest.TestCase):
    """Onset/offset SD: complete days only, circular SD (decision 2026-10-21)."""

    def _series(self, onsets, t0=0.0, t_end=None, length=6):
        """1 h bins; activity 10 for `length` h from each day's onset, else 0."""
        t = np.arange(t0, len(onsets) * 24 if t_end is None else t_end, 1.0)
        a = np.zeros_like(t)
        for d, on in enumerate(onsets):
            for h in range(length):
                a[np.isclose(t, d * 24 + (on + h) % 24)] = 10
        return t, a

    def test_circular_sd(self):
        from core.visualization_circadian_metrics import _circular_sd_h
        # 1 h apart, each 0.5 h from the circular mean (ZT 0)
        self.assertAlmostEqual(_circular_sd_h([23.5, 0.5] * 3), 0.5, delta=0.01)
        self.assertAlmostEqual(_circular_sd_h([12.0] * 4), 0.0, places=6)

    def test_onsets_across_midnight_have_small_sd(self):
        from core.visualization_circadian_metrics import compute_alpha_rho
        # One active hour per day, alternating ZT 23 and ZT 1 (2 h apart)
        r = compute_alpha_rho(*self._series([23, 1] * 3, length=1))
        self.assertEqual(r['onset_zt'], [23.0, 1.0] * 3)
        self.assertLess(r['onset_sd'], 1.1)       # linear SD would be 11 h

    def test_partial_days_are_excluded(self):
        from core.visualization_circadian_metrics import compute_alpha_rho
        # Recording starts at ZT 8.0 of day 1 and stops at ZT 2.0 of day 5
        t, a = self._series([12, 12, 12, 12, 12], t0=8.0, t_end=4 * 24 + 3)
        r = compute_alpha_rho(t, a)
        self.assertEqual(r['days'], [2, 3, 4])
        self.assertAlmostEqual(r['onset_sd'], 0.0, places=6)
        self.assertAlmostEqual(r['alpha_mean'], 6.0)


class TestTabularZT(unittest.TestCase):
    """Tabular time is clock hours; 'Lights on (ZT0)' converts it to ZT."""

    def test_lights_on_shifts_tabular_time_to_zt(self):
        df = _activity_frame(t0=0.0)              # clock hours from midnight
        res0, _ = _run_module(df, {'viz_lights_on_hour': 0}, source_type='csv')
        res8, _ = _run_module(df, {'viz_lights_on_hour': 8}, source_type='csv')
        # Midnight is ZT 16 of the day that started at the previous lights-on
        self.assertEqual(min(res0['profile_data']['wt']['zt_times']), 0.0)
        z8 = np.array(res8['profile_data']['wt']['zt_times'])
        m8 = np.array(res8['profile_data']['wt']['activity_matrix'], dtype=float)
        self.assertTrue(np.all(np.isnan(m8[0, z8 < 16.0])))
        self.assertFalse(np.any(np.isnan(m8[0, z8 >= 16.0])))


class TestLocomotorGUI(unittest.TestCase):
    """B-22 (GUI side), offscreen."""

    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_lights_on_row_only_in_locomotor_module(self):
        from PySide6.QtWidgets import QFormLayout
        from ui.analysis_panel import AnalysisPanel
        p = AnalysisPanel()
        p._source_type = 'csv'
        p._update_module_combo_for_source('csv')

        def hidden():
            lay = p._params_layout
            for i in range(lay.rowCount()):
                it = lay.itemAt(i, QFormLayout.LabelRole)
                if it and it.widget() and it.widget().text() == 'Lights on (ZT0):':
                    return it.widget().isHidden()

        p._module_combo.setCurrentText('Locomotor Activity Analysis')
        self.assertFalse(hidden())
        p._module_combo.setCurrentText('Classical Rhythm Analysis')
        self.assertTrue(hidden())

    def test_threshold_curve_drawn_for_every_condition(self):
        from ui.results_panel import PlotCanvas
        periods = np.arange(18, 32.1, 0.5)
        chi = {c: {'periods': periods, 'qp': np.ones_like(periods), 'qp_sem': None,
                   'tau': None, 'tau_sem': None, 'tau_qp': None,
                   'threshold_curve': np.linspace(70, 110, len(periods)) + k,
                   'alpha': 0.05, 'phase_used': ph}
               for k, (c, ph) in enumerate([('a', 'DD'), ('b', 'all')])}
        cv = PlotCanvas()
        cv.plot_chi_square_periodogram(chi, ['a', 'b'])
        ax = cv.fig.axes[0]
        dotted = [l for l in ax.get_lines() if l.get_linestyle() == ':']
        self.assertEqual(len(dotted), 2)
        self.assertIn('some individuals on full recording', ax.get_title())

    def test_actogram_is_drawn_in_zt(self):
        """Bars sit at their ZT (no shift by the lights-on hour), so each row
        is in chronological order."""
        from ui.results_panel import PlotCanvas
        zt = np.arange(0, 24, 1.0)
        row = np.zeros(48)
        row[20] = 5.0                              # ZT 20 of the row's day
        data = {'c': {'days': [1, 2], 'zt_times': zt.tolist(), 'matrix': [row.tolist()]}}
        cv = PlotCanvas()
        cv.plot_actogram(data, ['c'], lights_on_hour=8)
        ax = cv.fig.axes[0]
        xs = [p.get_x() for p in ax.patches if p.get_height() > 0 and p.get_width() == 1.0]
        self.assertEqual(xs, [20.0])               # old code drew it at clock 4 h
        self.assertEqual(ax.get_xlabel(), 'ZT (h)')


if __name__ == '__main__':
    unittest.main()
