"""Validate that CRS-AI is gated on series length, not only on loader type.

Reviewer 1 (Minor point 3) asked whether CRS-AI, being unavailable for DAM/AWD
recordings because of their length and sampling density, is likewise unavailable
for long-term bioluminescence traces. Inspecting the application showed it was
not: the module was withheld purely on data-loader type, so a multi-day,
densely-sampled trace loaded through the generic CSV pathway reached the model
unchecked and returned a well-formed but unreliable probability.

The gate now also consults series length against the model's 6-48-timepoint
training window (core.meta_classifier.timepoint_applicability). This script
checks the decision boundary, confirms the bioluminescence case is now caught,
and shows that the previous loader-only rule would have let it through.

Run:  python validation/methods/validate_crs_ai_length_gating.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.meta_classifier import (  # noqa: E402
    APPLICABLE_MAX_TIMEPOINTS,
    TRAINING_MAX_TIMEPOINTS,
    TRAINING_MIN_TIMEPOINTS,
    timepoint_applicability,
)
from ui.analysis_panel import AnalysisPanel  # noqa: E402


class _StubLoader:
    def __init__(self, n_timepoints):
        self._n = n_timepoints

    def get_timepoints(self):
        if self._n is None:
            raise RuntimeError('no timepoints available')
        return list(range(self._n))


class _NoTimepointsLoader:
    """A loader that predates get_timepoints(); must not be gated on length."""


class _StubPanel:
    def __init__(self, loader):
        self._loader = loader

    _loaded_timepoint_count = AnalysisPanel._loaded_timepoint_count
    _crs_ai_availability = AnalysisPanel._crs_ai_availability


def old_rule(source_type):
    """The behaviour before this fix: loader type only."""
    return source_type not in ('dam', 'awd')


def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ''))
    return bool(condition)


CASES = [
    # label,                                    n_tp,  source, expect_available, expect_note
    ('typical transcriptomic series (24 tp)',      24, 'csv',  True,  False),
    ('minimum of training window (6 tp)',           6, 'csv',  True,  False),
    ('below training window (5 tp)',                5, 'csv',  False, False),
    ('top of training window (48 tp)',             48, 'csv',  True,  False),
    ('just above window (49 tp)',                  49, 'csv',  True,  True),
    ('twice the window (96 tp)',                   96, 'csv',  True,  True),
    ('beyond twice the window (97 tp)',            97, 'csv',  False, False),
    ('7-day bioluminescence, 10-min bins (1008)', 1008, 'csv',  False, False),
    ('scRNA-seq, 6 timepoints',                     6, 'rosbash', True, False),
    ('DAM recording',                            2880, 'dam',  False, False),
    ('AWD recording',                            2880, 'awd',  False, False),
]


def main():
    print('=' * 78)
    print('CRS-AI availability gated on series length -- validation')
    print(f'training window {TRAINING_MIN_TIMEPOINTS}-{TRAINING_MAX_TIMEPOINTS} '
          f'timepoints; withheld above {APPLICABLE_MAX_TIMEPOINTS}')
    print('=' * 78)
    ok = True

    print('\n1. APPLICABILITY CLASSIFIER\n' + '-' * 78)
    for n, expected in ((None, 'unknown'), (0, 'unknown'), (5, 'too_short'),
                        (6, 'ok'), (48, 'ok'), (49, 'marginal'),
                        (96, 'marginal'), (97, 'too_long')):
        got = timepoint_applicability(n)
        ok &= check(f'{str(n):>5} timepoints -> {expected}', got == expected, f'got {got}')

    print('\n2. MODULE AVAILABILITY\n' + '-' * 78)
    for label, n_tp, source, expect_avail, expect_note in CASES:
        panel = _StubPanel(_StubLoader(n_tp))
        avail, note = panel._crs_ai_availability(source)
        ok &= check(label,
                    avail == expect_avail and bool(note) == expect_note,
                    f"available={avail}, note={'yes' if note else 'no'}")

    print('\n3. THE CASE THE REVIEWER ASKED ABOUT\n' + '-' * 78)
    panel = _StubPanel(_StubLoader(1008))
    avail, _ = panel._crs_ai_availability('csv')
    ok &= check('long bioluminescence trace as CSV is now withheld', avail is False)
    ok &= check('the previous loader-only rule would have allowed it',
                old_rule('csv') is True,
                'confirms the gap this fix closes')

    print('\n4. NO REGRESSION FOR LOADERS THAT CANNOT REPORT LENGTH\n' + '-' * 78)
    panel = _StubPanel(_NoTimepointsLoader())
    avail, note = panel._crs_ai_availability('csv')
    ok &= check('length unknown leaves availability unchanged', avail is True and not note)
    panel = _StubPanel(_StubLoader(None))  # get_timepoints raises
    avail, _ = panel._crs_ai_availability('csv')
    ok &= check('a failing get_timepoints() does not hide the module', avail is True)
    panel = _StubPanel(None)
    avail, _ = panel._crs_ai_availability('csv')
    ok &= check('no loader at all does not hide the module', avail is True)

    print('\n' + '=' * 78)
    print('RESULT:', 'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    print('=' * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
