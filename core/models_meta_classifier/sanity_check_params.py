"""
Smoke test for _resolve_params policy and end-to-end extract_features.

Verifies:
  1. Defaults (no parameters) → trained-window settings.
  2. User period_range fully inside window → honored, no warning.
  3. User period_range partly outside → clipped + warning.
  4. User period_range fully outside → falls back to defaults + warning.
  5. User n_harmonics → honored freely.
  6. extract_features returns all 11 model features under both default
     and custom parameters.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extraction import extract_features, _resolve_params, FEATURE_NAMES


def _show(p):
    print(f"  jtk_periods     = {p['jtk_periods']}")
    print(f"  cosinor first 5 = {list(p['cosinor_periods'])[:5]} ... last = {list(p['cosinor_periods'])[-1]}")
    print(f"  ls_range        = {p['ls_range']}")
    print(f"  n_harmonics     = {p['n_harmonics']}")


def _capture_warnings(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = fn()
    return result, [str(wi.message) for wi in w]


print("=" * 60)
print("TEST 1: defaults (no parameters)")
print("=" * 60)
p, w = _capture_warnings(lambda: _resolve_params(None))
assert not w, f"unexpected warnings: {w}"
_show(p)
assert p['jtk_periods'] == list(range(20, 29)), p['jtk_periods']
assert p['ls_range'] == (18.0, 32.0)
assert p['n_harmonics'] == 2

print()
print("=" * 60)
print("TEST 2: period_range=(22, 26) fully inside window")
print("=" * 60)
p, w = _capture_warnings(lambda: _resolve_params({'period_range': (22, 26)}))
assert not w, f"unexpected warnings: {w}"
_show(p)
assert p['jtk_periods'] == [22, 23, 24, 25, 26]
assert p['ls_range'] == (22.0, 26.0)

print()
print("=" * 60)
print("TEST 3: period_range=(16, 30) — clips low end")
print("=" * 60)
p, w = _capture_warnings(lambda: _resolve_params({'period_range': (16, 30)}))
assert len(w) == 1 and 'clipping' in w[0].lower(), f"expected clip warning, got: {w}"
print(f"  WARN: {w[0]}")
_show(p)
assert p['ls_range'] == (18.0, 30.0)
assert p['jtk_periods'][0] == 18

print()
print("=" * 60)
print("TEST 4: period_range=(6, 12) — fully outside, falls back")
print("=" * 60)
p, w = _capture_warnings(lambda: _resolve_params({'period_range': (6, 12)}))
assert len(w) == 1 and 'outside' in w[0].lower(), f"expected fallback warning, got: {w}"
print(f"  WARN: {w[0]}")
_show(p)
assert p['ls_range'] == (18.0, 32.0)
assert p['jtk_periods'] == list(range(20, 29))

print()
print("=" * 60)
print("TEST 5: n_harmonics=3")
print("=" * 60)
p, w = _capture_warnings(lambda: _resolve_params({'n_harmonics': 3}))
assert not w
assert p['n_harmonics'] == 3
print(f"  n_harmonics = {p['n_harmonics']}")

print()
print("=" * 60)
print("TEST 6: extract_features end-to-end on 24h cosine + noise")
print("=" * 60)
np.random.seed(0)
t = np.arange(0, 48, 2.0)
y = 10 + 2 * np.cos(2 * np.pi * t / 24) + np.random.normal(0, 0.3, len(t))

f_default = extract_features(t, y)
f_custom = extract_features(t, y, parameters={'period_range': (22, 26), 'n_harmonics': 3})

assert all(name in f_default for name in FEATURE_NAMES)
assert all(name in f_custom for name in FEATURE_NAMES)

print(f"  {'feature':<22} {'default':>12} {'custom (22-26)':>14}")
for name in FEATURE_NAMES:
    print(f"  {name:<22} {f_default[name]:>12.4f} {f_custom[name]:>14.4f}")

print()
print("All sanity checks passed.")
