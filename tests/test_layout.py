"""Guards on the repository layout.

These catch the class of breakage that only shows up at PyInstaller build time
or when a validation script is re-run months later: a path that was moved in
the tree but not in the code that points at it.
"""
import ast
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPECS = ('ChronoScope.spec', 'ChronoScope_mac.spec')

# Folder names retired during the 2026-09 restructure. If one of these comes
# back in a path string, something was moved without updating its references.
RETIRED_PATHS = (
    'core/models_meta_classifier',
    'core/RhythmCount_docs',
    'core/CosinorPy_docs',
    'core/circacompare_docs',
    'core/rhythm_docs',
    'Rosbash_data/',
)


def _spec_datas(spec_text):
    """Extract the (source, destination) pairs from a spec's `datas` list."""
    match = re.search(r'^datas = (\[.*?^\])', spec_text, re.S | re.M)
    assert match, 'no `datas = [...]` block found in spec'
    return ast.literal_eval(match.group(1))


class TestBundledDataPaths(unittest.TestCase):
    """Everything the .spec files promise to bundle must actually exist."""

    def test_spec_data_sources_exist(self):
        for spec in SPECS:
            spec_path = PROJECT_ROOT / spec
            self.assertTrue(spec_path.is_file(), f'{spec} is missing')
            for source, _dest in _spec_datas(spec_path.read_text(encoding='utf-8')):
                with self.subTest(spec=spec, source=source):
                    self.assertTrue(
                        (PROJECT_ROOT / source).exists(),
                        f'{spec} bundles "{source}", which does not exist',
                    )

    def test_model_artifacts_are_bundled(self):
        """The trained model must be inside a bundled folder, or the frozen app
        silently ships without CRS-AI."""
        for spec in SPECS:
            sources = [s for s, _ in _spec_datas((PROJECT_ROOT / spec).read_text(encoding='utf-8'))]
            with self.subTest(spec=spec):
                self.assertIn('core/models', sources)

    def test_bundled_model_dir_matches_runtime_lookup(self):
        """The destination inside the bundle must match what
        ConsensusClassifier.get_model_dir() looks for when frozen."""
        for spec in SPECS:
            datas = _spec_datas((PROJECT_ROOT / spec).read_text(encoding='utf-8'))
            dest = dict(datas)['core/models']
            with self.subTest(spec=spec):
                self.assertEqual(dest, 'core/models')


class TestNoRetiredPaths(unittest.TestCase):
    """No source file may still point at a folder that no longer exists."""

    # Only the directories that hold first-party source. docs/reference holds
    # upstream copies that legitimately mention their original layout, and this
    # file itself lists the retired names on purpose.
    SOURCE_DIRS = ('core', 'ui', 'utils', 'validation', 'scripts', 'examples')

    def _tracked_sources(self):
        for name in self.SOURCE_DIRS:
            for pattern in ('*.py', '*.spec'):
                for path in (PROJECT_ROOT / name).rglob(pattern):
                    if '__pycache__' in path.parts:
                        continue
                    yield path
        for pattern in ('*.py', '*.spec'):
            yield from PROJECT_ROOT.glob(pattern)

    def test_no_references_to_retired_folders(self):
        offenders = []
        for path in self._tracked_sources():
            text = path.read_text(encoding='utf-8', errors='ignore')
            for retired in RETIRED_PATHS:
                if retired in text:
                    offenders.append(f'{path.relative_to(PROJECT_ROOT)} -> {retired}')
        self.assertEqual(offenders, [], 'stale path references:\n' + '\n'.join(offenders))


class TestCoreIsAnalysisOnly(unittest.TestCase):
    """core/ holds runtime analysis code -- not reports, figures or datasets."""

    def test_no_report_or_figure_artifacts_in_core(self):
        strays = [p.relative_to(PROJECT_ROOT).as_posix()
                  for p in (PROJECT_ROOT / 'core').rglob('*')
                  if p.suffix in {'.txt', '.png', '.csv', '.npy'}
                  and 'vendor' not in p.parts]
        self.assertEqual(strays, [], f'analysis artifacts left in core/: {strays}')

    def test_validation_scripts_live_under_validation(self):
        strays = [p.name for p in (PROJECT_ROOT / 'core').rglob('validate_*.py')]
        self.assertEqual(strays, [], f'validation scripts still in core/: {strays}')


if __name__ == '__main__':
    unittest.main()
