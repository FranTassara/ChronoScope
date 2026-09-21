"""The public surface of each package imports cleanly and is complete."""
import importlib
import unittest

from tests.helpers import PROJECT_ROOT  # noqa: F401  (inserts project root on sys.path)


class TestPackageImports(unittest.TestCase):

    def test_core_exports_resolve(self):
        core = importlib.import_module('core')
        missing = [name for name in core.__all__ if not hasattr(core, name)]
        self.assertEqual(missing, [], f'core.__all__ advertises missing names: {missing}')

    def test_analysis_modules_import(self):
        for name in ('core.analysis_engine', 'core.cosinor_analysis',
                     'core.circacompare_analysis', 'core.rhythm_analysis',
                     'core.rhythmcount_analysis', 'core.meta_classifier',
                     'core.feature_extraction', 'core.preprocessing',
                     'core.visualization_circadian_metrics'):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_vendored_rhythmcount_is_importable(self):
        """RhythmCount is vendored code, not documentation: if this import
        breaks, the count-data module silently disables itself."""
        for name in ('core.vendor.rhythmcount.data_processing',
                     'core.vendor.rhythmcount.helpers',
                     'core.vendor.rhythmcount.plot'):
            with self.subTest(module=name):
                importlib.import_module(name)

        from core.rhythmcount_analysis import RHYTHMCOUNT_AVAILABLE
        self.assertTrue(RHYTHMCOUNT_AVAILABLE,
                        'RhythmCount reported unavailable -- the vendored import failed')

    def test_utils_loaders_import(self):
        for name in ('utils.data_loader', 'utils.dam_loader', 'utils.awd_loader',
                     'utils.rosbash_loader', 'utils.export'):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_cli_is_importable_without_qt(self):
        """cli.py must stay GUI-free so it can run headless."""
        import ast
        source = (PROJECT_ROOT / 'cli.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split('.')[0])
        self.assertNotIn('PySide6', imported)
        self.assertNotIn('ui', imported)


if __name__ == '__main__':
    unittest.main()
