# -*- mode: python ; coding: utf-8 -*-
"""
ChronoScope - PyInstaller Build Specification (macOS)
======================================================

Genera la aplicación en modo --onedir (más estable con PySide6) y la
empaqueta como .app nativo de macOS con BUNDLE().

IMPORTANTE: PyInstaller no hace cross-compilation. Este .spec se debe
correr en una Mac (con el venv del proyecto activado ahí), no desde Windows.

Uso (en macOS):
    pyinstaller --clean ChronoScope_mac.spec

Salida:
    dist/ChronoScope.app          (app para distribuir / abrir con Finder)
    dist/ChronoScope/             (carpeta onedir equivalente, sin bundle)

Para distribuir: comprimir dist/ChronoScope.app en un .zip, o generar un .dmg.

Ícono:
    paper/logo/logo.icns (ver BUNDLE() más abajo).
"""

# ---------------------------------------------------------------------------
# Imports de utilidades de PyInstaller
# ---------------------------------------------------------------------------
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# ---------------------------------------------------------------------------
# Hidden imports: módulos que PyInstaller no detecta por carga dinámica
# ---------------------------------------------------------------------------
hidden_imports = [
    # scikit-learn (carga dinámica interna con joblib)
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.impute',
    'sklearn.impute._base',
    'sklearn.preprocessing',
    'sklearn.pipeline',
    'sklearn.model_selection',
    'sklearn.metrics',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors._partition_nodes',
    'sklearn.tree._utils',

    # joblib (usado por sklearn para serializar el modelo .pkl)
    'joblib',
    'joblib.externals.loky',
    'joblib.externals.cloudpickle',

    # scipy (submódulos de uso indirecto)
    'scipy.stats',
    'scipy.signal',
    'scipy.interpolate',
    'scipy.optimize',
    'scipy.linalg',
    'scipy.special',

    # statsmodels
    'statsmodels.tsa',
    'statsmodels.stats',
    'statsmodels.formula',
    'statsmodels.formula.api',

    # matplotlib - backends (Qt para render, file backends para exportar)
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_ps',   # EPS / PS
    'matplotlib.backends.backend_svg',  # SVG

    # PyWavelets
    'pywt',
    'pywt._extensions._cwt',

    # h5py (dataset Rosbash scRNA-seq)
    'h5py',
    'h5py._hl',
    'h5py.defs',
    'h5py.utils',
    'h5py.h5ac',
    'h5py.h5z',

    # Exportación
    'openpyxl',
    'xlsxwriter',

    # CosinorPy — detectado automáticamente por PyInstaller vía análisis de imports

    # Módulos del proyecto (por si el análisis dinámico los pierde)
    'core',
    'core.analysis_engine',
    'core.circacompare_analysis',
    'core.cosinor_analysis',
    'core.feature_extraction',
    'core.meta_classifier',
    'core.rhythm_analysis',
    'ui',
    'ui.main_window',
    'ui.data_panel',
    'ui.analysis_panel',
    'ui.results_panel',
    'utils',
    'utils.data_loader',
    'utils.dam_loader',
    'utils.export',
    'utils.rosbash_loader',
]

# ---------------------------------------------------------------------------
# Archivos de datos a incluir en el bundle
# Formato: (origen, destino_dentro_del_exe)
# ---------------------------------------------------------------------------
datas = [
    # Modelo Random Forest entrenado + metadatos de features
    ('core/models_meta_classifier', 'core/models_meta_classifier'),

    # Ejemplos de datos (cargados desde el GUI)
    ('examples', 'examples'),

    # Dataset scRNA-seq de Rosbash (archivo HDF5 ~pesado)
    ('Rosbash_data/rosbash_processed.h5', 'Rosbash_data'),
]

# ---------------------------------------------------------------------------
# Análisis de dependencias
# ---------------------------------------------------------------------------
a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={
        'matplotlib': {
            'backends': ['qtagg', 'pdf', 'ps', 'svg'],
        },
    },
    runtime_hooks=[],
    excludes=[
        # Scripts de entrenamiento, no necesarios en runtime
        'generate_training_data',
        'generate_real_training_data',
        'generate_synthetic_data',
        'train_consensus_model',
        # Herramientas de desarrollo innecesarias en el exe
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'tkinter',
        # Otros bindings de Qt instalados en el entorno: forzar PySide6 únicamente
        'PyQt5',
        'PyQt6',
        'PySide2',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# Ejecutable principal
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # onedir: los binarios van en COLLECT
    name='ChronoScope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,              # sin ventana de consola (app GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,           # arquitectura nativa de la Mac que compila (arm64 o x86_64)
    codesign_identity=None,     # setear si vas a firmar con un certificado de Apple Developer
    entitlements_file=None,
)

# ---------------------------------------------------------------------------
# Colección (modo onedir)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ChronoScope',
)

# ---------------------------------------------------------------------------
# Bundle .app nativo de macOS
# ---------------------------------------------------------------------------
app = BUNDLE(
    coll,
    name='ChronoScope.app',
    icon='paper/logo/logo.icns',
    bundle_identifier='ar.gov.leloir.chronoscope',
    info_plist={
        'NSHighResolutionCapable': True,
        'NSPrincipalClass': 'NSApplication',
        # UI stylesheet assumes a light palette; force Light Appearance so
        # macOS Dark Mode doesn't flip text to white on the hardcoded white
        # widget backgrounds.
        'NSRequiresAquaSystemAppearance': True,
    },
)
