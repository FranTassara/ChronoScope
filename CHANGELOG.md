# Changelog

All notable changes to ChronoScope are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `cli.py`: headless batch interface to the analysis engine, for scripted
  pipelines that do not need the Qt GUI. Covers the tabular-data methods
  reachable through `AnalysisEngine.run_analysis`.
- CRS-AI applicability window: series length is now classified against the
  6–48 timepoint training range (`timepoint_applicability`), so the module is
  flagged or withheld instead of silently extrapolating.
- `validation/`: the validation scripts and reports supporting the manuscript —
  external holdouts (GSE20635, GSE3416, GSE37332), nested feature-selection
  validation, simulated ground truth, parameter-override checks, and the
  method-level validations (chi-square periodogram fix, CRS-AI length gating,
  locomotor tabular input).
- `tests/`: a smoke-test suite runnable with `python -m unittest discover tests`
  — no extra dependencies. Covers imports, model/feature consistency,
  rhythm detection versus noise, the vendored RhythmCount path, and guards on
  the repository layout and the PyInstaller bundle contents.
- `CITATION.cff`, so GitHub renders a "Cite this repository" entry.

### Fixed
- Chi-square periodogram: the significance threshold was too low by roughly a
  factor of `n_slots`, which made essentially any input come out "significantly
  rhythmic". Corrected following Tackenberg & Hughey (2021), *PLOS Comput Biol*
  17:e1008567. See `validation/methods/chi_square_periodogram_fix_validation.txt`.
- Paper figure scripts and validation scripts now resolve the model and holdout
  artifacts through the restructured paths.

### Changed
- Repository restructure:
  - `core/` now holds only runtime analysis code, plus `core/models/` (the
    trained CRS-AI artifacts) and `core/vendor/` (third-party code that is
    imported, not documentation).
  - `core/RhythmCount_docs/` → `core/vendor/rhythmcount/`. It was never
    documentation: `rhythmcount_analysis.py` imports it at runtime.
  - `core/models_meta_classifier/` split into `core/models/` (the `.pkl` and
    `feature_names.json` that ship with the app) and `validation/crs_ai/`
    (training script, sanity checks, holdout arrays and reports). The frozen
    app no longer bundles training scripts and reports.
  - Reference material (`CosinorPy_docs`, `circacompare_docs`, `rhythm_docs`)
    moved out of the import path to `docs/reference/`.
  - `Rosbash_data/` split into `data/rosbash/` (the HDF5 dataset) and
    `scripts/` (the one-off preprocessing script).
- `ChronoScope.spec` / `ChronoScope_mac.spec`: data paths updated for the new
  layout, and the hidden-import list extended to cover the modules that were
  relying on PyInstaller's static analysis alone.

### Removed
- ~1850 lines of superseded CosinorPy dispatch methods in
  `core/analysis_engine.py` (`_run_cosinorpy_single`, `_multi`, `_comparison`,
  `_limorhyde`, `_compare_nonlinear`, `_compare_all`, `_compare_all_limo`,
  `_population`, `_count`, `_nonlinear`, `_periodogram`). They had no call
  sites; `run_analysis` has dispatched to the `*_new` implementations since the
  CosinorPy refactor. Analysis output is unchanged.
- `Rosbash_data/rosbash_data_loader.py`, an unused earlier version of
  `utils/rosbash_loader.py`.

## [1.0.0] - 2026-07-20

Initial public release: multi-method circadian rhythm analysis in a PySide6
desktop application, with packaged builds for Windows and macOS.

- CosinorPy, CircaCompare and RhythmCount modules.
- Classical rhythm analysis: JTK_CYCLE, ARSER-style JTK, Cosine-Kendall,
  cosinor OLS, harmonic cosinor, Fourier F24, Lomb–Scargle, spectral analysis,
  CWT and linear mixed-effects models.
- CRS-AI: random-forest consensus rhythmicity score over the individual methods.
- Locomotor activity analysis for DAM and AWD recordings.
- Single-cell RNA-seq support for the Rosbash CLK856 clock-neuron dataset.

[Unreleased]: https://github.com/FranTassara/ChronoScope/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FranTassara/ChronoScope/releases/tag/v1.0.0
