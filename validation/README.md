# Validation

Scripts and reports that back the claims made in the manuscript. They are not
part of the application and are not bundled into the packaged builds.

Every script writes its report next to itself, and reads the trained model from
`core/models/`. Run them from the repository root, e.g.:

```bash
python validation/crs_ai/validate_simulated_ground_truth.py
python validation/methods/validate_chi_square_periodogram_fix.py
```

## `crs_ai/` — the consensus classifier

| File | What it does |
|---|---|
| `train_consensus_model.py` | Trains the random forest and writes `core/models/consensus_rf_model.pkl` + `feature_names.json`; the holdout arrays and `training_report.txt` stay here. |
| `validate_external_holdout_gse20635.py`, `..._gse3416.py`, `..._gse37332.py` | Evaluate the shipped model on external GEO datasets it never saw in training. |
| `validate_nested_feature_selection.py` | Repeats feature selection inside each cross-validation fold, to check the headline holdout figure is not inflated by selection on the full dataset. |
| `validate_simulated_ground_truth.py` | Scores the model against simulated series with known rhythmicity across a range of noise levels. |
| `validate_period_range_override.py` | Confirms user-supplied period ranges propagate into feature extraction, and that the default settings reproduce the cached holdout features. |
| `sanity_check_*.py` | Fast checks on feature computation and parameter propagation. |

Re-running the training or the external holdouts needs the raw datasets under
`training_data_meta_classifier/`, which are not tracked in the repository
because of their size.

## `methods/` — individual analysis methods

| File | What it does |
|---|---|
| `validate_chi_square_periodogram_fix.py` | Monte-Carlo calibration of the corrected chi-square periodogram significance threshold against the old one. |
| `validate_crs_ai_length_gating.py` | Checks the 6–48 timepoint applicability window is enforced where CRS-AI is offered. |
| `validate_locomotor_tabular_input.py` | Checks locomotor metrics computed from tabular input match those from the native DAM/AWD path. |
