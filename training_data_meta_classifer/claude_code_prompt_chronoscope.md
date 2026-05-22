# ChronoScope CRS-AI: Add Drosophila & Human Datasets + External LODO Validation

## Goal

Extend the ChronoScope meta-classifier training pipeline by adding two new biological datasets and an external leave-one-dataset-out (LODO) validation step. Currently the pipeline uses only mouse liver data (GSE11923 + GSE11516). The two additions bring in *Drosophila* and human, broadening biological diversity for stronger generalization claims in the planned JBR submission.

**Scope of this work:**
- **(a)** Implement two new dataset loaders in `training_data_meta_classifer/generate_real_training_data.py`.
- **(b)** Integrate them into `core/models_meta_classifier/train_consensus_model.py`.
- **(c)** Implement a new external LODO validation script using a held-out *Drosophila* dataset never seen during training.
- **(d)** Retrain the model end-to-end and regenerate the training report.

---

## Context

The current CRS-AI is a calibrated Random Forest with 11 features extracted from time-series expression data. It uses GroupShuffleSplit by gene to prevent leakage, and `class_weight='balanced'` to handle imbalance. The existing `generate_from_geo()` function in `generate_real_training_data.py` is the template for adding new datasets — it downloads a GEO Series Matrix, parses probes → genes, applies label heuristics from a `KNOWN_CIRCADIAN_GENES` set + a `NON_RHYTHMIC_GENES` set + optionally a BioCycle XLSX lookup. The output is a list of instance metadata dicts + a parallel list of long-format DataFrames.

Each new dataset must follow the **same instance format** as the existing ones:

```python
metadata = {
    'instance_id': int,
    'variable': str,           # 'var_<id>'
    'signal_type': str,        # e.g., 'real_per_LNv_GSE77451'
    'is_rhythmic': 0 | 1,
    'n_timepoints': int,
    'n_replicates': int,
    'sampling_hours': float,
    'snr': 0.0,                # constant for biological
    'period': 24.0 if rhythmic else 0.0,
    'has_outliers': False,
    'source': 'biological',
    'gene': str,               # bare gene symbol — used as grouping key
    # New optional field, only for GSE77451:
    'cell_type': str,          # 'LNv' | 'LNd' | 'DN1' | 'TH'
}
```

DataFrames are long-format with columns: `time`, `condition='control'`, `replicate='rep1'`, `var_<id>`.

---

## Prerequisites (already in place — confirm before starting)

These three files must exist at the paths below. They were downloaded from the original publications. **If they are missing, stop and ask the user — do not attempt to re-derive labels from raw data, that would be circular for this design.**

```
training_data_meta_classifer/abruzzi_2017_cycling.xlsx
    # PLOS Genetics 2017 S3 File — https://doi.org/10.1371/journal.pgen.1006613.s003
    # Sheets: 'TH_cyclers', 'LNv_cyclers', 'LNd_cyclers', 'DN1_cyclers', 'NOTES'
    # Column A = gene symbol; expression columns at ZT2,6,10,14,18,22 × 2 reps
    # (DN1 sheet uses ZT3,7,11,15,19,23). 'cycling?' column = 'HC-cycler' or 'LC-cycler'.

training_data_meta_classifer/moller_levet_2013_circadian.xlsx
    # PNAS 2013 Dataset S2 — https://doi.org/10.1073/pnas.1217154110
    # Sheet 'Main_list', 3741 rows. Columns include 'Identifier (Gene/ Probe/ Accession)',
    # 'Sleep Condition effect', 'Circadian in Control', 'Circadian in Sleep Restriction',
    # and trend flags.
```

Verify these exist as a first action. The existing BioCycle file at `training_data_meta_classifer/rhythmicdb_query_bioCycle.xlsx` is the template for how to load XLSX label sources.

---

## Phase 1: Add GSE77451 (*Drosophila*, Abruzzi et al. 2017)

### Design

GSE77451 contains bulk RNA-seq of four FACS-sorted *Drosophila* neuron populations: three clock neuron groups (LNv, LNd, DN1) and one non-clock outgroup (TH = tyrosine-hydroxylase neurons). Sampling is every 4 h at ZT 2/6/10/14/18/22 for LNv/LNd/TH, and ZT 3/7/11/15/19/23 for DN1, with 2 biological replicates per timepoint. Total: 4 cell types × 6 timepoints × 2 reps = 48 samples.

**Label strategy (per-cell-type, option iii):**
- **Positives (label=1):** A gene's instance for cell type C is rhythmic if and only if the gene is listed as `HC-cycler` in the C sheet of `abruzzi_2017_cycling.xlsx`. Use HC only — skip LC-cyclers (lower confidence, would muddy training labels). Restrict positives to LNv, LNd, DN1 only.
- **Negatives (label=0):**
    - Fly housekeeping genes (defined below) for ALL four cell types (LNv, LNd, DN1, TH).
    - Fly core clock genes for TH only (TH lacks the canonical molecular clock, so clock genes should not cycle there — biologically validated negative).
- **Skip:** TH HC-cyclers (ambiguous biology — the paper itself flags TH as a non-circadian outgroup).

**Capping:** Up to 200 HC-cycler positive instances per cell type (LNv, LNd, DN1), sampled with the lowest `JTK p-value` first (most rhythmic) to deterministically prioritize strongest signals. This gives up to 600 positives; combined with ~70–90 negatives, the dataset contributes ~670–690 instances. Class imbalance is handled downstream by `class_weight='balanced'`.

**Replicates:** Use the `extract_timepoints_from_samples` + `_extract_features_for_instances` pattern already in `train_consensus_model.py` — biological replicates at the same ZT are averaged implicitly via the `unique_times` deduplication. This matches GSE11923 behavior.

### Gene lists to add as constants in `generate_real_training_data.py`

Add new module-level sets next to the existing `KNOWN_CIRCADIAN_GENES` / `NON_RHYTHMIC_GENES`. Use **FlyBase symbols, case-sensitive** as published (most are lowercase, some mixed):

```python
# Drosophila melanogaster core clock genes
KNOWN_CIRCADIAN_GENES_FLY = {
    'per', 'tim', 'Clk', 'cyc', 'vri', 'Pdp1', 'cry', 'cwo',
    'Pdf', 'sgg', 'dco', 'nmo', 'jet', 'twins', 'ck1', 'NPF',
    'shaggy', 'dbt',  # dbt is alternative symbol for dco
}

# Drosophila housekeeping genes — should always be non-rhythmic
NON_RHYTHMIC_GENES_FLY = {
    'RpL32', 'Act5C', 'Act5c', 'Act88F', 'alphaTub84B', 'αTub84B',
    'Gapdh1', 'Gapdh2', 'Sdha', 'eIF1A', 'eEF1alpha1', 'eEF1α1',
    'Rpl13', 'Rps17', 'Tbp', 'GstD1', 'Hsc70-4', 'Hsc70Cb',
    'CG8187', 'CG7434',  # common ribosomal/structural
}
```

### Implementation: new function `generate_from_GSE77451(...)`

Add a new function in `generate_real_training_data.py` (do not modify `generate_from_geo`; build alongside it). Signature roughly:

```python
def generate_from_GSE77451(
    abruzzi_xlsx_path: Path,
    geo_cache_dir: Path,
    starting_instance_id: int,
    max_positives_per_cell_type: int = 200,
    hc_only: bool = True,
    seed: int = 42,
) -> tuple[list[dict], list[pd.DataFrame]]:
    ...
```

Steps:

1. **Download GSE77451 series matrix** using the existing `download_geo_series_matrix('GSE77451', geo_cache_dir)`. If this helper currently hardcodes anything, generalize it.
2. **Parse the series matrix.** Extract per-sample metadata: `geo_accession`, `cell_type`, `timepoint_zt`, `replicate`. The `!Sample_characteristics_ch1` or `!Sample_title` fields encode these — verify by printing a few samples. Build a `samples_df` DataFrame with columns `[sample_id, cell_type, zt, rep, expression_vector]`.
3. **Map probes/transcripts to gene symbols.** Use the platform annotation file (GPL accession found in the series matrix header). Build a `probe_to_gene` dict. If multiple probes map to one gene, average the expression vectors per timepoint (existing pattern in `generate_from_geo`).
4. **Load Abruzzi S3** with a new helper, e.g.:

   ```python
   def load_abruzzi_cycling_labels(xlsx_path: Path, hc_only: bool = True) -> dict[str, set[str]]:
       """Returns {'LNv': {gene1, gene2, ...}, 'LNd': {...}, 'DN1': {...}, 'TH': {...}}."""
   ```
   Read each cyclers sheet, filter by `cycling?` column (strip whitespace — values are `'HC-cycler '` with trailing space, observed). Return symbol sets per cell type. Note that gene-symbol column name varies slightly across sheets (`'symbol (HC cyclers in yellow)'` vs `'symbol (yellow=HC cycler)'`) — match by position (column 0) or substring `'symbol'`.

5. **Generate per-(gene, cell_type) instances.** For each cell type in `['LNv', 'LNd', 'DN1', 'TH']`:
    - Subset `samples_df` to that cell type (12 samples = 6 ZTs × 2 reps).
    - For each gene in the expression matrix with sufficient data:
        - Decide label per the **Label strategy** above.
        - Build a long-format DataFrame: rows = (zt, replicate, expression). Use `condition='control'` and `replicate=f'rep{r}'`.
        - Append metadata dict and DataFrame.

    Set `metadata['cell_type']` to the cell type and `metadata['signal_type']` to `f'real_{gene}_{cell_type}_GSE77451'`. Use the bare gene symbol in `metadata['gene']` (no cell-type suffix) so that GroupShuffleSplit naturally groups all four instances of the same gene together — preventing intra-gene cross-contamination between train and holdout.

6. **Apply caps.** After collecting all candidate instances, cap positives per cell type to `max_positives_per_cell_type`. Selection: sort candidates by `JTK p-value` from the S3 table ascending, take the top N. This is deterministic and prioritizes the strongest rhythms.

7. **Return** `(metadata_list, dataframe_list)`. Each metadata dict's `instance_id` increments from `starting_instance_id`.

### Tests for Phase 1

Add an inline assertion block at the bottom of the function (or a small standalone smoke test) that prints:
- Total instances generated, broken down by cell type and label.
- Confirmation that no instance has `gene` that appears in both labels (sanity check on label conflicts: a clock gene labeled 1 in LNv and 0 in TH is correct and expected; flag any *within-cell-type* contradictions).
- Confirmation that ≥80% of fly housekeeping genes are present in negatives.
- Confirmation that no TH cyclers are in positives.

---

## Phase 2: Add GSE39445 (Human, Möller-Levet et al. 2013)

### Design

GSE39445 contains 438 whole-blood microarray samples from 26 subjects under two conditions: sleep-sufficient (control) and sleep-restricted. Each subject was sampled every 3 h across an extended-wakefulness period (~33 h). The custom Agilent platform is `GPL15331`.

**Label strategy (option iii):**

Use `moller_levet_2013_circadian.xlsx` (`Main_list` sheet) classification columns. For each gene that maps to the expression matrix:

- **Positive (label=1):**
    - `Circadian in Control == 1` AND `Circadian in Sleep Restriction == 1` → strong positive (~826 candidates). Use these first.
    - `Circadian in Control == 1` AND `Circadian in Sleep Restriction == 0` → hard positive (rhythmic with reduced amplitude under sleep restriction, ~1029 candidates). Use to fill up to cap.
    - Also force-include core human clock genes (`KNOWN_CIRCADIAN_GENES_HUMAN`, listed below) regardless of sd02 classification — defends against false negatives in known core clock genes (e.g., CRY2 is not flagged circadian in sd02 but is a canonical clock gene).
- **Negative (label=0):**
    - `Circadian in Control == 0` AND `Circadian in Sleep Restriction == 0` AND `Sleep Condition effect == 0` → ~633 clean candidates.
    - Also force-include human housekeeping genes (`NON_RHYTHMIC_GENES_HUMAN`, below).
- **Filter out probe IDs.** Rows where the `Identifier` starts with `A_` (Agilent probe IDs without gene mapping) — skip them. ~498 rows of 3741 are probe IDs.

**Instance generation:** Two strategies are acceptable, choose based on what cleanly integrates with existing code:

1. **(Preferred)** Treat all subjects' samples as a single pooled time-course per gene: merge across subjects, let the existing `unique_times` averaging handle per-timepoint pooling. Generates 1 instance per (gene, condition) — i.e., 2 instances per gene (control + sleep-restricted). For positives that are rhythmic in both conditions, both instances get label=1. For "hard positives" (rhythmic only in control), the control instance gets label=1, the SR instance is **not** included (avoids polluting positives with a known non-rhythmic copy).
2. **(Alternative)** One instance per (gene, subject, condition). Many more instances but introduces subject-correlation issues. Skip this unless (1) yields too few instances.

**Capping:** Up to 800 positives + 800 negatives total, matching the existing GSE11516 pattern. Selection order for positives: strong-rhythmic-in-both first, then hard-positives until cap. Selection for negatives: cleanest (all flags off) first.

### Gene lists to add as constants

```python
# Homo sapiens core clock genes (HGNC symbols)
KNOWN_CIRCADIAN_GENES_HUMAN = {
    'ARNTL', 'BMAL1', 'ARNTL2', 'BMAL2',
    'PER1', 'PER2', 'PER3',
    'CRY1', 'CRY2',
    'NR1D1', 'NR1D2',
    'DBP', 'TEF', 'HLF',
    'RORA', 'RORB', 'RORC',
    'CLOCK', 'NPAS2',
    'NFIL3', 'BHLHE40', 'BHLHE41',
    'CIART',
    'CSNK1D', 'CSNK1E', 'FBXL3',
    'PROK2', 'AVP', 'VIP',
}

# Homo sapiens housekeeping genes — should always be non-rhythmic in blood
NON_RHYTHMIC_GENES_HUMAN = {
    'ACTB', 'GAPDH', 'HPRT1', 'TBP', 'RPL13A', 'B2M', 'UBC',
    'PPIA', 'RPL32', 'EEF1A1', 'SDHA', 'HMBS', 'YWHAZ', 'PGK1',
    'TFRC', 'POLR2A', 'PSMB4', 'PSMB2', 'CHMP2A', 'EMC7',
    'GPI', 'C1orf43', 'REEP5', 'SNRPD3', 'VCP', 'VPS29',
}
```

### Implementation: new function `generate_from_GSE39445(...)`

```python
def generate_from_GSE39445(
    moller_xlsx_path: Path,
    geo_cache_dir: Path,
    starting_instance_id: int,
    max_per_class: int = 800,
    seed: int = 42,
) -> tuple[list[dict], list[pd.DataFrame]]:
    ...
```

Steps:

1. **Download GSE39445 series matrix.** This is a large file (~438 samples) — confirm the existing `download_geo_series_matrix` handles large series matrices; if there's a chunking/memory concern, log and adapt.
2. **Parse sample metadata.** Per sample, extract: `subject_id`, `condition` (`'control'` or `'sleep_restriction'`), `timepoint_hours`. These are typically in `!Sample_characteristics_ch1` — print a few raw entries to confirm the format before parsing.
3. **Map probes to genes.** Use platform annotation (`GPL15331`) to build `probe_to_gene`. Average duplicate probes per gene.
4. **Load Möller-Levet labels** with a new helper:

   ```python
   def load_moller_levet_labels(xlsx_path: Path) -> pd.DataFrame:
       """Returns DataFrame with columns: ['gene', 'circ_control', 'circ_sr', 'sleep_effect']."""
   ```
   Filter out probe-ID rows (`Identifier.startswith('A_')`).
5. **Generate instances.** For each gene with expression data + classification in Möller-Levet (or in forced clock/housekeeping lists):
    - Determine label per the strategy above.
    - For positives: generate a control-condition instance always; generate an SR-condition instance only if it remains positive under SR.
    - For negatives: generate both control and SR instances (since gene is non-rhythmic in both, this doubles the negative pool).
    - Build the long-format DataFrame, set `metadata['signal_type']` to `f'real_{gene}_{condition}_GSE39445'`. Use bare gene symbol in `metadata['gene']`.
6. **Cap and return.** Random sample (with `seed=42`) within each class if over cap. Strong positives prioritized.

### Tests for Phase 2

- Total instances by label after capping.
- Verify clock genes ARNTL, PER1, PER2, NR1D1 appear as positives.
- Verify housekeeping genes ACTB, GAPDH appear as negatives.
- Verify all `gene` values are uppercase (HGNC convention) — catches probe-ID leakage.

---

## Phase 3: Integrate into `train_consensus_model.py`

### Changes required

Locate the section where existing biological datasets are loaded (currently calls `generate_from_geo()` for GSE11923 and GSE11516). Add two new calls:

```python
# Dataset 3 — Drosophila clock neurons (Abruzzi 2017)
gse77451_meta, gse77451_dfs = generate_from_GSE77451(
    abruzzi_xlsx_path=Path('training_data_meta_classifer/abruzzi_2017_cycling.xlsx'),
    geo_cache_dir=GEO_CACHE_DIR,
    starting_instance_id=current_max_id + 1,
    max_positives_per_cell_type=200,
    hc_only=True,
    seed=42,
)

# Dataset 4 — Human blood (Möller-Levet 2013)
gse39445_meta, gse39445_dfs = generate_from_GSE39445(
    moller_xlsx_path=Path('training_data_meta_classifer/moller_levet_2013_circadian.xlsx'),
    geo_cache_dir=GEO_CACHE_DIR,
    starting_instance_id=current_max_id + 1,
    max_per_class=800,
    seed=42,
)
```

### Group-key safety

Since `GroupShuffleSplit` and `StratifiedGroupKFold` use `metadata['gene']` as the group key, confirm:
- `per` (fly) ≠ `Per1` (mouse) ≠ `PER1` (human) — case-sensitive string comparison ensures no cross-species collision. Document this assumption with a comment near the group construction.
- Within the fly dataset, all 4 cell-type instances of the same gene share the same group (correct — prevents gene leakage). The model can still learn cell-type-dependent rhythmicity because instances within a group are visible in either train or holdout (not both), but the cell-type pattern across the full training set is learnable.

### Report (Section 5.2) updates

The current report Section 5.2 documents Dataset 1 (GSE11923) and Dataset 2 (GSE11516). Add Datasets 3 and 4 in the same format:

```
Dataset 3: GSE77451 (Abruzzi et al. 2017, PLOS Genetics)
  Organism: Drosophila melanogaster
  Tissue: FACS-sorted clock neurons (LNv, LNd, DN1) + dopaminergic outgroup (TH)
  Platform: RNA-seq (Illumina HiSeq)
  Sampling: 4 cell types × 6 timepoints (4-hour intervals) × 2 biological replicates
  Label source: Abruzzi 2017 Supplementary S3 (HC-cyclers identified by JTK + F24)
  Label assignment:
    - Positives: per-cell-type HC-cyclers in LNv, LNd, DN1 (max 200/cell type)
    - Negatives: fly housekeeping genes (all cell types) + fly clock genes (TH only)
    - Excluded: TH HC-cyclers (ambiguous — non-circadian outgroup)
  Instances generated: <N>  (positives: <P>, negatives: <Q>)

Dataset 4: GSE39445 (Möller-Levet et al. 2013, PNAS)
  Organism: Homo sapiens
  Tissue: whole blood
  Platform: Custom Agilent microarray (GPL15331)
  Sampling: 26 subjects × ~17 timepoints (3-hour intervals) × 2 conditions
  Label source: Möller-Levet 2013 Supplementary Dataset S2 (Main_list)
  Label assignment:
    - Strong positives: rhythmic in both conditions
    - Hard positives: rhythmic in Control only (amplitude reduced under sleep restriction)
    - Negatives: not rhythmic + no sleep effect
    - Forced positives: KNOWN_CIRCADIAN_GENES_HUMAN (overrides null classification)
  Instances generated: <N>  (positives: <P>, negatives: <Q>)
```

### Report (Section 5.4) updates

The current Section 5.4 lists 20 mouse circadian + 16 mouse housekeeping genes used as label overrides. Extend it to list the fly + human equivalents added in Phase 1 + Phase 2.

---

## Phase 4: External LODO validation with Hughes 2012 (GSE29972)

### Design

Create a **new standalone script** `core/models_meta_classifier/validate_external_holdout_hughes2012.py`. It loads the trained model + scaler artifacts, downloads GSE29972 (Hughes et al. 2012, *Drosophila* brain bulk RNA-seq, LD timecourse), generates labeled instances using only `KNOWN_CIRCADIAN_GENES_FLY` + `NON_RHYTHMIC_GENES_FLY` (no per-cell-type complexity — this is brain whole-tissue), computes features for each instance, predicts probabilities, and reports metrics. The dataset must **never** appear in training.

This script does for fly what `validate_period_range_override.py` (referenced in Section 9.3 of the report) does for the period-range override sensitivity test: it's a sidecar validation. Use that script as a pattern.

### Implementation

Steps:

1. **Load artifacts** from `core/models_meta_classifier/`:
    - Trained model pipeline (calibrated RF + isotonic).
    - Feature list (11 features used by the model).
2. **Generate instances from GSE29972.** Reuse `generate_from_geo()` if it can be parameterized to use the fly gene lists, otherwise write a small wrapper `generate_from_GSE29972()` that:
    - Downloads GSE29972 series matrix.
    - Parses timepoints (likely ZT 0-24 every 4h; verify by inspecting raw metadata).
    - Maps probes/transcripts to fly gene symbols.
    - Labels: gene in `KNOWN_CIRCADIAN_GENES_FLY` → 1; gene in `NON_RHYTHMIC_GENES_FLY` → 0; everything else → exclude.
    - **No capping** — use all available labeled instances. Expect ~20–40 instances total given the small candidate gene pool and modest data quality (the paper itself acknowledges sparse temporal sampling).
3. **Extract features** for each instance via `core.feature_extraction.extract_features` (the same function `train_consensus_model.py` uses). Apply the same sentinel imputation logic. No re-scaling needed (the production pipeline doesn't use StandardScaler).
4. **Predict** with `model.predict_proba(X)`.
5. **Compute and report:**
    - ROC-AUC with bootstrap 95% CI (1000 resamples).
    - Average precision with bootstrap CI.
    - Accuracy at threshold 0.5 with bootstrap CI.
    - Brier score.
    - Confusion matrix.
    - Per-instance table: gene, label, predicted probability, predicted class.
6. **Output report** to `core/models_meta_classifier/validate_external_holdout_hughes2012.txt` in the same plain-text style as `training_report.txt`. Include:
    - Header with date, model artifact hash/version, dataset accession.
    - Dataset summary (N positives, N negatives, gene list).
    - Metrics block with CIs.
    - Per-instance predictions table.
    - A brief interpretive note: this is the most stringent generalization test (different organism, different lab, different platform, different tissue scope from any training instance).

### Why this matters for the manuscript

This LODO test is the strongest external generalization claim available for the JBR submission. Reviewers will ask: does CRS-AI generalize beyond mouse liver? With Phase 4 in place, the answer is a concrete AUROC + CI on a never-seen *Drosophila* dataset from a different lab using only canonical gene labels (zero circularity). Even if performance drops vs. the standard holdout, *reporting* it is the credible move — and Hughes 2012's known sparse-sampling limitation provides a built-in caveat.

---

## Phase 5: Retrain and update report

### Steps

1. Run `train_consensus_model.py` end-to-end with all four datasets. Note: total instance count should jump from ~3395 to ~5500–6000 (rough estimate: +600 fly + +1500 human).
2. Verify training_report.txt now contains:
    - Section 1 (header): updated total instance count.
    - Section 5.2: four dataset entries.
    - Section 5.4: fly + human gene lists added.
    - Section 6: CV and holdout metrics, expected to remain in the same range (CV ROC-AUC 0.90–0.94) or slightly improve. **A drop is acceptable and informative** if it reflects honest generalization across the more diverse training set.
    - Section 6.5: ambiguous holdout — expect improvement here. Adding Möller-Levet's "rhythmic-in-control-but-not-SR" hard positives should specifically help with borderline cases (AUROC > 0.70 is the goal).
    - Section 7: feature importances — the MDI vs Permutation discrepancy for `jtk_p_value`, `cosinor_p_value`, `cosinor_r_squared`, `ls_p_value` should shrink (the current divergence is a symptom of training-set homogeneity).
3. Run `validate_external_holdout_hughes2012.py`. Save its report.
4. Stash both report files together for review.

### Diagnostics to print at the end of retraining

Add a brief summary block printed at the end of `train_consensus_model.py` (or as an appendix to the report):

```
=== Dataset diversity summary ===
Organisms:    Mus musculus (mouse), Drosophila melanogaster (fly), Homo sapiens (human)
Tissues:      liver (mouse), clock neurons + dopaminergic outgroup (fly), whole blood (human)
Platforms:    microarray (mouse, human), RNA-seq (fly)
Total real biological instances: <N>
Total synthetic instances:        <M>
GroupShuffleSplit groups (genes): <G>
Cross-species group collisions:  0 (case-sensitive symbol distinct across species)
```

---

## Verification checklist

After Claude Code completes all five phases, the user should be able to verify:

- [ ] `training_data_meta_classifer/abruzzi_2017_cycling.xlsx` and `moller_levet_2013_circadian.xlsx` are loaded without error.
- [ ] `generate_from_GSE77451()` returns ~600–700 instances with `cell_type` populated.
- [ ] `generate_from_GSE39445()` returns ~1200–1600 instances.
- [ ] Total real instances in `training_report.txt` is ~3500–4000 (up from 1699).
- [ ] No `gene` symbol appears in both `KNOWN_CIRCADIAN_GENES_*` and `NON_RHYTHMIC_GENES_*` within the same species.
- [ ] CV ROC-AUC ± std reported and within plausible range (0.88–0.94).
- [ ] Holdout ROC-AUC with bootstrap CI reported.
- [ ] Ambiguous-holdout AUROC (Section 6.5) reported and ideally improved.
- [ ] `validate_external_holdout_hughes2012.txt` exists with AUROC + CI + per-instance table.
- [ ] The Section 7 feature-importance MDI–permutation gap for `jtk_p_value`, `cosinor_p_value`, `cosinor_r_squared`, `ls_p_value` is reported (and ideally reduced vs. the current report).

---

## General guidelines for execution

- **Read before writing.** Before adding a new function, read the existing `generate_from_geo()` end-to-end to match style and conventions. Reuse helpers (`download_geo_series_matrix`, `parse_series_matrix`, probe-mapping logic) rather than re-implementing.
- **Don't modify existing dataset behavior.** GSE11923 and GSE11516 generation must remain bit-identical to before this change.
- **Print verbose progress** to stdout during dataset generation — counts of instances per cell type, per label, dropped probes, etc. This makes debugging easier when something doesn't match expectations.
- **Pin random seeds.** Use `seed=42` everywhere a deterministic sample is taken (capping selections, etc.).
- **Don't silently re-derive labels.** If a supplementary XLSX is missing or malformed, fail loudly with a clear message pointing to the source URL. Do **not** fall back to running JTK/F24 in-house on the new datasets — that would be circular for this design choice.
- **Run Phase 1 + Phase 2 to completion and inspect outputs before starting Phase 3.** Print the first 5 instances generated per dataset (metadata dict + DataFrame head) so the user can sanity-check before retraining.
- **Run Phase 3 + Phase 5 (training) before Phase 4.** The LODO script depends on the trained model artifact.

When in doubt about a design decision not covered here, ask the user before proceeding — especially regarding caps, label edge cases, and platform annotation parsing for GPL15331 (human) and the Drosophila platform.
