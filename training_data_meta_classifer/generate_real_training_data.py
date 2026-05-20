"""
Generate Real Biological Training Data for ChronoScope Consensus Model
=====================================================================

Downloads and processes real gene expression time series from public
databases for training the meta-classifier alongside synthetic data.

Data sources:
  - Expression data: NCBI GEO (automatic download via urllib)
  - Gene labels: Built-in known circadian/housekeeping genes
  - Gene labels (optional): RhythmicDB BioCycle results (manual CSV)

Supported GEO datasets:
  - GSE11923: Hughes 2009, mouse liver, hourly x 48h (Affymetrix 430 2.0)

Usage:
    # Default: download GSE11923, label known genes only
    python generate_real_training_data.py

    # With BioCycle labels from RhythmicDB
    python generate_real_training_data.py --biocycle biocycle_results.csv

    # Custom GEO dataset
    python generate_real_training_data.py --geo GSE11923 --platform GPL1261

Output:
    Returns (metadata_list, dataframes_list) compatible with
    train_consensus_model.py

Author: Francisco Tassara
"""

import os
import sys
import gzip
import re
import urllib.request
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
from io import StringIO


# =============================================================================
# KNOWN GROUND TRUTH GENES
# =============================================================================

# Core clock genes - definitively circadian (~24h) from decades of
# knockout studies and molecular biology. These are TRUE ground truth.
KNOWN_CIRCADIAN_GENES: Set[str] = {
    # Core TTFL (transcription-translation feedback loop)
    'Per1', 'Per2', 'Per3',           # Period genes
    'Cry1', 'Cry2',                   # Cryptochrome genes
    'Arntl',                          # Bmal1
    'Clock',                          # CLOCK
    'Npas2',                          # NPAS2 (CLOCK paralog)
    'Nr1d1', 'Nr1d2',                # Rev-erb alpha/beta
    # Clock-controlled output genes (robust oscillators)
    'Dbp', 'Tef', 'Hlf',             # PAR bZip family
    'Rora', 'Rorb', 'Rorc',          # ROR family
    'Ciart',                          # Chrono
    'Bhlhe40', 'Bhlhe41',            # Dec1, Dec2
    'Nfil3',                          # E4BP4
}

# Housekeeping genes - definitively non-rhythmic. Standard reference
# genes used in qPCR normalization across circadian studies.
KNOWN_NON_RHYTHMIC_GENES: Set[str] = {
    'Gapdh',     # Glyceraldehyde-3-phosphate dehydrogenase
    'Actb',      # Beta-actin
    'Tbp',       # TATA-binding protein
    'Hprt',      # Hypoxanthine phosphoribosyltransferase (also Hprt1)
    'Hprt1',     # Alias
    'Rpl13a',    # Ribosomal protein L13a
    'B2m',       # Beta-2-microglobulin
    'Ubc',       # Ubiquitin C
    'Ppia',      # Cyclophilin A
    'Rpl32',     # Ribosomal protein L32
    'Eef1a1',    # Elongation factor 1-alpha
    'Sdha',      # Succinate dehydrogenase subunit A
    'Hmbs',      # Hydroxymethylbilane synthase
    'Ywhaz',     # 14-3-3 protein zeta
    'Pgk1',      # Phosphoglycerate kinase 1
    'Tfrc',      # Transferrin receptor
}


# =============================================================================
# GEO DATA DOWNLOAD
# =============================================================================

GEO_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'geo')


def _geo_ftp_prefix(accession: str) -> str:
    """Convert GEO accession to FTP directory prefix (e.g. GSE11nnn)."""
    return accession[:-3] + 'nnn'


def download_file(url: str, output_path: str) -> str:
    """Download a file from URL if not already cached."""
    if os.path.exists(output_path):
        print(f"  [cached] {os.path.basename(output_path)}")
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Saved: {output_path} ({size_kb:.0f} KB)")
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Download failed: {e}\n  URL: {url}") from e
    return output_path


def download_geo_series_matrix(accession: str) -> str:
    """Download GEO series matrix file."""
    prefix = _geo_ftp_prefix(accession)
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/"
           f"{accession}/matrix/{accession}_series_matrix.txt.gz")
    output = os.path.join(GEO_DATA_DIR, f"{accession}_series_matrix.txt.gz")
    return download_file(url, output)


def download_geo_platform_annot(platform_id: str) -> str:
    """Download GEO platform annotation file (.annot.gz or .soft.gz fallback)."""
    prefix = _geo_ftp_prefix(platform_id)

    # Try .annot.gz first (Affymetrix platforms typically have this)
    annot_url = (f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/"
                 f"{platform_id}/annot/{platform_id}.annot.gz")
    annot_output = os.path.join(GEO_DATA_DIR, f"{platform_id}.annot.gz")

    try:
        return download_file(annot_url, annot_output)
    except RuntimeError:
        print(f"  .annot.gz not available, trying SOFT format...")

    # Fallback: download SOFT file (Illumina and other platforms)
    soft_url = (f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/"
                f"{platform_id}/soft/{platform_id}_family.soft.gz")
    soft_output = os.path.join(GEO_DATA_DIR, f"{platform_id}_family.soft.gz")
    return download_file(soft_url, soft_output)


# =============================================================================
# GEO DATA PARSING
# =============================================================================

def parse_series_matrix(filepath: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse a GEO series matrix file.

    Returns:
        expression_df: rows=probe_ids, columns=sample_ids, values=expression
        sample_info:   {sample_id: {key: [values]}} from !Sample_ lines
    """
    metadata_lines = []
    data_lines = []
    in_data = False

    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if line.startswith('!'):
                metadata_lines.append(line)
            elif line.startswith('"ID_REF"') or line.startswith('ID_REF'):
                in_data = True
                data_lines.append(line)
            elif in_data and line and not line.startswith('!'):
                if line.strip() == '':
                    continue
                data_lines.append(line)

    # Parse expression matrix
    data_text = '\n'.join(data_lines)
    expression_df = pd.read_csv(
        StringIO(data_text), sep='\t', index_col=0, na_values=['null', 'NA']
    )
    expression_df.index = expression_df.index.astype(str)
    expression_df.index.name = 'probe_id'

    # Parse sample metadata from !Sample_ lines
    sample_ids = list(expression_df.columns)
    sample_info = {sid: {} for sid in sample_ids}

    for line in metadata_lines:
        if not line.startswith('!Sample_'):
            continue
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        key = parts[0].lstrip('!')
        values = [v.strip('"').strip() for v in parts[1:]]

        for i, sid in enumerate(sample_ids):
            if i < len(values):
                if key not in sample_info[sid]:
                    sample_info[sid][key] = []
                sample_info[sid][key].append(values[i])

    return expression_df, sample_info


def extract_timepoints_from_samples(sample_info: Dict) -> Dict[str, float]:
    """
    Extract time in hours from GEO sample characteristics.

    Handles common patterns:
      - "time point: CT18", "zeitgeber time: ZT4"
      - "time: 18h", "circadian time: 18"
      - "time (hours): 48"

    Returns: {sample_id: time_in_hours}
    """
    sample_times = {}

    for sample_id, info in sample_info.items():
        found = False
        for key, values in info.items():
            if found:
                break
            if 'characteristic' not in key.lower() and 'title' not in key.lower():
                continue
            for val in values:
                match = re.search(
                    r'(?:time\s*(?:point)?|CT|ZT|circadian\s+time|'
                    r'hours?\s*(?:post)?|time\s*\(hours?\))\s*'
                    r'[:\s=]*\s*(?:CT|ZT)?\s*(\d+(?:\.\d+)?)',
                    val, re.IGNORECASE
                )
                if match:
                    sample_times[sample_id] = float(match.group(1))
                    found = True
                    break

    return sample_times


def parse_platform_annotation(filepath: str) -> Dict[str, str]:
    """
    Parse GEO platform annotation to get probe -> gene symbol.

    Supports two formats:
      - .annot.gz: Tab-delimited with header row containing 'Gene Symbol'
      - .soft.gz:  SOFT format with !platform_table_begin/end markers

    Returns: {probe_id: gene_symbol}
    """
    is_soft = '_family.soft' in filepath or filepath.endswith('.soft.gz')

    if is_soft:
        return _parse_soft_platform(filepath)
    else:
        return _parse_annot_platform(filepath)


def _parse_annot_platform(filepath: str) -> Dict[str, str]:
    """Parse .annot.gz format."""
    probe_to_gene = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    header = None
    gene_col = None

    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if line.startswith('#'):
                continue
            if header is None:
                cols = line.split('\t')
                for i, col in enumerate(cols):
                    if 'gene symbol' in col.lower() or col == 'Gene Symbol':
                        gene_col = i
                        header = cols
                        break
                continue
            if gene_col is not None:
                parts = line.split('\t')
                if len(parts) > gene_col:
                    probe_id = parts[0].strip().strip('"')
                    raw_symbol = parts[gene_col].strip().strip('"')
                    if raw_symbol and raw_symbol != '---' and raw_symbol != '':
                        primary = raw_symbol.split('///')[0].strip()
                        if primary:
                            probe_to_gene[probe_id] = primary

    print(f"  Platform annotation: {len(probe_to_gene)} probes mapped to genes")
    return probe_to_gene


def _parse_soft_platform(filepath: str) -> Dict[str, str]:
    """Parse SOFT format platform file (.soft.gz)."""
    probe_to_gene = {}
    opener = gzip.open if filepath.endswith('.gz') else open
    in_table = False
    header = None
    gene_col = None
    id_col = 0

    with opener(filepath, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')

            if line == '!platform_table_begin':
                in_table = True
                continue
            if line == '!platform_table_end':
                break

            if not in_table:
                continue

            # First line in table is the header
            if header is None:
                header = line.split('\t')
                for i, col in enumerate(header):
                    cl = col.strip().lower()
                    if cl == 'id':
                        id_col = i
                    if ('symbol' in cl or cl == 'gene'
                            or cl == 'ilmn_gene'
                            or cl == 'gene_symbol'):
                        gene_col = i
                if gene_col is None:
                    # Try broader match
                    for i, col in enumerate(header):
                        if 'gene' in col.lower():
                            gene_col = i
                            break
                continue

            if gene_col is not None:
                parts = line.split('\t')
                if len(parts) > max(id_col, gene_col):
                    probe_id = parts[id_col].strip().strip('"')
                    raw_symbol = parts[gene_col].strip().strip('"')
                    if raw_symbol and raw_symbol != '' and raw_symbol != '---':
                        primary = raw_symbol.split('///')[0].strip()
                        if primary:
                            probe_to_gene[probe_id] = primary

    print(f"  Platform annotation (SOFT): {len(probe_to_gene)} probes mapped to genes")
    return probe_to_gene


def map_expression_to_genes(
    expression_df: pd.DataFrame,
    probe_to_gene: Dict[str, str]
) -> pd.DataFrame:
    """
    Map probe-level expression to gene-level expression.

    When multiple probes map to the same gene, keeps the probe with
    highest mean expression (standard practice in microarray analysis).

    Returns: DataFrame with gene symbols as index, samples as columns
    """
    # Add gene symbol column
    gene_symbols = expression_df.index.map(
        lambda pid: probe_to_gene.get(pid, None)
    )
    expr_with_genes = expression_df.copy()
    expr_with_genes['gene_symbol'] = gene_symbols

    # Drop probes without gene mapping
    expr_with_genes = expr_with_genes.dropna(subset=['gene_symbol'])

    # For duplicate genes, keep probe with highest mean expression
    expr_with_genes['mean_expr'] = expr_with_genes.drop(
        columns=['gene_symbol']
    ).mean(axis=1)
    expr_with_genes = expr_with_genes.sort_values('mean_expr', ascending=False)
    expr_with_genes = expr_with_genes.drop_duplicates(
        subset='gene_symbol', keep='first'
    )

    # Set gene symbol as index, drop helper columns
    expr_with_genes = expr_with_genes.drop(columns=['mean_expr'])
    expr_with_genes = expr_with_genes.set_index('gene_symbol')
    expr_with_genes.index.name = 'gene'

    print(f"  Gene-level expression: {len(expr_with_genes)} unique genes")
    return expr_with_genes


# =============================================================================
# BIOCYCLE LABELS (from RhythmicDB manual download)
# =============================================================================

def load_biocycle_labels_xlsx(
    xlsx_path: str,
    dataset_id: str,
    q_threshold: float = 0.01,
    q_non_rhythmic_threshold: float = 0.2,
) -> Tuple[Set[str], Set[str], Set[str], Dict[str, float]]:
    """
    Load BioCycle classification results from the RhythmicDB Excel export.

    Filters by a specific dataset to avoid cross-experiment label noise.
    Uses an ambiguity gap to exclude genes with borderline significance.

    Args:
        xlsx_path: Path to rhythmicdb_query_bioCycle.xlsx
        dataset_id: RhythmicDB dataset ID (e.g. 'E-GEOD-11516')
        q_threshold: Q-value cutoff for rhythmic (default 0.01)
        q_non_rhythmic_threshold: Q-value above which genes are labeled
            non-rhythmic (default 0.2). Genes between q_threshold and
            this value are excluded as ambiguous.

    Returns:
        (circadian_genes, non_rhythmic_genes, ambiguous_genes, gene_q_values)
        - circadian_genes: genes with q <= q_threshold
        - non_rhythmic_genes: genes with q > q_non_rhythmic_threshold
        - ambiguous_genes: genes in the gap (excluded from training)
        - gene_q_values: {gene: best (lowest) q-value} for ALL genes; used
            downstream to assign provisional labels to ambiguous genes for
            holdout evaluation (label = 1 if q < midpoint, else 0).
    """
    df = pd.read_excel(xlsx_path)

    # Filter by dataset
    ds_data = df[df['Dataset'] == dataset_id].copy()
    if len(ds_data) == 0:
        available = df['Dataset'].unique().tolist()
        raise ValueError(
            f"Dataset '{dataset_id}' not found. Available: {available}"
        )

    # For genes with multiple probes, use the best (lowest) Q-value
    gene_best_q = ds_data.groupby('Gene info')['Q-value'].min()

    circadian = set(gene_best_q[gene_best_q <= q_threshold].index)
    non_rhythmic = set(gene_best_q[gene_best_q > q_non_rhythmic_threshold].index)
    ambiguous = set(gene_best_q.index) - circadian - non_rhythmic

    print(f"  BioCycle labels from {dataset_id}:")
    print(f"    Total unique genes: {len(gene_best_q)}")
    print(f"    Circadian (q <= {q_threshold}): {len(circadian)}")
    print(f"    Non-rhythmic (q > {q_non_rhythmic_threshold}): {len(non_rhythmic)}")
    print(f"    Ambiguous (excluded): {len(ambiguous)}")

    gene_q_values = gene_best_q.to_dict()
    return circadian, non_rhythmic, ambiguous, gene_q_values


# =============================================================================
# TRAINING INSTANCE GENERATION
# =============================================================================

def _subsample_timeseries(
    times: np.ndarray,
    values: np.ndarray,
    interval_h: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample a time series to a given interval (e.g. every 2h, every 4h).
    Selects the closest existing timepoint to each target time.
    """
    t_min, t_max = times.min(), times.max()
    target_times = np.arange(t_min, t_max + interval_h / 2, interval_h)
    indices = []
    for t_target in target_times:
        idx = np.argmin(np.abs(times - t_target))
        if idx not in indices:
            indices.append(idx)

    return times[indices], values[indices]


def generate_real_training_instances(
    expression_df: pd.DataFrame,
    sample_times: Dict[str, float],
    circadian_genes: Set[str],
    non_rhythmic_genes: Set[str],
    starting_id: int = 2000,
    subsample_intervals: Optional[List[float]] = None,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """
    Generate ChronoScope-compatible training instances from real expression data.

    Args:
        expression_df: Gene-level expression (genes x samples)
        sample_times: {sample_id: time_in_hours}
        circadian_genes: Set of gene symbols labeled as circadian (label=1)
        non_rhythmic_genes: Set of gene symbols labeled as non-rhythmic (label=0)
        starting_id: Starting instance ID (to avoid conflicts with synthetic)
        subsample_intervals: List of intervals (hours) to subsample. Creates
            additional instances at lower temporal resolution. E.g. [2.0, 4.0]
            Default: None (use original resolution only)

    Returns:
        (metadata_list, dataframes_list) compatible with train_consensus_model.py
    """
    if subsample_intervals is None:
        subsample_intervals = []

    # Get ordered samples with valid times
    valid_samples = [s for s in expression_df.columns if s in sample_times]
    valid_samples.sort(key=lambda s: sample_times[s])
    base_times = np.array([sample_times[s] for s in valid_samples])

    # Normalize times to start from 0
    base_times = base_times - base_times.min()

    # Sampling interval (use unique times to handle biological replicates)
    unique_base = np.unique(base_times)
    if len(unique_base) > 1:
        base_interval = float(np.median(np.diff(unique_base)))
    else:
        base_interval = 1.0

    print(f"\n  Time series: {len(base_times)} timepoints, "
          f"{base_interval:.1f}h intervals, "
          f"{base_times[-1]:.0f}h total span")

    # Build all gene labels
    all_labeled = {}
    for gene in circadian_genes:
        if gene in expression_df.index:
            all_labeled[gene] = 1
    for gene in non_rhythmic_genes:
        if gene in expression_df.index:
            all_labeled[gene] = 0

    print(f"  Labeled genes found in expression data: {len(all_labeled)}")
    n_rhythmic = sum(1 for v in all_labeled.values() if v == 1)
    n_non = sum(1 for v in all_labeled.values() if v == 0)
    print(f"    Rhythmic: {n_rhythmic}")
    print(f"    Non-rhythmic: {n_non}")

    # Generate instances
    metadata_list = []
    all_dataframes = []
    instance_id = starting_id

    resolutions = [('original', base_interval, None)]
    for interval in subsample_intervals:
        if interval > base_interval:
            resolutions.append((f'subsample_{interval:.0f}h', interval, interval))

    for gene, label in sorted(all_labeled.items()):
        # Get expression values for this gene across valid samples
        gene_values = expression_df.loc[gene, valid_samples].values.astype(float)

        # Skip genes with too many NaN
        valid_mask = ~np.isnan(gene_values)
        if valid_mask.sum() < 6:
            continue

        for res_name, res_interval, subsample_h in resolutions:
            if subsample_h is not None:
                times, values = _subsample_timeseries(
                    base_times[valid_mask],
                    gene_values[valid_mask],
                    subsample_h
                )
            else:
                times = base_times[valid_mask]
                values = gene_values[valid_mask]

            if len(times) < 6:
                continue

            # Build DataFrame in ChronoScope format
            var_name = f'var_{instance_id}'
            rows = []
            for i, t in enumerate(times):
                rows.append({
                    'time': float(t),
                    'condition': 'control',
                    'replicate': 'rep1',
                    var_name: float(values[i])
                })

            df = pd.DataFrame(rows)
            all_dataframes.append(df)

            metadata_list.append({
                'instance_id': instance_id,
                'variable': var_name,
                'signal_type': f'real_{gene}_{res_name}',
                'is_rhythmic': label,
                'n_timepoints': len(times),
                'n_replicates': 1,
                'sampling_hours': res_interval,
                'snr': 0.0,   # Unknown for real data
                'period': 24.0 if label == 1 else 0.0,
                'has_outliers': False,
                'source': 'biological',
                'gene': gene,
            })
            instance_id += 1

    return metadata_list, all_dataframes


# =============================================================================
# MAIN: COMPLETE PIPELINE
# =============================================================================

def generate_from_geo(
    geo_accession: str = 'GSE11923',
    platform_id: str = 'GPL1261',
    biocycle_xlsx: Optional[str] = None,
    biocycle_dataset_id: Optional[str] = None,
    biocycle_q_threshold: float = 0.01,
    biocycle_q_non_rhythmic: float = 0.2,
    max_rhythmic: Optional[int] = None,
    max_non_rhythmic: Optional[int] = None,
    starting_id: int = 2000,
    subsample_intervals: Optional[List[float]] = None,
    return_ambiguous: bool = False,
):
    """
    Complete pipeline: download GEO data -> parse -> label -> generate instances.

    Args:
        geo_accession: GEO series accession (e.g. 'GSE11923')
        platform_id: GEO platform ID for probe annotation (e.g. 'GPL1261')
        biocycle_xlsx: Path to RhythmicDB BioCycle Excel file (optional)
        biocycle_dataset_id: RhythmicDB dataset ID to filter (e.g. 'E-GEOD-11516')
        biocycle_q_threshold: Q-value cutoff for rhythmic (default 0.01)
        biocycle_q_non_rhythmic: Q-value above which -> non-rhythmic (default 0.2)
        max_rhythmic: Cap on rhythmic instances (None = no cap)
        max_non_rhythmic: Cap on non-rhythmic instances (None = no cap)
        starting_id: Starting instance ID
        subsample_intervals: Intervals for subsampling (e.g. [2.0, 4.0])
        return_ambiguous: If True, also returns a separate set of instances
            for genes in the BioCycle ambiguity gap (q in (q_threshold,
            q_non_rhythmic]). Used for holdout evaluation on borderline
            cases. These instances are labeled provisionally based on the
            q-value midpoint: q <= midpoint => label 1, else 0.

    Returns:
        If return_ambiguous=False (default):
            (metadata_list, dataframes_list)
        If return_ambiguous=True:
            (metadata_list, dataframes_list,
             ambiguous_metadata, ambiguous_dataframes)
        — all compatible with train_consensus_model.py
    """
    print("=" * 70)
    print("REAL BIOLOGICAL TRAINING DATA GENERATOR")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Download from GEO
    # ------------------------------------------------------------------
    print(f"\n[1/5] Downloading GEO data ({geo_accession})...")
    matrix_path = download_geo_series_matrix(geo_accession)
    annot_path = download_geo_platform_annot(platform_id)

    # ------------------------------------------------------------------
    # Step 2: Parse expression data
    # ------------------------------------------------------------------
    print(f"\n[2/5] Parsing expression data...")
    expression_df, sample_info = parse_series_matrix(matrix_path)
    print(f"  Probes: {len(expression_df)}")
    print(f"  Samples: {len(expression_df.columns)}")

    # Extract timepoints
    sample_times = extract_timepoints_from_samples(sample_info)
    if len(sample_times) == 0:
        raise RuntimeError(
            "Could not extract timepoints from sample metadata. "
            "Check the GEO dataset format."
        )
    print(f"  Timepoints extracted: {len(sample_times)} samples")

    t_values = sorted(sample_times.values())
    print(f"  Time range: {t_values[0]:.0f}h to {t_values[-1]:.0f}h")

    # ------------------------------------------------------------------
    # Step 3: Map probes to genes
    # ------------------------------------------------------------------
    print(f"\n[3/5] Mapping probes to gene symbols ({platform_id})...")
    probe_to_gene = parse_platform_annotation(annot_path)
    gene_expression = map_expression_to_genes(expression_df, probe_to_gene)

    # ------------------------------------------------------------------
    # Step 4: Label genes
    # ------------------------------------------------------------------
    print(f"\n[4/5] Labeling genes...")

    circadian_genes = set(KNOWN_CIRCADIAN_GENES)
    non_rhythmic_genes = set(KNOWN_NON_RHYTHMIC_GENES)

    # Count known genes found
    known_circ_found = circadian_genes & set(gene_expression.index)
    known_non_found = non_rhythmic_genes & set(gene_expression.index)
    print(f"  Known circadian genes found: {len(known_circ_found)}/{len(KNOWN_CIRCADIAN_GENES)}")
    print(f"    {sorted(known_circ_found)}")
    print(f"  Known non-rhythmic genes found: {len(known_non_found)}/{len(KNOWN_NON_RHYTHMIC_GENES)}")
    print(f"    {sorted(known_non_found)}")

    # Load BioCycle labels if provided
    bc_ambiguous_genes: Set[str] = set()
    bc_gene_q_values: Dict[str, float] = {}
    if biocycle_xlsx and biocycle_dataset_id and os.path.exists(biocycle_xlsx):
        bc_circadian, bc_non_rhythmic, bc_ambiguous_genes, bc_gene_q_values = (
            load_biocycle_labels_xlsx(
                biocycle_xlsx, biocycle_dataset_id,
                biocycle_q_threshold, biocycle_q_non_rhythmic,
            )
        )
        circadian_genes |= bc_circadian
        non_rhythmic_genes |= bc_non_rhythmic
        # Ambiguous genes are excluded from both training sets

    # ------------------------------------------------------------------
    # Step 5: Generate training instances
    # ------------------------------------------------------------------
    print(f"\n[5/5] Generating training instances...")
    metadata, dataframes = generate_real_training_instances(
        gene_expression,
        sample_times,
        circadian_genes,
        non_rhythmic_genes,
        starting_id=starting_id,
        subsample_intervals=subsample_intervals,
    )

    # Cap instances if requested (random sample to avoid bias)
    n_before = len(metadata)
    if max_rhythmic is not None or max_non_rhythmic is not None:
        rng = np.random.RandomState(42)
        rhythmic_idx = [i for i, m in enumerate(metadata) if m['is_rhythmic'] == 1]
        non_rhythmic_idx = [i for i, m in enumerate(metadata) if m['is_rhythmic'] == 0]

        if max_rhythmic is not None and len(rhythmic_idx) > max_rhythmic:
            rhythmic_idx = list(rng.choice(rhythmic_idx, max_rhythmic, replace=False))
        if max_non_rhythmic is not None and len(non_rhythmic_idx) > max_non_rhythmic:
            non_rhythmic_idx = list(rng.choice(non_rhythmic_idx, max_non_rhythmic, replace=False))

        keep_idx = sorted(rhythmic_idx + non_rhythmic_idx)
        metadata = [metadata[i] for i in keep_idx]
        dataframes = [dataframes[i] for i in keep_idx]

    # Summary
    n_rhythmic = sum(1 for m in metadata if m['is_rhythmic'] == 1)
    n_non = sum(1 for m in metadata if m['is_rhythmic'] == 0)
    print(f"\n  Generated {n_before} real biological instances")
    if n_before != len(metadata):
        print(f"  Capped to {len(metadata)} instances")
    print(f"    Rhythmic:     {n_rhythmic}")
    print(f"    Non-rhythmic: {n_non}")

    if metadata:
        genes_used = set(m['gene'] for m in metadata)
        print(f"    Unique genes: {len(genes_used)}")

    # ------------------------------------------------------------------
    # Optional: build ambiguous-gene holdout for borderline-case evaluation
    # ------------------------------------------------------------------
    if return_ambiguous:
        if not bc_ambiguous_genes:
            print("\n  No ambiguous genes available (BioCycle data not loaded);"
                  " returning empty ambiguous holdout.")
            return metadata, dataframes, [], []

        # Provisional labeling: midpoint of the q-value gap.
        # Genes closer to q_threshold get label 1 (closer to "rhythmic");
        # genes closer to q_non_rhythmic get label 0.
        midpoint = (biocycle_q_threshold + biocycle_q_non_rhythmic) / 2.0
        amb_rhythmic = {
            g for g in bc_ambiguous_genes
            if bc_gene_q_values.get(g, 1.0) <= midpoint
        }
        amb_non_rhythmic = bc_ambiguous_genes - amb_rhythmic

        print(f"\n  Generating ambiguous holdout (midpoint q = {midpoint}):")
        print(f"    Provisional rhythmic (q <= midpoint):     {len(amb_rhythmic)}")
        print(f"    Provisional non-rhythmic (q > midpoint):  {len(amb_non_rhythmic)}")

        # Use a distinct starting_id to avoid collisions with training data
        amb_start_id = starting_id + 50000
        amb_metadata, amb_dataframes = generate_real_training_instances(
            gene_expression,
            sample_times,
            amb_rhythmic,
            amb_non_rhythmic,
            starting_id=amb_start_id,
            subsample_intervals=subsample_intervals,
        )

        # Tag with q-value for downstream inspection
        for m in amb_metadata:
            m['biocycle_q_value'] = bc_gene_q_values.get(m['gene'])
            m['source'] = 'biological_ambiguous'

        print(f"  Ambiguous holdout instances: {len(amb_metadata)}")
        return metadata, dataframes, amb_metadata, amb_dataframes

    return metadata, dataframes


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate real biological training data for ChronoScope'
    )
    parser.add_argument(
        '--geo', default='GSE11923',
        help='GEO series accession (default: GSE11923, Hughes 2009 mouse liver)'
    )
    parser.add_argument(
        '--platform', default='GPL1261',
        help='GEO platform ID (default: GPL1261, Affymetrix Mouse 430 2.0)'
    )
    parser.add_argument(
        '--biocycle-xlsx', default=None,
        help='Path to RhythmicDB BioCycle Excel file (optional)'
    )
    parser.add_argument(
        '--biocycle-dataset', default=None,
        help='RhythmicDB dataset ID to filter (e.g. E-GEOD-11516)'
    )
    parser.add_argument(
        '--biocycle-q', type=float, default=0.01,
        help='Q-value threshold for rhythmic (default: 0.01)'
    )
    parser.add_argument(
        '--biocycle-q-nr', type=float, default=0.2,
        help='Q-value above which genes are non-rhythmic (default: 0.2)'
    )
    parser.add_argument(
        '--max-rhythmic', type=int, default=None,
        help='Cap on rhythmic instances (default: no cap)'
    )
    parser.add_argument(
        '--max-non-rhythmic', type=int, default=None,
        help='Cap on non-rhythmic instances (default: no cap)'
    )
    parser.add_argument(
        '--subsample', nargs='*', type=float, default=[2.0, 4.0],
        help='Subsampling intervals in hours (default: 2.0 4.0)'
    )

    args = parser.parse_args()

    metadata, dataframes = generate_from_geo(
        geo_accession=args.geo,
        platform_id=args.platform,
        biocycle_xlsx=args.biocycle_xlsx,
        biocycle_dataset_id=args.biocycle_dataset,
        biocycle_q_threshold=args.biocycle_q,
        biocycle_q_non_rhythmic=args.biocycle_q_nr,
        max_rhythmic=args.max_rhythmic,
        max_non_rhythmic=args.max_non_rhythmic,
        subsample_intervals=args.subsample,
    )

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Total instances: {len(metadata)}")
    print(f"\n  To use in training, modify train_consensus_model.py to call:")
    print(f"    from generate_real_training_data import generate_from_geo")
    print(f"    real_meta, real_dfs = generate_from_geo()")
