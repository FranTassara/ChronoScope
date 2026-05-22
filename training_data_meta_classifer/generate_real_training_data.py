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

# Drosophila melanogaster core clock genes (FlyBase symbols, case-sensitive)
KNOWN_CIRCADIAN_GENES_FLY: Set[str] = {
    'per', 'tim', 'Clk', 'cyc', 'vri', 'Pdp1', 'cry', 'cwo',
    'Pdf', 'sgg', 'dco', 'nmo', 'jet', 'twins', 'ck1', 'NPF',
    'shaggy', 'dbt',
}

# Drosophila housekeeping genes — should always be non-rhythmic
NON_RHYTHMIC_GENES_FLY: Set[str] = {
    'RpL32', 'Act5C', 'Act5c', 'Act88F', 'alphaTub84B',
    'Gapdh1', 'Gapdh2', 'Sdha', 'eIF1A', 'eEF1alpha1',
    'Rpl13', 'Rps17', 'Tbp', 'GstD1', 'Hsc70-4', 'Hsc70Cb',
    'CG8187', 'CG7434',
}

# Homo sapiens core clock genes (HGNC symbols, uppercase)
KNOWN_CIRCADIAN_GENES_HUMAN: Set[str] = {
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
NON_RHYTHMIC_GENES_HUMAN: Set[str] = {
    'ACTB', 'GAPDH', 'HPRT1', 'TBP', 'RPL13A', 'B2M', 'UBC',
    'PPIA', 'RPL32', 'EEF1A1', 'SDHA', 'HMBS', 'YWHAZ', 'PGK1',
    'TFRC', 'POLR2A', 'PSMB4', 'PSMB2', 'CHMP2A', 'EMC7',
    'GPI', 'C1orf43', 'REEP5', 'SNRPD3', 'VCP', 'VPS29',
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


def download_geo_series_matrix(accession: str,
                               cache_dir: Optional[str] = None) -> str:
    """Download GEO series matrix file.

    Args:
        accession:  GEO series accession (e.g. 'GSE11923').
        cache_dir:  Directory for cached downloads. Defaults to GEO_DATA_DIR.
    """
    prefix = _geo_ftp_prefix(accession)
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/"
           f"{accession}/matrix/{accession}_series_matrix.txt.gz")
    out_dir = cache_dir if cache_dir is not None else GEO_DATA_DIR
    output = os.path.join(out_dir, f"{accession}_series_matrix.txt.gz")
    return download_file(url, output)


def download_geo_platform_annot(platform_id: str,
                                cache_dir: Optional[str] = None) -> str:
    """Download GEO platform annotation file (.annot.gz or .soft.gz fallback).

    Args:
        platform_id: GEO platform accession (e.g. 'GPL1261').
        cache_dir:   Directory for cached downloads. Defaults to GEO_DATA_DIR.
    """
    prefix = _geo_ftp_prefix(platform_id)
    out_dir = cache_dir if cache_dir is not None else GEO_DATA_DIR

    # Try .annot.gz first (Affymetrix platforms typically have this)
    annot_url = (f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/"
                 f"{platform_id}/annot/{platform_id}.annot.gz")
    annot_output = os.path.join(out_dir, f"{platform_id}.annot.gz")

    try:
        return download_file(annot_url, annot_output)
    except RuntimeError:
        print(f"  .annot.gz not available, trying SOFT format...")

    # Fallback: download SOFT file (Illumina and other platforms)
    soft_url = (f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/"
                f"{platform_id}/soft/{platform_id}_family.soft.gz")
    soft_output = os.path.join(out_dir, f"{platform_id}_family.soft.gz")
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
# XLSX LABEL LOADERS (for new biological datasets)
# =============================================================================

def load_abruzzi_cycling_labels(
    xlsx_path: Path,
    hc_only: bool = True,
) -> Dict[str, Set[str]]:
    """
    Load per-cell-type cycling gene labels from Abruzzi 2017 S3 XLSX.

    Returns {cell_type: set_of_gene_symbols} for 'LNv', 'LNd', 'DN1', 'TH'.
    Symbols are case-sensitive FlyBase symbols exactly as published.

    If hc_only=True (default), only HC-cyclers are returned; otherwise both
    HC- and LC-cyclers are included.

    Source: https://doi.org/10.1371/journal.pgen.1006613.s003
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Abruzzi 2017 cycling XLSX not found: {xlsx_path}\n"
            f"  Source: https://doi.org/10.1371/journal.pgen.1006613.s003"
        )

    sheet_map = {
        'LNv': 'LNv_cyclers',
        'LNd': 'LNd_cyclers',
        'DN1': 'DN1_cyclers',
        'TH':  'TH_cyclers',
    }
    result: Dict[str, Set[str]] = {}
    for cell_type, sheet_name in sheet_map.items():
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        sym_col = df.columns[0]  # gene symbol always column 0

        # 'cycling?' column name varies across sheets — locate by keyword
        cyc_col = next(
            (c for c in df.columns if 'cycling' in str(c).lower()), None
        )
        if cyc_col is None:
            raise ValueError(
                f"No 'cycling?' column found in sheet '{sheet_name}'. "
                f"Columns: {list(df.columns)}"
            )

        cycling_vals = df[cyc_col].fillna('').astype(str).str.strip()
        if hc_only:
            mask = cycling_vals == 'HC-cycler'
        else:
            mask = cycling_vals.isin(['HC-cycler', 'LC-cycler'])

        genes = set(df.loc[mask, sym_col].dropna().astype(str).str.strip())
        result[cell_type] = genes
        print(f"    Abruzzi ({cell_type}): {len(genes)} "
              f"{'HC-' if hc_only else ''}cyclers")

    return result


def load_moller_levet_labels(xlsx_path: Path) -> pd.DataFrame:
    """
    Load circadian labels from Möller-Levet 2013 PNAS Dataset S2 (Main_list sheet).

    Returns DataFrame with columns:
        gene         : identifier (gene symbol or accession)
        circ_control : 1 if circadian in control condition, else 0
        circ_sr      : 1 if circadian in sleep restriction, else 0
        sleep_effect : 1 if sleep condition has a significant effect, else 0

    Probe-ID rows (Identifier starting with 'A_') are filtered out because
    they lack gene-symbol mapping and cannot be matched to the expression matrix.

    Source: https://doi.org/10.1073/pnas.1217154110
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Möller-Levet 2013 XLSX not found: {xlsx_path}\n"
            f"  Source: https://doi.org/10.1073/pnas.1217154110"
        )

    df = pd.read_excel(xlsx_path, sheet_name='Main_list')

    col_map = {
        'Identifier (Gene/ Probe/ Accession)': 'gene',
        'Sleep Condition effect':              'sleep_effect',
        'Circadian in Control':                'circ_control',
        'Circadian in Sleep Restriction':      'circ_sr',
    }
    df = df.rename(columns=col_map)
    df = df[['gene', 'circ_control', 'circ_sr', 'sleep_effect']].copy()

    n_before = len(df)
    df = df[~df['gene'].astype(str).str.startswith('A_')].copy()
    n_probe = n_before - len(df)

    print(f"  Möller-Levet 2013 Main_list: {n_before} total rows")
    print(f"    Agilent probe IDs removed: {n_probe}")
    print(f"    Gene/accession rows:       {len(df)}")

    circ_both      = ((df['circ_control'] == 1) & (df['circ_sr'] == 1)).sum()
    circ_ctrl_only = ((df['circ_control'] == 1) & (df['circ_sr'] == 0)).sum()
    neg_clean      = ((df['circ_control'] == 0) & (df['circ_sr'] == 0)
                      & (df['sleep_effect'] == 0)).sum()
    print(f"    Rhythmic in both conditions:  {circ_both}")
    print(f"    Rhythmic in control only:     {circ_ctrl_only}")
    print(f"    Clean negatives (all flags=0):{neg_clean}")

    return df.reset_index(drop=True)


# =============================================================================
# DATASET 3: GSE77451 (Abruzzi 2017, Drosophila clock neurons)
# =============================================================================

def generate_from_GSE77451(
    abruzzi_xlsx_path: Path,
    geo_cache_dir: Optional[Path] = None,
    starting_instance_id: int = 5000,
    max_positives_per_cell_type: int = 200,
    hc_only: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """
    Generate training instances from GSE77451 (Abruzzi et al. 2017, PLOS Genetics).

    Positives (label=1):
      HC-cyclers in LNv, LNd, DN1 from Abruzzi 2017 S3 XLSX (embedded expression
      data). Sorted by JTK p-value (ascending), capped at max_positives_per_cell_type.
      TH positives are excluded — TH is a non-circadian dopaminergic outgroup.

    Negatives (label=0):
      NON_RHYTHMIC_GENES_FLY in ALL four cell types (requires GEO expression).
      KNOWN_CIRCADIAN_GENES_FLY in TH only (clock genes don't cycle in TH).

    NOTE: Positive expression comes directly from the XLSX supplementary file.
    Negative expression requires downloading the GSE77451 series matrix from GEO.
    If the GEO download fails, only positives are returned with a warning.

    Cross-species group safety: Drosophila symbols like 'per' are distinct from
    mouse 'Per1' and human 'PER1' in case-sensitive string comparison, so
    GroupShuffleSplit groups remain disjoint across species.

    Returns (metadata_list, dataframes_list) in ChronoScope format.
    """
    print("=" * 70)
    print("GSE77451: Drosophila Clock Neurons (Abruzzi et al. 2017)")
    print("=" * 70)

    abruzzi_xlsx_path = Path(abruzzi_xlsx_path)
    if not abruzzi_xlsx_path.exists():
        raise FileNotFoundError(
            f"Abruzzi 2017 XLSX not found: {abruzzi_xlsx_path}\n"
            f"  Source: https://doi.org/10.1371/journal.pgen.1006613.s003"
        )

    # Per-cell-type XLSX configuration:
    #   (sheet_name, rep1_zt_columns, rep2_zt_columns)
    # Rep1 columns are integers (ZT values); rep2 columns are the string
    # versions with '.1' suffix (e.g. 2 → '2.1') as read by pd.read_excel.
    CELL_CFG: Dict[str, tuple] = {
        'LNv': ('LNv_cyclers',
                [2, 6, 10, 14, 18, 22],
                ['2.1', '6.1', '10.1', '14.1', '18.1', '22.1']),
        'LNd': ('LNd_cyclers',
                [2, 6, 10, 14, 18, 22],
                ['2.1', '6.1', '10.1', '14.1', '18.1', '22.1']),
        'DN1': ('DN1_cyclers',
                [3, 7, 11, 15, 19, 23],
                ['3.1', '7.1', '11.1', '15.1', '19.1', '23.1']),
        'TH':  ('TH_cyclers',
                [2, 6, 10, 14, 18, 22],
                ['2.1', '6.1', '10.1', '14.1', '18.1', '22.1']),
    }

    metadata_list: List[Dict] = []
    dataframe_list: List[pd.DataFrame] = []
    instance_id = starting_instance_id

    # ------------------------------------------------------------------ #
    # Step 1: Positive instances from XLSX expression data                #
    # ------------------------------------------------------------------ #
    print("\n[1/3] Generating POSITIVE instances from XLSX HC-cyclers...")
    pos_by_ct: Dict[str, int] = {}

    for cell_type, (sheet_name, zt_rep1, zt_rep2) in CELL_CFG.items():
        if cell_type == 'TH':
            print(f"  {cell_type}: positives skipped "
                  f"(non-circadian outgroup per Abruzzi 2017)")
            pos_by_ct['TH'] = 0
            continue

        df_sheet = pd.read_excel(abruzzi_xlsx_path, sheet_name=sheet_name)
        sym_col = df_sheet.columns[0]

        cyc_col = next(
            (c for c in df_sheet.columns if 'cycling' in str(c).lower()), None
        )
        jtk_col = next(
            (c for c in df_sheet.columns
             if str(c).strip().lower().startswith('jtk p')), None
        )
        if cyc_col is None:
            raise ValueError(f"No 'cycling?' column in sheet '{sheet_name}'")
        if jtk_col is None:
            raise ValueError(f"No 'JTK p-value' column in sheet '{sheet_name}'")

        cycling_str = df_sheet[cyc_col].fillna('').astype(str).str.strip()
        if hc_only:
            df_hc = df_sheet[cycling_str == 'HC-cycler'].copy()
        else:
            df_hc = df_sheet[
                cycling_str.isin(['HC-cycler', 'LC-cycler'])
            ].copy()

        print(f"  {cell_type}: {len(df_hc)} HC-cyclers "
              f"({'hc_only' if hc_only else 'hc+lc'})")

        # Deterministic cap: sort by JTK p-value ascending (most rhythmic first)
        df_hc = df_hc.sort_values(jtk_col, ascending=True, na_position='last')
        if len(df_hc) > max_positives_per_cell_type:
            df_hc = df_hc.iloc[:max_positives_per_cell_type]
            print(f"    Capped to {max_positives_per_cell_type} "
                  f"by lowest JTK p-value")

        n_built = 0
        for _, row in df_hc.iterrows():
            gene = str(row[sym_col]).strip()
            if not gene or gene.lower() == 'nan':
                continue

            try:
                r1_vals = [float(row[c]) for c in zt_rep1]
                r2_vals = [float(row[c]) for c in zt_rep2]
            except (KeyError, ValueError, TypeError):
                continue
            if any(np.isnan(v) for v in r1_vals + r2_vals):
                continue

            var_name = f'var_{instance_id}'
            rows = []
            for zt, v1, v2 in zip(zt_rep1, r1_vals, r2_vals):
                rows.append({
                    'time': float(zt), 'condition': 'control',
                    'replicate': 'rep1', var_name: v1,
                })
                rows.append({
                    'time': float(zt), 'condition': 'control',
                    'replicate': 'rep2', var_name: v2,
                })

            metadata_list.append({
                'instance_id': instance_id,
                'variable': var_name,
                'signal_type': f'real_{gene}_{cell_type}_GSE77451',
                'is_rhythmic': 1,
                'n_timepoints': len(zt_rep1),
                'n_replicates': 2,
                'sampling_hours': 4.0,
                'snr': 0.0,
                'period': 24.0,
                'has_outliers': False,
                'source': 'biological',
                'gene': gene,
                'cell_type': cell_type,
            })
            dataframe_list.append(pd.DataFrame(rows))
            instance_id += 1
            n_built += 1

        pos_by_ct[cell_type] = n_built
        print(f"    Generated {n_built} positive instances for {cell_type}")

    # ------------------------------------------------------------------ #
    # Step 2: Negative instances from GEO series matrix                   #
    # ------------------------------------------------------------------ #
    print("\n[2/3] Generating NEGATIVE instances from GEO supplementary files...")
    neg_by_ct: Dict[str, int] = {ct: 0 for ct in CELL_CFG}

    _SUPPL_URL = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77451/"
        "suppl/GSE77451_{ct}_gene_expression.txt.gz"
    )

    def _load_gse77451_suppl(cell_type: str) -> Dict[str, list]:
        """Download and parse per-cell-type supplementary expression file.

        Format: col0=FlyBase transcript ID (ignored), col1=gene symbol,
        cols 2-7=rep1 expression (ZT order per CELL_CFG),
        cols 8-13=rep2 expression.  Multiple transcripts per gene: first row
        kept (highest-expressed canonical isoform in GEO file ordering).

        Returns {gene_symbol: [v0..v5_rep1, v0..v5_rep2]} (12 floats).
        """
        cache_base = str(geo_cache_dir) if geo_cache_dir is not None else GEO_DATA_DIR
        out_path = os.path.join(
            cache_base, f"GSE77451_{cell_type}_gene_expression.txt.gz"
        )
        url = _SUPPL_URL.format(ct=cell_type)
        download_file(url, out_path)

        gene_map: Dict[str, list] = {}
        opener = gzip.open if out_path.endswith('.gz') else open
        with opener(out_path, 'rt', encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh):
                line = line.rstrip('\n').rstrip('\r')
                if i == 0:           # header row — skip
                    continue
                parts = line.split('\t')
                if len(parts) < 14:  # need 2 ID cols + 12 expression cols
                    continue
                gene = parts[1].strip()
                if not gene or gene.lower() in ('', 'na', 'nan'):
                    continue
                try:
                    vals = [float(parts[c]) for c in range(2, 14)]
                except (ValueError, IndexError):
                    continue
                if gene not in gene_map:
                    gene_map[gene] = vals  # keep first transcript per gene
        print(f"    {cell_type}: {len(gene_map)} unique genes in supplementary file")
        return gene_map

    for cell_type, (sheet_name, zt_rep1, zt_rep2) in CELL_CFG.items():
        try:
            gene_map = _load_gse77451_suppl(cell_type)
        except Exception as e:
            print(f"  WARNING: Could not load {cell_type} supplementary file: {e}")
            continue

        gene_lower = {g.lower(): g for g in gene_map}

        # Genes that should be non-rhythmic in this cell type
        target_neg = set(NON_RHYTHMIC_GENES_FLY)
        if cell_type == 'TH':
            target_neg |= KNOWN_CIRCADIAN_GENES_FLY

        n_neg_ct = 0
        for gene in sorted(target_neg):
            if gene in gene_map:
                vals = gene_map[gene]
            elif gene.lower() in gene_lower:
                vals = gene_map[gene_lower[gene.lower()]]
            else:
                continue

            rep1_vals = vals[:6]
            rep2_vals = vals[6:]

            if any(np.isnan(v) for v in rep1_vals + rep2_vals):
                continue

            var_name = f'var_{instance_id}'
            rows = []
            for zt, v1, v2 in zip(zt_rep1, rep1_vals, rep2_vals):
                rows.append({
                    'time': float(zt), 'condition': 'control',
                    'replicate': 'rep1', var_name: v1,
                })
                rows.append({
                    'time': float(zt), 'condition': 'control',
                    'replicate': 'rep2', var_name: v2,
                })

            metadata_list.append({
                'instance_id': instance_id,
                'variable': var_name,
                'signal_type': f'real_{gene}_{cell_type}_GSE77451',
                'is_rhythmic': 0,
                'n_timepoints': len(zt_rep1),
                'n_replicates': 2,
                'sampling_hours': 4.0,
                'snr': 0.0,
                'period': 0.0,
                'has_outliers': False,
                'source': 'biological',
                'gene': gene,
                'cell_type': cell_type,
            })
            dataframe_list.append(pd.DataFrame(rows))
            instance_id += 1
            n_neg_ct += 1

        neg_by_ct[cell_type] = n_neg_ct
        print(f"  {cell_type}: {n_neg_ct} negative instances")

    # ------------------------------------------------------------------ #
    # Step 3: Sanity checks                                               #
    # ------------------------------------------------------------------ #
    print("\n[3/3] Sanity checks...")
    n_pos_total = sum(1 for m in metadata_list if m['is_rhythmic'] == 1)
    n_neg_total = sum(1 for m in metadata_list if m['is_rhythmic'] == 0)
    th_pos = sum(
        1 for m in metadata_list
        if m.get('cell_type') == 'TH' and m['is_rhythmic'] == 1
    )
    print(f"  Total: {len(metadata_list)} instances  "
          f"(pos={n_pos_total}, neg={n_neg_total})")
    print(f"  TH positives (must be 0): {th_pos}")
    assert th_pos == 0, f"SANITY FAIL: TH positives={th_pos}"

    # Within-cell-type label conflict check
    seen_labels: Dict[tuple, Set[int]] = {}
    for m in metadata_list:
        key = (m['gene'], m.get('cell_type', ''))
        seen_labels.setdefault(key, set()).add(m['is_rhythmic'])
    conflicts = [
        f"{g}@{ct}"
        for (g, ct), lbls in seen_labels.items()
        if len(lbls) > 1
    ]
    if conflicts:
        print(f"  Within-cell-type label conflicts (unexpected): {conflicts}")
    else:
        print(f"  Label conflict check: OK (no within-cell-type conflicts)")

    hk_neg_genes = {
        m['gene'] for m in metadata_list
        if m['is_rhythmic'] == 0 and m['gene'] in NON_RHYTHMIC_GENES_FLY
    }
    pct_hk = (len(hk_neg_genes) / len(NON_RHYTHMIC_GENES_FLY) * 100
               if NON_RHYTHMIC_GENES_FLY else 0.0)
    print(f"  Housekeeping gene coverage in negatives: "
          f"{len(hk_neg_genes)}/{len(NON_RHYTHMIC_GENES_FLY)} ({pct_hk:.0f}%)")

    for ct_name in ('LNv', 'LNd', 'DN1'):
        p = pos_by_ct.get(ct_name, 0)
        n = neg_by_ct.get(ct_name, 0)
        print(f"  {ct_name}: {p} pos, {n} neg")
    print(f"  TH: 0 pos (skipped), {neg_by_ct.get('TH', 0)} neg")

    # Print first 5 instances for user sanity-check
    print("\n  First 5 instances:")
    for m in metadata_list[:5]:
        df_i = dataframe_list[metadata_list.index(m)]
        print(f"    id={m['instance_id']} gene={m['gene']} "
              f"ct={m.get('cell_type')} label={m['is_rhythmic']} "
              f"nrows={len(df_i)}")

    return metadata_list, dataframe_list


# =============================================================================
# DATASET 4: GSE39445 (Möller-Levet 2013, Human whole blood)
# =============================================================================

def generate_from_GSE39445(
    moller_xlsx_path: Path,
    geo_cache_dir: Optional[Path] = None,
    starting_instance_id: int = 20000,
    max_per_class: int = 800,
    seed: int = 42,
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """
    Generate training instances from GSE39445 (Möller-Levet et al. 2013, PNAS).

    Label strategy:
      Strong positive (circ_control=1, circ_sr=1):
        → control instance (label=1) + SR instance (label=1)
      Hard positive (circ_control=1, circ_sr=0):
        → control instance only (label=1); SR not included
      Negative (circ_control=0, circ_sr=0, sleep_effect=0):
        → control instance (label=0) + SR instance (label=0)
      Forced positives: KNOWN_CIRCADIAN_GENES_HUMAN (override null label)
      Forced negatives: NON_RHYTHMIC_GENES_HUMAN (override null label)

    Expression: subjects pooled (averaged) per condition per timepoint.
    Timepoints are rounded to the nearest 3-hour bin before averaging.

    Cross-species group safety: human symbols (UPPERCASE HGNC) are case-
    sensitively distinct from mouse ('Per1') and fly ('per') symbols.

    Returns (metadata_list, dataframes_list) in ChronoScope format.
    """
    print("=" * 70)
    print("GSE39445: Human Blood Transcriptome (Möller-Levet et al. 2013)")
    print("=" * 70)

    moller_xlsx_path = Path(moller_xlsx_path)
    rng = np.random.RandomState(seed)
    cache_str = str(geo_cache_dir) if geo_cache_dir is not None else None

    # ------------------------------------------------------------------ #
    # Step 1: Download and parse GEO series matrix (large: ~438 samples)  #
    # ------------------------------------------------------------------ #
    print("\n[1/6] Downloading GSE39445 series matrix (large file, may take time)...")
    matrix_path = download_geo_series_matrix('GSE39445', cache_dir=cache_str)
    expr_df, sinfo = parse_series_matrix(matrix_path)
    print(f"  Matrix: {len(expr_df)} probes × {len(expr_df.columns)} samples")

    # ------------------------------------------------------------------ #
    # Step 2: Parse sample metadata (condition + timepoint)               #
    # ------------------------------------------------------------------ #
    print("\n[2/6] Parsing sample metadata (condition, timepoint)...")

    def _extract_condition(meta: dict) -> Optional[str]:
        """Return 'control' or 'sleep_restriction' from sample characteristics."""
        for key, vals in meta.items():
            if ('characteristic' not in key.lower()
                    and 'title' not in key.lower()):
                continue
            for v in vals:
                vl = v.lower()
                if any(kw in vl for kw in
                       ('sleep sufficient', 'normal sleep', '10 h sleep',
                        'control', 'non-sleep deprived')):
                    return 'control'
                if any(kw in vl for kw in
                       ('sleep restrict', 'restricted', 'sleep depri',
                        '5.7 h sleep', 'sleep-restricted')):
                    return 'sleep_restriction'
        return None

    sample_times = extract_timepoints_from_samples(sinfo)
    sample_cond: Dict[str, str] = {}
    for sid in expr_df.columns:
        if sid in sinfo:
            cond = _extract_condition(sinfo[sid])
            if cond:
                sample_cond[sid] = cond

    ctrl_sids = [s for s, c in sample_cond.items() if c == 'control']
    sr_sids   = [s for s, c in sample_cond.items() if c == 'sleep_restriction']
    print(f"  Control samples:           {len(ctrl_sids)}")
    print(f"  Sleep-restriction samples: {len(sr_sids)}")
    print(f"  Samples with timepoints:   {len(sample_times)}")

    if not ctrl_sids:
        # Diagnosis: print first 5 sample titles for user to fix the parser
        print("\n  WARNING: No control samples found. Printing raw metadata "
              "for the first 5 samples for diagnosis:")
        for sid in list(expr_df.columns)[:5]:
            meta = sinfo.get(sid, {})
            for k, vs in meta.items():
                if 'title' in k.lower() or 'characteristic' in k.lower():
                    print(f"    {sid} | {k}: {vs[:3]}")
        raise RuntimeError(
            "Cannot classify GSE39445 samples by sleep condition. "
            "Update _extract_condition() in generate_from_GSE39445() "
            "to match the actual Sample_characteristics_ch1 values."
        )

    # ------------------------------------------------------------------ #
    # Step 3: Map probes → genes (GPL15331, custom Agilent array)         #
    # ------------------------------------------------------------------ #
    print("\n[3/6] Mapping probes to gene symbols (GPL15331)...")
    try:
        annot_path = download_geo_platform_annot('GPL15331', cache_dir=cache_str)
        probe_to_gene = parse_platform_annotation(annot_path)
        gene_expr = map_expression_to_genes(expr_df, probe_to_gene)
    except Exception as e:
        print(f"  WARNING: GPL15331 annotation failed ({e}). "
              f"Trying row IDs as gene symbols directly.")
        gene_expr = expr_df.copy()
        gene_expr.index.name = 'gene'

    print(f"  Gene-level expression: {len(gene_expr)} genes")

    # ------------------------------------------------------------------ #
    # Step 4: Load Möller-Levet circadian labels                          #
    # ------------------------------------------------------------------ #
    print("\n[4/6] Loading Möller-Levet 2013 labels...")
    ml_df = load_moller_levet_labels(moller_xlsx_path)
    ml_by_gene = ml_df.set_index('gene')

    # ------------------------------------------------------------------ #
    # Step 5: Pool time-series per (gene, condition)                      #
    # ------------------------------------------------------------------ #
    print("\n[5/6] Classifying genes and pooling time-series...")

    BIN_HOURS = 3.0  # round timepoints to nearest 3-hour bin

    def _pool(gene_row: pd.Series, sids: List[str]) -> Optional[tuple]:
        """Average expression across subjects per timepoint bin.

        Returns (unique_times_arr, avg_values_arr) or None if < 6 unique bins.
        """
        expr_dict = gene_row.to_dict()
        by_time: Dict[float, List[float]] = {}
        for s in sids:
            if s not in sample_times:
                continue
            val = expr_dict.get(s, np.nan)
            if pd.isna(val):
                continue
            t_bin = round(float(sample_times[s]) / BIN_HOURS) * BIN_HOURS
            by_time.setdefault(t_bin, []).append(float(val))
        if len(by_time) < 6:
            return None
        t_arr = np.array(sorted(by_time.keys()))
        v_arr = np.array([float(np.mean(by_time[t])) for t in t_arr])
        return t_arr, v_arr

    all_genes = set(gene_expr.index)

    # Build candidate gene lists
    strong_pos: List[str] = []   # circ in both
    hard_pos:   List[str] = []   # circ in control only
    neg_cands:  List[str] = []   # clean non-rhythmic

    for gene in all_genes:
        gene_str = str(gene)
        gene_up = gene_str.upper()

        # Force-include clock/housekeeping genes regardless of ML label
        if gene_str in KNOWN_CIRCADIAN_GENES_HUMAN or gene_up in KNOWN_CIRCADIAN_GENES_HUMAN:
            strong_pos.append(gene)
            continue
        if gene_str in NON_RHYTHMIC_GENES_HUMAN or gene_up in NON_RHYTHMIC_GENES_HUMAN:
            neg_cands.append(gene)
            continue

        # Use Möller-Levet label (match by gene symbol exactly)
        if gene_str not in ml_by_gene.index:
            continue
        row_ml = ml_by_gene.loc[gene_str]
        cc = int(row_ml['circ_control'])
        cs = int(row_ml['circ_sr'])
        se = int(row_ml['sleep_effect'])

        if cc == 1 and cs == 1:
            strong_pos.append(gene)
        elif cc == 1 and cs == 0:
            hard_pos.append(gene)
        elif cc == 0 and cs == 0 and se == 0:
            neg_cands.append(gene)
        # else: ambiguous — skip

    print(f"  Strong positives (circ in both):   {len(strong_pos)}")
    print(f"  Hard positives (ctrl only):        {len(hard_pos)}")
    print(f"  Negative candidates:               {len(neg_cands)}")

    # ------------------------------------------------------------------ #
    # Step 6: Build instances with caps                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[6/6] Building instances (cap: {max_per_class} per class)...")

    metadata_list: List[Dict] = []
    dataframe_list: List[pd.DataFrame] = []
    instance_id = starting_instance_id

    def _add_instance(gene: str, condition_label: str, label: int,
                      t_arr: np.ndarray, v_arr: np.ndarray) -> bool:
        nonlocal instance_id
        if len(t_arr) < 6:
            return False
        var_name = f'var_{instance_id}'
        df_inst = pd.DataFrame([
            {'time': float(t), 'condition': 'control',
             'replicate': 'rep1', var_name: float(v)}
            for t, v in zip(t_arr, v_arr)
        ])
        metadata_list.append({
            'instance_id': instance_id,
            'variable': var_name,
            'signal_type': f'real_{gene}_{condition_label}_GSE39445',
            'is_rhythmic': label,
            'n_timepoints': len(t_arr),
            'n_replicates': 1,
            'sampling_hours': BIN_HOURS,
            'snr': 0.0,
            'period': 24.0 if label == 1 else 0.0,
            'has_outliers': False,
            'source': 'biological',
            'gene': str(gene),
        })
        dataframe_list.append(df_inst)
        instance_id += 1
        return True

    # Shuffle for diversity under capping
    rng.shuffle(strong_pos)
    rng.shuffle(hard_pos)
    rng.shuffle(neg_cands)

    n_pos = 0
    # Strong positives → control + SR instances
    for gene in strong_pos:
        if n_pos >= max_per_class:
            break
        gene_row = gene_expr.loc[gene]
        for cond_name, sids in [('control', ctrl_sids),
                                 ('sleep_restriction', sr_sids)]:
            if n_pos >= max_per_class:
                break
            pooled = _pool(gene_row, sids)
            if pooled and _add_instance(gene, cond_name, 1, *pooled):
                n_pos += 1

    # Hard positives → control instance only
    for gene in hard_pos:
        if n_pos >= max_per_class:
            break
        gene_row = gene_expr.loc[gene]
        pooled = _pool(gene_row, ctrl_sids)
        if pooled and _add_instance(gene, 'control', 1, *pooled):
            n_pos += 1

    n_neg = 0
    # Negatives → control + SR instances
    for gene in neg_cands:
        if n_neg >= max_per_class:
            break
        gene_row = gene_expr.loc[gene]
        for cond_name, sids in [('control', ctrl_sids),
                                 ('sleep_restriction', sr_sids)]:
            if n_neg >= max_per_class:
                break
            pooled = _pool(gene_row, sids)
            if pooled and _add_instance(gene, cond_name, 0, *pooled):
                n_neg += 1

    print(f"\n  Generated {len(metadata_list)} instances")
    print(f"    Positives: {n_pos}")
    print(f"    Negatives: {n_neg}")

    # Sanity checks
    clock_in_pos = sorted({
        m['gene'] for m in metadata_list
        if m['is_rhythmic'] == 1
        and m['gene'] in KNOWN_CIRCADIAN_GENES_HUMAN
    })
    hk_in_neg = sorted({
        m['gene'] for m in metadata_list
        if m['is_rhythmic'] == 0
        and m['gene'] in NON_RHYTHMIC_GENES_HUMAN
    })
    core_expected = ['ARNTL', 'NR1D1', 'PER1', 'PER2']
    missing_core = [g for g in core_expected if g not in clock_in_pos]
    if missing_core:
        print(f"  NOTE: Core clock genes not in positives: {missing_core} "
              f"(may not be in expression matrix)")
    else:
        print(f"  Core clock genes confirmed in positives: {core_expected}")

    hk_expected = ['ACTB', 'GAPDH']
    missing_hk = [g for g in hk_expected if g not in hk_in_neg]
    if missing_hk:
        print(f"  NOTE: Housekeeping genes not in negatives: {missing_hk}")
    else:
        print(f"  Housekeeping genes confirmed in negatives: {hk_expected}")

    non_upper = [
        m['gene'] for m in metadata_list
        if m['gene'] != m['gene'].upper() and m['gene'] in ml_by_gene.index
    ]
    if non_upper:
        print(f"  NOTE: {len(non_upper)} non-uppercase identifiers "
              f"(RefSeq/other accessions): {non_upper[:5]}")

    # Print first 5 instances for sanity check
    print("\n  First 5 instances:")
    for m in metadata_list[:5]:
        df_i = dataframe_list[metadata_list.index(m)]
        print(f"    id={m['instance_id']} gene={m['gene']} "
              f"cond={m['signal_type'].split('_')[-2]} "
              f"label={m['is_rhythmic']} nrows={len(df_i)}")

    return metadata_list, dataframe_list


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
