"""
Rosbash Dataset Loader
======================

Handles loading and querying the preprocessed Rosbash CLK856 circadian neuron dataset.
The dataset contains single-cell RNA sequencing data from Drosophila clock neurons.

Paper: "A transcriptomic taxonomy of Drosophila circadian neurons around the clock"
       Elife 2021

HDF5 Structure:
    /expression/log1p      - Log-normalized expression (cells x genes)
    /expression/tp10k      - TP10K normalized expression (cells x genes)
    /genes/names           - Gene names
    /cells/names           - Cell barcodes
    /metadata/...          - Cell metadata columns
    /info/...              - Dataset information
"""

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import warnings

import pandas as pd
import numpy as np

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not installed. Rosbash dataset loading will not be available.")


@dataclass
class RosbashDatasetInfo:
    """Information about the loaded Rosbash dataset."""
    filepath: str
    n_cells: int
    n_genes: int
    clusters: List[str]
    cluster_ids: List[int]
    cluster_names: List[str]
    conditions: List[str]  # LD, DD
    timepoints_ld: List[str]
    timepoints_dd: List[str]
    normalization: str
    creation_date: str


@dataclass
class GeneExpressionData:
    """Container for gene expression data extracted for analysis."""
    gene: str
    condition: str
    cluster: str
    times: np.ndarray  # Numeric time values
    values: np.ndarray  # Expression values (log1p normalized)
    cell_ids: List[str]
    n_cells_per_timepoint: Dict[str, int]


class RosbashDataLoader:
    """
    Loader for the preprocessed Rosbash circadian neuron dataset.
    
    This loader provides efficient access to single-cell RNA-seq data
    without requiring scanpy in the main application.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the Rosbash data loader.
        
        Args:
            filepath: Path to the preprocessed HDF5 file
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for loading the Rosbash dataset")
        
        self._filepath: Optional[str] = None
        self._h5file: Optional[h5py.File] = None
        self._dataset_info: Optional[RosbashDatasetInfo] = None
        
        # Cached data
        self._gene_names: Optional[List[str]] = None
        self._cell_names: Optional[List[str]] = None
        self._metadata: Optional[pd.DataFrame] = None
        
        if filepath:
            self.load(filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load the HDF5 dataset file.
        
        Args:
            filepath: Path to the HDF5 file
        """
        self._filepath = filepath
        
        # Open file in read mode
        self._h5file = h5py.File(filepath, 'r')
        
        # Load gene and cell names
        self._gene_names = [
            name.decode('utf-8') if isinstance(name, bytes) else name
            for name in self._h5file['genes/names'][:]
        ]
        
        self._cell_names = [
            name.decode('utf-8') if isinstance(name, bytes) else name
            for name in self._h5file['cells/names'][:]
        ]
        
        # Load metadata
        self._load_metadata()
        
        # Generate dataset info
        self._generate_dataset_info()
    
    def _load_metadata(self) -> None:
        """Load cell metadata from HDF5."""
        if self._h5file is None:
            return
        
        meta_group = self._h5file['metadata']
        meta_dict = {}
        
        for key in meta_group.keys():
            data = meta_group[key][:]
            # Decode bytes to strings if necessary
            if data.dtype.kind in ['S', 'O']:
                data = [
                    d.decode('utf-8') if isinstance(d, bytes) else d
                    for d in data
                ]
            meta_dict[key] = data
        
        self._metadata = pd.DataFrame(meta_dict, index=self._cell_names)
    
    def _generate_dataset_info(self) -> None:
        """Generate dataset summary information."""
        if self._h5file is None:
            return
        
        info = self._h5file['info']
        
        # Get clusters
        clusters = [
            c.decode('utf-8') if isinstance(c, bytes) else c
            for c in info['clusters'][:]
        ]
        
        # Parse cluster IDs and names
        cluster_ids = []
        cluster_names = []
        for c in clusters:
            parts = c.split(':')
            cluster_ids.append(int(parts[0]))
            cluster_names.append(parts[1] if len(parts) > 1 else c)
        
        # Get timepoints
        timepoints_ld = [
            t.decode('utf-8') if isinstance(t, bytes) else t
            for t in info['timepoints_LD'][:]
        ]
        
        timepoints_dd = [
            t.decode('utf-8') if isinstance(t, bytes) else t
            for t in info['timepoints_DD'][:]
        ]
        
        self._dataset_info = RosbashDatasetInfo(
            filepath=self._filepath,
            n_cells=info.attrs['n_cells'],
            n_genes=info.attrs['n_genes'],
            clusters=clusters,
            cluster_ids=cluster_ids,
            cluster_names=cluster_names,
            conditions=['LD', 'DD'],
            timepoints_ld=timepoints_ld,
            timepoints_dd=timepoints_dd,
            normalization=info.attrs.get('normalization', 'TP10K'),
            creation_date=info.attrs.get('creation_date', 'Unknown')
        )
    
    def close(self) -> None:
        """Close the HDF5 file."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def get_dataset_info(self) -> Optional[RosbashDatasetInfo]:
        """Get dataset summary information."""
        return self._dataset_info
    
    def get_gene_names(self) -> List[str]:
        """Get list of all gene names."""
        return self._gene_names.copy() if self._gene_names else []
    
    def get_clusters(self) -> List[str]:
        """Get list of cluster identifiers (e.g., '1:LNd')."""
        if self._dataset_info:
            return self._dataset_info.clusters.copy()
        return []
    
    def get_cluster_names(self) -> List[str]:
        """Get list of cluster names only (e.g., 'LNd')."""
        if self._dataset_info:
            return self._dataset_info.cluster_names.copy()
        return []
    
    def get_conditions(self) -> List[str]:
        """Get list of conditions (LD, DD)."""
        return ['LD', 'DD']
    
    def get_timepoints(self, condition: str = 'LD') -> List[str]:
        """Get timepoints for a specific condition."""
        if self._dataset_info is None:
            return []
        
        if condition.upper() == 'LD':
            return self._dataset_info.timepoints_ld.copy()
        elif condition.upper() == 'DD':
            return self._dataset_info.timepoints_dd.copy()
        else:
            return []
    
    def search_genes(self, query: str, max_results: int = 50) -> List[str]:
        """
        Search for genes matching a query string.
        
        Args:
            query: Search string (case-insensitive)
            max_results: Maximum number of results to return
        
        Returns:
            List of matching gene names
        """
        if not self._gene_names:
            return []
        
        query_lower = query.lower()
        matches = [
            gene for gene in self._gene_names
            if query_lower in gene.lower()
        ]
        
        # Sort by relevance (exact matches first, then starts-with, then contains)
        def sort_key(gene):
            gene_lower = gene.lower()
            if gene_lower == query_lower:
                return (0, gene)
            elif gene_lower.startswith(query_lower):
                return (1, gene)
            else:
                return (2, gene)
        
        matches.sort(key=sort_key)
        return matches[:max_results]
    
    def gene_exists(self, gene: str) -> bool:
        """Check if a gene exists in the dataset."""
        if not self._gene_names:
            return False
        return gene in self._gene_names
    
    # =========================================================================
    # DATA EXTRACTION
    # =========================================================================
    
    def get_gene_expression(
        self,
        gene: str,
        condition: str = 'LD',
        cluster: Optional[str] = None,
        use_log1p: bool = False
    ) -> GeneExpressionData:
        """
        Extract expression data for a specific gene.

        Args:
            gene: Gene name
            condition: 'LD' or 'DD'
            cluster: Cluster identifier (e.g., '1:LNd') or None for all clusters
            use_log1p: If True, use log1p normalized data; if False, use TP10K (default)
        
        Returns:
            GeneExpressionData object with expression values
        """
        if self._h5file is None:
            raise ValueError("No dataset loaded")
        
        if gene not in self._gene_names:
            raise ValueError(f"Gene '{gene}' not found in dataset")
        
        # Get gene index
        gene_idx = self._gene_names.index(gene)
        
        # Filter cells by condition and cluster
        mask = self._metadata['condition'] == condition.upper()
        
        if cluster is not None:
            mask = mask & (self._metadata['Idents'] == cluster)
        
        cell_indices = np.where(mask)[0]
        
        if len(cell_indices) == 0:
            raise ValueError(f"No cells found for condition={condition}, cluster={cluster}")
        
        # Get expression data
        expr_key = 'expression/log1p' if use_log1p else 'expression/tp10k'
        expression = self._h5file[expr_key][cell_indices, gene_idx]
        
        # Get time values
        filtered_meta = self._metadata.iloc[cell_indices]
        times_str = filtered_meta['time'].values
        times_numeric = filtered_meta['time_numeric'].values.astype(float)
        cell_ids = filtered_meta.index.tolist()
        
        # Count cells per timepoint
        n_cells_per_tp = filtered_meta['time'].value_counts().to_dict()
        
        return GeneExpressionData(
            gene=gene,
            condition=condition,
            cluster=cluster or "All",
            times=times_numeric,
            values=expression,
            cell_ids=cell_ids,
            n_cells_per_timepoint=n_cells_per_tp
        )
    
    def get_gene_expression_df(
        self,
        gene: str,
        condition: str = 'LD',
        cluster: Optional[str] = None,
        use_log1p: bool = False
    ) -> pd.DataFrame:
        """
        Get gene expression as a DataFrame suitable for analysis modules.

        Args:
            gene: Gene name
            condition: 'LD' or 'DD'
            cluster: Cluster identifier or None for all
            use_log1p: If True, use log1p; if False, use TP10K (default)
        
        Returns:
            DataFrame with columns: time, condition, [gene_name]
        """
        data = self.get_gene_expression(gene, condition, cluster, use_log1p)
        
        return pd.DataFrame({
            'time': data.times,
            'condition': data.condition,
            gene: data.values
        })
    
    def get_multiple_genes_df(
        self,
        genes: List[str],
        condition: str = 'LD',
        cluster: Optional[str] = None,
        use_log1p: bool = True
    ) -> pd.DataFrame:
        """
        Get expression data for multiple genes.
        
        Args:
            genes: List of gene names
            condition: 'LD' or 'DD'
            cluster: Cluster identifier or None
            use_log1p: Use log1p normalized data
        
        Returns:
            DataFrame with time, condition, and gene columns
        """
        if self._h5file is None:
            raise ValueError("No dataset loaded")
        
        # Validate genes
        valid_genes = [g for g in genes if g in self._gene_names]
        if not valid_genes:
            raise ValueError("None of the specified genes were found")
        
        # Get gene indices
        gene_indices = [self._gene_names.index(g) for g in valid_genes]
        
        # Filter cells
        mask = self._metadata['condition'] == condition.upper()
        if cluster is not None:
            mask = mask & (self._metadata['Idents'] == cluster)
        
        cell_indices = np.where(mask)[0]
        
        # Get expression for all genes
        expr_key = 'expression/log1p' if use_log1p else 'expression/tp10k'
        expression = self._h5file[expr_key][cell_indices, :][:, gene_indices]
        
        # Build DataFrame
        filtered_meta = self._metadata.iloc[cell_indices]
        
        df = pd.DataFrame({
            'time': filtered_meta['time_numeric'].values.astype(float),
            'condition': condition.upper()
        })
        
        for i, gene in enumerate(valid_genes):
            df[gene] = expression[:, i]
        
        return df
    
    def get_mean_expression_by_timepoint(
        self,
        gene: str,
        condition: str = 'LD',
        cluster: Optional[str] = None,
        use_log1p: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get mean expression and SEM grouped by timepoint.
        
        Args:
            gene: Gene name
            condition: 'LD' or 'DD'
            cluster: Cluster identifier or None
            use_log1p: Use log1p normalized data
        
        Returns:
            Tuple of (timepoints, mean_values, sem_values)
        """
        df = self.get_gene_expression_df(gene, condition, cluster, use_log1p)
        
        grouped = df.groupby('time')[gene]
        timepoints = np.array(sorted(df['time'].unique()))
        means = grouped.mean().loc[timepoints].values
        sems = grouped.sem().loc[timepoints].values
        
        return timepoints, means, sems
    
    # =========================================================================
    # CLUSTER ANALYSIS HELPERS
    # =========================================================================
    
    def get_cells_per_cluster(self, condition: str = 'LD') -> Dict[str, int]:
        """Get number of cells per cluster for a condition."""
        if self._metadata is None:
            return {}
        
        filtered = self._metadata[self._metadata['condition'] == condition.upper()]
        return filtered['Idents'].value_counts().to_dict()
    
    def get_cells_per_timepoint(
        self,
        condition: str = 'LD',
        cluster: Optional[str] = None
    ) -> Dict[str, int]:
        """Get number of cells per timepoint."""
        if self._metadata is None:
            return {}
        
        mask = self._metadata['condition'] == condition.upper()
        if cluster:
            mask = mask & (self._metadata['Idents'] == cluster)
        
        filtered = self._metadata[mask]
        return filtered['time'].value_counts().to_dict()
    
    # =========================================================================
    # CONVERSION TO ANALYSIS FORMAT
    # =========================================================================
    
    def prepare_for_cosinor(
        self,
        gene: str,
        condition: str = 'LD',
        cluster: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare data in format expected by CosinorAnalyzer.
        
        Returns DataFrame with columns: time, condition, [gene]
        where each row is a single cell (replicate).
        """
        return self.get_gene_expression_df(gene, condition, cluster, use_log1p=False)

    def prepare_for_circacompare(
        self,
        gene: str,
        condition1: str = 'LD',
        condition2: str = 'DD',
        cluster: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare data for CircaCompare analysis between two conditions.
        
        Returns DataFrame with both conditions combined.
        """
        df1 = self.get_gene_expression_df(gene, condition1, cluster, use_log1p=True)
        df2 = self.get_gene_expression_df(gene, condition2, cluster, use_log1p=True)
        
        return pd.concat([df1, df2], ignore_index=True)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_rosbash_dataset(filepath: str) -> RosbashDataLoader:
    """
    Convenience function to load the Rosbash dataset.
    
    Args:
        filepath: Path to the preprocessed HDF5 file
    
    Returns:
        Initialized RosbashDataLoader
    """
    return RosbashDataLoader(filepath)


def get_available_clock_genes() -> List[str]:
    """
    Get list of well-known Drosophila clock genes.
    
    Returns:
        List of clock gene names for quick selection in GUI
    """
    return [
        'per', 'tim', 'Clk', 'cyc', 'vri', 'Pdp1', 'cry',
        'cwo', 'dbt', 'sgg', 'Pdf', 'Dh31', 'Dh44', 'sNPF',
        'AstA', 'AstC', 'Nplp1', 'Tk', 'CCHa1'
    ]
