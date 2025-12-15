"""
rosbash_data_loader.py

Clase helper para cargar y consultar el dataset de Rosbash procesado.
Diseñada para ser usada en una GUI de PySide6 sin necesidad de scanpy.

Dependencias mínimas:
    - h5py
    - numpy
    - pandas (opcional, para DataFrames)

Ejemplo de uso:
    from rosbash_data_loader import RosbashDataLoader
    
    loader = RosbashDataLoader('rosbash_processed.h5')
    
    # Obtener lista de genes y clusters
    genes = loader.get_gene_names()
    clusters = loader.get_clusters()
    
    # Obtener expresión de un gen en un cluster específico
    expr_df = loader.get_gene_expression(
        gene='tim',
        cluster='25:l_LNv',
        condition='LD',
        use_log=False  # TP10K raw
    )
    
    # Calcular expresión media por timepoint
    time_series = loader.get_timeseries_mean(
        gene='tim',
        cluster='25:l_LNv',
        condition='LD'
    )
"""

import h5py
import numpy as np
from typing import Optional, List, Dict, Union
from functools import lru_cache


class RosbashDataLoader:
    """
    Cargador de datos del dataset de Rosbash CLK856.
    Optimizado para consultas rápidas en una GUI.
    """
    
    def __init__(self, h5_path: str):
        """
        Inicializa el cargador.
        
        Args:
            h5_path: Ruta al archivo HDF5 procesado
        """
        self.h5_path = h5_path
        self._h5_file = None
        
        # Cargar metadata en memoria (es pequeña)
        self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadata en memoria para acceso rápido."""
        with h5py.File(self.h5_path, 'r') as f:
            # Info general
            self.n_cells = f['info'].attrs['n_cells']
            self.n_genes = f['info'].attrs['n_genes']
            
            # Nombres
            self._gene_names = self._decode_strings(f['genes/names'][:])
            self._cell_names = self._decode_strings(f['cells/names'][:])
            
            # Crear índice de genes para búsqueda rápida
            self._gene_to_idx = {gene: i for i, gene in enumerate(self._gene_names)}
            
            # Clusters y timepoints
            self._clusters = self._decode_strings(f['info/clusters'][:])
            self._timepoints_ld = self._decode_strings(f['info/timepoints_LD'][:])
            self._timepoints_dd = self._decode_strings(f['info/timepoints_DD'][:])
            
            # Metadata de células
            self._metadata = {
                'condition': self._decode_strings(f['metadata/condition'][:]),
                'cluster': self._decode_strings(f['metadata/cluster_full'][:]),
                'time': self._decode_strings(f['metadata/time'][:]),
                'time_numeric': f['metadata/time_numeric'][:],
                'cluster_id': f['metadata/cluster_id'][:],
                'cluster_name': self._decode_strings(f['metadata/cluster_name'][:]),
                'experiment': self._decode_strings(f['metadata/experiment'][:]),
            }
    
    @staticmethod
    def _decode_strings(arr):
        """Decodifica arrays de bytes a strings."""
        return [x.decode() if isinstance(x, bytes) else x for x in arr]
    
    def get_gene_names(self) -> List[str]:
        """Retorna lista de todos los genes disponibles."""
        return self._gene_names.copy()
    
    def search_genes(self, query: str, case_sensitive: bool = False) -> List[str]:
        """
        Busca genes que contengan el query.
        
        Args:
            query: Texto a buscar
            case_sensitive: Si True, búsqueda sensible a mayúsculas
            
        Returns:
            Lista de genes que matchean
        """
        if not case_sensitive:
            query = query.lower()
            return [g for g in self._gene_names if query in g.lower()]
        return [g for g in self._gene_names if query in g]
    
    def get_clusters(self) -> List[str]:
        """Retorna lista de clusters disponibles."""
        return self._clusters.copy()
    
    def get_cluster_names(self) -> List[str]:
        """Retorna solo los nombres de clusters (sin el número)."""
        return list(set(self._metadata['cluster_name']))
    
    def get_timepoints(self, condition: str = 'LD') -> List[str]:
        """
        Retorna timepoints para una condición.
        
        Args:
            condition: 'LD' o 'DD'
        """
        if condition.upper() == 'LD':
            return self._timepoints_ld.copy()
        return self._timepoints_dd.copy()
    
    def get_conditions(self) -> List[str]:
        """Retorna condiciones disponibles."""
        return ['LD', 'DD']
    
    def _get_cell_mask(self, 
                       cluster: Optional[str] = None,
                       condition: Optional[str] = None,
                       timepoint: Optional[str] = None) -> np.ndarray:
        """
        Crea máscara booleana para filtrar células.
        
        Args:
            cluster: Cluster específico (ej: '25:l_LNv') o None para todos
            condition: 'LD' o 'DD' o None para ambos
            timepoint: Timepoint específico (ej: 'ZT02') o None para todos
            
        Returns:
            Array booleano con True para células que pasan el filtro
        """
        mask = np.ones(self.n_cells, dtype=bool)
        
        if cluster is not None:
            mask &= np.array([c == cluster for c in self._metadata['cluster']])
        
        if condition is not None:
            mask &= np.array([c == condition for c in self._metadata['condition']])
        
        if timepoint is not None:
            mask &= np.array([t == timepoint for t in self._metadata['time']])
        
        return mask
    
    def get_gene_expression(self,
                           gene: str,
                           cluster: Optional[str] = None,
                           condition: Optional[str] = None,
                           timepoint: Optional[str] = None,
                           use_log: bool = False) -> Dict:
        """
        Obtiene expresión de un gen para células filtradas.
        
        Args:
            gene: Nombre del gen
            cluster: Filtrar por cluster
            condition: Filtrar por condición ('LD' o 'DD')
            timepoint: Filtrar por timepoint
            use_log: Si True, retorna valores log1p; si False, TP10K
            
        Returns:
            Dict con:
                - 'expression': array de expresión
                - 'cells': lista de nombres de células
                - 'times': lista de timepoints
                - 'time_numeric': array de horas numéricas
        """
        if gene not in self._gene_to_idx:
            raise ValueError(f"Gen '{gene}' no encontrado. Use search_genes() para buscar.")
        
        gene_idx = self._gene_to_idx[gene]
        mask = self._get_cell_mask(cluster, condition, timepoint)
        
        dataset = 'expression/log1p' if use_log else 'expression/tp10k'
        
        with h5py.File(self.h5_path, 'r') as f:
            # Leer solo la columna del gen para las células filtradas
            expression = f[dataset][mask, gene_idx]
        
        # Filtrar metadata correspondiente
        indices = np.where(mask)[0]
        
        return {
            'expression': expression,
            'cells': [self._cell_names[i] for i in indices],
            'times': [self._metadata['time'][i] for i in indices],
            'time_numeric': self._metadata['time_numeric'][mask],
            'condition': [self._metadata['condition'][i] for i in indices],
            'cluster': [self._metadata['cluster'][i] for i in indices],
        }
    
    def get_timeseries_mean(self,
                           gene: str,
                           cluster: Optional[str] = None,
                           condition: str = 'LD',
                           use_log: bool = False) -> Dict:
        """
        Calcula expresión media por timepoint (para graficar time series).
        
        Args:
            gene: Nombre del gen
            cluster: Filtrar por cluster (None = todos)
            condition: 'LD' o 'DD'
            use_log: Si True, usa valores log1p
            
        Returns:
            Dict con:
                - 'timepoints': lista ordenada de timepoints
                - 'time_numeric': horas numéricas correspondientes
                - 'mean': array de medias
                - 'sem': array de errores estándar
                - 'std': array de desviaciones estándar
                - 'n_cells': número de células por timepoint
        """
        timepoints = self.get_timepoints(condition)
        
        means = []
        sems = []
        stds = []
        n_cells = []
        time_numeric = []
        
        for tp in timepoints:
            data = self.get_gene_expression(
                gene=gene,
                cluster=cluster,
                condition=condition,
                timepoint=tp,
                use_log=use_log
            )
            
            expr = data['expression']
            n = len(expr)
            
            if n > 0:
                mean = np.mean(expr)
                std = np.std(expr)
                sem = std / np.sqrt(n) if n > 1 else 0
                # Extraer hora del timepoint (ZT02 -> 2, CT14 -> 14)
                hour = int(tp[2:])
            else:
                mean = np.nan
                std = np.nan
                sem = np.nan
                hour = int(tp[2:])
            
            means.append(mean)
            stds.append(std)
            sems.append(sem)
            n_cells.append(n)
            time_numeric.append(hour)
        
        return {
            'timepoints': timepoints,
            'time_numeric': np.array(time_numeric),
            'mean': np.array(means),
            'sem': np.array(sems),
            'std': np.array(stds),
            'n_cells': np.array(n_cells),
        }
    
    def get_expression_matrix(self,
                             genes: List[str],
                             cluster: Optional[str] = None,
                             condition: Optional[str] = None,
                             use_log: bool = False) -> Dict:
        """
        Obtiene matriz de expresión para múltiples genes.
        
        Args:
            genes: Lista de genes
            cluster: Filtrar por cluster
            condition: Filtrar por condición
            use_log: Si True, usa valores log1p
            
        Returns:
            Dict con:
                - 'matrix': array 2D (cells x genes)
                - 'genes': lista de genes (columnas)
                - 'cells': lista de células (filas)
                - 'metadata': dict con metadata de células
        """
        gene_indices = []
        valid_genes = []
        for gene in genes:
            if gene in self._gene_to_idx:
                gene_indices.append(self._gene_to_idx[gene])
                valid_genes.append(gene)
        
        if not gene_indices:
            raise ValueError("Ninguno de los genes fue encontrado.")
        
        mask = self._get_cell_mask(cluster, condition)
        dataset = 'expression/log1p' if use_log else 'expression/tp10k'
        
        with h5py.File(self.h5_path, 'r') as f:
            # Leer todas las columnas de los genes de interés
            matrix = f[dataset][mask][:, gene_indices]
        
        indices = np.where(mask)[0]
        
        return {
            'matrix': matrix,
            'genes': valid_genes,
            'cells': [self._cell_names[i] for i in indices],
            'metadata': {
                key: [self._metadata[key][i] for i in indices]
                for key in self._metadata
            }
        }
    
    def get_cells_per_cluster_timepoint(self, condition: str = 'LD') -> Dict:
        """
        Retorna el número de células por cluster y timepoint.
        Útil para verificar cobertura de datos.
        
        Args:
            condition: 'LD' o 'DD'
            
        Returns:
            Dict[cluster][timepoint] -> n_cells
        """
        timepoints = self.get_timepoints(condition)
        result = {}
        
        for cluster in self._clusters:
            result[cluster] = {}
            for tp in timepoints:
                mask = self._get_cell_mask(cluster, condition, tp)
                result[cluster][tp] = int(np.sum(mask))
        
        return result
    
    def get_all_metadata(self):
        """
        Retorna DataFrame con toda la metadata de las células.
        Requiere pandas.
        """
        try:
            import pandas as pd
            df = pd.DataFrame({
                'cell_name': self._cell_names,
                **self._metadata
            })
            return df
        except ImportError:
            return self._metadata


# Ejemplo de uso
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python rosbash_data_loader.py <archivo.h5>")
        sys.exit(1)
    
    loader = RosbashDataLoader(sys.argv[1])
    
    print(f"Dataset cargado: {loader.n_cells} células, {loader.n_genes} genes")
    print(f"Clusters: {len(loader.get_clusters())}")
    print(f"Timepoints LD: {loader.get_timepoints('LD')}")
    print(f"Timepoints DD: {loader.get_timepoints('DD')}")
    
    # Ejemplo: time series de 'tim' en l-LNv
    print("\n=== Ejemplo: 'tim' en l-LNv (LD) ===")
    ts = loader.get_timeseries_mean('tim', cluster='25:l_LNv', condition='LD')
    for tp, mean, sem, n in zip(ts['timepoints'], ts['mean'], ts['sem'], ts['n_cells']):
        print(f"  {tp}: {mean:.2f} ± {sem:.2f} TP10K (n={n})")
    
    # Buscar genes circadianos
    print("\n=== Genes que contienen 'per' ===")
    per_genes = loader.search_genes('per')
    print(f"  {per_genes[:10]}...")
