#!/usr/bin/env python3
"""
Script para procesar el dataset de Rosbash (CLK856 circadian neurons)
y generar un archivo HDF5 optimizado para usar en una GUI sin scanpy.

Paper: "A transcriptomic taxonomy of Drosophila circadian neurons around the clock"

Uso:
    python process_rosbash_dataset.py <input_dir> <annotations_csv> <output_h5>

Ejemplo:
    python process_rosbash_dataset.py ./dataset/ neuron_annotations.csv rosbash_processed.h5
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import h5py
from scipy import sparse
import scanpy as sc
from datetime import datetime


def parse_filename(filename):
    """
    Extrae metadata del nombre del archivo CSV.
    Ejemplo: GSM4768088_ZT14_20181231_AR01.csv -> (ZT14, 20181231, AR01)
    """
    basename = os.path.basename(filename)
    # Patrón: GSM*_TimePoint_Date_AR##.csv
    match = re.match(r'GSM\d+_([CZ]T\d+)_(\d+)_AR(\d+)\.csv', basename)
    if match:
        timepoint = match.group(1)
        date = match.group(2)
        ar = f"AR{match.group(3)}"
        return timepoint, date, ar
    return None, None, None


def load_csv_files(input_dir, verbose=True):
    """
    Carga todos los archivos CSV del directorio y los combina en una matriz.
    """
    csv_files = sorted(glob.glob(os.path.join(input_dir, "GSM*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos GSM*.csv en {input_dir}")
    
    if verbose:
        print(f"Encontrados {len(csv_files)} archivos CSV")
    
    all_data = []
    all_cells = []
    gene_names = None
    
    for i, csv_file in enumerate(csv_files):
        if verbose:
            print(f"  Cargando {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}", end='\r')
        
        df = pd.read_csv(csv_file, index_col=0)
        
        # Verificar que los genes sean consistentes
        if gene_names is None:
            gene_names = df.index.tolist()
        else:
            if df.index.tolist() != gene_names:
                print(f"\nWARNING: Genes no coinciden en {csv_file}")
                # Reindexar para asegurar consistencia
                df = df.reindex(gene_names, fill_value=0)
        
        # Remover el prefijo "X" de los nombres de células (añadido por R)
        cell_names = [col.lstrip('X') for col in df.columns]
        
        all_data.append(df.values)
        all_cells.extend(cell_names)
    
    if verbose:
        print(f"\n  Combinando matrices...")
    
    # Combinar todas las matrices
    combined_matrix = np.hstack(all_data)
    
    if verbose:
        print(f"  Matriz combinada: {len(gene_names)} genes x {len(all_cells)} células")
    
    return combined_matrix.T, gene_names, all_cells  # Transponer: cells x genes


def filter_cells_by_annotations(expression_matrix, cell_names, annotations_df, verbose=True):
    """
    Filtra las células para quedarse solo con las que están en las anotaciones.
    """
    # Crear un set de células anotadas para búsqueda rápida
    annotated_cells = set(annotations_df.index)
    
    # Encontrar índices de células que están en las anotaciones
    keep_indices = []
    kept_cells = []
    
    for i, cell in enumerate(cell_names):
        if cell in annotated_cells:
            keep_indices.append(i)
            kept_cells.append(cell)
    
    if verbose:
        print(f"Células totales: {len(cell_names)}")
        print(f"Células con anotación: {len(kept_cells)}")
        print(f"Células descartadas: {len(cell_names) - len(kept_cells)}")
    
    # Filtrar matriz
    filtered_matrix = expression_matrix[keep_indices, :]
    
    return filtered_matrix, kept_cells


def preprocess_with_scanpy(expression_matrix, gene_names, cell_names, verbose=True):
    """
    Preprocesa los datos usando scanpy siguiendo el protocolo de Rosbash:
    1. Normalización a TP10K (transcripts per 10,000)
    2. Log-transform: log(TP10K + 1)
    
    Returns:
        - normalized_data: matriz normalizada (log1p de TP10K)
        - tp10k_data: matriz en TP10K (sin log, para calcular mean etc.)
    """
    if verbose:
        print("Creando objeto AnnData...")
    
    # Crear AnnData (cells x genes)
    adata = sc.AnnData(
        X=expression_matrix.astype(np.float32),
        obs=pd.DataFrame(index=cell_names),
        var=pd.DataFrame(index=gene_names)
    )
    
    if verbose:
        print(f"  Shape: {adata.shape}")
        print("Normalizando a TP10K...")
    
    # Normalizar a 10,000 counts por célula (TP10K)
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Guardar copia de TP10K antes de log
    tp10k_data = adata.X.copy()
    
    if verbose:
        print("Aplicando log1p...")
    
    # Log transform
    sc.pp.log1p(adata)
    
    return adata.X, tp10k_data


def create_metadata_dataframe(cell_names, annotations_df):
    """
    Crea un DataFrame con toda la metadata de las células.
    """
    # Reindexar annotations para que coincida con el orden de cell_names
    metadata = annotations_df.loc[cell_names].copy()
    
    # Extraer información adicional del índice (nombre de célula)
    # Formato: 20181215_CLK856_LD_ZT14_AR07_ACAGGA
    metadata['cell_id'] = metadata.index
    
    # Parsear componentes del nombre de célula
    parsed = metadata.index.str.extract(
        r'(\d{8})_CLK856_(LD|DD)_([CZ]T\d+)_(AR\d+)_([A-Z]+)'
    )
    parsed.columns = ['date_str', 'condition_from_name', 'time_from_name', 'ar', 'barcode']
    
    # Agregar al metadata
    for col in parsed.columns:
        metadata[col] = parsed[col].values
    
    # Extraer hora numérica del timepoint (ej: ZT14 -> 14, CT02 -> 2)
    metadata['time_numeric'] = metadata['time'].str.extract(r'(\d+)').astype(int)
    
    # Agregar cluster_id (número) y cluster_name (nombre)
    metadata['cluster_full'] = metadata['Idents']
    cluster_split = metadata['Idents'].str.split(':', expand=True)
    metadata['cluster_id'] = cluster_split[0].astype(int)
    metadata['cluster_name'] = cluster_split[1]
    
    return metadata


def save_string_dataset(group, name, string_list):
    """Helper para guardar lista de strings en HDF5."""
    # Convertir todo a strings explícitamente
    str_array = [str(s) if s is not None else '' for s in string_list]
    dt = h5py.special_dtype(vlen=str)
    group.create_dataset(name, data=str_array, dtype=dt)


def save_to_hdf5(output_path, expression_log, expression_tp10k, gene_names, cell_names, metadata, verbose=True):
    """
    Guarda todos los datos en un archivo HDF5.
    
    Estructura del archivo:
    /expression/log1p      - Matriz de expresión log-normalizada (cells x genes)
    /expression/tp10k      - Matriz de expresión en TP10K (cells x genes)
    /genes/names           - Nombres de los genes
    /cells/names           - Nombres de las células
    /metadata/...          - Columnas de metadata
    /info/...              - Información del procesamiento
    """
    if verbose:
        print(f"Guardando en {output_path}...")
    
    with h5py.File(output_path, 'w') as f:
        # Grupo de expresión
        expr_grp = f.create_group('expression')
        
        # Guardar como sparse si es muy grande, sino como dense
        # Para este dataset (~2600 cells x 15000 genes) dense está bien
        expr_grp.create_dataset('log1p', data=expression_log, compression='gzip', compression_opts=4)
        expr_grp.create_dataset('tp10k', data=expression_tp10k, compression='gzip', compression_opts=4)
        
        # Genes
        genes_grp = f.create_group('genes')
        save_string_dataset(genes_grp, 'names', gene_names)
        
        # Células
        cells_grp = f.create_group('cells')
        save_string_dataset(cells_grp, 'names', cell_names)
        
        # Metadata
        meta_grp = f.create_group('metadata')
        for col in metadata.columns:
            data = metadata[col].values
            if data.dtype == object or data.dtype.kind in ['U', 'S', 'O']:
                save_string_dataset(meta_grp, col, data.tolist())
            elif pd.api.types.is_integer_dtype(data):
                meta_grp.create_dataset(col, data=data.astype(np.int64))
            elif pd.api.types.is_float_dtype(data):
                meta_grp.create_dataset(col, data=data.astype(np.float64))
            else:
                # Convertir a string si no sabemos qué es
                save_string_dataset(meta_grp, col, [str(x) for x in data])
        
        # Información del procesamiento
        info_grp = f.create_group('info')
        info_grp.attrs['creation_date'] = datetime.now().isoformat()
        info_grp.attrs['n_cells'] = len(cell_names)
        info_grp.attrs['n_genes'] = len(gene_names)
        info_grp.attrs['normalization'] = 'TP10K (target_sum=10000)'
        info_grp.attrs['log_transform'] = 'log1p'
        info_grp.attrs['source'] = 'Rosbash CLK856 dataset'
        info_grp.attrs['paper'] = 'A transcriptomic taxonomy of Drosophila circadian neurons around the clock'
        
        # Lista de clusters únicos
        unique_clusters = sorted(metadata['Idents'].unique())
        save_string_dataset(info_grp, 'clusters', unique_clusters)
        
        # Timepoints únicos por condición
        ld_times = sorted(metadata[metadata['condition'] == 'LD']['time'].unique())
        dd_times = sorted(metadata[metadata['condition'] == 'DD']['time'].unique())
        save_string_dataset(info_grp, 'timepoints_LD', ld_times)
        save_string_dataset(info_grp, 'timepoints_DD', dd_times)
    
    if verbose:
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Archivo guardado: {file_size:.1f} MB")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nError: Faltan argumentos")
        print("Uso: python process_rosbash_dataset.py <input_dir> <annotations_csv> <output_h5>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    annotations_path = sys.argv[2]
    output_path = sys.argv[3]
    
    print("=" * 60)
    print("Procesamiento de dataset Rosbash CLK856")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Annotations: {annotations_path}")
    print(f"Output: {output_path}")
    print()
    
    # 1. Cargar anotaciones
    print("1. Cargando anotaciones...")
    annotations = pd.read_csv(annotations_path, index_col=0)
    print(f"   {len(annotations)} células anotadas")
    print(f"   Clusters: {annotations['Idents'].nunique()}")
    print(f"   Condiciones: {annotations['condition'].unique().tolist()}")
    print()
    
    # 2. Cargar archivos CSV
    print("2. Cargando archivos CSV...")
    raw_matrix, gene_names, all_cell_names = load_csv_files(input_dir)
    print()
    
    # 3. Filtrar células
    print("3. Filtrando células por anotaciones...")
    filtered_matrix, cell_names = filter_cells_by_annotations(
        raw_matrix, all_cell_names, annotations
    )
    print()
    
    # 4. Preprocesar con scanpy
    print("4. Preprocesando (normalización TP10K + log1p)...")
    expression_log, expression_tp10k = preprocess_with_scanpy(
        filtered_matrix, gene_names, cell_names
    )
    print()
    
    # 5. Crear metadata
    print("5. Creando metadata...")
    metadata = create_metadata_dataframe(cell_names, annotations)
    print(f"   Columnas: {metadata.columns.tolist()}")
    print()
    
    # 6. Guardar
    print("6. Guardando archivo HDF5...")
    save_to_hdf5(output_path, expression_log, expression_tp10k, 
                 gene_names, cell_names, metadata)
    print()
    
    # 7. Verificación
    print("7. Verificación del archivo guardado:")
    with h5py.File(output_path, 'r') as f:
        print(f"   expression/log1p: {f['expression/log1p'].shape}")
        print(f"   expression/tp10k: {f['expression/tp10k'].shape}")
        print(f"   genes: {len(f['genes/names'])}")
        print(f"   cells: {len(f['cells/names'])}")
        print(f"   clusters: {len(f['info/clusters'])}")
    
    print()
    print("=" * 60)
    print("¡Procesamiento completado!")
    print("=" * 60)


if __name__ == "__main__":
    main()
