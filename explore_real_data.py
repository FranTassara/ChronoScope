"""
Script para explorar datos reales con CosinorPy
Permite seleccionar un archivo CSV y ejecutar cosinor.fit_group() o cosinor1.fit_group()
"""

import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
import sys

# Fix para compatibilidad con NumPy 2.0
# CosinorPy usa np.round_ que fue removido en NumPy 2.0
if not hasattr(np, 'round_'):
    np.round_ = np.round

# Agregar el directorio core al path para importar CosinorPy
sys.path.insert(0, str(Path(__file__).parent / 'core' / 'CosinorPy'))

from CosinorPy import cosinor, cosinor1


def select_file():
    """Abre ventana para seleccionar archivo CSV."""
    root = Tk()
    root.withdraw()  # Ocultar ventana principal

    data_dir = Path(__file__).parent / 'examples' / 'data_real'

    file_path = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        initialdir=data_dir,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    root.destroy()

    return file_path


def load_and_prepare_data(file_path):
    """
    Carga el archivo CSV y prepara los datos.
    """
    print(f"\n{'='*60}")
    print(f"Cargando archivo: {Path(file_path).name}")
    print(f"{'='*60}\n")

    # Intentar cargar con diferentes separadores (algunos CSV usan ; otros ,)
    try:
        df_raw = pd.read_csv(file_path, sep=';')
        if len(df_raw.columns) == 1:
            # Si solo hay 1 columna, probablemente el separador sea coma
            df_raw = pd.read_csv(file_path, sep=',')
    except:
        df_raw = pd.read_csv(file_path, sep=',')

    print(f"\nDimensiones: {df_raw.shape}")
    print(f"\nPrimeras filas del archivo:")
    print(df_raw.head(10))
    print(f"\nColumnas: {list(df_raw.columns)}")
    print(f"\nTipos de datos:")
    print(df_raw.dtypes)
    print(f"\nCondiciones únicas: {df_raw['Condition'].unique()}")
    print(f"\nTiempos únicos (ZT): {sorted(df_raw['ZT'].unique())}")

    return df_raw


def prepare_cosinorpy_format(df_raw):
    """
    Convierte los datos al formato que espera CosinorPy.

    CosinorPy espera un DataFrame con columnas:
    - 'test': identificador del grupo/condición
    - 'x': tiempo (variable independiente)
    - 'y': medición (variable dependiente)

    Tus datos tienen formato:
    - ZT: tiempo (Zeitgeber Time)
    - Condition: condición experimental
    - PDF/PER/Complexity: variable medida
    """

    print("\n" + "="*60)
    print("Preparando datos para CosinorPy")
    print("="*60 + "\n")

    # Identificar el nombre de la columna de la variable dependiente
    # (la tercera columna que no es ZT ni Condition)
    variable_col = [col for col in df_raw.columns if col not in ['ZT', 'Condition']][0]

    print(f"Variable detectada: {variable_col}")

    # Crear DataFrame en formato CosinorPy
    df_cosinor = pd.DataFrame({
        'test': df_raw['Condition'],
        'x': df_raw['ZT'],
        'y': df_raw[variable_col]
    })

    # Eliminar filas con valores NaN
    df_cosinor = df_cosinor.dropna()

    print(f"\nDataFrame preparado:")
    print(f"  Dimensiones: {df_cosinor.shape}")
    print(f"  Tests únicos: {df_cosinor['test'].unique()}")
    print(f"  Rango x: {df_cosinor['x'].min()} - {df_cosinor['x'].max()}")
    print(f"  Rango y: {df_cosinor['y'].min():.3f} - {df_cosinor['y'].max():.3f}")
    print(f"\nPrimeras filas:")
    print(df_cosinor.head(10))

    return df_cosinor


def run_cosinor_analysis(df):
    """
    Ejecuta análisis cosinor completo con parámetros exploratorios.

    Modificá los parámetros según lo que quieras probar:
    - n_components: número de componentes armónicos
    - period: período o lista de períodos a probar
    - lin_comp: incluir componente lineal
    """

    print("\n" + "="*60)
    print("PASO 1: Análisis Cosinor por Condición")
    print("="*60 + "\n")

    # PARÁMETROS PARA MODIFICAR
    # -------------------------
    n_components = [2]  # Probá con [1], [2], [1,2,3], etc.
    period = [24]       # Período en horas (modificá según tus datos)
    lin_comp = False    # True para incluir componente lineal

    print(f"Parámetros:")
    print(f"  n_components = {n_components}")
    print(f"  period = {period}")
    print(f"  lin_comp = {lin_comp}")
    print()

    # Ejecutar análisis por condición
    try:
        results = cosinor.fit_group(
            df,
            n_components=n_components,
            period=period,
        )

        print("\n" + "="*60)
        print("RESULTADOS POR CONDICIÓN")
        print("="*60 + "\n")
        print(results)

        # Guardar resultados
        output_file = Path(__file__).parent / 'cosinor_results_exploration.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResultados guardados en: {output_file}")

    except Exception as e:
        print(f"\n❌ Error ejecutando cosinor.fit_group(): {e}")
        import traceback
        traceback.print_exc()
        return None

    return results


def run_comparison_analysis(df):
    """
    Ejecuta comparaciones entre pares de condiciones.

    Usa dos métodos:
    1. compare_pairs(): Comparación directa (más simple)
    2. compare_pairs_limo(): Método LimoRhyde (más sofisticado)
    """

    print("\n" + "="*80)
    print("PASO 2: Comparación entre Condiciones")
    print("="*80 + "\n")

    # Parámetros
    n_components = 2  # Debe ser escalar, no lista
    period = 24       # Debe ser escalar, no lista

    # Generar pares automáticamente
    conditions = df['test'].unique()
    print(f"Condiciones detectadas: {list(conditions)}")

    # Generar todos los pares posibles
    from itertools import combinations
    pairs = list(combinations(conditions, 2))
    print(f"Pares a comparar: {pairs}\n")

    # ------------------------------------------------------------------
    # MÉTODO 1: compare_pairs() - Comparación Directa
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MÉTODO 1: cosinor.compare_pairs() - Comparación Directa")
    print("-"*80)

    try:
        comparison_direct = cosinor.compare_pairs(
            df,
            pairs=pairs,
            n_components=n_components,
            period=period,
            analysis='bootstrap',  # Opciones: 'CI' o 'bootstrap'
            parameters_to_analyse=['amplitude', 'acrophase', 'mesor'],
            parameters_angular=['acrophase']
        )

        print("\nResultados (Comparación Directa):")
        print(comparison_direct.to_string())

        # Guardar
        output_file = Path(__file__).parent / 'comparison_direct_results.csv'
        comparison_direct.to_csv(output_file, index=False)
        print(f"\nGuardado en: {output_file}")

    except Exception as e:
        print(f"\n❌ Error en compare_pairs(): {e}")
        import traceback
        traceback.print_exc()
        comparison_direct = None

    # ------------------------------------------------------------------
    # MÉTODO 2: compare_pairs_limo() - Método LimoRhyde
    # ------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MÉTODO 2: cosinor.compare_pairs_limo() - Método LimoRhyde")
    print("-"*80)

    try:
        comparison_limo = cosinor.compare_pairs_limo(
            df,
            pairs=pairs,
            n_components=n_components,
            period=period,
            analysis='bootstrap1',  # Opciones: '', 'CI1', 'bootstrap1', 'CI2', 'bootstrap2'
            parameters_to_analyse=['amplitude', 'acrophase', 'mesor'],
            parameters_angular=['acrophase'],
            folder=None  # Carpeta para guardar plots (None = no guardar)
        )

        print("\nResultados (LimoRhyde):")
        print(comparison_limo.to_string())

        # Guardar
        output_file = Path(__file__).parent / 'comparison_limo_results.csv'
        comparison_limo.to_csv(output_file, index=False)
        print(f"\nGuardado en: {output_file}")

    except Exception as e:
        print(f"\n❌ Error en compare_pairs_limo(): {e}")
        import traceback
        traceback.print_exc()
        comparison_limo = None

    return comparison_direct, comparison_limo


def main():
    """Función principal."""

    # 1. Seleccionar archivo
    file_path = select_file()

    if not file_path:
        print("No se seleccionó ningún archivo. Saliendo...")
        return

    # 2. Cargar datos
    df_raw = load_and_prepare_data(file_path)

    # 3. Preparar formato CosinorPy
    df_cosinor = prepare_cosinorpy_format(df_raw)

    # 4. Ejecutar análisis por condición
    results = run_cosinor_analysis(df_cosinor)

    if results is None:
        print("\n❌ No se pudo completar el análisis")
        return

    # 5. Ejecutar comparaciones entre condiciones
    comparisons = run_comparison_analysis(df_cosinor)

    # Resumen final
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print("\nArchivos generados:")
    print("  1. cosinor_results_exploration.csv - Resultados por condición")
    print("  2. comparison_direct_results.csv - Comparación directa entre pares")
    print("  3. comparison_limo_results.csv - Comparación LimoRhyde entre pares")
    print("\nPodés modificar los parámetros en:")
    print("  - run_cosinor_analysis(): n_components, period, lin_comp")
    print("  - run_comparison_analysis(): n_components, period, analysis method")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
