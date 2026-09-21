# CosinorPy GUI Implementation

## CosinorPy Library Structure

The library has 4 main modules:
- `file_parser`: Data I/O and synthetic data generation
- `cosinor`: Multi-component cosinor analysis (independent and population-based)
- `cosinor1`: Single-component cosinor with detailed statistics
- `cosinor_nonlin`: Non-linear cosinor analysis (amplification, linear component)

## Data Format Requirements

### Input CSV Format (user provides):
```
time,value,variable,condition,subject
0,1.2,GeneA,control,
0,1.3,GeneA,treatment,
2,0.8,GeneA,control,
...
```

### CosinorPy Internal Format:
```
x,y,test
0,1.2,GeneA_control
0,1.3,GeneA_treatment
2,0.8,GeneA_control
...
```
For dependent data, includes replicate identifier.

## Required Analysis Modules

### 1. Periodogram Analysis
- **Function**: `cosinor.periodogram_df(df)`
- **Purpose**: Visualize period detection
- **User inputs**: None (automatic)
- **Outputs**: Periodogram plots per variable
- Dependent data: Periodogram plots per subject!

### 2. Cosinor Analysis (Independent Data)
- **Functions**:
  - Single component (n=1): `cosinor1.fit_group(df, period=[24])`
  - Multi-component: `cosinor.fit_group(df, n_components=[1,2,3], period=24)`
  - Count Data Analysis (Analyze count data (RNA-seq, etc.) using Poisson or negative binomial models): `cosinor.fit_group(df, n_components, period, model_type='poisson')` or `model_type='negative_binomial'`
  - Best fits: `cosinor.get_best_fits(df_results, n_components, criterium='RSS')`
  - Best models: `cosinor.get_best_models(df, df_results, n_components)`
  - Extended analysis: `cosinor.analyse_best_models(df, df_best_models, analysis='CI')`
  - Plotting: `cosinor.plot_df_models(df, df_best_models, folder=None)`

- **User inputs**
  - Checkbox "Count Data" and model_type selection: model_type='poisson')` or `model_type='negative_binomial'
  - Period: single value or range (e.g., 24 or [20,21,22,23,24,25,26])
  - Components: list (e.g., [1], [1,2,3])
  - Criterium for best fit: 'RSS', 'AIC', 'BIC', 'log-likelihood'
  - Criterium for best model:the best fitting periods and the best fitting models - in dependence on the number of components; by default the criterium is p-value
  - Analysis model method: 'CI' or 'bootstrap'
  - Save plots checkbox + folder path

- **Outputs**: DataFrame with fitted parameters FOR ALL MODELS (ALL PERIODS AND COMPONENTS)
	-cosinor.fit_group(): period, n_components, p, q, p_reject, q_reject, RSS, R2, R2_adj, log-likelihood, amplitude, acrophase, mesor, peaks, heights, troughs, heights2, ME, resid_SE
	-cosinor.analyse_best_models(analysis="CI"): test ,period ,n_components, p, q, p_reject, q_reject, amplitude, acrophase, CI(amplitude), p(amplitude), q(amplitude), CI(acrophase), p(acrophase), q(acrophase), mesor, CI(mesor), p(mesor), q(mesor) 
	-cosinor.analyse_best_models(analysis="bootstrap"): test ,period ,n_components, p, q, p_reject, q_reject, amplitude, acrophase, CI(amplitude), p(amplitude), q(amplitude), CI(acrophase), p(acrophase), q(acrophase), mesor, CI(mesor), p(mesor), q(mesor)
	-cosinor1.fit_group(): test, period, p, q, amplitude, p(amplitude), q(amplitude), CI(amplitude), acrophase, p(acrophase), q(acrophase), CI(acrophase), acrophase[h]

Flujo de funciones según parámetros
	n_components = [1], un periodo:
		cosinor1.fit_group(period=[P])
		NO se llama get_best_fits, get_best_models, ni analyse_best_models
		Resultado: 1 fila, sin "Best model"
	n_components = [1], varios periodos:
		cosinor1.fit_group(period=[P1, P2, ...]) (una sola llamada)
		Comparación manual para mejor periodo (min p-value)
		NO se llama get_best_models ni analyse_best_models
		Resultado: N filas, mejor marcado como "Yes (min p-value)"
	n_components = [N] (N>1), un periodo:
		cosinor.fit_group(period=[P], n_components=[N])
		NO se llama get_best_fits (solo 1 periodo, innecesario)
		NO se llama get_best_models (solo 1 componente, innecesario)
		SÍ se llama cosinor.analyse_best_models() → CIs y p-values por parámetro
		Resultado: 1 fila con CIs extendidos
	n_components = [N] (N>1), varios periodos:
		Workflow de 2 pasos:
		Paso 1: cosinor.fit_group(period=[P1, P2, ...], n_components=[N])
		cosinor.get_best_fits(criterium, reverse) → selecciona mejor periodo
		Paso 2: cosinor.fit_group(period=[mejor_P], n_components=[N])
		NO se llama get_best_models (solo 1 componente)
		SÍ se llama cosinor.analyse_best_models() → CIs y p-values por parámetro
		Resultado: 1 fila (mejor periodo) con CIs extendidos
	n_components = [1, 2, ...], un periodo:
		cosinor.fit_group(period=[P], n_components=[1, 2, ...])
		NO se llama get_best_fits (solo 1 periodo, innecesario)
		cosinor.get_best_models() → F-test selecciona mejor
		cosinor.analyse_best_models() → CIs solo para mejor modelo
		Resultado: M filas, mejor marcado como "Best model (min p-value)"
	n_components = [1, 2, ...], varios periodos:
		Workflow de 2 pasos:
		Paso 1: cosinor.fit_group(period=[P1, P2, ...], n_components=[1])
		cosinor.get_best_fits(criterium, reverse) → mejor periodo
		Paso 2: cosinor.fit_group(period=[mejor_P], n_components=[1, 2, ...])
		cosinor.get_best_fits() → M filas
		cosinor.get_best_models() → F-test
		cosinor.analyse_best_models() → CIs para mejor modelo
		Resultado: M filas (mejor periodo), mejor modelo marcado


### 3. Cosinor Analysis (Dependent/Population Data)
- **Functions**:
  - Single component: `cosinor1.population_fit_group(df, period=24)`
  - Multi-component: `cosinor.population_fit_group(df, n_components=[1,2,3], period=24)`
  - Count Data Analysis (Analyze count data (RNA-seq, etc.) using Poisson or negative binomial models): `cosinor.fit_group(df, n_components, period, model_type='poisson')` or `model_type='negative_binomial'`
  - Best models: `cosinor.get_best_models_population(df, df_results, n_components, lin_comp = False, criterium = 'RSS', reverse = True)`
  - Extended analysis: `cosinor.analyse_best_models_population(df, df_best_models, params_CI_analysis='sampling')`
  - Plotting: `cosinor.plot_df_models_population(df, df_best_models, folder=None)`

- **User inputs**: Same as independent + params_CI_analysis option
	cosinor.get_best_models_population() - criterium:
		'RSS' (default) - Residual Sum of Squares - reverse=True (menor es mejor)
		'AIC' - Akaike Information Criterion
		'BIC' - Bayesian Information Criterion
		'log-likelihood' o 'Log-Likelihood' - Maximum Likelihood
	cosinor.analyse_best_models_population() - params_CI_analysis:
		'sampling' (default) - Parameter sampling (LHS o similar)
		'bootstrap' - Bootstrap resampling

- **Outputs**: DataFrame with fitted parameters
	-cosinor.population_fit_group(): test, period, n_components, p, q, p_reject, q_reject, RSS, amplitude, acrophase, mesor, ME, resid_SE, 'CI(amplitude)', 'CI(acrophase)', 'CI(mesor)'
	-cosinor1.population_fit_group():test, p, q, amplitude, p(amplitude), q(amplitude), CI(amplitude), mesor, p(mesor), q(mesor), CI(mesor), acrophase, p(acrophase), q(acrophase), CI(acrophase), acrophase[h]
	-cosinor.analyse_best_models_population(): test, period, n_components, p, q, p_reject, q_reject, amplitude, CI(amplitude), p(amplitude), q(amplitude), acrophase, CI(acrophase), p(acrophase), q(acrophase), mesor, CI(mesor), p(mesor), q(mesor)

Flujo de funciones según parámetros
	n_components = [1], un periodo
		Llama cosinor1.population_fit_group(df_test, period=, plot_on=False/True)
		Retorna 1 resultado (Dict) - amplitude, acrophase, acrophase_hours, mesor, p_value, q_value, amplitude_ci, acrophase_ci, mesor_ci, p_amplitude, p_acrophase, p_mesor, q_amplitude, q_acrophase, q_mesor
		No hay best_model indicator (solo 1 resultado)
	n_components = [1], varios periodos
		Itera sobre cada período: for p in [23, 24, 25]
		Para cada período, llama cosinor1.population_fit_group(df_test, period=p, plot_on=False/True)
		Concatena todos los DataFrames de resultados
		Identifica el mejor modelo (menor p-value)
		Retorna 3 resultados (List[Dict]), uno por período
		Cada resultado tiene best_model = 'Best model (min p-value)' o 'No'
	n_components = [N] (N>1), un periodo
		Llama cosinor.population_fit_group(df_test, n_components=[2], period=[24])
		Llama cosinor.get_best_models_population() → retorna 1 fila (solo hay 1 modelo lo que no tiene sentido pero asi usamos el mismo codigo)
		Llama cosinor.analyse_best_models_population() → análisis extendido con CIs
		Retorna 1 resultado (Dict)
		No hay best_model indicator (solo 1 combinación)
	n_components = [N] (N>1), varios periodos
 		Itera sobre cada período: for p in [23, 24, 25]
		Para cada período, llama cosinor.population_fit_group(df_test, n_components=[N], period=[p])
		Concatena todos los DataFrames de resultados → 3 filas
		Llama cosinor.get_best_models_population() → retorna 1 fila (mejor modelo global)
		Llama cosinor.analyse_best_models_population() → análisis extendido solo para el mejor
		Retorna 3 resultados (List[Dict]), uno por período
		Solo el mejor modelo global tiene datos extendidos (CIs, etc.)
		Cada resultado tiene best_model basado en comparación de (n_components, period)
	n_components = [1, 2, ...], un periodo
		Llama cosinor.population_fit_group(df_test, n_components=[1,2,3], period=[24])
		Retorna 3 filas (una por cada n_components)
		Llama cosinor.get_best_models_population() → retorna 1 fila
		Llama cosinor.analyse_best_models_population() → análisis extendido solo para el mejor
		Retorna 3 resultados (List[Dict]), uno por componente
		Solo el mejor modelo tiene datos extendidos
		Cada resultado tiene best_model basado en (n_components, period)
	n_components = [1, 2, ...], varios periodos
		Itera sobre cada período: for p in [23, 24]
		Para cada período, llama cosinor.population_fit_group(df_test, n_components=[1,2], period=[p])
		Período 23 → 2 filas (n_components=1 y 2)
		Período 24 → 2 filas (n_components=1 y 2)
		Concatena todos los DataFrames → 4 filas totales
		Llama cosinor.get_best_models_population() → retorna 1 fila (mejor modelo global entre las 4)
		Llama cosinor.analyse_best_models_population() → análisis extendido solo para el mejor
		Retorna 4 resultados (List[Dict])
		Solo el mejor modelo global tiene datos extendidos
		Cada resultado tiene best_model basado en comparación de (n_components, period)

### 4. Compare Conditions (Independent Data)
- **Prerequisite**: "condition" column must exist in data
- **Functions**:
  - Single component: `cosinor1.test_cosinor_pairs(df, pairs, period=24, folder=None)` para datos dependientes / test_cosinor_pairs_independent(), para datos independientes con 1 componente
  - Multi-component LimoRhyde: `cosinor.compare_pairs_limo(df, pairs, n_components, period, folder=None, analysis="")`, analysis: "", "CI1", "bootstrap1", "CI2", "bootstrap2"
  - Direct comparison: `cosinor.compare_pairs(df, pairs, n_components, period, analysis='CI')`, analysis: "CI" o "bootstrap"

- **User inputs**:
  - Pairs to compare: auto-generate from conditions
  - All parameters from analysis module
  
- **Outputs**: DataFrame with amplitude change (d_amplitude), acrophase shift (d_acrophase), p/q values
	-cosinor1.test_cosinor_pairs(): test, period, p, q, amplitude1, p(amplitude1), q(amplitude1), amplitude2, p(amplitude2), q(amplitude2),	..., p(acrophase1), q(acrophase1), acrophase2, p(acrophase2), q(acrophase2), d_acrophase, p(d_acrophase), q(d_acrophase), CI(d_acrophase), CI(d_amplitude)
	-cosinor1.test_cosinor_pairs_independent(): test, p1, q1, p2, q2, period1, period2, amplitude1, amplitude2, d_amplitude, p(d_amplitude), q(d_amplitude), CI(d_amplitude), acrophase1, acrophase2, d_acrophase, p(d_acrophase), q(d_acrophase), CI(d_acrophase)
	-cosinor.compare_pairs():test, period, n_components, p1, p2, q1, q2, d_amplitude, CI(d_amplitude), p(d_amplitude), q(d_amplitude), d_acrophase, CI(d_acrophase), p(d_acrophase), q(d_acrophase), d_mesor, CI(d_mesor), p(d_mesor), q(d_mesor)
	-cosinor.compare_pairs_limo():test, period, n_components, p, q, p params, q params, p(F test), q(F test), d_amplitude, d_acrophase

Flujo de funciones
Caso 1: Single component (n_components=[1])
El usuario puede elegir "Comparison Type":
├─> "Pooled Model" (Recommended for similar groups)
│   └─> cosinor1.test_cosinor_pairs(df, pairs, period, folder)
│       - Modelo conjunto con interacción
│       - Mismo periodo para ambos grupos
│       - Mayor poder estadístico
│       - Asume varianza común
└─> "Independent Models" (Recommended for different groups)
    └─> cosinor1.test_cosinor_pairs_independent(df, pairs, period, period2)
        - Dos modelos separados
        - Permite diferentes periodos
        - No asume varianza común
        - Más robusto
Caso 2: Multiple components (n_components=[1,2,3])
SI Comparison Method = "Independent":
└─> Iterar sobre cada combinación (period, n_components):
    └─> cosinor.compare_pairs(
          df, pairs,
          n_components=n_comp,  # scalar, no lista
          period=per,           # scalar, no lista
          analysis="CI" or "bootstrap",
          parameters_to_analyse=user_selection,  # e.g., ['amplitude', 'acrophase', 'mesor']
          parameters_angular=['acrophase'],
          lin_comp=True/False,
          bootstrap_size=bootstrap_size
        )
    └─> Concatenar todos los resultados en un solo DataFrame
SI Comparison Method = "LimoRhyde":
└─> Iterar sobre cada combinación (period, n_components):
    └─> cosinor.compare_pairs_limo(
          df, pairs,
          n_components=n_comp,  # scalar, no lista
          period=per,           # scalar, no lista
          analysis="" or "CI1" or "bootstrap1" or "CI2" or "bootstrap2",
          parameters_to_analyse=user_selection,
          parameters_angular=['acrophase'],
          folder=save_folder if save_plots else "",
          bootstrap_size=bootstrap_size
        )
    └─> Concatenar todos los resultados en un solo DataFrame


### 5. Compare Conditions (Dependent/Population Data)
- **Functions**:
  - Single component: `cosinor1.population_test_cosinor_pairs(df, pairs, period=24)`
  - Direct: `cosinor.compare_pairs_population(df, pairs, n_components, analysis='CI', parameters_to_analyse=['acrophase', 'amplitude'])`

- **Outputs**: DataFrame
	-cosinor1.population_test_cosinor_pairs(): test, d_amplitude, p(d_amplitude), q(d_amplitude), d_acrophase, p(d_acrophase), q(d_acrophase)
	-cosinor.compare_pairs_population(): test, period, n_components, p1, p2, q1, q2, d_amplitude, CI(d_amplitude), p(d_amplitude), q(d_amplitude), d_acrophase, CI(d_acrophase), p(d_acrophase), q(d_acrophase), d_mesor, CI(d_mesor), p(d_mesor), q(d_mesor)

Flujo de funciones
Caso 1: Single component (n_components=[1])
└─> Iterar sobre cada periodo
    └─> cosinor1.population_test_cosinor_pairs(df, pairs, period, save_folder, plot_on)
	- Modelo population-mean
	- Compara amplitud y acrofase SOLAMENTE (no MESOR)
	- Requiere columna 'subject' en el DataFrame
	- Formato de test: "condition_repN" (e.g., "control_rep1", "control_rep2")
	- ⚠️ CRITICAL: Siempre llama plt.show() si save_folder='' → Solución: Usar carpeta temporal para evitar GUI freeze
	- Pares: [('control', 'treatment')] (solo nombres de condiciones, SIN variable)
Caso 2: Multiple components (n_components=[1,2,3])
└─> Iterar sobre cada combinación (period, n_components):
    └─> cosinor.compare_pairs_population(
          df, pairs,
          n_components=[n_comp],      # ⚠️ LISTA (diferente a independent!)
          period=[per],                # ⚠️ LISTA (diferente a independent!)
          analysis="CI" or "permutation",
          parameters_to_analyse=user_selection,  # e.g., ['amplitude', 'acrophase', 'mesor']
          parameters_angular=['acrophase'],
          lin_comp=True/False,
          folder=save_folder or ""     # Empty string = no plots (no freeze risk)
        )
    └─> Concatenar todos los resultados en un solo DataFrame

### 6. Non-Linear Analysis (Independent Data)
│        Modelo: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t                    │
│        Parámetros adicionales: amplification (C), lin_comp (D) 
- **Purpose**: Detect amplification (damped/forced oscillations) and linear trends
- **Functions**:
  - Single component: `cosinor_nonlin.fit_generalized_cosinor_group(df, period=24, plot=True)`
  - Multi-component: `cosinor_nonlin.fit_generalized_cosinor_n_comp_group(df, period=24, n_components=3, plot=True)`
  - Best model: `cosinor_nonlin.fit_generalized_cosinor_n_comp_group_best(df, period=24, n_components=[1,2,3], plot=True)`
  - Bootstrap single: `cosinor_nonlin.bootstrap_generalized_cosinor_n_comp_group(df, period=24, n_components=3, bootstrap_size=100)`
  - Bootstrap best: `cosinor_nonlin.bootstrap_generalized_cosinor_n_comp_group_best(df, df_best_models, bootstrap_size=100)`

- **Additional outputs**: amplification, lin_comp parameters with q-values

Flujo de funciones según parámetros
	n_components = [1], un periodo:
		Función: cosinor_nonlin.fit_generalized_cosinor_group(df, period=P, plot=True)
         	├─ Internamente llama: fit_generalized_cosinor() para cada test
         	└─ Retorna: df_results con p, q, amplitude, acrophase, amplification, lin_comp + sus p-values y CIs
		Bootstrap: NO necesario (estadísticas calculadas analíticamente vía curve_fit)
		Resultado: N filas (una por test), con estadísticas completas incluyendo q-values
	n_components = [1], varios periodos:
		Función: Llamar cosinor_nonlin.fit_generalized_cosinor_group() MÚLTIPLES VECES (una por periodo)
	         for period in [P1, P2, P3, ...]:
	             df_results_p = cosinor_nonlin.fit_generalized_cosinor_group(df, period=period, plot=False)
         		Luego: Concatenar resultados y comparar manualmente por mejor p-value
		NO se llama get_best_models (no existe para nonlin con 1 comp)
		Resultado: N×M filas (N tests × M periodos), mejor marcado manualmente como "Yes (min p-value)"
	n_components = [N] (N>1), un periodo:
		Función: cosinor_nonlin.fit_generalized_cosinor_n_comp_group(df, period=P, n_components=N, plot=True)
		         ├─ Internamente llama: fit_generalized_cosinor_n_comp()
		         └─ Retorna: df_results con peaks, troughs, amplification, lin_comp, etc.
		Bootstrap (OPCIONAL pero recomendado para amplitude/acrophase):
		         cosinor_nonlin.bootstrap_generalized_cosinor_n_comp_group(
		             df, period=P, n_components=N, bootstrap_size=100
		         )
		         └─ Agrega: p(amplitude), q(amplitude), CI(amplitude), p(acrophase), q(acrophase), CI(acrophase)
		Resultado: 1 filas
	n_components = [N] (N>1), varios periodos:
		Función: Llamar cosinor_nonlin.fit_generalized_cosinor_n_comp_group() MÚLTIPLES VECES
		         for period in [P1, P2, P3, ...]:
		             df_results_p = cosinor_nonlin.fit_generalized_cosinor_n_comp_group(
		                 df, period=period, n_components=N, plot=False
		             )         
		         Luego: Concatenar y comparar manualmente por mejor p-value o RSS
		Bootstrap (OPCIONAL): Ídem caso 3, por cada periodo
		Resultado: 1×M filas
	n_components = [1, 2, ...], un periodo:
		Función: cosinor_nonlin.fit_generalized_cosinor_n_comp_group_best(
		             df, period=P, n_components=[1,2,3], plot=True
		         )
		         ├─ Internamente llama: get_best_model() que prueba todos los n_components
		         ├─ Selecciona el mejor modelo por criterio (p-value por defecto)
		         └─ Retorna: df_best_models con n_components óptimo por test
		Bootstrap (RECOMENDADO para amplitude/acrophase con n_comp > 1):
		         df_bootstrap = cosinor_nonlin.bootstrap_generalized_cosinor_n_comp_group_best(
		             df, df_best_models, bootstrap_size=100
		         )
		         └─ Usa los n_components y period de df_best_models
		Resultado: N filas (una por test), cada una con su n_components óptimo
	n_components = [1, 2, ...], varios periodos:
		Función: Llamar cosinor_nonlin.fit_generalized_cosinor_n_comp_group_best() MÚLTIPLES VECES
		         for period in [P1, P2, P3, ...]:
		             df_best_p = cosinor_nonlin.fit_generalized_cosinor_n_comp_group_best(
		                 df, period=period, n_components=[1,2,3], plot=False
		             )         
		         Luego: Concatenar y seleccionar mejor combinación (period, n_components) por test
		Bootstrap: Ídem caso 5, usando el df_best_models filtrado
		Resultado: Selección del mejor (period, n_components) por test

### 7. Non-Linear Analysis (Dependent/Population Data)
	NON-LINEAR ANALYSIS (DEPENDENT/POPULATION DATA)
        Modelo: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t
        Datos: Mismos sujetos medidos repetidamente (test5_rep1, test5_rep2)
- **Functions**:
  - Single component: `cosinor_nonlin.population_fit_generalized_cosinor_group(df, period=24, plot=True)`
  - Multi-component: `cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group(df, period=24, n_components=3, plot=True)`
  - Best model: `cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group_best(df, period=24, n_components=[1,2,3], plot=True)`

Flujo de funciones según parámetros
	n_components = [1], un periodo:
		Función: cosinor_nonlin.population_fit_generalized_cosinor_group(df, period=P, plot=True)
		         ├─ Ajusta modelo generalizado de 1 componente para cada población
		         ├─ Calcula estadísticas analíticamente (sin bootstrap)
		         └─ Retorna: df_results con amplitude, acrophase, amplification, lin_comp + p/q-values + CIs
		Bootstrap: NO necesario (stats calculados analíticamente)
		Resultado: 1 filas (una por población/test base)
	n_components = [1], varios periodos:
		Función: Llamar population_fit_generalized_cosinor_group() MÚLTIPLES VECES
		         for period in [P1, P2, P3, ...]:
		             df_results_p = cosinor_nonlin.population_fit_generalized_cosinor_group(df, period=period, plot=False)         
		         Luego: Concatenar y comparar manualmente por mejor p-value
		Resultado: N×M filas (N conditions × M periodos)
	n_components = [N] (N>1), un periodo:
		Función: cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group(df, period=P, n_components=N, plot=True)
		         ├─ Ajusta modelo con N componentes para cada población
		         └─ Retorna: df_results con peaks, troughs, amplification, lin_comp, etc.
		Bootstrap: NO disponible para population (stats se calculan desde la varianza entre réplicas)
		Resultado: 1 fila
	n_components = [N] (N>1), varios periodos:
		Función: Llamar population_fit_generalized_cosinor_n_comp_group() MÚLTIPLES VECES
		         for period in [P1, P2, ...]:
		             df_results_p = cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group(
		                 df, period=period, n_components=N, plot=False
		             )		
		Resultado: 1×M filas
	n_components = [1, 2, ...], un periodo:
		Función: cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group_best(
		             df, period=P, n_components=[1,2,3], plot=True
		         )
		         ├─ Prueba todos los n_components
		         ├─ Selecciona el mejor modelo por criterio estadístico
		         └─ Retorna: df_best_models con n_components óptimo por test
		Bootstrap: NO disponible para population
		Resultado: 1 fila
	n_components = [1, 2, ...], varios periodos:
		Función: Llamar population_fit_generalized_cosinor_n_comp_group_best() MÚLTIPLES VECES
		         for period in [P1, P2, ...]:
		             df_best_p = cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group_best(
		                 df, period=period, n_components=[1,2,3], plot=False
		             )
		Resultado: Selección del mejor (period, n_components) por condition

### 8. Non-Linear Compare Conditions (Independent Data)
- **Functions**:
  - Same period (dependent model): `cosinor_nonlin.fit_generalized_cosinor_compare_pairs_dependent(df, pairs, period=24, plot=True)`
  - Different periods (independent model): `cosinor_nonlin.fit_generalized_cosinor_compare_pairs_independent(df, pairs, period1=24, period2=24, plot=True)`
  - Bootstrap comparison: `cosinor_nonlin.compare_pairs_n_comp_bootstrap_group(df, pairs, df_best_models=df_best_models, df_bootstrap_single=df_bootstrap, plot=True)`
  - Or with parameters: `cosinor_nonlin.compare_pairs_n_comp_bootstrap_group(df, pairs, n_components=3, period=24, bootstrap_size=100, plot=True)`

Flujo de funciones según parámetros
	Opción A: Modelo Dependiente (1 componente, mismo periodo)
		Función: cosinor_nonlin.fit_generalized_cosinor_compare_pairs_dependent(
		             df, pairs, period=24, plot=True
		         )
		         └─ Solo 1 componente
		         └─ Retorna: d_amplitude, d_acrophase, d_amplification, d_lin_comp + p-values
		Uso: Cuando ambas condiciones comparten el mismo periodo
	Opción B: Modelo Independiente (1 componente, puede ser distinto periodo)
		Función: cosinor_nonlin.fit_generalized_cosinor_compare_pairs_independent(
		             df, pairs, period1=24, period2=24, plot=True
		         )
		         └─ Solo 1 componente
		         └─ Retorna: diferencias con CIs calculados independientemente
		Uso: Cuando cada condición puede tener su propio periodo
	Opción C: Multi-componente con Bootstrap. Itera sobre cada componente
		Función: cosinor_nonlin.compare_pairs_n_comp_bootstrap_group(
		             df, pairs,
		             n_components=3,                      # fijar n_components
		             period=24,
		             bootstrap_size=100,
		             plot=True
 		        )

### 9. Non-Linear Compare Conditions (Dependent/Population Data)
- **Functions**:
  - Single component: `cosinor_nonlin.population_fit_generalized_cosinor_compare_pairs(df, pairs, period1=24, period2=24, plot=True)`
  - Multi-component: `cosinor_nonlin.population_compare_pairs_n_comp_group(df, pairs, df_best_models=df_best_models, plot=True)`
  - Or with params: `cosinor_nonlin.population_compare_pairs_n_comp_group(df, pairs, n_components=4, period=24, plot=True)`

Flujo de funciones según parámetros
	Opción A: Single Component (1 comp, puede tener diferentes periodos)
		Función: cosinor_nonlin.population_fit_generalized_cosinor_compare_pairs(
		             df, pairs, period1=24, period2=24, plot=True
		         )
		         ├─ Compara pares usando modelo de 1 componente
		         ├─ Permite periodos diferentes por condición
		         └─ Retorna: d_amplitude, d_acrophase, d_amplification, d_lin_comp + p-values + CIs
		Uso: Comparación simple con 1 componente
	Opción B: Multi-component con parámetros fijos
		Función: cosinor_nonlin.population_compare_pairs_n_comp_group(
		             df, pairs,
		             n_components=N,   # Mismo N para todos, y se itera sobre cada uno
		             period=24,
		             plot=True
		         )
		         ├─ Usa el mismo n_components para todos los tests
		         └─ Retorna: comparaciones con N componentes
		Uso: Cuando querés forzar el mismo número de componentes para todas las comparaciones


### Key Implementation Details:

1. **Data Type Detection**:
   - If "subject" column exists → Dependent data (population methods)
   - If no "subject" column → Independent data

2. **Condition Detection**:
   - If "condition" column exists → Enable compare methods
   - Auto-generate pairs from unique conditions

3. **Dynamic UI**:
   - Show/hide parameters based on selected analysis method
   - Disable compare methods if no condition column

4. **NumPy 2.0 Compatibility**:
   Apply monkey patch for deprecated functions:
   ```python
   import numpy as np
   if not hasattr(np, 'float'):
       np.float = np.float64
   if not hasattr(np, 'int'):
       np.int = np.int64
   if not hasattr(np, 'bool'):
       np.bool = np.bool_
   ```