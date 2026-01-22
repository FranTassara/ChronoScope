# CircaScope - Referencia de Funciones para GUI

## Resumen de Módulos Disponibles

| Módulo | Clase Principal | Descripción |
|--------|-----------------|-------------|
| `cosinor_analysis.py` | `CosinorAnalyzer` | Wrapper de CosinorPy con análisis cosinor completo |
| `circacompare_analysis.py` | `CircaCompareAnalyzer` | Comparación de grupos con regresión robusta |
| `rhythm_analysis.py` | `RhythmAnalyzer` | Métodos adicionales (JTK, Fourier, Lomb-Scargle, etc.) |

---

## 1. MÓDULO: cosinor_analysis.py (CosinorPy Wrapper)

**Dependencia:** `pip install cosinorpy`

### Inicialización
```python
from cosinor_analysis import CosinorAnalyzer, DataType, AnalysisMode, ModelType

analyzer = CosinorAnalyzer(
    period=24.0,           # Período esperado (horas)
    n_components=1,        # Número de armónicos
    data_type=DataType.CONTINUOUS,      # CONTINUOUS o COUNT
    analysis_mode=AnalysisMode.INDEPENDENT  # INDEPENDENT o DEPENDENT
)
```

### Parámetros de Inicialización (GUI)
| Parámetro | Tipo | Default | Descripción GUI |
|-----------|------|---------|-----------------|
| `period` | float | 24.0 | Spinner/Input: "Período (horas)" [1-48] |
| `n_components` | int | 1 | Spinner: "Número de componentes" [1-4] |
| `data_type` | enum | CONTINUOUS | Dropdown: "Tipo de datos" [Continuo, Conteo] |
| `analysis_mode` | enum | auto-detect | Dropdown: "Modo" [Independiente, Dependiente, Auto] |

---

### Funciones de Análisis Disponibles

#### 1.1 `single_cosinor()` - Cosinor Simple
**Propósito:** Análisis cosinor de un componente para una variable/condición.

```python
result = analyzer.single_cosinor(
    variable="gene1",      # Selección de variable
    condition="control",   # Selección de condición
    period=24.0           # Opcional: override del período
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | Lista de variables detectadas |
| `condition` | str | Dropdown | Lista de condiciones detectadas |
| `period` | float | Spinner | Opcional, default del analyzer |

**Output:** `CosinorParameters` con MESOR, amplitud, acrofase, p-value, CI

---

#### 1.2 `single_cosinor_all()` - Cosinor Batch
**Propósito:** Análisis cosinor para todas las variables.

```python
df_results = analyzer.single_cosinor_all(
    condition="control",   # Opcional: filtrar por condición
    period=24.0           # Opcional
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `condition` | str/None | Dropdown + "Todas" | Opcional |
| `period` | float | Spinner | Opcional |

**Output:** DataFrame con resultados para todas las combinaciones

---

#### 1.3 `multi_cosinor()` - Cosinor Multi-componente
**Propósito:** Detecta armónicos adicionales (ritmos con forma compleja).

```python
result = analyzer.multi_cosinor(
    variable="gene1",
    condition="control",
    n_components=2,       # Número de armónicos
    period=24.0
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | - |
| `condition` | str | Dropdown | - |
| `n_components` | int | Spinner [1-4] | 1=básico, 2+=detecta waveforms complejas |
| `period` | float | Spinner | - |

---

#### 1.4 `find_best_model()` - Selección Automática de Modelo
**Propósito:** Encuentra el número óptimo de componentes usando criterio estadístico.

```python
result = analyzer.find_best_model(
    variable="gene1",
    condition="control",
    max_components=4,     # Máximo a probar
    period=24.0
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | - |
| `condition` | str | Dropdown | - |
| `max_components` | int | Spinner [2-6] | Default: 4 |
| `period` | float | Spinner | - |

---

#### 1.5 `population_cosinor()` - Cosinor Poblacional
**Propósito:** Para datos dependientes (mismos sujetos medidos en múltiples tiempos).

```python
result = analyzer.population_cosinor(
    variable="gene1",
    condition="control",
    period=24.0,
    n_components=1
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | - |
| `condition` | str | Dropdown | - |
| `period` | float | Spinner | - |
| `n_components` | int | Spinner [1-4] | - |

**Nota:** Requiere columna `subject` en el CSV.

---

#### 1.6 `compare_conditions()` - Ritmicidad Diferencial
**Propósito:** Compara parámetros rítmicos entre dos condiciones.

```python
result = analyzer.compare_conditions(
    variable="gene1",
    condition1="control",
    condition2="treatment",
    period=24.0,
    n_components=1
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | - |
| `condition1` | str | Dropdown | "Condición referencia" |
| `condition2` | str | Dropdown | "Condición comparación" |
| `period` | float | Spinner | - |
| `n_components` | int | Spinner [1-4] | - |

**Output:** `DifferentialResult` con diferencias en amplitud, acrofase, MESOR y p-values

---

#### 1.7 `compare_all_conditions()` - Comparación Múltiple
**Propósito:** Compara todas las parejas de condiciones.

```python
df_results = analyzer.compare_all_conditions(
    variable="gene1",
    period=24.0,
    n_components=1
)
```

---

#### 1.8 `compare_variables()` - Comparar Variables
**Propósito:** Compara ritmos de dos variables en la misma condición.

```python
result = analyzer.compare_variables(
    variable1="gene1",
    variable2="gene2",
    condition="control",
    period=24.0,
    n_components=1
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable1` | str | Dropdown | "Variable 1" |
| `variable2` | str | Dropdown | "Variable 2" |
| `condition` | str | Dropdown | - |

---

#### 1.9 `nonlinear_cosinor()` - Cosinor Generalizado
**Propósito:** Para ritmos no estacionarios (amplitud que cambia en el tiempo).

```python
result = analyzer.nonlinear_cosinor(
    variable="gene1",
    condition="control",
    period=24.0,
    n_components=1
)
```

**Incluye:** Coeficiente de damping y tendencia lineal.

---

#### 1.10 `fit_count_data()` - Análisis para Datos de Conteo
**Propósito:** Para RNA-seq counts usando regresión Poisson o NB.

```python
result = analyzer.fit_count_data(
    variable="gene1",
    condition="control",
    model_type=ModelType.POISSON,  # POISSON, GEN_POISSON, NEGATIVE_BINOMIAL
    period=24.0,
    n_components=1
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `model_type` | enum | Dropdown | [Poisson, Gen. Poisson, Neg. Binomial] |

---

## 2. MÓDULO: circacompare_analysis.py

**Dependencias:** Solo scipy, numpy, pandas (incluidas)

### Inicialización
```python
from circacompare_analysis import CircaCompareAnalyzer

analyzer = CircaCompareAnalyzer(
    period=24.0,
    loss='linear',        # Función de pérdida
    f_scale=1.0,          # Escala para loss robusta
    max_iterations=500    # Intentos de optimización
)
```

### Parámetros de Inicialización (GUI)
| Parámetro | Tipo | Default | Widget | Descripción |
|-----------|------|---------|--------|-------------|
| `period` | float | 24.0 | Spinner | Período en horas |
| `loss` | str | 'linear' | Dropdown | [linear, soft_l1, huber, cauchy, arctan] |
| `f_scale` | float | 1.0 | Spinner [0.1-10] | Solo para loss no-lineal |
| `max_iterations` | int | 500 | Spinner [100-2000] | Reintentos optimización |

**Nota sobre Loss Functions:**
- `linear`: Mínimos cuadrados estándar (más sensible a outliers)
- `soft_l1`: Suavemente robusta
- `huber`: Robusta, combina L1 y L2
- `cauchy`: Muy robusta a outliers
- `arctan`: Extremadamente robusta

---

### Funciones de Análisis

#### 2.1 `fit_single()` - Ajuste de Un Grupo
**Propósito:** Ajusta modelo cosinor a una condición con intervalos de confianza.

```python
result = analyzer.fit_single(
    variable="gene1",
    condition="control",
    period=24.0  # Opcional
)
```

| Parámetro GUI | Tipo | Widget |
|---------------|------|--------|
| `variable` | str | Dropdown |
| `condition` | str | Dropdown |
| `period` | float | Spinner (opcional) |

**Output:** `CircaSingleResult` con MESOR, amplitud, acrofase, CI 95%, SE

---

#### 2.2 `fit_single_all()` - Ajuste Batch
```python
df_results = analyzer.fit_single_all(
    condition=None,  # None = todas
    period=24.0
)
```

---

#### 2.3 `compare()` - Comparación de Dos Grupos ⭐
**Propósito:** Función principal para detectar ritmicidad diferencial.

```python
result = analyzer.compare(
    variable="gene1",
    condition1="control",    # Grupo referencia (g0)
    condition2="treatment",  # Grupo comparación (g1)
    period=24.0
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | - |
| `condition1` | str | Dropdown | "Referencia" |
| `condition2` | str | Dropdown | "Comparación" |
| `period` | float | Spinner | - |

**Output:** `CircaCompareResult` con:
- Parámetros grupo 0 (mesor_g0, amplitude_g0, acrophase_g0)
- Diferencias (d_mesor, d_amplitude, d_acrophase)
- Parámetros grupo 1 (derivados)
- CI 95% para todas las diferencias
- Métodos: `.is_mesor_different()`, `.is_amplitude_different()`, `.is_acrophase_different()`

---

#### 2.4 `compare_all_conditions()` - Comparaciones Múltiples
```python
df_results = analyzer.compare_all_conditions(
    variable="gene1",
    period=24.0
)
```

---

#### 2.5 `compare_all_variables()` - Comparar Todas las Variables
```python
df_results = analyzer.compare_all_variables(
    condition1="control",
    condition2="treatment",
    period=24.0
)
```

---

#### 2.6 `predict()` y `predict_compare()` - Para Plotting
```python
# Curva ajustada para un grupo
time_grid, y_fitted = analyzer.predict(result, n_points=100)

# Curvas para ambos grupos
time_grid, y_g0, y_g1 = analyzer.predict_compare(compare_result, n_points=100)
```

---

## 3. MÓDULO: rhythm_analysis.py

**Dependencias:** `pip install numpy scipy pandas statsmodels PyWavelets`

### Inicialización
```python
from rhythm_analysis import RhythmAnalyzer, DefaultPeriodRanges

analyzer = RhythmAnalyzer(
    period_range=[22, 23, 24, 25, 26],  # Períodos a probar
    default_period=24.0                  # Para métodos single-period
)
```

### Parámetros de Inicialización (GUI)
| Parámetro | Tipo | Default | Widget | Descripción |
|-----------|------|---------|--------|-------------|
| `period_range` | list | [20-28] | Multi-slider o input list | Rango de períodos |
| `default_period` | float | 24.0 | Spinner | Para Fourier F24 |

**Period Ranges Predefinidos:**
- `DefaultPeriodRanges.CIRCADIAN` = [20, 20.5, 21, ..., 28]
- `DefaultPeriodRanges.CIRCADIAN_INT` = [20, 21, 22, ..., 28]
- `DefaultPeriodRanges.ULTRADIAN` = [4, 5, 6, ..., 12]

---

### Funciones de Análisis

#### 3.1 `run_jtk()` - JTK Cycle ⭐
**Propósito:** Detección no-paramétrica de ritmicidad usando templates triangulares.

```python
result = analyzer.run_jtk(
    variable="gene1",
    condition="control",
    period_range=[22, 23, 24, 25, 26],  # Lista de períodos enteros
    lag_range=None,                      # Auto: 0 a period
    asymmetries=[0.5]                    # Simetría del waveform
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `variable` | str | Dropdown | - |
| `condition` | str | Dropdown | - |
| `period_range` | list[int] | Multi-select/Input | Solo enteros |
| `asymmetries` | list[float] | Checkboxes | [0.5]=simétrico, [0.25,0.5,0.75]=probar varios |

**Output:** `JTKResult` con period, amplitude, acrophase, tau, p_value, adj_p_value

---

#### 3.2 `run_ar_jtk()` - JTK con Manejo de Autocorrelación
**Propósito:** JTK que detecta y corrige autocorrelación en residuos.

```python
result, ar_applied = analyzer.run_ar_jtk(
    variable="gene1",
    condition="control",
    period_range=[22, 23, 24, 25, 26],
    ar_lag=1,              # Orden del modelo AR
    ljungbox_lag=10        # Lags para test de Ljung-Box
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `ar_lag` | int | Spinner [1-5] | Orden AR para prewhitening |
| `ljungbox_lag` | int | Spinner [5-20] | Lags para test autocorr |

**Output:** Tuple (`JTKResult`, `bool` indicando si AR fue aplicado)

---

#### 3.3 `run_cosine_kendall()` - Coseno-Kendall
**Propósito:** Similar a JTK pero usa templates coseno.

```python
result = analyzer.run_cosine_kendall(
    variable="gene1",
    condition="control",
    period_range=[22, 23, 24, 25, 26],
    interval=None  # Auto-detectado
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `period_range` | list[float] | Input | Acepta decimales |
| `interval` | float/None | Spinner | Intervalo muestreo (auto) |

---

#### 3.4 `run_cosinor()` - Cosinor con Selección de Período
**Propósito:** Cosinor OLS con selección automática de período por AIC.

```python
result = analyzer.run_cosinor(
    variable="gene1",
    condition="control",
    period_range=[22, 23, 24, 25, 26]
)
```

| Parámetro GUI | Tipo | Widget |
|---------------|------|--------|
| `period_range` | list[float] | Input/Slider |

**Output:** `CosinorResult` con mesor, amplitude, acrophase, CI, adj_p_value

---

#### 3.5 `run_harmonic_cosinor()` - Cosinor Armónico ⭐
**Propósito:** Detecta ritmos multi-modales (ej: bimodal con 2 picos/día).

```python
result = analyzer.run_harmonic_cosinor(
    variable="gene1",
    condition="control",
    period_range=[22, 23, 24, 25, 26],
    n_harmonics=2  # 1=unimodal, 2=bimodal, etc.
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `n_harmonics` | int | Spinner [1-4] | 1=normal, 2=bimodal, 3=trimodal |

**Output:** `HarmonicCosinorResult` con múltiples acrophases y amplitudes

---

#### 3.6 `run_fourier_f24()` - Análisis Fourier (Wijnen 2006) ⭐
**Propósito:** Calcula F24 score como medida de effect size rítmico.

⚠️ **REQUIERE EXACTAMENTE 2 RÉPLICAS**

```python
result = analyzer.run_fourier_f24(
    variable="gene1",
    condition="control",
    target_period=24.0,     # Período objetivo
    n_permutations=1000     # Para calcular baseline
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `target_period` | float | Spinner | Período a evaluar |
| `n_permutations` | int | Spinner [100-10000] | Más = más preciso |

**Output:** `FourierF24Result` con f24_score, power_spectrum
- `.is_rhythmic(threshold=2.0)` → True si F24 > threshold

**Nota:** F24 es effect size, NO p-value. Usar como filtro (F24 > 2 o F24 > 3).

---

#### 3.7 `run_lomb_scargle()` - Periodograma Lomb-Scargle
**Propósito:** Ideal para datos con muestreo irregular.

```python
result = analyzer.run_lomb_scargle(
    variable="gene1",
    condition="control",
    period_range=(20.0, 28.0),  # Tupla (min, max)
    n_periods=1000              # Resolución
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `period_range` | tuple | Dos spinners | (min_period, max_period) |
| `n_periods` | int | Spinner [100-5000] | Resolución espectral |

**Output:** `LombScargleResult` con dominant_period, power, FAP

---

#### 3.8 `run_cwt()` - Transformada Wavelet Continua
**Propósito:** Detecta cambios en período/amplitud a lo largo del tiempo.

⚠️ **REQUIERE PyWavelets** (`pip install PyWavelets`)

```python
result = analyzer.run_cwt(
    variable="gene1",
    condition="control",
    sampling_interval=None,    # Auto-detectado
    wavelet='cmor1.5-1.0',    # Morlet complejo
    period_range=(20.0, 28.0)
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `sampling_interval` | float/None | Spinner | Auto si None |
| `wavelet` | str | Dropdown | ['cmor1.5-1.0', 'morl', 'mexh'] |
| `period_range` | tuple | Dos spinners | (min, max) |

**Output:** `CWTResult` con dominant_period, period_variation, amplitude_modulations

---

#### 3.9 `run_lme()` - Modelo Mixto Lineal
**Propósito:** Análisis de datos jerárquicos con efectos aleatorios.

```python
result = analyzer.run_lme(
    dependent="gene1",              # Variable dependiente
    fixed_effects=["time", "condition"],  # Efectos fijos
    random_effect="subject",        # Agrupación aleatoria
    condition=None                  # Filtro opcional
)
```

| Parámetro GUI | Tipo | Widget | Notas |
|---------------|------|--------|-------|
| `dependent` | str | Dropdown | Variable a modelar |
| `fixed_effects` | list[str] | Multi-select | Columnas de efectos fijos |
| `random_effect` | str | Dropdown | Columna de agrupación |
| `condition` | str/None | Dropdown + "Todas" | Filtro opcional |

**Output:** `LMEResult` con términos, estimados, SE, z-values, p-values

---

### Funciones Batch (rhythm_analysis.py)

| Función | Descripción |
|---------|-------------|
| `run_all_jtk()` | JTK para todas las variables |
| `run_all_cosinor()` | Cosinor para todas las variables |
| `run_all_lomb_scargle()` | Lomb-Scargle para todas las variables |

Todas aceptan:
- `condition`: None para todas, o específica
- `variables`: None para todas, o lista específica
- `**kwargs`: Parámetros adicionales del método

---

## 4. RESUMEN DE WIDGETS GUI RECOMENDADOS

### Widgets Comunes
| Widget | Uso |
|--------|-----|
| **Dropdown** | variable, condition, model_type, loss, wavelet |
| **Spinner numérico** | period, n_components, n_harmonics, n_permutations |
| **Range slider** | period_range (min, max) |
| **Multi-select** | period_range (lista), fixed_effects |
| **Checkbox** | opciones booleanas |

### Organización Sugerida de GUI

```
┌─────────────────────────────────────────────────────────────┐
│  SELECCIÓN DE MÉTODO                                        │
│  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ Módulo:          │  │ ○ CosinorPy                     │  │
│  │ [Dropdown]       │  │ ○ CircaCompare                  │  │
│  │                  │  │ ○ Rhythm Analysis               │  │
│  └──────────────────┘  └─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  SELECCIÓN DE ANÁLISIS                                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ [Dropdown] Single Cosinor / Multi-Cosinor / Compare... ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  DATOS                                                      │
│  ┌─────────────────┐ ┌─────────────────┐                   │
│  │ Variable:       │ │ Condición:      │                   │
│  │ [Dropdown]      │ │ [Dropdown]      │                   │
│  └─────────────────┘ └─────────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│  PARÁMETROS (dinámicos según método seleccionado)          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Period: [24.0] │ Components: [1] │ Loss: [linear]      ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  [▶ EJECUTAR ANÁLISIS]                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. FORMATO CSV ESPERADO

### Datos Independientes (Standard)
```csv
time,condition,replicate,gene1,gene2,gene3
0,control,1,10.2,5.3,8.1
0,control,2,10.5,5.1,8.3
4,control,1,12.1,6.2,9.0
4,control,2,11.8,6.0,8.8
...
0,treatment,1,9.8,4.9,7.8
...
```

### Datos Dependientes (Population-mean)
```csv
time,condition,subject,gene1,gene2
0,control,mouse1,10.2,5.3
4,control,mouse1,12.1,6.2
8,control,mouse1,11.5,5.8
0,control,mouse2,10.5,5.1
4,control,mouse2,12.3,6.4
...
```

### Columnas Requeridas
| Columna | Requerida | Descripción |
|---------|-----------|-------------|
| `time` | ✅ | Tiempo en horas |
| `condition` | ✅ | Grupo experimental |
| `replicate` | Opcional | Para datos con réplicas técnicas |
| `subject` | Para population | ID de sujeto para datos dependientes |
| Variables | ✅ | Columnas numéricas con datos |
