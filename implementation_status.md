# Estado de Implementación de Métodos en CircaScope GUI

## 1. CosinorPy (cosinor_analysis.py)

| Método | En GUI? | En Engine? | Notas |
|--------|---------|------------|-------|
| `single_cosinor()` | ✅ | ✅ | Implementado como COSINORPY_SINGLE |
| `single_cosinor_all()` | ❌ | ❌ | Batch - no implementado |
| `multi_cosinor()` | ✅ | ✅ | Implementado como COSINORPY_MULTI |
| `find_best_model()` | ❌ | ❌ | No implementado |
| `population_cosinor()` | ✅ | ✅ | Implementado como COSINORPY_POPULATION ⭐ |
| `compare_conditions()` | ✅ | ✅ | Implementado como COSINORPY_COMPARE |
| `compare_all_conditions()` | ❌ | ❌ | Batch - no implementado |
| `compare_variables()` | ❌ | ❌ | No implementado |
| `nonlinear_cosinor()` | ✅ | ✅ | Implementado como COSINORPY_NONLINEAR ⭐ |
| `fit_count_data()` | ✅ | ✅ | Implementado como COSINORPY_COUNT ⭐ |

## 2. CircaCompare (circacompare_analysis.py)

| Método | En GUI? | En Engine? | Notas |
|--------|---------|------------|-------|
| `fit_single()` | ✅ | ✅ | Implementado correctamente |
| `fit_single_all()` | ❌ | ❌ | Batch - no implementado |
| `compare()` | ✅ | ✅ | Implementado correctamente |
| `compare_all_conditions()` | ❌ | ❌ | Batch - no implementado |
| `compare_all_variables()` | ❌ | ❌ | Batch - no implementado |

## 3. Rhythm Analysis (rhythm_analysis.py)

| Método | En GUI? | En Engine? | Notas |
|--------|---------|------------|-------|
| `run_jtk()` | ✅ | ✅ | Implementado como RHYTHM_JTK |
| `run_ar_jtk()` | ✅ | ✅ | Implementado como AR_JTK ⭐ |
| `run_cosine_kendall()` | ✅ | ✅ | Implementado como COSINE_KENDALL ⭐ |
| `run_cosinor()` | ✅ | ✅ | Implementado como RHYTHM_COSINOR |
| `run_harmonic_cosinor()` | ✅ | ✅ | Implementado como RHYTHM_HARMONIC |
| `run_fourier_f24()` | ✅ | ✅ | Implementado como FOURIER_F24 ⭐ |
| `run_lomb_scargle()` | ✅ | ✅ | Implementado como RHYTHM_LOMB |
| `run_cwt()` | ✅ | ✅ | Implementado como CWT ⭐ |
| `run_lme()` | ✅ | ✅ | Implementado como LME ⭐ |

## Resumen

### ✅ Completamente implementados (17/22):
1. CosinorPy Single
2. CosinorPy Multi
3. CosinorPy Population ⭐ (nuevo)
4. CosinorPy Compare
5. CosinorPy Count Data ⭐ (nuevo)
6. CosinorPy Nonlinear ⭐ (nuevo)
7. CircaCompare Single
8. CircaCompare Compare
9. JTK Cycle
10. AR-JTK ⭐ (nuevo)
11. Cosine-Kendall ⭐ (nuevo)
12. Cosinor (OLS)
13. Harmonic Cosinor
14. Fourier F24 ⭐ (nuevo)
15. Lomb-Scargle
16. Wavelet (CWT) ⭐ (nuevo)
17. Linear Mixed Effects (LME) ⭐ (nuevo)

### ⚠️ En GUI pero falta engine (0/22):
(Todos los métodos ahora tienen implementación completa en el engine)

### ❌ No implementados (5/22):
1. single_cosinor_all (batch) - CosinorPy
2. find_best_model - CosinorPy
3. compare_all_conditions - CosinorPy
4. compare_variables - CosinorPy
5. fit_single_all (batch) - CircaCompare
6. compare_all_conditions (CircaCompare)
7. compare_all_variables - CircaCompare

Nota: La mayoría son funciones batch (_all) que pueden implementarse después

## Prioridades para implementar:

### ✅ Alta prioridad - COMPLETADO:
1. ✅ **CosinorPy Population** - para datos dependientes
2. ✅ **CosinorPy Count Data** - para RNA-seq
3. ✅ **CosinorPy Nonlinear** - cosinor no lineal
4. ✅ **AR-JTK** - JTK con corrección de autocorrelación
5. ✅ **Cosine-Kendall** - análisis no paramétrico
6. ✅ **Fourier F24** - effect size measure
7. ✅ **Wavelet (CWT)** - análisis tiempo-frecuencia
8. ✅ **Linear Mixed Effects (LME)** - modelos mixtos

### Media prioridad (métodos especializados):
- find_best_model (CosinorPy)
- compare_variables (CosinorPy)

### Baja prioridad (funciones batch):
- Todas las funciones _all() que procesan múltiples variables
- Estas son útiles pero pueden implementarse después
