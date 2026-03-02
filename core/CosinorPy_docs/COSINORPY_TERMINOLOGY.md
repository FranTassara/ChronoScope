# CosinorPy Terminology: "Dependent" vs "Independent"

## Important Distinction

CosinorPy uses the terms "dependent" and "independent" in **TWO DIFFERENT CONTEXTS**, which can cause confusion:

---

## 1. DATA TYPE (Experimental Design)

Refers to how the data was collected:

### INDEPENDENT Data
- **Description**: Biological replicates - different subjects measured at each timepoint
- **Data structure**: No `subject` column required
- **Example**: Measuring gene expression in different mice at each timepoint
- **CircaScope detection**: `analysis_mode = AnalysisMode.INDEPENDENT`

### DEPENDENT Data
- **Description**: Repeated measures - same subjects measured multiple times
- **Data structure**: Requires `subject` column to track individuals
- **Example**: Measuring body temperature of the same person at different times
- **CircaScope detection**: `analysis_mode = AnalysisMode.DEPENDENT`

---

## 2. MODEL TYPE (Statistical Approach)

Refers to how the comparison model is fitted:

### INDEPENDENT Model
- **Description**: Each condition fitted with its own separate period
- **Use case**: When you expect different periods in each condition
- **CosinorPy function**: `fit_generalized_cosinor_compare_pairs_independent()`
- **CircaScope parameter**: `use_dependent_model = False`

### DEPENDENT Model
- **Description**: Both conditions constrained to share the same period
- **Use case**: When you expect the same period but different amplitude/phase
- **CosinorPy function**: `fit_generalized_cosinor_compare_pairs_dependent()`
- **CircaScope parameter**: `use_dependent_model = True`

---

## Function Mapping in CircaScope

The `compare_conditions_nonlinear()` method in `cosinor_analysis.py` selects the appropriate CosinorPy function based on BOTH factors:

| Data Type | Model Type | CosinorPy Function |
|-----------|------------|-------------------|
| INDEPENDENT | Independent | `fit_generalized_cosinor_compare_pairs_independent()` |
| INDEPENDENT | Dependent | `fit_generalized_cosinor_compare_pairs_dependent()` |
| DEPENDENT | Either | `population_fit_generalized_cosinor_compare_pairs()` |

---

## Code Example

```python
# Example 1: Independent data + Independent model
# Data: Different mice at each timepoint
# Model: Each condition can have different period
result = engine.run_comparison(
    data=data,
    variable='gene_expression',
    condition1='control',
    condition2='treated',
    analysis_type=AnalysisType.COSINORPY_COMPARE_NONLINEAR,
    parameters={
        'period': 24.0,
        'use_dependent_model': False  # MODEL TYPE: independent
    }
)
# Will use: fit_generalized_cosinor_compare_pairs_independent()

# Example 2: Independent data + Dependent model
# Data: Different mice at each timepoint
# Model: Both conditions constrained to same period
result = engine.run_comparison(
    data=data,
    variable='gene_expression',
    condition1='control',
    condition2='treated',
    analysis_type=AnalysisType.COSINORPY_COMPARE_NONLINEAR,
    parameters={
        'period': 24.0,
        'use_dependent_model': True  # MODEL TYPE: dependent
    }
)
# Will use: fit_generalized_cosinor_compare_pairs_dependent()

# Example 3: Dependent data (repeated measures)
# Data: Same subjects measured multiple times (has 'subject' column)
# Model: Uses population method (ignores use_dependent_model parameter)
result = engine.run_comparison(
    data=data_with_subjects,  # Must have 'subject' column
    variable='body_temperature',
    condition1='morning',
    condition2='evening',
    analysis_type=AnalysisType.COSINORPY_COMPARE_NONLINEAR,
    parameters={
        'period': 24.0,
        'use_dependent_model': False  # Ignored for repeated measures
    }
)
# Will use: population_fit_generalized_cosinor_compare_pairs()
```

---

## Key Takeaway

⚠️ **CRITICAL**: The function names `fit_generalized_cosinor_compare_pairs_dependent()` and `fit_generalized_cosinor_compare_pairs_independent()` refer to the **MODEL TYPE**, NOT the **DATA TYPE**!

Both can be used with INDEPENDENT data (biological replicates). The choice depends on whether you want to constrain both conditions to the same period (dependent model) or allow each condition to have its own period (independent model).

---

## References

- See `core/cosinor_analysis.py` lines 2217-2370 for implementation
- See `core/CosinorPy/demo_independent_nonlin.ipynb` cells 28-31 for CosinorPy examples
- See `core/CosinorPy/demo_dependent_nonlin.ipynb` for population/repeated measures examples
