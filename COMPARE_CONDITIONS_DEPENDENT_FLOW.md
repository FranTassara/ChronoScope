# Compare Conditions (Dependent Data) - Implementation Flow

## Overview

This document outlines the implementation flow for **"Compare Conditions (Dependent)"** in CircaScope, following the same careful planning approach used for Independent data.

## Key Differences: Independent vs Dependent Data

### Data Structure

**Independent Data:**
- Each measurement is independent
- DataFrame format: `test = "variable_condition"`
- Example: `gene1_control`, `gene1_treatment`

**Dependent Data (Population/Repeated Measures):**
- Multiple subjects/replicates per condition
- DataFrame format: `test = "variable_condition_repN"` (one per subject)
- Example: `gene1_control_rep1`, `gene1_control_rep2`, `gene1_control_rep3`
- Requires a **subject column** in the CSV

### Example CSV Structure

```csv
time,condition,subject,gene1,gene2,gene3
0,control,S1,10.2,8.5,20.0
2,control,S1,11.0,9.2,22.5
...
0,control,S2,9.8,8.1,19.5
...
0,treatment,S5,12.1,10.5,25.0
2,treatment,S5,13.5,12.0,28.5
```

## CosinorPy Functions Analysis

### 1. Single-Component Comparison: `population_test_cosinor_pairs()`

**Location:** `cosinor1.py:305`

**Function signature:**
```python
def population_test_cosinor_pairs(df, pairs, period=24, save_folder='', plot_on=True, **kwargs)
```

**Parameters:**
- `df`: DataFrame with columns `['x', 'y', 'test']` where `test` includes `_rep` suffix
- `pairs`: List of tuples `[('condition1', 'condition2'), ...]`
- `period`: Single period value (default: 24)
- `save_folder`: Folder for plots (empty string triggers `plt.show()` - **GUI FREEZE RISK**)
- `plot_on`: Whether to plot (default: True)

**Returns DataFrame with columns:**
```python
['test', 'd_amplitude', 'p(d_amplitude)', 'q(d_amplitude)', 'd_acrophase', 'p(d_acrophase)', 'q(d_acrophase)']
```

**How it works:**
1. For each pair `(cond1, cond2)`:
   - Filters `df` for rows where `test.startswith(f'{cond1}_rep')`
   - Filters `df` for rows where `test.startswith(f'{cond2}_rep')`
   - Fits population cosinor model for each condition
   - Compares using `population_test_cosinor(res1, res2)`
   - Plots both fits on same graph (black vs red)
   - Saves or shows plot
2. Applies FDR correction (Benjamini-Hochberg)

**Important notes:**
- Only works for **single component** (n_components=1)
- No `n_components` parameter
- Includes amplitude and acrophase comparison only (no mesor)
- **GUI FREEZE WARNING:** Same as Independent - must always provide `save_folder`

---

### 2. Multi-Component Comparison: `compare_pairs_population()`

**Location:** `cosinor.py:2086`

**Function signature:**
```python
def compare_pairs_population(
    df,
    pairs,
    n_components=3,           # Can be int or list
    period=24,                # Can be int or list
    folder='',                # For saving plots
    prefix='',                # Prefix for saved files
    analysis='CI',            # 'CI' or 'permutation'
    lin_comp=False,           # Include linear component
    model_type='lin',         # 'lin' or other
    df_results_extended=pd.DataFrame(columns=["test"]),  # Optional pre-computed single condition results
    parameters_to_analyse=['amplitude', 'acrophase', 'mesor'],
    parameters_angular=['acrophase'],
    **kwargs
)
```

**Parameters:**
- `df`: DataFrame with columns `['x', 'y', 'test']` where `test` includes `_rep` suffix
- `pairs`: List of tuples `[('condition1', 'condition2'), ...]`
- `n_components`: List of component counts `[1, 2, 3]` or single int
- `period`: List of periods `[24]` or single int
- `folder`: Empty string for no saving (no plt.show() in this function)
- `analysis`: `'CI'` (confidence intervals) or `'permutation'` (permutation test)
- `lin_comp`: Include linear component in model
- `parameters_to_analyse`: Which parameters to compare (default: all three)
- `parameters_angular`: Which parameters are angular (for proper circular stats)

**Returns DataFrame with columns:**
```python
['test', 'period', 'n_components', 'p1', 'p2', 'q1', 'q2',
 'd_amplitude', 'CI(d_amplitude)', 'p(d_amplitude)', 'q(d_amplitude)',
 'd_acrophase', 'CI(d_acrophase)', 'p(d_acrophase)', 'q(d_acrophase)',
 'd_mesor', 'CI(d_mesor)', 'p(d_mesor)', 'q(d_mesor)']
```

**How it works:**
1. Converts `period` and `n_components` to lists if they're ints
2. For each pair, period, and n_components combination:
   - Filters data for each condition (using `test.startswith()`)
   - Fits population models using `population_fit()`
   - Computes differences in rhythm parameters
   - If `analysis='CI'`:
     - Calls `compare_pair_population_CI()` to get confidence intervals
     - Computes p-values from CIs
   - If `analysis='permutation'`:
     - Calls `permutation_test_population_approx()` for p-values
3. Applies FDR correction for all parameters

**Important notes:**
- Supports **multi-component** models
- Includes **mesor** comparison (unlike single-component version)
- More flexible analysis options (CI vs permutation)
- **No GUI freeze risk** - doesn't call `plt.show()` directly

---

## GUI Parameter Mapping

### Existing GUI Parameters (Already Available)

From [ui/analysis_panel.py](ui/analysis_panel.py:844-949):

| GUI Parameter | CosinorPy Parameter | Notes |
|---------------|---------------------|-------|
| Period | `period` | Can be single value or comma-separated |
| Components | `n_components` | Can be single value or comma-separated |
| Comparison Type | N/A (single-component only) | Not used for dependent (only one comparison method) |
| Analysis Method | `analysis` | 'CI', 'Bootstrap', 'Sampling' → map to 'CI' or 'permutation' |
| Parameters to Compare | `parameters_to_analyse` | 'All', 'Amplitude', 'Acrophase', etc. |
| Include Linear Component | `lin_comp` | Boolean checkbox |
| Bootstrap Size | N/A | Not used in dependent methods |

### Parameters NOT Needed for Dependent

- **Comparison Type** (Pooled vs Independent Models): Only relevant for single-component independent data
- **Comparison Method** (Independent vs LimoRhyde): Only relevant for multi-component independent data
- **Bootstrap Size**: Dependent methods use CI or permutation, not bootstrap

### New Consideration: Analysis Method Mapping

**Current GUI options:**
- "CI" → Use `analysis='CI'`
- "Bootstrap" → ??? (no bootstrap for dependent)
- "Sampling" → ???

**Proposed mapping for Dependent:**
- "CI" → `analysis='CI'` (confidence intervals)
- "Permutation" → `analysis='permutation'` (permutation test)
- Hide "Bootstrap" and "Sampling" for dependent methods

---

## Implementation Flow

### Step 1: Data Preparation

**Already implemented in** `analysis_engine._convert_to_cosinorpy_format()` (line 3168):

```python
# For dependent data with subject column:
# Creates test = "variable_condition_rep1", "variable_condition_rep2", etc.
if subject_col and subject_col in df_filtered.columns:
    subjects = df_filtered[subject_col].unique()
    for i, subject in enumerate(subjects, 1):
        df_subject = df_filtered[df_filtered[subject_col] == subject].copy()
        df_subject_cosinorpy = pd.DataFrame({
            'x': df_subject[time_col].values,
            'y': df_subject[variable].values,
            'test': f"{variable}_{condition}_rep{i}"
        })
        df_cosinorpy = pd.concat([df_cosinorpy, df_subject_cosinorpy], ignore_index=True)
```

**Required:** Extract `subject_col` from data loader:
```python
subject_col = 'subject' if 'subject' in data.columns else None
```

This is already done in some methods (see line 3615).

---

### Step 2: Build Pairs List

**Format required by CosinorPy:**
```python
pairs = [('control', 'treatment'), ('control', 'mutant'), ...]
```

**NOT:**
```python
pairs = [('gene1_control', 'gene1_treatment'), ...]  # WRONG for dependent!
```

**Why:** CosinorPy's dependent functions use `test.startswith(condition_name)` to filter, so pairs should be condition names only (not including variable prefix).

**Implementation:**
```python
# In analysis_engine._run_cosinorpy_compare_dependent_new():
all_conditions = data[condition_col].unique().tolist()

# Generate all pairwise combinations
from itertools import combinations
pairs = list(combinations(all_conditions, 2))
# Example: [('control', 'treatment'), ('control', 'mutant'), ('treatment', 'mutant')]
```

---

### Step 3: Single-Component vs Multi-Component Branch

```python
# Parse n_components from parameters
n_components_str = parameters.get('n_components', '1')
n_components_list = [int(x.strip()) for x in n_components_str.split(',')]

# Parse period
period_str = parameters.get('period', '24')
period_list = [float(x.strip()) for x in period_str.split(',')]

# Determine if single or multi-component
is_single_component = len(n_components_list) == 1 and n_components_list[0] == 1

if is_single_component:
    # Use cosinor1.population_test_cosinor_pairs()
    df_results = _compare_dependent_single(
        df_cosinorpy, pairs, period_list[0], parameters
    )
else:
    # Use cosinor.compare_pairs_population()
    df_results = _compare_dependent_multi(
        df_cosinorpy, pairs, period_list, n_components_list, parameters
    )
```

---

### Step 4: Single-Component Implementation

```python
def _compare_dependent_single(
    self, df_cosinorpy, pairs, period, parameters
):
    """Single-component comparison using population_test_cosinor_pairs."""
    import tempfile
    import shutil

    save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)
    save_folder = parameters.get('cosinorpy_plot_folder', '')

    # CRITICAL: Always provide a folder to avoid GUI freeze
    use_temp_folder = False
    if save_cosinorpy_plots and save_folder:
        folder = save_folder
        plot_on = True
    else:
        folder = tempfile.mkdtemp(prefix='cosinorpy_temp_')
        plot_on = False  # Don't plot to avoid overhead
        use_temp_folder = True

    try:
        df_results = cosinor1.population_test_cosinor_pairs(
            df_cosinorpy,
            pairs=pairs,
            period=period,
            save_folder=folder,
            plot_on=plot_on
        )

        return {
            'comparison_type': 'dependent_single',
            'n_pairs': len(pairs),
            'results_df': df_results
        }
    finally:
        if use_temp_folder:
            shutil.rmtree(folder, ignore_errors=True)
```

**Output columns:**
- `test`: e.g., "control vs treatment"
- `d_amplitude`, `p(d_amplitude)`, `q(d_amplitude)`
- `d_acrophase`, `p(d_acrophase)`, `q(d_acrophase)`

**Note:** No mesor comparison in single-component.

---

### Step 5: Multi-Component Implementation

```python
def _compare_dependent_multi(
    self, df_cosinorpy, pairs, period_list, n_components_list, parameters
):
    """Multi-component comparison using compare_pairs_population."""

    # Get analysis method
    analysis_method = parameters.get('analysis_method', 'CI')
    # Map GUI options to CosinorPy options
    if analysis_method == 'CI':
        analysis_param = 'CI'
    elif analysis_method in ('Bootstrap', 'Sampling'):
        analysis_param = 'permutation'  # Use permutation instead
    else:
        analysis_param = 'CI'

    # Get parameters to compare
    params_to_compare_str = parameters.get('parameters_to_compare', 'All Parameters')
    if params_to_compare_str == 'All Parameters':
        parameters_to_analyse = ['amplitude', 'acrophase', 'mesor']
    elif params_to_compare_str == 'Amplitude':
        parameters_to_analyse = ['amplitude']
    elif params_to_compare_str == 'Acrophase':
        parameters_to_analyse = ['acrophase']
    elif params_to_compare_str == 'Amplitude & Acrophase':
        parameters_to_analyse = ['amplitude', 'acrophase']

    # Linear component
    lin_comp = parameters.get('include_lin_comp', False)

    results_list = []

    for per in period_list:
        for n_comp in n_components_list:
            print(f"[DEBUG] Processing period={per}, n_components={n_comp}")

            # NOTE: CosinorPy expects period and n_components as lists
            df_result = cosinor.compare_pairs_population(
                df_cosinorpy,
                pairs=pairs,
                n_components=[n_comp],  # Must be list
                period=[per],           # Must be list
                analysis=analysis_param,
                parameters_to_analyse=parameters_to_analyse,
                parameters_angular=['acrophase'],
                lin_comp=lin_comp,
                folder=''  # Don't save plots for now
            )

            results_list.append(df_result)

    # Combine results
    if len(results_list) == 1:
        df_combined = results_list[0]
    else:
        df_combined = pd.concat(results_list, ignore_index=True)

    return {
        'comparison_type': 'dependent_multi',
        'n_pairs': len(pairs),
        'results_df': df_combined
    }
```

**Output columns:**
- `test`: e.g., "control vs treatment"
- `period`, `n_components`
- `p1`, `p2`, `q1`, `q2` (p/q values for each condition individually)
- For each parameter in `parameters_to_analyse`:
  - `d_{param}`: difference
  - `CI(d_{param})`: confidence interval (if analysis='CI')
  - `p(d_{param})`: p-value
  - `q(d_{param})`: FDR-corrected q-value

---

### Step 6: Parsing Results

**Location:** `analysis_engine._parse_comparison_results()` (needs extension)

**Current parsing** (line 3784-3820) handles Independent format:
- Extracts condition names from `test` column (e.g., "gene1_control vs gene1_treatment")
- Conditional parsing for Independent vs LimoRhyde column names

**New requirement for Dependent:**

```python
# In _parse_comparison_results():

# Detect if this is dependent data format
is_dependent = 'p1' in df_results.columns or 'p2' in df_results.columns

if is_dependent:
    # Dependent format: test = "condition1 vs condition2"
    # Extract condition names differently
    cond1, cond2 = self._parse_dependent_test_column(row['test'])
else:
    # Independent format: test = "variable_condition1 vs variable_condition2"
    cond1, cond2 = self._parse_independent_test_column(row['test'], variable)
```

**Helper function:**
```python
def _parse_dependent_test_column(self, test_str: str) -> Tuple[str, str]:
    """Parse test column for dependent data (e.g., 'control vs treatment')."""
    parts = test_str.split(' vs ')
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        # Fallback
        return test_str, test_str
```

**Column differences:**

| Data Type | p-value columns | Notes |
|-----------|----------------|-------|
| Independent (Single) | `p(d_amplitude)`, `p(d_acrophase)` | No p1/p2 |
| Independent (Multi) | `p(d_amplitude)`, `p(d_acrophase)`, `p(d_mesor)` | No p1/p2 |
| Dependent (Single) | `p(d_amplitude)`, `p(d_acrophase)` | No p1/p2, no mesor |
| Dependent (Multi) | `p(d_amplitude)`, `p(d_acrophase)`, `p(d_mesor)`, `p1`, `p2` | Has p1/p2 |

**Mesor handling:**
- Single-component dependent: **No mesor** comparison
- Multi-component dependent: **Has mesor** comparison

---

### Step 7: Results Display

**Already implemented** in [ui/results_panel.py](ui/results_panel.py:466-475):

Columns include:
- `variable`, `condition1`, `condition2`, `method`, `n_components`, `period`
- Amplitude: `amplitude_g0`, `amplitude_g1`, `amplitude_diff`, `p_amplitude`, `q_amplitude`, `amplitude_diff_ci`
- Acrophase: similar structure
- MESOR: similar structure
- `me`, `resid_se`, `aic`, `bic`

**Note:** Dependent data doesn't have AIC/BIC in comparison results, only `p1`, `p2`, `q1`, `q2`.

**May need to add:**
- `p1`, `p2` columns for dependent multi-component

---

## UI Parameter Flow

### Current UI Behavior (for Independent)

From [ui/analysis_panel.py](ui/analysis_panel.py:1025-1070):

```python
def _update_compare_conditions_parameters(self):
    # Parse n_components
    is_single_component = len(components) == 1 and components[0] == 1

    if is_single_component:
        # Show: Comparison Type (Pooled vs Independent Models)
        self._show_param("Comparison Type:")
    else:
        # Show: Comparison Method (Independent vs LimoRhyde)
        self._show_param("Comparison Method:")
        self._show_param("Bootstrap Size:")

        if comparison_method == 'Independent':
            self._show_param("Analysis Method:")
            self._show_param("Parameters to Compare:")
            self._show_checkbox(self._include_lin_comp_check)
        else:  # LimoRhyde
            # ...
```

### Required UI Behavior for Dependent

**Single-component (n_components=1):**
- **Hide:** Comparison Type, Comparison Method, Bootstrap Size
- **Show:** Nothing extra (just Period, Components, Save plots checkbox)

**Multi-component (n_components>1):**
- **Hide:** Comparison Type, Comparison Method, Bootstrap Size
- **Show:**
  - Analysis Method (but map Bootstrap/Sampling to Permutation)
  - Parameters to Compare
  - Include Linear Component checkbox

**Implementation:**
```python
def _update_compare_conditions_parameters(self):
    method_text = self._method_combo.currentText()

    # Determine if this is Dependent data method
    is_dependent = "Dependent" in method_text

    # Parse n_components
    components_text = self._components_edit.text().strip()
    try:
        components = [int(x.strip()) for x in components_text.split(',') if x.strip()]
        is_single_component = len(components) == 1 and components[0] == 1
    except ValueError:
        is_single_component = False

    # Hide all first
    self._hide_param("Comparison Type:")
    self._hide_param("Comparison Method:")
    self._hide_param("Analysis Method:")
    self._hide_param("Parameters to Compare:")
    self._hide_param("Bootstrap Size:")
    self._hide_checkbox(self._include_lin_comp_check)

    if is_dependent:
        # Dependent data: simpler UI
        if not is_single_component:
            # Multi-component: show analysis options
            self._show_param("Analysis Method:")
            self._show_param("Parameters to Compare:")
            self._show_checkbox(self._include_lin_comp_check)
            # Could add tooltip: "Bootstrap and Sampling use Permutation for dependent data"
    else:
        # Independent data: existing complex logic
        if is_single_component:
            self._show_param("Comparison Type:")
        else:
            self._show_param("Comparison Method:")
            self._show_param("Bootstrap Size:")
            # ... rest of existing logic
```

---

## Testing Plan

### Test 1: Single-Component Dependent

**Test data:** [examples/population_mean_test_data.csv](examples/population_mean_test_data.csv)
- 2 conditions: control, treatment
- 4 subjects per condition (S1-S4 for control, S5-S8 for treatment)
- 3 variables: gene1, gene2, gene3
- Period: 24h

**Expected behavior:**
1. Load CSV → CircaScope detects `subject` column
2. Select method: "Compare Conditions (Dependent)"
3. Set Components: 1
4. Set Period: 24
5. Click Run Analysis
6. Should compare: control vs treatment (3 pairs for 3 genes × 1 pair = 3 result rows)

**Expected results:**
- 3 rows (gene1, gene2, gene3)
- Columns: d_amplitude, p(d_amplitude), q(d_amplitude), d_acrophase, p(d_acrophase), q(d_acrophase)
- NO mesor columns

### Test 2: Multi-Component Dependent

**Test data:** Same CSV, but with components=2

**Expected behavior:**
1. Set Components: 2
2. UI should show: Analysis Method, Parameters to Compare, Linear Component checkbox
3. Click Run Analysis

**Expected results:**
- 3 rows (gene1, gene2, gene3)
- Columns: period, n_components, p1, p2, q1, q2, d_amplitude, CI(d_amplitude), p(d_amplitude), q(d_amplitude), d_acrophase, CI(d_acrophase), p(d_acrophase), q(d_acrophase), d_mesor, CI(d_mesor), p(d_mesor), q(d_mesor)
- HAS mesor columns

### Test 3: Multiple Periods and Components

**Test data:** Same CSV

**Expected behavior:**
1. Set Components: 1, 2, 3
2. Set Period: 24, 12
3. Click Run Analysis

**Expected results:**
- 3 variables × 3 components × 2 periods = 18 result rows
- Each row should have correct period and n_components values

---

## Potential Issues & Solutions

### Issue 1: GUI Freeze with Plots

**Solution:** Same as Independent - always use temporary folder for single-component method.

### Issue 2: Missing Subject Column

**Symptom:** User loads CSV without `subject` column but selects Dependent method.

**Solution:**
1. Detect in data loader if subject column exists
2. Disable "Dependent" methods if no subject column detected
3. Show warning message: "Dependent data methods require a 'subject' column in your CSV"

**Implementation:**
```python
# In analysis_panel.py when populating method combo:
has_subject_col = loader.get_dataset_info().subject_column is not None

if has_subject_col:
    # Add dependent methods
    methods.append("Compare Conditions (Dependent)")
else:
    # Only add independent methods
    methods.append("Compare Conditions (Independent)")
```

### Issue 3: Pairs List Format

**Symptom:** Pairs formatted as `[('gene1_control', 'gene1_treatment')]` instead of `[('control', 'treatment')]`

**Solution:** For dependent methods, pairs should be **condition names only**, not including variable prefix.

### Issue 4: Results Table - Missing p1/p2 Columns

**Symptom:** Multi-component dependent results include `p1`, `p2`, `q1`, `q2` but results table doesn't show them.

**Solution:** Add these columns to results table for dependent methods:

```python
# In results_panel.py:
if is_comparison and is_dependent:
    columns = ['variable', 'condition1', 'condition2', 'method', 'n_components', 'period',
               'p1', 'q1', 'p2', 'q2',  # Individual condition p/q values
               'amplitude_g0', 'amplitude_g1', 'amplitude_diff', 'p_amplitude', 'q_amplitude', 'amplitude_diff_ci',
               # ... rest of columns
    ]
```

### Issue 5: MESOR Column Mismatch

**Symptom:** Single-component results show N/A for mesor (not computed), but table expects it.

**Solution:** Already handled by using `row.get('d_mesor')` which returns None if column doesn't exist.

---

## Summary Checklist

### New Code Required

- [ ] `analysis_engine._run_cosinorpy_compare_dependent_new()` (main entry point)
- [ ] `cosinor_analysis._compare_dependent_single()` (single-component wrapper)
- [ ] `cosinor_analysis._compare_dependent_multi()` (multi-component wrapper)
- [ ] `analysis_engine._parse_dependent_test_column()` (parse "control vs treatment" format)
- [ ] Update `analysis_engine._parse_comparison_results()` to handle dependent format
- [ ] Update `analysis_panel._update_compare_conditions_parameters()` for dependent UI flow
- [ ] Add validation to check for subject column before enabling dependent methods

### Existing Code to Reuse

- [x] `_convert_to_cosinorpy_format()` - already handles dependent data transformation
- [x] `_parse_ci()` - already handles CI parsing
- [x] Results table display - already has n_components column
- [x] Temporary folder solution - can reuse for single-component dependent

### Testing

- [ ] Test with single-component dependent data
- [ ] Test with multi-component dependent data
- [ ] Test with multiple periods and components
- [ ] Test UI parameter visibility for dependent methods
- [ ] Test error handling when subject column is missing

---

## Next Steps

1. **Review this flow document** with user
2. **Implement** `_run_cosinorpy_compare_dependent_new()` following the pattern established for Independent
3. **Add UI logic** to differentiate dependent vs independent parameter visibility
4. **Test** with population_mean_test_data.csv
5. **Iterate** based on test results

---

## Questions for User

1. Should we allow **both Analysis Method options** for multi-component dependent (CI and Permutation), or just CI?
2. For the **pairs list**, should we compare ALL conditions automatically (like we do for Independent), or allow user selection?
3. Should we **disable dependent methods** if no subject column is detected, or just show a warning?
4. Do we want to display **p1, p2, q1, q2** columns in the results table for multi-component dependent?

