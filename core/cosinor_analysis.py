"""
Cosinor Analysis Module (Refactored)
=====================================

Modular wrapper for CosinorPy library with 10 analysis methods:
1. Periodogram Analysis
2. Cosinor Analysis (Independent Data)
3. Cosinor Analysis (Dependent/Population Data)
4. Compare Conditions (Independent Data)
5. Compare Conditions (Dependent/Population Data)
6. Non-Linear Analysis (Independent Data)
7. Non-Linear Analysis (Dependent/Population Data)
8. Non-Linear Compare Conditions (Independent Data)
9. Non-Linear Compare Conditions (Dependent/Population Data)
10. Count Data Analysis (via model_type parameter)
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Check CosinorPy availability
try:
    import CosinorPy.cosinor as cosinor
    import CosinorPy.cosinor1 as cosinor1
    import CosinorPy.cosinor_nonlin as cosinor_nonlin
    COSINORPY_AVAILABLE = True
except ImportError:
    COSINORPY_AVAILABLE = False
    warnings.warn("CosinorPy not available. Cosinor analysis methods will be disabled.")


class DataType(Enum):
    """Type of data for analysis."""
    INDEPENDENT = "independent"  # Different subjects at each timepoint
    DEPENDENT = "dependent"      # Same subjects measured repeatedly (population/longitudinal)


class ModelType(Enum):
    """Statistical model type for count data."""
    NORMAL = "normal"              # Standard cosinor (continuous data)
    POISSON = "poisson"            # Count data (RNA-seq, etc.)
    NEGATIVE_BINOMIAL = "negative_binomial"  # Overdispersed count data


class AnalysisMethod(Enum):
    """Method for extended analysis (CI computation)."""
    CI = "CI"                # Confidence intervals via analytical methods
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling
    SAMPLING = "sampling"    # Monte Carlo sampling (for population data)


class Criterium(Enum):
    """Criterium for selecting best model."""
    RSS = "RSS"                      # Residual Sum of Squares (lower is better)
    AIC = "AIC"                      # Akaike Information Criterion
    BIC = "BIC"                      # Bayesian Information Criterion
    LOG_LIKELIHOOD = "log-likelihood"  # Log-likelihood (higher is better)


@dataclass
class CosinorParameters:
    """Parameters for cosinor analysis."""
    period: Union[float, List[float]] = 24.0  # Single value or list/range
    n_components: List[int] = None             # [1], [1,2,3], etc.
    model_type: ModelType = ModelType.NORMAL
    criterium: Criterium = Criterium.RSS
    analysis_method: AnalysisMethod = AnalysisMethod.CI
    params_ci_analysis: str = "sampling"       # For population data
    bootstrap_size: int = 100
    amplification: Optional[float] = None      # For nonlinear analysis
    lin_comp: Optional[float] = None           # For nonlinear analysis

    def __post_init__(self):
        """Initialize default values."""
        if self.n_components is None:
            self.n_components = [1]


class CosinorAnalyzer:
    """
    Modular wrapper for CosinorPy analysis.

    Provides 10 analysis methods with clean, consistent API.
    """

    def __init__(self):
        """Initialize the analyzer."""
        if not COSINORPY_AVAILABLE:
            raise ImportError("CosinorPy is required for cosinor analysis")

        self._data: Optional[pd.DataFrame] = None
        self._period: Optional[float] = None
        self._data_type: Optional[DataType] = None

    def load_data(
        self,
        data: pd.DataFrame,
        data_type: DataType = DataType.INDEPENDENT
    ) -> None:
        """
        Load data for analysis.

        Args:
            data: DataFrame in CosinorPy format (columns: x, y, test)
            data_type: INDEPENDENT or DEPENDENT
        """
        self._data = data.copy()
        self._data_type = data_type
        print(f"[DEBUG] CosinorAnalyzer loaded {len(data)} rows, data_type={data_type.value}")

    def set_period(self, period: Union[float, List[float]]) -> None:
        """Set the period for analysis."""
        self._period = period
        print(f"[DEBUG] CosinorAnalyzer period set to {period}")

    # ========================================================================
    # METHOD 1: PERIODOGRAM ANALYSIS
    # ========================================================================

    def periodogram(
        self,
        save_folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate periodogram plots for all variables.

        Uses: cosinor.periodogram_df(df, folder=None)

        Args:
            save_folder: Optional folder to save plots

        Returns:
            Dict with status and message
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print(f"[DEBUG periodogram] Generating periodograms for {len(self._data['test'].unique())} tests")
        print(f"[DEBUG periodogram] Save folder: {save_folder}")

        try:
            # Call CosinorPy periodogram function
            cosinor.periodogram_df(self._data, folder=save_folder)

            return {
                'success': True,
                'message': f'Periodograms generated for {len(self._data["test"].unique())} variables',
                'n_variables': len(self._data['test'].unique()),
                'save_folder': save_folder
            }

        except Exception as e:
            print(f"[DEBUG periodogram] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Periodogram failed: {str(e)}'
            }

    # ========================================================================
    # METHOD 2: COSINOR ANALYSIS (INDEPENDENT DATA)
    # ========================================================================

    def cosinor_independent(
        self,
        variable: str,
        condition: str,
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        model_type: ModelType = ModelType.NORMAL,
        criterium: Criterium = Criterium.RSS,
        analysis_method: AnalysisMethod = AnalysisMethod.CI,
        bootstrap_size: int = 1000,
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Cosinor analysis for independent data.

        Workflow:
        1. If n_components=[1]: Use cosinor1.fit_group() for single component
        2. If n_components=[1,2,3]: Use cosinor.fit_group() then cosinor.get_best_fits()
        3. Get best models: cosinor.get_best_models()
        4. Extended analysis: cosinor.analyse_best_models()
        5. Plot: cosinor.plot_df_models()

        Args:
            variable: Variable name
            condition: Condition name
            period: Single value or list (e.g., 24 or [20,21,22,23,24,25,26])
            n_components: List of components to test (e.g., [1], [1,2,3])
            model_type: NORMAL, POISSON, or NEGATIVE_BINOMIAL
            criterium: RSS, AIC, BIC, or LOG_LIKELIHOOD
            analysis_method: CI or BOOTSTRAP
            save_folder: Optional folder for plots

        Returns:
            Dict with analysis results including ME, resid_SE, etc.
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG cosinor_independent] variable={variable}, condition={condition}")
        print(f"[DEBUG cosinor_independent] period={period}, n_components={n_components}")
        print(f"[DEBUG cosinor_independent] model_type={model_type.value}, criterium={criterium.value}")

        # Filter data for this test
        test_name = f"{variable}_{condition}"
        df_test = self._data[self._data['test'] == test_name].copy()

        if len(df_test) == 0:
            raise ValueError(f"No data found for test: {test_name}")

        print(f"[DEBUG cosinor_independent] Filtered {len(df_test)} rows for {test_name}")

        try:
            # Single component analysis (only for normal/linear model - cosinor1 doesn't support Poisson/NB)
            if len(n_components) == 1 and n_components[0] == 1 and model_type == ModelType.NORMAL:
                return self._cosinor_independent_single(
                    df_test, test_name, period, save_folder, save_cosinorpy_plots
                )

            # Multi-component analysis (also handles non-normal model types for single component)
            else:
                return self._cosinor_independent_multi(
                    df_test, test_name, period, n_components,
                    criterium, analysis_method, bootstrap_size, save_folder, save_cosinorpy_plots,
                    model_type=model_type
                )

        except Exception as e:
            print(f"[DEBUG cosinor_independent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _cosinor_independent_single(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period: Union[float, List[float]],
        save_folder: Optional[str],
        save_cosinorpy_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Single-component cosinor analysis using cosinor1.fit_group().

        Note: cosinor1.fit_group() is richer in statistics than cosinor.fit_group()
        for single-component analysis, as mentioned in the CosinorPy notebooks.
        We calculate MESOR manually from the data.
        """
        print(f"[DEBUG] Using cosinor1.fit_group for single component")
        print(f"[DEBUG] save_cosinorpy_plots={save_cosinorpy_plots}")

        # Ensure period is a list
        if isinstance(period, (int, float)):
            period = [period]

        # Disable matplotlib interactive mode to prevent plots from showing
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Call cosinor1.fit_group
        # Use plot_on and save_folder parameters to control plot generation
        if save_cosinorpy_plots and save_folder:
            print(f"[DEBUG] Generating CosinorPy plots to {save_folder}")
            df_results = cosinor1.fit_group(df_test, period=period, save_folder=save_folder, plot_on=True)
            print(f"[DEBUG] CosinorPy plots saved to {save_folder}")
        else:
            # No plots - use plot_on=False to prevent matplotlib warnings
            df_results = cosinor1.fit_group(df_test, period=period, plot_on=False)

        print(f"[DEBUG] cosinor1.fit_group returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        # Extract results for this test
        test_results = df_results[df_results['test'] == test_name]

        # Calculate MESOR from the data (mean of y values for this test)
        test_data = df_test[df_test['test'] == test_name]
        mesor = test_data['y'].mean() if len(test_data) > 0 else None

        # If multiple periods were tested, return ALL results with best_model indicator
        if len(test_results) > 1:
            print(f"[DEBUG] Multiple periods tested ({len(test_results)}), returning all results")
            # Find best model (lowest p-value)
            best_idx = test_results['p'].idxmin()

            # Create a result dict for each period
            results_list = []
            for idx, row in test_results.iterrows():
                # Parse CI columns
                amp_ci_str = row.get('CI(amplitude)', '')
                acro_ci_str = row.get('CI(acrophase)', '')

                print(f"[DEBUG] CI parsing - amp_ci_str: {amp_ci_str} (type: {type(amp_ci_str)})")
                print(f"[DEBUG] CI parsing - acro_ci_str: {acro_ci_str} (type: {type(acro_ci_str)})")

                try:
                    if amp_ci_str:
                        if isinstance(amp_ci_str, (list, tuple)) and len(amp_ci_str) == 2:
                            # Already a list/tuple
                            amplitude_ci = (float(amp_ci_str[0]), float(amp_ci_str[1]))
                        elif isinstance(amp_ci_str, str):
                            # Parse string format
                            amp_ci_parts = amp_ci_str.strip('()[]').split(',')
                            amplitude_ci = (float(amp_ci_parts[0]), float(amp_ci_parts[1]))
                        else:
                            amplitude_ci = None
                    else:
                        amplitude_ci = None
                except Exception as e:
                    print(f"[DEBUG] Failed to parse amplitude CI: {e}")
                    amplitude_ci = None

                try:
                    if acro_ci_str:
                        if isinstance(acro_ci_str, (list, tuple)) and len(acro_ci_str) == 2:
                            # Already a list/tuple
                            acrophase_ci = (float(acro_ci_str[0]), float(acro_ci_str[1]))
                        elif isinstance(acro_ci_str, str):
                            # Parse string format
                            acro_ci_parts = acro_ci_str.strip('()[]').split(',')
                            acrophase_ci = (float(acro_ci_parts[0]), float(acro_ci_parts[1]))
                        else:
                            acrophase_ci = None
                    else:
                        acrophase_ci = None
                except Exception as e:
                    print(f"[DEBUG] Failed to parse acrophase CI: {e}")
                    acrophase_ci = None

                print(f"[DEBUG] Parsed amplitude_ci: {amplitude_ci}")
                print(f"[DEBUG] Parsed acrophase_ci: {acrophase_ci}")

                result_dict = {
                    'test_name': test_name,
                    'period': row['period'],
                    'n_components': 1,
                    'amplitude': row['amplitude'],
                    'acrophase': row['acrophase'],
                    'acrophase_hours': row.get('acrophase[h]', None),
                    'mesor': mesor,
                    'p_value': row['p'],
                    'q_value': row.get('q', None),
                    'amplitude_ci': amplitude_ci,
                    'acrophase_ci': acrophase_ci,
                    'mesor_ci': None,  # cosinor1.fit_group doesn't provide MESOR CI
                    'p_amplitude': row.get('p(amplitude)', None),
                    'p_acrophase': row.get('p(acrophase)', None),
                    'p_mesor': None,  # cosinor1.fit_group doesn't test MESOR
                    'q_amplitude': row.get('q(amplitude)', None),
                    'q_acrophase': row.get('q(acrophase)', None),
                    'q_mesor': None,  # cosinor1.fit_group doesn't test MESOR
                    'rss': None,
                    'r2': None,
                    'r2_adj': None,
                    'log_likelihood': None,
                    'aic': None,
                    'bic': None,
                    'me': None,
                    'resid_se': None,
                    'peaks': None,
                    'results_df': df_results,
                    'best_model': 'Yes (min p-value)' if idx == best_idx else 'No'
                }
                results_list.append(result_dict)

            return results_list  # Return list of dicts

        else:
            # Single period - return single dict (no best_model indicator needed)
            result_row = test_results.iloc[0]

            # Parse CI columns
            amp_ci_str = result_row.get('CI(amplitude)', '')
            acro_ci_str = result_row.get('CI(acrophase)', '')

            print(f"[DEBUG] Single period - CI parsing - amp_ci_str: {amp_ci_str} (type: {type(amp_ci_str)})")
            print(f"[DEBUG] Single period - CI parsing - acro_ci_str: {acro_ci_str} (type: {type(acro_ci_str)})")

            try:
                if amp_ci_str:
                    if isinstance(amp_ci_str, (list, tuple)) and len(amp_ci_str) == 2:
                        # Already a list/tuple
                        amplitude_ci = (float(amp_ci_str[0]), float(amp_ci_str[1]))
                    elif isinstance(amp_ci_str, str):
                        # Parse string format
                        amp_ci_parts = amp_ci_str.strip('()[]').split(',')
                        amplitude_ci = (float(amp_ci_parts[0]), float(amp_ci_parts[1]))
                    else:
                        amplitude_ci = None
                else:
                    amplitude_ci = None
            except Exception as e:
                print(f"[DEBUG] Failed to parse amplitude CI: {e}")
                amplitude_ci = None

            try:
                if acro_ci_str:
                    if isinstance(acro_ci_str, (list, tuple)) and len(acro_ci_str) == 2:
                        # Already a list/tuple
                        acrophase_ci = (float(acro_ci_str[0]), float(acro_ci_str[1]))
                    elif isinstance(acro_ci_str, str):
                        # Parse string format
                        acro_ci_parts = acro_ci_str.strip('()[]').split(',')
                        acrophase_ci = (float(acro_ci_parts[0]), float(acro_ci_parts[1]))
                    else:
                        acrophase_ci = None
                else:
                    acrophase_ci = None
            except Exception as e:
                print(f"[DEBUG] Failed to parse acrophase CI: {e}")
                acrophase_ci = None

            print(f"[DEBUG] Single period - Parsed amplitude_ci: {amplitude_ci}")
            print(f"[DEBUG] Single period - Parsed acrophase_ci: {acrophase_ci}")

            return {
                'test_name': test_name,
                'period': result_row['period'],
                'n_components': 1,
                'amplitude': result_row['amplitude'],
                'acrophase': result_row['acrophase'],
                'acrophase_hours': result_row.get('acrophase[h]', None),
                'mesor': mesor,
                'p_value': result_row['p'],
                'q_value': result_row.get('q', None),
                'amplitude_ci': amplitude_ci,
                'acrophase_ci': acrophase_ci,
                'mesor_ci': None,  # cosinor1.fit_group doesn't provide MESOR CI
                'p_amplitude': result_row.get('p(amplitude)', None),
                'p_acrophase': result_row.get('p(acrophase)', None),
                'p_mesor': None,  # cosinor1.fit_group doesn't test MESOR
                'q_amplitude': result_row.get('q(amplitude)', None),
                'q_acrophase': result_row.get('q(acrophase)', None),
                'q_mesor': None,  # cosinor1.fit_group doesn't test MESOR
                'rss': None,
                'r2': None,
                'r2_adj': None,
                'log_likelihood': None,
                'aic': None,
                'bic': None,
                'me': None,
                'resid_se': None,
                'peaks': None,
                'results_df': df_results,
                'best_model': None  # Not applicable for single period
            }

    def _cosinor_independent_multi(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period: Union[float, List[float]],
        n_components: List[int],
        criterium: Criterium,
        analysis_method: AnalysisMethod,
        bootstrap_size: int,
        save_folder: Optional[str],
        save_cosinorpy_plots: bool = False,
        model_type: ModelType = ModelType.NORMAL
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Multi-component cosinor analysis using cosinor.

        Returns:
            - If only one combination (single period + single component): Dict
            - If multiple combinations: List[Dict] with best_model indicators
        """
        # Map our ModelType enum to CosinorPy's expected string values
        _model_type_map = {
            ModelType.NORMAL: 'lin',
            ModelType.POISSON: 'poisson',
            ModelType.NEGATIVE_BINOMIAL: 'nb',
        }
        cosinorpy_model_type = _model_type_map.get(model_type, 'lin')

        print(f"[DEBUG] Using cosinor.fit_group for multi-component")
        print(f"[DEBUG] save_cosinorpy_plots={save_cosinorpy_plots}")

        # Ensure period is a list
        if isinstance(period, (int, float)):
            period = [period]

        # Disable matplotlib interactive mode to prevent plots from showing
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # =====================================================================
        # COSINORPY WORKFLOW FOR MULTIPLE PERIODS
        # =====================================================================
        # IMPORTANT: CosinorPy is designed to work with either:
        #   - Multiple periods + single n_components (to find best period)
        #   - Single period + multiple n_components (to find best model via F-test)
        #
        # It CANNOT properly handle multiple periods × multiple n_components simultaneously
        # because the F-test in get_best_models() assumes all models have the same period.
        #
        # Therefore, when the user specifies multiple periods, we implement a two-step workflow:
        #   Step 1: Find the best period (using only n_components=[1])
        #   Step 2: With the best period, find the best n_components (using F-test)
        #
        # RESULT: The results table will ONLY show models for the BEST PERIOD.
        #         It will NOT show all combinations of periods × n_components.
        #         This is the correct CosinorPy workflow, not a limitation of CircaScope.
        #
        # If you need to see results for all periods, run separate analyses with single periods.
        # =====================================================================

        # Map criterium to available columns in fit_group results
        # Note: get_best_fits with reverse=False means MAXIMIZE, reverse=True means MINIMIZE
        criterium_map = {
            'RSS': ('RSS', True),              # minimize RSS
            'AIC': ('AIC', True),              # minimize AIC (computed below)
            'BIC': ('BIC', True),              # minimize BIC (computed below)
            'log-likelihood': ('log-likelihood', False)  # maximize log-likelihood
        }
        criterium_column, reverse = criterium_map.get(criterium.value, ('R2_adj', False))
        print(f"[DEBUG] Using criterium column: {criterium_column} (requested: {criterium.value}), reverse={reverse}")

        N_obs = len(df_test)  # Number of observations (needed for BIC)

        def _add_aic_bic(df: pd.DataFrame) -> pd.DataFrame:
            """Compute AIC and BIC from log-likelihood and n_components and add as columns."""
            df = df.copy()
            k = df['n_components'] * 2 + 1  # params: 2 per harmonic + MESOR
            logL = df['log-likelihood']
            df['AIC'] = 2 * k - 2 * logL
            df['BIC'] = k * np.log(N_obs) - 2 * logL
            return df

        # Check if we have multiple periods
        if isinstance(period, list) and len(period) > 1:
            # Determine which n_components to use for finding best period
            # - If user specified single component (e.g., [2]): use that component
            # - If user specified multiple components (e.g., [1,2,3]): use n=1 as baseline
            if len(n_components) == 1:
                period_search_n = n_components  # Use the specified component
                print(f"[DEBUG] Multiple periods detected, using CosinorPy workflow:")
                print(f"[DEBUG]   Step 1: Find best period (with n_components={n_components})")
            else:
                period_search_n = [1]  # Use baseline for multiple components case
                print(f"[DEBUG] Multiple periods detected, using CosinorPy workflow:")
                print(f"[DEBUG]   Step 1: Find best period (with n_components=[1] as baseline)")

            # Step 1a: Fit all periods to find best period
            df_period_results = cosinor.fit_group(
                df_test,
                n_components=period_search_n,
                period=period,
                plot=save_cosinorpy_plots,
                model_type=cosinorpy_model_type
            )
            df_period_results = _add_aic_bic(df_period_results)
            print(f"[DEBUG]   Fitted {len(df_period_results)} periods with n_components={period_search_n}")

            # Step 1b: Select best period based on criterium
            df_best_period = cosinor.get_best_fits(
                df_period_results,
                n_components=period_search_n,
                criterium=criterium_column,
                reverse=reverse
            )
            best_period = df_best_period[df_best_period['test'] == test_name].iloc[0]['period']
            # Convert to float to avoid numpy.float64 (which is not iterable)
            best_period = float(best_period)
            print(f"[DEBUG]   Best period: {best_period} (based on {criterium_column})")

            # Step 2: Now fit all n_components with the best period
            if len(n_components) == 1:
                # Already fitted in Step 1 with the specified component - just filter for best period
                print(f"[DEBUG]   Step 2: Using Step 1 results (already fitted with n_components={n_components})")
                df_results = df_period_results[df_period_results['period'] == best_period]
            else:
                # Need to fit multiple components with best period
                print(f"[DEBUG]   Step 2: Find best n_components (with period={best_period})")
                df_results = cosinor.fit_group(
                    df_test,
                    n_components=n_components,
                    period=[best_period],  # Must be a list for fit_group
                    plot=save_cosinorpy_plots,
                    model_type=cosinorpy_model_type
                )
                df_results = _add_aic_bic(df_results)
                print(f"[DEBUG]   Fitted {len(df_results)} n_components with best period")

        else:
            # Single period: fit all n_components directly
            single_period = period[0] if isinstance(period, list) else period
            print(f"[DEBUG] Single period ({single_period}), fitting all n_components")
            df_results = cosinor.fit_group(
                df_test,
                n_components=n_components,
                period=[single_period],  # Wrap in list - fit_group expects a list
                plot=save_cosinorpy_plots,
                model_type=cosinorpy_model_type
            )
            df_results = _add_aic_bic(df_results)
            print(f"[DEBUG] fit_group returned {len(df_results)} rows")

        print(f"[DEBUG] Available columns: {df_results.columns.tolist()}")

        # Optimization: Only call get_best_fits/get_best_models when necessary
        # get_best_fits: Only needed when we have multiple periods
        # get_best_models: Only needed when we have multiple n_components
        # analyse_best_models: Only needed when we call get_best_models

        if len(n_components) > 1:
            # Multiple components: need F-test to select best model
            print(f"[DEBUG] Multiple n_components, using F-test to select best model")

            # Step 3: Get best fits (only if multiple periods - otherwise skip)
            if isinstance(period, list) and len(period) > 1:
                # Multiple periods: get best period for each n_component
                df_best_fits = cosinor.get_best_fits(
                    df_results,
                    n_components=n_components,
                    criterium=criterium_column,
                    reverse=reverse
                )
                print(f"[DEBUG] get_best_fits returned {len(df_best_fits)} rows")
            else:
                # Single period: all results are already "best fits"
                df_best_fits = df_results
                print(f"[DEBUG] Single period, skipping get_best_fits")

            # Step 4: Get best model (overall best n_components using F-test)
            df_best_models = cosinor.get_best_models(
                df_test,
                df_best_fits,  # Use df_best_fits instead of df_results
                n_components=n_components
            )
            print(f"[DEBUG] get_best_models returned {len(df_best_models)} rows")
            print(f"[DEBUG] Best models columns: {df_best_models.columns.tolist()}")

            # Step 5: Extended analysis (only for best model)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='skopt')
                df_extended = cosinor.analyse_best_models(
                    df_test,
                    df_best_models,
                    analysis=analysis_method.value,
                    bootstrap_size=bootstrap_size
                )
            print(f"[DEBUG] analyse_best_models returned {len(df_extended)} rows")
        else:
            # Single component (but N>1, not N=1 which uses cosinor1)
            # F-test not needed, but analyse_best_models IS useful for CIs and p-values per parameter
            print(f"[DEBUG] Single n_component (N={n_components[0]}), skipping F-test but calling analyse_best_models for CIs")
            df_best_fits = df_results  # All results are "best fits" trivially
            df_best_models = df_results  # The single component is the "best model" trivially

            # Call analyse_best_models to get CIs and p-values per parameter
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='skopt')
                df_extended = cosinor.analyse_best_models(
                    df_test,
                    df_best_models,
                    analysis=analysis_method.value,
                    bootstrap_size=bootstrap_size
                )
            print(f"[DEBUG] analyse_best_models returned {len(df_extended)} rows")
            print(f"[DEBUG] analyse_best_models columns: {df_extended.columns.tolist()}")

        # Step 5: Plot (only if checkbox is checked)
        if save_cosinorpy_plots and save_folder:
            print(f"[DEBUG] Generating CosinorPy plots to {save_folder}")
            cosinor.plot_df_models(df_test, df_best_models, folder=save_folder)
            print(f"[DEBUG] CosinorPy plots saved to {save_folder}")

        # Filter results for this test
        test_results = df_results[df_results['test'] == test_name]

        # If only one combination, return single dict (no best_model indicator needed)
        if len(test_results) == 1:
            result_row = test_results.iloc[0]
            original_result_row = result_row.copy()

            # Get extended analysis if available (only for best model with multiple components)
            if df_extended is not None:
                extended_row = df_extended[df_extended['test'] == test_name]
                if len(extended_row) > 0:
                    print(f"[DEBUG] Found extended row for {test_name}")
                    # Combine extended row (CIs, p-values) with original row (fit metrics)
                    result_row = extended_row.iloc[0].copy()
                    # Preserve fit quality metrics from original row
                    fit_metrics = ['RSS', 'R2', 'R2_adj', 'log-likelihood', 'AIC', 'BIC',
                                  'peaks', 'heights', 'troughs', 'heights2', 'ME', 'resid_SE']
                    for metric in fit_metrics:
                        if metric in original_result_row.index and metric not in result_row.index:
                            result_row[metric] = original_result_row[metric]
                            print(f"[DEBUG] Preserved {metric} = {original_result_row[metric]} from original row")
                else:
                    print(f"[DEBUG] No extended row found for {test_name}")

            print(f"[DEBUG] result_row columns: {result_row.index.tolist()}")

            # Parse CI columns
            amplitude_ci = None
            acrophase_ci = None
            mesor_ci = None

            # Helper to parse CI values (can be list/tuple or string)
            def parse_ci(ci_val):
                if ci_val is None:
                    return None
                # Check for list/tuple first (before pd.isna which doesn't work on arrays)
                if isinstance(ci_val, (list, tuple)) and len(ci_val) == 2:
                    return (float(ci_val[0]), float(ci_val[1]))
                # For scalars, check if NaN
                try:
                    if pd.isna(ci_val):
                        return None
                except (ValueError, TypeError):
                    pass  # Not a scalar, continue
                # Try parsing as string
                if isinstance(ci_val, str):
                    try:
                        parts = ci_val.strip('()[]').split(',')
                        return (float(parts[0]), float(parts[1]))
                    except:
                        return None
                return None

            # Try CI(parameter) format (from analyse_best_models)
            if 'CI(amplitude)' in result_row:
                amp_ci_val = result_row.get('CI(amplitude)')
                print(f"[DEBUG] CI(amplitude): {amp_ci_val} (type: {type(amp_ci_val)})")
                amplitude_ci = parse_ci(amp_ci_val)
            else:
                print(f"[DEBUG] CI(amplitude) not found in result_row")

            if 'CI(acrophase)' in result_row:
                acro_ci_val = result_row.get('CI(acrophase)')
                print(f"[DEBUG] CI(acrophase): {acro_ci_val} (type: {type(acro_ci_val)})")
                acrophase_ci = parse_ci(acro_ci_val)
            else:
                print(f"[DEBUG] CI(acrophase) not found in result_row")

            if 'CI(mesor)' in result_row:
                mesor_ci_val = result_row.get('CI(mesor)')
                print(f"[DEBUG] CI(mesor): {mesor_ci_val} (type: {type(mesor_ci_val)})")
                mesor_ci = parse_ci(mesor_ci_val)
            else:
                print(f"[DEBUG] CI(mesor) not found in result_row")

            print(f"[DEBUG] Parsed CIs - amplitude_ci: {amplitude_ci}, acrophase_ci: {acrophase_ci}, mesor_ci: {mesor_ci}")

            # Helper function
            def safe_get(key, default=None):
                try:
                    if key in result_row:
                        val = result_row[key]
                        # Check for NaN (but skip if it's an array/list)
                        if isinstance(val, (list, np.ndarray)):
                            # Arrays/lists are valid values, return as-is
                            return val
                        if pd.isna(val):
                            return default
                        return val
                    return default
                except:
                    return default

            return {
                'test_name': test_name,
                'period': result_row['period'],
                'n_components': int(result_row['n_components']),
                # Model parameters
                'amplitude': safe_get('amplitude'),
                'acrophase': safe_get('acrophase'),
                'acrophase_hours': safe_get('acrophase[h]') if safe_get('acrophase[h]') is not None
                    else (-result_row['period'] * safe_get('acrophase') / (2 * np.pi)
                          if safe_get('acrophase') is not None else None),
                'mesor': safe_get('MESOR', safe_get('mesor')),
                # Basic statistics
                'p_value': safe_get('p'),
                'q_value': safe_get('q'),
                'p_reject': safe_get('p_reject'),
                'q_reject': safe_get('q_reject'),
                # Fit quality metrics
                'rss': safe_get('RSS'),
                'r_squared': safe_get('R2'),
                'r_squared_adj': safe_get('R2_adj'),
                'log_likelihood': safe_get('log-likelihood'),
                'aic': safe_get('AIC'),
                'bic': safe_get('BIC'),
                'me': safe_get('ME'),
                'resid_se': safe_get('resid_SE'),
                # Confidence intervals
                'amplitude_ci': amplitude_ci,
                'acrophase_ci': acrophase_ci,
                'mesor_ci': mesor_ci,
                # p-values for parameters
                'p_amplitude': safe_get('p(amplitude)'),
                'p_acrophase': safe_get('p(acrophase)'),
                'p_mesor': safe_get('p(mesor)'),
                # q-values for parameters
                'q_amplitude': safe_get('q(amplitude)'),
                'q_acrophase': safe_get('q(acrophase)'),
                'q_mesor': safe_get('q(mesor)'),
                # Other
                'peak_times': safe_get('peaks'),
                'trough_times': safe_get('troughs'),
                # Internal
                'results_df': df_extended,
                'best_models_df': df_best_models,
                'best_model': None
            }
        else:
            # Multiple combinations - return all with best_model indicators
            print(f"[DEBUG] Multiple combinations ({len(test_results)}), returning all results")

            # Create sets for quick lookup
            # Note: best_fits_set should only be populated when we actually called get_best_fits()
            # (i.e., when there are multiple periods). Otherwise it should be empty.
            best_fits_set = set()  # (period, n_components) tuples
            if isinstance(period, list) and len(period) > 1:
                # Only populate best_fits_set if we had multiple periods
                for _, row in df_best_fits[df_best_fits['test'] == test_name].iterrows():
                    best_fits_set.add((row['period'], int(row['n_components'])))

            best_model_info = None  # (period, n_components) tuple
            best_model_row = df_best_models[df_best_models['test'] == test_name]
            if len(best_model_row) > 0:
                best_model_info = (best_model_row.iloc[0]['period'], int(best_model_row.iloc[0]['n_components']))

            print(f"[DEBUG] Best fits: {best_fits_set}")
            print(f"[DEBUG] Best model: {best_model_info}")

            # Create result dict for each combination
            results_list = []
            for idx, row in test_results.iterrows():
                period_val = row['period']
                n_comp = int(row['n_components'])
                combo = (period_val, n_comp)

                # Determine best_model indicator
                is_best_model = (combo == best_model_info)
                is_best_fit = (combo in best_fits_set)

                print(f"[DEBUG] Processing row: period={period_val}, n_comp={n_comp}, is_best_model={is_best_model}")

                # Store original row for preserving fit metrics
                original_row = row.copy()

                if is_best_model:
                    # Use extended analysis data for best model (if available)
                    if df_extended is not None:
                        extended_row = df_extended[(df_extended['test'] == test_name) &
                                                  (df_extended['period'] == period_val) &
                                                  (df_extended['n_components'] == n_comp)]
                        if len(extended_row) > 0:
                            print(f"[DEBUG] Using extended row for best model")
                            print(f"[DEBUG] Extended row columns: {extended_row.iloc[0].index.tolist()}")
                            # Combine extended row (CIs, p-values) with original row (fit metrics)
                            # Start with extended row
                            row = extended_row.iloc[0].copy()
                            # Preserve fit quality metrics from original row
                            fit_metrics = ['RSS', 'R2', 'R2_adj', 'log-likelihood', 'AIC', 'BIC',
                                          'peaks', 'heights', 'troughs', 'heights2', 'ME', 'resid_SE']
                            for metric in fit_metrics:
                                if metric in original_row.index and metric not in row.index:
                                    row[metric] = original_row[metric]
                                    print(f"[DEBUG] Preserved {metric} = {original_row[metric]} from original row")
                        else:
                            print(f"[DEBUG] No extended row found for best model")
                    else:
                        print(f"[DEBUG] df_extended is None")

                    best_model_value = f'Best model (min p-value)'
                elif is_best_fit:
                    # Use the actual criterium column name (mapped from user's choice)
                    if criterium_column == 'RSS':
                        best_model_value = f'Best fit (min RSS)'
                    elif criterium_column == 'AIC':
                        best_model_value = f'Best fit (min AIC)'
                    elif criterium_column == 'BIC':
                        best_model_value = f'Best fit (min BIC)'
                    elif criterium_column == 'log-likelihood':
                        best_model_value = f'Best fit (max log-likelihood)'
                    else:
                        best_model_value = f'Best fit ({criterium_column})'
                else:
                    best_model_value = 'No'
    
                # Helper to parse CI values (can be list/tuple or string)
                def parse_ci(ci_val):
                    if ci_val is None:
                        return None
                    # Check for list/tuple first (before pd.isna which doesn't work on arrays)
                    if isinstance(ci_val, (list, tuple)) and len(ci_val) == 2:
                        return (float(ci_val[0]), float(ci_val[1]))
                    # For scalars, check if NaN
                    try:
                        if pd.isna(ci_val):
                            return None
                    except (ValueError, TypeError):
                        pass
                    # Try parsing as string
                    if isinstance(ci_val, str):
                        try:
                            parts = ci_val.strip('()[]').split(',')
                            return (float(parts[0]), float(parts[1]))
                        except:
                            return None
                    return None

                # Parse CI columns from extended analysis (only available for best model)
                amplitude_ci = None
                acrophase_ci = None
                mesor_ci = None

                # Try CI(parameter) format (from analyse_best_models)
                if 'CI(amplitude)' in row:
                    amplitude_ci = parse_ci(row.get('CI(amplitude)'))
                # Fallback to LB/UB format
                elif 'LB(amplitude)' in row and 'UB(amplitude)' in row:
                    lb = row.get('LB(amplitude)')
                    ub = row.get('UB(amplitude)')
                    if lb is not None and ub is not None:
                        amplitude_ci = (lb, ub)

                if 'CI(acrophase)' in row:
                    acrophase_ci = parse_ci(row.get('CI(acrophase)'))
                elif 'LB(acrophase)' in row and 'UB(acrophase)' in row:
                    lb = row.get('LB(acrophase)')
                    ub = row.get('UB(acrophase)')
                    if lb is not None and ub is not None:
                        acrophase_ci = (lb, ub)

                if 'CI(mesor)' in row:
                    mesor_ci = parse_ci(row.get('CI(mesor)'))
    
                # Helper function to safely get value from pandas Series
                def safe_get(key, default=None):
                    try:
                        if key in row:
                            val = row[key]
                            # Check for NaN (but skip if it's an array/list)
                            if isinstance(val, (list, np.ndarray)):
                                # Arrays/lists are valid values, return as-is
                                return val
                            if pd.isna(val):
                                return default
                            return val
                        return default
                    except:
                        return default
    
                result_dict = {
                    'test_name': test_name,
                    'period': period_val,
                    'n_components': n_comp,
                    # Model parameters
                    'amplitude': safe_get('amplitude'),
                    'acrophase': safe_get('acrophase'),
                    'acrophase_hours': safe_get('acrophase[h]') if safe_get('acrophase[h]') is not None
                        else (-period_val * safe_get('acrophase') / (2 * np.pi)
                              if safe_get('acrophase') is not None else None),
                    'mesor': safe_get('MESOR', safe_get('mesor')),
                    # Basic statistics (from fit_group)
                    'p_value': safe_get('p'),
                    'q_value': safe_get('q'),
                    'p_reject': safe_get('p_reject'),
                    'q_reject': safe_get('q_reject'),
                    # Fit quality metrics
                    'rss': safe_get('RSS'),
                    'r_squared': safe_get('R2'),
                    'r_squared_adj': safe_get('R2_adj'),
                    'log_likelihood': safe_get('log-likelihood'),
                    'aic': safe_get('AIC'),
                    'bic': safe_get('BIC'),
                    'me': safe_get('ME'),
                    'resid_se': safe_get('resid_SE'),
                    # Confidence intervals (from analyse_best_models - only for best model)
                    'amplitude_ci': amplitude_ci,
                    'acrophase_ci': acrophase_ci,
                    'mesor_ci': mesor_ci,
                    # p-values for parameters (from analyse_best_models - only for best model)
                    'p_amplitude': safe_get('p(amplitude)'),
                    'p_acrophase': safe_get('p(acrophase)'),
                    'p_mesor': safe_get('p(mesor)'),
                    # q-values for parameters (from analyse_best_models - only for best model)
                    'q_amplitude': safe_get('q(amplitude)'),
                    'q_acrophase': safe_get('q(acrophase)'),
                    'q_mesor': safe_get('q(mesor)'),
                    # Other
                    'peak_times': safe_get('peaks'),
                    'trough_times': safe_get('troughs'),
                    # Internal
                    'results_df': df_results,
                    'best_models_df': df_best_models,
                    'best_model': best_model_value
                }
                print(f"[DEBUG] result_dict peak_times = {result_dict.get('peak_times')}")
                print(f"[DEBUG] result_dict trough_times = {result_dict.get('trough_times')}")
                results_list.append(result_dict)

            return results_list

    # ========================================================================
    # METHOD 3: COSINOR ANALYSIS (DEPENDENT/POPULATION DATA)
    # ========================================================================

    def cosinor_dependent(
        self,
        variable: str,
        condition: str,
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        model_type: ModelType = ModelType.NORMAL,
        criterium: Criterium = Criterium.RSS,
        params_ci_analysis: str = "sampling",
        bootstrap_size: int = 1000,
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Cosinor analysis for dependent/population data (repeated measures).

        Workflow:
        1. If n_components=[1]: Use cosinor1.population_fit_group()
        2. If n_components=[1,2,3]: Use cosinor.population_fit_group() then get_best_models_population()
        3. Extended analysis: cosinor.analyse_best_models_population()
        4. Plot: cosinor.plot_df_models_population()

        Args:
            variable: Variable name
            condition: Condition name
            period: Single value or list
            n_components: List of components to test
            criterium: RSS, AIC, BIC, or LOG_LIKELIHOOD
            params_ci_analysis: 'sampling' or 'bootstrap' for population CI
            save_folder: Optional folder for plots

        Returns:
            Dict with analysis results including ME and resid_SE
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG cosinor_dependent] variable={variable}, condition={condition}")
        print(f"[DEBUG cosinor_dependent] period={period}, n_components={n_components}")

        # Filter data
        # For dependent data, test names are "{condition}_rep1", "{condition}_rep2", etc.
        # NOT "{variable}_{condition}_rep1" - this is critical for CosinorPy
        test_name = condition  # Just the condition name for dependent data
        print(f"[DEBUG cosinor_dependent] Looking for test starting with: {test_name}")
        print(f"[DEBUG cosinor_dependent] Available tests in data: {self._data['test'].unique().tolist()}")
        df_test = self._data[self._data['test'].str.startswith(f"{test_name}_rep")].copy()

        if len(df_test) == 0:
            raise ValueError(f"No data found for test: {test_name}_rep*")

        print(f"[DEBUG cosinor_dependent] Filtered {len(df_test)} rows")

        try:
            # Single component (only Normal model uses cosinor1; Poisson/NB need cosinor path)
            if len(n_components) == 1 and n_components[0] == 1 and model_type == ModelType.NORMAL:
                return self._cosinor_dependent_single(
                    df_test, test_name, period, save_folder, save_cosinorpy_plots
                )

            # Multi-component (also handles non-normal model types for single component)
            else:
                return self._cosinor_dependent_multi(
                    df_test, test_name, period, n_components,
                    criterium, params_ci_analysis, bootstrap_size, save_folder, save_cosinorpy_plots,
                    model_type=model_type
                )

        except Exception as e:
            print(f"[DEBUG cosinor_dependent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _cosinor_dependent_single(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period: Union[float, List[float]],
        save_folder: Optional[str],
        save_cosinorpy_plots: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Single-component population cosinor.

        Returns:
            - If single period: Dict
            - If multiple periods: List[Dict] with best_model indicators
        """
        print(f"[DEBUG] Using cosinor1.population_fit_group for single component")
        print(f"[DEBUG] save_cosinorpy_plots={save_cosinorpy_plots}")

        # Ensure period is a list for iteration, but cosinor1.population_fit_group needs scalar values
        if isinstance(period, (int, float)):
            period_list = [period]
        else:
            period_list = period

        # Disable matplotlib interactive mode to prevent plots from showing
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Call cosinor1.population_fit_group for each period
        # Note: population_fit_group expects a single period value, not a list
        results_dfs = []
        for p in period_list:
            if save_cosinorpy_plots and save_folder:
                print(f"[DEBUG] Generating CosinorPy plots to {save_folder}")
                df_result = cosinor1.population_fit_group(df_test, period=p, save_folder=save_folder, plot_on=True)
                print(f"[DEBUG] CosinorPy plots saved to {save_folder}")
            else:
                # No plots - use plot_on=False to prevent matplotlib warnings
                df_result = cosinor1.population_fit_group(df_test, period=p, plot_on=False)

            # Add period column if it doesn't exist
            if 'period' not in df_result.columns:
                df_result['period'] = p

            results_dfs.append(df_result)

        # Concatenate all results
        if len(results_dfs) > 1:
            df_results = pd.concat(results_dfs, ignore_index=True)
        else:
            df_results = results_dfs[0]

        print(f"[DEBUG] population_fit_group returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        # Extract result
        # The test name in results doesn't include "_repX" suffix
        base_test_name = test_name.rsplit('_rep', 1)[0] if '_rep' in test_name else test_name
        test_results = df_results[df_results['test'] == base_test_name]

        # If multiple periods were tested, return ALL results with best_model indicator
        if len(test_results) > 1:
            print(f"[DEBUG] Multiple periods tested ({len(test_results)}), returning all results")
            # Find best model (lowest p-value)
            best_idx = test_results['p'].idxmin()

            # Create a result dict for each period
            results_list = []
            for idx, row in test_results.iterrows():
                # Parse CI columns (format: "[lower, upper]")
                amp_ci = self._parse_ci_str(row.get('CI(amplitude)'))
                acro_ci = self._parse_ci_str(row.get('CI(acrophase)'))
                mesor_ci = self._parse_ci_str(row.get('CI(mesor)'))

                result_dict = {
                    'test_name': test_name,
                    'period': row.get('period', period_list[0]),
                    'n_components': 1,
                    'amplitude': row.get('amplitude'),
                    'acrophase': row.get('acrophase'),
                    'acrophase_hours': row.get('acrophase[h]') if row.get('acrophase[h]') is not None
                        else (-row.get('period', period_list[0]) * row.get('acrophase') / (2 * np.pi)
                              if row.get('acrophase') is not None else None),
                    'mesor': row.get('mesor'),
                    'p_value': row.get('p'),
                    'q_value': row.get('q'),
                    'amplitude_ci': amp_ci,
                    'acrophase_ci': acro_ci,
                    'mesor_ci': mesor_ci,
                    'p_amplitude': row.get('p(amplitude)'),
                    'p_acrophase': row.get('p(acrophase)'),
                    'p_mesor': row.get('p(mesor)'),
                    'q_amplitude': row.get('q(amplitude)'),
                    'q_acrophase': row.get('q(acrophase)'),
                    'q_mesor': row.get('q(mesor)'),
                    'results_df': df_results,
                    'best_model': 'Yes (min p-value)' if idx == best_idx else 'No'
                }
                results_list.append(result_dict)

            return results_list

        # Single period - return single dict (no best_model indicator needed)
        else:
            result_row = test_results.iloc[0]

            # Parse CI columns (format: "[lower, upper]")
            amp_ci = self._parse_ci_str(result_row.get('CI(amplitude)'))
            acro_ci = self._parse_ci_str(result_row.get('CI(acrophase)'))
            mesor_ci = self._parse_ci_str(result_row.get('CI(mesor)'))

            result_dict = {
                'test_name': test_name,
                'period': result_row.get('period', period_list[0]),
                'n_components': 1,
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'acrophase_hours': result_row.get('acrophase[h]') if result_row.get('acrophase[h]') is not None
                    else (-result_row.get('period', period_list[0]) * result_row.get('acrophase') / (2 * np.pi)
                          if result_row.get('acrophase') is not None else None),
                'mesor': result_row.get('mesor'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'amplitude_ci': amp_ci,
                'acrophase_ci': acro_ci,
                'mesor_ci': mesor_ci,
                'p_amplitude': result_row.get('p(amplitude)'),
                'p_acrophase': result_row.get('p(acrophase)'),
                'p_mesor': result_row.get('p(mesor)'),
                'q_amplitude': result_row.get('q(amplitude)'),
                'q_acrophase': result_row.get('q(acrophase)'),
                'q_mesor': result_row.get('q(mesor)'),
                'results_df': df_results,
                'best_model': None  # Not applicable for single period
            }

            print(f"[DEBUG] result_dict q_value: {result_dict.get('q_value')}")
            print(f"[DEBUG] result_dict mesor_ci: {result_dict.get('mesor_ci')}")
            print(f"[DEBUG] result_dict p_amplitude: {result_dict.get('p_amplitude')}")
            print(f"[DEBUG] result_dict q_amplitude: {result_dict.get('q_amplitude')}")

            return result_dict

    def _parse_ci_str(self, ci_val) -> Optional[Tuple[float, float]]:
        """Parse CI from CosinorPy output (can be string '[lower, upper]' or numpy array)."""
        if ci_val is None:
            return None

        # Check if it's a numpy array or list
        if isinstance(ci_val, (list, np.ndarray)):
            try:
                if len(ci_val) == 2:
                    return (float(ci_val[0]), float(ci_val[1]))
            except:
                pass
            return None

        # Check if it's NaN (scalar)
        try:
            if pd.isna(ci_val):
                return None
        except:
            pass

        # Handle string format
        if isinstance(ci_val, str):
            try:
                # Remove brackets and split
                ci_val = ci_val.strip('[]')
                parts = ci_val.split(',')
                if len(parts) == 2:
                    return (float(parts[0].strip()), float(parts[1].strip()))
            except:
                pass

        return None

    def _cosinor_dependent_multi(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period: Union[float, List[float]],
        n_components: List[int],
        criterium: Criterium,
        params_ci_analysis: str,
        bootstrap_size: int,
        save_folder: Optional[str],
        save_cosinorpy_plots: bool = False,
        model_type: ModelType = ModelType.NORMAL
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Multi-component population cosinor.

        Returns:
            - If only one combination (single component): Dict
            - If multiple combinations: List[Dict] with best_model indicators
        """
        # Map ModelType enum to CosinorPy expected strings
        _model_type_map = {
            ModelType.NORMAL: 'lin',
            ModelType.POISSON: 'poisson',
            ModelType.NEGATIVE_BINOMIAL: 'nb',
        }
        cosinorpy_model_type = _model_type_map.get(model_type, 'lin')

        # Map criterium to column name and sort direction for get_best_models_population
        # NOTE: population_fit_group does NOT return log-likelihood, so AIC/BIC/log-likelihood
        # are not valid criteria for dependent data. We fall back to RSS with a warning.
        criterium_map = {
            'RSS': ('RSS', True),
            'AIC': ('AIC', True),
            'BIC': ('BIC', True),
            'log-likelihood': ('log-likelihood', False),
        }
        criterium_column, reverse = criterium_map.get(criterium.value, ('RSS', True))

        N_obs = len(df_test)

        def _add_aic_bic(df: pd.DataFrame) -> pd.DataFrame:
            """Compute AIC and BIC from log-likelihood when available."""
            df = df.copy()
            if 'log-likelihood' in df.columns:
                k = df['n_components'] * 2 + 1
                logL = df['log-likelihood']
                df['AIC'] = 2 * k - 2 * logL
                df['BIC'] = k * np.log(N_obs) - 2 * logL
            return df

        print(f"[DEBUG] Using cosinor.population_fit_group for multi-component")
        print(f"[DEBUG] save_cosinorpy_plots={save_cosinorpy_plots}")

        # Disable matplotlib interactive mode to prevent plots from showing
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        # Convert period to list if it's a single value
        period_list = [period] if not isinstance(period, list) else period

        # Step 1: Fit all combinations for each period
        # Note: cosinor.population_fit_group expects period as a list, but we need to iterate
        # over each period separately to get results for all periods
        results_dfs = []
        for p in period_list:
            if save_cosinorpy_plots and save_folder:
                print(f"[DEBUG] Generating CosinorPy plots for period {p} to {save_folder}")
                df_result = cosinor.population_fit_group(
                    df_test,
                    n_components=n_components,
                    period=[p],
                    folder=save_folder,
                    model_type=cosinorpy_model_type
                )
                print(f"[DEBUG] CosinorPy plots saved to {save_folder}")
            else:
                df_result = cosinor.population_fit_group(
                    df_test,
                    n_components=n_components,
                    period=[p],
                    model_type=cosinorpy_model_type
                )
            print(f"[DEBUG] Period {p} returned {len(df_result)} rows with periods: {df_result['period'].unique().tolist()}")
            results_dfs.append(df_result)

        # Concatenate all results
        if len(results_dfs) > 1:
            df_results = pd.concat(results_dfs, ignore_index=True)
        else:
            df_results = results_dfs[0]

        df_results = _add_aic_bic(df_results)
        print(f"[DEBUG] population_fit_group returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        # Guard: population_fit_group does not return log-likelihood, so AIC/BIC/log-likelihood
        # columns may not exist. Fall back to RSS to avoid KeyError in get_best_fits.
        if criterium_column not in df_results.columns:
            print(f"[WARNING] Criterium column '{criterium_column}' not available for dependent data "
                  f"(population_fit_group does not return log-likelihood). Falling back to RSS.")
            criterium_column = 'RSS'
            reverse = True

        # Step 2: Get best models using the user-selected criterium
        df_best_models = cosinor.get_best_models_population(
            df_test,
            df_results,
            n_components=n_components,
            criterium=criterium_column,
            reverse=reverse
        )

        print(f"[DEBUG] get_best_models_population returned {len(df_best_models)} rows")
        print(f"[DEBUG] Best models columns: {df_best_models.columns.tolist()}")

        # Step 3: Extended analysis (only for best model)
        df_extended = cosinor.analyse_best_models_population(
            df_test,
            df_best_models,
            params_CI_analysis=params_ci_analysis,
            bootstrap_size=bootstrap_size
        )

        print(f"[DEBUG] analyse_best_models_population returned {len(df_extended)} rows")

        # Step 4: Plot (only if checkbox is checked)
        if save_cosinorpy_plots and save_folder:
            print(f"[DEBUG] Generating CosinorPy plots to {save_folder}")
            cosinor.plot_df_models_population(df_test, df_best_models, folder=save_folder)
            print(f"[DEBUG] CosinorPy plots saved to {save_folder}")

        # Extract results for this test
        base_test_name = test_name.rsplit('_rep', 1)[0] if '_rep' in test_name else test_name
        test_results = df_results[df_results['test'] == base_test_name]

        # If only one combination, return single dict (no best_model indicator needed)
        if len(test_results) == 1:
            result_row = test_results.iloc[0]
            original_result_row = result_row.copy()

            # Get extended analysis if available (only for best model)
            extended_row = df_extended[df_extended['test'] == base_test_name]
            if len(extended_row) > 0:
                print(f"[DEBUG] Found extended row for {base_test_name}")
                # Combine extended row (CIs, p-values) with original row (fit metrics)
                result_row = extended_row.iloc[0].copy()
                # Preserve fit quality metrics from original row
                fit_metrics = ['RSS', 'R2', 'R2_adj', 'log-likelihood', 'AIC', 'BIC',
                              'peaks', 'heights', 'troughs', 'heights2', 'ME', 'resid_SE']
                for metric in fit_metrics:
                    if metric in original_result_row.index and metric not in result_row.index:
                        result_row[metric] = original_result_row[metric]
                        print(f"[DEBUG] Preserved {metric} = {original_result_row[metric]} from original row")

            # Helper to parse CI values (can be list/tuple or string)
            def parse_ci(ci_val):
                if ci_val is None:
                    return None
                # Check for list/tuple first
                if isinstance(ci_val, (list, tuple, np.ndarray)) and len(ci_val) == 2:
                    return (float(ci_val[0]), float(ci_val[1]))
                # For scalars, check if NaN
                try:
                    if pd.isna(ci_val):
                        return None
                except (ValueError, TypeError):
                    pass
                # Try parsing as string
                if isinstance(ci_val, str):
                    try:
                        parts = ci_val.strip('()[]').split(',')
                        return (float(parts[0]), float(parts[1]))
                    except:
                        return None
                return None

            # Parse CIs
            amplitude_ci = None
            acrophase_ci = None
            mesor_ci = None

            # Try CI(parameter) format first (from analyse_best_models_population)
            if 'CI(amplitude)' in result_row:
                amplitude_ci = parse_ci(result_row.get('CI(amplitude)'))
            # Fallback to LB/UB format
            elif 'LB(amplitude)' in result_row and 'UB(amplitude)' in result_row:
                lb = result_row.get('LB(amplitude)')
                ub = result_row.get('UB(amplitude)')
                if lb is not None and ub is not None:
                    amplitude_ci = (lb, ub)

            if 'CI(acrophase)' in result_row:
                acrophase_ci = parse_ci(result_row.get('CI(acrophase)'))
            elif 'LB(acrophase)' in result_row and 'UB(acrophase)' in result_row:
                lb = result_row.get('LB(acrophase)')
                ub = result_row.get('UB(acrophase)')
                if lb is not None and ub is not None:
                    acrophase_ci = (lb, ub)

            if 'CI(mesor)' in result_row:
                mesor_ci = parse_ci(result_row.get('CI(mesor)'))
            elif 'CI(MESOR)' in result_row:
                mesor_ci = parse_ci(result_row.get('CI(MESOR)'))

            # Helper function to safely get value
            def safe_get(key, default=None):
                try:
                    if key in result_row.index:
                        val = result_row[key]
                        # Check for NaN (but skip if it's an array/list)
                        if isinstance(val, (list, np.ndarray)):
                            return val
                        if pd.isna(val):
                            return default
                        return val
                    return default
                except:
                    return default

            return {
                'test_name': test_name,
                'period': safe_get('period'),
                'n_components': int(result_row['n_components']),
                'amplitude': safe_get('amplitude'),
                'acrophase': safe_get('acrophase'),
                'acrophase_hours': safe_get('acrophase[h]'),
                'mesor': safe_get('MESOR', safe_get('mesor')),
                'p_value': safe_get('p'),
                'q_value': safe_get('q'),
                'p_reject': safe_get('p_reject'),
                'q_reject': safe_get('q_reject'),
                'rss': safe_get('RSS'),
                'aic': safe_get('AIC'),
                'bic': safe_get('BIC'),
                'me': safe_get('ME'),
                'resid_se': safe_get('resid_SE'),
                'amplitude_ci': amplitude_ci,
                'acrophase_ci': acrophase_ci,
                'mesor_ci': mesor_ci,
                'p_amplitude': safe_get('p(amplitude)'),
                'p_acrophase': safe_get('p(acrophase)'),
                'p_mesor': safe_get('p(mesor)'),
                'q_amplitude': safe_get('q(amplitude)'),
                'q_acrophase': safe_get('q(acrophase)'),
                'q_mesor': safe_get('q(mesor)'),
                'results_df': df_extended,
                'best_models_df': df_best_models,
                'best_model': None
            }

        # Multiple combinations - return all with best_model indicators
        print(f"[DEBUG] Multiple combinations ({len(test_results)}), returning all results")

        # Get best model info (n_components AND period)
        best_model_n_comp = None
        best_model_period = None
        best_model_row = df_best_models[df_best_models['test'] == base_test_name]
        if len(best_model_row) > 0:
            best_model_n_comp = int(best_model_row.iloc[0]['n_components'])
            best_model_period = best_model_row.iloc[0]['period']

        print(f"[DEBUG] Best model n_components: {best_model_n_comp}, period: {best_model_period}")

        # Helper to parse CI values (can be list/tuple or string)
        def parse_ci(ci_val):
            if ci_val is None:
                return None
            # Check for list/tuple first
            if isinstance(ci_val, (list, tuple, np.ndarray)) and len(ci_val) == 2:
                return (float(ci_val[0]), float(ci_val[1]))
            # For scalars, check if NaN
            try:
                if pd.isna(ci_val):
                    return None
            except (ValueError, TypeError):
                pass
            # Try parsing as string
            if isinstance(ci_val, str):
                try:
                    parts = ci_val.strip('()[]').split(',')
                    return (float(parts[0]), float(parts[1]))
                except:
                    return None
            return None

        # Helper function to safely get value
        def safe_get(row, key, default=None):
            try:
                if key in row.index:
                    val = row[key]
                    # Check for NaN (but skip if it's an array/list)
                    if isinstance(val, (list, np.ndarray)):
                        return val
                    if pd.isna(val):
                        return default
                    return val
                return default
            except:
                return default

        # Create result dict for each combination
        results_list = []
        for idx, row in test_results.iterrows():
            n_comp = int(row['n_components'])
            period_val = row['period']  # Store original period value
            original_row = row.copy()

            # Determine best_model indicator
            # Note: For population data, we mark only the best model by BOTH n_components AND period
            is_best_model = (n_comp == best_model_n_comp and period_val == best_model_period)

            if is_best_model:
                # Use extended analysis data for best model
                extended_row = df_extended[(df_extended['test'] == base_test_name) &
                                          (df_extended['n_components'] == n_comp)]
                if len(extended_row) > 0:
                    print(f"[DEBUG] Using extended row for best model")
                    # Combine extended row (CIs, p-values) with original row (fit metrics)
                    row = extended_row.iloc[0].copy()
                    # Update period_val from extended row if available
                    period_val = row['period']
                    # Preserve fit quality metrics from original row
                    fit_metrics = ['RSS', 'R2', 'R2_adj', 'log-likelihood', 'AIC', 'BIC',
                                  'peaks', 'heights', 'troughs', 'heights2', 'ME', 'resid_SE']
                    for metric in fit_metrics:
                        if metric in original_row.index and metric not in row.index:
                            row[metric] = original_row[metric]

                best_model_value = 'Best model (min p-value)'
            else:
                best_model_value = 'No'

            print(f"[DEBUG] Creating result for n_components={n_comp}, period={period_val}, best_model={best_model_value}")
            print(f"[DEBUG] p_reject={safe_get(row, 'p_reject')}, q_reject={safe_get(row, 'q_reject')}")

            # Parse CIs
            amplitude_ci = None
            acrophase_ci = None
            mesor_ci = None

            # Try CI(parameter) format first (from analyse_best_models_population)
            if 'CI(amplitude)' in row:
                amplitude_ci = parse_ci(row.get('CI(amplitude)'))
            # Fallback to LB/UB format
            elif 'LB(amplitude)' in row and 'UB(amplitude)' in row:
                lb = safe_get(row, 'LB(amplitude)')
                ub = safe_get(row, 'UB(amplitude)')
                if lb is not None and ub is not None:
                    amplitude_ci = (lb, ub)

            if 'CI(acrophase)' in row:
                acrophase_ci = parse_ci(row.get('CI(acrophase)'))
            elif 'LB(acrophase)' in row and 'UB(acrophase)' in row:
                lb = safe_get(row, 'LB(acrophase)')
                ub = safe_get(row, 'UB(acrophase)')
                if lb is not None and ub is not None:
                    acrophase_ci = (lb, ub)

            if 'CI(mesor)' in row:
                mesor_ci = parse_ci(row.get('CI(mesor)'))
            elif 'CI(MESOR)' in row:
                mesor_ci = parse_ci(row.get('CI(MESOR)'))

            result_dict = {
                'test_name': test_name,
                'period': period_val,
                'n_components': n_comp,
                'amplitude': safe_get(row, 'amplitude'),
                'acrophase': safe_get(row, 'acrophase'),
                'acrophase_hours': safe_get(row, 'acrophase[h]') if safe_get(row, 'acrophase[h]') is not None
                    else (-period_val * safe_get(row, 'acrophase') / (2 * np.pi)
                          if safe_get(row, 'acrophase') is not None else None),
                'mesor': safe_get(row, 'MESOR', safe_get(row, 'mesor')),
                'p_value': safe_get(row, 'p'),
                'q_value': safe_get(row, 'q'),
                'p_reject': safe_get(row, 'p_reject'),
                'q_reject': safe_get(row, 'q_reject'),
                'rss': safe_get(row, 'RSS'),
                'aic': safe_get(row, 'AIC'),
                'bic': safe_get(row, 'BIC'),
                'me': safe_get(row, 'ME'),
                'resid_se': safe_get(row, 'resid_SE'),
                'amplitude_ci': amplitude_ci,
                'acrophase_ci': acrophase_ci,
                'mesor_ci': mesor_ci,
                'p_amplitude': safe_get(row, 'p(amplitude)'),
                'p_acrophase': safe_get(row, 'p(acrophase)'),
                'p_mesor': safe_get(row, 'p(mesor)'),
                'q_amplitude': safe_get(row, 'q(amplitude)'),
                'q_acrophase': safe_get(row, 'q(acrophase)'),
                'q_mesor': safe_get(row, 'q(mesor)'),
                'results_df': df_results,
                'best_models_df': df_best_models,
                'best_model': best_model_value
            }
            results_list.append(result_dict)

        return results_list

    # ========================================================================
    # METHOD 4: COMPARE CONDITIONS (INDEPENDENT DATA)
    # ========================================================================

    def compare_independent(
        self,
        variable: str,
        conditions: List[str],
        period: Union[float, List[float]],
        period1: Optional[float] = None,
        period2: Optional[float] = None,
        n_components: List[int] = [1],
        comparison_type: str = "Pooled Model",
        comparison_method: str = "Independent",
        analysis_method: str = "CI",
        parameters_to_compare: List[str] = None,
        lin_comp: bool = False,
        bootstrap_size: int = 1000,
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Compare conditions for independent data using CosinorPy methods.

        Workflow:
        1. Auto-generate all possible pairs from conditions
        2. If n_components=[1]:
           - Pooled Model: cosinor1.test_cosinor_pairs()
           - Independent Models: cosinor1.test_cosinor_pairs_independent()
        3. If multi-component:
           - Independent method: cosinor.compare_pairs()
           - LimoRhyde method: cosinor.compare_pairs_limo()
           - Iterate over all (period, n_components) combinations

        Args:
            variable: Variable name
            conditions: List of conditions to compare
            period: Period value or list (used for Pooled Model and multi-component)
            period1: Period for condition 1 (only for Independent Models with 2 conditions)
            period2: Period for condition 2 (only for Independent Models with 2 conditions)
            n_components: Components to test
            comparison_type: "Pooled Model" or "Independent Models" (for single component)
            comparison_method: "Independent" or "LimoRhyde" (for multi-component)
            analysis_method: Analysis method string (depends on comparison_method)
            parameters_to_compare: List of parameters to compare (amplitude, acrophase, mesor)
            lin_comp: Include linear component (only for Independent method)
            bootstrap_size: Number of bootstrap iterations
            save_folder: Optional folder for plots
            save_cosinorpy_plots: Whether to save CosinorPy plots

        Returns:
            Dict with comparison results
        """
        if self._data is None:
            raise ValueError("No data loaded")

        # Default parameters_to_compare
        if parameters_to_compare is None:
            parameters_to_compare = ['amplitude', 'acrophase', 'mesor']

        print(f"[DEBUG compare_independent] variable={variable}, conditions={conditions}")
        print(f"[DEBUG compare_independent] period={period}, n_components={n_components}")
        print(f"[DEBUG compare_independent] comparison_type={comparison_type}, comparison_method={comparison_method}")
        print(f"[DEBUG compare_independent] analysis_method={analysis_method}, parameters_to_compare={parameters_to_compare}")

        # Generate all pairs
        condition_pairs = self._generate_all_pairs(conditions)
        test_pairs = self._format_pairs_for_cosinorpy(condition_pairs, variable)

        print(f"[DEBUG compare_independent] Generated {len(test_pairs)} pairs: {test_pairs}")

        try:
            # Single component
            if len(n_components) == 1 and n_components[0] == 1:
                if comparison_type == "Pooled Model":
                    return self._compare_independent_single_pooled(
                        test_pairs, period, save_folder, save_cosinorpy_plots
                    )
                else:  # Independent Models
                    return self._compare_independent_single_independent(
                        test_pairs, period, period1, period2, save_folder
                    )
            # Multi-component
            else:
                return self._compare_independent_multi(
                    test_pairs, period, n_components,
                    comparison_method, analysis_method, parameters_to_compare,
                    lin_comp, bootstrap_size, save_folder, save_cosinorpy_plots
                )

        except Exception as e:
            print(f"[DEBUG compare_independent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _compare_independent_single_pooled(
        self,
        test_pairs: List[Tuple[str, str]],
        period: Union[float, List[float]],
        save_folder: Optional[str],
        save_cosinorpy_plots: bool
    ) -> Dict[str, Any]:
        """Single-component comparison using pooled model (cosinor1.test_cosinor_pairs)."""
        print(f"[DEBUG] Using cosinor1.test_cosinor_pairs (pooled model)")

        # Ensure period is single value
        if isinstance(period, list):
            period = period[0]

        # Prepare folder and plot settings
        # CRITICAL BUG FIX: cosinor1.test_cosinor_pairs ALWAYS calls plot_pair() which calls plt.show()
        # when folder=''. This freezes the GUI. Solution: ALWAYS provide a folder path.
        # If user doesn't want to save plots, use a temporary folder that we delete after.
        import tempfile
        import shutil

        use_temp_folder = False
        if save_cosinorpy_plots and save_folder:
            folder = save_folder
            plot_measurements = True
        else:
            # Create temporary folder to avoid plt.show() calls
            folder = tempfile.mkdtemp(prefix='cosinorpy_temp_')
            plot_measurements = False
            use_temp_folder = True
            print(f"[DEBUG] Using temporary folder to avoid plot display: {folder}")

        try:
            # Call cosinor1.test_cosinor_pairs
            df_results = cosinor1.test_cosinor_pairs(
                self._data,
                pairs=test_pairs,
                period=period,
                folder=folder,
                plot_measurements=plot_measurements,
                legend=True
            )

            print(f"[DEBUG] test_cosinor_pairs returned {len(df_results)} rows")
            print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

            return {
                'comparison_type': 'pooled_model',
                'n_pairs': len(test_pairs),
                'results_df': df_results
            }
        finally:
            # Clean up temporary folder if used
            if use_temp_folder and folder:
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                    print(f"[DEBUG] Cleaned up temporary folder: {folder}")
                except Exception as e:
                    print(f"[DEBUG] Failed to clean up temp folder: {e}")

    def _compare_independent_single_independent(
        self,
        test_pairs: List[Tuple[str, str]],
        period: Union[float, List[float]],
        period1: Optional[float] = None,
        period2: Optional[float] = None,
        save_folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """Single-component comparison using independent models (cosinor1.test_cosinor_pairs_independent)."""
        print(f"[DEBUG] Using cosinor1.test_cosinor_pairs_independent (independent models)")

        # Determine which periods to use
        # If period1 and period2 are provided (from UI), use them
        # Otherwise, use the default period parameter
        if period1 is not None and period2 is not None:
            # User specified different periods for each condition
            print(f"[DEBUG] Using per-condition periods: period1={period1}, period2={period2}")
            final_period = period1
            final_period2 = period2
        else:
            # Use default period (same for both conditions)
            if isinstance(period, list):
                final_period = period[0]
            else:
                final_period = period
            final_period2 = None  # Same period for both conditions
            print(f"[DEBUG] Using same period for both conditions: {final_period}")

        # Call cosinor1.test_cosinor_pairs_independent
        df_results = cosinor1.test_cosinor_pairs_independent(
            self._data,
            pairs=test_pairs,
            period=final_period,
            period2=final_period2
        )

        print(f"[DEBUG] test_cosinor_pairs_independent returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        return {
            'comparison_type': 'independent_models',
            'n_pairs': len(test_pairs),
            'results_df': df_results
        }

    def _compare_independent_multi(
        self,
        test_pairs: List[Tuple[str, str]],
        period: Union[float, List[float]],
        n_components: List[int],
        comparison_method: str,
        analysis_method: str,
        parameters_to_compare: List[str],
        lin_comp: bool,
        bootstrap_size: int,
        save_folder: Optional[str],
        save_cosinorpy_plots: bool
    ) -> Dict[str, Any]:
        """Multi-component comparison using Independent or LimoRhyde method."""
        print(f"[DEBUG] Multi-component comparison: method={comparison_method}, analysis={analysis_method}")

        # Convert period to list for iteration
        if isinstance(period, (int, float)):
            period_list = [period]
        else:
            period_list = period

        # Disable matplotlib interactive mode
        import matplotlib
        matplotlib.use('Agg')

        # Prepare folder for plots
        # IMPORTANT: Only set folder if we actually want to save plots for LimoRhyde
        # For compare_pairs_limo, if folder is empty string it might try to show plots
        if save_cosinorpy_plots and save_folder:
            folder = save_folder
        else:
            folder = None  # Don't pass folder parameter to avoid plot generation

        # Map GUI analysis_method to CosinorPy parameters
        if comparison_method == 'LimoRhyde':
            # LimoRhyde: '', 'CI1', 'Bootstrap1', 'CI2', 'Bootstrap2'
            analysis_map = {
                'None': '',
                'CI1': 'CI1',
                'Bootstrap1': 'bootstrap1',
                'CI2': 'CI2',
                'Bootstrap2': 'bootstrap2'
            }
            analysis_param = analysis_map.get(analysis_method, '')
        else:
            # Independent: 'CI', 'Bootstrap'
            analysis_map = {
                'CI': 'CI',
                'Bootstrap': 'bootstrap'
            }
            analysis_param = analysis_map.get(analysis_method, 'CI')

        # Iterate over all combinations of (period, n_components)
        results_list = []
        for per in period_list:
            for n_comp in n_components:
                print(f"[DEBUG] Processing period={per}, n_components={n_comp}")

                if comparison_method == 'Independent':
                    # Use cosinor.compare_pairs
                    # NOTE: CosinorPy expects period as a list (it iterates internally)
                    # We're already iterating, so pass single period as list
                    df_result = cosinor.compare_pairs(
                        self._data,
                        pairs=test_pairs,
                        n_components=[n_comp],  # Must be list
                        period=[per],  # Must be list, even for single value
                        analysis=analysis_param,
                        parameters_to_analyse=parameters_to_compare,
                        parameters_angular=['acrophase'],
                        lin_comp=lin_comp,
                        bootstrap_size=bootstrap_size
                    )
                else:
                    # Use cosinor.compare_pairs_limo
                    # Only pass folder if it's set (to save plots)
                    limo_kwargs = {
                        'pairs': test_pairs,
                        'n_components': [n_comp],  # Must be list
                        'period': [per],  # Must be list, even for single value
                        'analysis': analysis_param,
                        'parameters_to_analyse': parameters_to_compare,
                        'parameters_angular': ['acrophase'],
                        'bootstrap_size': bootstrap_size
                    }
                    if folder is not None:
                        limo_kwargs['folder'] = folder

                    df_result = cosinor.compare_pairs_limo(self._data, **limo_kwargs)

                print(f"[DEBUG] Got {len(df_result)} rows for period={per}, n_components={n_comp}")
                results_list.append(df_result)

        # Concatenate all results
        if len(results_list) > 1:
            df_results = pd.concat(results_list, ignore_index=True)
        else:
            df_results = results_list[0]

        print(f"[DEBUG] Total comparison results: {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        return {
            'comparison_type': f'multi_{comparison_method.lower()}',
            'n_pairs': len(test_pairs),
            'n_components': n_components,
            'n_periods': len(period_list),
            'results_df': df_results
        }

    # ========================================================================
    # METHOD 5: COMPARE CONDITIONS (DEPENDENT/POPULATION DATA)
    # ========================================================================

    def compare_dependent(
        self,
        variable: str,
        conditions: List[str],
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        analysis_method: str = 'CI',
        parameters_to_analyse: List[str] = None,
        lin_comp: bool = False,
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Compare conditions for dependent/population data.

        Workflow:
        1. Auto-generate all possible pairs from conditions
        2. If n_components=[1]: Use cosinor1.population_test_cosinor_pairs()
        3. If multi-component: Use cosinor.compare_pairs_population()

        IMPORTANT: For dependent data, pairs should be CONDITION NAMES ONLY,
        not including variable prefix. CosinorPy uses test.startswith() to filter.

        Args:
            variable: Variable name
            conditions: List of conditions
            period: Period value(s)
            n_components: Components to test
            analysis_method: 'CI' or 'permutation'
            parameters_to_analyse: Parameters to compare (default: all)
            lin_comp: Include linear component
            save_folder: Optional folder for plots
            save_cosinorpy_plots: Whether to save plots

        Returns:
            Dict with comparison results
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG compare_dependent] variable={variable}, conditions={conditions}")
        print(f"[DEBUG] analysis_method={analysis_method}, lin_comp={lin_comp}")

        # Generate pairs from condition names
        condition_pairs = self._generate_all_pairs(conditions)
        # Pairs must include the variable prefix to match test names in the DataFrame
        # Test names are formatted as "{variable}-{condition}_rep{i}" (e.g., "circadian_noisy-control_rep1")
        # CosinorPy filters with df.test.str.startswith(f'{pair[0]}_rep'), so pair[0] must be "{variable}-{condition}"
        test_pairs = [(f"{variable}-{c1}", f"{variable}-{c2}") for c1, c2 in condition_pairs]

        print(f"[DEBUG compare_dependent] Generated {len(test_pairs)} pairs: {test_pairs}")

        try:
            # Single component
            if len(n_components) == 1 and n_components[0] == 1:
                return self._compare_dependent_single(
                    test_pairs, period, save_folder, save_cosinorpy_plots
                )
            # Multi-component
            else:
                return self._compare_dependent_multi(
                    test_pairs, period, n_components, analysis_method,
                    parameters_to_analyse, lin_comp, save_folder
                )

        except Exception as e:
            print(f"[DEBUG compare_dependent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _compare_dependent_single(
        self,
        test_pairs: List[Tuple[str, str]],
        period: Union[float, List[float]],
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Dict[str, Any]:
        """
        Single-component population comparison using pooled model.

        CRITICAL: cosinor1.population_test_cosinor_pairs ALWAYS calls plt.show()
        when save_folder=''. We must ALWAYS provide a folder to avoid GUI freeze.
        """
        print(f"[DEBUG _compare_dependent_single] Using cosinor1.population_test_cosinor_pairs")
        print(f"[DEBUG] save_cosinorpy_plots={save_cosinorpy_plots}, save_folder={save_folder}")

        # Convert period to list if needed
        period_list = period if isinstance(period, list) else [period]
        print(f"[DEBUG] Will iterate over {len(period_list)} period(s): {period_list}")

        # CRITICAL BUG FIX: cosinor1.population_test_cosinor_pairs ALWAYS calls plot_pair()
        # which calls plt.show() when save_folder=''. Solution: ALWAYS provide a folder.
        import tempfile
        import shutil

        use_temp_folder = False
        if save_cosinorpy_plots and save_folder:
            folder = save_folder
            plot_on = True
            print(f"[DEBUG] Will save plots to: {folder}")
        else:
            # Create temporary folder to avoid plt.show() calls
            folder = tempfile.mkdtemp(prefix='cosinorpy_temp_')
            plot_on = False
            use_temp_folder = True
            print(f"[DEBUG] Using temporary folder: {folder}")

        try:
            results_list = []
            import os

            # Iterate over each period
            for per in period_list:
                print(f"[DEBUG] Processing period={per}")

                # For multiple periods, create a subfolder to avoid overwriting plots
                if len(period_list) > 1 and save_cosinorpy_plots and save_folder:
                    # Create period-specific subfolder
                    period_folder = os.path.join(folder, f"period_{per}h")
                    os.makedirs(period_folder, exist_ok=True)
                    current_folder = period_folder
                    print(f"[DEBUG] Using period-specific folder: {current_folder}")
                else:
                    current_folder = folder

                print(f"[DEBUG] Calling population_test_cosinor_pairs with:")
                print(f"  - pairs={test_pairs}")
                print(f"  - period={per}")
                print(f"  - save_folder='{current_folder}'")
                print(f"  - plot_on={plot_on}")

                df_result = cosinor1.population_test_cosinor_pairs(
                    self._data,
                    pairs=test_pairs,
                    period=per,
                    save_folder=current_folder,
                    plot_on=plot_on
                )

                # Add period column to results (single-component doesn't include it by default)
                df_result['period'] = per

                print(f"[DEBUG] population_test_cosinor_pairs returned {len(df_result)} rows for period={per}")
                results_list.append(df_result)

            # Concatenate all results
            if len(results_list) > 1:
                df_results = pd.concat(results_list, ignore_index=True)
            else:
                df_results = results_list[0]

            print(f"[DEBUG] Total results: {len(df_results)} rows")

            if not use_temp_folder and save_cosinorpy_plots:
                print(f"[DEBUG] Checking if plots were saved:")

                if len(period_list) > 1:
                    # Multiple periods: check in subfolders
                    for per in period_list:
                        period_folder = os.path.join(folder, f"period_{per}h")
                        print(f"[DEBUG] Period {per}h folder:")
                        for pair in test_pairs:
                            pdf_path = os.path.join(period_folder, f"{pair[0]}_vs_{pair[1]}.pdf")
                            png_path = os.path.join(period_folder, f"{pair[0]}_vs_{pair[1]}.png")
                            print(f"  - PDF exists: {os.path.exists(pdf_path)} at {pdf_path}")
                            print(f"  - PNG exists: {os.path.exists(png_path)} at {png_path}")
                else:
                    # Single period: check in main folder
                    for pair in test_pairs:
                        pdf_path = os.path.join(folder, f"{pair[0]}_vs_{pair[1]}.pdf")
                        png_path = os.path.join(folder, f"{pair[0]}_vs_{pair[1]}.png")
                        print(f"  - PDF exists: {os.path.exists(pdf_path)} at {pdf_path}")
                        print(f"  - PNG exists: {os.path.exists(png_path)} at {png_path}")

            return {
                'comparison_type': 'dependent_single',
                'n_pairs': len(test_pairs),
                'n_periods': len(period_list),
                'results_df': df_results
            }
        finally:
            # Clean up temporary folder if used
            if use_temp_folder and folder:
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                    print(f"[DEBUG] Cleaned up temporary folder: {folder}")
                except Exception as e:
                    print(f"[DEBUG] Failed to clean up temp folder: {e}")

    def _compare_dependent_multi(
        self,
        test_pairs: List[Tuple[str, str]],
        period: Union[float, List[float]],
        n_components: List[int],
        analysis_method: str = 'CI',
        parameters_to_analyse: List[str] = None,
        lin_comp: bool = False,
        save_folder: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Multi-component population comparison.

        Args:
            test_pairs: List of condition pairs to compare
            period: Period value(s) - can be single or list
            n_components: List of component counts
            analysis_method: 'CI' or 'permutation'
            parameters_to_analyse: Parameters to compare (default: all)
            lin_comp: Include linear component
            save_folder: Folder for saving plots (optional)
        """
        print(f"[DEBUG _compare_dependent_multi] Using cosinor.compare_pairs_population")
        print(f"[DEBUG] analysis_method={analysis_method}")
        print(f"[DEBUG] parameters_to_analyse={parameters_to_analyse}")
        print(f"[DEBUG] lin_comp={lin_comp}")

        # Default parameters
        if parameters_to_analyse is None:
            parameters_to_analyse = ['amplitude', 'acrophase', 'mesor']

        # Convert period to list if needed
        period_list = period if isinstance(period, list) else [period]

        results_list = []

        # Iterate over all period and n_components combinations
        for per in period_list:
            for n_comp in n_components:
                print(f"[DEBUG] Processing period={per}, n_components={n_comp}")

                # NOTE: CosinorPy expects period and n_components as lists
                df_result = cosinor.compare_pairs_population(
                    self._data,
                    pairs=test_pairs,
                    n_components=[n_comp],  # Must be list
                    period=[per],           # Must be list
                    analysis=analysis_method,
                    parameters_to_analyse=parameters_to_analyse,
                    parameters_angular=['acrophase'],
                    lin_comp=lin_comp,
                    folder=save_folder or ''  # Empty string for no plots
                )

                print(f"[DEBUG] Got {len(df_result)} rows for period={per}, n_components={n_comp}")
                results_list.append(df_result)

        # Concatenate all results
        if len(results_list) > 1:
            df_results = pd.concat(results_list, ignore_index=True)
        else:
            df_results = results_list[0]

        print(f"[DEBUG] compare_pairs_population returned {len(df_results)} total rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        return {
            'comparison_type': 'dependent_multi',
            'n_pairs': len(test_pairs),
            'n_components': n_components,
            'n_periods': len(period_list),
            'results_df': df_results
        }

    # ========================================================================
    # METHOD 6: NON-LINEAR ANALYSIS (INDEPENDENT DATA)
    # ========================================================================

    def nonlinear_independent(
        self,
        variable: str,
        condition: str,
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        bootstrap_size: int = 100,
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Non-linear cosinor analysis for independent data.

        Detects amplification (damped/forced oscillations) and linear trends.
        Model: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t

        Workflow based on n_components and period:

        Case 1: n_components=[1], single period
            → fit_generalized_cosinor_group() - stats calculated analytically

        Case 2: n_components=[1], multiple periods
            → fit_generalized_cosinor_group() × N periods, compare by p-value

        Case 3: n_components=[N] (N>1, fixed), single period
            → fit_generalized_cosinor_n_comp_group()
            → bootstrap_generalized_cosinor_n_comp_group() for amplitude/acrophase stats

        Case 4: n_components=[N] (N>1, fixed), multiple periods
            → fit_generalized_cosinor_n_comp_group() × N periods

        Case 5: n_components=[1,2,...] (auto-select), single period
            → fit_generalized_cosinor_n_comp_group_best()
            → bootstrap_generalized_cosinor_n_comp_group_best()

        Case 6: n_components=[1,2,...] (auto-select), multiple periods
            → fit_generalized_cosinor_n_comp_group_best() × N periods

        Args:
            variable: Variable name
            condition: Condition name
            period: Period value or list of periods to test
            n_components: List of components [1], [3], or [1,2,3] for auto-selection
            bootstrap_size: Number of bootstrap samples (for multi-component)
            save_folder: Folder for saving plots
            save_cosinorpy_plots: If True, generate and save CosinorPy plots

        Returns:
            Dict with nonlinear results, or List of Dicts for multiple periods
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG nonlinear_independent] variable={variable}, condition={condition}")
        print(f"[DEBUG nonlinear_independent] period={period}, n_components={n_components}")
        print(f"[DEBUG nonlinear_independent] save_cosinorpy_plots={save_cosinorpy_plots}")

        test_name = f"{variable}_{condition}"
        df_test = self._data[self._data['test'] == test_name].copy()

        if len(df_test) == 0:
            raise ValueError(f"No data found for test: {test_name}")

        # Normalize period to list
        period_list = period if isinstance(period, list) else [period]
        is_multi_period = len(period_list) > 1

        # Determine plot settings
        should_plot = save_cosinorpy_plots and save_folder is not None
        plot_folder = save_folder if should_plot else None

        try:
            # Case 1 & 2: Single component (n_components=[1])
            if len(n_components) == 1 and n_components[0] == 1:
                return self._nonlinear_independent_single_comp(
                    df_test, test_name, period_list, plot_folder, bootstrap_size
                )

            # Case 3 & 4: Fixed N components (n_components=[N] where N>1)
            elif len(n_components) == 1 and n_components[0] > 1:
                return self._nonlinear_independent_fixed_ncomp(
                    df_test, test_name, period_list, n_components[0],
                    bootstrap_size, plot_folder
                )

            # Case 5 & 6: Auto-select components (n_components=[1,2,3,...])
            else:
                return self._nonlinear_independent_best_ncomp(
                    df_test, test_name, period_list, n_components,
                    bootstrap_size, plot_folder
                )

        except Exception as e:
            print(f"[DEBUG nonlinear_independent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _nonlinear_independent_single_comp(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period_list: List[float],
        plot_folder: Optional[str],
        bootstrap_size: int
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Single component (n_components=1) nonlinear analysis.
        Uses fit_generalized_cosinor_group() which calculates stats analytically.
        No bootstrap needed for 1-component.
        """
        print(f"[DEBUG] Using fit_generalized_cosinor_group (1 component)")

        all_results = []

        for period in period_list:
            print(f"[DEBUG] Fitting period={period}")

            # Build params - fit_generalized_cosinor_group only takes scalar period
            params = {
                'period': period,
                'plot': plot_folder is not None
            }
            if plot_folder:
                params['folder'] = plot_folder

            # Call CosinorPy - this returns full statistics for 1-component
            df_results = cosinor_nonlin.fit_generalized_cosinor_group(
                df_test,
                **params
            )

            print(f"[DEBUG] fit_generalized_cosinor_group returned {len(df_results)} rows")
            print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

            result_row = df_results[df_results['test'] == test_name].iloc[0]

            # Parse CI strings to tuples if they exist
            amp_ci = self._parse_ci(result_row.get('CI(amplitude)'))
            acro_ci = self._parse_ci(result_row.get('CI(acrophase)'))
            amplification_ci = self._parse_ci(result_row.get('CI(amplification)'))
            lin_comp_ci = self._parse_ci(result_row.get('CI(lin_comp)'))

            result_dict = {
                'test_name': test_name,
                'period': period,
                'n_components': 1,
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'mesor': df_test['y'].mean(),  # Calculate from data
                'amplification': result_row.get('amplification'),
                'lin_comp': result_row.get('lin_comp'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'p_amplitude': result_row.get('p(amplitude)'),
                'p_acrophase': result_row.get('p(acrophase)'),
                'p_amplification': result_row.get('p(amplification)'),
                'p_lin_comp': result_row.get('p(lin_comp)'),
                'q_amplitude': result_row.get('q(amplitude)'),
                'q_acrophase': result_row.get('q(acrophase)'),
                'q_amplification': result_row.get('q(amplification)'),
                'q_lin_comp': result_row.get('q(lin_comp)'),
                'amplitude_ci': amp_ci,
                'acrophase_ci': acro_ci,
                'amplification_ci': amplification_ci,
                'lin_comp_ci': lin_comp_ci,
                'results_df': df_results
            }

            all_results.append(result_dict)

        # If multiple periods, mark best model by min p-value
        if len(all_results) > 1:
            best_idx = min(range(len(all_results)),
                          key=lambda i: all_results[i].get('p_value', float('inf')) or float('inf'))
            for i, r in enumerate(all_results):
                r['best_model'] = 'Yes (min p-value)' if i == best_idx else 'No'
            return all_results
        else:
            return all_results[0]

    def _nonlinear_independent_fixed_ncomp(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period_list: List[float],
        n_components: int,
        bootstrap_size: int,
        plot_folder: Optional[str]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fixed N components (N>1) nonlinear analysis.
        Uses fit_generalized_cosinor_n_comp_group() then bootstrap for amplitude/acrophase stats.
        """
        print(f"[DEBUG] Using fit_generalized_cosinor_n_comp_group (fixed {n_components} components)")

        all_results = []

        for period in period_list:
            print(f"[DEBUG] Fitting period={period}, n_components={n_components}")

            # Build params
            params = {
                'period': period,
                'n_components': n_components,
                'plot': plot_folder is not None
            }
            if plot_folder:
                params['folder'] = plot_folder

            # Fit model
            df_results = cosinor_nonlin.fit_generalized_cosinor_n_comp_group(
                df_test,
                **params
            )

            print(f"[DEBUG] fit_generalized_cosinor_n_comp_group returned {len(df_results)} rows")

            # Bootstrap for amplitude/acrophase statistics (recommended for n_comp > 1)
            df_bootstrap = cosinor_nonlin.bootstrap_generalized_cosinor_n_comp_group(
                df_test,
                period=period,
                n_components=n_components,
                bootstrap_size=bootstrap_size
            )

            print(f"[DEBUG] bootstrap returned {len(df_bootstrap)} rows")

            # Use bootstrap results which have more complete statistics
            result_row = df_bootstrap[df_bootstrap['test'] == test_name].iloc[0]

            # Parse CI strings
            amp_ci = self._parse_ci(result_row.get('CI(amplitude)'))
            acro_ci = self._parse_ci(result_row.get('CI(acrophase)'))
            amplification_ci = self._parse_ci(result_row.get('CI(amplification)'))
            lin_comp_ci = self._parse_ci(result_row.get('CI(lin_comp)'))

            result_dict = {
                'test_name': test_name,
                'period': period,
                'n_components': n_components,
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'mesor': df_test['y'].mean(),
                'amplification': result_row.get('amplification'),
                'lin_comp': result_row.get('lin_comp'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'p_amplitude': result_row.get('p(amplitude)'),
                'p_acrophase': result_row.get('p(acrophase)'),
                'p_amplification': result_row.get('p(amplification)'),
                'p_lin_comp': result_row.get('p(lin_comp)'),
                'q_amplitude': result_row.get('q(amplitude)'),
                'q_acrophase': result_row.get('q(acrophase)'),
                'q_amplification': result_row.get('q(amplification)'),
                'q_lin_comp': result_row.get('q(lin_comp)'),
                'amplitude_ci': amp_ci,
                'acrophase_ci': acro_ci,
                'amplification_ci': amplification_ci,
                'lin_comp_ci': lin_comp_ci,
                'results_df': df_bootstrap
            }

            all_results.append(result_dict)

        # If multiple periods, mark best model
        if len(all_results) > 1:
            best_idx = min(range(len(all_results)),
                          key=lambda i: all_results[i].get('p_value', float('inf')) or float('inf'))
            for i, r in enumerate(all_results):
                r['best_model'] = 'Yes (min p-value)' if i == best_idx else 'No'
            return all_results
        else:
            return all_results[0]

    def _nonlinear_independent_best_ncomp(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period_list: List[float],
        n_components: List[int],
        bootstrap_size: int,
        plot_folder: Optional[str]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Auto-select best n_components with bootstrap.
        Uses fit_generalized_cosinor_n_comp_group_best() then bootstrap.
        """
        print(f"[DEBUG] Using fit_generalized_cosinor_n_comp_group_best (auto-select from {n_components})")

        all_results = []

        for period in period_list:
            print(f"[DEBUG] Fitting period={period}, testing n_components={n_components}")

            # Build params
            params = {
                'period': period,
                'n_components': n_components,
                'plot': plot_folder is not None
            }
            if plot_folder:
                params['folder'] = plot_folder

            # Get best models
            df_best = cosinor_nonlin.fit_generalized_cosinor_n_comp_group_best(
                df_test,
                **params
            )

            print(f"[DEBUG] fit_generalized_cosinor_n_comp_group_best returned {len(df_best)} rows")

            # Bootstrap using best models
            df_bootstrap = cosinor_nonlin.bootstrap_generalized_cosinor_n_comp_group_best(
                df_test,
                df_best_models=df_best,
                bootstrap_size=bootstrap_size
            )

            print(f"[DEBUG] bootstrap returned {len(df_bootstrap)} rows")

            result_row = df_bootstrap[df_bootstrap['test'] == test_name].iloc[0]
            best_row = df_best[df_best['test'] == test_name].iloc[0]

            # Parse CI strings
            amp_ci = self._parse_ci(result_row.get('CI(amplitude)'))
            acro_ci = self._parse_ci(result_row.get('CI(acrophase)'))
            amplification_ci = self._parse_ci(best_row.get('CI(amplification)'))
            lin_comp_ci = self._parse_ci(best_row.get('CI(lin_comp)'))

            result_dict = {
                'test_name': test_name,
                'period': period,
                'n_components': int(result_row.get('n_components', 1)),
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'mesor': df_test['y'].mean(),
                'amplification': result_row.get('amplification'),
                'lin_comp': result_row.get('lin_comp'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'p_amplitude': result_row.get('p(amplitude)'),
                'p_acrophase': result_row.get('p(acrophase)'),
                'p_amplification': best_row.get('p(amplification)'),
                'p_lin_comp': best_row.get('p(lin_comp)'),
                'q_amplitude': result_row.get('q(amplitude)'),
                'q_acrophase': result_row.get('q(acrophase)'),
                'q_amplification': best_row.get('q(amplification)'),
                'q_lin_comp': best_row.get('q(lin_comp)'),
                'amplitude_ci': amp_ci,
                'acrophase_ci': acro_ci,
                'amplification_ci': amplification_ci,
                'lin_comp_ci': lin_comp_ci,
                'results_df': df_bootstrap,
                'best_models_df': df_best
            }

            all_results.append(result_dict)

        # If multiple periods, mark best model
        if len(all_results) > 1:
            best_idx = min(range(len(all_results)),
                          key=lambda i: all_results[i].get('p_value', float('inf')) or float('inf'))
            for i, r in enumerate(all_results):
                r['best_model'] = 'Yes (min p-value)' if i == best_idx else 'No'
            return all_results
        else:
            return all_results[0]

    def _parse_ci(self, ci_value) -> Optional[Tuple[float, float]]:
        """Parse CI value from various formats to tuple."""
        if ci_value is None:
            return None
        if isinstance(ci_value, (list, tuple)) and len(ci_value) == 2:
            return (float(ci_value[0]), float(ci_value[1]))
        if isinstance(ci_value, str):
            # Try to parse string format like "[0.5, 1.5]"
            try:
                import ast
                parsed = ast.literal_eval(ci_value)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                    return (float(parsed[0]), float(parsed[1]))
            except:
                pass
        return None

    # ========================================================================
    # METHOD 7: NON-LINEAR ANALYSIS (DEPENDENT/POPULATION DATA)
    # ========================================================================

    def nonlinear_dependent(
        self,
        variable: str,
        condition: str,
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Non-linear cosinor analysis for dependent/population data.

        Model: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t
        Data format: test names like "var_cond_rep1", "var_cond_rep2", etc.

        Workflow based on n_components and period:

        Case 1: n_components=[1], single period
            → population_fit_generalized_cosinor_group() - stats from replicate variance

        Case 2: n_components=[1], multiple periods
            → population_fit_generalized_cosinor_group() × N periods

        Case 3: n_components=[N] (N>1, fixed), single period
            → population_fit_generalized_cosinor_n_comp_group()

        Case 4: n_components=[N] (N>1, fixed), multiple periods
            → population_fit_generalized_cosinor_n_comp_group() × N periods

        Case 5: n_components=[1,2,...] (auto-select), single period
            → population_fit_generalized_cosinor_n_comp_group_best()

        Case 6: n_components=[1,2,...] (auto-select), multiple periods
            → population_fit_generalized_cosinor_n_comp_group_best() × N periods

        Note: No bootstrap for population data - stats calculated from variance between replicates.

        Args:
            variable: Variable name
            condition: Condition name
            period: Period value or list of periods to test
            n_components: List of components [1], [3], or [1,2,3] for auto-selection
            save_folder: Folder for saving plots
            save_cosinorpy_plots: If True, generate and save CosinorPy plots

        Returns:
            Dict with nonlinear results, or List of Dicts for multiple periods
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG nonlinear_dependent] variable={variable}, condition={condition}")
        print(f"[DEBUG nonlinear_dependent] period={period}, n_components={n_components}")
        print(f"[DEBUG nonlinear_dependent] save_cosinorpy_plots={save_cosinorpy_plots}")

        # For dependent data, test names are in format "variable-condition_repN" (with hyphen)
        # This is because CosinorPy uses test.split("_")[0] to get base name
        test_name = f"{variable}_{condition}"  # Keep original for return value
        test_pattern = f"{variable}-{condition}"  # Pattern with hyphen for filtering

        print(f"[DEBUG] Looking for tests starting with: {test_pattern}")
        df_test = self._data[self._data['test'].str.startswith(test_pattern)].copy()

        if len(df_test) == 0:
            # Fallback: try with underscore (in case data wasn't converted)
            df_test = self._data[self._data['test'].str.startswith(test_name)].copy()

        if len(df_test) == 0:
            print(f"[DEBUG] Available tests: {self._data['test'].unique().tolist()}")
            raise ValueError(f"No data found for test pattern: {test_pattern}* or {test_name}*")

        # Normalize period to list
        period_list = period if isinstance(period, list) else [period]

        # Determine plot settings
        should_plot = save_cosinorpy_plots and save_folder is not None
        plot_folder = save_folder if should_plot else None

        try:
            # Case 1 & 2: Single component (n_components=[1])
            if len(n_components) == 1 and n_components[0] == 1:
                return self._nonlinear_dependent_single_comp(
                    df_test, test_name, period_list, plot_folder
                )

            # Case 3 & 4: Fixed N components (n_components=[N] where N>1)
            elif len(n_components) == 1 and n_components[0] > 1:
                return self._nonlinear_dependent_fixed_ncomp(
                    df_test, test_name, period_list, n_components[0], plot_folder
                )

            # Case 5 & 6: Auto-select components (n_components=[1,2,3,...])
            else:
                return self._nonlinear_dependent_best_ncomp(
                    df_test, test_name, period_list, n_components, plot_folder
                )

        except Exception as e:
            print(f"[DEBUG nonlinear_dependent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _nonlinear_dependent_single_comp(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period_list: List[float],
        plot_folder: Optional[str]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Single component (n_components=1) nonlinear analysis for population data.
        Uses population_fit_generalized_cosinor_group() which provides rich statistics.
        """
        print(f"[DEBUG] Using population_fit_generalized_cosinor_group (1 component)")

        all_results = []

        for period in period_list:
            print(f"[DEBUG] Fitting period={period}")

            params = {
                'period': period,
                'plot': plot_folder is not None
            }
            if plot_folder:
                params['folder'] = plot_folder

            df_results = cosinor_nonlin.population_fit_generalized_cosinor_group(
                df_test,
                **params
            )

            print(f"[DEBUG] population_fit_generalized_cosinor_group returned {len(df_results)} rows")
            print(f"[DEBUG] Columns: {df_results.columns.tolist()}")
            print(f"[DEBUG] Available tests in results: {df_results['test'].tolist()}")

            # CosinorPy population functions use test.split("_")[0] to get base name
            # Our test names are in format "variable-condition_repN"
            # So CosinorPy will return "variable-condition" as test name
            #
            # test_name here is "variable_condition" (from caller)
            # We need to convert it to "variable-condition" format to match
            expected_test_name = test_name.replace('_', '-')
            print(f"[DEBUG] Looking for test: {expected_test_name}")

            # Find the result row
            matching_rows = df_results[df_results['test'] == expected_test_name]

            # Fallback: if only one result, use it
            if len(matching_rows) == 0 and len(df_results) == 1:
                print(f"[DEBUG] Only one result, using it directly")
                matching_rows = df_results

            if len(matching_rows) == 0:
                raise ValueError(f"No results found for test: {expected_test_name}")

            result_row = matching_rows.iloc[0]

            # Parse CI strings
            amp_ci = self._parse_ci(result_row.get('CI(amplitude)'))
            acro_ci = self._parse_ci(result_row.get('CI(acrophase)'))
            amplification_ci = self._parse_ci(result_row.get('CI(amplification)'))
            lin_comp_ci = self._parse_ci(result_row.get('CI(lin_comp)'))

            result_dict = {
                'test_name': test_name,
                'period': period,
                'n_components': 1,
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'mesor': df_test['y'].mean(),
                'amplification': result_row.get('amplification'),
                'lin_comp': result_row.get('lin_comp'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'p_amplitude': result_row.get('p(amplitude)'),
                'p_acrophase': result_row.get('p(acrophase)'),
                'p_amplification': result_row.get('p(amplification)'),
                'p_lin_comp': result_row.get('p(lin_comp)'),
                'q_amplitude': result_row.get('q(amplitude)'),
                'q_acrophase': result_row.get('q(acrophase)'),
                'q_amplification': result_row.get('q(amplification)'),
                'q_lin_comp': result_row.get('q(lin_comp)'),
                'amplitude_ci': amp_ci,
                'acrophase_ci': acro_ci,
                'amplification_ci': amplification_ci,
                'lin_comp_ci': lin_comp_ci,
                'results_df': df_results
            }

            all_results.append(result_dict)

        # If multiple periods, mark best model by min p-value
        if len(all_results) > 1:
            best_idx = min(range(len(all_results)),
                          key=lambda i: all_results[i].get('p_value', float('inf')) or float('inf'))
            for i, r in enumerate(all_results):
                r['best_model'] = 'Yes (min p-value)' if i == best_idx else 'No'
            return all_results
        else:
            return all_results[0]

    def _nonlinear_dependent_fixed_ncomp(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period_list: List[float],
        n_components: int,
        plot_folder: Optional[str]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fixed N components (N>1) nonlinear analysis for population data.
        Uses population_fit_generalized_cosinor_n_comp_group().
        No bootstrap for population data - stats from replicate variance.
        """
        print(f"[DEBUG] Using population_fit_generalized_cosinor_n_comp_group (fixed {n_components} components)")

        all_results = []

        for period in period_list:
            print(f"[DEBUG] Fitting period={period}, n_components={n_components}")

            params = {
                'period': period,
                'n_components': n_components,
                'plot': plot_folder is not None
            }
            if plot_folder:
                params['folder'] = plot_folder

            df_results = cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group(
                df_test,
                **params
            )

            print(f"[DEBUG] population_fit_generalized_cosinor_n_comp_group returned {len(df_results)} rows")
            print(f"[DEBUG] Available tests: {df_results['test'].tolist()}")

            # CosinorPy uses test.split("_")[0], so convert test_name to match
            expected_test_name = test_name.replace('_', '-')
            print(f"[DEBUG] Looking for test: {expected_test_name}")

            matching_rows = df_results[df_results['test'] == expected_test_name]

            # Fallback: if only one result, use it
            if len(matching_rows) == 0 and len(df_results) == 1:
                print(f"[DEBUG] Only one result, using it directly")
                matching_rows = df_results

            if len(matching_rows) == 0:
                raise ValueError(f"No results found for test: {expected_test_name}")

            result_row = matching_rows.iloc[0]

            # Parse CI strings
            amplification_ci = self._parse_ci(result_row.get('CI(amplification)'))
            lin_comp_ci = self._parse_ci(result_row.get('CI(lin_comp)'))

            result_dict = {
                'test_name': test_name,
                'period': period,
                'n_components': n_components,
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'mesor': result_row.get('mesor'),
                'amplification': result_row.get('amplification'),
                'lin_comp': result_row.get('lin_comp'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'rss': result_row.get('RSS'),
                'p_amplification': result_row.get('p(amplification)'),
                'p_lin_comp': result_row.get('p(lin_comp)'),
                'q_amplification': result_row.get('q(amplification)'),
                'q_lin_comp': result_row.get('q(lin_comp)'),
                'amplification_ci': amplification_ci,
                'lin_comp_ci': lin_comp_ci,
                'peaks': result_row.get('peaks'),
                'troughs': result_row.get('troughs'),
                'results_df': df_results
            }

            all_results.append(result_dict)

        # If multiple periods, mark best model
        if len(all_results) > 1:
            best_idx = min(range(len(all_results)),
                          key=lambda i: all_results[i].get('p_value', float('inf')) or float('inf'))
            for i, r in enumerate(all_results):
                r['best_model'] = 'Yes (min p-value)' if i == best_idx else 'No'
            return all_results
        else:
            return all_results[0]

    def _nonlinear_dependent_best_ncomp(
        self,
        df_test: pd.DataFrame,
        test_name: str,
        period_list: List[float],
        n_components: List[int],
        plot_folder: Optional[str]
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Auto-select best n_components for population data.
        Uses population_fit_generalized_cosinor_n_comp_group_best().
        No bootstrap for population data.
        """
        print(f"[DEBUG] Using population_fit_generalized_cosinor_n_comp_group_best (auto-select from {n_components})")

        all_results = []

        for period in period_list:
            print(f"[DEBUG] Fitting period={period}, testing n_components={n_components}")

            params = {
                'period': period,
                'n_components': n_components,
                'plot': plot_folder is not None
            }
            if plot_folder:
                params['folder'] = plot_folder

            df_best = cosinor_nonlin.population_fit_generalized_cosinor_n_comp_group_best(
                df_test,
                **params
            )

            print(f"[DEBUG] population_fit_generalized_cosinor_n_comp_group_best returned {len(df_best)} rows")
            print(f"[DEBUG] Available tests: {df_best['test'].tolist()}")

            # CosinorPy uses test.split("_")[0], so convert test_name to match
            expected_test_name = test_name.replace('_', '-')
            print(f"[DEBUG] Looking for test: {expected_test_name}")

            matching_rows = df_best[df_best['test'] == expected_test_name]

            # Fallback: if only one result, use it
            if len(matching_rows) == 0 and len(df_best) == 1:
                print(f"[DEBUG] Only one result, using it directly")
                matching_rows = df_best

            if len(matching_rows) == 0:
                raise ValueError(f"No results found for test: {expected_test_name}")

            result_row = matching_rows.iloc[0]

            # Parse CI strings
            amplification_ci = self._parse_ci(result_row.get('CI(amplification)'))
            lin_comp_ci = self._parse_ci(result_row.get('CI(lin_comp)'))

            result_dict = {
                'test_name': test_name,
                'period': period,
                'n_components': int(result_row.get('n_components', 1)),
                'amplitude': result_row.get('amplitude'),
                'acrophase': result_row.get('acrophase'),
                'mesor': result_row.get('mesor'),
                'amplification': result_row.get('amplification'),
                'lin_comp': result_row.get('lin_comp'),
                'p_value': result_row.get('p'),
                'q_value': result_row.get('q'),
                'rss': result_row.get('RSS'),
                'p_amplification': result_row.get('p(amplification)'),
                'p_lin_comp': result_row.get('p(lin_comp)'),
                'q_amplification': result_row.get('q(amplification)'),
                'q_lin_comp': result_row.get('q(lin_comp)'),
                'amplification_ci': amplification_ci,
                'lin_comp_ci': lin_comp_ci,
                'peaks': result_row.get('peaks'),
                'troughs': result_row.get('troughs'),
                'results_df': df_best
            }

            all_results.append(result_dict)

        # If multiple periods, mark best model
        if len(all_results) > 1:
            best_idx = min(range(len(all_results)),
                          key=lambda i: all_results[i].get('p_value', float('inf')) or float('inf'))
            for i, r in enumerate(all_results):
                r['best_model'] = 'Yes (min p-value)' if i == best_idx else 'No'
            return all_results
        else:
            return all_results[0]

    # ========================================================================
    # METHOD 8: NON-LINEAR COMPARE CONDITIONS (INDEPENDENT DATA)
    # ========================================================================

    def nonlinear_compare_independent(
        self,
        variable: str,
        conditions: List[str],
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        bootstrap_size: int = 100,
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Non-linear comparison for independent data.

        Iterates over each value in n_components and accumulates results.
        For each n_comp value:
            - n_comp == 1:
                → fit_generalized_cosinor_compare_pairs_independent(period1, period2)
                → Stats from model fitting, no bootstrap needed
            - n_comp > 1:
                → compare_pairs_n_comp_bootstrap_group(n_components=n_comp)
                → Requires bootstrap for stats

        Args:
            variable: Variable name
            conditions: List of conditions to compare
            period: Single period or [period1, period2] for different periods
            n_components: List of component counts to compare (e.g. [1], [2], [2,3])
            bootstrap_size: Bootstrap samples (for multi-component)
            save_folder: Folder for saving plots
            save_cosinorpy_plots: If True, generate and save CosinorPy plots

        Returns:
            List of comparison result Dicts (one per pair per n_components value)
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG nonlinear_compare_independent] variable={variable}, conditions={conditions}")
        print(f"[DEBUG nonlinear_compare_independent] period={period}, n_components={n_components}")
        print(f"[DEBUG nonlinear_compare_independent] save_cosinorpy_plots={save_cosinorpy_plots}")

        # Generate pairs
        condition_pairs = self._generate_all_pairs(conditions)
        test_pairs = self._format_pairs_for_cosinorpy(condition_pairs, variable)

        print(f"[DEBUG nonlinear_compare_independent] Pairs: {test_pairs}")

        # Determine plot settings
        should_plot = save_cosinorpy_plots and save_folder is not None
        plot_folder = save_folder if should_plot else None

        try:
            all_results = []
            for n_comp in n_components:
                print(f"[DEBUG nonlinear_compare_independent] Processing n_components={n_comp}")
                if n_comp == 1:
                    results = self._nonlinear_compare_independent_single_comp(
                        test_pairs, variable, period, plot_folder
                    )
                else:
                    results = self._nonlinear_compare_independent_fixed_ncomp(
                        test_pairs, variable, period, n_comp, bootstrap_size, plot_folder
                    )
                all_results.extend(results)

            return all_results

        except Exception as e:
            print(f"[DEBUG nonlinear_compare_independent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _nonlinear_compare_independent_single_comp(
        self,
        test_pairs: List[Tuple[str, str]],
        variable: str,
        period: Union[float, List[float]],
        plot_folder: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Single component comparison for independent data.
        Uses fit_generalized_cosinor_compare_pairs_independent().
        """
        print(f"[DEBUG] Using fit_generalized_cosinor_compare_pairs_independent (1 component)")

        # Determine periods
        if isinstance(period, (int, float)):
            period1, period2 = period, period
        elif isinstance(period, list) and len(period) >= 2:
            period1, period2 = period[0], period[1]
        else:
            period1, period2 = period[0] if isinstance(period, list) else period, period[0] if isinstance(period, list) else period

        params = {
            'pairs': test_pairs,
            'period1': period1,
            'period2': period2,
            'plot': plot_folder is not None
        }
        if plot_folder:
            params['folder'] = plot_folder

        df_results = cosinor_nonlin.fit_generalized_cosinor_compare_pairs_independent(
            self._data,
            **params
        )

        print(f"[DEBUG] fit_generalized_cosinor_compare_pairs_independent returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        # Convert to list of result dicts
        results = []
        for _, row in df_results.iterrows():
            # Parse test names from "test1 vs. test2" format
            test_str = row.get('test', '')
            parts = test_str.split(' vs. ')
            if len(parts) == 2:
                cond1 = parts[0].replace(f"{variable}-", "")
                cond2 = parts[1].replace(f"{variable}-", "")
            else:
                cond1, cond2 = "unknown", "unknown"

            result_dict = {
                'variable': variable,
                'condition1': cond1,
                'condition2': cond2,
                'period1': row.get('period1', period1),
                'period2': row.get('period2', period2),
                'n_components1': 1,
                'n_components2': 1,
                'd_amplitude': row.get('d_amplitude'),
                'd_acrophase': row.get('d_acrophase'),
                'd_amplification': row.get('d_amplification'),
                'd_lin_comp': row.get('d_lin_comp'),
                'p_d_amplitude': row.get('p(d_amplitude)'),
                'p_d_acrophase': row.get('p(d_acrophase)'),
                'p_d_amplification': row.get('p(d_amplification)'),
                'p_d_lin_comp': row.get('p(d_lin_comp)'),
                'q_d_amplitude': row.get('q(d_amplitude)'),
                'q_d_acrophase': row.get('q(d_acrophase)'),
                'q_d_amplification': row.get('q(d_amplification)'),
                'q_d_lin_comp': row.get('q(d_lin_comp)'),
                'd_amplitude_ci': self._parse_ci(row.get('CI(d_amplitude)')),
                'd_acrophase_ci': self._parse_ci(row.get('CI(d_acrophase)')),
                'd_amplification_ci': self._parse_ci(row.get('CI(d_amplification)')),
                'd_lin_comp_ci': self._parse_ci(row.get('CI(d_lin_comp)')),
            }
            results.append(result_dict)

        return results

    def _nonlinear_compare_independent_fixed_ncomp(
        self,
        test_pairs: List[Tuple[str, str]],
        variable: str,
        period: Union[float, List[float]],
        n_comp: int,
        bootstrap_size: int,
        plot_folder: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Multi-component comparison with fixed n_components for independent data.
        Uses compare_pairs_n_comp_bootstrap_group().
        """
        print(f"[DEBUG] Using compare_pairs_n_comp_bootstrap_group (n_components={n_comp})")

        # Use single period for both
        single_period = period[0] if isinstance(period, list) else period

        params = {
            'pairs': test_pairs,
            'n_components': n_comp,
            'period': single_period,
            'bootstrap_size': bootstrap_size,
            'plot': plot_folder is not None
        }
        if plot_folder:
            params['folder'] = plot_folder

        df_results = cosinor_nonlin.compare_pairs_n_comp_bootstrap_group(
            self._data,
            **params
        )

        print(f"[DEBUG] compare_pairs_n_comp_bootstrap_group returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        return self._parse_bootstrap_compare_results(df_results, variable)

    def _nonlinear_compare_independent_auto_select(
        self,
        test_pairs: List[Tuple[str, str]],
        variable: str,
        conditions: List[str],
        period: Union[float, List[float]],
        n_components: List[int],
        bootstrap_size: int,
        plot_folder: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Auto-select best models first, then compare for independent data.
        Uses fit_generalized_cosinor_n_comp_group_best() then compare_pairs_n_comp_bootstrap_group().
        """
        print(f"[DEBUG] Auto-selecting best models, then comparing")

        single_period = period[0] if isinstance(period, list) else period

        # First, get best models for each test
        df_best_models = cosinor_nonlin.fit_generalized_cosinor_n_comp_group_best(
            self._data,
            period=single_period,
            n_components=n_components,
            plot=False  # Don't plot individual fits
        )

        print(f"[DEBUG] Best models identified: {len(df_best_models)} tests")
        if not df_best_models.empty:
            print(f"[DEBUG] Best models:\n{df_best_models[['test', 'n_components']].to_string()}")

        # Run bootstrap comparison using best models
        params = {
            'pairs': test_pairs,
            'df_best_models': df_best_models,
            'bootstrap_size': bootstrap_size,
            'plot': plot_folder is not None
        }
        if plot_folder:
            params['folder'] = plot_folder

        df_results = cosinor_nonlin.compare_pairs_n_comp_bootstrap_group(
            self._data,
            **params
        )

        print(f"[DEBUG] compare_pairs_n_comp_bootstrap_group returned {len(df_results)} rows")

        return self._parse_bootstrap_compare_results(df_results, variable)

    def _parse_bootstrap_compare_results(
        self,
        df_results: pd.DataFrame,
        variable: str
    ) -> List[Dict[str, Any]]:
        """Parse results from compare_pairs_n_comp_bootstrap_group."""
        results = []
        for _, row in df_results.iterrows():
            # Parse test names from "test1 vs. test2" format
            test_str = row.get('test', '')
            parts = test_str.split(' vs. ')
            if len(parts) == 2:
                cond1 = parts[0].replace(f"{variable}-", "")
                cond2 = parts[1].replace(f"{variable}-", "")
            else:
                cond1, cond2 = "unknown", "unknown"

            result_dict = {
                'variable': variable,
                'condition1': cond1,
                'condition2': cond2,
                'period1': row.get('period1', 24.0),
                'period2': row.get('period2', 24.0),
                'n_components1': int(row.get('n_components1', 1)),
                'n_components2': int(row.get('n_components2', 1)),
                'd_amplitude': row.get('d_amplitude'),
                'd_acrophase': row.get('d_acrophase'),
                'd_amplification': row.get('d_amplification'),
                'd_lin_comp': row.get('d_lin_comp'),
                'p_d_amplitude': row.get('p(d_amplitude)'),
                'p_d_acrophase': row.get('p(d_acrophase)'),
                'p_d_amplification': row.get('p(d_amplification)'),
                'p_d_lin_comp': row.get('p(d_lin_comp)'),
                'q_d_amplitude': row.get('q(d_amplitude)'),
                'q_d_acrophase': row.get('q(d_acrophase)'),
                'q_d_amplification': row.get('q(d_amplification)'),
                'q_d_lin_comp': row.get('q(d_lin_comp)'),
                'd_amplitude_ci': self._parse_ci(row.get('CI(d_amplitude)')),
                'd_acrophase_ci': self._parse_ci(row.get('CI(d_acrophase)')),
                'd_amplification_ci': self._parse_ci(row.get('CI(d_amplification)')),
                'd_lin_comp_ci': self._parse_ci(row.get('CI(d_lin_comp)')),
            }
            results.append(result_dict)

        return results

    # ========================================================================
    # METHOD 9: NON-LINEAR COMPARE CONDITIONS (DEPENDENT/POPULATION DATA)
    # ========================================================================

    def nonlinear_compare_dependent(
        self,
        variable: str,
        conditions: List[str],
        period: Union[float, List[float]],
        n_components: List[int] = [1],
        save_folder: Optional[str] = None,
        save_cosinorpy_plots: bool = False,
        df_best_models: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Non-linear comparison for dependent/population data.

        If df_best_models is provided externally, uses it directly via
        population_compare_pairs_n_comp_group(df_best_models=...).

        Otherwise, iterates over each value in n_components and accumulates results.
        For each n_comp value:
            - n_comp == 1:
                → population_fit_generalized_cosinor_compare_pairs(period1, period2)
                → Allows different periods per condition
            - n_comp > 1:
                → population_compare_pairs_n_comp_group(n_components=n_comp, period=period)
                → Uses fixed n_components and period

        Args:
            variable: Variable name
            conditions: List of conditions to compare
            period: Single period or [period1, period2] for different periods
            n_components: List of component counts to compare (e.g. [1], [2], [2,3])
            save_folder: Folder for saving plots
            save_cosinorpy_plots: If True, generate and save CosinorPy plots
            df_best_models: Optional pre-computed best models DataFrame

        Returns:
            List of comparison result Dicts (one per pair per n_components value)
        """
        if self._data is None:
            raise ValueError("No data loaded")

        print(f"[DEBUG nonlinear_compare_dependent] variable={variable}, conditions={conditions}")
        print(f"[DEBUG nonlinear_compare_dependent] period={period}, n_components={n_components}")
        print(f"[DEBUG nonlinear_compare_dependent] save_cosinorpy_plots={save_cosinorpy_plots}")

        # Generate pairs
        condition_pairs = self._generate_all_pairs(conditions)
        test_pairs = self._format_pairs_for_cosinorpy(condition_pairs, variable)

        print(f"[DEBUG nonlinear_compare_dependent] Pairs: {test_pairs}")

        # Determine plot settings
        should_plot = save_cosinorpy_plots and save_folder is not None
        plot_folder = save_folder if should_plot else None

        try:
            # If df_best_models provided externally, use it directly
            if df_best_models is not None:
                return self._nonlinear_compare_dependent_multi_comp(
                    test_pairs, variable, period, n_components, plot_folder, df_best_models
                )

            # Iterate over each n_components value
            all_results = []
            for n_comp in n_components:
                print(f"[DEBUG nonlinear_compare_dependent] Processing n_components={n_comp}")
                if n_comp == 1:
                    results = self._nonlinear_compare_dependent_single_comp(
                        test_pairs, variable, period, plot_folder
                    )
                else:
                    results = self._nonlinear_compare_dependent_multi_comp(
                        test_pairs, variable, period, [n_comp], plot_folder, None
                    )
                all_results.extend(results)

            return all_results

        except Exception as e:
            print(f"[DEBUG nonlinear_compare_dependent] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _nonlinear_compare_dependent_single_comp(
        self,
        test_pairs: List[Tuple[str, str]],
        variable: str,
        period: Union[float, List[float]],
        plot_folder: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Single component comparison for population data.
        Uses population_fit_generalized_cosinor_compare_pairs().
        """
        print(f"[DEBUG] Using population_fit_generalized_cosinor_compare_pairs (1 component)")

        # Determine periods
        if isinstance(period, (int, float)):
            period1, period2 = period, period
        elif isinstance(period, list) and len(period) >= 2:
            period1, period2 = period[0], period[1]
        else:
            period1, period2 = period[0] if isinstance(period, list) else period, period[0] if isinstance(period, list) else period

        params = {
            'pairs': test_pairs,
            'period1': period1,
            'period2': period2,
            'plot': plot_folder is not None
        }
        if plot_folder:
            params['folder'] = plot_folder

        df_results = cosinor_nonlin.population_fit_generalized_cosinor_compare_pairs(
            self._data,
            **params
        )

        print(f"[DEBUG] population_fit_generalized_cosinor_compare_pairs returned {len(df_results)} rows")
        print(f"[DEBUG] Columns: {df_results.columns.tolist()}")

        # Convert to list of result dicts
        results = []
        for _, row in df_results.iterrows():
            # Parse test names from "test1 vs. test2" format
            test_str = row.get('test', '')
            parts = test_str.split(' vs. ')
            if len(parts) == 2:
                cond1 = parts[0].replace(f"{variable}-", "")
                cond2 = parts[1].replace(f"{variable}-", "")
            else:
                cond1, cond2 = "unknown", "unknown"

            result_dict = {
                'variable': variable,
                'condition1': cond1,
                'condition2': cond2,
                'period1': row.get('period1', period1),
                'period2': row.get('period2', period2),
                'n_components1': 1,
                'n_components2': 1,
                'd_amplitude': row.get('d_amplitude'),
                'd_acrophase': row.get('d_acrophase'),
                'd_amplification': row.get('d_amplification'),
                'd_lin_comp': row.get('d_lin_comp'),
                'p_d_amplitude': row.get('p(d_amplitude)'),
                'p_d_acrophase': row.get('p(d_acrophase)'),
                'p_d_amplification': row.get('p(d_amplification)'),
                'p_d_lin_comp': row.get('p(d_lin_comp)'),
                'q_d_amplitude': row.get('q(d_amplitude)'),
                'q_d_acrophase': row.get('q(d_acrophase)'),
                'q_d_amplification': row.get('q(d_amplification)'),
                'q_d_lin_comp': row.get('q(d_lin_comp)'),
                'd_amplitude_ci': self._parse_ci(row.get('CI(d_amplitude)')),
                'd_acrophase_ci': self._parse_ci(row.get('CI(d_acrophase)')),
                'd_amplification_ci': self._parse_ci(row.get('CI(d_amplification)')),
                'd_lin_comp_ci': self._parse_ci(row.get('CI(d_lin_comp)')),
            }
            results.append(result_dict)

        return results

    def _nonlinear_compare_dependent_multi_comp(
        self,
        test_pairs: List[Tuple[str, str]],
        variable: str,
        period: Union[float, List[float]],
        n_components: List[int],
        plot_folder: Optional[str],
        df_best_models: Optional[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Multi-component comparison for population data.
        Uses population_compare_pairs_n_comp_group().
        """
        print(f"[DEBUG] Using population_compare_pairs_n_comp_group")

        # Build params
        params = {
            'pairs': test_pairs,
            'plot': plot_folder is not None
        }
        if plot_folder:
            params['folder'] = plot_folder

        # If df_best_models provided, use it (takes n_components and period from there)
        if df_best_models is not None:
            params['df_best_models'] = df_best_models
        else:
            # Fixed n_components and period
            params['n_components'] = n_components[0] if len(n_components) == 1 else max(n_components)
            params['period'] = period if isinstance(period, (int, float)) else period[0]

        df_results = cosinor_nonlin.population_compare_pairs_n_comp_group(
            self._data,
            **params
        )

        print(f"[DEBUG] population_compare_pairs_n_comp_group returned {len(df_results)} rows")

        # Convert to list of result dicts
        results = []
        for _, row in df_results.iterrows():
            test_str = row.get('test', '')
            parts = test_str.split(' vs. ')
            if len(parts) == 2:
                cond1 = parts[0].replace(f"{variable}-", "")
                cond2 = parts[1].replace(f"{variable}-", "")
            else:
                cond1, cond2 = "unknown", "unknown"

            result_dict = {
                'variable': variable,
                'condition1': cond1,
                'condition2': cond2,
                'period1': row.get('period1'),
                'period2': row.get('period2'),
                'n_components1': int(row.get('n_components1', 1)),
                'n_components2': int(row.get('n_components2', 1)),
                'd_amplitude': row.get('d_amplitude'),
                'd_acrophase': row.get('d_acrophase'),
                'd_amplification': row.get('d_amplification'),
                'd_lin_comp': row.get('d_lin_comp'),
                'p_d_amplification': row.get('p(d_amplification)'),
                'p_d_lin_comp': row.get('p(d_lin_comp)'),
                'q_d_amplification': row.get('q(d_amplification)'),
                'q_d_lin_comp': row.get('q(d_lin_comp)'),
                'd_amplification_ci': self._parse_ci(row.get('CI(d_amplification)')),
                'd_lin_comp_ci': self._parse_ci(row.get('CI(d_lin_comp)')),
            }
            results.append(result_dict)

        return results

    def _nonlinear_compare_dependent_auto_select(
        self,
        test_pairs: List[Tuple[str, str]],
        variable: str,
        conditions: List[str],
        period: Union[float, List[float]],
        n_components: List[int],
        plot_folder: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Auto-select best n_components for each condition, then compare.

        NOTE: We cannot use population_fit_generalized_cosinor_n_comp_group_best()
        directly because it groups tests using t.split("_")[0], which fails when
        the variable name contains underscores (e.g., "circadian_noisy").
        Instead, we group conditions manually using str.startswith() and call
        get_best_model_population() for each condition separately.
        """
        print(f"[DEBUG] Auto-selecting best models before comparison")

        period_val = period if isinstance(period, (int, float)) else period[0]

        # Collect unique condition prefixes from all pairs
        all_test_names = set()
        for t1, t2 in test_pairs:
            all_test_names.add(t1)
            all_test_names.add(t2)

        # Build df_best_models by fitting each condition separately
        rows = []
        for test_name in sorted(all_test_names):
            df_pop = self._data[self._data.test.str.startswith(f'{test_name}_rep')]

            if len(df_pop) == 0:
                print(f"[DEBUG] No data found for {test_name}")
                continue

            try:
                best_comps, stats, p_dict, rhythm_params = cosinor_nonlin.get_best_model_population(
                    df_pop, period=period_val, n_components=n_components, plot=False
                )
                rows.append({
                    'test': test_name,
                    'period': period_val,
                    'n_components': best_comps
                })
                print(f"[DEBUG] Best model for {test_name}: n_components={best_comps}")
            except Exception as e:
                print(f"[DEBUG] Could not find best model for {test_name}: {e}")

        df_best_models = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['test', 'period', 'n_components'])
        print(f"[DEBUG] Built df_best_models with {len(df_best_models)} entries")

        if len(df_best_models) == 0:
            print(f"[DEBUG] No best models found, cannot compare")
            return []

        # Now compare using the correctly built best models
        return self._nonlinear_compare_dependent_multi_comp(
            test_pairs, variable, period, n_components, plot_folder, df_best_models
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _generate_all_pairs(self, conditions: List[str]) -> List[Tuple[str, str]]:
        """Generate all possible pairs from list of conditions."""
        pairs = []
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                pairs.append((conditions[i], conditions[j]))
        return pairs

    def _format_pairs_for_cosinorpy(
        self,
        pairs: List[Tuple[str, str]],
        variable: str
    ) -> List[Tuple[str, str]]:
        """Format pairs as CosinorPy test names.

        Uses hyphen separator (variable-condition) because CosinorPy
        splits on underscore to get base test names.
        """
        return [(f"{variable}-{p[0]}", f"{variable}-{p[1]}") for p in pairs]


# Module-level function for backward compatibility
def get_cosinorpy_analyzer() -> CosinorAnalyzer:
    """Get a CosinorAnalyzer instance."""
    return CosinorAnalyzer()
