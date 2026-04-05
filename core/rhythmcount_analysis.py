"""
RhythmCount Analysis Module
============================

Modular wrapper for RhythmCount library, which fits cosinor models using
count-data distributions. Designed for discrete, non-negative data such as
RNA-seq counts, locomotor activity events, and spike trains.

Source library: https://github.com/ninavelikajne/RhythmCount

Available count models:
    - Poisson
    - Generalized Poisson
    - Zero-Inflated Poisson
    - Negative Binomial
    - Zero-Inflated Negative Binomial

Analysis methods:
    1. Fit Single Model        – fit one (distribution, n_components) combination
    2. Fit All Models          – fit all combinations and return a results table
    3. Select Best Model       – choose best n_components and/or distribution
    4. Parameter CIs           – bootstrap confidence intervals for rhythm params
    5. Group Comparison        – compare rhythm fits across groups/conditions
    6. Clean Data              – remove outliers via quantile filtering

CSV Format Requirements:
------------------------
    X  : time variable (e.g., hour of day 0–23)
    Y  : count variable (non-negative integers)
    For group comparison, an additional column identifying the group is needed.

    Example:
        X,Y
        0,12
        1,9
        2,5
        ...

Author: Francisco Tassara
Version: 1.0.0
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

import pandas as pd
import numpy as np

# Check RhythmCount availability
try:
    from .RhythmCount_docs import data_processing as dproc
    from .RhythmCount_docs import helpers as hlp
    from .RhythmCount_docs import plot as rplot
    RHYTHMCOUNT_AVAILABLE = True
except ImportError:
    RHYTHMCOUNT_AVAILABLE = False
    warnings.warn("RhythmCount dependencies not available. Install statsmodels to enable count data analysis.")


# =============================================================================
# ENUMS AND PARAMETER CLASSES
# =============================================================================

class CountModel(Enum):
    """Statistical distribution for count data fitting."""
    POISSON = "poisson"
    GENERALIZED_POISSON = "gen_poisson"
    ZERO_INFLATED_POISSON = "zero_poisson"
    NEGATIVE_BINOMIAL = "nb"
    ZERO_INFLATED_NB = "zero_nb"


class SelectionTest(Enum):
    """Criterion for selecting the best model."""
    AIC = "AIC"
    BIC = "BIC"
    VUONG = "Vuong"
    F = "F"


@dataclass
class RhythmCountParameters:
    """Parameters for RhythmCount analysis."""
    period: float = 24.0
    n_components: List[int] = field(default_factory=lambda: [1, 2, 3])
    count_models: List[CountModel] = field(default_factory=lambda: [
        CountModel.POISSON,
        CountModel.GENERALIZED_POISSON,
        CountModel.NEGATIVE_BINOMIAL,
        CountModel.ZERO_INFLATED_POISSON,
        CountModel.ZERO_INFLATED_NB,
    ])
    selection_test: SelectionTest = SelectionTest.AIC
    eval_order: bool = True       # True: select best n_components first, then model
    maxiter: int = 5000
    maxfun: int = 5000
    method: str = 'nm'            # Nelder-Mead (robust for count models)
    repetitions: int = 20         # Bootstrap repetitions for CIs
    precision_rate: float = 2.0   # Tolerance for peak matching in CIs
    clean_data: bool = False       # Apply quantile-based outlier removal

    def count_model_values(self) -> List[str]:
        """Return list of count model string values for RhythmCount functions."""
        return [m.value for m in self.count_models]


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class SingleFitResult:
    """Result from fitting a single count model."""
    count_model: str
    n_components: int
    amplitude: float
    mesor: float
    peaks: np.ndarray
    heights: np.ndarray
    llr_pvalue: float
    AIC: float
    BIC: float
    RSS: float
    log_likelihood: float
    prsquared: float         # McFadden's pseudo R²
    resid_mean: float
    resid_std: float
    data_mean: float
    data_std: float
    X_test: np.ndarray       # Time values for fitted curve
    Y_test: np.ndarray       # Fitted curve values
    success: bool = True
    error: Optional[str] = None


@dataclass
class BestModelResult:
    """Result from automatic model selection."""
    count_model: str
    n_components: int
    selection_test: str
    amplitude: float
    mesor: float
    peaks: np.ndarray
    heights: np.ndarray
    llr_pvalue: float
    AIC: float
    BIC: float
    RSS: float
    prsquared: float
    X_test: np.ndarray
    Y_test: np.ndarray
    all_results: pd.DataFrame    # Full table with all fitted models
    success: bool = True
    error: Optional[str] = None


@dataclass
class ParameterCIsResult:
    """Bootstrap confidence intervals for rhythm parameters."""
    amplitude_CIs: np.ndarray    # [lower, upper]
    mesor_CIs: np.ndarray        # [lower, upper]
    peaks_CIs: np.ndarray        # shape (n_peaks, 2)
    heights_CIs: np.ndarray      # shape (n_peaks, 2)
    count_model: str
    n_components: int
    repetitions: int
    success: bool = True
    error: Optional[str] = None


@dataclass
class GroupComparisonResult:
    """Result from comparing rhythm fits across groups."""
    group_column: str
    results_table: pd.DataFrame  # One row per group with fit stats and CIs
    selection_test: str
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# ANALYZER CLASS
# =============================================================================

class RhythmCountAnalyzer:
    """
    Modular wrapper for RhythmCount count-data circadian analysis.

    Designed for datasets where Y values are discrete counts (non-negative
    integers) and standard Gaussian cosinor is inappropriate.

    Typical use cases:
        - RNA-seq read counts with circadian variation
        - Zero-inflated behavioral event counts (e.g., DAM locomotor data)
        - Overdispersed count time series

    Usage:
        analyzer = RhythmCountAnalyzer()
        analyzer.load_data(df)
        result = analyzer.fit_best_model(params)
    """

    def __init__(self):
        if not RHYTHMCOUNT_AVAILABLE:
            raise ImportError(
                "RhythmCount dependencies are not available. "
                "Install statsmodels: pip install statsmodels"
            )
        self._data: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------------

    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load data for analysis.

        Args:
            data: DataFrame with at minimum columns 'X' (time) and 'Y' (count).
                  For group comparison, include an additional grouping column.
                  Y values must be non-negative integers.

        Raises:
            ValueError: If required columns are missing or Y contains negatives.
        """
        if 'X' not in data.columns or 'Y' not in data.columns:
            raise ValueError("DataFrame must have 'X' (time) and 'Y' (count) columns.")
        if (data['Y'].dropna() < 0).any():
            raise ValueError("Y column contains negative values. RhythmCount requires count data (Y >= 0).")

        self._data = data.copy()

    # -------------------------------------------------------------------------
    # METHOD 1: FIT SINGLE MODEL
    # -------------------------------------------------------------------------

    def fit_single_model(
        self,
        count_model: CountModel,
        n_components: int,
        params: Optional[RhythmCountParameters] = None,
    ) -> SingleFitResult:
        """
        Fit one specific count model with a given number of harmonic components.

        Args:
            count_model: The count distribution to use (e.g., CountModel.POISSON).
            n_components: Number of harmonic components (1 = fundamental only).
            params: Analysis parameters. Uses defaults if None.

        Returns:
            SingleFitResult with fit statistics and rhythm parameters.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if params is None:
            params = RhythmCountParameters()

        df = self._prepare_data(params)

        try:
            _, df_result, _ = dproc.fit_to_model(
                df,
                n_components=n_components,
                count_model=count_model.value,
                period=params.period,
                maxiter=params.maxiter,
                maxfun=params.maxfun,
                method=params.method,
                disp=0,
            )

            return SingleFitResult(
                count_model=count_model.value,
                n_components=n_components,
                amplitude=df_result['amplitude'],
                mesor=df_result['mesor'],
                peaks=df_result['peaks'],
                heights=df_result['heights'],
                llr_pvalue=df_result['llr_pvalue'],
                AIC=df_result['AIC'],
                BIC=df_result['BIC'],
                RSS=df_result['RSS'],
                log_likelihood=df_result['log_likelihood'],
                prsquared=df_result['prsquared'],
                resid_mean=df_result['resid_mean'],
                resid_std=df_result['resid_std'],
                data_mean=df_result['data_mean'],
                data_std=df_result['data_std'],
                X_test=df_result['X_test'],
                Y_test=df_result['Y_test'],
            )

        except Exception as e:
            return SingleFitResult(
                count_model=count_model.value,
                n_components=n_components,
                amplitude=float('nan'), mesor=float('nan'),
                peaks=np.array([]), heights=np.array([]),
                llr_pvalue=float('nan'), AIC=float('nan'),
                BIC=float('nan'), RSS=float('nan'),
                log_likelihood=float('nan'), prsquared=float('nan'),
                resid_mean=float('nan'), resid_std=float('nan'),
                data_mean=float('nan'), data_std=float('nan'),
                X_test=np.array([]), Y_test=np.array([]),
                success=False,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # METHOD 2: FIT ALL MODELS
    # -------------------------------------------------------------------------

    def fit_all_models(
        self,
        params: Optional[RhythmCountParameters] = None,
        plot_models: bool = False,
    ) -> pd.DataFrame:
        """
        Fit all combinations of count models and harmonic components.

        Args:
            params: Analysis parameters (models and n_components to try).
            plot_models: If True, display a comparison plot grid.

        Returns:
            DataFrame with one row per (count_model, n_components) combination,
            containing fit statistics and rhythm parameters.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if params is None:
            params = RhythmCountParameters()

        df = self._prepare_data(params)

        return dproc.fit_to_models(
            df,
            count_models=params.count_model_values(),
            n_components=params.n_components,
            maxiter=params.maxiter,
            maxfun=params.maxfun,
            method=params.method,
            plot_models=plot_models,
            period=params.period,
        )

    # -------------------------------------------------------------------------
    # METHOD 3: FIT BEST MODEL (automatic selection)
    # -------------------------------------------------------------------------

    def fit_best_model(
        self,
        params: Optional[RhythmCountParameters] = None,
    ) -> BestModelResult:
        """
        Fit all model combinations and automatically select the best one.

        Selection strategy (controlled by params.eval_order):
            - eval_order=True  (default): first pick best n_components, then
              best count distribution given that component count.
            - eval_order=False: first pick best distribution, then best
              n_components given that distribution.

        Args:
            params: Analysis parameters including selection criterion.

        Returns:
            BestModelResult with the winning model and full results table.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if params is None:
            params = RhythmCountParameters()

        df = self._prepare_data(params)
        test = params.selection_test.value

        try:
            all_results = dproc.fit_to_models(
                df,
                count_models=params.count_model_values(),
                n_components=params.n_components,
                maxiter=params.maxiter,
                maxfun=params.maxfun,
                method=params.method,
                plot_models=False,
                period=params.period,
            )

            if params.eval_order:
                best_component = dproc.get_best_n_components(all_results, test)
                best = dproc.get_best_count_model(
                    all_results, test,
                    n_components=int(best_component['n_components'])
                )
            else:
                best_count_model = dproc.get_best_count_model(all_results, test)
                best = dproc.get_best_n_components(
                    all_results, test,
                    count_model=best_count_model['count_model']
                )

            return BestModelResult(
                count_model=best['count_model'],
                n_components=int(best['n_components']),
                selection_test=test,
                amplitude=best['amplitude'],
                mesor=best['mesor'],
                peaks=best['peaks'],
                heights=best['heights'],
                llr_pvalue=best['llr_pvalue'],
                AIC=best['AIC'],
                BIC=best['BIC'],
                RSS=best['RSS'],
                prsquared=best['prsquared'],
                X_test=best['X_test'],
                Y_test=best['Y_test'],
                all_results=all_results,
            )

        except Exception as e:
            return BestModelResult(
                count_model='', n_components=0,
                selection_test=test,
                amplitude=float('nan'), mesor=float('nan'),
                peaks=np.array([]), heights=np.array([]),
                llr_pvalue=float('nan'), AIC=float('nan'),
                BIC=float('nan'), RSS=float('nan'),
                prsquared=float('nan'),
                X_test=np.array([]), Y_test=np.array([]),
                all_results=pd.DataFrame(),
                success=False,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # METHOD 4: PARAMETER CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------

    def calculate_parameter_cis(
        self,
        count_model: CountModel,
        n_components: int,
        reference_peaks: np.ndarray,
        params: Optional[RhythmCountParameters] = None,
    ) -> ParameterCIsResult:
        """
        Compute bootstrap confidence intervals for amplitude, mesor, and peaks.

        Uses bootstrap resampling: fits the model on random subsamples and
        derives 95% CIs from the distribution of estimated parameters.

        Args:
            count_model: The count distribution to use.
            n_components: Number of harmonic components.
            reference_peaks: Peak locations from the primary fit (used for
                             matching bootstrap peaks via tolerance window).
            params: Analysis parameters (repetitions, precision_rate, etc.).

        Returns:
            ParameterCIsResult with 95% CI arrays for each rhythm parameter.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if params is None:
            params = RhythmCountParameters()

        df = self._prepare_data(params)

        try:
            ci_result = dproc.calculate_confidence_intervals_parameters(
                df,
                n_components=n_components,
                count_model=count_model.value,
                all_peaks=reference_peaks,
                repetitions=params.repetitions,
                maxiter=params.maxiter,
                maxfun=params.maxfun,
                method=params.method,
                period=params.period,
                precision_rate=params.precision_rate,
            )

            return ParameterCIsResult(
                amplitude_CIs=ci_result['amplitude_CIs'],
                mesor_CIs=ci_result['mesor_CIs'],
                peaks_CIs=ci_result['peaks_CIs'],
                heights_CIs=ci_result['heights_CIs'],
                count_model=count_model.value,
                n_components=n_components,
                repetitions=params.repetitions,
            )

        except Exception as e:
            return ParameterCIsResult(
                amplitude_CIs=np.array([]), mesor_CIs=np.array([]),
                peaks_CIs=np.array([]), heights_CIs=np.array([]),
                count_model=count_model.value,
                n_components=n_components,
                repetitions=params.repetitions,
                success=False,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # METHOD 5: GROUP COMPARISON
    # -------------------------------------------------------------------------

    def compare_groups(
        self,
        group_column: str,
        count_models: Optional[List[CountModel]] = None,
        n_components: Optional[List[int]] = None,
        params: Optional[RhythmCountParameters] = None,
        ax_indices: Optional[List[int]] = None,
        ax_titles: Optional[List[str]] = None,
        labels: Optional[Dict] = None,
        plot_comparison: bool = True,
    ) -> GroupComparisonResult:
        """
        Fit and compare count-based cosinor models across groups/conditions.

        For each unique value in group_column, independently selects the best
        model and computes bootstrap CIs for rhythm parameters.

        Args:
            group_column: Column name identifying groups (e.g., 'condition').
            count_models: Models to try. Defaults to params.count_models.
            n_components: Components to try. Defaults to params.n_components.
            params: Analysis parameters.
            ax_indices: Subplot indices for the comparison figure (1-based).
            ax_titles: Subplot titles, one per group.
            labels: Dict mapping group names to display labels.
            plot_comparison: If True, display comparison plots.

        Returns:
            GroupComparisonResult with per-group fit statistics and CIs.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        if group_column not in self._data.columns:
            raise ValueError(f"Column '{group_column}' not found in data.")

        if params is None:
            params = RhythmCountParameters()

        df = self._prepare_data(params)
        groups = df[group_column].unique()
        n_groups = len(groups)

        if count_models is None:
            count_models = params.count_models
        if n_components is None:
            n_components = params.n_components

        if ax_indices is None:
            ax_indices = list(range(1, n_groups + 1))
        if ax_titles is None:
            ax_titles = [str(g) for g in groups]

        count_model_values = [m.value for m in count_models]
        test = params.selection_test.value

        try:
            if plot_comparison:
                rows = max(1, n_groups // 2)
                cols = max(1, (n_groups + 1) // 2)
                df_results = dproc.compare_by_component(
                    df,
                    component=group_column,
                    n_components=n_components,
                    count_models=count_model_values,
                    ax_indices=ax_indices,
                    ax_titles=ax_titles,
                    rows=rows,
                    cols=cols,
                    labels=labels,
                    eval_order=params.eval_order,
                    maxiter=params.maxiter,
                    maxfun=params.maxfun,
                    method=params.method,
                    period=params.period,
                    precision_rate=params.precision_rate,
                    repetitions=params.repetitions,
                    test=test,
                )
            else:
                # Run without plotting: fit and select best per group manually
                df_results = pd.DataFrame()
                for group_name in groups:
                    df_group = df[df[group_column] == group_name].copy()
                    all_results = dproc.fit_to_models(
                        df_group,
                        count_models=count_model_values,
                        n_components=n_components,
                        maxiter=params.maxiter,
                        maxfun=params.maxfun,
                        method=params.method,
                        plot_models=False,
                        period=params.period,
                    )

                    if params.eval_order:
                        best_nc = dproc.get_best_n_components(all_results, test)
                        best = dproc.get_best_count_model(
                            all_results, test,
                            n_components=int(best_nc['n_components'])
                        )
                    else:
                        best_cm = dproc.get_best_count_model(all_results, test)
                        best = dproc.get_best_n_components(
                            all_results, test,
                            count_model=best_cm['count_model']
                        )

                    ci_result = dproc.calculate_confidence_intervals_parameters(
                        df_group,
                        n_components=int(best['n_components']),
                        count_model=best['count_model'],
                        all_peaks=best['peaks'],
                        repetitions=params.repetitions,
                        maxiter=params.maxiter,
                        maxfun=params.maxfun,
                        method=params.method,
                        period=params.period,
                        precision_rate=params.precision_rate,
                    )

                    row = best.to_dict()
                    row[group_column] = group_name
                    row.update(ci_result)
                    df_results = pd.concat(
                        [df_results, pd.DataFrame([row])], ignore_index=True
                    )

            return GroupComparisonResult(
                group_column=group_column,
                results_table=df_results,
                selection_test=test,
            )

        except Exception as e:
            return GroupComparisonResult(
                group_column=group_column,
                results_table=pd.DataFrame(),
                selection_test=test,
                success=False,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # HELPER: format results for GUI display
    # -------------------------------------------------------------------------

    def format_single_result(self, result: SingleFitResult) -> Dict[str, Any]:
        """
        Format a SingleFitResult into a flat dict suitable for GUI display.

        Returns:
            Dict with human-readable keys and formatted values.
        """
        model_name = hlp.get_model_name(result.count_model)
        peaks_str = ', '.join([f'{p:.2f}h' for p in result.peaks]) if len(result.peaks) > 0 else 'N/A'

        return {
            'Model': model_name,
            'Components (N)': result.n_components,
            'Amplitude': f'{result.amplitude:.4f}',
            'MESOR': f'{result.mesor:.4f}',
            'Peak Time(s)': peaks_str,
            'LLR p-value': f'{result.llr_pvalue:.4e}' if not np.isnan(result.llr_pvalue) else 'N/A',
            'AIC': f'{result.AIC:.2f}' if not np.isnan(result.AIC) else 'N/A',
            'BIC': f'{result.BIC:.2f}' if not np.isnan(result.BIC) else 'N/A',
            'McFadden R²': f'{result.prsquared:.4f}' if not np.isnan(result.prsquared) else 'N/A',
            'Data Mean': f'{result.data_mean:.4f}' if not np.isnan(result.data_mean) else 'N/A',
            'Residual Mean': f'{result.resid_mean:.4f}' if not np.isnan(result.resid_mean) else 'N/A',
            'Success': result.success,
            'Error': result.error or '',
        }

    def format_best_model_result(self, result: BestModelResult) -> Dict[str, Any]:
        """
        Format a BestModelResult into a flat dict suitable for GUI display.

        Returns:
            Dict with selected model info plus selection criterion used.
        """
        model_name = hlp.get_model_name(result.count_model)
        peaks_str = ', '.join([f'{p:.2f}h' for p in result.peaks]) if len(result.peaks) > 0 else 'N/A'

        return {
            'Best Model': model_name,
            'Components (N)': result.n_components,
            'Selection Criterion': result.selection_test,
            'Amplitude': f'{result.amplitude:.4f}',
            'MESOR': f'{result.mesor:.4f}',
            'Peak Time(s)': peaks_str,
            'LLR p-value': f'{result.llr_pvalue:.4e}' if not np.isnan(result.llr_pvalue) else 'N/A',
            'AIC': f'{result.AIC:.2f}' if not np.isnan(result.AIC) else 'N/A',
            'BIC': f'{result.BIC:.2f}' if not np.isnan(result.BIC) else 'N/A',
            'McFadden R²': f'{result.prsquared:.4f}' if not np.isnan(result.prsquared) else 'N/A',
            'Models Evaluated': len(result.all_results) if not result.all_results.empty else 0,
            'Success': result.success,
            'Error': result.error or '',
        }

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _prepare_data(self, params: RhythmCountParameters) -> pd.DataFrame:
        """Apply optional cleaning and return a copy of the data."""
        df = self._data.copy()
        if params.clean_data:
            df = dproc.clean_data(df)
        return df
