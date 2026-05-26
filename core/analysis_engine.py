"""
Analysis Engine
================

Central engine that connects the GUI to the analysis modules.
Handles data preparation, method execution, and result formatting.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import os
from pathlib import Path

import pandas as pd
import numpy as np

# Import analysis modules
from .cosinor_analysis import (
    CosinorAnalyzer, CosinorParameters, DataType, ModelType, AnalysisMethod as CosinorMethod, Criterium,
    COSINORPY_AVAILABLE
)
from .circacompare_analysis import (
    CircaCompareAnalyzer, CircaSingleResult, CircaCompareResult
)
from .rhythm_analysis import (
    RhythmAnalyzer, AnalysisMethod as RhythmMethod
)
from .meta_classifier import ConsensusClassifier, SKLEARN_AVAILABLE
from .rhythmcount_analysis import (
    RhythmCountAnalyzer, RhythmCountParameters, CountModel, SelectionTest,
    RHYTHMCOUNT_AVAILABLE
)


def get_cosinorpy_plot_folder(data_file_path: Optional[str]) -> Optional[str]:
    """
    Get the folder path for saving CosinorPy plots.

    Args:
        data_file_path: Path to the data file

    Returns:
        Path to cosinorpy_plots folder, or None if no data file path
    """
    if data_file_path is None:
        return None

    # Get the directory containing the data file
    data_dir = Path(data_file_path).parent

    # Create cosinorpy_plots subdirectory
    plot_folder = data_dir / "cosinorpy_plots"
    plot_folder.mkdir(exist_ok=True)

    return str(plot_folder)


class AnalysisType(Enum):
    """Types of analysis available."""
    # CosinorPy - New Refactored Methods
    # 1. Periodogram
    COSINORPY_PERIODOGRAM = "cosinorpy_periodogram"

    # 2. Cosinor Analysis (Independent Data)
    COSINORPY_INDEPENDENT = "cosinorpy_independent"

    # 3. Cosinor Analysis (Dependent/Population Data)
    COSINORPY_DEPENDENT = "cosinorpy_dependent"

    # 4. Compare Conditions (Independent Data)
    COSINORPY_COMPARE_INDEPENDENT = "cosinorpy_compare_independent"

    # 5. Compare Conditions (Dependent/Population Data)
    COSINORPY_COMPARE_DEPENDENT = "cosinorpy_compare_dependent"

    # 6. Non-Linear Analysis (Independent Data)
    COSINORPY_NONLINEAR_INDEPENDENT = "cosinorpy_nonlinear_independent"

    # 7. Non-Linear Analysis (Dependent/Population Data)
    COSINORPY_NONLINEAR_DEPENDENT = "cosinorpy_nonlinear_dependent"

    # 8. Non-Linear Compare Conditions (Independent Data)
    COSINORPY_NONLINEAR_COMPARE_INDEPENDENT = "cosinorpy_nonlinear_compare_independent"

    # 9. Non-Linear Compare Conditions (Dependent/Population Data)
    COSINORPY_NONLINEAR_COMPARE_DEPENDENT = "cosinorpy_nonlinear_compare_dependent"

    # CircaCompare
    CIRCACOMPARE_SINGLE = "circacompare_single"
    CIRCACOMPARE_COMPARE = "circacompare_compare"

    # Rhythm Analysis
    JTK = "jtk"
    AR_JTK = "ar_jtk"
    COSINE_KENDALL = "cosine_kendall"
    COSINOR_OLS = "cosinor_ols"
    HARMONIC_COSINOR = "harmonic_cosinor"
    FOURIER_F24 = "fourier_f24"
    LOMB_SCARGLE = "lomb_scargle"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    CWT = "cwt"
    LME = "lme"

    # AI Meta-Classifier
    CONSENSUS_AI = "consensus_ai"

    # RhythmCount
    RHYTHMCOUNT_SINGLE = "rhythmcount_single"
    RHYTHMCOUNT_ALL_MODELS = "rhythmcount_all_models"
    RHYTHMCOUNT_BEST_MODEL = "rhythmcount_best_model"
    RHYTHMCOUNT_PARAMETER_CIS = "rhythmcount_parameter_cis"
    RHYTHMCOUNT_COMPARE_GROUPS = "rhythmcount_compare_groups"


@dataclass
class AnalysisResult:
    """Standardized analysis result."""
    # Identification
    variable: str
    condition: str
    method: str
    
    # Core parameters
    mesor: Optional[float] = None
    amplitude: Optional[float] = None
    acrophase: Optional[float] = None  # in radians
    acrophase_hours: Optional[float] = None
    period: float = 24.0
    
    # Statistics
    p_value: Optional[float] = None
    q_value: Optional[float] = None  # FDR-corrected p-value
    p_reject: Optional[bool] = None  # Whether p-value indicates rejection
    q_reject: Optional[bool] = None  # Whether q-value indicates rejection
    p_amplitude: Optional[float] = None
    p_acrophase: Optional[float] = None
    p_mesor: Optional[float] = None  # p-value for MESOR parameter
    q_amplitude: Optional[float] = None  # FDR-corrected p-value for amplitude
    q_acrophase: Optional[float] = None  # FDR-corrected p-value for acrophase
    q_mesor: Optional[float] = None  # FDR-corrected p-value for MESOR
    r_squared: Optional[float] = None
    r_squared_adj: Optional[float] = None  # Adjusted R²
    rss: Optional[float] = None  # Residual Sum of Squares
    log_likelihood: Optional[float] = None  # Log-likelihood
    me: Optional[float] = None  # Model Error (ME) - for population/dependent data
    resid_se: Optional[float] = None  # Residual Standard Error - for population/dependent data
    aic: Optional[float] = None  # Akaike Information Criterion
    bic: Optional[float] = None  # Bayesian Information Criterion

    # Confidence intervals
    mesor_ci: Optional[Tuple[float, float]] = None
    amplitude_ci: Optional[Tuple[float, float]] = None
    acrophase_ci: Optional[Tuple[float, float]] = None

    # Peaks and troughs
    peak_times: Optional[List[float]] = None  # Times of rhythm peaks (hours)
    trough_times: Optional[List[float]] = None  # Times of rhythm troughs (hours)
    
    # Method-specific
    tau: Optional[float] = None  # JTK Kendall tau
    bonf_p_value: Optional[float] = None  # Bonferroni-corrected p-value (JTK)
    raw_p_value: Optional[float] = None   # Raw (uncorrected) p-value (JTK)
    lag: Optional[float] = None           # Phase lag in hours (JTK/Cosine-Kendall)
    asymmetry: Optional[float] = None     # Waveform asymmetry parameter (JTK)
    n_tests: Optional[int] = None         # Number of hypotheses tested (JTK)
    power: Optional[float] = None  # Lomb-Scargle
    dominant_period: Optional[float] = None
    dominant_power: Optional[float] = None  # Fourier F24
    target_power: Optional[float] = None    # Fourier F24
    correlation_r: Optional[float] = None   # Fourier F24 (replicate correlation)
    n_components: Optional[int] = None  # Multi-component cosinor
    amplification: Optional[float] = None  # Nonlinear cosinor - damping/forcing coefficient
    lin_comp: Optional[float] = None  # Nonlinear cosinor - linear trend component
    p_amplification: Optional[float] = None  # P-value for amplification
    p_lin_comp: Optional[float] = None  # P-value for linear component
    q_amplification: Optional[float] = None  # FDR-corrected p-value for amplification
    q_lin_comp: Optional[float] = None  # FDR-corrected p-value for linear component
    amplification_ci: Optional[Tuple[float, float]] = None  # CI for amplification
    lin_comp_ci: Optional[Tuple[float, float]] = None  # CI for linear component

    # Periodogram-specific
    periods: Optional[np.ndarray] = None  # Array of period values
    power_spectrum: Optional[np.ndarray] = None  # Power spectral density
    threshold: Optional[float] = None  # Significance threshold
    significant_peaks: Optional[List[float]] = None  # List of significant periods

    # Scalogram-specific (CWT)
    scalogram_power: Optional[np.ndarray] = None  # 2D power matrix (periods x time)
    scalogram_times: Optional[np.ndarray] = None  # Time array for scalogram
    scalogram_periods: Optional[np.ndarray] = None  # Period array for scalogram
    period_variation: Optional[float] = None  # CWT period variation
    amplitude_modulations: Optional[int] = None  # CWT amplitude modulation count

    # LME-specific
    random_effect_var: Optional[float] = None  # Between-group (random intercept) variance
    residual_var: Optional[float] = None        # Within-group residual variance

    # CircaCompare-specific (standard errors)
    se_mesor: Optional[float] = None
    se_amplitude: Optional[float] = None
    se_acrophase: Optional[float] = None

    # Raw data for plotting
    times: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    
    # Status
    success: bool = True
    message: str = ""

    # Best model indicator (for multiple period testing)
    best_model: Optional[str] = None  # e.g., "Yes (min p-value)", "No", None if not applicable

    # Serialized results table (for methods that return multiple model comparisons)
    results_table_json: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, converting numpy arrays to lists."""
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, np.ndarray):
                # Convert numpy arrays to lists for JSON serialization
                result[key] = value.tolist()
            else:
                result[key] = value
        return result


@dataclass
class ComparisonResult:
    """Result from comparing two conditions."""
    variable: str
    condition1: str
    condition2: str
    method: str

    # Group 0 (condition1) parameters
    mesor_g0: Optional[float] = None
    amplitude_g0: Optional[float] = None
    acrophase_g0: Optional[float] = None

    # Group 1 (condition2) parameters
    mesor_g1: Optional[float] = None
    amplitude_g1: Optional[float] = None
    acrophase_g1: Optional[float] = None

    # Comparison p-values
    p_mesor: Optional[float] = None
    p_amplitude: Optional[float] = None
    p_acrophase: Optional[float] = None

    # CircaCompare CI-based significance ("Yes" / "No")
    sig_mesor: Optional[str] = None
    sig_amplitude: Optional[str] = None
    sig_acrophase: Optional[str] = None

    # Comparison q-values (FDR-corrected)
    q_mesor: Optional[float] = None
    q_amplitude: Optional[float] = None
    q_acrophase: Optional[float] = None

    # Population-specific p/q values (for dependent data multi-component)
    # These represent the individual rhythm detection p/q values for each condition
    p1: Optional[float] = None  # Condition 1 rhythm detection p-value
    p2: Optional[float] = None  # Condition 2 rhythm detection p-value
    q1: Optional[float] = None  # Condition 1 rhythm detection q-value
    q2: Optional[float] = None  # Condition 2 rhythm detection q-value

    # Acrophase in hours (CircaCompare)
    acrophase_g0_hours: Optional[float] = None
    acrophase_g1_hours: Optional[float] = None

    # Differences
    mesor_diff: Optional[float] = None
    amplitude_diff: Optional[float] = None
    acrophase_diff: Optional[float] = None
    acrophase_diff_hours: Optional[float] = None

    # Confidence intervals for differences (tuple of lower, upper)
    amplitude_diff_ci: Optional[Tuple[float, float]] = None
    acrophase_diff_ci: Optional[Tuple[float, float]] = None
    mesor_diff_ci: Optional[Tuple[float, float]] = None

    # Nonlinear cosinor-specific fields (for generalized cosinor comparison)
    amplification_g0: Optional[float] = None
    amplification_g1: Optional[float] = None
    amplification_diff: Optional[float] = None
    amplification_diff_ci: Optional[Tuple[float, float]] = None
    p_amplification: Optional[float] = None
    q_amplification: Optional[float] = None

    lin_comp_g0: Optional[float] = None
    lin_comp_g1: Optional[float] = None
    lin_comp_diff: Optional[float] = None
    lin_comp_diff_ci: Optional[Tuple[float, float]] = None
    p_lin_comp: Optional[float] = None
    q_lin_comp: Optional[float] = None

    period: float = 24.0
    n_components: Optional[int] = None

    # Raw data for plotting (optional)
    times_g0: Optional[np.ndarray] = None
    values_g0: Optional[np.ndarray] = None
    times_g1: Optional[np.ndarray] = None
    values_g1: Optional[np.ndarray] = None

    success: bool = True
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AnalysisEngine:
    """
    Central analysis engine for ChronoScope.
    
    Provides a unified interface to all analysis methods and handles
    data preparation, execution, and result formatting.
    """
    
    def __init__(self):
        """Initialize the analysis engine."""
        self._cosinor = CosinorAnalyzer() if COSINORPY_AVAILABLE else None
        self._circacompare = CircaCompareAnalyzer()
        self._rhythm = RhythmAnalyzer()
        self._consensus = ConsensusClassifier() if SKLEARN_AVAILABLE else None
        self._rhythmcount = RhythmCountAnalyzer() if RHYTHMCOUNT_AVAILABLE else None
    
    def check_method_available(self, analysis_type: AnalysisType) -> Tuple[bool, str]:
        """
        Check if an analysis method is available.
        
        Returns:
            Tuple of (is_available, message)
        """
        if analysis_type.value.startswith('cosinorpy'):
            if not COSINORPY_AVAILABLE:
                return False, "CosinorPy not installed. Install with: pip install cosinorpy"
        
        if analysis_type == AnalysisType.CWT:
            try:
                import pywt
            except ImportError:
                return False, "PyWavelets not installed. Install with: pip install PyWavelets"

        if analysis_type == AnalysisType.CONSENSUS_AI:
            if not SKLEARN_AVAILABLE:
                return False, "scikit-learn not installed. Install with: pip install scikit-learn"
            if self._consensus and not self._consensus.is_loaded:
                success = self._consensus.load_model()
                if not success:
                    return False, "AI model file not found. Run train_consensus_model.py first."

        if analysis_type.value.startswith('rhythmcount'):
            if not RHYTHMCOUNT_AVAILABLE:
                return False, "RhythmCount dependencies not available. Install statsmodels: pip install statsmodels"

        return True, "Available"
    
    def run_analysis(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        analysis_type: AnalysisType,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None,
        data_file_path: Optional[str] = None
    ) -> AnalysisResult:
        """
        Run a single analysis on one variable/condition combination.

        Args:
            data: DataFrame with the data
            variable: Column name of the variable to analyze
            condition: Condition value to filter for
            analysis_type: Type of analysis to run
            time_col: Name of the time column
            condition_col: Name of the condition column
            parameters: Analysis-specific parameters
            data_file_path: Optional path to the data file for saving plots

        Returns:
            AnalysisResult object
        """
        print(f"[DEBUG] run_analysis called with analysis_type: {analysis_type}")
        print(f"[DEBUG] Variable: {variable}, Condition: {condition}")

        parameters = parameters or {}

        # Route to new refactored CosinorPy methods FIRST (they handle data filtering internally)
        if analysis_type == AnalysisType.COSINORPY_PERIODOGRAM:
            return self._run_cosinorpy_periodogram_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_INDEPENDENT:
            return self._run_cosinorpy_independent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_DEPENDENT:
            return self._run_cosinorpy_dependent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_COMPARE_INDEPENDENT:
            return self._run_cosinorpy_compare_independent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_COMPARE_DEPENDENT:
            return self._run_cosinorpy_compare_dependent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_NONLINEAR_INDEPENDENT:
            return self._run_cosinorpy_nonlinear_independent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_NONLINEAR_DEPENDENT:
            return self._run_cosinorpy_nonlinear_dependent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT:
            return self._run_cosinorpy_nonlinear_compare_independent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        elif analysis_type == AnalysisType.COSINORPY_NONLINEAR_COMPARE_DEPENDENT:
            return self._run_cosinorpy_nonlinear_compare_dependent_new(
                data, variable, condition, time_col, condition_col, parameters, data_file_path
            )

        # AI Meta-Classifier (handles data filtering internally via feature extraction)
        elif analysis_type == AnalysisType.CONSENSUS_AI:
            return self._run_consensus_ai(
                data, variable, condition, time_col, condition_col, parameters
            )

        # RhythmCount methods (handle their own data filtering and column renaming)
        elif analysis_type in (
            AnalysisType.RHYTHMCOUNT_SINGLE,
            AnalysisType.RHYTHMCOUNT_ALL_MODELS,
            AnalysisType.RHYTHMCOUNT_BEST_MODEL,
            AnalysisType.RHYTHMCOUNT_PARAMETER_CIS,
            AnalysisType.RHYTHMCOUNT_COMPARE_GROUPS,
        ):
            return self._run_rhythmcount(
                data, variable, condition, analysis_type,
                time_col, condition_col, parameters
            )

        # For other methods, filter data by variable and condition
        period = parameters.get('period', 24.0)

        # Filter data
        mask = data[condition_col] == condition
        filtered = data[mask].copy()

        if len(filtered) == 0:
            return AnalysisResult(
                variable=variable,
                condition=condition,
                method=analysis_type.value,
                success=False,
                message=f"No data for condition: {condition}"
            )

        times = filtered[time_col].values.astype(float)
        values = filtered[variable].values.astype(float)

        # Remove NaN values
        valid_mask = ~(np.isnan(times) | np.isnan(values))
        times = times[valid_mask]
        values = values[valid_mask]

        if len(times) < 4:
            return AnalysisResult(
                variable=variable,
                condition=condition,
                method=analysis_type.value,
                success=False,
                message="Insufficient data points (minimum 4 required)"
            )

        try:
            # Route to appropriate method (OLD methods and other analysis types)
            # CircaCompare
            if analysis_type == AnalysisType.CIRCACOMPARE_SINGLE:
                return self._run_circacompare_single(
                    times, values, variable, condition, parameters
                )

            # Rhythm Analysis Methods
            elif analysis_type == AnalysisType.JTK:
                return self._run_jtk(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.COSINOR_OLS:
                return self._run_cosinor_ols(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.HARMONIC_COSINOR:
                return self._run_harmonic_cosinor(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.LOMB_SCARGLE:
                return self._run_lomb_scargle(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.AR_JTK:
                return self._run_ar_jtk(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.COSINE_KENDALL:
                return self._run_cosine_kendall(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.FOURIER_F24:
                return self._run_fourier_f24(
                    data, variable, condition, time_col, condition_col, parameters
                )

            elif analysis_type == AnalysisType.SPECTRAL_ANALYSIS:
                return self._run_spectral_analysis(
                    times, values, variable, condition, parameters
                )

            elif analysis_type == AnalysisType.CWT:
                return self._run_cwt(
                    data, variable, condition, time_col, condition_col, parameters
                )

            elif analysis_type == AnalysisType.LME:
                return self._run_lme(
                    data, variable, condition, time_col, condition_col, parameters
                )

            else:
                return AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method=analysis_type.value,
                    success=False,
                    message=f"Method not yet implemented: {analysis_type.value}"
                )
        
        except Exception as e:
            import traceback
            print(f"[ERROR] run_analysis exception ({analysis_type.value}, {variable}, {condition}): {e}")
            traceback.print_exc()
            return AnalysisResult(
                variable=variable,
                condition=condition,
                method=analysis_type.value,
                success=False,
                message=str(e)
            )

    def run_comparison(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        analysis_type: AnalysisType,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None,
        data_file_path: Optional[str] = None
    ) -> Union[ComparisonResult, List[ComparisonResult]]:
        """
        Run a comparison analysis between two conditions.

        Args:
            data: DataFrame with the data
            variable: Column name of the variable
            condition1: First condition (reference)
            condition2: Second condition (comparison)
            analysis_type: Type of comparison analysis
            time_col: Name of time column
            condition_col: Name of condition column
            parameters: Analysis parameters
            data_file_path: Optional path to the data file for saving plots

        Returns:
            ComparisonResult object
        """
        parameters = parameters or {}
        period = parameters.get('period', 24.0)
        
        try:
            if analysis_type == AnalysisType.CIRCACOMPARE_COMPARE:
                return self._run_circacompare_comparison(
                    data, variable, condition1, condition2,
                    time_col, condition_col, parameters
                )
            
            else:
                return ComparisonResult(
                    variable=variable,
                    condition1=condition1,
                    condition2=condition2,
                    method=analysis_type.value,
                    success=False,
                    message=f"Comparison not supported for: {analysis_type.value}"
                )
        
        except Exception as e:
            print(f"[DEBUG] Exception in run_comparison: {e}")
            import traceback
            traceback.print_exc()
            return ComparisonResult(
                variable=variable,
                condition1=condition1,
                condition2=condition2,
                method=analysis_type.value,
                success=False,
                message=str(e)
            )

    def run_multi_comparison(
        self,
        data: pd.DataFrame,
        variable: str,
        analysis_type: AnalysisType,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None,
        data_file_path: Optional[str] = None
    ) -> List[ComparisonResult]:
        """
        Run comparison analysis across ALL conditions for a variable.

        This performs pairwise comparisons between all combinations of conditions.

        Args:
            data: DataFrame with the data
            variable: Column name of the variable
            analysis_type: Type of comparison analysis
            time_col: Name of time column
            condition_col: Name of condition column
            parameters: Analysis parameters
            data_file_path: Optional path to the data file for saving plots

        Returns:
            List of ComparisonResult objects (one per pair)
        """
        parameters = parameters or {}

        try:
            # NOTE: This method is obsolete in the refactored architecture.
            # The new comparison methods (COSINORPY_COMPARE_INDEPENDENT, COSINORPY_COMPARE_DEPENDENT)
            # handle pair generation automatically within their implementations.
            # This method is kept for backward compatibility but should not be called.
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method=analysis_type.value,
                success=False,
                message=f"run_multi_comparison is obsolete. Use run_comparison or run_analysis with new comparison methods."
            )]

        except Exception as e:
            import traceback
            traceback.print_exc()
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method=analysis_type.value,
                success=False,
                message=f"Error: {str(e)}"
            )]

    # =========================================================================
    # CosinorPy Methods
    # =========================================================================
    
    def _run_cosinorpy_single(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> AnalysisResult:
        """Run CosinorPy single-component cosinor."""
        print(f"[DEBUG] _run_cosinorpy_single called")
        print(f"[DEBUG] self._cosinor is None: {self._cosinor is None}")

        # Get folder for saving plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        if plot_folder:
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_single",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        print(f"[DEBUG] _run_cosinorpy_single - Period received from parameters: {period}")
        print(f"[DEBUG] _run_cosinorpy_single - Full parameters: {parameters}")

        # Prepare data - CosinorAnalyzer expects data to be loaded first
        print(f"[DEBUG] Preparing DataFrame...")
        df = pd.DataFrame({
            'time': times,
            'condition': condition,
            variable: values
        })
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")

        # Load data into CosinorAnalyzer and run analysis
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            # Import AnalysisMode
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes
            self._cosinor._raw_data = df
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition]
            self._cosinor._time_col = 'time'
            self._cosinor._condition_col = 'condition'
            self._cosinor._replicate_col = None  # No explicit replicate column
            self._cosinor._subject_col = None  # Not population data
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Converted DataFrame columns: {self._cosinor._cosinorpy_df.columns.tolist()}")
            print(f"[DEBUG] Converted DataFrame head:\n{self._cosinor._cosinorpy_df.head(10)}")
            print(f"[DEBUG] Data loaded successfully")

            print(f"[DEBUG] Setting period...")
            self._cosinor.set_period(period)
            print(f"[DEBUG] Period set successfully")

            print(f"[DEBUG] Calling self._cosinor.single_cosinor()...")
            result = self._cosinor.single_cosinor(
                variable=variable,
                condition=condition,
                period=period,
                save_folder=plot_folder
            )
            print(f"[DEBUG] single_cosinor returned successfully")
        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_single",
                success=False, message=f"Error: {str(e)}"
            )

        print(f"[DEBUG] CosinorPy single_cosinor result: {result}")
        print(f"[DEBUG] Result type: {type(result)}")
        if result is not None:
            print(f"[DEBUG] Result keys: {result.keys() if hasattr(result, 'keys') else 'N/A'}")

        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_single",
                success=False, message="Analysis failed"
            )

        # Extract parameters from CosinorParameters object
        # result is a CosinorParameters dataclass, access attributes directly
        acrophase_rad = result.acrophase
        acrophase_hours = result.acrophase_hours  # Already calculated by CosinorAnalyzer

        # Get confidence intervals if available
        amplitude_ci = None
        acrophase_ci = None
        if result.confidence_intervals:
            amplitude_ci = result.confidence_intervals.get('amplitude')
            acrophase_ci = result.confidence_intervals.get('acrophase')

        print(f"[DEBUG] Extracted values - mesor: {result.mesor}, amplitude: {result.amplitude}, p_value: {result.p_value}")

        # Generate phase plot if we have a valid acrophase and save folder
        if plot_folder and result.acrophase is not None and result.amplitude is not None:
            test_name = f"{variable}_{condition}"
            CI_acrs = [acrophase_ci] if acrophase_ci else []
            CI_amps = [amplitude_ci] if amplitude_ci else []

            self._cosinor._generate_phase_plot(
                acrophases=[result.acrophase],
                amplitudes=[result.amplitude],
                tests=[test_name],
                save_folder=plot_folder,
                prefix="single_",
                period=result.period,
                CI_acrs=CI_acrs,
                CI_amps=CI_amps
            )

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="cosinorpy_single",
            mesor=result.mesor,
            amplitude=result.amplitude,
            acrophase=acrophase_rad,
            acrophase_hours=acrophase_hours,
            period=result.period,
            p_value=result.p_value,
            r_squared=result.r_squared,
            rss=result.rss,
            log_likelihood=result.log_likelihood,
            peak_times=result.peak_times,
            trough_times=result.trough_times,
            amplitude_ci=amplitude_ci,
            acrophase_ci=acrophase_ci,
            times=times,
            values=values,
            success=True
        )
    
    def _run_cosinorpy_multi(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> AnalysisResult:
        """Run CosinorPy multi-component cosinor."""
        print(f"[DEBUG] _run_cosinorpy_multi called")
        print(f"[DEBUG] self._cosinor is None: {self._cosinor is None}")

        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_multi",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 2)
        auto_components = parameters.get('auto_components', False)
        print(f"[DEBUG] Period: {period}, n_components: {n_components}, auto_components: {auto_components}")

        # Get plot folder for CosinorPy plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        print(f"[DEBUG] Plot folder: {plot_folder}")

        # Prepare data - CosinorAnalyzer expects data to be loaded first
        print(f"[DEBUG] Preparing DataFrame...")
        df = pd.DataFrame({
            'time': times,
            'condition': condition,
            variable: values
        })
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        print(f"[DEBUG] DataFrame columns: {df.columns.tolist()}")

        # Load data into CosinorAnalyzer
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            # Import AnalysisMode
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes
            self._cosinor._raw_data = df
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition]
            self._cosinor._time_col = 'time'
            self._cosinor._condition_col = 'condition'
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Data loaded successfully")

            print(f"[DEBUG] Setting period and n_components...")
            self._cosinor.set_period(period)
            self._cosinor.n_components = n_components
            print(f"[DEBUG] Parameters set successfully")

            # Check if auto-select components is enabled
            if auto_components:
                print(f"[DEBUG] Auto-select components enabled, calling get_best_models()...")
                # Test with 1, 2, 3 components and select best
                df_best = self._cosinor.get_best_models(
                    n_components=[1, 2, 3],
                    period=period,
                    criterium='p'  # Use p-value as selection criterion
                )
                print(f"[DEBUG] get_best_models() result:\n{df_best}")

                # Extract the selected number of components
                if len(df_best) > 0:
                    n_components_selected = int(df_best['n_components'].values[0])
                    print(f"[DEBUG] Best model has {n_components_selected} components")

                    # Use the raw_results from get_best_models for result extraction
                    result = {
                        'test_name': f"{variable}_{condition}",
                        'n_components': n_components_selected,
                        'period': period,
                        'p_value': df_best['p'].values[0] if 'p' in df_best.columns else None,
                        'raw_results': df_best
                    }
                else:
                    print(f"[DEBUG] get_best_models() returned empty DataFrame")
                    result = None
            else:
                print(f"[DEBUG] Calling self._cosinor.multi_cosinor()...")
                result = self._cosinor.multi_cosinor(
                    variable=variable,
                    condition=condition,
                    n_components=n_components,
                    period=period,
                    save_folder=plot_folder
                )

            print(f"[DEBUG] Analysis completed")
            print(f"[DEBUG] Result: {result}")
        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy multi: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_multi",
                success=False, message=f"Error: {str(e)}"
            )

        if result is None or 'raw_results' not in result or result['raw_results'] is None:
            print(f"[DEBUG] Result check failed - result: {result}")
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_multi",
                success=False, message="Analysis failed"
            )

        # Extract results from the raw_results DataFrame
        # Multi-component results contain amplitude, acrophase for each component
        # We'll extract the first (dominant) component's parameters
        df_results = result['raw_results']
        print(f"[DEBUG] Raw results DataFrame type: {type(df_results)}")
        print(f"[DEBUG] Raw results DataFrame shape: {df_results.shape if hasattr(df_results, 'shape') else 'N/A'}")
        print(f"[DEBUG] Raw results DataFrame columns: {df_results.columns.tolist() if hasattr(df_results, 'columns') else 'N/A'}")
        print(f"[DEBUG] Raw results DataFrame:\n{df_results}")

        # Extract dominant period results (first row/component)
        if len(df_results) > 0:
            # CosinorPy multi-component returns: amplitude, acrophase, mesor, p-value
            amplitude = df_results['amplitude'].values[0] if 'amplitude' in df_results.columns else None
            acrophase_rad = df_results['acrophase'].values[0] if 'acrophase' in df_results.columns else None
            mesor = df_results['mesor'].values[0] if 'mesor' in df_results.columns else None
            p_value = df_results['p'].values[0] if 'p' in df_results.columns else None
            n_components_result = df_results['n_components'].values[0] if 'n_components' in df_results.columns else None

            # Convert to int if possible
            if n_components_result is not None:
                n_components_result = int(n_components_result)

            # Convert acrophase to hours if available
            acrophase_hours = None
            if acrophase_rad is not None and period is not None:
                # CosinorPy acrophase is already in radians, convert to hours
                acrophase_hours = (-acrophase_rad * period / (2 * np.pi)) % period

            print(f"[DEBUG] Extracted - mesor: {mesor}, amplitude: {amplitude}, acrophase_rad: {acrophase_rad}, acrophase_hours: {acrophase_hours}, p_value: {p_value}, n_components: {n_components_result}")

            # Generate phase plot if we have valid acrophase and save folder
            if plot_folder and acrophase_rad is not None and amplitude is not None:
                test_name = f"{variable}_{condition}"

                # Extract confidence intervals if available
                CI_acrs = []
                CI_amps = []
                if 'CI(acrophase)' in df_results.columns:
                    ci_acr = df_results['CI(acrophase)'].values[0]
                    if ci_acr is not None and isinstance(ci_acr, (list, tuple)) and len(ci_acr) == 2:
                        CI_acrs = [ci_acr]

                if 'CI(amplitude)' in df_results.columns:
                    ci_amp = df_results['CI(amplitude)'].values[0]
                    if ci_amp is not None and isinstance(ci_amp, (list, tuple)) and len(ci_amp) == 2:
                        CI_amps = [ci_amp]

                print(f"[DEBUG] Generating phase plot for Multi-Component: {test_name}")
                self._cosinor._generate_phase_plot(
                    acrophases=[acrophase_rad],
                    amplitudes=[amplitude],
                    tests=[test_name],
                    save_folder=plot_folder,
                    prefix="multi_",
                    period=period,
                    CI_acrs=CI_acrs if CI_acrs else None,
                    CI_amps=CI_amps if CI_amps else None
                )

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cosinorpy_multi",
                mesor=mesor,
                amplitude=amplitude,
                acrophase=acrophase_rad,
                acrophase_hours=acrophase_hours,
                period=period,
                p_value=p_value,
                dominant_period=period,  # For multi-component, base period
                n_components=n_components_result,  # Number of components in the model
                times=times,
                values=values,
                success=True
            )
        else:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_multi",
                success=False, message="No results returned"
            )
    
    def _run_cosinorpy_comparison(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> ComparisonResult:
        """Run CosinorPy differential rhythmicity."""
        print(f"[DEBUG] _run_cosinorpy_comparison called")
        print(f"[DEBUG] self._cosinor is None: {self._cosinor is None}")

        # Get folder for saving plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        if plot_folder:
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        if self._cosinor is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)
        print(f"[DEBUG] Period: {period}, n_components: {n_components}")

        # Filter for both conditions
        mask = data[condition_col].isin([condition1, condition2])
        filtered = data[mask].copy()
        print(f"[DEBUG] Filtered data shape: {filtered.shape}")

        # Extract raw data for each group (for plotting)
        data_g0 = filtered[filtered[condition_col] == condition1]
        data_g1 = filtered[filtered[condition_col] == condition2]
        times_g0 = data_g0[time_col].values.astype(float) if len(data_g0) > 0 else None
        values_g0 = data_g0[variable].values.astype(float) if len(data_g0) > 0 else None
        times_g1 = data_g1[time_col].values.astype(float) if len(data_g1) > 0 else None
        values_g1 = data_g1[variable].values.astype(float) if len(data_g1) > 0 else None

        # Load data into CosinorAnalyzer
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes
            self._cosinor._raw_data = filtered
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition1, condition2]
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Data loaded successfully")

            print(f"[DEBUG] Setting period...")
            self._cosinor.set_period(period)
            print(f"[DEBUG] Parameters set successfully")

            print(f"[DEBUG] Calling self._cosinor.compare_conditions()...")
            result = self._cosinor.compare_conditions(
                variable=variable,
                condition1=condition1,
                condition2=condition2,
                period=period,
                n_components=n_components,
                save_folder=plot_folder
            )
            print(f"[DEBUG] compare_conditions() returned successfully")
            print(f"[DEBUG] Result: {result}")
        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy comparison: {e}")
            import traceback
            traceback.print_exc()
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare",
                success=False, message=f"Error: {str(e)}"
            )

        if result is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare",
                success=False, message="Analysis failed"
            )

        # Extract results from DifferentialResult object
        print(f"[DEBUG] Extracting comparison results...")

        # Extract q-values if available
        q_amplitude = None
        q_acrophase = None
        q_mesor = None
        if result.q_values:
            q_amplitude = result.q_values.get('amplitude')
            q_acrophase = result.q_values.get('acrophase')
            q_mesor = result.q_values.get('mesor')

        # Generate phase plot if we have valid acrophases and amplitudes for both groups
        if plot_folder and result.acrophase_g0 is not None and result.acrophase_g1 is not None:
            if result.amplitude_g0 is not None and result.amplitude_g1 is not None:
                test_name_g0 = f"{variable}_{condition1}"
                test_name_g1 = f"{variable}_{condition2}"

                # Extract confidence intervals if available
                CI_acrs = []
                CI_amps = []

                # Note: result has CI for differences, not for individual groups
                # We'll generate the plot without individual CIs for now

                print(f"[DEBUG] Generating phase plot for Compare Conditions: {condition1} vs {condition2}")
                self._cosinor._generate_phase_plot(
                    acrophases=[result.acrophase_g0, result.acrophase_g1],
                    amplitudes=[result.amplitude_g0, result.amplitude_g1],
                    tests=[test_name_g0, test_name_g1],
                    save_folder=plot_folder,
                    prefix="compare_",
                    period=period,
                    CI_acrs=None,  # Individual group CIs not available from comparison
                    CI_amps=None   # Individual group CIs not available from comparison
                )

        return ComparisonResult(
            variable=variable,
            condition1=condition1,
            condition2=condition2,
            method="cosinorpy_compare",
            mesor_g0=result.mesor_g0,
            mesor_g1=result.mesor_g1,
            amplitude_g0=result.amplitude_g0,
            amplitude_g1=result.amplitude_g1,
            acrophase_g0=result.acrophase_g0,
            acrophase_g1=result.acrophase_g1,
            mesor_diff=result.mesor_diff,
            amplitude_diff=result.amplitude_diff,
            acrophase_diff=result.acrophase_diff,
            p_mesor=result.mesor_p_value,
            p_amplitude=result.amplitude_p_value,
            p_acrophase=result.acrophase_p_value,
            q_mesor=q_mesor,
            q_amplitude=q_amplitude,
            q_acrophase=q_acrophase,
            amplitude_diff_ci=result.amplitude_diff_ci,
            acrophase_diff_ci=result.acrophase_diff_ci,
            mesor_diff_ci=result.mesor_diff_ci,
            period=period,
            times_g0=times_g0,
            values_g0=values_g0,
            times_g1=times_g1,
            values_g1=values_g1,
            success=True
        )

    def _run_cosinorpy_limorhyde(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> ComparisonResult:
        """Run CosinorPy LimoRhyde comparison."""
        print(f"[DEBUG] _run_cosinorpy_limorhyde called")
        print(f"[DEBUG] self._cosinor is None: {self._cosinor is None}")

        # Get folder for saving plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        if plot_folder:
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        if self._cosinor is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_limorhyde",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)
        print(f"[DEBUG] Period: {period}, n_components: {n_components}")

        # Filter for both conditions
        mask = data[condition_col].isin([condition1, condition2])
        filtered = data[mask].copy()
        print(f"[DEBUG] Filtered data shape: {filtered.shape}")

        # Extract raw data for each group (for plotting)
        data_g0 = filtered[filtered[condition_col] == condition1]
        data_g1 = filtered[filtered[condition_col] == condition2]
        times_g0 = data_g0[time_col].values.astype(float) if len(data_g0) > 0 else None
        values_g0 = data_g0[variable].values.astype(float) if len(data_g0) > 0 else None
        times_g1 = data_g1[time_col].values.astype(float) if len(data_g1) > 0 else None
        values_g1 = data_g1[variable].values.astype(float) if len(data_g1) > 0 else None

        # Load data into CosinorAnalyzer
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes
            self._cosinor._raw_data = filtered
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition1, condition2]
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Data loaded successfully")

            print(f"[DEBUG] Setting period...")
            self._cosinor.set_period(period)
            print(f"[DEBUG] Parameters set successfully")

            print(f"[DEBUG] Calling self._cosinor.compare_conditions_limo()...")
            result = self._cosinor.compare_conditions_limo(
                variable=variable,
                condition1=condition1,
                condition2=condition2,
                period=period,
                n_components=n_components,
                save_folder=plot_folder
            )
            print(f"[DEBUG] compare_conditions_limo() returned successfully")
            print(f"[DEBUG] Result: {result}")
        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy LimoRhyde comparison: {e}")
            import traceback
            traceback.print_exc()
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_limorhyde",
                success=False, message=f"Error: {str(e)}"
            )

        if result is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_limorhyde",
                success=False, message="Analysis failed"
            )

        # Extract results from DifferentialResult object
        print(f"[DEBUG] Extracting LimoRhyde comparison results...")

        # Extract q-values if available
        q_amplitude = None
        q_acrophase = None
        q_mesor = None
        if result.q_values:
            q_amplitude = result.q_values.get('amplitude')
            q_acrophase = result.q_values.get('acrophase')
            q_mesor = result.q_values.get('mesor')

        return ComparisonResult(
            variable=variable,
            condition1=condition1,
            condition2=condition2,
            method="cosinorpy_limorhyde",
            mesor_g0=result.mesor_g0,
            mesor_g1=result.mesor_g1,
            amplitude_g0=result.amplitude_g0,
            amplitude_g1=result.amplitude_g1,
            acrophase_g0=result.acrophase_g0,
            acrophase_g1=result.acrophase_g1,
            mesor_diff=result.mesor_diff,
            amplitude_diff=result.amplitude_diff,
            acrophase_diff=result.acrophase_diff,
            p_mesor=result.mesor_p_value,
            p_amplitude=result.amplitude_p_value,
            p_acrophase=result.acrophase_p_value,
            q_mesor=q_mesor,
            q_amplitude=q_amplitude,
            q_acrophase=q_acrophase,
            amplitude_diff_ci=result.amplitude_diff_ci,
            acrophase_diff_ci=result.acrophase_diff_ci,
            mesor_diff_ci=result.mesor_diff_ci,
            period=period,
            times_g0=times_g0,
            values_g0=values_g0,
            times_g1=times_g1,
            values_g1=values_g1,
            success=True
        )

    def _run_cosinorpy_compare_nonlinear(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> ComparisonResult:
        """Run CosinorPy nonlinear cosinor comparison."""
        print(f"[DEBUG] _run_cosinorpy_compare_nonlinear called")
        print(f"[DEBUG] Variable: {variable}, Condition1: {condition1}, Condition2: {condition2}")

        # Get folder for saving plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        if plot_folder:
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        if self._cosinor is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare_nonlinear",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        use_dependent_model = parameters.get('use_dependent_model', False)
        auto_components = parameters.get('auto_components', False)
        print(f"[DEBUG] Period: {period}, use_dependent_model: {use_dependent_model}, auto_components: {auto_components}")

        # Filter for both conditions
        mask = data[condition_col].isin([condition1, condition2])
        filtered = data[mask].copy()
        print(f"[DEBUG] Filtered data shape: {filtered.shape}")

        # Extract raw data for each group (for plotting)
        data_g0 = filtered[filtered[condition_col] == condition1]
        data_g1 = filtered[filtered[condition_col] == condition2]
        times_g0 = data_g0[time_col].values.astype(float) if len(data_g0) > 0 else None
        values_g0 = data_g0[variable].values.astype(float) if len(data_g0) > 0 else None
        times_g1 = data_g1[time_col].values.astype(float) if len(data_g1) > 0 else None
        values_g1 = data_g1[variable].values.astype(float) if len(data_g1) > 0 else None

        # Load data into CosinorAnalyzer
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes
            self._cosinor._raw_data = filtered
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition1, condition2]
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None

            # Detect analysis mode from data
            # DEPENDENT data = same subject measured multiple times (requires "subject" column)
            # INDEPENDENT data = different subjects (biological replicates, no "subject" column)

            # First, look for subject column
            subject_col = None
            for col_name in ['subject', 'Subject', 'id', 'ID', 'individual', 'Individual']:
                if col_name in filtered.columns:
                    subject_col = col_name
                    print(f"[DEBUG] Found subject column: {subject_col}")
                    break

            self._cosinor._subject_col = subject_col

            # Determine analysis mode based on presence of subject column
            if subject_col is not None:
                # Has subject column → DEPENDENT data (repeated measures)
                self._cosinor.analysis_mode = AnalysisMode.DEPENDENT
                print(f"[DEBUG] Detected DEPENDENT data (repeated measures with subject column)")
            else:
                # No subject column → INDEPENDENT data (biological replicates or single measurements)
                self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT
                print(f"[DEBUG] Detected INDEPENDENT data (biological replicates or single measures)")

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Data loaded successfully")

            print(f"[DEBUG] Setting period...")
            self._cosinor.set_period(period)
            print(f"[DEBUG] Parameters set successfully")

            # If auto_components is enabled, determine best n_components for each condition
            n_components_c1 = 1
            n_components_c2 = 1
            if auto_components:
                print(f"[DEBUG] Auto-select components enabled")
                print(f"[DEBUG] Finding best n_components for {condition1}...")

                try:
                    # Analyze condition1
                    result_c1 = self._cosinor.nonlinear_cosinor_best_fit(
                        variable=variable,
                        condition=condition1,
                        period=period,
                        n_components_range=[1, 2, 3],
                        save_folder=None  # Don't save plots for auto-selection
                    )
                    n_components_c1 = result_c1.get('n_components_selected', 1)
                    print(f"[DEBUG] Best n_components for {condition1}: {n_components_c1}")
                except Exception as e:
                    print(f"[DEBUG] ERROR finding best n for {condition1}: {e}")
                    n_components_c1 = 1

                print(f"[DEBUG] Finding best n_components for {condition2}...")
                try:
                    # Analyze condition2
                    result_c2 = self._cosinor.nonlinear_cosinor_best_fit(
                        variable=variable,
                        condition=condition2,
                        period=period,
                        n_components_range=[1, 2, 3],
                        save_folder=None  # Don't save plots for auto-selection
                    )
                    n_components_c2 = result_c2.get('n_components_selected', 1)
                    print(f"[DEBUG] Best n_components for {condition2}: {n_components_c2}")
                except Exception as e:
                    print(f"[DEBUG] ERROR finding best n for {condition2}: {e}")
                    n_components_c2 = 1

                print(f"[DEBUG] Auto-selection complete: {condition1}={n_components_c1}, {condition2}={n_components_c2}")
                # Note: For comparison, we'll use the max of the two
                # This ensures both conditions are modeled with sufficient complexity
                # Alternatively, we could use the average or let each condition use its own optimal value
                # For now, we'll proceed with the comparison using the standard single n_components approach
                # Future enhancement: CosinorPy doesn't support different n_components per condition in comparison

            print(f"[DEBUG] Calling self._cosinor.compare_conditions_nonlinear()...")
            result = self._cosinor.compare_conditions_nonlinear(
                variable=variable,
                condition1=condition1,
                condition2=condition2,
                period=period,
                use_dependent_model=use_dependent_model,
                save_folder=plot_folder
            )
            print(f"[DEBUG] compare_conditions_nonlinear() returned successfully")
            print(f"[DEBUG] Result: {result}")
        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy nonlinear comparison: {e}")
            import traceback
            traceback.print_exc()
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare_nonlinear",
                success=False, message=f"Error: {str(e)}"
            )

        if result is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare_nonlinear",
                success=False, message="Analysis failed"
            )

        # Extract results from DifferentialResult object
        print(f"[DEBUG] Extracting nonlinear comparison results...")

        # Extract q-values if available
        q_amplitude = None
        q_acrophase = None
        q_amplification = None
        q_lin_comp = None
        if result.q_values:
            q_amplitude = result.q_values.get('amplitude')
            q_acrophase = result.q_values.get('acrophase')
            q_amplification = result.q_values.get('amplification')
            q_lin_comp = result.q_values.get('lin_comp')

        # Create ComparisonResult with all fields including nonlinear-specific ones
        print(f"[DEBUG] About to create ComparisonResult")
        print(f"[DEBUG]   result.amplification_diff = {result.amplification_diff}")
        print(f"[DEBUG]   result.amplification_diff type = {type(result.amplification_diff)}")
        print(f"[DEBUG]   result.lin_comp_diff = {result.lin_comp_diff}")

        # Build message with auto-selection info if applicable
        message = None
        if auto_components:
            message = f"Auto-selected n_components: {condition1}={n_components_c1}, {condition2}={n_components_c2}"
            print(f"[DEBUG] {message}")

        # Generate phase plot if we have valid acrophases and amplitudes for both groups
        if plot_folder and result.acrophase_g0 is not None and result.acrophase_g1 is not None:
            if result.amplitude_g0 is not None and result.amplitude_g1 is not None:
                test_name_g0 = f"{variable}_{condition1}"
                test_name_g1 = f"{variable}_{condition2}"

                # Extract confidence intervals if available
                CI_acrs = []
                CI_amps = []

                # Note: result has CI for differences, not for individual groups
                # We'll generate the plot without individual CIs for now

                print(f"[DEBUG] Generating phase plot for Compare Nonlinear: {condition1} vs {condition2}")
                self._cosinor._generate_phase_plot(
                    acrophases=[result.acrophase_g0, result.acrophase_g1],
                    amplitudes=[result.amplitude_g0, result.amplitude_g1],
                    tests=[test_name_g0, test_name_g1],
                    save_folder=plot_folder,
                    prefix="compare_nonlinear_",
                    period=period,
                    CI_acrs=None,  # Individual group CIs not available from comparison
                    CI_amps=None   # Individual group CIs not available from comparison
                )

        comp_result = ComparisonResult(
            variable=variable,
            condition1=condition1,
            condition2=condition2,
            method="cosinorpy_compare_nonlinear",
            mesor_g0=result.mesor_g0,
            mesor_g1=result.mesor_g1,
            amplitude_g0=result.amplitude_g0,
            amplitude_g1=result.amplitude_g1,
            acrophase_g0=result.acrophase_g0,
            acrophase_g1=result.acrophase_g1,
            mesor_diff=result.mesor_diff,
            amplitude_diff=result.amplitude_diff,
            acrophase_diff=result.acrophase_diff,
            p_mesor=result.mesor_p_value,
            p_amplitude=result.amplitude_p_value,
            p_acrophase=result.acrophase_p_value,
            q_mesor=None,
            q_amplitude=q_amplitude,
            q_acrophase=q_acrophase,
            amplitude_diff_ci=result.amplitude_diff_ci,
            acrophase_diff_ci=result.acrophase_diff_ci,
            mesor_diff_ci=result.mesor_diff_ci,
            # Nonlinear-specific fields
            # Note: CosinorPy comparison only returns differences, not individual group values
            amplification_g0=None,  # Not available from nonlinear comparison
            amplification_g1=None,  # Not available from nonlinear comparison
            amplification_diff=result.amplification_diff,
            amplification_diff_ci=result.amplification_diff_ci,
            p_amplification=result.amplification_p_value,
            q_amplification=q_amplification,
            lin_comp_g0=None,  # Not available from nonlinear comparison
            lin_comp_g1=None,  # Not available from nonlinear comparison
            lin_comp_diff=result.lin_comp_diff,
            lin_comp_diff_ci=result.lin_comp_diff_ci,
            p_lin_comp=result.lin_comp_p_value,
            q_lin_comp=q_lin_comp,
            period=period,
            times_g0=times_g0,
            values_g0=values_g0,
            times_g1=times_g1,
            values_g1=values_g1,
            success=True,
            message=message  # Include auto-selection info if applicable
        )

        print(f"[DEBUG] ComparisonResult created")
        print(f"[DEBUG]   amplification_diff attribute: {comp_result.amplification_diff}")
        print(f"[DEBUG]   lin_comp_diff attribute: {comp_result.lin_comp_diff}")
        print(f"[DEBUG]   to_dict amplification_diff: {comp_result.to_dict().get('amplification_diff')}")
        print(f"[DEBUG]   to_dict lin_comp_diff: {comp_result.to_dict().get('lin_comp_diff')}")

        return comp_result

    def _run_cosinorpy_compare_all(
        self,
        data: pd.DataFrame,
        variable: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> List[ComparisonResult]:
        """
        Run CosinorPy comparison across ALL condition pairs for a variable.

        Args:
            data: Input DataFrame
            variable: Variable to analyze
            time_col: Time column name
            condition_col: Condition column name
            parameters: Analysis parameters
            data_file_path: Path to data file for plot saving

        Returns:
            List of ComparisonResult objects
        """
        print(f"[DEBUG] _run_cosinorpy_compare_all called")
        print(f"[DEBUG] Variable: {variable}")

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)

        # Determine plot folder
        plot_folder = None
        if data_file_path:
            import os
            data_dir = os.path.dirname(data_file_path)
            plot_folder = os.path.join(data_dir, 'cosinorpy_plots')
            os.makedirs(plot_folder, exist_ok=True)
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        # Initialize CosinorAnalyzer if needed
        if self._cosinor is None:
            from core.cosinor_analysis import CosinorAnalyzer
            self._cosinor = CosinorAnalyzer()

        print(f"[DEBUG] Period: {period}, n_components: {n_components}")

        # Filter data for this variable
        if variable in data.columns:
            filtered_data = data[[time_col, condition_col, variable]].copy()
            print(f"[DEBUG] Filtered data shape: {filtered_data.shape}")
        else:
            print(f"[ERROR] Variable {variable} not found in data columns")
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method="cosinorpy_compare_all",
                success=False,
                message=f"Variable {variable} not found in data"
            )]

        # Load data into CosinorAnalyzer
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            self._cosinor._raw_data = filtered_data
            self._cosinor._variables = [variable]
            self._cosinor._conditions = filtered_data[condition_col].unique().tolist()
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")

            print(f"[DEBUG] Data loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method="cosinorpy_compare_all",
                success=False,
                message=f"Failed to load data: {str(e)}"
            )]

        # Set period
        try:
            print(f"[DEBUG] Setting period...")
            self._cosinor.set_period(period)
            print(f"[DEBUG] Parameters set successfully")
        except Exception as e:
            print(f"[ERROR] Failed to set parameters: {e}")

        # Call compare_all_conditions
        try:
            print(f"[DEBUG] Calling self._cosinor.compare_all_conditions()...")

            diff_results = self._cosinor.compare_all_conditions(
                variable=variable,
                period=period,
                n_components=n_components,
                save_folder=plot_folder
            )
            print(f"[DEBUG] compare_all_conditions() returned {len(diff_results)} results")

        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy compare_all: {e}")
            import traceback
            traceback.print_exc()
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method="cosinorpy_compare_all",
                success=False,
                message=f"Error: {str(e)}"
            )]

        # Convert DifferentialResult objects to ComparisonResult objects
        print(f"[DEBUG] Converting {len(diff_results)} DifferentialResults to ComparisonResults...")
        comparison_results = []

        for diff_result in diff_results:
            # Extract q-values
            q_amplitude = None
            q_acrophase = None
            q_mesor = None
            if diff_result.q_values:
                q_amplitude = diff_result.q_values.get('amplitude')
                q_acrophase = diff_result.q_values.get('acrophase')
                q_mesor = diff_result.q_values.get('mesor')

            comp_result = ComparisonResult(
                variable=variable,
                condition1=diff_result.condition1,
                condition2=diff_result.condition2,
                method="cosinorpy_compare_all",
                mesor_g0=diff_result.mesor_g0,
                mesor_g1=diff_result.mesor_g1,
                amplitude_g0=diff_result.amplitude_g0,
                amplitude_g1=diff_result.amplitude_g1,
                acrophase_g0=diff_result.acrophase_g0,
                acrophase_g1=diff_result.acrophase_g1,
                mesor_diff=diff_result.mesor_diff,
                amplitude_diff=diff_result.amplitude_diff,
                acrophase_diff=diff_result.acrophase_diff,
                p_mesor=diff_result.mesor_p_value,
                p_amplitude=diff_result.amplitude_p_value,
                p_acrophase=diff_result.acrophase_p_value,
                q_mesor=q_mesor,
                q_amplitude=q_amplitude,
                q_acrophase=q_acrophase,
                amplitude_diff_ci=diff_result.amplitude_diff_ci,
                acrophase_diff_ci=diff_result.acrophase_diff_ci,
                mesor_diff_ci=diff_result.mesor_diff_ci,
                period=period,
                success=True
            )
            comparison_results.append(comp_result)

        print(f"[DEBUG] Returning {len(comparison_results)} ComparisonResults")
        return comparison_results

    def _run_cosinorpy_compare_all_limo(
        self,
        data: pd.DataFrame,
        variable: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> List[ComparisonResult]:
        """
        Run CosinorPy LimoRhyde comparison across ALL condition pairs for a variable.

        This uses the LimoRhyde method which is an alternative statistical approach
        that can be more powerful in certain scenarios.

        Args:
            data: Input DataFrame
            variable: Variable to analyze
            time_col: Time column name
            condition_col: Condition column name
            parameters: Analysis parameters
            data_file_path: Path to data file for plot saving

        Returns:
            List of ComparisonResult objects
        """
        print(f"[DEBUG] _run_cosinorpy_compare_all_limo called")
        print(f"[DEBUG] Variable: {variable}")

        # Determine plot folder
        plot_folder = None
        if data_file_path:
            import os
            data_dir = os.path.dirname(data_file_path)
            plot_folder = os.path.join(data_dir, 'cosinorpy_plots')
            os.makedirs(plot_folder, exist_ok=True)
            print(f"[DEBUG] Plot folder: {plot_folder}")

        # Get parameters
        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)

        print(f"[DEBUG] Parameters - period: {period}, n_components: {n_components}")

        # Initialize CosinorAnalyzer if needed
        if self._cosinor is None:
            from core.cosinor_analysis import CosinorAnalyzer
            self._cosinor = CosinorAnalyzer()

        # Filter data for this variable
        if variable in data.columns:
            filtered_data = data[[time_col, condition_col, variable]].copy()
            print(f"[DEBUG] Filtered data shape: {filtered_data.shape}")
        else:
            print(f"[ERROR] Variable {variable} not found in data columns")
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method="cosinorpy_compare_all_limo",
                success=False,
                message=f"Variable {variable} not found in data"
            )]

        # Load data into CosinorAnalyzer
        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            self._cosinor._raw_data = filtered_data
            self._cosinor._variables = [variable]
            self._cosinor._conditions = filtered_data[condition_col].unique().tolist()
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT
            self._cosinor.period = period

            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")

            print(f"[DEBUG] Data loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            import traceback
            traceback.print_exc()
            return [ComparisonResult(
                variable=variable,
                condition1="",
                condition2="",
                method="cosinorpy_compare_all_limo",
                success=False,
                message=f"Failed to load data: {str(e)}"
            )]

        # Call the LimoRhyde compare all method
        diff_results = self._cosinor.compare_all_conditions_limo(
            variable=variable,
            period=period,
            n_components=n_components,
            save_folder=plot_folder
        )

        print(f"[DEBUG] compare_all_conditions_limo returned {len(diff_results)} results")

        # Convert DifferentialResult objects to ComparisonResult objects
        comparison_results = []
        for diff_result in diff_results:
            # Extract q-values
            q_amp = diff_result.q_values.get('amplitude') if diff_result.q_values else None
            q_acro = diff_result.q_values.get('acrophase') if diff_result.q_values else None
            q_overall = diff_result.q_values.get('overall') if diff_result.q_values else None

            comp_result = ComparisonResult(
                variable=variable,
                condition1=diff_result.condition1,
                condition2=diff_result.condition2,
                method="LimoRhyde Compare All",
                success=True,
                message="",
                # Amplitude
                amplitude_g0=diff_result.amplitude_g0,
                amplitude_g1=diff_result.amplitude_g1,
                amplitude_diff=diff_result.amplitude_diff,
                p_amplitude=diff_result.amplitude_p_value,
                q_amplitude=q_amp,
                amplitude_diff_ci=diff_result.amplitude_diff_ci,
                # Acrophase
                acrophase_g0=diff_result.acrophase_g0,
                acrophase_g1=diff_result.acrophase_g1,
                acrophase_diff=diff_result.acrophase_diff,
                p_acrophase=diff_result.acrophase_p_value,
                q_acrophase=q_acro,
                acrophase_diff_ci=diff_result.acrophase_diff_ci,
                # MESOR
                mesor_g0=diff_result.mesor_g0,
                mesor_g1=diff_result.mesor_g1,
                mesor_diff=diff_result.mesor_diff,
                p_mesor=diff_result.mesor_p_value,
                q_mesor=None,  # LimoRhyde doesn't provide MESOR q-value
                mesor_diff_ci=diff_result.mesor_diff_ci,
                # Additional metadata
                period=period
            )
            comparison_results.append(comp_result)

        print(f"[DEBUG] Returning {len(comparison_results)} ComparisonResult objects")
        return comparison_results

    # =========================================================================
    # CircaCompare Methods
    # =========================================================================
    
    def _run_circacompare_single(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Run CircaCompare single fit. Supports single period or a list of periods."""
        period = parameters.get('period', 24.0)
        loss = parameters.get('loss', 'huber')
        f_scale = parameters.get('f_scale', 1.0)
        max_iterations = parameters.get('max_iterations', 500)

        # Normalise period to a list so single and range cases share the same loop
        periods = list(period) if isinstance(period, (list, tuple, np.ndarray)) else [float(period)]
        is_single = len(periods) == 1

        # Build the DataFrame once (same data for every period)
        df = pd.DataFrame({
            'time': times,
            'condition': condition,
            variable: values
        })
        self._circacompare.load_dataframe(df)
        self._circacompare.set_loss(loss)
        self._circacompare.set_f_scale(f_scale)
        self._circacompare.set_max_iterations(max_iterations)

        results = []
        for p in periods:
            p = float(p)
            self._circacompare.set_period(p)

            result = self._circacompare.fit_single(
                variable=variable, condition=condition
            )

            if result is None:
                if is_single:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="circacompare_single",
                        success=False, message="Fit failed"
                    )
                continue

            acrophase_hours = (result.acrophase * p) / (2 * np.pi)
            if acrophase_hours < 0:
                acrophase_hours += p

            se = result.standard_errors
            results.append(AnalysisResult(
                variable=variable,
                condition=condition,
                method="circacompare_single",
                mesor=result.mesor,
                amplitude=result.amplitude,
                acrophase=result.acrophase,
                acrophase_hours=acrophase_hours,
                period=p,
                mesor_ci=result.mesor_ci,
                amplitude_ci=result.amplitude_ci,
                acrophase_ci=result.acrophase_ci,
                se_mesor=float(se[0]) if se is not None else None,
                se_amplitude=float(se[1]) if se is not None else None,
                se_acrophase=float(se[2]) if se is not None else None,
                times=times,
                values=values,
                success=getattr(result, 'success', True)
            ))

        if not results:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="circacompare_single",
                success=False, message="All period fits failed"
            )

        return results[0] if is_single else results
    
    def _run_circacompare_comparison(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> Union['ComparisonResult', List['ComparisonResult']]:
        """Run CircaCompare group comparison. Returns a list when a period range is given."""
        print(f"[DEBUG] CircaCompare comparison called:")
        print(f"  Variable: {variable}")
        print(f"  Condition1: {condition1}, Condition2: {condition2}")
        print(f"  Data shape: {data.shape}")

        period = parameters.get('period', 24.0)
        loss = parameters.get('loss', 'huber')
        f_scale = parameters.get('f_scale', 1.0)
        max_iterations = parameters.get('max_iterations', 500)

        # Normalize period to a list so we can iterate
        periods = list(period) if isinstance(period, (list, tuple, np.ndarray)) else [float(period)]
        is_single = len(periods) == 1

        # Prepare data - only include the two conditions being compared
        mask = data[condition_col].isin([condition1, condition2])
        filtered = data[mask].copy()

        print(f"  Filtered data shape: {filtered.shape}")
        print(f"  Unique conditions in filtered: {filtered[condition_col].unique()}")

        # Extract raw data for each group (for plotting) - same for all periods
        data_g0 = filtered[filtered[condition_col] == condition1]
        data_g1 = filtered[filtered[condition_col] == condition2]
        times_g0 = data_g0[time_col].values.astype(float) if len(data_g0) > 0 else None
        values_g0 = data_g0[variable].values.astype(float) if len(data_g0) > 0 else None
        times_g1 = data_g1[time_col].values.astype(float) if len(data_g1) > 0 else None
        values_g1 = data_g1[variable].values.astype(float) if len(data_g1) > 0 else None

        # Load data and common settings once
        self._circacompare.load_dataframe(filtered)
        self._circacompare.set_loss(loss)
        self._circacompare.set_f_scale(f_scale)
        self._circacompare.set_max_iterations(max_iterations)

        def calc_sig(ci_key, confidence_intervals):
            if ci_key in confidence_intervals:
                ci_lower, ci_upper = confidence_intervals[ci_key]
                return "Yes" if (ci_lower > 0 or ci_upper < 0) else "No"
            return None

        comparison_results = []
        for p in periods:
            p = float(p)
            self._circacompare.set_period(p)

            print(f"  Calling compare() for period={p}...")
            result = self._circacompare.compare(
                variable=variable,
                condition1=condition1,
                condition2=condition2
            )

            print(f"  Result: {result}")

            if result is None:
                if is_single:
                    return ComparisonResult(
                        variable=variable, condition1=condition1, condition2=condition2,
                        method="circacompare_compare",
                        success=False, message="Comparison failed"
                    )
                continue  # skip failed period in range

            sig_mesor = calc_sig('d_mesor', result.confidence_intervals)
            sig_amplitude = calc_sig('d_amplitude', result.confidence_intervals)
            sig_acrophase = calc_sig('d_acrophase', result.confidence_intervals)

            ci = result.confidence_intervals
            mesor_diff_ci = ci.get('d_mesor') if ci else None
            amplitude_diff_ci = ci.get('d_amplitude') if ci else None
            acrophase_diff_ci = ci.get('d_acrophase') if ci else None

            comparison_results.append(ComparisonResult(
                variable=variable,
                condition1=condition1,
                condition2=condition2,
                method="circacompare_compare",
                mesor_g0=result.mesor_g0,
                mesor_g1=result.mesor_g1,
                amplitude_g0=result.amplitude_g0,
                amplitude_g1=result.amplitude_g1,
                acrophase_g0=result.acrophase_g0,
                acrophase_g1=result.acrophase_g1,
                acrophase_g0_hours=result.acrophase_g0_hours,
                acrophase_g1_hours=result.acrophase_g1_hours,
                sig_mesor=sig_mesor,
                sig_amplitude=sig_amplitude,
                sig_acrophase=sig_acrophase,
                mesor_diff=result.d_mesor,
                amplitude_diff=result.d_amplitude,
                acrophase_diff=result.d_acrophase,
                acrophase_diff_hours=result.d_acrophase_hours,
                mesor_diff_ci=mesor_diff_ci,
                amplitude_diff_ci=amplitude_diff_ci,
                acrophase_diff_ci=acrophase_diff_ci,
                period=p,
                times_g0=times_g0,
                values_g0=values_g0,
                times_g1=times_g1,
                values_g1=values_g1,
                success=True
            ))

        if not comparison_results:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="circacompare_compare",
                success=False, message="All period fits failed"
            )

        return comparison_results[0] if is_single else comparison_results

    # =========================================================================
    # Rhythm Analysis Methods
    # =========================================================================
    
    def _run_jtk(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Run JTK Cycle analysis."""
        from .rhythm_analysis import _run_discrete_jtk, _run_discrete_jtk_all_periods
        import pandas as pd

        # Get period range - support both period_range tuple and period list
        period = parameters.get('period', None)
        if period is not None:
            if isinstance(period, list):
                period_range = [float(p) for p in period]
            else:
                period_range = [float(period)]
        else:
            period_range = list(range(20, 29))  # Default circadian range

        # Get asymmetry parameter (default: 0.5 for symmetric)
        asymmetry = parameters.get('asymmetry', 0.5)
        asymmetries = [asymmetry] if asymmetry else [0.5]

        # Create series with time as index
        series = pd.Series(values, index=times)

        show_all = len(period_range) > 1

        if show_all:
            per_period = _run_discrete_jtk_all_periods(
                series, period_range=period_range, asymmetries=asymmetries
            )
            if not per_period:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="jtk", success=False, message="JTK analysis failed"
                )
            best_p = min(r.p_value for r in per_period)
            results = []
            for r in per_period:
                results.append(AnalysisResult(
                    variable=variable, condition=condition,
                    method="jtk",
                    period=r.period,
                    acrophase_hours=r.acrophase,
                    p_value=r.bh_p_value,
                    raw_p_value=r.p_value,
                    bonf_p_value=r.bonf_p_value,
                    tau=r.tau,
                    lag=r.lag,
                    asymmetry=r.asymmetry,
                    n_tests=r.n_tests,
                    amplitude=r.amplitude,
                    times=times, values=values,
                    success=True,
                    best_model="Yes" if r.p_value == best_p else "No"
                ))
            return results

        result = _run_discrete_jtk(
            series,
            period_range=period_range,
            asymmetries=asymmetries
        )

        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="jtk",
                success=False, message="JTK analysis failed"
            )

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="jtk",
            period=result.period,
            acrophase_hours=result.acrophase,
            p_value=result.bh_p_value,
            raw_p_value=result.p_value,
            bonf_p_value=result.bonf_p_value,
            tau=result.tau,
            lag=result.lag,
            asymmetry=result.asymmetry,
            n_tests=result.n_tests,
            amplitude=result.amplitude,
            times=times,
            values=values,
            success=True
        )
    
    def _run_cosinor_ols(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run OLS cosinor with period search."""
        from .rhythm_analysis import _fit_cosinor, DefaultPeriodRanges

        # Handle search mode
        search_mode = parameters.get('search_mode', 'Optimize Period')
        period = parameters.get('period', None)

        if search_mode == 'Fixed Period':
            # Use single fixed period
            if period is not None:
                if isinstance(period, list):
                    period_range = [period[0]]  # Use first value if list
                else:
                    period_range = [period]
            else:
                period_range = [24.0]  # Default to 24h
        else:
            # Optimize Period mode - use period range
            if period is not None:
                if isinstance(period, list):
                    period_range = period
                else:
                    period_range = DefaultPeriodRanges.CIRCADIAN
            else:
                period_range = DefaultPeriodRanges.CIRCADIAN

        result = _fit_cosinor(times, values, period_range=period_range)

        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinor_ols",
                success=False, message="Cosinor analysis failed"
            )

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="cosinor_ols",
            mesor=result.mesor,
            amplitude=result.amplitude,
            acrophase=result.acrophase_rad,
            acrophase_hours=result.acrophase_hours,
            period=result.period,
            p_value=result.p_value,
            bonf_p_value=result.adj_p_value,
            amplitude_ci=result.amplitude_ci,
            acrophase_ci=result.acrophase_ci,
            p_amplitude=result.amplitude_p,
            p_acrophase=result.acrophase_p,
            times=times,
            values=values,
            success=True
        )
    
    def _run_harmonic_cosinor(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run harmonic cosinor analysis."""
        from .rhythm_analysis import _fit_harmonic_cosinor, DefaultPeriodRanges

        n_harmonics = parameters.get('n_harmonics', 2)
        period_range = parameters.get('period_range', None)
        if period_range is None:
            period_range = DefaultPeriodRanges.CIRCADIAN

        result = _fit_harmonic_cosinor(times, values, period_range=period_range, n_harmonics=n_harmonics)

        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="harmonic_cosinor",
                success=False, message="Harmonic cosinor failed"
            )

        # Get primary amplitude and acrophase from lists
        primary_amplitude = result.amplitudes[0] if result.amplitudes else None
        primary_acrophase = result.acrophases[0] if result.acrophases else None

        # Build amplitude/acrophase info for all harmonics in message
        harmonic_details = ", ".join(
            f"H{i+1}: A={a:.4f} φ={p:.2f}h"
            for i, (a, p) in enumerate(zip(result.amplitudes, result.acrophases))
        )
        msg = result.warning or ""
        if len(result.amplitudes) > 1:
            msg = (msg + f" | Harmonics: [{harmonic_details}]").lstrip(" | ")

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="harmonic_cosinor",
            amplitude=primary_amplitude,
            acrophase_hours=primary_acrophase,
            period=result.period,
            p_value=result.adj_p_value,
            n_components=result.n_harmonics,
            peak_times=result.acrophases,    # all harmonic acrophases (h)
            trough_times=result.amplitudes,  # all harmonic amplitudes (repurposed field)
            times=times,
            values=values,
            success=True,
            message=msg
        )
    
    def _run_lomb_scargle(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Lomb-Scargle periodogram."""
        from .rhythm_analysis import _compute_lomb_scargle

        period_range = parameters.get('period_range', (18.0, 32.0))
        n_periods = parameters.get('n_periods', 1000)
        alpha = parameters.get('alpha', 0.05)

        result = _compute_lomb_scargle(times, values, period_range=period_range, n_periods=n_periods)

        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="lomb_scargle",
                success=False, message="Lomb-Scargle failed"
            )

        fap = result.false_alarm_probability
        is_significant = fap is not None and fap < alpha
        msg = (f"FAP={fap:.4f} ({'significant' if is_significant else 'not significant'}, "
               f"α={alpha}), dominant period={result.dominant_period:.2f}h")

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="lomb_scargle",
            dominant_period=result.dominant_period,
            power=result.dominant_power,
            p_value=result.false_alarm_probability,
            period=result.dominant_period,
            periods=result.periods,
            power_spectrum=result.power_spectrum,
            times=times,
            values=values,
            success=True,
            message=msg
        )
    
    # =========================================================================
    # Batch Analysis
    # =========================================================================
    
    def run_batch_analysis(
        self,
        data: pd.DataFrame,
        variables: List[str],
        conditions: List[str],
        analysis_type: AnalysisType,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[AnalysisResult]:
        """
        Run analysis on multiple variables and conditions.
        
        Args:
            data: DataFrame with all data
            variables: List of variable column names
            conditions: List of conditions to analyze
            analysis_type: Type of analysis
            time_col: Time column name
            condition_col: Condition column name
            parameters: Analysis parameters
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of AnalysisResult objects
        """
        results = []
        total = len(variables) * len(conditions)
        current = 0
        
        for var in variables:
            for cond in conditions:
                if progress_callback:
                    progress_callback(current, total, f"Analyzing {var} in {cond}")
                
                result = self.run_analysis(
                    data, var, cond, analysis_type,
                    time_col, condition_col, parameters
                )
                results.append(result)
                current += 1
        
        if progress_callback:
            progress_callback(total, total, "Complete")
        
        return results
    
    def run_batch_comparison(
        self,
        data: pd.DataFrame,
        variables: List[str],
        condition1: str,
        condition2: str,
        analysis_type: AnalysisType,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ComparisonResult]:
        """
        Run comparison analysis on multiple variables.
        
        Returns:
            List of ComparisonResult objects
        """
        results = []
        total = len(variables)
        
        for i, var in enumerate(variables):
            if progress_callback:
                progress_callback(i, total, f"Comparing {var}")
            
            result = self.run_comparison(
                data, var, condition1, condition2, analysis_type,
                time_col, condition_col, parameters
            )
            results.append(result)

        if progress_callback:
            progress_callback(total, total, "Complete")

        return results

    # =========================================================================
    # ADDITIONAL ANALYSIS METHODS
    # =========================================================================

    def _run_cosinorpy_population(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> AnalysisResult:
        """Run CosinorPy population-mean cosinor analysis."""
        # Get folder for saving plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        if plot_folder:
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        print(f"[DEBUG] _run_cosinorpy_population called")
        print(f"[DEBUG] self._cosinor is None: {self._cosinor is None}")

        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_population",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)
        auto_components = parameters.get('auto_components', False)
        amplification = parameters.get('amplification', None)
        lin_comp = parameters.get('lin_comp', None)
        print(f"[DEBUG] Period: {period}, n_components: {n_components}, auto_components: {auto_components}")

        # Filter for the specific condition
        subset = data[data[condition_col] == condition].copy()
        print(f"[DEBUG] Filtered data shape: {subset.shape}")
        print(f"[DEBUG] Columns: {subset.columns.tolist()}")

        # Check for subject/replicate column (subject identifier for population mean)
        replicate_col = None
        if 'subject' in subset.columns:
            replicate_col = 'subject'
            print(f"[DEBUG] Found 'subject' column")
            print(f"[DEBUG] Unique subjects: {subset[replicate_col].unique()}")
        elif 'replicate' in subset.columns:
            replicate_col = 'replicate'
            print(f"[DEBUG] Found 'replicate' column")
            print(f"[DEBUG] Unique subjects: {subset[replicate_col].unique()}")
        else:
            print(f"[DEBUG] WARNING: No 'subject' or 'replicate' column found - population mean requires subject identifiers")

        # Get times and values for plotting
        times = subset[time_col].values
        values = subset[variable].values

        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes for DEPENDENT (population) analysis
            self._cosinor._raw_data = subset
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition]
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = replicate_col  # Subject column for population mean
            self._cosinor.analysis_mode = AnalysisMode.DEPENDENT  # Important: DEPENDENT for population mean

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Converted DataFrame head:\n{self._cosinor._cosinorpy_df.head(10)}")
            print(f"[DEBUG] Data loaded successfully")

            print(f"[DEBUG] Setting period...")
            self._cosinor.set_period(period)

            # Check if auto-select components is enabled
            # For population data, auto-select always uses the nonlinear version
            # because it can test multi-component models
            if auto_components:
                print(f"[DEBUG] Auto-select components enabled for population nonlinear, calling population_nonlinear_cosinor_best_fit()...")
                try:
                    result = self._cosinor.population_nonlinear_cosinor_best_fit(
                        variable=variable,
                        condition=condition,
                        period=period,
                        n_components_range=[1, 2, 3],
                        amplification=amplification,
                        lin_comp=lin_comp,
                        save_folder=plot_folder
                    )
                    print(f"[DEBUG] Best model selected: n_components={result.get('n_components_selected')}")
                except Exception as nonlinear_error:
                    print(f"[DEBUG] Nonlinear auto-select failed (convergence issue): {nonlinear_error}")
                    print(f"[DEBUG] Falling back to standard population_cosinor with n_components=1...")
                    # Fallback to standard method when nonlinear fails (e.g., convergence issues)
                    result = self._cosinor.population_cosinor(
                        variable=variable,
                        condition=condition,
                        period=period,
                        n_components=1,  # Use n=1 for simple data
                        save_folder=plot_folder
                    )
            else:
                print(f"[DEBUG] Calling self._cosinor.population_cosinor()...")
                result = self._cosinor.population_cosinor(
                    variable=variable,
                    condition=condition,
                    period=period,
                    n_components=n_components,
                    save_folder=plot_folder
                )
            print(f"[DEBUG] Analysis returned successfully")
            print(f"[DEBUG] Result keys: {result.keys()}")
            print(f"[DEBUG] Result: {result}")

        except Exception as e:
            print(f"[DEBUG] ERROR in CosinorPy population: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_population",
                success=False, message=f"Error: {str(e)}"
            )

        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_population",
                success=False, message="Analysis failed"
            )

        # Check if this is from population_nonlinear_cosinor_best_fit (dict format)
        # or from population_cosinor (population_results dict format)
        if 'n_components_selected' in result:
            # Result from population_nonlinear_cosinor_best_fit
            print(f"[DEBUG] Processing nonlinear population result (auto-selected)")

            amplitude = result.get('amplitude')
            acrophase_rad = result.get('acrophase')
            mesor = result.get('mesor')
            p_value = result.get('p_value')
            n_components_result = result.get('n_components_selected')

            # Nonlinear-specific fields
            amplification = result.get('amplification')
            lin_comp_val = result.get('lin_comp')
            p_amplification = result.get('p_amplification')
            p_lin_comp = result.get('p_lin_comp')

            # CIs and q-values
            amplitude_ci = result.get('amplitude_ci')
            acrophase_ci = result.get('acrophase_ci')
            mesor_ci = None  # Not available from nonlinear

            # Convert acrophase to hours
            acrophase_hours = None
            if acrophase_rad is not None:
                acrophase_hours = (-acrophase_rad * period / (2 * np.pi)) % period

            print(f"[DEBUG] Extracted nonlinear - n={n_components_result}, amplitude={amplitude}, p={p_value}")

        elif 'population_results' in result:
            # Result from population_cosinor (standard format)
            print(f"[DEBUG] Processing standard population result")
            pop_results = result['population_results']
            print(f"[DEBUG] Population results:\n{pop_results}")

            if pop_results is not None and isinstance(pop_results, dict):
                # Extract population parameters from dictionary
                # pop_results['means'] = [MESOR, rrr, sss, amplitude, acrophase]
                means = pop_results.get('means')
                confint = pop_results.get('confint', {})

                if means is not None and len(means) >= 5:
                    mesor = means[0]  # Index 0: MESOR (Intercept)
                    amplitude = means[3]  # Index 3: amplitude
                    acrophase_rad = means[4]  # Index 4: acrophase in radians

                    # Get p-value (overall model)
                    p_value = pop_results.get('p_value')

                    # Get confidence intervals
                    amplitude_ci = confint.get('amp')
                    acrophase_ci = confint.get('acr')
                    mesor_ci = confint.get('MESOR')

                    # Convert acrophase to hours
                    acrophase_hours = (-acrophase_rad * period / (2 * np.pi)) % period

                    # Not nonlinear
                    n_components_result = n_components
                    amplification = None
                    lin_comp_val = None
                    p_amplification = None
                    p_lin_comp = None

                    print(f"[DEBUG] Extracted - mesor: {mesor}, amplitude: {amplitude}, acrophase_hours: {acrophase_hours}, p_value: {p_value}")
                else:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="cosinorpy_population",
                        success=False, message="Invalid means array in population results"
                    )
            else:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cosinorpy_population",
                    success=False, message="No population results returned"
                )
        else:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_population",
                success=False, message="Unknown result format"
            )

        # Generate phase plot if we have valid acrophase and save folder
        if plot_folder and acrophase_rad is not None and amplitude is not None:
            test_name = f"{variable}_{condition}"

            # Extract confidence intervals if available
            CI_acrs = []
            CI_amps = []
            if acrophase_ci is not None and isinstance(acrophase_ci, (list, tuple)) and len(acrophase_ci) == 2:
                CI_acrs = [acrophase_ci]

            if amplitude_ci is not None and isinstance(amplitude_ci, (list, tuple)) and len(amplitude_ci) == 2:
                CI_amps = [amplitude_ci]

            print(f"[DEBUG] Generating phase plot for Population-Mean: {test_name}")
            self._cosinor._generate_phase_plot(
                acrophases=[acrophase_rad],
                amplitudes=[amplitude],
                tests=[test_name],
                save_folder=plot_folder,
                prefix="population_",
                period=period,
                CI_acrs=CI_acrs if CI_acrs else None,
                CI_amps=CI_amps if CI_amps else None
            )

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="cosinorpy_population",
            mesor=mesor,
            amplitude=amplitude,
            acrophase=acrophase_rad,
            acrophase_hours=acrophase_hours,
            period=period,
            p_value=p_value,
            amplitude_ci=amplitude_ci,
            acrophase_ci=acrophase_ci,
            mesor_ci=mesor_ci,
            n_components=n_components_result,
            amplification=amplification,
            lin_comp=lin_comp_val,
            p_amplification=p_amplification,
            p_lin_comp=p_lin_comp,
            times=times,
            values=values,
            success=True
        )

    def _run_cosinorpy_count(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run CosinorPy count data analysis."""
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_count",
                success=False, message="CosinorPy not available"
            )

        from core.cosinor_analysis import ModelType

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)
        model_type_str = parameters.get('model_type', 'poisson')

        # Convert string to ModelType enum
        model_type = ModelType.POISSON if model_type_str.lower() == 'poisson' else ModelType.NEGBINOMIAL

        try:
            result = self._cosinor.fit_count_data(
                variable=variable,
                condition=condition,
                model_type=model_type,
                period=period,
                n_components=n_components
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cosinorpy_count",
                    success=False, message="Analysis failed"
                )

            # Extract parameters
            acrophase_rad = result.get('acrophase', 0)
            acrophase_hours = (acrophase_rad * period) / (2 * np.pi)
            if acrophase_hours < 0:
                acrophase_hours += period

            # Get times and values for plotting
            subset = data[data[condition_col] == condition]
            times = subset[time_col].values
            values = subset[variable].values

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cosinorpy_count",
                mesor=result.get('mesor'),
                amplitude=result.get('amplitude'),
                acrophase=acrophase_rad,
                acrophase_hours=acrophase_hours,
                period=period,
                p_value=result.get('p_value'),
                times=times,
                values=values,
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_count",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosinorpy_nonlinear(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> AnalysisResult:
        """Run CosinorPy nonlinear cosinor analysis."""
        print(f"[DEBUG] _run_cosinorpy_nonlinear called")
        print(f"[DEBUG] Variable: {variable}, Condition: {condition}")

        # Get folder for saving plots
        plot_folder = get_cosinorpy_plot_folder(data_file_path)
        if plot_folder:
            print(f"[DEBUG] CosinorPy plots will be saved to: {plot_folder}")

        if self._cosinor is None:
            print(f"[DEBUG] ERROR: CosinorPy not available")
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)
        amplification = parameters.get('amplification', None)
        lin_comp = parameters.get('lin_comp', None)
        auto_components = parameters.get('auto_components', False)

        print(f"[DEBUG] Parameters: period={period}, n_components={n_components}, amplification={amplification}, lin_comp={lin_comp}, auto_components={auto_components}")

        # Filter for the specific condition
        subset = data[data[condition_col] == condition].copy()
        print(f"[DEBUG] Filtered data shape: {subset.shape}")

        # Get times and values for result
        times = subset[time_col].values
        values = subset[variable].values

        try:
            print(f"[DEBUG] Loading data into CosinorAnalyzer...")
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes for INDEPENDENT analysis
            self._cosinor._raw_data = subset
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition]
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            # Convert to CosinorPy format
            print(f"[DEBUG] Converting to CosinorPy format...")
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()
            print(f"[DEBUG] Converted DataFrame shape: {self._cosinor._cosinorpy_df.shape}")
            print(f"[DEBUG] Data loaded successfully")

            # Check if auto-select components is enabled
            if auto_components:
                print(f"[DEBUG] Auto-select components enabled, calling nonlinear_cosinor_best_fit()...")
                result = self._cosinor.nonlinear_cosinor_best_fit(
                    variable=variable,
                    condition=condition,
                    period=period,
                    n_components_range=[1, 2, 3],
                    amplification=amplification,
                    lin_comp=lin_comp,
                    save_folder=plot_folder
                )
                print(f"[DEBUG] Best model selected: n_components={result.get('n_components_selected')}")
            else:
                print(f"[DEBUG] Calling self._cosinor.nonlinear_cosinor()...")
                result = self._cosinor.nonlinear_cosinor(
                    variable=variable,
                    condition=condition,
                    period=period,
                    n_components=n_components,
                    amplification=amplification,
                    lin_comp=lin_comp,
                    save_folder=plot_folder
                )

            print(f"[DEBUG] nonlinear_cosinor() returned")
            print(f"[DEBUG] Result is None: {result is None}")
            if result:
                print(f"[DEBUG] Result keys: {result.keys()}")
                print(f"[DEBUG] Result: {result}")

            if result is None:
                print(f"[DEBUG] ERROR: Analysis returned None")
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cosinorpy_nonlinear",
                    success=False, message="Analysis failed"
                )

            # Extract parameters
            acrophase_rad = result.get('acrophase', 0)
            acrophase_hours = (acrophase_rad * period) / (2 * np.pi)
            if acrophase_hours < 0:
                acrophase_hours += period

            # Get times and values for plotting
            subset = data[data[condition_col] == condition]
            times = subset[time_col].values
            values = subset[variable].values

            print(f"[DEBUG] Extracted values:")
            print(f"  mesor: {result.get('mesor')}")
            print(f"  amplitude: {result.get('amplitude')}")
            print(f"  acrophase_hours: {acrophase_hours}")
            print(f"  p_value: {result.get('p_value')}")
            print(f"  amplification: {result.get('amplification')}")
            print(f"  lin_comp: {result.get('lin_comp')}")

            # Generate phase plot if we have valid acrophase and save folder
            if plot_folder and acrophase_rad is not None and result.get('amplitude') is not None:
                test_name = f"{variable}_{condition}"

                # Extract confidence intervals if available
                CI_acrs = []
                CI_amps = []
                if 'acrophase_ci' in result:
                    ci_acr = result['acrophase_ci']
                    if ci_acr is not None and isinstance(ci_acr, (list, tuple)) and len(ci_acr) == 2:
                        CI_acrs = [ci_acr]

                if 'amplitude_ci' in result:
                    ci_amp = result['amplitude_ci']
                    if ci_amp is not None and isinstance(ci_amp, (list, tuple)) and len(ci_amp) == 2:
                        CI_amps = [ci_amp]

                print(f"[DEBUG] Generating phase plot for Nonlinear Cosinor: {test_name}")
                self._cosinor._generate_phase_plot(
                    acrophases=[acrophase_rad],
                    amplitudes=[result.get('amplitude')],
                    tests=[test_name],
                    save_folder=plot_folder,
                    prefix="nonlinear_",
                    period=period,
                    CI_acrs=CI_acrs if CI_acrs else None,
                    CI_amps=CI_amps if CI_amps else None
                )

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cosinorpy_nonlinear",
                mesor=result.get('mesor'),
                amplitude=result.get('amplitude'),
                acrophase=acrophase_rad,
                acrophase_hours=acrophase_hours,
                period=period,
                p_value=result.get('p_value'),
                r_squared=result.get('r_squared'),
                n_components=result.get('n_components'),
                amplification=result.get('amplification'),
                lin_comp=result.get('lin_comp'),
                p_amplification=result.get('p_amplification'),
                p_lin_comp=result.get('p_lin_comp'),
                times=times,
                values=values,
                success=True
            )
        except Exception as e:
            print(f"[DEBUG] ERROR in nonlinear cosinor: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosinorpy_periodogram(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str] = None
    ) -> AnalysisResult:
        """Run CosinorPy periodogram analysis - generates plot in cosinorpy_plots folder."""
        print(f"[DEBUG] _run_cosinorpy_periodogram called")
        print(f"  variable: {variable}, condition: {condition}")
        print(f"  per_type: {parameters.get('per_type')}")
        print(f"  max_per: {parameters.get('max_per')}")
        print(f"  prominent: {parameters.get('prominent', False)}")

        try:
            # Get plot folder
            plot_folder = get_cosinorpy_plot_folder(data_file_path)
            print(f"[DEBUG] Plot folder: {plot_folder}")

            # Extract data for this condition
            subset = data[data[condition_col] == condition]

            # Load data into CosinorAnalyzer
            from core.cosinor_analysis import AnalysisMode

            # Set the internal data attributes for INDEPENDENT analysis
            self._cosinor._raw_data = subset
            self._cosinor._variables = [variable]
            self._cosinor._conditions = [condition]
            self._cosinor._time_col = time_col
            self._cosinor._condition_col = condition_col
            self._cosinor._replicate_col = None
            self._cosinor._subject_col = None
            self._cosinor.analysis_mode = AnalysisMode.INDEPENDENT

            # Convert to CosinorPy format
            self._cosinor._cosinorpy_df = self._cosinor._convert_to_cosinorpy_format()

            # Run CosinorPy's periodogram function (generates plot)
            per_type = parameters.get('per_type', 'per')
            max_per = parameters.get('max_per', 240.0)
            prominent = parameters.get('prominent', False)

            result = self._cosinor.periodogram(
                variable=variable,
                condition=condition,
                per_type=per_type,
                max_per=max_per,
                prominent=prominent,
                save_folder=plot_folder
            )

            print(f"[DEBUG] Periodogram result: {result}")

            # Return result (no table data, just message)
            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cosinorpy_periodogram",
                success=result.get('success', False),
                message=result.get('message', 'Unknown error')
            )

        except Exception as e:
            print(f"[DEBUG] ERROR in periodogram: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_periodogram",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_spectral_analysis(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Spectral Analysis (Periodogram) from Rhythm Analysis module.

        Uses scipy.signal for spectral analysis with multiple methods:
        - 'per': Standard FFT periodogram
        - 'welch': Welch's method (averaged periodogram)
        - 'lombscargle': Lomb-Scargle (for unevenly sampled data)
        """
        from scipy import signal

        try:
            per_type = parameters.get('per_type', 'per')
            max_per = parameters.get('max_per', 240.0)
            detrending = parameters.get('detrending', True)
            prominent = parameters.get('prominent', False)

            X = times
            Y = values

            if per_type == 'per':
                # Standard FFT periodogram - need evenly sampled data
                X_u = np.unique(X)
                Y_u = []
                for x_u in X_u:
                    Y_u.append(np.median(Y[X == x_u]))
                Y_u = np.array(Y_u)

                if len(X_u) > 1:
                    time_diffs = np.diff(X_u)
                    sampling_interval = np.median(time_diffs)
                    sampling_f = 1 / sampling_interval
                else:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="spectral_analysis",
                        success=False, message="Need at least 2 time points"
                    )

                f, Pxx_den = signal.periodogram(
                    Y_u, sampling_f, detrend='constant' if detrending else False
                )

            elif per_type == 'welch':
                # Welch's method
                X_u = np.unique(X)
                Y_u = []
                for x_u in X_u:
                    Y_u.append(np.median(Y[X == x_u]))
                Y_u = np.array(Y_u)

                if len(X_u) > 1:
                    time_diffs = np.diff(X_u)
                    sampling_interval = np.median(time_diffs)
                    sampling_f = 1 / sampling_interval
                else:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="spectral_analysis",
                        success=False, message="Need at least 2 time points"
                    )

                f, Pxx_den = signal.welch(
                    Y_u, sampling_f, detrend='constant' if detrending else False
                )

            elif per_type == 'lombscargle':
                # Lomb-Scargle can handle uneven sampling
                min_per = 2
                f = np.linspace(1/max_per, 1/min_per, 1000)
                Y_proc = Y - np.mean(Y) if detrending else Y
                Pxx_den = signal.lombscargle(X, Y_proc, f)
                Y_u = Y_proc  # For significance calculation

            else:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="spectral_analysis",
                    success=False, message=f"Invalid periodogram type: {per_type}"
                )

            # Convert frequency to period
            if f[0] == 0:
                per = 1 / f[1:]
                Pxx = Pxx_den[1:]
            else:
                per = 1 / f
                Pxx = Pxx_den

            # Filter to max_per
            mask = per <= max_per
            Pxx = Pxx[mask]
            per = per[mask]

            # Calculate significance threshold (Refinetti et al. 2007)
            p_t = 0.05
            N = len(Y_u)
            T = (1 - (p_t/N)**(1/(N-1))) * sum(Pxx_den)

            # Find dominant period (highest power peak — always computed)
            max_idx = np.argmax(Pxx)
            dominant_per = float(per[max_idx])

            # Find significant peaks only when requested via 'prominent' flag
            significant_peaks = []
            if prominent:
                if len(Pxx) >= 10:
                    locs, heights = signal.find_peaks(Pxx, height=T)
                    if len(locs) > 0:
                        heights = heights['peak_heights']
                        s = list(zip(heights, locs))
                        s.sort(reverse=True)
                        significant_peaks = [float(per[loc]) for _, loc in s]
                else:
                    significant_indices = np.where(Pxx > T)[0]
                    significant_peaks = [float(per[i]) for i in significant_indices]

            msg = (f"Found {len(significant_peaks)} significant peaks"
                   if prominent else f"Dominant period: {dominant_per:.1f}h")

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="spectral_analysis",
                dominant_period=dominant_per,
                power=float(Pxx[max_idx]),
                period=dominant_per,
                periods=per,
                power_spectrum=Pxx,
                threshold=T,
                significant_peaks=significant_peaks,
                times=times,
                values=values,
                success=True,
                message=msg
            )

        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="spectral_analysis",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_ar_jtk(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Run AR-JTK Cycle analysis (JTK with autocorrelation correction)."""
        from .rhythm_analysis import _run_ar_jtk, _run_discrete_jtk_all_periods
        from .rhythm_analysis import _is_white_noise_ranked, _prewhiten_ranked_residuals, _generate_triangle_template_time
        import pandas as pd

        # Get period range - support both period_range tuple and period list
        period = parameters.get('period', None)
        if period is not None:
            if isinstance(period, list):
                period_range = [float(p) for p in period]
            else:
                period_range = [float(period)]
        else:
            period_range = list(range(20, 29))  # Default circadian range

        # Get asymmetry parameter (default: 0.5 for symmetric)
        asymmetry = parameters.get('asymmetry', 0.5)
        asymmetries = [asymmetry] if asymmetry else [0.5]

        ar_lag = parameters.get('ar_lag', 1)
        ljungbox_lag = parameters.get('ljungbox_lag', 10)
        prewhiten = parameters.get('prewhiten', False)

        try:
            # AR-JTK requires a proper time series (one value per timepoint).
            # Independent data with multiple replicates per timepoint would cause
            # the Ljung-Box test to flag spurious within-timepoint autocorrelation,
            # leading to incorrect prewhitening and a wrong acrophase.
            # Fix: average replicates at each unique timepoint before running AR-JTK.
            unique_times = np.unique(times)
            if len(unique_times) < len(times):
                avg_values = np.array([np.mean(values[times == t]) for t in unique_times])
                series = pd.Series(avg_values, index=unique_times)
            else:
                series = pd.Series(values, index=times)

            result, autocorr_detected = _run_ar_jtk(
                series,
                period_range=period_range,
                asymmetries=asymmetries,
                ar_lag=ar_lag,
                ljungbox_lag=ljungbox_lag,
                force_prewhiten=prewhiten
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="ar_jtk",
                    success=False, message="Analysis failed"
                )

            msg = "AR correction applied" if autocorr_detected else "No autocorrelation detected"
            show_all = len(period_range) > 1

            if show_all:
                # Re-run all-periods on the (possibly prewhitened) series used internally
                # We use the same prewhitening logic as _run_ar_jtk
                import numpy as _np
                unique_t = _np.unique(times)
                if len(unique_t) < len(times):
                    avg_v = _np.array([_np.mean(values[times == t]) for t in unique_t])
                    series_ap = pd.Series(avg_v, index=unique_t)
                else:
                    series_ap = pd.Series(values, index=times)

                # Always run all-periods on the original (non-prewhitened) series.
                # Running on the prewhitened series (rank space) causes two bugs:
                # (1) Amplitude in rank-space units instead of original data units.
                # (2) Phase inversion: when the first JTK selects a template with tau<0,
                #     the prewhitened series peaks where the original data troughs, so
                #     the second JTK reports the trough time as the acrophase (off by T/2).
                per_period = _run_discrete_jtk_all_periods(
                    series_ap, period_range=period_range, asymmetries=asymmetries
                )
                amp = (series.values.max() - series.values.min()) / 2  # from original data
                best_p = min(r.p_value for r in per_period)
                results = []
                for r in per_period:
                    results.append(AnalysisResult(
                        variable=variable, condition=condition,
                        method="ar_jtk",
                        period=r.period,
                        acrophase_hours=r.acrophase,
                        p_value=r.bh_p_value,
                        raw_p_value=r.p_value,
                        bonf_p_value=r.bonf_p_value,
                        tau=r.tau,
                        lag=r.lag,
                        asymmetry=r.asymmetry,
                        n_tests=r.n_tests,
                        amplitude=amp,  # from original data (not rank space)
                        times=times, values=values,
                        success=True,
                        message=msg,
                        best_model="Yes" if r.p_value == best_p else "No"
                    ))
                return results

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="ar_jtk",
                period=result.period,
                p_value=result.bh_p_value,
                raw_p_value=result.p_value,
                bonf_p_value=result.bonf_p_value,
                acrophase_hours=result.acrophase,
                amplitude=result.amplitude,
                tau=result.tau,
                lag=result.lag,
                asymmetry=result.asymmetry,
                n_tests=result.n_tests,
                times=times,
                values=values,
                success=True,
                message=msg
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="ar_jtk",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosine_kendall(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Run Cosine-Kendall nonparametric analysis."""
        from .rhythm_analysis import _run_cosine_kendall, _run_cosine_kendall_all_periods, DefaultPeriodRanges
        import pandas as pd

        # Get period range - support both period_range tuple and period list
        period = parameters.get('period', None)
        if period is not None:
            if isinstance(period, list):
                period_range = [float(p) for p in period]
            else:
                period_range = [float(period)]
        else:
            period_range = DefaultPeriodRanges.CIRCADIAN

        # Get resolution/interval parameter
        interval = parameters.get('resolution', parameters.get('interval', 1.0))

        try:
            # Create series with time as index
            series = pd.Series(values, index=times)

            result = _run_cosine_kendall(series, period_range=period_range, interval=interval)

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cosine_kendall",
                    success=False, message="Analysis failed"
                )

            show_all = len(period_range) > 1

            if show_all:
                per_period = _run_cosine_kendall_all_periods(
                    series, period_range=period_range, interval=interval
                )
                best_p = min(r.p_value for r in per_period)
                results = []
                for r in per_period:
                    results.append(AnalysisResult(
                        variable=variable, condition=condition,
                        method="cosine_kendall",
                        period=r.period,
                        acrophase_hours=r.acrophase,
                        p_value=r.bh_p_value,
                        raw_p_value=r.p_value,
                        bonf_p_value=r.bonf_p_value,
                        tau=r.tau,
                        lag=r.lag,
                        asymmetry=r.asymmetry,
                        n_tests=r.n_tests,
                        amplitude=r.amplitude,
                        times=times, values=values,
                        success=True,
                        best_model="Yes" if r.p_value == best_p else "No"
                    ))
                return results

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cosine_kendall",
                period=result.period,
                p_value=result.bh_p_value,
                raw_p_value=result.p_value,
                bonf_p_value=result.bonf_p_value,
                acrophase_hours=result.acrophase,
                amplitude=result.amplitude,
                tau=result.tau,
                lag=result.lag,
                asymmetry=result.asymmetry,
                n_tests=result.n_tests,
                times=times,
                values=values,
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosine_kendall",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_fourier_f24(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Fourier F24 analysis (effect size measure).

        Uses the two-replicate method (with correlation_r) when a 'replicate'
        or 'subject' column is present and has at least 2 unique values.
        Falls back to the single-series method otherwise.
        """
        from .rhythm_analysis import _compute_fourier_f24, _compute_fourier_f24_with_replicates

        target_period = parameters.get('target_period', 24.0)
        n_permutations = parameters.get('n_permutations', 1000)

        try:
            # Filter data for this condition
            cond_data = data[data[condition_col] == condition].copy()

            # Raw times/values for storing in result (used for plotting)
            all_times = cond_data[time_col].values.astype(float)
            all_values = cond_data[variable].values.astype(float)
            valid = ~(np.isnan(all_times) | np.isnan(all_values))
            all_times = all_times[valid]
            all_values = all_values[valid]

            # Detect replicate column
            rep_col = None
            for col in ('replicate', 'subject'):
                if col in cond_data.columns:
                    rep_col = col
                    break

            use_replicates = rep_col is not None and cond_data[rep_col].nunique() >= 2

            if use_replicates:
                rep_ids = cond_data[rep_col].unique()

                if len(rep_ids) == 2:
                    rep1 = cond_data[cond_data[rep_col] == rep_ids[0]]
                    rep2 = cond_data[cond_data[rep_col] == rep_ids[1]]
                    times1 = rep1[time_col].values.astype(float)
                    values1 = rep1[variable].values.astype(float)
                    times2 = rep2[time_col].values.astype(float)
                    values2 = rep2[variable].values.astype(float)
                else:
                    # More than 2 replicates: randomly split into two groups and average
                    rng = np.random.default_rng()
                    indices = np.arange(len(rep_ids))
                    rng.shuffle(indices)
                    mid = len(indices) // 2
                    group1_ids = rep_ids[indices[:mid]]
                    group2_ids = rep_ids[indices[mid:]]

                    g1_avg = (cond_data[cond_data[rep_col].isin(group1_ids)]
                              .groupby(time_col)[variable].mean().reset_index())
                    g2_avg = (cond_data[cond_data[rep_col].isin(group2_ids)]
                              .groupby(time_col)[variable].mean().reset_index())

                    times1 = g1_avg[time_col].values.astype(float)
                    values1 = g1_avg[variable].values.astype(float)
                    times2 = g2_avg[time_col].values.astype(float)
                    values2 = g2_avg[variable].values.astype(float)

                # Remove NaN per replicate
                mask1 = ~(np.isnan(times1) | np.isnan(values1))
                mask2 = ~(np.isnan(times2) | np.isnan(values2))
                times1, values1 = times1[mask1], values1[mask1]
                times2, values2 = times2[mask2], values2[mask2]

                result = _compute_fourier_f24_with_replicates(
                    times1, values1, times2, values2,
                    target_period=target_period,
                    n_permutations=n_permutations
                )
            else:
                result = _compute_fourier_f24(
                    all_times, all_values,
                    target_period=target_period,
                    n_permutations=n_permutations
                )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="fourier_f24",
                    success=False, message="Analysis failed"
                )

            # F24 > 2 is typically considered rhythmic (Wijnen et al., 2006)
            is_rhythmic = result.f24_score > 2.0
            rep_note = " (2-replicate method)" if use_replicates else ""
            msg = f"F24={result.f24_score:.2f} ({'rhythmic' if is_rhythmic else 'not rhythmic'}, threshold=2.0){rep_note}"

            # Convert frequencies to periods for plotting
            frequencies = result.frequencies
            periods = None
            if frequencies is not None:
                valid_freq = frequencies > 0
                periods = np.zeros_like(frequencies)
                periods[valid_freq] = 1.0 / frequencies[valid_freq]

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="fourier_f24",
                period=result.target_period,
                dominant_period=result.dominant_period,
                power=result.f24_score,  # F24 score as effect size
                dominant_power=result.dominant_power,
                target_power=result.target_power,
                correlation_r=result.correlation_r,
                times=all_times,
                values=all_values,
                periods=periods,
                power_spectrum=result.power_spectrum,
                success=True,
                message=msg
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="fourier_f24",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cwt(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Continuous Wavelet Transform analysis.

        Uses per-subject CWT when a 'replicate' or 'subject' column is present
        and has at least 2 unique values, then aggregates by averaging global
        wavelet spectra across subjects.  Falls back to single-series CWT otherwise.
        """
        from .rhythm_analysis import _compute_cwt

        sampling_interval_param = parameters.get('sampling_interval', None)
        wavelet = parameters.get('wavelet', 'cmor1.5-1.0')
        period_range = parameters.get('period_range', (20.0, 28.0))

        try:
            cond_data = data[data[condition_col] == condition].copy()

            # Full times/values for storing in result (used for raw-data plot)
            all_times = cond_data[time_col].values.astype(float)
            all_values = cond_data[variable].values.astype(float)
            valid = ~(np.isnan(all_times) | np.isnan(all_values))
            all_times = all_times[valid]
            all_values = all_values[valid]

            # Detect replicate/subject column
            rep_col = None
            for col in ('replicate', 'subject'):
                if col in cond_data.columns:
                    rep_col = col
                    break

            use_subjects = rep_col is not None and cond_data[rep_col].nunique() >= 2

            def _auto_si(t: np.ndarray) -> float:
                """Auto-detect sampling interval from a time array."""
                diffs = np.diff(np.sort(t))
                nonzero = diffs[diffs > 0]
                return float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

            if use_subjects:
                subject_ids = cond_data[rep_col].unique()
                subject_results = []

                for sid in subject_ids:
                    subj = cond_data[cond_data[rep_col] == sid]
                    t = subj[time_col].values.astype(float)
                    v = subj[variable].values.astype(float)
                    mask = ~(np.isnan(t) | np.isnan(v))
                    t, v = t[mask], v[mask]
                    if len(t) < 4:
                        continue
                    si = sampling_interval_param if sampling_interval_param is not None else _auto_si(t)
                    r = _compute_cwt(t, v, sampling_interval=si, wavelet=wavelet, period_range=period_range)
                    if r is not None and not np.isnan(r.dominant_period):
                        subject_results.append(r)

                if not subject_results:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="cwt", success=False,
                        message="CWT failed for all subjects"
                    )

                # Average global wavelet spectra (power averaged over time per period)
                # global_power shape is (n_periods,) — same for all subjects since
                # period_range is identical; only the time dimension may differ.
                global_powers = [
                    r.power_matrix.mean(axis=1)
                    for r in subject_results
                    if r.power_matrix is not None
                ]

                if global_powers and all(len(gp) == len(global_powers[0]) for gp in global_powers):
                    avg_global = np.mean(global_powers, axis=0)
                    period_array = subject_results[0].periods
                    dominant_period = float(period_array[np.argmax(avg_global)])
                    mean_power = float(np.mean(avg_global))
                else:
                    # Fallback: average scalar outputs
                    dominant_period = float(np.mean([r.dominant_period for r in subject_results]))
                    mean_power = float(np.mean([r.mean_power for r in subject_results]))

                period_variation = float(np.mean([r.period_variation for r in subject_results]))
                amplitude_modulations = int(round(np.mean([r.amplitude_modulations for r in subject_results])))

                n = len(subject_results)
                msg = (f"Multi-subject CWT (n={n}): dominant period={dominant_period:.2f}h, "
                       f"period variation={period_variation:.2f}h, "
                       f"amplitude modulations={amplitude_modulations}")

                # Use first subject's scalogram as representative for the plot
                rep = subject_results[0]
                return AnalysisResult(
                    variable=variable, condition=condition, method="cwt",
                    period=dominant_period, dominant_period=dominant_period,
                    power=mean_power,
                    period_variation=period_variation,
                    amplitude_modulations=amplitude_modulations,
                    times=all_times, values=all_values,
                    scalogram_power=rep.power_matrix,
                    scalogram_times=rep.times,
                    scalogram_periods=rep.periods,
                    success=True, message=msg
                )

            else:
                # Single series: original behaviour
                si = sampling_interval_param if sampling_interval_param is not None else _auto_si(all_times)
                result = _compute_cwt(
                    all_times, all_values,
                    sampling_interval=si,
                    wavelet=wavelet,
                    period_range=period_range
                )

                if result is None:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="cwt", success=False, message="Analysis failed"
                    )

                if np.isnan(result.dominant_period):
                    total_duration = float(all_times.max() - all_times.min()) if len(all_times) > 1 else 0.0
                    max_analyzable = total_duration / 3.0
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="cwt", success=False,
                        message=(f"Time series too short for period range "
                                 f"[{period_range[0]:.1f}, {period_range[1]:.1f}]h. "
                                 f"Data spans {total_duration:.1f}h; maximum analyzable period "
                                 f"is {max_analyzable:.1f}h. Reduce the period range or use "
                                 f"longer time series.")
                    )

                return AnalysisResult(
                    variable=variable, condition=condition, method="cwt",
                    period=result.dominant_period,
                    dominant_period=result.dominant_period,
                    power=result.mean_power,
                    period_variation=result.period_variation,
                    amplitude_modulations=result.amplitude_modulations,
                    times=all_times, values=all_values,
                    scalogram_power=result.power_matrix,
                    scalogram_times=result.times,
                    scalogram_periods=result.periods,
                    success=True,
                    message=f"Period variation: {result.period_variation:.2f}h, Amplitude modulations: {result.amplitude_modulations}"
                )

        except ImportError:
            return AnalysisResult(
                variable=variable, condition=condition, method="cwt",
                success=False, message="PyWavelets not installed. Install with: pip install PyWavelets"
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition, method="cwt",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_consensus_ai(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Consensus Rhythmicity Score (AI) meta-classifier."""
        import json as _json

        if self._consensus is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="consensus_ai",
                success=False,
                message="scikit-learn not available. Install with: pip install scikit-learn"
            )

        # Filter data for this condition
        mask = data[condition_col] == condition
        filtered = data[mask].copy()

        if len(filtered) == 0:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="consensus_ai",
                success=False,
                message=f"No data for condition: {condition}"
            )

        times = filtered[time_col].values.astype(float)
        values = filtered[variable].values.astype(float)

        # Remove NaN
        valid_mask = ~(np.isnan(times) | np.isnan(values))
        times = times[valid_mask]
        values = values[valid_mask]

        if len(times) < 4:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="consensus_ai",
                success=False,
                message="Insufficient data points (minimum 4 required)"
            )

        # Average replicates at each timepoint and compute SEM
        unique_times = np.unique(times)
        avg_values = np.array([values[times == t].mean() for t in unique_times])
        sem_values = np.array([
            values[times == t].std(ddof=1) / np.sqrt(np.sum(times == t))
            if np.sum(times == t) > 1 else 0.0
            for t in unique_times
        ])

        try:
            result = self._consensus.predict(
                unique_times, avg_values, data, variable, condition,
                time_col, condition_col, parameters
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="consensus_ai",
                success=False,
                message=f"Consensus AI error: {str(e)}"
            )

        if result.get('error'):
            return AnalysisResult(
                variable=variable, condition=condition,
                method="consensus_ai",
                success=False,
                message=result['error']
            )

        probability = result['probability']
        classification = result['classification']
        features = result['features']

        # Build JSON message with full details for the results panel
        details_json = _json.dumps({
            'classification': classification,
            'probability': probability,
            'method_results': result['method_results'],
            'feature_importances': result['feature_importances'],
            'sub_method_details': result['sub_method_details'],
            'sem_values': sem_values.tolist(),
        })

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="consensus_ai",
            # p_value = 1 - probability so lower = more significant (standard sorting)
            p_value=1.0 - probability,
            # r_squared stores the raw probability score (0-1)
            r_squared=probability,
            # Cosinor parameters from sub-methods
            amplitude=features.get('cosinor_amplitude'),
            period=features.get('cosinor_period', 24.0),
            # Method agreement as tau field
            tau=features.get('method_agreement'),
            # Period concordance
            period_variation=features.get('period_concordance'),
            # Raw data for plotting
            times=unique_times,
            values=avg_values,
            success=True,
            message=details_json,
        )

    def _run_lme(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Cosinor Linear Mixed Effects model analysis for circadian rhythms."""
        print("[DEBUG _run_lme] Starting LME analysis...")
        from .rhythm_analysis import _fit_lme_model

        period = parameters.get('period', 24.0)
        if isinstance(period, (list, tuple)):
            period_list = [float(p) for p in period] if len(period) > 0 else [24.0]
        else:
            period_list = [float(period)]
        random_effect = parameters.get('random_effect', 'replicate')

        try:
            # Filter data for this condition
            if condition is not None:
                df = data[data[condition_col] == condition].copy()
            else:
                df = data.copy()

            # Resolve random effect column
            if random_effect not in df.columns:
                potential_groups = ['replicate', 'subject', 'animal', 'id', 'sample', 'day']
                found_group = None
                for col in potential_groups:
                    if col in df.columns:
                        found_group = col
                        break
                if found_group is None:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="lme", success=False,
                        message=f"Random effect column '{random_effect}' not found. "
                                f"Available columns: {list(df.columns)}. "
                                f"Please specify a grouping variable (e.g., subject ID, replicate)."
                    )
                random_effect = found_group

            times = df[time_col].values.astype(float)
            values = df[variable].values.astype(float)
            groups = df[random_effect].values

            if len(times) < 5:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="lme", success=False,
                    message="Not enough data points for LME analysis (need at least 5)"
                )

            unique_groups = np.unique(groups)
            if len(unique_groups) < 2:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="lme", success=False,
                    message=f"LME requires at least 2 groups for random effect. "
                            f"Found only {len(unique_groups)} unique value(s) in '{random_effect}' column."
                )

            def _make_result(lme_res, p):
                return AnalysisResult(
                    variable=variable, condition=condition, method="lme",
                    mesor=lme_res.mesor,
                    amplitude=lme_res.amplitude,
                    acrophase=lme_res.acrophase_rad,
                    acrophase_hours=lme_res.acrophase,
                    period=lme_res.period,
                    p_value=lme_res.p_value,
                    r_squared=lme_res.r_squared,
                    aic=lme_res.aic,
                    bic=lme_res.bic,
                    random_effect_var=lme_res.random_effect_var,
                    residual_var=lme_res.residual_var,
                    times=times, values=values,
                    success=True,
                    message=f"Random effect: {random_effect} ({len(unique_groups)} groups), period={p:.2f}h"
                )

            if len(period_list) == 1:
                result = _fit_lme_model(
                    times=times, values=values, random_groups=groups, period=period_list[0]
                )
                if result is None:
                    return AnalysisResult(
                        variable=variable, condition=condition,
                        method="lme", success=False, message="Analysis failed"
                    )
                return _make_result(result, period_list[0])

            # Multi-period scan: fit at each period, mark best by lowest p-value
            per_period = []
            for p in period_list:
                try:
                    r = _fit_lme_model(
                        times=times, values=values, random_groups=groups, period=p
                    )
                    if r is not None:
                        per_period.append((p, r))
                except Exception:
                    continue

            if not per_period:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="lme", success=False,
                    message="LME analysis failed for all tested periods"
                )

            best_p_val = min(r.p_value for _, r in per_period)
            results = []
            for p, r in per_period:
                ar = _make_result(r, p)
                ar.best_model = "Yes" if r.p_value == best_p_val else "No"
                results.append(ar)
            return results

        except Exception as e:
            import traceback
            print(f"[DEBUG _run_lme] Exception caught: {e}")
            print(traceback.format_exc())
            return AnalysisResult(
                variable=variable, condition=condition,
                method="lme", success=False, message=f"Error: {str(e)}"
            )

    # ========================================================================
    # COSINORPY NEW REFACTORED METHODS
    # ========================================================================

    def _convert_to_cosinorpy_format(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        subject_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert data to CosinorPy format (x, y, test).

        For independent data: test = "variable_condition"
        For dependent data: test = "variable_condition_repN" (one per subject)
        """
        print(f"[DEBUG _convert_to_cosinorpy_format] variable={variable}, condition={condition}")
        print(f"[DEBUG] subject_col={subject_col}")

        # Filter for this condition
        df_filtered = data[data[condition_col] == condition].copy()

        if subject_col and subject_col in df_filtered.columns:
            # Dependent data: create test_repN for each subject
            # CRITICAL: For dependent data, test name must be "condition_repN"
            # NOT "variable_condition_repN" because CosinorPy uses:
            # df[df.test.str.startswith(f'{condition}_rep')]
            df_cosinorpy = pd.DataFrame()
            subjects = df_filtered[subject_col].unique()

            print(f"[DEBUG] Found {len(subjects)} subjects: {subjects}")

            for i, subject in enumerate(subjects, 1):
                df_subject = df_filtered[df_filtered[subject_col] == subject].copy()
                df_subject_cosinorpy = pd.DataFrame({
                    'x': df_subject[time_col].values,
                    'y': df_subject[variable].values,
                    'test': f"{condition}_rep{i}"  # ONLY condition name, no variable!
                })
                df_cosinorpy = pd.concat([df_cosinorpy, df_subject_cosinorpy], ignore_index=True)

        else:
            # Independent data: simple conversion
            df_cosinorpy = pd.DataFrame({
                'x': df_filtered[time_col].values,
                'y': df_filtered[variable].values,
                'test': f"{variable}_{condition}"
            })

        print(f"[DEBUG] Converted to CosinorPy format: {len(df_cosinorpy)} rows")
        return df_cosinorpy

    def _convert_to_cosinorpy_format_population(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        subject_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert data to CosinorPy format for population/dependent data.

        IMPORTANT: CosinorPy population functions use test.split("_")[0] to get the base name,
        so we must use format "basename_repN" where basename has NO underscores.
        We combine variable and condition with a hyphen.

        Format: test = "variable-condition_repN"
        """
        print(f"[DEBUG _convert_to_cosinorpy_format_population] variable={variable}, condition={condition}")

        # Filter for this condition
        df_filtered = data[data[condition_col] == condition].copy()

        # Create base name WITHOUT underscores (CosinorPy splits on "_")
        # Use hyphen to join variable and condition
        base_name = f"{variable}-{condition}"

        # Determine how to identify replicates
        rep_col = None
        if subject_col and subject_col in df_filtered.columns:
            rep_col = subject_col
        elif 'subject' in df_filtered.columns:
            rep_col = 'subject'
        elif 'replicate' in df_filtered.columns:
            rep_col = 'replicate'

        if rep_col:
            # Create test_repN for each replicate
            df_cosinorpy = pd.DataFrame()
            replicates = df_filtered[rep_col].unique()

            print(f"[DEBUG] Found {len(replicates)} replicates using column '{rep_col}'")
            print(f"[DEBUG] Using base_name='{base_name}' (no underscores for CosinorPy compatibility)")

            for i, rep in enumerate(replicates, 1):
                df_rep = df_filtered[df_filtered[rep_col] == rep].copy()
                df_rep_cosinorpy = pd.DataFrame({
                    'x': df_rep[time_col].values,
                    'y': df_rep[variable].values,
                    'test': f"{base_name}_rep{i}"
                })
                df_cosinorpy = pd.concat([df_cosinorpy, df_rep_cosinorpy], ignore_index=True)

        else:
            # No replicate info - treat as single replicate
            print(f"[DEBUG] No replicate column found, treating as single replicate")
            df_cosinorpy = pd.DataFrame({
                'x': df_filtered[time_col].values,
                'y': df_filtered[variable].values,
                'test': f"{base_name}_rep1"
            })

        print(f"[DEBUG] Converted to CosinorPy population format: {len(df_cosinorpy)} rows")
        print(f"[DEBUG] Test names: {df_cosinorpy['test'].unique().tolist()}")
        return df_cosinorpy

    def _convert_to_cosinorpy_format_population_all_conditions(
        self,
        data: pd.DataFrame,
        variable: str,
        time_col: str,
        condition_col: str,
        subject_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Convert data to CosinorPy format for ALL conditions (for comparison).

        IMPORTANT: CosinorPy population functions use test.split("_")[0] to get the base name,
        so we must use format "basename_repN" where basename has NO underscores.

        Format: test = "variable-condition_repN" for each condition.
        """
        print(f"[DEBUG _convert_to_cosinorpy_format_population_all_conditions] variable={variable}")

        conditions = data[condition_col].unique()
        print(f"[DEBUG] Converting {len(conditions)} conditions")

        # Determine replicate column
        rep_col = None
        if subject_col and subject_col in data.columns:
            rep_col = subject_col
        elif 'subject' in data.columns:
            rep_col = 'subject'
        elif 'replicate' in data.columns:
            rep_col = 'replicate'

        df_cosinorpy = pd.DataFrame()

        for condition in conditions:
            df_filtered = data[data[condition_col] == condition].copy()
            # Use hyphen to join variable and condition (no underscores in base name)
            base_name = f"{variable}-{condition}"

            if rep_col:
                replicates = df_filtered[rep_col].unique()
                for i, rep in enumerate(replicates, 1):
                    df_rep = df_filtered[df_filtered[rep_col] == rep].copy()
                    df_rep_cosinorpy = pd.DataFrame({
                        'x': df_rep[time_col].values,
                        'y': df_rep[variable].values,
                        'test': f"{base_name}_rep{i}"
                    })
                    df_cosinorpy = pd.concat([df_cosinorpy, df_rep_cosinorpy], ignore_index=True)
            else:
                # No replicate info
                df_cond_cosinorpy = pd.DataFrame({
                    'x': df_filtered[time_col].values,
                    'y': df_filtered[variable].values,
                    'test': f"{base_name}_rep1"
                })
                df_cosinorpy = pd.concat([df_cosinorpy, df_cond_cosinorpy], ignore_index=True)

        print(f"[DEBUG] Converted all conditions to CosinorPy format: {len(df_cosinorpy)} rows")
        print(f"[DEBUG] Test names: {df_cosinorpy['test'].unique().tolist()}")
        return df_cosinorpy

    def _convert_to_cosinorpy_format_all_conditions(
        self,
        data: pd.DataFrame,
        variable: str,
        time_col: str,
        condition_col: str
    ) -> pd.DataFrame:
        """
        Convert data to CosinorPy format for ALL conditions (for independent data comparison).

        For independent data: test = "variable-condition" for each condition.
        Uses hyphen to avoid CosinorPy's underscore splitting issues.
        """
        print(f"[DEBUG _convert_to_cosinorpy_format_all_conditions] variable={variable}")

        conditions = data[condition_col].unique()
        print(f"[DEBUG] Converting {len(conditions)} conditions")

        df_cosinorpy = pd.DataFrame()

        for condition in conditions:
            df_filtered = data[data[condition_col] == condition].copy()
            # Use hyphen to join variable and condition
            test_name = f"{variable}-{condition}"

            df_cond_cosinorpy = pd.DataFrame({
                'x': df_filtered[time_col].values,
                'y': df_filtered[variable].values,
                'test': test_name
            })
            df_cosinorpy = pd.concat([df_cosinorpy, df_cond_cosinorpy], ignore_index=True)

        print(f"[DEBUG] Converted all conditions to CosinorPy format: {len(df_cosinorpy)} rows")
        print(f"[DEBUG] Test names: {df_cosinorpy['test'].unique().tolist()}")
        return df_cosinorpy

    def _run_cosinorpy_periodogram_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> AnalysisResult:
        """Run periodogram analysis for period detection."""
        print(f"[DEBUG _run_cosinorpy_periodogram_new] ENTERED")
        print(f"[DEBUG] variable={variable}, condition={condition}")
        print(f"[DEBUG] data.shape={data.shape}")
        print(f"[DEBUG] parameters keys={list(parameters.keys())}")

        if self._cosinor is None:
            print(f"[DEBUG] ERROR: CosinorPy not available")
            return AnalysisResult(
                variable="Periodogram", condition="All",
                method="cosinorpy_periodogram",
                success=False, message="CosinorPy not available"
            )

        print(f"[DEBUG _run_cosinorpy_periodogram_new] CosinorPy is available")

        try:
            # Get plot folder
            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            # Get selected variables and conditions from parameters
            selected_variables = parameters.get('selected_variables', None)
            selected_conditions = parameters.get('selected_conditions', None)

            # If not provided in parameters, use all available
            if selected_variables is None:
                selected_variables = [col for col in data.columns
                                    if col not in [time_col, condition_col, 'subject', 'replicate']]

            if selected_conditions is None:
                selected_conditions = data[condition_col].unique().tolist()

            print(f"[DEBUG] Generating periodograms for:")
            print(f"  Variables: {selected_variables}")
            print(f"  Conditions: {selected_conditions}")

            # WORKAROUND: Generate periodograms one-by-one and rename files
            # CosinorPy adds "per_" prefix to test names, so we need to rename the files
            # after generation to get the correct names
            from .cosinor_analysis import DataType
            import os
            import glob

            total_combinations = 0
            successful_combinations = 0

            for var in selected_variables:
                for cond in selected_conditions:
                    df_cond = data[data[condition_col] == cond].copy()
                    if len(df_cond) > 0 and var in df_cond.columns:
                        total_combinations += 1

                        # Create DataFrame for this single combination
                        # Use var_cond as test name (CosinorPy will prepend "per_")
                        df_temp = pd.DataFrame({
                            'x': df_cond[time_col].values,
                            'y': df_cond[var].values,
                            'test': f"{var}_{cond}"
                        })

                        print(f"[DEBUG] Generating periodogram for {var}_{cond}")

                        # Load and run periodogram for this combination only
                        self._cosinor.load_data(df_temp, DataType.INDEPENDENT)
                        result_temp = self._cosinor.periodogram(save_folder=plot_folder)

                        if result_temp.get('success', False):
                            successful_combinations += 1

                            # Rename files to correct names
                            # CosinorPy generates: per_{var}_{cond}.png and per_{var}_{cond}.pdf
                            # We want: {var}_{cond}.png and {var}_{cond}.pdf
                            if plot_folder and os.path.exists(plot_folder):
                                old_pattern = f"per_{var}_{cond}"
                                new_pattern = f"{var}_{cond}"

                                for ext in ['.png', '.pdf']:
                                    old_file = os.path.join(plot_folder, f"{old_pattern}{ext}")
                                    new_file = os.path.join(plot_folder, f"{new_pattern}{ext}")

                                    if os.path.exists(old_file):
                                        if os.path.exists(new_file):
                                            os.remove(new_file)  # Remove old file if exists
                                        os.rename(old_file, new_file)
                                        print(f"[DEBUG] Renamed: {old_pattern}{ext} -> {new_pattern}{ext}")
                        else:
                            print(f"[DEBUG] WARNING: Failed to generate periodogram for {var}_{cond}")

            # Create final result based on success count
            result = {
                'success': successful_combinations > 0,
                'message': f"Generated {successful_combinations}/{total_combinations} periodogram(s)"
            }

            if result['success']:
                # Create summary message
                var_summary = f"{len(selected_variables)} variable(s)" if len(selected_variables) > 1 else selected_variables[0]
                cond_summary = f"{len(selected_conditions)} condition(s)" if len(selected_conditions) > 1 else selected_conditions[0]

                return AnalysisResult(
                    variable=var_summary,
                    condition=cond_summary,
                    method="cosinorpy_periodogram",  # Use lowercase identifier for detection
                    success=True,
                    message=f"Plots saved in: {plot_folder}\n{successful_combinations} periodogram(s) generated for {len(selected_variables)} variable(s) × {len(selected_conditions)} condition(s)"
                )
            else:
                return AnalysisResult(
                    variable="Periodogram", condition="All",
                    method="cosinorpy_periodogram",
                    success=False, message=result['message']
                )

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_periodogram_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_periodogram",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosinorpy_independent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> AnalysisResult:
        """Run cosinor analysis for independent data."""
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_independent",
                success=False, message="CosinorPy not available"
            )

        print(f"[DEBUG _run_cosinorpy_independent_new] variable={variable}, condition={condition}")

        try:
            # Get parameters
            # Period: use period_range (min, max) with step
            # If min == max, it's a single period. If min != max, it's a range.
            period_range = parameters.get('period_range', (24.0, 24.0))
            period_step = parameters.get('period_step', 1.0)

            # If period_range min != max, use the range as a list with step
            if period_range[0] != period_range[1]:
                import numpy as np
                # Use arange to support decimal steps
                # Add small epsilon to include the end point
                period = list(np.arange(period_range[0], period_range[1] + period_step/2, period_step))
                # Round to avoid floating point precision issues
                period = [round(p, 1) for p in period]
                print(f"[DEBUG] Using period range: {period} (step={period_step})")
            else:
                # Otherwise use single period value
                period = period_range[0]
                print(f"[DEBUG] Using single period: {period}")

            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            model_type_str = parameters.get('model_type', 'normal')
            from .cosinor_analysis import ModelType, AnalysisMethod, Criterium, DataType
            model_type = ModelType[model_type_str.upper().replace(' ', '_')] if model_type_str else ModelType.NORMAL

            criterium_str = parameters.get('criterium', 'RSS')
            # Replace hyphen with underscore for LOG-LIKELIHOOD
            criterium_str = criterium_str.replace('-', '_') if criterium_str else 'RSS'
            criterium = Criterium[criterium_str.upper()] if criterium_str else Criterium.RSS

            analysis_method_str = parameters.get('analysis_method', 'CI')
            analysis_method = AnalysisMethod[analysis_method_str.upper()] if analysis_method_str else AnalysisMethod.CI

            bootstrap_size = parameters.get('bootstrap_size', 1000)

            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            # Convert to CosinorPy format
            df_cosinorpy = self._convert_to_cosinorpy_format(
                data, variable, condition, time_col, condition_col
            )

            # Load data and set period
            self._cosinor.load_data(df_cosinorpy, DataType.INDEPENDENT)
            # Note: set_period() is for internal tracking, actual period passed to cosinor_independent()
            if isinstance(period, list):
                self._cosinor.set_period(period[0])  # Set first period for tracking
            else:
                self._cosinor.set_period(period)

            # Run analysis
            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)
            result = self._cosinor.cosinor_independent(
                variable=variable,
                condition=condition,
                period=period,
                n_components=n_components,
                model_type=model_type,
                criterium=criterium,
                analysis_method=analysis_method,
                bootstrap_size=bootstrap_size,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            # Get data for plotting
            times = df_cosinorpy['x'].values
            values = df_cosinorpy['y'].values

            # Check if result is a list (multiple periods) or dict (single period)
            if isinstance(result, list):
                # Multiple periods - create AnalysisResult for each
                print(f"[DEBUG] Creating {len(result)} AnalysisResult objects for multiple periods")
                analysis_results = []
                for res_dict in result:
                    acrophase_rad = res_dict.get('acrophase')
                    acrophase_hours = res_dict.get('acrophase_hours')

                    analysis_results.append(AnalysisResult(
                        variable=variable,
                        condition=condition,
                        method="cosinorpy_independent",
                        mesor=res_dict.get('mesor'),
                        amplitude=res_dict.get('amplitude'),
                        acrophase=acrophase_rad,
                        acrophase_hours=acrophase_hours,
                        period=res_dict.get('period'),
                        p_value=res_dict.get('p_value'),
                        q_value=res_dict.get('q_value'),
                        p_reject=res_dict.get('p_reject'),
                        q_reject=res_dict.get('q_reject'),
                        n_components=res_dict.get('n_components'),
                        rss=res_dict.get('rss'),
                        r_squared=res_dict.get('r_squared'),
                        r_squared_adj=res_dict.get('r_squared_adj'),
                        log_likelihood=res_dict.get('log_likelihood'),
                        aic=res_dict.get('aic'),
                        bic=res_dict.get('bic'),
                        me=res_dict.get('me'),
                        resid_se=res_dict.get('resid_se'),
                        amplitude_ci=res_dict.get('amplitude_ci'),
                        acrophase_ci=res_dict.get('acrophase_ci'),
                        mesor_ci=res_dict.get('mesor_ci'),
                        p_amplitude=res_dict.get('p_amplitude'),
                        p_acrophase=res_dict.get('p_acrophase'),
                        p_mesor=res_dict.get('p_mesor'),
                        q_amplitude=res_dict.get('q_amplitude'),
                        q_acrophase=res_dict.get('q_acrophase'),
                        q_mesor=res_dict.get('q_mesor'),
                        peak_times=res_dict.get('peak_times'),
                        trough_times=res_dict.get('trough_times'),
                        times=times,
                        values=values,
                        best_model=res_dict.get('best_model'),
                        success=True
                    ))
                return analysis_results  # Return list of AnalysisResult

            else:
                # Single period - return single AnalysisResult
                acrophase_rad = result.get('acrophase')
                acrophase_hours = result.get('acrophase_hours')

                return AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method="cosinorpy_independent",
                    mesor=result.get('mesor'),
                    amplitude=result.get('amplitude'),
                    acrophase=acrophase_rad,
                    acrophase_hours=acrophase_hours,
                    period=result.get('period', period if isinstance(period, (int, float)) else period[0]),
                    p_value=result.get('p_value'),
                    q_value=result.get('q_value'),
                    p_reject=result.get('p_reject'),
                    q_reject=result.get('q_reject'),
                    n_components=result.get('n_components'),
                    rss=result.get('rss'),
                    r_squared=result.get('r_squared'),
                    r_squared_adj=result.get('r_squared_adj'),
                    log_likelihood=result.get('log_likelihood'),
                    aic=result.get('aic'),
                    bic=result.get('bic'),
                    me=result.get('me'),
                    resid_se=result.get('resid_se'),
                    amplitude_ci=result.get('amplitude_ci'),
                    acrophase_ci=result.get('acrophase_ci'),
                    mesor_ci=result.get('mesor_ci'),
                    p_amplitude=result.get('p_amplitude'),
                    p_acrophase=result.get('p_acrophase'),
                    p_mesor=result.get('p_mesor'),
                    q_amplitude=result.get('q_amplitude'),
                    q_acrophase=result.get('q_acrophase'),
                    q_mesor=result.get('q_mesor'),
                    peak_times=result.get('peak_times'),
                    trough_times=result.get('trough_times'),
                    times=times,
                    values=values,
                    best_model=result.get('best_model'),
                    success=True
                )

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_independent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_independent",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosinorpy_dependent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """Run cosinor analysis for dependent/population data.

        Returns:
            - Single AnalysisResult if only one combination tested
            - List[AnalysisResult] if multiple periods/components tested
        """
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_dependent",
                success=False, message="CosinorPy not available"
            )

        print(f"[DEBUG _run_cosinorpy_dependent_new] variable={variable}, condition={condition}")

        try:
            # Get parameters
            # Period: use period_range (min, max) with step
            # If min == max, it's a single period. If min != max, it's a range.
            period_range = parameters.get('period_range', (24.0, 24.0))
            period_step = parameters.get('period_step', 1.0)

            # If period_range min != max, use the range as a list with step
            if period_range[0] != period_range[1]:
                import numpy as np
                # Use arange to support decimal steps
                # Add small epsilon to include the end point
                period = list(np.arange(period_range[0], period_range[1] + period_step/2, period_step))
                # Round to avoid floating point precision issues
                period = [round(p, 1) for p in period]
                print(f"[DEBUG] Using period range: {period} (step={period_step})")
            else:
                # Otherwise use single period value
                period = period_range[0]
                print(f"[DEBUG] Using single period: {period}")

            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            from .cosinor_analysis import ModelType, Criterium, DataType
            model_type_str = parameters.get('model_type', 'normal')
            model_type = ModelType[model_type_str.upper().replace(' ', '_')] if model_type_str else ModelType.NORMAL

            criterium_str = parameters.get('criterium', 'RSS')
            # Replace hyphen with underscore for LOG-LIKELIHOOD
            criterium_str = criterium_str.replace('-', '_') if criterium_str else 'RSS'
            criterium = Criterium[criterium_str.upper()] if criterium_str else Criterium.RSS

            # Map GUI analysis_method to CosinorPy params_CI_analysis parameter
            # For dependent data, CosinorPy accepts: 'sampling' or 'bootstrap'
            analysis_method_str = parameters.get('analysis_method', 'Sampling')
            params_ci_analysis_map = {
                'Sampling': 'sampling',
                'Bootstrap': 'bootstrap'
            }
            params_ci_analysis = params_ci_analysis_map.get(analysis_method_str, 'sampling')

            bootstrap_size = parameters.get('bootstrap_size', 1000)

            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            # Get subject column
            subject_col = 'subject' if 'subject' in data.columns else None

            # Convert to CosinorPy format
            df_cosinorpy = self._convert_to_cosinorpy_format(
                data, variable, condition, time_col, condition_col, subject_col
            )

            # Load data
            self._cosinor.load_data(df_cosinorpy, DataType.DEPENDENT)
            # Note: set_period() is for internal tracking, actual period passed to cosinor_dependent()
            if isinstance(period, list):
                self._cosinor.set_period(period[0])  # Set first period for tracking
            else:
                self._cosinor.set_period(period)

            # Run analysis
            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)
            result = self._cosinor.cosinor_dependent(
                variable=variable,
                condition=condition,
                period=period,
                n_components=n_components,
                model_type=model_type,
                criterium=criterium,
                params_ci_analysis=params_ci_analysis,
                bootstrap_size=bootstrap_size,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            # Get data for plotting
            times = df_cosinorpy['x'].values
            values = df_cosinorpy['y'].values

            # Check if result is a list (multiple periods/components) or dict (single result)
            if isinstance(result, list):
                # Multiple periods/components - create AnalysisResult for each
                print(f"[DEBUG] Creating {len(result)} AnalysisResult objects for multiple periods/components")
                analysis_results = []
                for res_dict in result:
                    analysis_results.append(AnalysisResult(
                        variable=variable,
                        condition=condition,
                        method="cosinorpy_dependent",
                        mesor=res_dict.get('mesor'),
                        amplitude=res_dict.get('amplitude'),
                        acrophase=res_dict.get('acrophase'),
                        acrophase_hours=res_dict.get('acrophase_hours'),
                        period=res_dict.get('period'),
                        p_value=res_dict.get('p_value'),
                        q_value=res_dict.get('q_value'),
                        p_reject=res_dict.get('p_reject'),
                        q_reject=res_dict.get('q_reject'),
                        n_components=res_dict.get('n_components'),
                        rss=res_dict.get('rss'),
                        aic=res_dict.get('aic'),
                        bic=res_dict.get('bic'),
                        me=res_dict.get('me'),
                        resid_se=res_dict.get('resid_se'),
                        amplitude_ci=res_dict.get('amplitude_ci'),
                        acrophase_ci=res_dict.get('acrophase_ci'),
                        mesor_ci=res_dict.get('mesor_ci'),
                        p_amplitude=res_dict.get('p_amplitude'),
                        p_acrophase=res_dict.get('p_acrophase'),
                        p_mesor=res_dict.get('p_mesor'),
                        q_amplitude=res_dict.get('q_amplitude'),
                        q_acrophase=res_dict.get('q_acrophase'),
                        q_mesor=res_dict.get('q_mesor'),
                        times=times,
                        values=values,
                        best_model=res_dict.get('best_model'),
                        success=True
                    ))
                return analysis_results  # Return list of AnalysisResult
            else:
                # Single result - return single AnalysisResult
                return AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method="cosinorpy_dependent",
                    mesor=result.get('mesor'),
                    amplitude=result.get('amplitude'),
                    acrophase=result.get('acrophase'),
                    acrophase_hours=result.get('acrophase_hours'),
                    period=result.get('period', period if isinstance(period, (int, float)) else period[0]),
                    p_value=result.get('p_value'),
                    q_value=result.get('q_value'),
                    p_reject=result.get('p_reject'),
                    q_reject=result.get('q_reject'),
                    n_components=result.get('n_components'),
                    rss=result.get('rss'),
                    aic=result.get('aic'),
                    bic=result.get('bic'),
                    me=result.get('me'),
                    resid_se=result.get('resid_se'),
                    amplitude_ci=result.get('amplitude_ci'),
                    acrophase_ci=result.get('acrophase_ci'),
                    mesor_ci=result.get('mesor_ci'),
                    p_amplitude=result.get('p_amplitude'),
                    p_acrophase=result.get('p_acrophase'),
                    p_mesor=result.get('p_mesor'),
                    q_amplitude=result.get('q_amplitude'),
                    q_acrophase=result.get('q_acrophase'),
                    q_mesor=result.get('q_mesor'),
                    times=times,
                    values=values,
                    best_model=result.get('best_model'),
                    success=True
                )

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_dependent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_dependent",
                success=False, message=f"Error: {str(e)}"
            )

    def _parse_cosinorpy_comparison_results(
        self,
        result: Dict[str, Any],
        variable: str,
        comparison_type: str,
        comparison_method: str
    ) -> List[ComparisonResult]:
        """
        Parse CosinorPy comparison DataFrame results into ComparisonResult objects.

        Args:
            result: Dictionary from compare_independent() containing 'results_df'
            variable: Variable name being analyzed
            comparison_type: 'pooled_model', 'independent_models', or 'multi_*'
            comparison_method: 'Independent' or 'LimoRhyde' (for multi-component)

        Returns:
            List of ComparisonResult objects
        """
        df_results = result.get('results_df')
        if df_results is None or df_results.empty:
            return []

        comparison_results = []

        # Iterate over each row in the results DataFrame
        for idx, row in df_results.iterrows():
            # Extract pair names from 'test' column
            # cosinor (multi-component) uses "cond1 vs. cond2" (with dot)
            # cosinor1 (single-component) uses "cond1 vs cond2" (without dot)
            test_str = row.get('test', '')
            if ' vs. ' in test_str:
                cond1, cond2 = test_str.split(' vs. ', 1)
            elif ' vs ' in test_str:
                cond1, cond2 = test_str.split(' vs ', 1)
            else:
                cond1, cond2 = 'condition1', 'condition2'

            # Strip variable prefix from condition names if present
            # Data conversion creates test names as "{variable}-{condition}_rep{i}",
            # so after CosinorPy splits on '_rep', the condition names may have
            # the format "{variable}-{condition}" (e.g., "circadian_noisy-control")
            var_prefix = f"{variable}-"
            if cond1.strip().startswith(var_prefix):
                cond1 = cond1.strip()[len(var_prefix):]
            if cond2.strip().startswith(var_prefix):
                cond2 = cond2.strip()[len(var_prefix):]

            # Extract period (handle both single-component and multi-component).
            # test_cosinor_pairs_independent returns 'period1'/'period2' (not 'period'),
            # so fall back to 'period1' when 'period' is absent.
            period = row.get('period') or row.get('period1', 24.0)
            if isinstance(period, (list, np.ndarray)):
                period = period[0] if len(period) > 0 else 24.0

            # Build method string from the result dict's comparison_type (snake_case),
            # which is more reliable than the function argument (which may be a UI string).
            actual_ct = result.get('comparison_type', comparison_type)
            if actual_ct == 'pooled_model':
                method_str = 'cosinorpy_compare_pooled'
            elif actual_ct == 'independent_models':
                method_str = 'cosinorpy_compare_independent_models'
            elif actual_ct in ('multi_independent', 'multi_limorhyde', 'multi_limo'):
                if 'limo' in actual_ct:
                    method_str = 'cosinorpy_compare_limorhyde'
                else:
                    method_str = 'cosinorpy_compare_multi'
            elif actual_ct == 'dependent_single':
                method_str = 'cosinorpy_compare_dependent'
            elif actual_ct == 'dependent_multi':
                method_str = 'cosinorpy_compare_dependent_multi'
            else:
                method_str = 'cosinorpy_compare'

            # Extract parameters for condition 1 (g0 or 1)
            # CosinorPy uses different naming: amplitude_g0/g1 OR amplitude1/amplitude2
            amplitude_g0 = row.get('amplitude_g0') or row.get('amplitude1')
            acrophase_g0 = row.get('acrophase_g0') or row.get('acrophase1')
            mesor_g0 = row.get('mesor_g0') or row.get('mesor1')

            # Extract parameters for condition 2 (g1 or 2)
            amplitude_g1 = row.get('amplitude_g1') or row.get('amplitude2')
            acrophase_g1 = row.get('acrophase_g1') or row.get('acrophase2')
            mesor_g1 = row.get('mesor_g1') or row.get('mesor2')

            # Extract differences
            amplitude_diff = row.get('d_amplitude')
            acrophase_diff = row.get('d_acrophase')
            mesor_diff = row.get('d_mesor')

            # Extract p-values and q-values
            # Different methods have different column names:
            # - Independent: 'p(d_amplitude)', 'q(d_amplitude)', 'CI(d_amplitude)'
            # - LimoRhyde: 'p params', 'q params', 'p(F test)' (no p/q for individual parameters)
            # - Dependent (multi): 'p(d_amplitude)', 'q(d_amplitude)', 'p1', 'p2', 'q1', 'q2'

            # Check if this is LimoRhyde format
            is_limo = 'p params' in df_results.columns
            # Check if this is dependent multi-component format (has p1, p2)
            is_dependent_multi = 'p1' in df_results.columns

            if is_limo:
                # LimoRhyde with empty analysis ('None'): only 'p params'/'q params' are
                # available (joint test across all parameters).
                # LimoRhyde with non-empty analysis (CI1/CI2/Bootstrap1/Bootstrap2):
                # compare_pairs_limo also populates 'p(d_amplitude)', 'CI(d_amplitude)', etc.
                # In that case, use the per-parameter values instead of the joint test.
                has_per_param_limo = 'p(d_amplitude)' in df_results.columns

                if has_per_param_limo:
                    # Per-parameter p-values and CIs are available
                    p_amplitude = row.get('p(d_amplitude)')
                    p_acrophase = row.get('p(d_acrophase)')
                    p_mesor = row.get('p(d_mesor)')
                    # q values remain NaN in limo (only q params / q(F test) are FDR-corrected)
                    q_amplitude = row.get('q(d_amplitude)')  # will be NaN → N/A
                    q_acrophase = row.get('q(d_acrophase)')
                    q_mesor = row.get('q(d_mesor)')
                    amplitude_diff_ci = self._parse_ci(row.get('CI(d_amplitude)'))
                    acrophase_diff_ci = self._parse_ci(row.get('CI(d_acrophase)'))
                    mesor_diff_ci = self._parse_ci(row.get('CI(d_mesor)'))
                else:
                    # Empty analysis: only joint p/q params available
                    p_amplitude = row.get('p params')
                    p_acrophase = row.get('p params')
                    p_mesor = row.get('p params')
                    q_amplitude = row.get('q params')
                    q_acrophase = row.get('q params')
                    q_mesor = row.get('q params')
                    amplitude_diff_ci = None
                    acrophase_diff_ci = None
                    mesor_diff_ci = None
            else:
                # Independent or Dependent: has specific p/q/CI for each parameter
                p_amplitude = row.get('p(d_amplitude)')
                p_acrophase = row.get('p(d_acrophase)')
                p_mesor = row.get('p(d_mesor)')

                q_amplitude = row.get('q(d_amplitude)')
                q_acrophase = row.get('q(d_acrophase)')
                q_mesor = row.get('q(d_mesor)')

                # Extract confidence intervals (parse from string if needed).
                # cosinor1.test_cosinor_pairs (pooled model) has a typo in the library:
                # the column is 'CI(d_amplitde)' instead of 'CI(d_amplitude)'.
                # We try the correct spelling first, then the typo as fallback.
                amplitude_diff_ci = self._parse_ci(
                    row.get('CI(d_amplitude)') or row.get('CI(d_amplitde)')
                )
                acrophase_diff_ci = self._parse_ci(row.get('CI(d_acrophase)'))
                mesor_diff_ci = self._parse_ci(row.get('CI(d_mesor)'))

            # Extract population-specific p/q values (for dependent multi-component)
            p1 = row.get('p1') if is_dependent_multi else None
            p2 = row.get('p2') if is_dependent_multi else None
            q1 = row.get('q1') if is_dependent_multi else None
            q2 = row.get('q2') if is_dependent_multi else None

            # Extract n_components (for multi-component comparisons)
            n_comp_value = row.get('n_components', row.get('n_components1'))
            if n_comp_value is not None:
                try:
                    n_comp_value = int(n_comp_value)
                except (ValueError, TypeError):
                    n_comp_value = None

            # Create ComparisonResult object
            comp_result = ComparisonResult(
                variable=variable,
                condition1=cond1.strip(),
                condition2=cond2.strip(),
                method=method_str,
                period=float(period),
                n_components=n_comp_value,
                # Group 0 parameters
                amplitude_g0=amplitude_g0,
                acrophase_g0=acrophase_g0,
                mesor_g0=mesor_g0,
                # Group 1 parameters
                amplitude_g1=amplitude_g1,
                acrophase_g1=acrophase_g1,
                mesor_g1=mesor_g1,
                # Differences
                amplitude_diff=amplitude_diff,
                acrophase_diff=acrophase_diff,
                mesor_diff=mesor_diff,
                # P-values
                p_amplitude=p_amplitude,
                p_acrophase=p_acrophase,
                p_mesor=p_mesor,
                # Q-values
                q_amplitude=q_amplitude,
                q_acrophase=q_acrophase,
                q_mesor=q_mesor,
                # Population-specific p/q values (for dependent multi-component)
                p1=p1,
                p2=p2,
                q1=q1,
                q2=q2,
                # Confidence intervals
                amplitude_diff_ci=amplitude_diff_ci,
                acrophase_diff_ci=acrophase_diff_ci,
                mesor_diff_ci=mesor_diff_ci,
                success=True,
                message=""
            )

            comparison_results.append(comp_result)

        return comparison_results

    def _parse_ci(self, ci_value: Any) -> Optional[Tuple[float, float]]:
        """
        Parse confidence interval from various formats.

        Args:
            ci_value: Can be a list, array, tuple, or string representation

        Returns:
            Tuple of (lower, upper) or None if cannot parse
        """
        # Check if None first (before any other checks)
        if ci_value is None:
            return None

        # If it's already a list, tuple, or array
        if isinstance(ci_value, (list, tuple, np.ndarray)):
            try:
                if len(ci_value) == 2:
                    return (float(ci_value[0]), float(ci_value[1]))
            except (TypeError, ValueError, IndexError):
                return None

        # Check if it's a scalar NaN (for string or scalar types)
        # Use try-except to avoid "truth value of array" error
        try:
            if pd.isna(ci_value):
                return None
        except (ValueError, TypeError):
            # If pd.isna() fails (e.g., on arrays), continue
            pass

        # If it's a string representation like "[0.5, 1.5]"
        if isinstance(ci_value, str):
            try:
                # Remove brackets and split
                ci_str = ci_value.strip('[]()').replace(' ', '')
                parts = ci_str.split(',')
                if len(parts) == 2:
                    return (float(parts[0]), float(parts[1]))
            except (ValueError, AttributeError):
                pass

        return None

    def _run_cosinorpy_compare_independent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> List[ComparisonResult]:
        """Run comparison analysis for independent data (all pairs)."""
        if self._cosinor is None:
            return []

        print(f"[DEBUG _run_cosinorpy_compare_independent_new] variable={variable}")

        try:
            # Get all conditions
            conditions = data[condition_col].unique().tolist()
            print(f"[DEBUG] Found conditions: {conditions}")

            if len(conditions) < 2:
                print(f"[DEBUG] Not enough conditions for comparison")
                return []

            # Get parameters
            period = parameters.get('period', 24.0)
            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            # New comparison parameters
            comparison_type = parameters.get('comparison_type', 'Pooled Model')
            comparison_method = parameters.get('comparison_method', 'Independent')
            analysis_method = parameters.get('analysis_method', 'CI')
            parameters_to_compare = parameters.get('parameters_to_compare', ['amplitude', 'acrophase', 'mesor'])
            include_lin_comp = parameters.get('include_lin_comp', False)
            bootstrap_size = parameters.get('bootstrap_size', 1000)
            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)

            # Per-condition periods: only valid for "Independent Models" with exactly 2 conditions.
            # The UI always sends period1/period2 as spinbox floats (never None), so we must
            # explicitly set them to None when the per-condition period widgets are hidden,
            # otherwise _compare_independent_single_independent always uses them and ignores
            # the shared period (the else/shared-period branch would never be reached).
            if comparison_type == 'Independent Models' and len(conditions) == 2:
                period1 = parameters.get('period1', None)
                period2 = parameters.get('period2', None)
            else:
                period1 = None
                period2 = None

            from .cosinor_analysis import DataType

            plot_folder = get_cosinorpy_plot_folder(data_file_path) if save_cosinorpy_plots else None

            # Convert all conditions to CosinorPy format for comparison
            # Use the specialized function that formats with hyphen (not underscore)
            df_cosinorpy = self._convert_to_cosinorpy_format_all_conditions(
                data, variable, time_col, condition_col
            )

            # Load data
            self._cosinor.load_data(df_cosinorpy, DataType.INDEPENDENT)
            self._cosinor.set_period(period)

            # Run comparison (auto-generates all pairs)
            result = self._cosinor.compare_independent(
                variable=variable,
                conditions=conditions,
                period=period,
                period1=period1,
                period2=period2,
                n_components=n_components,
                comparison_type=comparison_type,
                comparison_method=comparison_method,
                analysis_method=analysis_method,
                parameters_to_compare=parameters_to_compare,
                lin_comp=include_lin_comp,
                bootstrap_size=bootstrap_size,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            # Convert results to ComparisonResult objects
            print(f"[DEBUG] Comparison returned: {result.keys()}")

            # Parse results_df and create ComparisonResult objects
            comparison_results = self._parse_cosinorpy_comparison_results(
                result, variable, comparison_type, comparison_method
            )

            # Enrich results with raw data and individual fit parameters (including mesor)
            self._enrich_independent_comparison_results(
                comparison_results, data, df_cosinorpy,
                variable, time_col, condition_col, period
            )

            return comparison_results

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_compare_independent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _run_cosinorpy_compare_dependent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> List[ComparisonResult]:
        """Run comparison analysis for dependent/population data (all pairs)."""
        if self._cosinor is None:
            return []

        print(f"[DEBUG _run_cosinorpy_compare_dependent_new] variable={variable}")

        try:
            # Get all conditions
            conditions = data[condition_col].unique().tolist()
            print(f"[DEBUG] Found conditions: {conditions}")

            if len(conditions) < 2:
                print(f"[DEBUG] Not enough conditions for comparison")
                return []

            # Check for subject column (required for dependent data)
            subject_col = 'subject' if 'subject' in data.columns else None
            if subject_col is None:
                print(f"[ERROR] Dependent data requires a 'subject' column")
                return []

            # Get parameters
            period = parameters.get('period', 24.0)
            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            # Comparison parameters for dependent data
            analysis_method = parameters.get('analysis_method', 'CI')
            parameters_to_compare = parameters.get('parameters_to_compare', ['amplitude', 'acrophase', 'mesor'])
            include_lin_comp = parameters.get('include_lin_comp', False)
            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)

            # Map GUI analysis method to CosinorPy parameter
            # 'CI' -> 'CI', 'Permutation' -> 'permutation'
            analysis_param = analysis_method if analysis_method in ['CI', 'permutation'] else 'CI'
            if analysis_method == 'Permutation':
                analysis_param = 'permutation'

            print(f"[DEBUG] analysis_method (GUI)={analysis_method}, analysis_param (CosinorPy)={analysis_param}")

            from .cosinor_analysis import DataType

            plot_folder = get_cosinorpy_plot_folder(data_file_path) if save_cosinorpy_plots else None
            print(f"[DEBUG] save_cosinorpy_plots={save_cosinorpy_plots}, data_file_path={data_file_path}, plot_folder={plot_folder}")

            # Convert all conditions to CosinorPy format for comparison (dependent/population format)
            # Use the specialized function that formats with hyphen (not underscore)
            df_cosinorpy = self._convert_to_cosinorpy_format_population_all_conditions(
                data, variable, time_col, condition_col, subject_col
            )

            print(f"[DEBUG] Converted DataFrame shape: {df_cosinorpy.shape}")
            print(f"[DEBUG] Unique test values: {df_cosinorpy['test'].unique()}")

            # Load data as DEPENDENT type
            self._cosinor.load_data(df_cosinorpy, DataType.DEPENDENT)
            self._cosinor.set_period(period)

            # Run comparison (auto-generates all pairs)
            result = self._cosinor.compare_dependent(
                variable=variable,
                conditions=conditions,
                period=period,
                n_components=n_components,
                analysis_method=analysis_param,
                parameters_to_analyse=parameters_to_compare,
                lin_comp=include_lin_comp,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            # Convert results to ComparisonResult objects
            print(f"[DEBUG] Comparison returned: {result.keys()}")

            # Parse results_df and create ComparisonResult objects
            comparison_results = self._parse_cosinorpy_comparison_results(
                result, variable, 'dependent', analysis_method
            )

            # Enrich results with raw data and individual population fit parameters
            # (CosinorPy single-component results only contain differences, not per-group values)
            self._enrich_dependent_comparison_results(
                comparison_results, data, df_cosinorpy,
                variable, time_col, condition_col, period
            )

            return comparison_results

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_compare_dependent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _enrich_dependent_comparison_results(
        self,
        comparison_results: List[ComparisonResult],
        data: pd.DataFrame,
        df_cosinorpy: pd.DataFrame,
        variable: str,
        time_col: str,
        condition_col: str,
        period: float
    ) -> None:
        """Enrich dependent comparison results with raw data and individual fit parameters.

        CosinorPy single-component population comparison only returns differences
        (d_amplitude, d_acrophase) but not individual group parameters. The visualization
        needs raw data and per-group amplitude/acrophase/mesor to render fit curves.

        For nonlinear compare dependent results also populates amplification_g0/g1
        and lin_comp_g0/g1 via a 1-component nonlinear population fit per condition.
        """
        try:
            from CosinorPy import cosinor1 as cosinor1_mod
        except ImportError:
            print("[DEBUG] CosinorPy not available for enrichment")
            return

        for comp_result in comparison_results:
            cond1_name = comp_result.condition1
            cond2_name = comp_result.condition2

            # Extract raw data from original DataFrame
            data_g0 = data[data[condition_col] == cond1_name]
            data_g1 = data[data[condition_col] == cond2_name]
            comp_result.times_g0 = data_g0[time_col].values.astype(float) if len(data_g0) > 0 else None
            comp_result.values_g0 = data_g0[variable].values.astype(float) if len(data_g0) > 0 else None
            comp_result.times_g1 = data_g1[time_col].values.astype(float) if len(data_g1) > 0 else None
            comp_result.values_g1 = data_g1[variable].values.astype(float) if len(data_g1) > 0 else None

            # Run individual population fits to get per-condition parameters
            period_val = comp_result.period
            cond1_prefix = f"{variable}-{cond1_name}"
            cond2_prefix = f"{variable}-{cond2_name}"

            df_pop1 = df_cosinorpy[df_cosinorpy.test.str.startswith(f'{cond1_prefix}_rep')]
            df_pop2 = df_cosinorpy[df_cosinorpy.test.str.startswith(f'{cond2_prefix}_rep')]

            # Fit condition 1 (linear, for mesor/amplitude/acrophase)
            try:
                res1 = cosinor1_mod.population_fit_cosinor(df_pop1, period=period_val, plot_on=False)
                comp_result.mesor_g0 = float(res1['means'][0])
                comp_result.amplitude_g0 = float(res1['means'][3])
                comp_result.acrophase_g0 = float(res1['means'][4])
                # Extract individual rhythmicity p-value (p1) for condition 1
                p1_val = res1.get('p_value')
                if p1_val is not None and not (isinstance(p1_val, float) and np.isnan(p1_val)):
                    comp_result.p1 = float(p1_val)
            except Exception as e:
                print(f"[DEBUG] Could not fit population for {cond1_name}: {e}")

            # Fit condition 2 (linear, for mesor/amplitude/acrophase)
            try:
                res2 = cosinor1_mod.population_fit_cosinor(df_pop2, period=period_val, plot_on=False)
                comp_result.mesor_g1 = float(res2['means'][0])
                comp_result.amplitude_g1 = float(res2['means'][3])
                comp_result.acrophase_g1 = float(res2['means'][4])
                # Extract individual rhythmicity p-value (p2) for condition 2
                p2_val = res2.get('p_value')
                if p2_val is not None and not (isinstance(p2_val, float) and np.isnan(p2_val)):
                    comp_result.p2 = float(p2_val)
            except Exception as e:
                print(f"[DEBUG] Could not fit population for {cond2_name}: {e}")

            # For nonlinear compare: additionally fit generalized cosinor to get
            # amplification_g0/g1 and lin_comp_g0/g1 per condition.
            if comp_result.method == 'cosinorpy_nonlinear_compare_dependent':
                try:
                    from CosinorPy import cosinor_nonlin as cosinor_nonlin_mod
                    sp1 = cosinor_nonlin_mod.population_fit_generalized_cosinor(
                        df_pop1, period=period_val, plot=False
                    )
                    comp_result.amplification_g0 = float(sp1['params']['C'])
                    comp_result.lin_comp_g0 = float(sp1['params']['D'])
                except Exception as e:
                    print(f"[DEBUG] Could not fit nonlinear population for {cond1_name}: {e}")

                try:
                    from CosinorPy import cosinor_nonlin as cosinor_nonlin_mod
                    sp2 = cosinor_nonlin_mod.population_fit_generalized_cosinor(
                        df_pop2, period=period_val, plot=False
                    )
                    comp_result.amplification_g1 = float(sp2['params']['C'])
                    comp_result.lin_comp_g1 = float(sp2['params']['D'])
                except Exception as e:
                    print(f"[DEBUG] Could not fit nonlinear population for {cond2_name}: {e}")

    def _enrich_independent_comparison_results(
        self,
        comparison_results: List[ComparisonResult],
        data: pd.DataFrame,
        df_cosinorpy: pd.DataFrame,
        variable: str,
        time_col: str,
        condition_col: str,
        period: float
    ) -> None:
        """Enrich independent comparison results with raw data and individual fit parameters.

        CosinorPy independent comparisons may not return individual group parameters
        (mesor, amplitude, acrophase per condition). The visualization needs these
        to render fit curves and scatter plots.
        """
        try:
            from CosinorPy import cosinor1 as cosinor1_mod
        except ImportError:
            print("[DEBUG] CosinorPy not available for enrichment")
            return

        for comp_result in comparison_results:
            cond1_name = comp_result.condition1
            cond2_name = comp_result.condition2

            # Extract raw data from original DataFrame
            data_g0 = data[data[condition_col] == cond1_name]
            data_g1 = data[data[condition_col] == cond2_name]
            comp_result.times_g0 = data_g0[time_col].values.astype(float) if len(data_g0) > 0 else None
            comp_result.values_g0 = data_g0[variable].values.astype(float) if len(data_g0) > 0 else None
            comp_result.times_g1 = data_g1[time_col].values.astype(float) if len(data_g1) > 0 else None
            comp_result.values_g1 = data_g1[variable].values.astype(float) if len(data_g1) > 0 else None

            # Run individual fits to get per-condition parameters if missing
            period_val = comp_result.period
            test_name_g0 = f"{variable}-{cond1_name}"
            test_name_g1 = f"{variable}-{cond2_name}"

            # Fit condition 1 (if mesor is missing)
            if comp_result.mesor_g0 is None:
                df_g0 = df_cosinorpy[df_cosinorpy.test == test_name_g0]
                if len(df_g0) > 0:
                    try:
                        fit_res, amp, acr, _ = cosinor1_mod.fit_cosinor(
                            df_g0.x.values, df_g0.y.values, period=period_val, plot_on=False
                        )
                        comp_result.mesor_g0 = float(fit_res.params.iloc[0])
                        if comp_result.amplitude_g0 is None:
                            comp_result.amplitude_g0 = float(amp)
                        if comp_result.acrophase_g0 is None:
                            comp_result.acrophase_g0 = float(acr)
                    except Exception as e:
                        print(f"[DEBUG] Could not fit cosinor for {cond1_name}: {e}")

            # Fit condition 2 (if mesor is missing)
            if comp_result.mesor_g1 is None:
                df_g1 = df_cosinorpy[df_cosinorpy.test == test_name_g1]
                if len(df_g1) > 0:
                    try:
                        fit_res, amp, acr, _ = cosinor1_mod.fit_cosinor(
                            df_g1.x.values, df_g1.y.values, period=period_val, plot_on=False
                        )
                        comp_result.mesor_g1 = float(fit_res.params.iloc[0])
                        if comp_result.amplitude_g1 is None:
                            comp_result.amplitude_g1 = float(amp)
                        if comp_result.acrophase_g1 is None:
                            comp_result.acrophase_g1 = float(acr)
                    except Exception as e:
                        print(f"[DEBUG] Could not fit cosinor for {cond2_name}: {e}")

    def _run_cosinorpy_nonlinear_independent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """
        Run nonlinear cosinor analysis for independent data.

        Model: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t
        Where C = amplification (damped/forced), D = lin_comp (linear trend)
        """
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear_independent",
                success=False, message="CosinorPy not available"
            )

        print(f"[DEBUG _run_cosinorpy_nonlinear_independent_new] variable={variable}, condition={condition}")

        try:
            # Get parameters
            # Period: use period_range (min, max) with step
            period_range = parameters.get('period_range', (24.0, 24.0))
            period_step = parameters.get('period_step', 1.0)

            # If period_range min != max, use the range as a list with step
            if period_range[0] != period_range[1]:
                import numpy as np
                period = list(np.arange(period_range[0], period_range[1] + period_step/2, period_step))
                period = [round(p, 1) for p in period]
                print(f"[DEBUG] Using period range: {period} (step={period_step})")
            else:
                period = period_range[0]
                print(f"[DEBUG] Using single period: {period}")

            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            bootstrap_size = parameters.get('bootstrap_size', 100)
            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)

            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            # Convert to CosinorPy format
            df_cosinorpy = self._convert_to_cosinorpy_format(
                data, variable, condition, time_col, condition_col
            )

            # Load data
            from .cosinor_analysis import DataType
            self._cosinor.load_data(df_cosinorpy, DataType.INDEPENDENT)

            # Run analysis
            result = self._cosinor.nonlinear_independent(
                variable=variable,
                condition=condition,
                period=period,
                n_components=n_components,
                bootstrap_size=bootstrap_size,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            # Get data for plotting
            times = df_cosinorpy['x'].values
            values = df_cosinorpy['y'].values

            # Helper to convert acrophase to hours
            def acrophase_to_hours(acr_rad, per):
                if acr_rad is None or per is None:
                    return None
                # Acrophase in radians to hours: hours = -acrophase * period / (2 * pi)
                import math
                hours = -acr_rad * per / (2 * math.pi)
                # Normalize to [0, period)
                hours = hours % per
                return hours

            # Check if result is a list (multiple periods) or dict (single)
            if isinstance(result, list):
                print(f"[DEBUG] Creating {len(result)} AnalysisResult objects for multiple periods")
                analysis_results = []
                for res_dict in result:
                    acrophase_rad = res_dict.get('acrophase')
                    period_val = res_dict.get('period')
                    acrophase_hours = acrophase_to_hours(acrophase_rad, period_val)

                    analysis_results.append(AnalysisResult(
                        variable=variable,
                        condition=condition,
                        method="cosinorpy_nonlinear_independent",
                        mesor=res_dict.get('mesor'),
                        amplitude=res_dict.get('amplitude'),
                        acrophase=acrophase_rad,
                        acrophase_hours=acrophase_hours,
                        period=period_val,
                        p_value=res_dict.get('p_value'),
                        q_value=res_dict.get('q_value'),
                        n_components=res_dict.get('n_components'),
                        amplification=res_dict.get('amplification'),
                        lin_comp=res_dict.get('lin_comp'),
                        p_amplitude=res_dict.get('p_amplitude'),
                        p_acrophase=res_dict.get('p_acrophase'),
                        p_amplification=res_dict.get('p_amplification'),
                        p_lin_comp=res_dict.get('p_lin_comp'),
                        q_amplitude=res_dict.get('q_amplitude'),
                        q_acrophase=res_dict.get('q_acrophase'),
                        q_amplification=res_dict.get('q_amplification'),
                        q_lin_comp=res_dict.get('q_lin_comp'),
                        amplitude_ci=res_dict.get('amplitude_ci'),
                        acrophase_ci=res_dict.get('acrophase_ci'),
                        amplification_ci=res_dict.get('amplification_ci'),
                        lin_comp_ci=res_dict.get('lin_comp_ci'),
                        times=times,
                        values=values,
                        best_model=res_dict.get('best_model'),
                        success=True
                    ))
                return analysis_results

            else:
                # Single period result
                acrophase_rad = result.get('acrophase')
                period_val = result.get('period', period if isinstance(period, (int, float)) else period[0])
                acrophase_hours = acrophase_to_hours(acrophase_rad, period_val)

                return AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method="cosinorpy_nonlinear_independent",
                    mesor=result.get('mesor'),
                    amplitude=result.get('amplitude'),
                    acrophase=acrophase_rad,
                    acrophase_hours=acrophase_hours,
                    period=period_val,
                    p_value=result.get('p_value'),
                    q_value=result.get('q_value'),
                    n_components=result.get('n_components'),
                    amplification=result.get('amplification'),
                    lin_comp=result.get('lin_comp'),
                    p_amplitude=result.get('p_amplitude'),
                    p_acrophase=result.get('p_acrophase'),
                    p_amplification=result.get('p_amplification'),
                    p_lin_comp=result.get('p_lin_comp'),
                    q_amplitude=result.get('q_amplitude'),
                    q_acrophase=result.get('q_acrophase'),
                    q_amplification=result.get('q_amplification'),
                    q_lin_comp=result.get('q_lin_comp'),
                    amplitude_ci=result.get('amplitude_ci'),
                    acrophase_ci=result.get('acrophase_ci'),
                    amplification_ci=result.get('amplification_ci'),
                    lin_comp_ci=result.get('lin_comp_ci'),
                    times=times,
                    values=values,
                    best_model=result.get('best_model'),
                    success=True
                )

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_nonlinear_independent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear_independent",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosinorpy_nonlinear_dependent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> Union[AnalysisResult, List[AnalysisResult]]:
        """
        Run nonlinear cosinor analysis for dependent/population data.

        Model: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t
        Data format: Same subjects measured repeatedly (replicates).
        No bootstrap - stats calculated from variance between replicates.
        """
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear_dependent",
                success=False, message="CosinorPy not available"
            )

        print(f"[DEBUG _run_cosinorpy_nonlinear_dependent_new] variable={variable}, condition={condition}")

        try:
            # Get parameters
            period_range = parameters.get('period_range', (24.0, 24.0))
            period_step = parameters.get('period_step', 1.0)

            if period_range[0] != period_range[1]:
                import numpy as np
                period = list(np.arange(period_range[0], period_range[1] + period_step/2, period_step))
                period = [round(p, 1) for p in period]
                print(f"[DEBUG] Using period range: {period} (step={period_step})")
            else:
                period = period_range[0]
                print(f"[DEBUG] Using single period: {period}")

            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)
            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            # Convert to CosinorPy format for DEPENDENT data
            df_cosinorpy = self._convert_to_cosinorpy_format_population(
                data, variable, condition, time_col, condition_col
            )

            # Load data as DEPENDENT
            from .cosinor_analysis import DataType
            self._cosinor.load_data(df_cosinorpy, DataType.DEPENDENT)

            # Run analysis
            result = self._cosinor.nonlinear_dependent(
                variable=variable,
                condition=condition,
                period=period,
                n_components=n_components,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            # Get data for plotting
            times = df_cosinorpy['x'].values
            values = df_cosinorpy['y'].values

            # Helper to convert acrophase to hours
            def acrophase_to_hours(acr_rad, per):
                if acr_rad is None or per is None:
                    return None
                import math
                hours = -acr_rad * per / (2 * math.pi)
                hours = hours % per
                return hours

            # Check if result is a list (multiple periods) or dict (single)
            if isinstance(result, list):
                print(f"[DEBUG] Creating {len(result)} AnalysisResult objects for multiple periods")
                analysis_results = []
                for res_dict in result:
                    acrophase_rad = res_dict.get('acrophase')
                    period_val = res_dict.get('period')
                    acrophase_hours = acrophase_to_hours(acrophase_rad, period_val)

                    analysis_results.append(AnalysisResult(
                        variable=variable,
                        condition=condition,
                        method="cosinorpy_nonlinear_dependent",
                        mesor=res_dict.get('mesor'),
                        amplitude=res_dict.get('amplitude'),
                        acrophase=acrophase_rad,
                        acrophase_hours=acrophase_hours,
                        period=period_val,
                        p_value=res_dict.get('p_value'),
                        q_value=res_dict.get('q_value'),
                        n_components=res_dict.get('n_components'),
                        rss=res_dict.get('rss'),
                        amplification=res_dict.get('amplification'),
                        lin_comp=res_dict.get('lin_comp'),
                        p_amplitude=res_dict.get('p_amplitude'),
                        p_acrophase=res_dict.get('p_acrophase'),
                        p_amplification=res_dict.get('p_amplification'),
                        p_lin_comp=res_dict.get('p_lin_comp'),
                        q_amplitude=res_dict.get('q_amplitude'),
                        q_acrophase=res_dict.get('q_acrophase'),
                        q_amplification=res_dict.get('q_amplification'),
                        q_lin_comp=res_dict.get('q_lin_comp'),
                        amplitude_ci=res_dict.get('amplitude_ci'),
                        acrophase_ci=res_dict.get('acrophase_ci'),
                        amplification_ci=res_dict.get('amplification_ci'),
                        lin_comp_ci=res_dict.get('lin_comp_ci'),
                        peak_times=res_dict.get('peaks'),
                        trough_times=res_dict.get('troughs'),
                        times=times,
                        values=values,
                        best_model=res_dict.get('best_model'),
                        success=True
                    ))
                return analysis_results

            else:
                # Single period result
                acrophase_rad = result.get('acrophase')
                period_val = result.get('period', period if isinstance(period, (int, float)) else period[0])
                acrophase_hours = acrophase_to_hours(acrophase_rad, period_val)

                return AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method="cosinorpy_nonlinear_dependent",
                    mesor=result.get('mesor'),
                    amplitude=result.get('amplitude'),
                    acrophase=acrophase_rad,
                    acrophase_hours=acrophase_hours,
                    period=period_val,
                    p_value=result.get('p_value'),
                    q_value=result.get('q_value'),
                    n_components=result.get('n_components'),
                    rss=result.get('rss'),
                    amplification=result.get('amplification'),
                    lin_comp=result.get('lin_comp'),
                    p_amplitude=result.get('p_amplitude'),
                    p_acrophase=result.get('p_acrophase'),
                    p_amplification=result.get('p_amplification'),
                    p_lin_comp=result.get('p_lin_comp'),
                    q_amplitude=result.get('q_amplitude'),
                    q_acrophase=result.get('q_acrophase'),
                    q_amplification=result.get('q_amplification'),
                    q_lin_comp=result.get('q_lin_comp'),
                    amplitude_ci=result.get('amplitude_ci'),
                    acrophase_ci=result.get('acrophase_ci'),
                    amplification_ci=result.get('amplification_ci'),
                    lin_comp_ci=result.get('lin_comp_ci'),
                    peak_times=result.get('peaks'),
                    trough_times=result.get('troughs'),
                    times=times,
                    values=values,
                    best_model=result.get('best_model'),
                    success=True
                )

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_nonlinear_dependent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear_dependent",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cosinorpy_nonlinear_compare_independent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> List[ComparisonResult]:
        """
        Run nonlinear comparison for independent data.

        Compares all pairs of conditions using nonlinear model.
        For independent data - uses bootstrap for multi-component stats.
        """
        if self._cosinor is None:
            return []

        print(f"[DEBUG _run_cosinorpy_nonlinear_compare_independent_new] variable={variable}")

        try:
            # Get all unique conditions
            conditions = data[condition_col].unique().tolist()
            print(f"[DEBUG] Conditions to compare: {conditions}")

            if len(conditions) < 2:
                print(f"[DEBUG] Need at least 2 conditions to compare")
                return []

            # Get parameters
            period_range = parameters.get('period_range', (24.0, 24.0))
            period = period_range[0]  # Primary period

            use_dependent_model = parameters.get('use_dependent_model', True)

            # Per-condition periods: only valid when use_dependent_model=False and
            # exactly 2 conditions (CosinorPy's compare function uses global period1/period2)
            if not use_dependent_model and len(conditions) == 2:
                period1 = parameters.get('period1', period)
                period2 = parameters.get('period2', period)
            else:
                period1 = period
                period2 = period

            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            bootstrap_size = parameters.get('bootstrap_size', 1000)
            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)
            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            print(f"[DEBUG _run_cosinorpy_nonlinear_compare_independent_new] period={period}, period1={period1}, period2={period2}, use_dependent_model={use_dependent_model}")

            # Convert to CosinorPy format for INDEPENDENT data
            df_cosinorpy = self._convert_to_cosinorpy_format_all_conditions(
                data, variable, time_col, condition_col
            )

            # Load data
            from .cosinor_analysis import DataType
            self._cosinor.load_data(df_cosinorpy, DataType.INDEPENDENT)

            # Run comparison
            results = self._cosinor.nonlinear_compare_independent(
                variable=variable,
                conditions=conditions,
                period=period,
                period1=period1,
                period2=period2,
                n_components=n_components,
                bootstrap_size=bootstrap_size,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots,
                use_dependent_model=use_dependent_model
            )

            print(f"[DEBUG] Comparison returned {len(results)} pair results")

            # Convert to ComparisonResult objects
            comparison_results = []
            for res_dict in results:
                comparison_results.append(ComparisonResult(
                    variable=variable,
                    condition1=res_dict.get('condition1', ''),
                    condition2=res_dict.get('condition2', ''),
                    method="cosinorpy_nonlinear_compare_independent",
                    # Global F-test p/q (joint model only; None for independent model)
                    p1=res_dict.get('p_global'),
                    q1=res_dict.get('q_global'),
                    # Differences
                    amplitude_diff=res_dict.get('d_amplitude'),
                    acrophase_diff=res_dict.get('d_acrophase'),
                    # P-values for differences
                    p_amplitude=res_dict.get('p_d_amplitude'),
                    p_acrophase=res_dict.get('p_d_acrophase'),
                    # Q-values
                    q_amplitude=res_dict.get('q_d_amplitude'),
                    q_acrophase=res_dict.get('q_d_acrophase'),
                    # CIs for differences
                    amplitude_diff_ci=res_dict.get('d_amplitude_ci'),
                    acrophase_diff_ci=res_dict.get('d_acrophase_ci'),
                    # Nonlinear-specific fields
                    amplification_diff=res_dict.get('d_amplification'),
                    amplification_diff_ci=res_dict.get('d_amplification_ci'),
                    p_amplification=res_dict.get('p_d_amplification'),
                    q_amplification=res_dict.get('q_d_amplification'),
                    lin_comp_diff=res_dict.get('d_lin_comp'),
                    lin_comp_diff_ci=res_dict.get('d_lin_comp_ci'),
                    p_lin_comp=res_dict.get('p_d_lin_comp'),
                    q_lin_comp=res_dict.get('q_d_lin_comp'),
                    # Period and components info
                    period=res_dict.get('period1', period),
                    n_components=res_dict.get('n_components1', 1),
                    success=True
                ))

            # Enrich results with raw data and individual fit parameters
            self._enrich_independent_comparison_results(
                comparison_results, data, df_cosinorpy,
                variable, time_col, condition_col, period
            )

            return comparison_results

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_nonlinear_compare_independent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _run_cosinorpy_nonlinear_compare_dependent_new(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        data_file_path: Optional[str]
    ) -> List[ComparisonResult]:
        """
        Run nonlinear comparison for dependent/population data.

        Compares all pairs of conditions using nonlinear model.
        For population data - stats from replicate variance, no bootstrap.
        """
        if self._cosinor is None:
            return []

        print(f"[DEBUG _run_cosinorpy_nonlinear_compare_dependent_new] variable={variable}")

        try:
            # Get all unique conditions
            conditions = data[condition_col].unique().tolist()
            print(f"[DEBUG] Conditions to compare: {conditions}")

            if len(conditions) < 2:
                print(f"[DEBUG] Need at least 2 conditions to compare")
                return []

            # Get parameters
            period_range = parameters.get('period_range', (24.0, 24.0))
            period = period_range[0]  # Use single period for comparison

            n_components = parameters.get('n_components', [1])
            if not isinstance(n_components, list):
                n_components = [n_components]

            save_cosinorpy_plots = parameters.get('save_cosinorpy_plots', False)
            plot_folder = get_cosinorpy_plot_folder(data_file_path)

            # Convert to CosinorPy format for DEPENDENT data
            df_cosinorpy = self._convert_to_cosinorpy_format_population_all_conditions(
                data, variable, time_col, condition_col
            )

            # Load data
            from .cosinor_analysis import DataType
            self._cosinor.load_data(df_cosinorpy, DataType.DEPENDENT)

            # Run comparison
            results = self._cosinor.nonlinear_compare_dependent(
                variable=variable,
                conditions=conditions,
                period=period,
                n_components=n_components,
                save_folder=plot_folder,
                save_cosinorpy_plots=save_cosinorpy_plots
            )

            print(f"[DEBUG] Comparison returned {len(results)} pair results")

            # Convert to ComparisonResult objects
            comparison_results = []
            for res_dict in results:
                comparison_results.append(ComparisonResult(
                    variable=variable,
                    condition1=res_dict.get('condition1', ''),
                    condition2=res_dict.get('condition2', ''),
                    method="cosinorpy_nonlinear_compare_dependent",
                    # Differences
                    amplitude_diff=res_dict.get('d_amplitude'),
                    acrophase_diff=res_dict.get('d_acrophase'),
                    # P-values for differences (use amplification/lin_comp p-values)
                    p_amplitude=res_dict.get('p_d_amplitude'),
                    p_acrophase=res_dict.get('p_d_acrophase'),
                    # Q-values
                    q_amplitude=res_dict.get('q_d_amplitude'),
                    q_acrophase=res_dict.get('q_d_acrophase'),
                    # CIs for differences
                    amplitude_diff_ci=res_dict.get('d_amplitude_ci'),
                    acrophase_diff_ci=res_dict.get('d_acrophase_ci'),
                    # Nonlinear-specific fields
                    amplification_diff=res_dict.get('d_amplification'),
                    amplification_diff_ci=res_dict.get('d_amplification_ci'),
                    p_amplification=res_dict.get('p_d_amplification'),
                    q_amplification=res_dict.get('q_d_amplification'),
                    lin_comp_diff=res_dict.get('d_lin_comp'),
                    lin_comp_diff_ci=res_dict.get('d_lin_comp_ci'),
                    p_lin_comp=res_dict.get('p_d_lin_comp'),
                    q_lin_comp=res_dict.get('q_d_lin_comp'),
                    # Period and components info
                    period=res_dict.get('period1', period),
                    n_components=res_dict.get('n_components1', 1),
                    success=True
                ))

            # Enrich results with raw data and individual population fit parameters
            self._enrich_dependent_comparison_results(
                comparison_results, data, df_cosinorpy,
                variable, time_col, condition_col, period
            )

            return comparison_results

        except Exception as e:
            print(f"[DEBUG _run_cosinorpy_nonlinear_compare_dependent_new] ERROR: {e}")
            import traceback
            traceback.print_exc()
            return []

    # =========================================================================
    # RhythmCount Methods
    # =========================================================================

    def _build_rhythmcount_params(self, parameters: Dict[str, Any]) -> RhythmCountParameters:
        """Build a RhythmCountParameters object from the GUI parameters dict."""
        period = parameters.get('period', 24.0)
        if isinstance(period, list):
            period = period[0] if period else 24.0

        # Count models: GUI passes list of string values
        raw_models = parameters.get('rc_count_models', None)
        if raw_models:
            count_models = []
            for m in raw_models:
                try:
                    count_models.append(CountModel(m))
                except ValueError:
                    pass
            if not count_models:
                count_models = list(CountModel)
        else:
            count_models = list(CountModel)

        # n_components
        n_components = parameters.get('rc_n_components', [1, 2, 3])

        # selection_test
        raw_test = parameters.get('rc_selection_test', 'AIC')
        try:
            selection_test = SelectionTest(raw_test)
        except ValueError:
            selection_test = SelectionTest.AIC

        return RhythmCountParameters(
            period=float(period),
            n_components=list(n_components),
            count_models=count_models,
            selection_test=selection_test,
            eval_order=parameters.get('rc_eval_order', True),
            repetitions=parameters.get('rc_repetitions', 20),
            precision_rate=parameters.get('rc_precision_rate', 2.0),
            clean_data=parameters.get('rc_clean_data', False),
        )

    def _prepare_rhythmcount_df(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
    ) -> pd.DataFrame:
        """Filter data by condition and rename columns to X/Y for RhythmCount."""
        mask = data[condition_col] == condition
        filtered = data[mask][[time_col, variable]].copy()
        filtered = filtered.rename(columns={time_col: 'X', variable: 'Y'})
        filtered = filtered.dropna()
        return filtered

    def _run_rhythmcount(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        analysis_type: 'AnalysisType',
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
    ) -> AnalysisResult:
        """Dispatcher for all RhythmCount analysis types."""
        if self._rhythmcount is None:
            return AnalysisResult(
                variable=variable or '',
                condition=condition or '',
                method=analysis_type.value,
                success=False,
                message="RhythmCount not available. Install statsmodels: pip install statsmodels",
            )

        params = self._build_rhythmcount_params(parameters)

        try:
            if analysis_type == AnalysisType.RHYTHMCOUNT_COMPARE_GROUPS:
                return self._run_rhythmcount_compare_groups(
                    data, variable, condition, time_col, condition_col, parameters, params
                )

            # All other types require single-condition data
            if not condition:
                return AnalysisResult(
                    variable=variable or '',
                    condition='',
                    method=analysis_type.value,
                    success=False,
                    message="No condition specified for RhythmCount analysis.",
                )

            df_xy = self._prepare_rhythmcount_df(data, variable, condition, time_col, condition_col)
            if len(df_xy) < 4:
                return AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method=analysis_type.value,
                    success=False,
                    message="Insufficient data points (minimum 4 required).",
                )

            self._rhythmcount.load_data(df_xy)

            if analysis_type == AnalysisType.RHYTHMCOUNT_SINGLE:
                return self._run_rhythmcount_single(variable, condition, parameters, params)
            elif analysis_type == AnalysisType.RHYTHMCOUNT_ALL_MODELS:
                return self._run_rhythmcount_all_models(variable, condition, parameters, params)
            elif analysis_type == AnalysisType.RHYTHMCOUNT_BEST_MODEL:
                return self._run_rhythmcount_best_model(variable, condition, parameters, params)
            elif analysis_type == AnalysisType.RHYTHMCOUNT_PARAMETER_CIS:
                return self._run_rhythmcount_parameter_cis(variable, condition, parameters, params)
            else:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method=analysis_type.value, success=False,
                    message=f"Unknown RhythmCount type: {analysis_type.value}",
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=variable or '',
                condition=condition or '',
                method=analysis_type.value,
                success=False,
                message=str(e),
            )

    def _run_rhythmcount_single(
        self,
        variable: str,
        condition: str,
        parameters: Dict[str, Any],
        params: RhythmCountParameters,
    ) -> AnalysisResult:
        """
        Fit one count distribution across all (period, n_components) combinations.

        Returns a full results table serialized in results_table_json, with the
        best row (lowest AIC) surfaced as the top-level AnalysisResult fields.
        """
        from dataclasses import replace as dc_replace

        raw_model = parameters.get('rc_single_count_model', 'poisson')
        try:
            count_model = CountModel(raw_model)
        except ValueError:
            count_model = CountModel.POISSON

        # N components: comma-separated list from GUI text field
        n_comps = parameters.get('rc_single_n_components_list', [1])
        if not n_comps:
            n_comps = [1]

        # Period: either a single float or a list generated from the range widget
        period_param = parameters.get('period', 24.0)
        periods = period_param if isinstance(period_param, list) else [period_param]

        rows = []
        for period in periods:
            per_params = dc_replace(params, period=float(period))
            for n_comp in n_comps:
                result = self._rhythmcount.fit_single_model(count_model, n_comp, per_params)
                peaks_str = ', '.join(f'{p:.2f}h' for p in result.peaks) if len(result.peaks) > 0 else ''
                rows.append({
                    'period': period,
                    'count_model': result.count_model,
                    'n_components': result.n_components,
                    'amplitude': result.amplitude,
                    'mesor': result.mesor,
                    'peaks': peaks_str,
                    'llr_pvalue': result.llr_pvalue,
                    'AIC': result.AIC,
                    'BIC': result.BIC,
                    'RSS': result.RSS,
                    'log_likelihood': result.log_likelihood,
                    'McFadden_R2': result.prsquared,
                    'success': result.success,
                    'error': result.error or '',
                })

        if not rows:
            return AnalysisResult(
                variable=variable, condition=condition,
                method='rhythmcount_single', success=False,
                message="No results produced.",
            )

        df = pd.DataFrame(rows)

        def _safe(val):
            try:
                return None if np.isnan(float(val)) else float(val)
            except (TypeError, ValueError):
                return None

        def _row_to_result(row) -> AnalysisResult:
            err = row.get('error', '')
            msg = err if err else (
                f"Period={row.get('period')}h, N={row.get('n_components')}, "
                f"Model={row.get('count_model')}"
            )
            return AnalysisResult(
                variable=variable,
                condition=condition,
                method='rhythmcount_single',
                mesor=_safe(row.get('mesor')),
                amplitude=_safe(row.get('amplitude')),
                p_value=_safe(row.get('llr_pvalue')),
                aic=_safe(row.get('AIC')),
                bic=_safe(row.get('BIC')),
                rss=_safe(row.get('RSS')),
                log_likelihood=_safe(row.get('log_likelihood')),
                r_squared=_safe(row.get('McFadden_R2')),
                n_components=int(row.get('n_components', 1)),
                period=float(row.get('period', periods[0])),
                success=bool(row.get('success', False)),
                message=msg,
            )

        # Multiple (period, n_comp) combinations → return one result per row,
        # matching CosinorPy's behaviour so the UI shows a row per period.
        if len(rows) > 1:
            return [_row_to_result(row) for _, row in df.iterrows()]

        # Single combination → keep original single-result behaviour.
        return _row_to_result(df.iloc[0])

    def _run_rhythmcount_all_models(
        self,
        variable: str,
        condition: str,
        parameters: Dict[str, Any],
        params: RhythmCountParameters,
    ) -> AnalysisResult:
        """Fit all (count_model, n_components) combinations across the period range.

        Returns one AnalysisResult per (period, count_model, n_components) combination
        so the results table shows every fit, not just the best-by-AIC one.
        """
        from dataclasses import replace as dc_replace

        def _safe(val):
            try:
                f = float(val)
                return None if np.isnan(f) else f
            except (TypeError, ValueError):
                return None

        period_param = parameters.get('period', 24.0)
        periods = period_param if isinstance(period_param, list) else [period_param]

        results = []
        for period in periods:
            per_params = dc_replace(params, period=float(period))
            df_results = self._rhythmcount.fit_all_models(per_params)

            if df_results is None or df_results.empty:
                continue

            for _, row in df_results.iterrows():
                err = str(row.get('error', '') or '')
                msg = err if err else (
                    f"Period={period}h, Model={row.get('count_model', 'N/A')}, "
                    f"N={row.get('n_components', 'N/A')}"
                )
                results.append(AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method='rhythmcount_all_models',
                    mesor=_safe(row.get('mesor')),
                    amplitude=_safe(row.get('amplitude')),
                    p_value=_safe(row.get('llr_pvalue')),
                    aic=_safe(row.get('AIC')),
                    bic=_safe(row.get('BIC')),
                    rss=_safe(row.get('RSS')),
                    log_likelihood=_safe(row.get('log_likelihood')),
                    r_squared=_safe(row.get('prsquared')),
                    n_components=int(row.get('n_components', 1)),
                    period=float(period),
                    success=bool(row.get('success', False)),
                    message=msg,
                ))

        if not results:
            return AnalysisResult(
                variable=variable, condition=condition,
                method='rhythmcount_all_models', success=False,
                message="fit_all_models returned empty results.",
            )

        if len(results) > 1:
            return results

        return results[0]

    def _run_rhythmcount_best_model(
        self,
        variable: str,
        condition: str,
        parameters: Dict[str, Any],
        params: RhythmCountParameters,
    ) -> AnalysisResult:
        """Fit all models and automatically select the best one per period."""
        from dataclasses import replace as dc_replace

        def _safe(val):
            try:
                f = float(val)
                return None if np.isnan(f) else f
            except (TypeError, ValueError):
                return None

        period_param = parameters.get('period', 24.0)
        periods = period_param if isinstance(period_param, list) else [period_param]

        results = []
        for period in periods:
            per_params = dc_replace(params, period=float(period))
            result = self._rhythmcount.fit_best_model(per_params)

            table_json = None
            if result.success and not result.all_results.empty:
                table_json = result.all_results.to_json(orient='records', default_handler=str)

            results.append(AnalysisResult(
                variable=variable,
                condition=condition,
                method='rhythmcount_best_model',
                mesor=_safe(result.mesor),
                amplitude=_safe(result.amplitude),
                peak_times=result.peaks.tolist() if len(result.peaks) > 0 else None,
                p_value=_safe(result.llr_pvalue),
                aic=_safe(result.AIC),
                bic=_safe(result.BIC),
                rss=_safe(result.RSS),
                r_squared=_safe(result.prsquared),
                n_components=result.n_components,
                period=float(period),
                times=result.X_test if len(result.X_test) > 0 else None,
                fitted_values=result.Y_test if len(result.Y_test) > 0 else None,
                results_table_json=table_json,
                success=result.success,
                message=(
                    f"Period={period}h — Best: {result.count_model}, N={result.n_components}, "
                    f"criterion={result.selection_test}"
                    + (f" | Error: {result.error}" if result.error else "")
                ),
            ))

        if not results:
            return AnalysisResult(
                variable=variable, condition=condition,
                method='rhythmcount_best_model', success=False,
                message="fit_best_model returned no results.",
            )

        if len(results) > 1:
            return results

        return results[0]

    def _run_rhythmcount_parameter_cis(
        self,
        variable: str,
        condition: str,
        parameters: Dict[str, Any],
        params: RhythmCountParameters,
    ) -> AnalysisResult:
        """Compute bootstrap confidence intervals for each (period, n_components) combination."""
        from dataclasses import replace as dc_replace

        raw_model = parameters.get('rc_single_count_model', 'poisson')
        try:
            count_model = CountModel(raw_model)
        except ValueError:
            count_model = CountModel.POISSON

        n_comps_list = parameters.get('rc_single_n_components_list', [1])
        if not n_comps_list:
            n_comps_list = [1]

        period_param = parameters.get('period', 24.0)
        periods = period_param if isinstance(period_param, list) else [period_param]

        ref_peaks_raw = parameters.get('rc_reference_peaks', None)

        results = []
        for period in periods:
            per_params = dc_replace(params, period=float(period))

            if ref_peaks_raw is not None:
                reference_peaks = np.array(ref_peaks_raw, dtype=float)
            else:
                reference_peaks = np.array([float(period) / 2.0])

            for n_comp in n_comps_list:
                result = self._rhythmcount.calculate_parameter_cis(
                    count_model, int(n_comp), reference_peaks, per_params
                )

                amplitude_ci = None
                mesor_ci = None
                if result.success and len(result.amplitude_CIs) >= 2:
                    amplitude_ci = (float(result.amplitude_CIs[0]), float(result.amplitude_CIs[1]))
                if result.success and len(result.mesor_CIs) >= 2:
                    mesor_ci = (float(result.mesor_CIs[0]), float(result.mesor_CIs[1]))

                results.append(AnalysisResult(
                    variable=variable,
                    condition=condition,
                    method='rhythmcount_parameter_cis',
                    n_components=int(n_comp),
                    period=float(period),
                    amplitude_ci=amplitude_ci,
                    mesor_ci=mesor_ci,
                    success=result.success,
                    message=(
                        f"Bootstrap CIs ({result.repetitions} reps), "
                        f"period={period}h, model={result.count_model}, N={result.n_components}"
                        + (f" | Error: {result.error}" if result.error else "")
                    ),
                ))

        if not results:
            return AnalysisResult(
                variable=variable, condition=condition,
                method='rhythmcount_parameter_cis', success=False,
                message="No results produced.",
            )

        if len(results) > 1:
            return results

        return results[0]

    def _run_rhythmcount_compare_groups(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any],
        params: RhythmCountParameters,
    ) -> AnalysisResult:
        """Compare count-based cosinor models across all groups, for each period in the range."""
        from dataclasses import replace as dc_replace

        # Build X/Y/group DataFrame with all conditions (done once, period-independent)
        df_xy = data[[time_col, variable, condition_col]].copy()
        df_xy = df_xy.rename(columns={time_col: 'X', variable: 'Y'})
        df_xy = df_xy.dropna(subset=['X', 'Y'])
        self._rhythmcount.load_data(df_xy)

        period_param = parameters.get('period', 24.0)
        periods = period_param if isinstance(period_param, list) else [period_param]

        results = []
        for period in periods:
            per_params = dc_replace(params, period=float(period))
            result = self._rhythmcount.compare_groups(
                group_column=condition_col,
                params=per_params,
                plot_comparison=False,
            )

            table_json = None
            if result.success and not result.results_table.empty:
                table_json = result.results_table.to_json(orient='records', default_handler=str)

            results.append(AnalysisResult(
                variable=variable or '',
                condition='all',
                method='rhythmcount_compare_groups',
                period=float(period),
                results_table_json=table_json,
                success=result.success,
                message=(
                    f"Period={period}h — Group comparison ({result.group_column}), "
                    f"criterion={result.selection_test}"
                    + (f" | Error: {result.error}" if result.error else "")
                ),
            ))

        if not results:
            return AnalysisResult(
                variable=variable or '', condition='all',
                method='rhythmcount_compare_groups', success=False,
                message="compare_groups returned no results.",
            )

        if len(results) > 1:
            return results

        return results[0]
