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

import pandas as pd
import numpy as np

# Import analysis modules
from .cosinor_analysis import (
    CosinorAnalyzer, CosinorParameters, DataType, ModelType, AnalysisMode,
    COSINORPY_AVAILABLE
)
from .circacompare_analysis import (
    CircaCompareAnalyzer, CircaSingleResult, CircaCompareResult
)
from .rhythm_analysis import (
    RhythmAnalyzer, AnalysisMethod as RhythmMethod
)


class AnalysisType(Enum):
    """Types of analysis available."""
    # CosinorPy
    COSINORPY_SINGLE = "cosinorpy_single"
    COSINORPY_MULTI = "cosinorpy_multi"
    COSINORPY_POPULATION = "cosinorpy_population"
    COSINORPY_COMPARE = "cosinorpy_compare"
    COSINORPY_COUNT = "cosinorpy_count"
    COSINORPY_NONLINEAR = "cosinorpy_nonlinear"
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
    CWT = "cwt"
    LME = "lme"


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
    p_amplitude: Optional[float] = None
    p_acrophase: Optional[float] = None
    r_squared: Optional[float] = None
    
    # Confidence intervals
    mesor_ci: Optional[Tuple[float, float]] = None
    amplitude_ci: Optional[Tuple[float, float]] = None
    acrophase_ci: Optional[Tuple[float, float]] = None
    
    # Method-specific
    tau: Optional[float] = None  # JTK
    power: Optional[float] = None  # Lomb-Scargle
    dominant_period: Optional[float] = None
    
    # Raw data for plotting
    times: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    fitted_values: Optional[np.ndarray] = None
    
    # Status
    success: bool = True
    message: str = ""
    
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
    
    # Differences
    mesor_diff: Optional[float] = None
    amplitude_diff: Optional[float] = None
    acrophase_diff: Optional[float] = None
    
    period: float = 24.0
    success: bool = True
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AnalysisEngine:
    """
    Central analysis engine for CircaScope.
    
    Provides a unified interface to all analysis methods and handles
    data preparation, execution, and result formatting.
    """
    
    def __init__(self):
        """Initialize the analysis engine."""
        self._cosinor = CosinorAnalyzer() if COSINORPY_AVAILABLE else None
        self._circacompare = CircaCompareAnalyzer()
        self._rhythm = RhythmAnalyzer()
    
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
        
        return True, "Available"
    
    def run_analysis(
        self,
        data: pd.DataFrame,
        variable: str,
        condition: str,
        analysis_type: AnalysisType,
        time_col: str = 'time',
        condition_col: str = 'condition',
        parameters: Optional[Dict[str, Any]] = None
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
        
        Returns:
            AnalysisResult object
        """
        parameters = parameters or {}
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
            # Route to appropriate method
            if analysis_type == AnalysisType.COSINORPY_SINGLE:
                return self._run_cosinorpy_single(
                    times, values, variable, condition, parameters
                )
            
            elif analysis_type == AnalysisType.COSINORPY_MULTI:
                return self._run_cosinorpy_multi(
                    times, values, variable, condition, parameters
                )
            
            elif analysis_type == AnalysisType.CIRCACOMPARE_SINGLE:
                return self._run_circacompare_single(
                    times, values, variable, condition, parameters
                )
            
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

            elif analysis_type == AnalysisType.COSINORPY_POPULATION:
                return self._run_cosinorpy_population(
                    data, variable, condition, time_col, condition_col, parameters
                )

            elif analysis_type == AnalysisType.COSINORPY_COUNT:
                return self._run_cosinorpy_count(
                    data, variable, condition, time_col, condition_col, parameters
                )

            elif analysis_type == AnalysisType.COSINORPY_NONLINEAR:
                return self._run_cosinorpy_nonlinear(
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

            elif analysis_type == AnalysisType.CWT:
                return self._run_cwt(
                    times, values, variable, condition, parameters
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
        parameters: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
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
            
            elif analysis_type == AnalysisType.COSINORPY_COMPARE:
                return self._run_cosinorpy_comparison(
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
            return ComparisonResult(
                variable=variable,
                condition1=condition1,
                condition2=condition2,
                method=analysis_type.value,
                success=False,
                message=str(e)
            )
    
    # =========================================================================
    # CosinorPy Methods
    # =========================================================================
    
    def _run_cosinorpy_single(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run CosinorPy single-component cosinor."""
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_single",
                success=False, message="CosinorPy not available"
            )
        
        period = parameters.get('period', 24.0)
        
        # Prepare data
        df = pd.DataFrame({
            'time': times,
            'condition': condition,
            variable: values
        })
        
        # Run analysis
        result = self._cosinor.single_cosinor(
            df, time_col='time', value_col=variable,
            period=period, save_to=None
        )
        
        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_single",
                success=False, message="Analysis failed"
            )
        
        # Extract parameters
        acrophase_rad = result.get('acrophase', 0)
        acrophase_hours = (acrophase_rad * period) / (2 * np.pi)
        if acrophase_hours < 0:
            acrophase_hours += period
        
        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="cosinorpy_single",
            mesor=result.get('mesor'),
            amplitude=result.get('amplitude'),
            acrophase=acrophase_rad,
            acrophase_hours=acrophase_hours,
            period=period,
            p_value=result.get('p_value'),
            r_squared=result.get('r_squared'),
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
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run CosinorPy multi-component cosinor."""
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_multi",
                success=False, message="CosinorPy not available"
            )
        
        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 2)
        
        df = pd.DataFrame({
            'time': times,
            'condition': condition,
            variable: values
        })
        
        result = self._cosinor.multi_component_cosinor(
            df, time_col='time', value_col=variable,
            period=period, n_components=n_components
        )
        
        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_multi",
                success=False, message="Analysis failed"
            )
        
        acrophase_rad = result.get('acrophase', 0)
        acrophase_hours = (acrophase_rad * period) / (2 * np.pi)
        if acrophase_hours < 0:
            acrophase_hours += period
        
        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="cosinorpy_multi",
            mesor=result.get('mesor'),
            amplitude=result.get('amplitude'),
            acrophase=acrophase_rad,
            acrophase_hours=acrophase_hours,
            period=period,
            p_value=result.get('p_value'),
            r_squared=result.get('r_squared'),
            times=times,
            values=values,
            success=True
        )
    
    def _run_cosinorpy_comparison(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> ComparisonResult:
        """Run CosinorPy differential rhythmicity."""
        if self._cosinor is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare",
                success=False, message="CosinorPy not available"
            )
        
        period = parameters.get('period', 24.0)
        
        # Filter for both conditions
        mask = data[condition_col].isin([condition1, condition2])
        filtered = data[mask].copy()
        
        # Run differential analysis
        result = self._cosinor.differential_rhythmicity(
            filtered, time_col=time_col, condition_col=condition_col,
            value_col=variable, period=period
        )
        
        if result is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="cosinorpy_compare",
                success=False, message="Analysis failed"
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
            p_mesor=result.p_mesor,
            p_amplitude=result.p_amplitude,
            p_acrophase=result.p_acrophase,
            period=period,
            success=True
        )
    
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
    ) -> AnalysisResult:
        """Run CircaCompare single fit."""
        period = parameters.get('period', 24.0)
        loss = parameters.get('loss', 'huber')
        f_scale = parameters.get('f_scale', 1.0)
        max_iterations = parameters.get('max_iterations', 500)

        # CircaCompare requires loading data first
        # Create a simple dataframe for this analysis
        df = pd.DataFrame({
            'time': times,
            'condition': condition,
            variable: values
        })

        self._circacompare.load_dataframe(df)
        self._circacompare.set_period(period)
        self._circacompare.set_loss(loss)
        self._circacompare.set_f_scale(f_scale)
        self._circacompare.set_max_iterations(max_iterations)

        result = self._circacompare.fit_single(
            variable=variable, condition=condition
        )
        
        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="circacompare_single",
                success=False, message="Fit failed"
            )
        
        acrophase_hours = (result.acrophase * period) / (2 * np.pi)
        if acrophase_hours < 0:
            acrophase_hours += period

        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="circacompare_single",
            mesor=result.mesor,
            amplitude=result.amplitude,
            acrophase=result.acrophase,
            acrophase_hours=acrophase_hours,
            period=result.period,
            mesor_ci=result.mesor_ci,
            amplitude_ci=result.amplitude_ci,
            acrophase_ci=result.acrophase_ci,
            times=times,
            values=values,
            success=True
        )
    
    def _run_circacompare_comparison(
        self,
        data: pd.DataFrame,
        variable: str,
        condition1: str,
        condition2: str,
        time_col: str,
        condition_col: str,
        parameters: Dict[str, Any]
    ) -> ComparisonResult:
        """Run CircaCompare group comparison."""
        period = parameters.get('period', 24.0)
        loss = parameters.get('loss', 'huber')
        f_scale = parameters.get('f_scale', 1.0)
        max_iterations = parameters.get('max_iterations', 500)

        # Prepare data - only include the two conditions being compared
        mask = data[condition_col].isin([condition1, condition2])
        filtered = data[mask].copy()

        # Load data into CircaCompare analyzer
        self._circacompare.load_dataframe(filtered)
        self._circacompare.set_period(period)
        self._circacompare.set_loss(loss)
        self._circacompare.set_f_scale(f_scale)
        self._circacompare.set_max_iterations(max_iterations)

        result = self._circacompare.compare(
            variable=variable,
            condition1=condition1,
            condition2=condition2
        )
        
        if result is None:
            return ComparisonResult(
                variable=variable, condition1=condition1, condition2=condition2,
                method="circacompare_compare",
                success=False, message="Comparison failed"
            )
        
        return ComparisonResult(
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
            p_mesor=result.p_mesor,
            p_amplitude=result.p_amplitude,
            p_acrophase=result.p_acrophase,
            mesor_diff=result.mesor_diff,
            amplitude_diff=result.amplitude_diff,
            acrophase_diff=result.acrophase_diff,
            period=period,
            success=True
        )
    
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
    ) -> AnalysisResult:
        """Run JTK Cycle analysis."""
        period_range = parameters.get('period_range', [20, 28])
        
        result = self._rhythm.run_jtk(
            times, values, period_range=period_range
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
            acrophase_hours=result.phase,
            p_value=result.p_value,
            tau=result.tau,
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
        period_range = parameters.get('period_range', (20, 28))
        
        result = self._rhythm.run_cosinor(
            times, values, period_range=period_range
        )
        
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
            acrophase=result.acrophase,
            acrophase_hours=result.acrophase_hours,
            period=result.period,
            p_value=result.p_value,
            r_squared=result.r_squared,
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
        n_harmonics = parameters.get('n_harmonics', 2)
        period = parameters.get('period', 24.0)
        
        result = self._rhythm.run_harmonic_cosinor(
            times, values, period=period, n_harmonics=n_harmonics
        )
        
        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="harmonic_cosinor",
                success=False, message="Harmonic cosinor failed"
            )
        
        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="harmonic_cosinor",
            mesor=result.mesor,
            amplitude=result.amplitude,
            acrophase=result.acrophase,
            acrophase_hours=result.acrophase_hours,
            period=period,
            p_value=result.p_value,
            r_squared=result.r_squared,
            times=times,
            values=values,
            success=True
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
        period_range = parameters.get('period_range', (4, 48))
        
        result = self._rhythm.run_lomb_scargle(
            times, values, period_range=period_range
        )
        
        if result is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="lomb_scargle",
                success=False, message="Lomb-Scargle failed"
            )
        
        return AnalysisResult(
            variable=variable,
            condition=condition,
            method="lomb_scargle",
            dominant_period=result.dominant_period,
            power=result.peak_power,
            p_value=result.p_value,
            period=result.dominant_period,
            times=times,
            values=values,
            success=True
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
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run CosinorPy population-mean cosinor analysis."""
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_population",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)

        try:
            result = self._cosinor.population_cosinor(
                variable=variable,
                condition=condition,
                period=period,
                n_components=n_components
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cosinorpy_population",
                    success=False, message="Analysis failed"
                )

            # Extract parameters from population result
            acrophase_rad = result.get('population_acrophase', 0)
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
                method="cosinorpy_population",
                mesor=result.get('population_mesor'),
                amplitude=result.get('population_amplitude'),
                acrophase=acrophase_rad,
                acrophase_hours=acrophase_hours,
                period=period,
                p_value=result.get('p_value'),
                r_squared=result.get('r_squared'),
                times=times,
                values=values,
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_population",
                success=False, message=f"Error: {str(e)}"
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
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run CosinorPy nonlinear cosinor analysis."""
        if self._cosinor is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear",
                success=False, message="CosinorPy not available"
            )

        period = parameters.get('period', 24.0)
        n_components = parameters.get('n_components', 1)

        try:
            result = self._cosinor.nonlinear_cosinor(
                variable=variable,
                condition=condition,
                period=period,
                n_components=n_components
            )

            if result is None:
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
                times=times,
                values=values,
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosinorpy_nonlinear",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_ar_jtk(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run AR-JTK Cycle analysis (JTK with autocorrelation correction)."""
        if self._rhythm is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="ar_jtk",
                success=False, message="RhythmAnalysis not available"
            )

        period_range = parameters.get('period_range', None)
        ar_lag = parameters.get('ar_lag', 1)
        ljungbox_lag = parameters.get('ljungbox_lag', 10)

        try:
            result, autocorr_detected = self._rhythm.run_ar_jtk(
                variable=variable,
                condition=condition,
                period_range=period_range,
                ar_lag=ar_lag,
                ljungbox_lag=ljungbox_lag
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="ar_jtk",
                    success=False, message="Analysis failed"
                )

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="ar_jtk",
                period=result.period,
                p_value=result.p_value,
                acrophase_hours=result.peak_time,
                amplitude=result.amplitude if hasattr(result, 'amplitude') else None,
                times=times,
                values=values,
                metadata={'autocorrelation_detected': autocorr_detected},
                success=True
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
    ) -> AnalysisResult:
        """Run Cosine-Kendall nonparametric analysis."""
        if self._rhythm is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cosine_kendall",
                success=False, message="RhythmAnalysis not available"
            )

        period_range = parameters.get('period_range', None)
        interval = parameters.get('interval', None)

        try:
            result = self._rhythm.run_cosine_kendall(
                variable=variable,
                condition=condition,
                period_range=period_range,
                interval=interval
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cosine_kendall",
                    success=False, message="Analysis failed"
                )

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cosine_kendall",
                period=result.period,
                p_value=result.p_value,
                acrophase_hours=result.peak_time,
                amplitude=result.amplitude if hasattr(result, 'amplitude') else None,
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
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Fourier F24 analysis (effect size measure)."""
        if self._rhythm is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="fourier_f24",
                success=False, message="RhythmAnalysis not available"
            )

        target_period = parameters.get('target_period', 24.0)
        n_permutations = parameters.get('n_permutations', 1000)

        try:
            result = self._rhythm.run_fourier_f24(
                variable=variable,
                condition=condition,
                target_period=target_period,
                n_permutations=n_permutations
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="fourier_f24",
                    success=False, message="Analysis failed"
                )

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="fourier_f24",
                period=target_period,
                amplitude=result.f24_statistic,  # F24 is an effect size measure
                p_value=result.p_value,
                times=times,
                values=values,
                metadata={'f24_statistic': result.f24_statistic},
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="fourier_f24",
                success=False, message=f"Error: {str(e)}"
            )

    def _run_cwt(
        self,
        times: np.ndarray,
        values: np.ndarray,
        variable: str,
        condition: str,
        parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """Run Continuous Wavelet Transform analysis."""
        if self._rhythm is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cwt",
                success=False, message="RhythmAnalysis not available"
            )

        sampling_interval = parameters.get('sampling_interval', None)
        wavelet = parameters.get('wavelet', 'cmor1.5-1.0')
        period_range = parameters.get('period_range', None)

        try:
            result = self._rhythm.run_cwt(
                variable=variable,
                condition=condition,
                sampling_interval=sampling_interval,
                wavelet=wavelet,
                period_range=period_range
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="cwt",
                    success=False, message="Analysis failed"
                )

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="cwt",
                period=result.dominant_period,
                amplitude=result.max_power,
                times=times,
                values=values,
                metadata={
                    'dominant_period': result.dominant_period,
                    'max_power': result.max_power,
                    'period_at_max': result.period_at_max
                },
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="cwt",
                success=False, message=f"Error: {str(e)}"
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
        """Run Linear Mixed Effects model analysis."""
        if self._rhythm is None:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="lme",
                success=False, message="RhythmAnalysis not available"
            )

        fixed_effects = parameters.get('fixed_effects', [time_col])
        random_effect = parameters.get('random_effect', 'replicate')

        try:
            result = self._rhythm.run_lme(
                dependent=variable,
                fixed_effects=fixed_effects,
                random_effect=random_effect,
                condition=condition
            )

            if result is None:
                return AnalysisResult(
                    variable=variable, condition=condition,
                    method="lme",
                    success=False, message="Analysis failed"
                )

            # Get times and values for plotting
            subset = data[data[condition_col] == condition]
            times = subset[time_col].values
            values = subset[variable].values

            return AnalysisResult(
                variable=variable,
                condition=condition,
                method="lme",
                p_value=result.p_value if hasattr(result, 'p_value') else None,
                times=times,
                values=values,
                metadata={
                    'aic': result.aic if hasattr(result, 'aic') else None,
                    'bic': result.bic if hasattr(result, 'bic') else None
                },
                success=True
            )
        except Exception as e:
            return AnalysisResult(
                variable=variable, condition=condition,
                method="lme",
                success=False, message=f"Error: {str(e)}"
            )
