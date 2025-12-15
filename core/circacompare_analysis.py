"""
CircaCompare Analysis Module
============================

A comprehensive module for circadian rhythm analysis using the CircaCompare method.
Based on the CircaCompare implementation by RWParsons:
https://github.com/RWParsons/circacompare_py

This module is designed to be integrated with a GUI interface and provides:
- Single Cosinor Analysis (circa_single)
- Differential Rhythmicity Analysis (circacompare)

CSV Format Requirements:
------------------------
The module expects CSV files with the following structure:

    time,condition,variable1,variable2,...
    0,winter,1.2,3.4
    4,winter,2.3,4.5
    8,winter,1.8,3.9
    ...
    0,summer,1.5,3.2
    ...

If replicates exist at the same timepoint:
    time,condition,replicate,variable1,variable2,...
    0,winter,1,1.2,3.4
    0,winter,2,1.3,3.5
    ...

Dependencies:
-------------
    pip install numpy scipy pandas

Note on Period Conversion:
--------------------------
CircaCompare internally works with time in radians. This module automatically
converts time values to radians based on the specified period:
    t_radians = 2 * π * t / period

For a 24-hour period, time=12 hours would be converted to π radians.

Author: Francisco Tassara Generated for GUI integration
Version: 1.0.0
"""

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

import pandas as pd
import numpy as np
from scipy.optimize import least_squares


# =============================================================================
# CIRCACOMPARE CORE FUNCTIONS (from RWParsons/circacompare_py)
# =============================================================================

class CircaCompareConstants:
    """Default constants for CircaCompare optimization."""
    LOSS = 'linear'
    F_SCALE = 1.0
    MAX_ITERATIONS = 500


def _fun_circa_single(x: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Residual function for single-group cosinor fitting.
    
    Model: y = mesor + amplitude * cos(t - acrophase)
    
    Args:
        x: Parameter array [mesor, amplitude, acrophase]
        t: Time values in radians
        y: Observed values
    
    Returns:
        Residuals (predicted - observed)
    """
    return x[0] + x[1] * np.cos(t - x[2]) - y


def _fun_circacompare(x: np.ndarray, t: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Residual function for two-group cosinor comparison.
    
    Model: y = (mesor1 + d_mesor*g) + (amp1 + d_amp*g) * cos(t - (acr1 + d_acr*g))
    
    Where g is the group indicator (0 or 1).
    
    Args:
        x: Parameter array [mesor1, d_mesor, amp1, d_amp, acr1, d_acr]
        t: Time values in radians
        y: Observed values
        g: Group indicator (0 or 1)
    
    Returns:
        Residuals (predicted - observed)
    """
    return (x[0] + x[1] * g) + (x[2] + x[3] * g) * np.cos(t - (x[4] + x[5] * g)) - y


def _param_standard_errors(optimised_result, y: np.ndarray) -> np.ndarray:
    """
    Calculate standard errors for fitted parameters.
    
    Uses the Jacobian from the optimization to estimate parameter uncertainty.
    
    Args:
        optimised_result: Result from scipy.optimize.least_squares
        y: Observed values
    
    Returns:
        Array of standard errors for each parameter
    """
    ssr = np.nansum(optimised_result.fun ** 2)
    dof = y.size - 2
    mse = ssr / dof
    rmse = np.sqrt(mse)
    neg_hess = np.dot(optimised_result.jac.T, optimised_result.jac)
    try:
        inv_neg_hess = np.linalg.inv(neg_hess)
        res_lsq_params_se = np.sqrt(np.diagonal(inv_neg_hess)) * rmse
    except np.linalg.LinAlgError:
        # If matrix is singular, return NaN for standard errors
        res_lsq_params_se = np.full(optimised_result.x.shape, np.nan)
    return res_lsq_params_se


def _circa_single_core(
    t: np.ndarray,
    y: np.ndarray,
    loss: str = CircaCompareConstants.LOSS,
    f_scale: float = CircaCompareConstants.F_SCALE,
    max_iterations: int = CircaCompareConstants.MAX_ITERATIONS
) -> Optional[Any]:
    """
    Core function for fitting single-group cosinor model.
    
    Fits the model: y = mesor + amplitude * cos(t - acrophase)
    
    Args:
        t: Time values IN RADIANS
        y: Observed values
        loss: Loss function for robust regression ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan')
        f_scale: Scaling factor for loss function
        max_iterations: Maximum optimization attempts with random starting values
    
    Returns:
        OptimizeResult object with fitted parameters and confidence intervals,
        or None if optimization failed
    """
    counter = 0
    while counter < max_iterations:
        # Random starting values
        start_args = np.random.rand(1, 3)[0] * np.array([
            2 * np.median(y),
            y.max() - y.min(),
            2 * np.pi
        ])
        
        result_least_squares = least_squares(
            _fun_circa_single,
            start_args,
            loss=loss,
            f_scale=f_scale,
            args=(t, y)
        )
        
        # Check constraints: amplitude > 0 and 0 < acrophase < 2π
        if result_least_squares.x[1] > 0 and 0 < result_least_squares.x[2] < 2 * np.pi:
            break
        counter += 1

    if counter == max_iterations:
        return None

    # Calculate confidence intervals (95%)
    se = _param_standard_errors(result_least_squares, y)
    confidence_intervals = (
        result_least_squares.x + se * 1.96,
        result_least_squares.x - se * 1.96
    )
    result_least_squares.confidence_intervals = confidence_intervals
    result_least_squares.standard_errors = se

    return result_least_squares


def _circacompare_core(
    t: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    loss: str = CircaCompareConstants.LOSS,
    f_scale: float = CircaCompareConstants.F_SCALE,
    max_iterations: int = CircaCompareConstants.MAX_ITERATIONS
) -> Optional[Any]:
    """
    Core function for comparing cosinor models between two groups.
    
    Fits the model: y = (M + dM*g) + (A + dA*g) * cos(t - (φ + dφ*g))
    
    Where:
        M = mesor for group 0
        dM = difference in mesor (group 1 - group 0)
        A = amplitude for group 0
        dA = difference in amplitude
        φ = acrophase for group 0
        dφ = difference in acrophase
    
    Args:
        t: Time values IN RADIANS
        y: Observed values
        g: Group indicator (0 or 1)
        loss: Loss function for robust regression
        f_scale: Scaling factor for loss function
        max_iterations: Maximum optimization attempts
    
    Returns:
        OptimizeResult object with fitted parameters and confidence intervals,
        or None if optimization failed
    """
    counter = 0
    while counter < max_iterations:
        # Random starting values
        random_array = np.concatenate((
            np.random.rand(1, 5)[0],
            (np.random.rand(1, 1)[0] - 0.5) * 2
        ))
        
        start_args = random_array * np.array([
            2 * np.median(y[g == 0]),                              # mesor group 0
            2 * np.median(y[g == 1]),                              # d_mesor
            y[g == 0].max() - y[g == 0].min(),                     # amplitude group 0
            (y[g == 0].max() - y[g == 0].min()) - 
            (y[g == 1].max() - y[g == 1].min()),                   # d_amplitude
            2 * np.pi,                                              # acrophase group 0
            np.pi                                                   # d_acrophase
        ])
        
        result_least_squares = least_squares(
            _fun_circacompare,
            start_args,
            loss=loss,
            f_scale=f_scale,
            args=(t, y, g)
        )
        
        # Check constraints
        amp_g0 = result_least_squares.x[2]
        amp_g1 = result_least_squares.x[2] + result_least_squares.x[3]
        acr_g0 = result_least_squares.x[4]
        d_acr = result_least_squares.x[5]
        
        if (amp_g0 > 0 and amp_g1 > 0 and 
            0 < acr_g0 < 2 * np.pi and 
            -np.pi < d_acr < np.pi):
            break
        counter += 1

    if counter == max_iterations:
        return None

    # Calculate confidence intervals (95%)
    se = _param_standard_errors(result_least_squares, y)
    confidence_intervals = (
        result_least_squares.x + se * 1.96,
        result_least_squares.x - se * 1.96
    )
    result_least_squares.confidence_intervals = confidence_intervals
    result_least_squares.standard_errors = se

    return result_least_squares


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CircaSingleResult:
    """
    Data class containing results from single-group cosinor analysis.
    
    Attributes:
        mesor: Midline Estimating Statistic Of Rhythm (rhythm-adjusted mean)
        amplitude: Half the peak-to-trough difference
        acrophase: Time of peak in radians
        acrophase_hours: Time of peak converted to hours
        period: The period used for analysis
        mesor_ci: 95% confidence interval for MESOR (lower, upper)
        amplitude_ci: 95% confidence interval for amplitude
        acrophase_ci: 95% confidence interval for acrophase (in radians)
        standard_errors: Standard errors for [mesor, amplitude, acrophase]
        success: Whether the optimization was successful
    """
    mesor: float
    amplitude: float
    acrophase: float
    acrophase_hours: float
    period: float
    mesor_ci: Tuple[float, float]
    amplitude_ci: Tuple[float, float]
    acrophase_ci: Tuple[float, float]
    standard_errors: np.ndarray
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'mesor': self.mesor,
            'amplitude': self.amplitude,
            'acrophase': self.acrophase,
            'acrophase_hours': self.acrophase_hours,
            'period': self.period,
            'mesor_ci_lower': self.mesor_ci[0],
            'mesor_ci_upper': self.mesor_ci[1],
            'amplitude_ci_lower': self.amplitude_ci[0],
            'amplitude_ci_upper': self.amplitude_ci[1],
            'acrophase_ci_lower': self.acrophase_ci[0],
            'acrophase_ci_upper': self.acrophase_ci[1],
            'se_mesor': self.standard_errors[0] if self.standard_errors is not None else np.nan,
            'se_amplitude': self.standard_errors[1] if self.standard_errors is not None else np.nan,
            'se_acrophase': self.standard_errors[2] if self.standard_errors is not None else np.nan,
            'success': self.success
        }


@dataclass  
class CircaCompareResult:
    """
    Data class containing results from two-group cosinor comparison.
    
    Parameters for Group 0 (reference):
        mesor_g0, amplitude_g0, acrophase_g0
    
    Differences (Group 1 - Group 0):
        d_mesor, d_amplitude, d_acrophase
    
    Parameters for Group 1 (derived):
        mesor_g1, amplitude_g1, acrophase_g1
    
    Attributes:
        condition1: Name of reference condition (group 0)
        condition2: Name of comparison condition (group 1)
        mesor_g0: MESOR for group 0
        amplitude_g0: Amplitude for group 0
        acrophase_g0: Acrophase for group 0 (radians)
        acrophase_g0_hours: Acrophase for group 0 (hours)
        d_mesor: Difference in MESOR (g1 - g0)
        d_amplitude: Difference in amplitude (g1 - g0)
        d_acrophase: Difference in acrophase (g1 - g0, radians)
        d_acrophase_hours: Difference in acrophase (hours)
        mesor_g1: MESOR for group 1
        amplitude_g1: Amplitude for group 1
        acrophase_g1: Acrophase for group 1 (radians)
        acrophase_g1_hours: Acrophase for group 1 (hours)
        confidence_intervals: 95% CIs for all parameters
        standard_errors: Standard errors for all parameters
        period: The period used for analysis
        success: Whether the optimization was successful
    """
    condition1: str
    condition2: str
    mesor_g0: float
    amplitude_g0: float
    acrophase_g0: float
    acrophase_g0_hours: float
    d_mesor: float
    d_amplitude: float
    d_acrophase: float
    d_acrophase_hours: float
    mesor_g1: float
    amplitude_g1: float
    acrophase_g1: float
    acrophase_g1_hours: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    standard_errors: np.ndarray
    period: float
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'condition1': self.condition1,
            'condition2': self.condition2,
            'mesor_g0': self.mesor_g0,
            'amplitude_g0': self.amplitude_g0,
            'acrophase_g0': self.acrophase_g0,
            'acrophase_g0_hours': self.acrophase_g0_hours,
            'd_mesor': self.d_mesor,
            'd_amplitude': self.d_amplitude,
            'd_acrophase': self.d_acrophase,
            'd_acrophase_hours': self.d_acrophase_hours,
            'mesor_g1': self.mesor_g1,
            'amplitude_g1': self.amplitude_g1,
            'acrophase_g1': self.acrophase_g1,
            'acrophase_g1_hours': self.acrophase_g1_hours,
            'period': self.period,
            'success': self.success,
            **{f'ci_{k}': v for k, v in self.confidence_intervals.items()}
        }
    
    def is_mesor_different(self, alpha: float = 0.05) -> bool:
        """
        Check if MESOR differs significantly between groups.
        
        Uses the confidence interval: if CI for d_mesor doesn't include 0,
        the difference is significant.
        """
        ci = self.confidence_intervals.get('d_mesor', (np.nan, np.nan))
        return not (min(ci) <= 0 <= max(ci))
    
    def is_amplitude_different(self, alpha: float = 0.05) -> bool:
        """Check if amplitude differs significantly between groups."""
        ci = self.confidence_intervals.get('d_amplitude', (np.nan, np.nan))
        return not (min(ci) <= 0 <= max(ci))
    
    def is_acrophase_different(self, alpha: float = 0.05) -> bool:
        """Check if acrophase differs significantly between groups."""
        ci = self.confidence_intervals.get('d_acrophase', (np.nan, np.nan))
        return not (min(ci) <= 0 <= max(ci))


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class CircaCompareAnalyzer:
    """
    Main class for performing CircaCompare-based rhythmometry analysis.
    
    This class provides a unified interface for single-group cosinor fitting
    and two-group differential rhythmicity analysis.
    
    Attributes:
        period: Expected period of oscillation (default: 24 hours)
        loss: Loss function for robust regression
        f_scale: Scaling factor for loss function
        max_iterations: Maximum optimization attempts
    
    Example:
        >>> analyzer = CircaCompareAnalyzer(period=24)
        >>> df = analyzer.load_csv("data.csv")
        >>> result = analyzer.fit_single(variable="geneA", condition="winter")
        >>> print(result.amplitude, result.acrophase_hours)
        
        >>> comparison = analyzer.compare(variable="geneA", 
        ...                                condition1="winter", 
        ...                                condition2="summer")
        >>> print(comparison.d_amplitude, comparison.is_amplitude_different())
    """
    
    def __init__(
        self,
        period: float = 24.0,
        loss: str = CircaCompareConstants.LOSS,
        f_scale: float = CircaCompareConstants.F_SCALE,
        max_iterations: int = CircaCompareConstants.MAX_ITERATIONS
    ):
        """
        Initialize the CircaCompareAnalyzer.
        
        Args:
            period: The expected period of oscillation in hours (default: 24)
            loss: Loss function for robust regression. Options:
                  'linear' - Standard least squares
                  'soft_l1' - Smooth approximation to L1 (absolute value)
                  'huber' - Huber loss (linear for large residuals)
                  'cauchy' - Cauchy loss (strong downweighting of outliers)
                  'arctan' - Arctan loss
            f_scale: Scaling factor for the loss function. Only affects non-linear
                    loss functions. Residuals below f_scale are treated ~linearly.
            max_iterations: Maximum number of optimization attempts with random
                           starting values before declaring failure.
        """
        self.period = period
        self.loss = loss
        self.f_scale = f_scale
        self.max_iterations = max_iterations
        
        self._raw_data: Optional[pd.DataFrame] = None
        self._variables: List[str] = []
        self._conditions: List[str] = []
        self._time_col: str = "time"
        self._condition_col: str = "condition"
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_csv(
        self,
        filepath: str,
        time_column: str = "time",
        condition_column: str = "condition",
        variable_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and preprocess CSV data for CircaCompare analysis.
        
        Args:
            filepath: Path to the CSV file
            time_column: Name of the column containing time values (in hours)
            condition_column: Name of the column containing condition labels
            variable_columns: List of columns to analyze. If None, auto-detects
                            all numeric columns except time/condition.
        
        Returns:
            Loaded pandas DataFrame
        
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing
        """
        # Load raw data
        self._raw_data = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = [time_column, condition_column]
        missing = [c for c in required_cols if c not in self._raw_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Store column names
        self._time_col = time_column
        self._condition_col = condition_column
        
        # Auto-detect variable columns if not specified
        if variable_columns is None:
            exclude_cols = {time_column, condition_column, 'replicate', 'subject'}
            variable_columns = [
                col for col in self._raw_data.columns
                if col not in exclude_cols
                and pd.api.types.is_numeric_dtype(self._raw_data[col])
            ]
        
        self._variables = variable_columns
        self._conditions = self._raw_data[condition_column].unique().tolist()
        
        return self._raw_data.copy()
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        time_column: str = "time",
        condition_column: str = "condition",
        variable_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from an existing DataFrame.
        
        Args:
            df: pandas DataFrame with the data
            time_column: Name of the column containing time values
            condition_column: Name of the column containing condition labels
            variable_columns: List of columns to analyze
        
        Returns:
            Copy of the loaded DataFrame
        """
        self._raw_data = df.copy()
        
        # Validate required columns
        required_cols = [time_column, condition_column]
        missing = [c for c in required_cols if c not in self._raw_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self._time_col = time_column
        self._condition_col = condition_column
        
        # Auto-detect variable columns
        if variable_columns is None:
            exclude_cols = {time_column, condition_column, 'replicate', 'subject'}
            variable_columns = [
                col for col in self._raw_data.columns
                if col not in exclude_cols
                and pd.api.types.is_numeric_dtype(self._raw_data[col])
            ]
        
        self._variables = variable_columns
        self._conditions = self._raw_data[condition_column].unique().tolist()
        
        return self._raw_data.copy()
    
    def get_variables(self) -> List[str]:
        """Get list of available variables for analysis."""
        return self._variables.copy()
    
    def get_conditions(self) -> List[str]:
        """Get list of available conditions."""
        return self._conditions.copy()
    
    # =========================================================================
    # TIME CONVERSION
    # =========================================================================
    
    def _time_to_radians(self, time: np.ndarray) -> np.ndarray:
        """
        Convert time values to radians based on the period.
        
        Args:
            time: Time values in hours (or whatever unit matches period)
        
        Returns:
            Time values in radians (0 to 2π for one complete cycle)
        """
        return 2 * np.pi * time / self.period
    
    def _radians_to_hours(self, radians: float) -> float:
        """
        Convert acrophase from radians to hours.
        
        Args:
            radians: Acrophase in radians
        
        Returns:
            Acrophase in hours (within one period)
        """
        hours = (radians * self.period) / (2 * np.pi)
        return hours % self.period
    
    # =========================================================================
    # SINGLE GROUP ANALYSIS
    # =========================================================================
    
    def fit_single(
        self,
        variable: str,
        condition: str,
        period: Optional[float] = None
    ) -> CircaSingleResult:
        """
        Fit a single-group cosinor model.
        
        Fits the model: y = mesor + amplitude * cos(2π*t/T - acrophase)
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            period: Period to use. If None, uses self.period
        
        Returns:
            CircaSingleResult object with fitted parameters and confidence intervals
        
        Raises:
            ValueError: If variable or condition not found in data
        """
        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_csv() or load_dataframe() first.")
        
        if variable not in self._variables:
            raise ValueError(f"Variable '{variable}' not found. Available: {self._variables}")
        if condition not in self._conditions:
            raise ValueError(f"Condition '{condition}' not found. Available: {self._conditions}")
        
        period = period or self.period
        
        # Get data for this condition
        cond_data = self._raw_data[self._raw_data[self._condition_col] == condition]
        
        # Extract time and values
        t = cond_data[self._time_col].values.astype(float)
        y = cond_data[variable].values.astype(float)
        
        # Remove NaN values
        mask = ~(np.isnan(t) | np.isnan(y))
        t = t[mask]
        y = y[mask]
        
        # Convert time to radians
        t_rad = self._time_to_radians(t)
        
        # Fit the model
        result = _circa_single_core(
            t_rad, y,
            loss=self.loss,
            f_scale=self.f_scale,
            max_iterations=self.max_iterations
        )
        
        if result is None:
            # Optimization failed
            return CircaSingleResult(
                mesor=np.nan,
                amplitude=np.nan,
                acrophase=np.nan,
                acrophase_hours=np.nan,
                period=period,
                mesor_ci=(np.nan, np.nan),
                amplitude_ci=(np.nan, np.nan),
                acrophase_ci=(np.nan, np.nan),
                standard_errors=np.array([np.nan, np.nan, np.nan]),
                success=False
            )
        
        # Extract parameters
        mesor = result.x[0]
        amplitude = result.x[1]
        acrophase = result.x[2]
        acrophase_hours = self._radians_to_hours(acrophase)
        
        # Extract confidence intervals
        ci_upper, ci_lower = result.confidence_intervals
        
        return CircaSingleResult(
            mesor=float(mesor),
            amplitude=float(amplitude),
            acrophase=float(acrophase),
            acrophase_hours=float(acrophase_hours),
            period=period,
            mesor_ci=(float(ci_lower[0]), float(ci_upper[0])),
            amplitude_ci=(float(ci_lower[1]), float(ci_upper[1])),
            acrophase_ci=(float(ci_lower[2]), float(ci_upper[2])),
            standard_errors=result.standard_errors,
            success=True
        )
    
    def fit_single_all(
        self,
        condition: Optional[str] = None,
        period: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Fit single-group cosinor models to all variables.
        
        Args:
            condition: Specific condition to analyze. If None, analyzes all conditions.
            period: Period to use. If None, uses self.period
        
        Returns:
            DataFrame with results for all variable-condition combinations
        """
        results = []
        conditions = [condition] if condition else self._conditions
        
        for cond in conditions:
            for var in self._variables:
                try:
                    result = self.fit_single(var, cond, period)
                    result_dict = result.to_dict()
                    result_dict['variable'] = var
                    result_dict['condition'] = cond
                    results.append(result_dict)
                except Exception as e:
                    warnings.warn(f"Failed to fit {var}/{cond}: {e}")
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # TWO-GROUP COMPARISON (DIFFERENTIAL RHYTHMICITY)
    # =========================================================================
    
    def compare(
        self,
        variable: str,
        condition1: str,
        condition2: str,
        period: Optional[float] = None
    ) -> CircaCompareResult:
        """
        Compare rhythmicity between two conditions.
        
        Fits a joint model to estimate differences in MESOR, amplitude, and acrophase
        between two groups.
        
        Args:
            variable: Name of the variable to compare
            condition1: Reference condition (group 0)
            condition2: Comparison condition (group 1)
            period: Period to use. If None, uses self.period
        
        Returns:
            CircaCompareResult object with comparison results
        
        Raises:
            ValueError: If variable or conditions not found in data
        """
        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_csv() or load_dataframe() first.")
        
        if variable not in self._variables:
            raise ValueError(f"Variable '{variable}' not found. Available: {self._variables}")
        if condition1 not in self._conditions:
            raise ValueError(f"Condition '{condition1}' not found. Available: {self._conditions}")
        if condition2 not in self._conditions:
            raise ValueError(f"Condition '{condition2}' not found. Available: {self._conditions}")
        
        period = period or self.period
        
        # Get data for both conditions
        data1 = self._raw_data[self._raw_data[self._condition_col] == condition1]
        data2 = self._raw_data[self._raw_data[self._condition_col] == condition2]
        
        # Combine data
        t1 = data1[self._time_col].values.astype(float)
        y1 = data1[variable].values.astype(float)
        t2 = data2[self._time_col].values.astype(float)
        y2 = data2[variable].values.astype(float)
        
        # Create combined arrays
        t = np.concatenate([t1, t2])
        y = np.concatenate([y1, y2])
        g = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
        
        # Remove NaN values
        mask = ~(np.isnan(t) | np.isnan(y))
        t = t[mask]
        y = y[mask]
        g = g[mask]
        
        # Convert time to radians
        t_rad = self._time_to_radians(t)
        
        # Fit the comparison model
        result = _circacompare_core(
            t_rad, y, g,
            loss=self.loss,
            f_scale=self.f_scale,
            max_iterations=self.max_iterations
        )
        
        if result is None:
            # Optimization failed
            return CircaCompareResult(
                condition1=condition1,
                condition2=condition2,
                mesor_g0=np.nan,
                amplitude_g0=np.nan,
                acrophase_g0=np.nan,
                acrophase_g0_hours=np.nan,
                d_mesor=np.nan,
                d_amplitude=np.nan,
                d_acrophase=np.nan,
                d_acrophase_hours=np.nan,
                mesor_g1=np.nan,
                amplitude_g1=np.nan,
                acrophase_g1=np.nan,
                acrophase_g1_hours=np.nan,
                confidence_intervals={},
                standard_errors=np.array([np.nan] * 6),
                period=period,
                success=False
            )
        
        # Extract parameters
        # x = [mesor_g0, d_mesor, amp_g0, d_amp, acr_g0, d_acr]
        mesor_g0 = result.x[0]
        d_mesor = result.x[1]
        amplitude_g0 = result.x[2]
        d_amplitude = result.x[3]
        acrophase_g0 = result.x[4]
        d_acrophase = result.x[5]
        
        # Derive group 1 parameters
        mesor_g1 = mesor_g0 + d_mesor
        amplitude_g1 = amplitude_g0 + d_amplitude
        acrophase_g1 = acrophase_g0 + d_acrophase
        
        # Convert acrophases to hours
        acrophase_g0_hours = self._radians_to_hours(acrophase_g0)
        acrophase_g1_hours = self._radians_to_hours(acrophase_g1)
        d_acrophase_hours = d_acrophase * self.period / (2 * np.pi)
        
        # Extract confidence intervals
        ci_upper, ci_lower = result.confidence_intervals
        
        confidence_intervals = {
            'mesor_g0': (float(ci_lower[0]), float(ci_upper[0])),
            'd_mesor': (float(ci_lower[1]), float(ci_upper[1])),
            'amplitude_g0': (float(ci_lower[2]), float(ci_upper[2])),
            'd_amplitude': (float(ci_lower[3]), float(ci_upper[3])),
            'acrophase_g0': (float(ci_lower[4]), float(ci_upper[4])),
            'd_acrophase': (float(ci_lower[5]), float(ci_upper[5]))
        }
        
        return CircaCompareResult(
            condition1=condition1,
            condition2=condition2,
            mesor_g0=float(mesor_g0),
            amplitude_g0=float(amplitude_g0),
            acrophase_g0=float(acrophase_g0),
            acrophase_g0_hours=float(acrophase_g0_hours),
            d_mesor=float(d_mesor),
            d_amplitude=float(d_amplitude),
            d_acrophase=float(d_acrophase),
            d_acrophase_hours=float(d_acrophase_hours),
            mesor_g1=float(mesor_g1),
            amplitude_g1=float(amplitude_g1),
            acrophase_g1=float(acrophase_g1),
            acrophase_g1_hours=float(acrophase_g1_hours),
            confidence_intervals=confidence_intervals,
            standard_errors=result.standard_errors,
            period=period,
            success=True
        )
    
    def compare_all_conditions(
        self,
        variable: str,
        period: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compare all pairs of conditions for a given variable.
        
        Args:
            variable: Name of the variable to compare
            period: Period to use. If None, uses self.period
        
        Returns:
            DataFrame with all pairwise comparisons
        """
        from itertools import combinations
        
        results = []
        for cond1, cond2 in combinations(self._conditions, 2):
            try:
                result = self.compare(variable, cond1, cond2, period)
                results.append(result.to_dict())
            except Exception as e:
                warnings.warn(f"Failed to compare {cond1} vs {cond2}: {e}")
        
        return pd.DataFrame(results)
    
    def compare_all_variables(
        self,
        condition1: str,
        condition2: str,
        period: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Compare two conditions across all variables.
        
        Args:
            condition1: Reference condition
            condition2: Comparison condition
            period: Period to use. If None, uses self.period
        
        Returns:
            DataFrame with comparison results for all variables
        """
        results = []
        for var in self._variables:
            try:
                result = self.compare(var, condition1, condition2, period)
                result_dict = result.to_dict()
                result_dict['variable'] = var
                results.append(result_dict)
            except Exception as e:
                warnings.warn(f"Failed to compare {var}: {e}")
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # PREDICTION / FITTED VALUES
    # =========================================================================
    
    def predict(
        self,
        result: CircaSingleResult,
        time_points: Optional[np.ndarray] = None,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predicted values from a fitted model.
        
        Useful for plotting the fitted curve.
        
        Args:
            result: CircaSingleResult from fit_single()
            time_points: Specific time points for prediction (in hours).
                        If None, generates n_points evenly spaced across one period.
            n_points: Number of points to generate if time_points is None
        
        Returns:
            Tuple of (time_hours, predicted_values)
        """
        if time_points is None:
            time_points = np.linspace(0, result.period, n_points)
        
        t_rad = self._time_to_radians(time_points)
        y_pred = result.mesor + result.amplitude * np.cos(t_rad - result.acrophase)
        
        return time_points, y_pred
    
    def predict_compare(
        self,
        result: CircaCompareResult,
        time_points: Optional[np.ndarray] = None,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predicted values for both groups from a comparison model.
        
        Args:
            result: CircaCompareResult from compare()
            time_points: Specific time points for prediction (in hours)
            n_points: Number of points to generate if time_points is None
        
        Returns:
            Tuple of (time_hours, predicted_group0, predicted_group1)
        """
        if time_points is None:
            time_points = np.linspace(0, result.period, n_points)
        
        t_rad = self._time_to_radians(time_points)
        
        y_pred_g0 = result.mesor_g0 + result.amplitude_g0 * np.cos(t_rad - result.acrophase_g0)
        y_pred_g1 = result.mesor_g1 + result.amplitude_g1 * np.cos(t_rad - result.acrophase_g1)
        
        return time_points, y_pred_g0, y_pred_g1
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    def set_period(self, period: float) -> None:
        """Update the analysis period."""
        self.period = period
    
    def set_loss(self, loss: str) -> None:
        """
        Update the loss function.
        
        Args:
            loss: Loss function ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan')
        """
        valid_losses = ['linear', 'soft_l1', 'huber', 'cauchy', 'arctan']
        if loss not in valid_losses:
            raise ValueError(f"Invalid loss function. Must be one of: {valid_losses}")
        self.loss = loss
    
    def set_f_scale(self, f_scale: float) -> None:
        """Update the f_scale parameter for loss function."""
        self.f_scale = f_scale
    
    def set_max_iterations(self, max_iterations: int) -> None:
        """Update the maximum number of optimization attempts."""
        self.max_iterations = max_iterations
    
    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Get the raw loaded data."""
        return self._raw_data.copy() if self._raw_data is not None else None


# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def generate_sample_data(
    n_timepoints: int = 6,
    n_conditions: int = 2,
    n_variables: int = 2,
    n_replicates: int = 3,
    period: float = 24.0,
    noise_level: float = 0.1
) -> pd.DataFrame:
    """
    Generate sample data for testing the CircaCompareAnalyzer.
    
    Args:
        n_timepoints: Number of timepoints per cycle
        n_conditions: Number of conditions
        n_variables: Number of variables to generate
        n_replicates: Number of replicates per timepoint
        period: Period of oscillation in hours
        noise_level: Standard deviation of noise (as fraction of amplitude)
    
    Returns:
        DataFrame with synthetic rhythmic data
    """
    condition_names = [f"condition{i+1}" for i in range(n_conditions)]
    variable_names = [f"var{i+1}" for i in range(n_variables)]
    
    timepoints = np.linspace(0, period - period/n_timepoints, n_timepoints)
    
    rows = []
    np.random.seed(42)
    
    for cond_idx, condition in enumerate(condition_names):
        for var_idx, variable in enumerate(variable_names):
            # Different parameters for each combination
            mesor = 10 + var_idx * 5
            amplitude = 3 + var_idx + cond_idx * 0.5
            acrophase = np.pi/4 + cond_idx * np.pi/6
            
            for t in timepoints:
                for rep in range(n_replicates):
                    t_rad = 2 * np.pi * t / period
                    value = mesor + amplitude * np.cos(t_rad - acrophase)
                    value += np.random.normal(0, amplitude * noise_level)
                    
                    rows.append({
                        'time': t,
                        'condition': condition,
                        'replicate': rep + 1,
                        variable: value
                    })
    
    df = pd.DataFrame(rows)
    
    # Consolidate variable columns
    if n_variables > 1:
        df = df.groupby(['time', 'condition', 'replicate']).first().reset_index()
    
    return df


# =============================================================================
# MAIN - TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("CircaCompare Analysis Module - Test Suite")
    print("=" * 60)
    
    # Generate sample data
    print("\n--- Generating Sample Data ---")
    df_sample = generate_sample_data(
        n_timepoints=6,
        n_conditions=2,
        n_variables=2,
        n_replicates=3,
        period=24.0,
        noise_level=0.15
    )
    
    csv_path = "/tmp/test_circacompare_data.csv"
    df_sample.to_csv(csv_path, index=False)
    print(f"Sample data saved to: {csv_path}")
    print(f"Data shape: {df_sample.shape}")
    print(f"\nSample data head:\n{df_sample.head(10)}")
    
    # Test the analyzer
    print("\n--- Testing CircaCompareAnalyzer ---")
    
    try:
        # Initialize analyzer
        analyzer = CircaCompareAnalyzer(period=24.0)
        print("✓ Analyzer initialized")
        print(f"  Loss function: {analyzer.loss}")
        print(f"  F-scale: {analyzer.f_scale}")
        print(f"  Max iterations: {analyzer.max_iterations}")
        
        # Load data
        df = analyzer.load_csv(csv_path)
        print("\n✓ Data loaded successfully")
        print(f"  Variables detected: {analyzer.get_variables()}")
        print(f"  Conditions detected: {analyzer.get_conditions()}")
        
        # Test single cosinor fit
        print("\n--- Single-Group Cosinor Fit ---")
        result = analyzer.fit_single('var1', 'condition1')
        print(f"Variable: var1, Condition: condition1")
        print(f"  Success: {result.success}")
        print(f"  MESOR: {result.mesor:.3f}")
        print(f"  Amplitude: {result.amplitude:.3f}")
        print(f"  Acrophase (hours): {result.acrophase_hours:.2f}")
        print(f"  Amplitude 95% CI: ({result.amplitude_ci[0]:.3f}, {result.amplitude_ci[1]:.3f})")
        
        # Test all variables
        print("\n--- All Variables Analysis ---")
        df_results = analyzer.fit_single_all()
        print(df_results[['variable', 'condition', 'amplitude', 'acrophase_hours', 'success']])
        
        # Test comparison
        print("\n--- Differential Rhythmicity (CircaCompare) ---")
        comparison = analyzer.compare('var1', 'condition1', 'condition2')
        print(f"Comparing: condition1 vs condition2 (var1)")
        print(f"  Success: {comparison.success}")
        print(f"  Group 0 - MESOR: {comparison.mesor_g0:.3f}, Amplitude: {comparison.amplitude_g0:.3f}")
        print(f"  Group 1 - MESOR: {comparison.mesor_g1:.3f}, Amplitude: {comparison.amplitude_g1:.3f}")
        print(f"  Difference in amplitude: {comparison.d_amplitude:.3f}")
        print(f"    CI: ({comparison.confidence_intervals['d_amplitude'][0]:.3f}, "
              f"{comparison.confidence_intervals['d_amplitude'][1]:.3f})")
        print(f"    Significant: {comparison.is_amplitude_different()}")
        print(f"  Difference in acrophase: {comparison.d_acrophase_hours:.2f} hours")
        print(f"    Significant: {comparison.is_acrophase_different()}")
        
        # Test prediction
        print("\n--- Prediction ---")
        time_points, y_pred = analyzer.predict(result, n_points=5)
        print(f"Predictions at times {time_points[:5]}:")
        print(f"  Values: {y_pred[:5]}")
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print usage example
    print("\n--- Usage Example for GUI Integration ---")
    print("""
# Basic usage:
from circacompare_analysis import CircaCompareAnalyzer

# Initialize with custom parameters
analyzer = CircaCompareAnalyzer(
    period=24.0,
    loss='linear',      # or 'huber', 'cauchy' for robust fitting
    f_scale=1.0,
    max_iterations=500
)

# Load CSV
df = analyzer.load_csv(
    "your_data.csv",
    time_column="time",
    condition_column="condition"
)

# Get available variables and conditions
variables = analyzer.get_variables()
conditions = analyzer.get_conditions()

# Fit single group
result = analyzer.fit_single("geneA", "winter")
print(result.amplitude, result.acrophase_hours)

# Compare two conditions
comparison = analyzer.compare("geneA", "winter", "summer")
print(comparison.d_amplitude, comparison.is_amplitude_different())

# Get predictions for plotting
time_points, y_pred_g0, y_pred_g1 = analyzer.predict_compare(comparison)
""")
