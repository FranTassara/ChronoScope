"""
Rhythm Analysis Module
======================

A comprehensive module for circadian rhythm analysis using multiple methods.
Designed to be compatible with the CosinorPy and CircaCompare modules for GUI integration.

This module provides:
- Python-JTK Cycle Analysis (nonparametric, triangle templates)
- AR-JTK (JTK with autoregressive noise handling)
- Cosine-Kendall (nonparametric cosine template correlation)
- Standard Cosinor Analysis (parametric, OLS-based with period optimization)
- Harmonic Cosinor Analysis (multi-harmonic detection)
- Fourier Analysis with F24 Score (Wijnen et al., 2006 method)
- Lomb-Scargle Periodogram (for unevenly sampled data)
- Continuous Wavelet Transform (CWT) Analysis
- Linear Mixed Effects (LME) Model

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
    pip install numpy scipy pandas statsmodels pywt

Author: Francisco Tassara
Version: 1.0.0
"""

from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from scipy.signal import find_peaks, lombscargle
from scipy.optimize import least_squares
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

# Optional PyWavelets import
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn(
        "PyWavelets (pywt) is not installed. CWT analysis will not be available. "
        "Install with: pip install PyWavelets"
    )


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class AnalysisMethod(Enum):
    """Enumeration for available analysis methods."""
    PYTHON_JTK = "Python-JTK"
    AR_JTK = "AR-JTK"
    COSINE_KENDALL = "Cosine-Kendall"
    COSINOR = "Cosinor"
    HARMONIC_COSINOR = "Harmonic-Cosinor"
    FOURIER_F24 = "Fourier-F24"
    LOMB_SCARGLE = "Lomb-Scargle"
    SPECTRAL_ANALYSIS = "Spectral-Analysis"
    CWT = "CWT"
    LME = "LME"


class DefaultPeriodRanges:
    """Default period ranges for different analysis types."""
    CIRCADIAN = [20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27, 27.5, 28]
    CIRCADIAN_INT = list(range(20, 29))  # For discrete JTK
    ULTRADIAN = list(range(4, 13))
    SINGLE = [24.0]


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class JTKResult:
    """
    Data class containing JTK Cycle analysis results.
    
    Attributes:
        p_value: Raw p-value from Kendall tau test
        adj_p_value: Bonferroni-adjusted p-value
        period: Best-fit period (hours)
        amplitude: Estimated amplitude (90th - 10th percentile / 2)
        acrophase: Peak time in hours
        asymmetry: Waveform asymmetry parameter
        tau: Kendall tau correlation coefficient
        lag: Phase lag parameter
        method: Analysis method identifier
    """
    p_value: float
    adj_p_value: float
    period: float
    amplitude: float
    acrophase: float
    asymmetry: float
    tau: float
    lag: float
    method: str
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the rhythm is statistically significant."""
        return self.adj_p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'p_value': self.p_value,
            'adj_p_value': self.adj_p_value,
            'period': self.period,
            'amplitude': self.amplitude,
            'acrophase': self.acrophase,
            'asymmetry': self.asymmetry,
            'tau': self.tau,
            'lag': self.lag,
            'method': self.method
        }


@dataclass
class CosinorResult:
    """
    Data class containing Cosinor analysis results.
    
    Attributes:
        mesor: Midline Estimating Statistic Of Rhythm
        amplitude: Rhythm amplitude
        acrophase_rad: Acrophase in radians
        acrophase_hours: Acrophase in hours
        period: Best-fit period
        p_value: F-test p-value
        adj_p_value: Bonferroni-adjusted p-value
        amplitude_ci: 95% CI for amplitude
        acrophase_ci: 95% CI for acrophase (radians)
        amplitude_p: P-value for amplitude significance
        acrophase_p: P-value for acrophase significance
        method: Analysis method identifier
    """
    mesor: float
    amplitude: float
    acrophase_rad: float
    acrophase_hours: float
    period: float
    p_value: float
    adj_p_value: float
    amplitude_ci: Tuple[float, float]
    acrophase_ci: Tuple[float, float]
    amplitude_p: float
    acrophase_p: float
    method: str = "Cosinor"
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the rhythm is statistically significant."""
        return self.adj_p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'mesor': self.mesor,
            'amplitude': self.amplitude,
            'acrophase_rad': self.acrophase_rad,
            'acrophase_hours': self.acrophase_hours,
            'period': self.period,
            'p_value': self.p_value,
            'adj_p_value': self.adj_p_value,
            'amplitude_ci_lower': self.amplitude_ci[0],
            'amplitude_ci_upper': self.amplitude_ci[1],
            'acrophase_ci_lower': self.acrophase_ci[0],
            'acrophase_ci_upper': self.acrophase_ci[1],
            'amplitude_p': self.amplitude_p,
            'acrophase_p': self.acrophase_p,
            'method': self.method
        }


@dataclass
class HarmonicCosinorResult:
    """
    Data class containing Harmonic Cosinor analysis results.
    
    Supports detection of multi-modal rhythms (e.g., bimodal with 2 peaks per cycle).
    
    Attributes:
        adj_p_value: Bonferroni-adjusted p-value
        period: Best-fit period
        amplitudes: List of amplitudes for each peak
        acrophases: List of acrophase times (hours) for each peak
        n_harmonics: Number of harmonics used
        method: Analysis method identifier
        fit_model: Dictionary with fitted model parameters for plotting
    """
    adj_p_value: float
    period: float
    amplitudes: List[float]
    acrophases: List[float]
    n_harmonics: int
    method: str = "Harmonic-Cosinor"
    fit_model: Optional[Dict[str, Any]] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the rhythm is statistically significant."""
        return self.adj_p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result = {
            'adj_p_value': self.adj_p_value,
            'period': self.period,
            'n_harmonics': self.n_harmonics,
            'method': self.method
        }
        for i, (amp, acro) in enumerate(zip(self.amplitudes, self.acrophases), 1):
            result[f'amplitude_{i}'] = amp
            result[f'acrophase_{i}'] = acro
        return result


@dataclass
class FourierF24Result:
    """
    Data class containing Fourier F24 analysis results (Wijnen et al., 2006 method).
    
    The F24 score represents the power at the 24-hour frequency relative to noise.
    This is an effect size measure, not a significance test.
    
    Attributes:
        f24_score: F24 rhythmicity index (power at target period / mean random power)
        power_spectrum: Full power spectrum
        frequencies: Corresponding frequencies
        dominant_period: Period with highest power
        dominant_power: Power at dominant period
        target_period: The period that was tested (e.g., 24h)
        target_power: Power at the target period
        correlation_r: Correlation between replicates (if applicable)
        method: Analysis method identifier
    """
    f24_score: float
    power_spectrum: np.ndarray
    frequencies: np.ndarray
    dominant_period: float
    dominant_power: float
    target_period: float
    target_power: float
    correlation_r: Optional[float] = None
    method: str = "Fourier-F24"
    
    def is_rhythmic(self, threshold: float = 2.0) -> bool:
        """
        Check if F24 score exceeds threshold (effect size filter).
        
        Note: This is NOT a significance test. F24 is an effect size measure.
        Wijnen et al. used F24 > 2 or F24 > 3 as filters.
        """
        return self.f24_score > threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'f24_score': self.f24_score,
            'dominant_period': self.dominant_period,
            'dominant_power': self.dominant_power,
            'target_period': self.target_period,
            'target_power': self.target_power,
            'correlation_r': self.correlation_r,
            'method': self.method
        }


@dataclass
class LombScargleResult:
    """
    Data class containing Lomb-Scargle periodogram results.
    
    Attributes:
        dominant_period: Period with highest power
        dominant_power: Power at dominant period
        false_alarm_probability: FAP for dominant peak
        periods: Array of periods tested
        power_spectrum: Power at each period
        method: Analysis method identifier
    """
    dominant_period: float
    dominant_power: float
    false_alarm_probability: Optional[float]
    periods: np.ndarray
    power_spectrum: np.ndarray
    method: str = "Lomb-Scargle"
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if dominant period is significant based on FAP."""
        if self.false_alarm_probability is None:
            return False
        return self.false_alarm_probability < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'dominant_period': self.dominant_period,
            'dominant_power': self.dominant_power,
            'false_alarm_probability': self.false_alarm_probability,
            'method': self.method
        }


@dataclass
class CWTResult:
    """
    Data class containing Continuous Wavelet Transform analysis results.
    
    Attributes:
        dominant_period: Period with highest average power
        mean_power: Mean power across the dominant period
        period_variation: Standard deviation of dominant period over time
        amplitude_modulations: Number of amplitude peaks detected
        method: Analysis method identifier
    """
    dominant_period: float
    mean_power: float
    period_variation: float
    amplitude_modulations: int
    method: str = "CWT"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'dominant_period': self.dominant_period,
            'mean_power': self.mean_power,
            'period_variation': self.period_variation,
            'amplitude_modulations': self.amplitude_modulations,
            'method': self.method
        }


@dataclass
class LMEResult:
    """
    Data class containing Linear Mixed Effects model results.
    
    Attributes:
        terms: List of model terms
        estimates: Parameter estimates
        std_errors: Standard errors
        z_values: Z-statistics
        p_values: P-values for each term
        method: Analysis method identifier
    """
    terms: List[str]
    estimates: List[float]
    std_errors: List[float]
    z_values: List[float]
    p_values: List[float]
    method: str = "LME"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'terms': self.terms,
            'estimates': self.estimates,
            'std_errors': self.std_errors,
            'z_values': self.z_values,
            'p_values': self.p_values,
            'method': self.method
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame({
            'Term': self.terms,
            'Estimate': self.estimates,
            'StdErr': self.std_errors,
            'z-value': self.z_values,
            'p-value': self.p_values
        })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def acrophase_to_hours(rad_phase: float, period: float = 24.0) -> float:
    """
    Convert acrophase from radians to hours.
    
    Args:
        rad_phase: Acrophase in radians
        period: Period in hours
    
    Returns:
        Acrophase in hours (0 to period)
    """
    hours = (rad_phase * period) / (2 * np.pi)
    return hours % period


def hours_to_radians(hours: float, period: float = 24.0) -> float:
    """
    Convert time in hours to radians.
    
    Args:
        hours: Time in hours
        period: Period in hours
    
    Returns:
        Time in radians
    """
    return 2 * np.pi * hours / period


# =============================================================================
# JTK CYCLE FUNCTIONS
# =============================================================================

def _generate_triangle_template(length: int, peak_index: int) -> np.ndarray:
    """
    Create triangle waveform of specified length and peak location.
    
    Args:
        length: Length of the template
        peak_index: Index of the peak
    
    Returns:
        Triangle waveform array
    """
    template = np.zeros(length)
    if peak_index > 0:
        template[:peak_index] = np.linspace(1, peak_index, peak_index)
    if length - peak_index > 0:
        template[peak_index:] = np.linspace(length - peak_index, 1, length - peak_index)
    return template


def _generate_triangle_template_time(
    times: np.ndarray, 
    period: float, 
    lag: float, 
    asymmetry: float = 0.5
) -> np.ndarray:
    """
    Generate a triangle template aligned to real timepoints.
    
    Args:
        times: Array of time values
        period: Desired period in same units as time
        lag: Phase shift in time units (e.g., hours)
        asymmetry: Float between 0 and 1 indicating peak position in the cycle
    
    Returns:
        Ranked triangle template array
    """
    peak_time = asymmetry * period
    template = np.zeros_like(times, dtype=float)
    
    for i, t in enumerate(times):
        t_mod = (t - lag) % period
        if t_mod <= peak_time:
            template[i] = t_mod / peak_time if peak_time != 0 else 1.0
        else:
            template[i] = (period - t_mod) / (period - peak_time) if period != peak_time else 0.0
    
    return pd.Series(template).rank().values


def _run_discrete_jtk(
    series: pd.Series,
    period_range: List[float] = None,
    lag_range: Optional[np.ndarray] = None,
    asymmetries: List[float] = None
) -> JTKResult:
    """
    Run JTK using triangle templates aligned to actual timepoints.
    
    Args:
        series: pandas Series with time as index and values as data
        period_range: List of periods to test
        lag_range: Array of lag values to test (default: 0 to period)
        asymmetries: List of asymmetry values to test
    
    Returns:
        JTKResult object with best-fit parameters
    """
    if period_range is None:
        period_range = DefaultPeriodRanges.CIRCADIAN_INT
    if asymmetries is None:
        asymmetries = [0.5]
    
    times = series.index.to_numpy()
    y = series.rank().values
    n = len(y)

    best_p = 1.0
    best_tau = 0.0
    best_per = None
    best_lag = None
    best_asym = None

    test_results = []

    for period in period_range:
        lags = lag_range if lag_range is not None else np.arange(0, period, 1)
        for asym in asymmetries:
            for lag in lags:
                ref = _generate_triangle_template_time(times, period, lag, asym)
                tau, pval = kendalltau(y, ref)
                test_results.append((pval, tau, period, lag, asym))
                if pval < best_p:
                    best_p = pval
                    best_tau = tau
                    best_per = period
                    best_lag = lag
                    best_asym = asym

    bonf_p = min(1.0, best_p * len(test_results))
    amp = (np.percentile(series.values, 90) - np.percentile(series.values, 10)) / 2

    # Correct acrophase based on tau sign
    acrophase = (
        (best_lag + best_asym * best_per + best_per / 2) % best_per if best_tau < 0
        else (best_lag + best_asym * best_per) % best_per
    )

    return JTKResult(
        p_value=round(best_p, 6),
        adj_p_value=round(bonf_p, 6),
        period=round(best_per, 2),
        amplitude=round(amp, 4),
        acrophase=round(acrophase, 2),
        asymmetry=round(best_asym, 2),
        tau=round(best_tau, 2),
        lag=round(best_lag, 2),
        method='Python-JTK'
    )


# =============================================================================
# AR-JTK NOISE HANDLING FUNCTIONS
# =============================================================================

def _is_white_noise_ranked(residuals: pd.Series, lags: int = 10) -> bool:
    """
    Test if residuals are white noise using Ljung-Box test.
    
    Args:
        residuals: Residual series
        lags: Number of lags for the test
    
    Returns:
        True if white noise (p > 0.05), False otherwise
    """
    residuals = residuals.dropna()
    if len(residuals) < lags + 1:
        return True
    try:
        lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        p_value = lb_test['lb_pvalue'].iloc[0]
        return p_value > 0.05
    except Exception:
        return True


def _prewhiten_ranked_residuals(residuals: pd.Series, maxlag: int = 1) -> pd.Series:
    """
    Fit AR model to rank residuals and return whitened residuals.
    
    Args:
        residuals: Residual series (rank-domain noise)
        maxlag: Maximum AR lag
    
    Returns:
        Prewhitened residuals
    """
    e = residuals.dropna()
    if len(e) < maxlag + 2:
        return residuals

    try:
        model = AutoReg(e, lags=maxlag, old_names=False).fit()
        phi = model.params.values

        e_pw = e.copy()
        for i in range(maxlag, len(e)):
            e_pw.iloc[i] = e.iloc[i] - np.dot(phi[1:], e.iloc[i-maxlag:i][::-1])

        e_pw = pd.Series(e_pw, index=e.index)
        return e_pw

    except Exception as ex:
        warnings.warn(f"AR prewhitening failed: {ex}")
        return residuals


def _run_ar_jtk(
    series: pd.Series,
    period_range: List[float] = None,
    lag_range: Optional[np.ndarray] = None,
    asymmetries: List[float] = None,
    ar_lag: int = 1,
    ljungbox_lag: int = 10
) -> Tuple[JTKResult, bool]:
    """
    Run JTK with autoregressive noise handling.
    
    Args:
        series: pandas Series with time as index
        period_range: List of periods to test
        lag_range: Array of lag values
        asymmetries: List of asymmetry values
        ar_lag: AR model lag for prewhitening
        ljungbox_lag: Lag for Ljung-Box test
    
    Returns:
        Tuple of (JTKResult, ar_applied_flag)
    """
    if period_range is None:
        period_range = DefaultPeriodRanges.CIRCADIAN_INT
    if asymmetries is None:
        asymmetries = [0.5]
    
    # Initial JTK
    temp_res = _run_discrete_jtk(series, period_range=period_range,
                                  lag_range=lag_range, asymmetries=asymmetries)
    
    if temp_res.period is None:
        return temp_res, False

    times = series.index.to_numpy()

    # Best template for that PER/LAG/ASYM
    template_vals = _generate_triangle_template_time(
        times, temp_res.period, temp_res.lag, temp_res.asymmetry
    )
    template_rank = pd.Series(template_vals, index=series.index).rank()

    # Rank series & compute rank-residuals
    r = series.rank()
    e = r - template_rank

    # Autocorrelation test on rank-residuals
    if _is_white_noise_ranked(e, lags=ljungbox_lag):
        return temp_res, False  # No AR detected

    # AR prewhiten residuals only
    e_pw = _prewhiten_ranked_residuals(e, maxlag=ar_lag)

    # Reconstruct whitened rank-series
    r_pw = template_rank + e_pw

    # JTK on reconstructed prewhitened series
    jtk_res = _run_discrete_jtk(r_pw, period_range=period_range,
                                 lag_range=lag_range, asymmetries=asymmetries)
    
    # Update method and recalculate amplitude from original data
    amp = (np.percentile(series.values, 90) - np.percentile(series.values, 10)) / 2
    
    return JTKResult(
        p_value=jtk_res.p_value,
        adj_p_value=jtk_res.adj_p_value,
        period=jtk_res.period,
        amplitude=round(amp, 4),
        acrophase=jtk_res.acrophase,
        asymmetry=jtk_res.asymmetry,
        tau=jtk_res.tau,
        lag=jtk_res.lag,
        method='AR-JTK'
    ), True


# =============================================================================
# COSINE-KENDALL FUNCTION
# =============================================================================

def _run_cosine_kendall(
    series: pd.Series,
    period_range: List[float] = None,
    interval: float = 1.0
) -> JTKResult:
    """
    Run Cosine-Kendall nonparametric rhythm detection.
    
    Uses cosine templates instead of triangle templates.
    
    Args:
        series: pandas Series with time as index
        period_range: List of periods to test
        interval: Time interval between samples (used if times are indices)
    
    Returns:
        JTKResult object
    """
    if period_range is None:
        period_range = DefaultPeriodRanges.CIRCADIAN
    
    y = series.rank().values
    n = len(y)
    
    # Use actual time values from index if available
    t = series.index.to_numpy().astype(float)
    
    best_p = 1.0
    best_tau = 0.0
    best_per = period_range[0] if period_range else 24.0  # Default value
    best_lag = 0.0

    test_results = []
    
    for period in period_range:
        for lag in np.arange(0, period, 0.5):
            radians = 2 * np.pi * (t - lag) / period
            ref = np.cos(radians)
            ref_ranked = pd.Series(ref).rank().values
            tau, pval = kendalltau(y, ref_ranked)
            test_results.append((pval, tau, period, lag))
            if pval < best_p:
                best_p = pval
                best_tau = tau
                best_per = period
                best_lag = lag

    bonf_p = min(1.0, best_p * len(test_results)) if test_results else 1.0
    amp = (np.percentile(series.values, 90) - np.percentile(series.values, 10)) / 2
    
    # Correct lag when tau < 0 (cosine has opposite correlation)
    corrected_lag = (best_lag + best_per / 2) % best_per if best_tau < 0 else best_lag

    return JTKResult(
        p_value=round(best_p, 6),
        adj_p_value=round(bonf_p, 6),
        period=round(best_per, 2),
        amplitude=round(amp, 4),
        acrophase=round(corrected_lag, 2),
        asymmetry=0.5,  # Cosine is symmetric
        tau=round(best_tau, 2),
        lag=round(best_lag, 2),
        method='Cosine-Kendall'
    )


# =============================================================================
# COSINOR ANALYSIS FUNCTION
# =============================================================================

def _fit_cosinor(
    times: np.ndarray,
    values: np.ndarray,
    period_range: List[float] = None
) -> CosinorResult:
    """
    Fit cosinor model with period optimization using OLS.
    
    Model: y = mesor + amplitude * cos(2π*t/T - acrophase)
         = mesor + beta_cos * cos(2π*t/T) + beta_sin * sin(2π*t/T)
    
    Args:
        times: Time values
        values: Measurement values
        period_range: List of periods to test
    
    Returns:
        CosinorResult object with best-fit parameters
    """
    if period_range is None:
        period_range = DefaultPeriodRanges.CIRCADIAN
    
    x = times
    y = values
    
    test_results = []
    best_aic = np.inf
    best_result = None
    best_p = 1.0

    for per in period_range:
        omega = 2 * np.pi / per
        cos_term = np.cos(omega * x)
        sin_term = np.sin(omega * x)
        X = np.column_stack([np.ones(len(x)), cos_term, sin_term])
        model = sm.OLS(y, X).fit()
        
        pval = model.f_pvalue
        test_results.append(pval)

        if model.aic < best_aic:
            beta_cos, beta_sin = model.params[1], model.params[2]
            amp = np.sqrt(beta_cos ** 2 + beta_sin ** 2)
            # Phase calculation: y = M + A*cos(ωt - φ) = M + A*cos(φ)*cos(ωt) + A*sin(φ)*sin(ωt)
            # So β_cos = A*cos(φ), β_sin = A*sin(φ)
            # Therefore φ = arctan2(β_sin, β_cos)
            phase = np.arctan2(beta_sin, beta_cos)
            # Ensure phase is in [0, 2π)
            if phase < 0:
                phase += 2 * np.pi

            cov = model.cov_params()
            
            # Amplitude CI using delta method
            var_amp = (beta_cos**2 * cov[2, 2] +
                       beta_sin**2 * cov[1, 1] +
                       2 * beta_cos * beta_sin * cov[1, 2]) / (amp**2 + 1e-10)
            se_amp = np.sqrt(max(var_amp, 0))
            ci_amp = (amp - 1.96 * se_amp, amp + 1.96 * se_amp)

            # Acrophase CI
            var_phase = ((beta_sin**2 * cov[1, 1] +
                          beta_cos**2 * cov[2, 2] -
                          2 * beta_cos * beta_sin * cov[1, 2]) /
                         ((beta_cos**2 + beta_sin**2)**2 + 1e-10))
            se_phase = np.sqrt(max(var_phase, 0))
            ci_phase = (phase - 1.96 * se_phase, phase + 1.96 * se_phase)

            best_aic = model.aic
            best_p = pval
            best_result = {
                'period': per,
                'p_value': pval,
                'mesor': model.params[0],
                'amplitude': amp,
                'amplitude_p': model.pvalues[1],
                'amplitude_ci': ci_amp,
                'acrophase_rad': phase,
                'acrophase_p': model.pvalues[2],
                'acrophase_ci': ci_phase,
                'acrophase_hours': acrophase_to_hours(phase, per)
            }

    if best_result is None:
        return CosinorResult(
            mesor=np.nan, amplitude=np.nan, acrophase_rad=np.nan,
            acrophase_hours=np.nan, period=np.nan, p_value=np.nan,
            adj_p_value=np.nan, amplitude_ci=(np.nan, np.nan),
            acrophase_ci=(np.nan, np.nan), amplitude_p=np.nan,
            acrophase_p=np.nan, method='Cosinor'
        )

    m = len(test_results)
    bonf_p = min(1.0, best_p * m)

    return CosinorResult(
        mesor=round(best_result['mesor'], 4),
        amplitude=round(best_result['amplitude'], 4),
        acrophase_rad=round(best_result['acrophase_rad'], 4),
        acrophase_hours=round(best_result['acrophase_hours'], 2),
        period=round(best_result['period'], 2),
        p_value=round(best_result['p_value'], 6),
        adj_p_value=round(bonf_p, 6),
        amplitude_ci=(round(best_result['amplitude_ci'][0], 4), 
                      round(best_result['amplitude_ci'][1], 4)),
        acrophase_ci=(round(best_result['acrophase_ci'][0], 4),
                      round(best_result['acrophase_ci'][1], 4)),
        amplitude_p=round(best_result['amplitude_p'], 6),
        acrophase_p=round(best_result['acrophase_p'], 6),
        method='Cosinor'
    )


# =============================================================================
# HARMONIC COSINOR FUNCTION
# =============================================================================

def _fit_harmonic_cosinor(
    times: np.ndarray,
    values: np.ndarray,
    period_range: List[float] = None,
    n_harmonics: int = 2
) -> HarmonicCosinorResult:
    """
    Fit harmonic cosinor model for multi-modal rhythm detection.
    
    Uses Kendall's tau test with multi-harmonic templates.
    
    Args:
        times: Time values
        values: Measurement values
        period_range: List of periods to test
        n_harmonics: Number of harmonics (1 = unimodal, 2 = bimodal, etc.)
    
    Returns:
        HarmonicCosinorResult object with peaks and fit model
    """
    if period_range is None:
        period_range = DefaultPeriodRanges.CIRCADIAN
    
    x = times
    y = values
    y_ranked = pd.Series(y).rank().values
    
    best_p = 1.0
    best_tau = 0.0
    best_per = None
    best_lag = None

    test_results = []

    for period in period_range:
        for lag in np.arange(0, period, 0.5):
            ref = np.zeros(len(x))
            for h in range(1, n_harmonics + 1):
                ref += np.cos(2 * np.pi * h * (x - lag) / period)
            ref_ranked = pd.Series(ref).rank().values

            if len(y_ranked) != len(ref_ranked):
                continue

            tau, pval = kendalltau(y_ranked, ref_ranked)
            test_results.append((pval, tau, period, lag))

            if pval < best_p:
                best_p = pval
                best_tau = tau
                best_per = period
                best_lag = lag

    bonf_p = min(1.0, best_p * len(test_results))
    amp_est = (np.percentile(y, 90) - np.percentile(y, 10)) / 2
    
    # Generate model for parameter estimation (1 cycle)
    t_grid = np.linspace(0, best_per, 1000)
    model_wave = np.zeros_like(t_grid)
    for h in range(1, n_harmonics + 1):
        model_wave += np.cos(2 * np.pi * h * (t_grid - best_lag) / best_per)
    
    # For plotting (full time)
    t_grid_full = np.linspace(x.min(), x.max(), 1000)
    model_wave_full = np.zeros_like(t_grid_full)
    for h in range(1, n_harmonics + 1):
        model_wave_full += np.cos(2 * np.pi * h * (t_grid_full - best_lag) / best_per)
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(model_wave, distance=200)
    
    # Fallback if not enough peaks detected
    n_peaks_expected = n_harmonics
    if len(peaks) < n_peaks_expected:
        peak_indices = np.argsort(model_wave)[-n_peaks_expected:]
    else:
        peak_indices = peaks[np.argsort(model_wave[peaks])[-n_peaks_expected:]]

    peak_indices = sorted(peak_indices)
    
    # Calculate acrophase per peak
    acrophases = [acrophase_to_hours(t_grid[i] / best_per * 2 * np.pi, best_per) 
                  for i in peak_indices]
    if best_tau < 0:
        acrophases = [(a + best_per / 2) % best_per for a in acrophases]

    # Normalize the model wave to [0, 1]
    norm_model_wave = (model_wave - np.min(model_wave)) / (np.ptp(model_wave) + 1e-10)

    # Get amplitude estimates based on normalized peak height
    peak_amps = [round(norm_model_wave[i] * amp_est * 2, 4) for i in peak_indices]

    # Pair acrophases with amps, sort by acrophase time
    acrophase_amp_pairs = sorted(zip(acrophases, peak_amps), key=lambda x: x[0])
    acrophases = [round(a, 2) for a, _ in acrophase_amp_pairs]
    peak_amps = [round(a, 4) for _, a in acrophase_amp_pairs]

    # Return model wave parameters for plotting
    fit_model = {
        't_grid': t_grid,
        'model_wave': amp_est * norm_model_wave,
        't_grid_full': t_grid_full,
        'model_wave_full': amp_est * (model_wave_full - np.min(model_wave)) / (np.ptp(model_wave) + 1e-10),
        'params': {
            'period': best_per,
            'lag': best_lag,
            'acrophases': acrophases,
            'amplitudes': peak_amps,
            'bonferroni_p': bonf_p
        }
    }

    return HarmonicCosinorResult(
        adj_p_value=round(bonf_p, 6),
        period=round(best_per, 2),
        amplitudes=peak_amps,
        acrophases=acrophases,
        n_harmonics=n_harmonics,
        method='Harmonic-Cosinor',
        fit_model=fit_model
    )


# =============================================================================
# FOURIER F24 ANALYSIS (WIJNEN ET AL., 2006)
# =============================================================================

def _compute_fourier_f24(
    times: np.ndarray,
    values: np.ndarray,
    target_period: float = 24.0,
    n_permutations: int = 1000
) -> FourierF24Result:
    """
    Compute F24 score using Fourier analysis (Wijnen et al., 2006 method).
    
    F24 = power at target frequency / mean power from random permutations
    
    This is an effect size measure, not a significance test.
    
    Args:
        times: Time values (evenly or unevenly spaced)
        values: Measurement values
        target_period: Target period to evaluate (default: 24h)
        n_permutations: Number of random permutations for baseline
    
    Returns:
        FourierF24Result object
    """
    n = len(values)
    
    # Compute FFT
    y_centered = values - np.mean(values)
    fft_result = np.fft.fft(y_centered)
    power_spectrum = np.abs(fft_result[:n//2])**2 / n
    
    # Compute frequencies (assuming evenly spaced data)
    dt = np.median(np.diff(times))  # Sample interval
    frequencies = np.fft.fftfreq(n, dt)[:n//2]
    periods = np.where(frequencies > 0, 1 / frequencies, np.inf)
    
    # Find power at target period
    target_freq = 1 / target_period
    freq_idx = np.argmin(np.abs(frequencies - target_freq))
    target_power = power_spectrum[freq_idx]
    
    # Find dominant period
    valid_mask = (periods > 0) & (periods < np.inf)
    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        dom_idx = valid_idx[np.argmax(power_spectrum[valid_mask])]
        dominant_period = periods[dom_idx]
        dominant_power = power_spectrum[dom_idx]
    else:
        dominant_period = np.nan
        dominant_power = np.nan
    
    # Compute mean power from random permutations
    rng = np.random.default_rng()
    perm_powers = []
    for _ in range(n_permutations):
        perm_values = rng.permutation(y_centered)
        perm_fft = np.fft.fft(perm_values)
        perm_power = np.abs(perm_fft[:n//2])**2 / n
        perm_powers.append(perm_power[freq_idx])
    
    mean_perm_power = np.mean(perm_powers)
    
    # F24 score
    f24_score = target_power / (mean_perm_power + 1e-10)
    
    return FourierF24Result(
        f24_score=round(f24_score, 4),
        power_spectrum=power_spectrum,
        frequencies=frequencies,
        dominant_period=round(dominant_period, 2),
        dominant_power=round(dominant_power, 4),
        target_period=target_period,
        target_power=round(target_power, 4),
        correlation_r=None,
        method='Fourier-F24'
    )


def _compute_fourier_f24_with_replicates(
    times_rep1: np.ndarray,
    values_rep1: np.ndarray,
    times_rep2: np.ndarray,
    values_rep2: np.ndarray,
    target_period: float = 24.0,
    n_permutations: int = 1000
) -> FourierF24Result:
    """
    Compute F24 score using two replicates (Wijnen et al., 2006 method).
    
    Uses correlation between replicates' Fourier coefficients at the target frequency.
    
    Args:
        times_rep1: Time values for replicate 1
        values_rep1: Values for replicate 1
        times_rep2: Time values for replicate 2
        values_rep2: Values for replicate 2
        target_period: Target period to evaluate
        n_permutations: Number of random permutations
    
    Returns:
        FourierF24Result object
    """
    # Center data
    y1 = values_rep1 - np.mean(values_rep1)
    y2 = values_rep2 - np.mean(values_rep2)
    
    n1, n2 = len(y1), len(y2)
    
    # Compute FFT for both replicates
    fft1 = np.fft.fft(y1)
    fft2 = np.fft.fft(y2)
    
    # Get power spectra
    power1 = np.abs(fft1[:n1//2])**2 / n1
    power2 = np.abs(fft2[:n2//2])**2 / n2
    
    # Average power spectrum
    min_len = min(len(power1), len(power2))
    avg_power = (power1[:min_len] + power2[:min_len]) / 2
    
    # Frequencies
    dt1 = np.median(np.diff(times_rep1))
    frequencies = np.fft.fftfreq(n1, dt1)[:n1//2][:min_len]
    periods = np.where(frequencies > 0, 1 / frequencies, np.inf)
    
    # Find target frequency index
    target_freq = 1 / target_period
    freq_idx = np.argmin(np.abs(frequencies - target_freq))
    target_power = avg_power[freq_idx]
    
    # Correlation between replicates at target frequency
    # Using complex coefficients
    if freq_idx < len(fft1) and freq_idx < len(fft2):
        coef1 = fft1[freq_idx]
        coef2 = fft2[freq_idx]
        # Correlation of real and imaginary parts
        correlation_r = np.corrcoef([coef1.real, coef1.imag], 
                                     [coef2.real, coef2.imag])[0, 1]
    else:
        correlation_r = None
    
    # Find dominant period
    valid_mask = (periods > 0) & (periods < np.inf)
    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        dom_idx = valid_idx[np.argmax(avg_power[valid_mask])]
        dominant_period = periods[dom_idx]
        dominant_power = avg_power[dom_idx]
    else:
        dominant_period = np.nan
        dominant_power = np.nan
    
    # Compute F24 using permutations
    rng = np.random.default_rng()
    perm_powers = []
    for _ in range(n_permutations):
        perm1 = rng.permutation(y1)
        perm2 = rng.permutation(y2)
        perm_fft1 = np.fft.fft(perm1)
        perm_fft2 = np.fft.fft(perm2)
        perm_power1 = np.abs(perm_fft1[:n1//2])**2 / n1
        perm_power2 = np.abs(perm_fft2[:n2//2])**2 / n2
        perm_avg = (perm_power1[freq_idx] + perm_power2[freq_idx]) / 2 if freq_idx < min(len(perm_power1), len(perm_power2)) else 0
        perm_powers.append(perm_avg)
    
    mean_perm_power = np.mean(perm_powers)
    f24_score = target_power / (mean_perm_power + 1e-10)
    
    return FourierF24Result(
        f24_score=round(f24_score, 4),
        power_spectrum=avg_power,
        frequencies=frequencies,
        dominant_period=round(dominant_period, 2),
        dominant_power=round(dominant_power, 4),
        target_period=target_period,
        target_power=round(target_power, 4),
        correlation_r=round(correlation_r, 4) if correlation_r is not None else None,
        method='Fourier-F24'
    )


# =============================================================================
# LOMB-SCARGLE PERIODOGRAM
# =============================================================================

def _compute_lomb_scargle(
    times: np.ndarray,
    values: np.ndarray,
    period_range: Tuple[float, float] = (18.0, 32.0),
    n_periods: int = 1000
) -> LombScargleResult:
    """
    Compute Lomb-Scargle periodogram for unevenly sampled data.
    
    Args:
        times: Time values (can be unevenly spaced)
        values: Measurement values
        period_range: (min_period, max_period) tuple
        n_periods: Number of periods to evaluate
    
    Returns:
        LombScargleResult object
    """
    # Remove NaN values
    mask = ~(np.isnan(times) | np.isnan(values))
    t = times[mask].astype(float)
    y = values[mask].astype(float)
    
    # Center the data
    y_centered = y - np.mean(y)
    
    # Define period grid
    periods = np.linspace(period_range[0], period_range[1], n_periods)
    angular_freqs = 2 * np.pi / periods
    
    # Compute Lomb-Scargle periodogram
    pgram = lombscargle(t, y_centered, angular_freqs, normalize=True)
    
    # Find dominant period
    max_idx = np.argmax(pgram)
    dominant_period = periods[max_idx]
    dominant_power = pgram[max_idx]
    
    # Estimate false alarm probability (simplified)
    # Using the approximation: FAP ≈ 1 - (1 - exp(-power))^M
    # where M is the effective number of independent frequencies
    M = len(t)  # Simplified approximation
    fap = 1 - (1 - np.exp(-dominant_power))**M
    fap = min(1.0, max(0.0, fap))
    
    return LombScargleResult(
        dominant_period=round(dominant_period, 2),
        dominant_power=round(dominant_power, 4),
        false_alarm_probability=round(fap, 6),
        periods=periods,
        power_spectrum=pgram,
        method='Lomb-Scargle'
    )


# =============================================================================
# CONTINUOUS WAVELET TRANSFORM (CWT)
# =============================================================================

def _compute_cwt(
    times: np.ndarray,
    values: np.ndarray,
    sampling_interval: float = 0.5,
    wavelet: str = 'cmor1.5-1.0',
    period_range: Tuple[float, float] = (20.0, 28.0)
) -> CWTResult:
    """
    Compute Continuous Wavelet Transform analysis.
    
    Args:
        times: Time values
        values: Measurement values
        sampling_interval: Sampling interval in hours
        wavelet: Wavelet type (default: complex Morlet)
        period_range: (min_period, max_period) tuple
    
    Returns:
        CWTResult object
    
    Raises:
        ImportError: If PyWavelets is not available
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets (pywt) is required for CWT analysis. "
                         "Install with: pip install PyWavelets")
    
    # Sort by time
    sort_idx = np.argsort(times)
    x = times[sort_idx]
    y = values[sort_idx]
    
    # Detrend the signal
    y_detrended = y - np.mean(y)
    
    # Create period array with coarser resolution
    period_array = np.arange(period_range[0], period_range[1] + 0.5, 0.5)
    
    # For complex Morlet wavelet (cmor), the central frequency is approximately 1.0
    # The scale relates to period as: scale ≈ (period / dt) / central_freq
    # For cmor1.5-1.0: fb=1.5 (bandwidth), fc=1.0 (center frequency)
    central_freq = 1.0
    scales = period_array / (central_freq * sampling_interval)
    
    # CWT works best when scale * wavelet_length fits within data length
    # A safe maximum scale is approximately n_samples / 2
    n_samples = len(y)
    max_safe_scale = n_samples / 4  # More conservative limit
    
    valid_mask = scales <= max_safe_scale
    if not valid_mask.any():
        # If no scales are valid, warn the user
        min_samples = int(min(period_array) / sampling_interval * 4)
        warnings.warn(f"Time series too short for CWT in period range. "
                     f"Need at least {min_samples} samples for minimum period {min(period_array)}h.")
        return CWTResult(
            dominant_period=np.nan,
            mean_power=np.nan,
            period_variation=np.nan,
            amplitude_modulations=0,
            method='CWT'
        )
    
    scales = scales[valid_mask]
    period_array = period_array[valid_mask]
    
    # Perform CWT
    try:
        coef, freqs = pywt.cwt(y_detrended, scales, wavelet, sampling_period=sampling_interval)
        power = np.abs(coef) ** 2
    except Exception as e:
        warnings.warn(f"CWT computation failed: {e}")
        return CWTResult(
            dominant_period=np.nan,
            mean_power=np.nan,
            period_variation=np.nan,
            amplitude_modulations=0,
            method='CWT'
        )
    
    # Average power across time (global wavelet spectrum)
    global_power = power.mean(axis=1)
    dominant_idx = np.argmax(global_power)
    dominant_period = period_array[dominant_idx]
    dominant_power = global_power[dominant_idx]
    
    # Period drift: find dominant period per timepoint
    dom_periods_over_time = period_array[np.argmax(power, axis=0)]
    period_variation = np.std(dom_periods_over_time)
    
    # Detect amplitude modulation
    mean_power = np.mean(power, axis=0)
    amp_peaks, _ = find_peaks(mean_power)
    amp_fluctuations = len(amp_peaks)
    
    return CWTResult(
        dominant_period=round(float(dominant_period), 2),
        mean_power=round(float(np.mean(global_power)), 4),
        period_variation=round(float(period_variation), 3),
        amplitude_modulations=amp_fluctuations,
        method='CWT'
    )


# =============================================================================
# LINEAR MIXED EFFECTS MODEL
# =============================================================================

def _fit_lme_model(
    df: pd.DataFrame,
    dependent: str,
    fixed_effects: List[str],
    random_effect: str
) -> LMEResult:
    """
    Fit Linear Mixed Effects model.
    
    Useful for analyzing data with hierarchical structure (e.g., multiple
    measurements per subject).
    
    Args:
        df: DataFrame with the data
        dependent: Name of dependent variable column
        fixed_effects: List of fixed effect column names
        random_effect: Name of random effect grouping column
    
    Returns:
        LMEResult object
    """
    # Construct formula
    fixed_str = " + ".join(fixed_effects)
    formula = f"{dependent} ~ {fixed_str}"
    
    # Fit model
    model = smf.mixedlm(formula, df, groups=df[random_effect])
    result = model.fit()
    
    return LMEResult(
        terms=result.params.index.tolist(),
        estimates=result.params.values.tolist(),
        std_errors=result.bse.values.tolist(),
        z_values=result.tvalues.values.tolist(),
        p_values=result.pvalues.values.tolist(),
        method='LME'
    )


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class RhythmAnalyzer:
    """
    Main class for comprehensive rhythm analysis.
    
    Provides a unified interface for multiple rhythm detection methods
    compatible with the CosinorPy and CircaCompare module structure.
    
    Attributes:
        period_range: List of periods to test
        default_period: Default period for single-period analyses
    
    Example:
        >>> analyzer = RhythmAnalyzer()
        >>> df = analyzer.load_csv("data.csv")
        >>> 
        >>> # JTK analysis
        >>> jtk_result = analyzer.run_jtk("geneA", "winter")
        >>> 
        >>> # Cosinor analysis
        >>> cosinor_result = analyzer.run_cosinor("geneA", "winter")
        >>> 
        >>> # Fourier F24 analysis
        >>> f24_result = analyzer.run_fourier_f24("geneA", "winter")
    """
    
    def __init__(
        self,
        period_range: Optional[List[float]] = None,
        default_period: float = 24.0
    ):
        """
        Initialize the RhythmAnalyzer.
        
        Args:
            period_range: List of periods to test. If None, uses default circadian range.
            default_period: Default period for methods that use a single period.
        """
        self.period_range = period_range or DefaultPeriodRanges.CIRCADIAN.copy()
        self.default_period = default_period
        
        self._raw_data: Optional[pd.DataFrame] = None
        self._variables: List[str] = []
        self._conditions: List[str] = []
        self._time_col: str = "time"
        self._condition_col: str = "condition"
        self._replicate_col: Optional[str] = None
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_csv(
        self,
        filepath: str,
        time_column: str = "time",
        condition_column: str = "condition",
        replicate_column: Optional[str] = None,
        variable_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and preprocess CSV data for rhythm analysis.
        
        Args:
            filepath: Path to the CSV file
            time_column: Name of the column containing time values (in hours)
            condition_column: Name of the column containing condition labels
            replicate_column: Name of column with replicate IDs (optional)
            variable_columns: List of columns to analyze. If None, auto-detects
                            all numeric columns except time/condition/replicate.
        
        Returns:
            Loaded pandas DataFrame
        
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If required columns are missing
        """
        self._raw_data = pd.read_csv(filepath)
        return self._setup_data(time_column, condition_column, replicate_column, variable_columns)
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        time_column: str = "time",
        condition_column: str = "condition",
        replicate_column: Optional[str] = None,
        variable_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from an existing DataFrame.
        
        Args:
            df: pandas DataFrame with the data
            time_column: Name of the column containing time values
            condition_column: Name of the column containing condition labels
            replicate_column: Name of column with replicate IDs
            variable_columns: List of columns to analyze
        
        Returns:
            Copy of the loaded DataFrame
        """
        self._raw_data = df.copy()
        return self._setup_data(time_column, condition_column, replicate_column, variable_columns)
    
    def _setup_data(
        self,
        time_column: str,
        condition_column: str,
        replicate_column: Optional[str],
        variable_columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Set up data after loading."""
        # Validate required columns
        required_cols = [time_column, condition_column]
        missing = [c for c in required_cols if c not in self._raw_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self._time_col = time_column
        self._condition_col = condition_column
        self._replicate_col = replicate_column
        
        # Auto-detect variable columns
        if variable_columns is None:
            exclude_cols = {time_column, condition_column}
            if replicate_column:
                exclude_cols.add(replicate_column)
            exclude_cols.add('subject')  # Common column to exclude
            
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
    
    def _get_data_for_analysis(
        self,
        variable: str,
        condition: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time and values for a specific variable and condition.
        
        Args:
            variable: Variable name
            condition: Condition name
        
        Returns:
            Tuple of (times, values) arrays
        """
        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_csv() or load_dataframe() first.")
        
        if variable not in self._variables:
            raise ValueError(f"Variable '{variable}' not found. Available: {self._variables}")
        if condition not in self._conditions:
            raise ValueError(f"Condition '{condition}' not found. Available: {self._conditions}")
        
        cond_data = self._raw_data[self._raw_data[self._condition_col] == condition]
        times = cond_data[self._time_col].values.astype(float)
        values = cond_data[variable].values.astype(float)
        
        # Remove NaN
        mask = ~(np.isnan(times) | np.isnan(values))
        return times[mask], values[mask]
    
    def _get_replicate_count(self, condition: str) -> int:
        """
        Count the number of replicates for a condition.
        
        Returns:
            Number of replicates (1 if no explicit replicates)
        """
        cond_data = self._raw_data[self._raw_data[self._condition_col] == condition]
        
        if self._replicate_col and self._replicate_col in cond_data.columns:
            return cond_data[self._replicate_col].nunique()
        
        # Check for implicit replicates (multiple values at same timepoint)
        time_counts = cond_data[self._time_col].value_counts()
        if (time_counts > 1).any():
            return time_counts.max()
        
        return 1
    
    def _get_replicate_data(
        self,
        variable: str,
        condition: str
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get data separated by replicates.
        
        Args:
            variable: Variable name
            condition: Condition name
        
        Returns:
            List of (times, values) tuples, one per replicate
        """
        cond_data = self._raw_data[self._raw_data[self._condition_col] == condition]
        
        if self._replicate_col and self._replicate_col in cond_data.columns:
            # Explicit replicates
            replicates = []
            for rep in cond_data[self._replicate_col].unique():
                rep_data = cond_data[cond_data[self._replicate_col] == rep]
                times = rep_data[self._time_col].values.astype(float)
                values = rep_data[variable].values.astype(float)
                mask = ~(np.isnan(times) | np.isnan(values))
                replicates.append((times[mask], values[mask]))
            return replicates
        
        # Check for implicit replicates
        time_counts = cond_data[self._time_col].value_counts()
        if (time_counts > 1).any():
            # Group by timepoint and assign implicit replicate numbers
            unique_times = sorted(cond_data[self._time_col].unique())
            max_reps = time_counts.max()
            
            replicates = [[] for _ in range(max_reps)]
            rep_times = [[] for _ in range(max_reps)]
            
            for t in unique_times:
                time_data = cond_data[cond_data[self._time_col] == t]
                for i, (_, row) in enumerate(time_data.iterrows()):
                    if i < max_reps:
                        rep_times[i].append(t)
                        replicates[i].append(row[variable])
            
            result = []
            for i in range(max_reps):
                times = np.array(rep_times[i], dtype=float)
                values = np.array(replicates[i], dtype=float)
                mask = ~(np.isnan(times) | np.isnan(values))
                if mask.any():
                    result.append((times[mask], values[mask]))
            
            return result
        
        # Single replicate
        times, values = self._get_data_for_analysis(variable, condition)
        return [(times, values)]
    
    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def run_jtk(
        self,
        variable: str,
        condition: str,
        period_range: Optional[List[float]] = None,
        lag_range: Optional[np.ndarray] = None,
        asymmetries: Optional[List[float]] = None
    ) -> JTKResult:
        """
        Run Python-JTK cycle analysis.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            period_range: List of periods to test (default: class period_range)
            lag_range: Array of lag values (default: 0 to period)
            asymmetries: List of waveform asymmetries (default: [0.5])
        
        Returns:
            JTKResult object
        """
        times, values = self._get_data_for_analysis(variable, condition)
        series = pd.Series(values, index=times)
        
        if period_range is None:
            # Use integer periods for discrete JTK
            period_range = [int(p) for p in self.period_range if p == int(p)]
            if not period_range:
                period_range = DefaultPeriodRanges.CIRCADIAN_INT
        
        return _run_discrete_jtk(series, period_range, lag_range, asymmetries)
    
    def run_ar_jtk(
        self,
        variable: str,
        condition: str,
        period_range: Optional[List[float]] = None,
        lag_range: Optional[np.ndarray] = None,
        asymmetries: Optional[List[float]] = None,
        ar_lag: int = 1,
        ljungbox_lag: int = 10
    ) -> Tuple[JTKResult, bool]:
        """
        Run AR-JTK analysis with autoregressive noise handling.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            period_range: List of periods to test
            lag_range: Array of lag values
            asymmetries: List of waveform asymmetries
            ar_lag: AR model lag for prewhitening
            ljungbox_lag: Lag for Ljung-Box autocorrelation test
        
        Returns:
            Tuple of (JTKResult, bool indicating if AR was applied)
        """
        times, values = self._get_data_for_analysis(variable, condition)
        series = pd.Series(values, index=times)
        
        if period_range is None:
            period_range = [int(p) for p in self.period_range if p == int(p)]
            if not period_range:
                period_range = DefaultPeriodRanges.CIRCADIAN_INT
        
        return _run_ar_jtk(series, period_range, lag_range, asymmetries, ar_lag, ljungbox_lag)
    
    def run_cosine_kendall(
        self,
        variable: str,
        condition: str,
        period_range: Optional[List[float]] = None,
        interval: Optional[float] = None
    ) -> JTKResult:
        """
        Run Cosine-Kendall nonparametric analysis.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            period_range: List of periods to test
            interval: Time interval between samples (auto-detected if None)
        
        Returns:
            JTKResult object (method = 'Cosine-Kendall')
        """
        times, values = self._get_data_for_analysis(variable, condition)
        series = pd.Series(values, index=times)
        
        if period_range is None:
            period_range = self.period_range
        
        if interval is None:
            interval = np.median(np.diff(np.sort(times)))
        
        return _run_cosine_kendall(series, period_range, interval)
    
    def run_cosinor(
        self,
        variable: str,
        condition: str,
        period_range: Optional[List[float]] = None
    ) -> CosinorResult:
        """
        Run standard Cosinor analysis with period optimization.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            period_range: List of periods to test
        
        Returns:
            CosinorResult object
        """
        times, values = self._get_data_for_analysis(variable, condition)
        
        if period_range is None:
            period_range = self.period_range
        
        return _fit_cosinor(times, values, period_range)
    
    def run_harmonic_cosinor(
        self,
        variable: str,
        condition: str,
        period_range: Optional[List[float]] = None,
        n_harmonics: int = 2
    ) -> HarmonicCosinorResult:
        """
        Run Harmonic Cosinor analysis for multi-modal rhythm detection.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            period_range: List of periods to test
            n_harmonics: Number of harmonics (1=unimodal, 2=bimodal, etc.)
        
        Returns:
            HarmonicCosinorResult object
        """
        times, values = self._get_data_for_analysis(variable, condition)
        
        if period_range is None:
            period_range = self.period_range
        
        return _fit_harmonic_cosinor(times, values, period_range, n_harmonics)
    
    def run_fourier_f24(
        self,
        variable: str,
        condition: str,
        target_period: Optional[float] = None,
        n_permutations: int = 1000
    ) -> FourierF24Result:
        """
        Run Fourier F24 analysis (Wijnen et al., 2006 method).
        
        Requires exactly 2 replicates. If more replicates exist, they are
        randomly split into 2 groups and averaged.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            target_period: Period to evaluate (default: default_period)
            n_permutations: Number of permutations for F24 calculation
        
        Returns:
            FourierF24Result object
        
        Raises:
            ValueError: If fewer than 2 replicates are available
        """
        if target_period is None:
            target_period = self.default_period
        
        n_reps = self._get_replicate_count(condition)
        
        if n_reps < 2:
            raise ValueError(
                f"Fourier F24 analysis requires at least 2 replicates. "
                f"Found {n_reps} replicate(s) for condition '{condition}'. "
                f"Please ensure your data has replicates at each timepoint "
                f"or specify a 'replicate' column."
            )
        
        replicate_data = self._get_replicate_data(variable, condition)
        
        if len(replicate_data) == 2:
            # Exactly 2 replicates - use directly
            times1, values1 = replicate_data[0]
            times2, values2 = replicate_data[1]
        else:
            # More than 2 replicates - randomly split into 2 groups and average
            rng = np.random.default_rng()
            indices = np.arange(len(replicate_data))
            rng.shuffle(indices)
            
            mid = len(indices) // 2
            group1_idx = indices[:mid]
            group2_idx = indices[mid:]
            
            # Average within groups
            # Assuming all replicates have the same timepoints
            times1 = replicate_data[group1_idx[0]][0]
            values1 = np.mean([replicate_data[i][1] for i in group1_idx], axis=0)
            
            times2 = replicate_data[group2_idx[0]][0]
            values2 = np.mean([replicate_data[i][1] for i in group2_idx], axis=0)
        
        return _compute_fourier_f24_with_replicates(
            times1, values1, times2, values2, target_period, n_permutations
        )
    
    def run_lomb_scargle(
        self,
        variable: str,
        condition: str,
        period_range: Optional[Tuple[float, float]] = None,
        n_periods: int = 1000
    ) -> LombScargleResult:
        """
        Run Lomb-Scargle periodogram analysis.
        
        Particularly useful for unevenly sampled data.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            period_range: (min_period, max_period) tuple
            n_periods: Number of periods to evaluate
        
        Returns:
            LombScargleResult object
        """
        times, values = self._get_data_for_analysis(variable, condition)
        
        if period_range is None:
            period_range = (min(self.period_range), max(self.period_range))
        
        return _compute_lomb_scargle(times, values, period_range, n_periods)
    
    def run_cwt(
        self,
        variable: str,
        condition: str,
        sampling_interval: Optional[float] = None,
        wavelet: str = 'cmor1.5-1.0',
        period_range: Optional[Tuple[float, float]] = None
    ) -> CWTResult:
        """
        Run Continuous Wavelet Transform analysis.
        
        Useful for detecting time-varying rhythms and non-stationary signals.
        
        Args:
            variable: Variable name to analyze
            condition: Condition to analyze
            sampling_interval: Sampling interval in hours (auto-detected if None)
            wavelet: Wavelet type (default: complex Morlet)
            period_range: (min_period, max_period) tuple
        
        Returns:
            CWTResult object
        
        Raises:
            ImportError: If PyWavelets is not installed
        """
        times, values = self._get_data_for_analysis(variable, condition)
        
        if sampling_interval is None:
            # Calculate sampling interval, excluding duplicates (replicates at same timepoint)
            sorted_times = np.sort(times)
            diffs = np.diff(sorted_times)
            nonzero_diffs = diffs[diffs > 0]
            if len(nonzero_diffs) > 0:
                sampling_interval = np.median(nonzero_diffs)
            else:
                sampling_interval = 1.0  # Default fallback
        
        if period_range is None:
            period_range = (min(self.period_range), max(self.period_range))
        
        return _compute_cwt(times, values, sampling_interval, wavelet, period_range)
    
    def run_lme(
        self,
        dependent: str,
        fixed_effects: List[str],
        random_effect: str,
        condition: Optional[str] = None
    ) -> LMEResult:
        """
        Run Linear Mixed Effects model analysis.
        
        Useful for analyzing hierarchical data with repeated measurements.
        
        Args:
            dependent: Name of dependent variable column
            fixed_effects: List of fixed effect column names
            random_effect: Name of random effect grouping column
            condition: Optional condition filter. If None, uses all data.
        
        Returns:
            LMEResult object
        """
        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_csv() or load_dataframe() first.")
        
        if condition is not None:
            df = self._raw_data[self._raw_data[self._condition_col] == condition].copy()
        else:
            df = self._raw_data.copy()
        
        return _fit_lme_model(df, dependent, fixed_effects, random_effect)

    def run_spectral_analysis(
        self,
        variable: str,
        condition: str,
        per_type: str = 'per',
        max_per: float = 240.0,
        prominent: bool = True
    ) -> Dict[str, Any]:
        """
        Run spectral analysis (periodogram) to identify dominant periods.

        This is an improved implementation that returns data for interactive
        visualization, unlike CosinorPy's periodogram which only generates plots.

        Uses scipy.signal for spectral analysis:
        - 'per': Standard FFT periodogram (for evenly sampled data)
        - 'welch': Welch's method (averaged periodogram, good for noisy data)
        - 'lombscargle': Lomb-Scargle (for unevenly sampled data or with replicates)

        Significance threshold based on Refinetti et al. 2007.

        Args:
            variable: Name of the variable to analyze
            condition: Condition to analyze
            per_type: Type of periodogram ('per', 'welch', or 'lombscargle')
            max_per: Maximum period to consider (in hours)
            prominent: Whether to identify prominent peaks

        Returns:
            Dictionary with:
                - periods: Array of period values
                - power: Array of power spectral density values
                - dominant_period: Period with highest power
                - threshold: Significance threshold
                - significant_peaks: List of significant periods
        """
        from scipy import signal

        if self._raw_data is None:
            raise ValueError("No data loaded. Call load_csv() or load_dataframe() first.")

        # Extract data for this variable and condition
        if condition is not None:
            subset = self._raw_data[self._raw_data[self._condition_col] == condition]
        else:
            subset = self._raw_data.copy()

        X = subset[self._time_col].values
        Y = subset[variable].values

        # Calculate periodogram based on type
        if per_type == 'per':
            # For standard periodogram, need evenly sampled data
            X_u = np.unique(X)
            Y_u = []
            for x_u in X_u:
                Y_u.append(np.median(Y[x_u == X]))

            if len(X_u) > 1:
                time_diffs = np.diff(X_u)
                sampling_interval = np.median(time_diffs)
                sampling_f = 1 / sampling_interval
            else:
                raise ValueError("Need at least 2 time points for periodogram")

            f, Pxx_den = signal.periodogram(Y_u, sampling_f)

        elif per_type == 'welch':
            # Welch's method also needs evenly sampled data
            X_u = np.unique(X)
            Y_u = []
            for x_u in X_u:
                Y_u.append(np.median(Y[x_u == X]))

            if len(X_u) > 1:
                time_diffs = np.diff(X_u)
                sampling_interval = np.median(time_diffs)
                sampling_f = 1 / sampling_interval
            else:
                raise ValueError("Need at least 2 time points for periodogram")

            f, Pxx_den = signal.welch(Y_u, sampling_f)

        elif per_type == 'lombscargle':
            # Lomb-Scargle can handle uneven sampling and replicates
            min_per = 2
            f = np.linspace(1/max_per, 1/min_per, 1000)
            Pxx_den = signal.lombscargle(X, Y, f)
        else:
            raise ValueError(f"Invalid periodogram type: {per_type}")

        # Convert frequency to period
        if f[0] == 0:
            per = 1 / f[1:]
            Pxx = Pxx_den[1:]
        else:
            per = 1 / f
            Pxx = Pxx_den

        # Filter to max_per
        Pxx = Pxx[per <= max_per]
        per = per[per <= max_per]

        # Calculate significance threshold (Refinetti et al. 2007)
        p_t = 0.05
        N = len(Y_u) if per_type in ['per', 'welch'] else len(Y)
        T = (1 - (p_t/N)**(1/(N-1))) * sum(Pxx_den)

        result = {
            'periods': per,
            'power': Pxx,
            'threshold': T
        }

        # Find prominent peaks if requested
        if prominent:
            # Always identify the dominant period (highest power)
            max_idx = np.argmax(Pxx)
            dominant_per = per[max_idx]
            result['dominant_period'] = float(dominant_per)

            # Find significant peaks
            if len(Pxx) < 10:
                # For sparse data, identify periods above threshold
                significant_indices = np.where(Pxx > T)[0]
                if len(significant_indices) > 0:
                    result['significant_peaks'] = [float(per[i]) for i in significant_indices]
                else:
                    result['significant_peaks'] = []
            else:
                # For denser data, use proper peak detection
                locs, heights = signal.find_peaks(Pxx, height=T)
                if len(locs) > 0:
                    heights = heights['peak_heights']
                    s = list(zip(heights, locs))
                    s.sort(reverse=True)
                    heights_sorted, locs_sorted = zip(*s)
                    result['significant_peaks'] = [float(per[loc]) for loc in locs_sorted]
                else:
                    result['significant_peaks'] = []

        return result

    # =========================================================================
    # BATCH ANALYSIS METHODS
    # =========================================================================
    
    def run_all_jtk(
        self,
        condition: Optional[str] = None,
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run JTK analysis on all variables.
        
        Args:
            condition: Condition to analyze (if None, analyzes all)
            variables: List of variables (if None, uses all)
            **kwargs: Additional arguments for run_jtk
        
        Returns:
            DataFrame with results for all variables
        """
        if variables is None:
            variables = self._variables
        
        conditions = [condition] if condition else self._conditions
        
        results = []
        for cond in conditions:
            for var in variables:
                try:
                    result = self.run_jtk(var, cond, **kwargs)
                    row = result.to_dict()
                    row['variable'] = var
                    row['condition'] = cond
                    results.append(row)
                except Exception as e:
                    warnings.warn(f"JTK failed for {var}/{cond}: {e}")
        
        return pd.DataFrame(results)
    
    def run_all_cosinor(
        self,
        condition: Optional[str] = None,
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run Cosinor analysis on all variables.
        
        Args:
            condition: Condition to analyze (if None, analyzes all)
            variables: List of variables (if None, uses all)
            **kwargs: Additional arguments for run_cosinor
        
        Returns:
            DataFrame with results for all variables
        """
        if variables is None:
            variables = self._variables
        
        conditions = [condition] if condition else self._conditions
        
        results = []
        for cond in conditions:
            for var in variables:
                try:
                    result = self.run_cosinor(var, cond, **kwargs)
                    row = result.to_dict()
                    row['variable'] = var
                    row['condition'] = cond
                    results.append(row)
                except Exception as e:
                    warnings.warn(f"Cosinor failed for {var}/{cond}: {e}")
        
        return pd.DataFrame(results)
    
    def run_all_lomb_scargle(
        self,
        condition: Optional[str] = None,
        variables: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Run Lomb-Scargle analysis on all variables.
        
        Args:
            condition: Condition to analyze (if None, analyzes all)
            variables: List of variables (if None, uses all)
            **kwargs: Additional arguments for run_lomb_scargle
        
        Returns:
            DataFrame with results for all variables
        """
        if variables is None:
            variables = self._variables
        
        conditions = [condition] if condition else self._conditions
        
        results = []
        for cond in conditions:
            for var in variables:
                try:
                    result = self.run_lomb_scargle(var, cond, **kwargs)
                    row = result.to_dict()
                    row['variable'] = var
                    row['condition'] = cond
                    results.append(row)
                except Exception as e:
                    warnings.warn(f"Lomb-Scargle failed for {var}/{cond}: {e}")
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # PREDICTION/FITTING METHODS (for GUI plotting)
    # =========================================================================
    
    def predict_cosinor(
        self,
        result: CosinorResult,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fitted curve from cosinor result.
        
        Args:
            result: CosinorResult object
            n_points: Number of points for the curve
        
        Returns:
            Tuple of (time_grid, fitted_values)
        """
        t = np.linspace(0, result.period, n_points)
        omega = 2 * np.pi / result.period
        y = result.mesor + result.amplitude * np.cos(omega * t - result.acrophase_rad)
        return t, y
    
    def predict_harmonic_cosinor(
        self,
        result: HarmonicCosinorResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fitted curve from harmonic cosinor result.
        
        Args:
            result: HarmonicCosinorResult object
        
        Returns:
            Tuple of (time_grid, fitted_values)
        """
        if result.fit_model is None:
            raise ValueError("Fit model not available")
        
        return (result.fit_model['t_grid'], result.fit_model['model_wave'])


# =============================================================================
# CONVENIENCE FUNCTIONS FOR DIRECT USE
# =============================================================================

def analyze_rhythm(
    times: np.ndarray,
    values: np.ndarray,
    method: str = 'cosinor',
    period_range: Optional[List[float]] = None,
    **kwargs
) -> Any:
    """
    Convenience function for quick rhythm analysis.
    
    Args:
        times: Time values
        values: Measurement values
        method: Analysis method ('jtk', 'ar_jtk', 'cosine_kendall', 
                'cosinor', 'harmonic_cosinor', 'lomb_scargle', 'cwt')
        period_range: List of periods to test
        **kwargs: Additional method-specific arguments
    
    Returns:
        Result object appropriate for the method
    """
    if period_range is None:
        period_range = DefaultPeriodRanges.CIRCADIAN
    
    method = method.lower().replace('-', '_').replace(' ', '_')
    
    if method in ('jtk', 'python_jtk'):
        series = pd.Series(values, index=times)
        int_periods = [int(p) for p in period_range if p == int(p)]
        return _run_discrete_jtk(series, int_periods or DefaultPeriodRanges.CIRCADIAN_INT, **kwargs)
    
    elif method == 'ar_jtk':
        series = pd.Series(values, index=times)
        int_periods = [int(p) for p in period_range if p == int(p)]
        return _run_ar_jtk(series, int_periods or DefaultPeriodRanges.CIRCADIAN_INT, **kwargs)
    
    elif method == 'cosine_kendall':
        series = pd.Series(values, index=times)
        return _run_cosine_kendall(series, period_range, **kwargs)
    
    elif method == 'cosinor':
        return _fit_cosinor(times, values, period_range)
    
    elif method == 'harmonic_cosinor':
        n_harmonics = kwargs.pop('n_harmonics', 2)
        return _fit_harmonic_cosinor(times, values, period_range, n_harmonics)
    
    elif method == 'lomb_scargle':
        period_tuple = (min(period_range), max(period_range))
        return _compute_lomb_scargle(times, values, period_tuple, **kwargs)
    
    elif method == 'cwt':
        period_tuple = (min(period_range), max(period_range))
        return _compute_cwt(times, values, period_range=period_tuple, **kwargs)
    
    elif method == 'fourier_f24':
        target_period = kwargs.pop('target_period', 24.0)
        return _compute_fourier_f24(times, values, target_period, **kwargs)
    
    else:
        raise ValueError(f"Unknown method: {method}. Available: jtk, ar_jtk, "
                        f"cosine_kendall, cosinor, harmonic_cosinor, lomb_scargle, cwt, fourier_f24")
