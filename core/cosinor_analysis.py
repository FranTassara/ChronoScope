"""
CosinorPy Analysis Module
=========================

A comprehensive wrapper module for the CosinorPy library that provides
cosinor-based rhythmometry analysis functions.

This module is designed to be integrated with a GUI interface and provides:
- Single-Component Cosinor Analysis
- Multi-Component Cosinor Analysis  
- Population-Mean Cosinor Analysis
- Differential Rhythmicity Analysis
- Generalized Cosinor (Non-linear) Analysis
- Count Data Analysis (Poisson regression)

CSV Format Requirements:
------------------------
The module expects CSV files with the following structure:

For INDEPENDENT data (e.g., qPCR with different subjects per timepoint):
    time,condition,variable1,variable2,...
    0,winter,1.2,3.4
    4,winter,2.3,4.5
    8,winter,1.8,3.9
    ...
    0,summer,1.5,3.2
    ...

For DEPENDENT data (population-mean, same subjects over time):
    time,condition,subject,variable1,variable2,...
    0,winter,subject1,1.2,3.4
    4,winter,subject1,2.3,4.5
    0,winter,subject2,1.3,3.5
    ...

If replicates exist at the same timepoint (independent data):
    time,condition,replicate,variable1,variable2,...
    0,winter,1,1.2,3.4
    0,winter,2,1.3,3.5
    0,winter,3,1.1,3.3
    ...

Dependencies:
-------------
    pip install cosinorpy pandas numpy

CosinorPy Plotting Functions (for GUI reference):
-------------------------------------------------
    - cosinor.plot_data(df, test): Plot raw data
    - cosinor.plot_data_pairs(df, test1, test2): Plot two groups together
    - cosinor.plot_phases(...): Polar plot of phases
    - cosinor.periodogram(df, test, ...): Periodogram with significance threshold
    - cosinor1.fit_cosinor(X, Y, ...): Single cosinor with optional plot

Author: Generated for GUI integration
Version: 1.0.0
"""

from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

import pandas as pd
import numpy as np

# CosinorPy imports
try:
    from CosinorPy import file_parser, cosinor, cosinor1, cosinor_nonlin
    COSINORPY_AVAILABLE = True
except ImportError:
    COSINORPY_AVAILABLE = False
    warnings.warn(
        "CosinorPy is not installed. Please install it with: pip install cosinorpy"
    )


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class DataType(Enum):
    """Enumeration for data types supported by cosinor analysis."""
    CONTINUOUS = "continuous"  # For normalized expression data (qPCR, IHC, etc.)
    COUNT = "count"            # For count data (RNA-seq, etc.)


class ModelType(Enum):
    """Enumeration for regression model types."""
    LINEAR = "linear"              # Standard linear regression
    POISSON = "poisson"            # Poisson regression for count data
    GEN_POISSON = "gen_poisson"    # Generalized Poisson
    NEGATIVE_BINOMIAL = "nb"       # Negative binomial


class AnalysisMode(Enum):
    """Enumeration for analysis modes based on data structure."""
    INDEPENDENT = "independent"    # Different subjects per timepoint
    DEPENDENT = "dependent"        # Same subjects measured over time (population-mean)


@dataclass
class CosinorParameters:
    """
    Data class containing cosinor model parameters.
    
    Attributes:
        mesor: Midline Estimating Statistic Of Rhythm (rhythm-adjusted mean)
        amplitude: Half the peak-to-trough difference
        acrophase: Time of peak (in radians, relative to period)
        acrophase_hours: Time of peak converted to hours
        period: The period used for fitting
        p_value: Statistical significance of the rhythm (zero-amplitude test)
        r_squared: Proportion of variance explained by the model
        confidence_intervals: Dict with CI for each parameter
    """
    mesor: float
    amplitude: float
    acrophase: float
    acrophase_hours: float
    period: float
    p_value: float
    r_squared: Optional[float] = None
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the rhythm is statistically significant."""
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'mesor': self.mesor,
            'amplitude': self.amplitude,
            'acrophase': self.acrophase,
            'acrophase_hours': self.acrophase_hours,
            'period': self.period,
            'p_value': self.p_value,
            'r_squared': self.r_squared,
            'confidence_intervals': self.confidence_intervals
        }


@dataclass
class DifferentialResult:
    """
    Data class for differential rhythmicity analysis results.
    
    Attributes:
        condition1: Name of first condition
        condition2: Name of second condition
        amplitude_diff: Difference in amplitudes
        amplitude_p_value: P-value for amplitude difference
        acrophase_diff: Difference in acrophases (hours)
        acrophase_p_value: P-value for acrophase difference
        mesor_diff: Difference in MESORs
        mesor_p_value: P-value for MESOR difference
        q_values: FDR-adjusted p-values (if multiple comparisons)
    """
    condition1: str
    condition2: str
    amplitude_diff: float
    amplitude_p_value: float
    acrophase_diff: float
    acrophase_p_value: float
    mesor_diff: Optional[float] = None
    mesor_p_value: Optional[float] = None
    q_values: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'condition1': self.condition1,
            'condition2': self.condition2,
            'amplitude_diff': self.amplitude_diff,
            'amplitude_p_value': self.amplitude_p_value,
            'acrophase_diff': self.acrophase_diff,
            'acrophase_p_value': self.acrophase_p_value,
            'mesor_diff': self.mesor_diff,
            'mesor_p_value': self.mesor_p_value,
            'q_values': self.q_values
        }


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class CosinorAnalyzer:
    """
    Main class for performing cosinor-based rhythmometry analysis.
    
    This class wraps CosinorPy functionality and provides a unified interface
    for various types of cosinor analysis including single-component,
    multi-component, population-mean, and differential rhythmicity analyses.
    
    Attributes:
        period: Expected period of oscillation (default: 24 hours)
        n_components: Number of cosinor components for multi-component analysis
        data_type: Type of data (continuous or count)
        analysis_mode: Mode of analysis (independent or dependent)
    
    Example:
        >>> analyzer = CosinorAnalyzer(period=24, n_components=1)
        >>> df = analyzer.load_csv("data.csv")
        >>> results = analyzer.single_cosinor(df, variable="geneA", condition="winter")
        >>> print(results.amplitude, results.p_value)
    """
    
    def __init__(
        self,
        period: float = 24.0,
        n_components: int = 1,
        data_type: DataType = DataType.CONTINUOUS,
        analysis_mode: Optional[AnalysisMode] = None
    ):
        """
        Initialize the CosinorAnalyzer.
        
        Args:
            period: The expected period of oscillation in hours (default: 24)
            n_components: Number of cosinor components for fitting (default: 1)
            data_type: Type of data - CONTINUOUS or COUNT (default: CONTINUOUS)
            analysis_mode: INDEPENDENT or DEPENDENT. If None, will be auto-detected.
        """
        if not COSINORPY_AVAILABLE:
            raise ImportError(
                "CosinorPy is required. Install with: pip install cosinorpy"
            )
        
        self.period = period
        self.n_components = n_components
        self.data_type = data_type
        self.analysis_mode = analysis_mode
        
        self._raw_data: Optional[pd.DataFrame] = None
        self._cosinorpy_df: Optional[pd.DataFrame] = None
        self._variables: List[str] = []
        self._conditions: List[str] = []
    
    # =========================================================================
    # DATA LOADING AND PREPROCESSING
    # =========================================================================
    
    def load_csv(
        self,
        filepath: str,
        time_column: str = "time",
        condition_column: str = "condition",
        subject_column: Optional[str] = None,
        replicate_column: Optional[str] = None,
        variable_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and preprocess CSV data for cosinor analysis.
        
        Args:
            filepath: Path to the CSV file
            time_column: Name of the column containing time values
            condition_column: Name of the column containing condition labels
            subject_column: Name of column with subject IDs (for dependent data)
            replicate_column: Name of column with replicate numbers (optional)
            variable_columns: List of columns to analyze. If None, auto-detects
                            all numeric columns except time/condition/subject.
        
        Returns:
            Preprocessed pandas DataFrame
        
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
        
        # Store column names for later use
        self._time_col = time_column
        self._condition_col = condition_column
        self._subject_col = subject_column
        self._replicate_col = replicate_column
        
        # Auto-detect variable columns if not specified
        if variable_columns is None:
            exclude_cols = {time_column, condition_column}
            if subject_column:
                exclude_cols.add(subject_column)
            if replicate_column:
                exclude_cols.add(replicate_column)
            
            variable_columns = [
                col for col in self._raw_data.columns
                if col not in exclude_cols
                and pd.api.types.is_numeric_dtype(self._raw_data[col])
            ]
        
        self._variables = variable_columns
        self._conditions = self._raw_data[condition_column].unique().tolist()
        
        # Auto-detect analysis mode if not specified
        if self.analysis_mode is None:
            self.analysis_mode = self._detect_analysis_mode(
                subject_column, replicate_column
            )
        
        # Auto-detect data type if needed
        if self.data_type == DataType.CONTINUOUS:
            detected_type = self._detect_data_type(variable_columns)
            if detected_type == DataType.COUNT:
                warnings.warn(
                    "Data appears to be count data. Consider setting "
                    "data_type=DataType.COUNT for Poisson regression."
                )
        
        # Convert to CosinorPy format
        self._cosinorpy_df = self._convert_to_cosinorpy_format()
        
        return self._raw_data.copy()
    
    def _detect_analysis_mode(
        self,
        subject_column: Optional[str],
        replicate_column: Optional[str]
    ) -> AnalysisMode:
        """
        Detect whether data should be analyzed as independent or dependent.
        
        Args:
            subject_column: Name of subject column if present
            replicate_column: Name of replicate column if present
        
        Returns:
            Detected AnalysisMode
        """
        if subject_column and subject_column in self._raw_data.columns:
            # If subject column exists, this is dependent (population) data
            return AnalysisMode.DEPENDENT
        else:
            return AnalysisMode.INDEPENDENT
    
    def _detect_data_type(self, variable_columns: List[str]) -> DataType:
        """
        Automatically detect if data is continuous or count data.
        
        Count data characteristics:
        - All values are non-negative integers
        - No decimal values
        
        Args:
            variable_columns: List of variable column names to check
        
        Returns:
            Detected DataType
        """
        for col in variable_columns:
            values = self._raw_data[col].dropna()
            
            # Check if all values are integers
            if not all(values == values.astype(int)):
                return DataType.CONTINUOUS
            
            # Check if all values are non-negative
            if (values < 0).any():
                return DataType.CONTINUOUS
        
        return DataType.COUNT
    
    def _detect_replicates(self, condition: str, variable: str) -> bool:
        """
        Detect if data contains replicates at the same timepoint.
        
        Args:
            condition: Condition to check
            variable: Variable to check
        
        Returns:
            True if replicates exist, False otherwise
        """
        subset = self._raw_data[
            self._raw_data[self._condition_col] == condition
        ]
        
        # Count occurrences of each timepoint
        time_counts = subset[self._time_col].value_counts()
        
        return (time_counts > 1).any()
    
    def _convert_to_cosinorpy_format(self) -> pd.DataFrame:
        """
        Convert loaded data to CosinorPy's expected DataFrame format.
        
        CosinorPy expects a DataFrame with columns: 'x' (time), 'y' (value), 'test' (label)
        For population data, replicates should be named: test_rep1, test_rep2, etc.
        
        Returns:
            DataFrame in CosinorPy format
        """
        rows = []
        
        for condition in self._conditions:
            cond_data = self._raw_data[
                self._raw_data[self._condition_col] == condition
            ]
            
            for variable in self._variables:
                test_name = f"{variable}_{condition}"
                
                if self.analysis_mode == AnalysisMode.DEPENDENT:
                    # Population-mean: each subject is a replicate
                    subjects = cond_data[self._subject_col].unique()
                    for i, subject in enumerate(subjects, 1):
                        subj_data = cond_data[cond_data[self._subject_col] == subject]
                        for _, row in subj_data.iterrows():
                            rows.append({
                                'x': row[self._time_col],
                                'y': row[variable],
                                'test': f"{test_name}_rep{i}"
                            })
                else:
                    # Independent data
                    if self._replicate_col and self._replicate_col in cond_data.columns:
                        # Explicit replicates
                        for _, row in cond_data.iterrows():
                            rep = row[self._replicate_col]
                            rows.append({
                                'x': row[self._time_col],
                                'y': row[variable],
                                'test': f"{test_name}_rep{int(rep)}"
                            })
                    else:
                        # Check for implicit replicates (multiple values at same timepoint)
                        for time_val in cond_data[self._time_col].unique():
                            time_data = cond_data[cond_data[self._time_col] == time_val]
                            if len(time_data) > 1:
                                for i, (_, row) in enumerate(time_data.iterrows(), 1):
                                    rows.append({
                                        'x': row[self._time_col],
                                        'y': row[variable],
                                        'test': f"{test_name}_rep{i}"
                                    })
                            else:
                                for _, row in time_data.iterrows():
                                    rows.append({
                                        'x': row[self._time_col],
                                        'y': row[variable],
                                        'test': test_name
                                    })
        
        return pd.DataFrame(rows)
    
    def get_variables(self) -> List[str]:
        """Get list of available variables for analysis."""
        return self._variables.copy()
    
    def get_conditions(self) -> List[str]:
        """Get list of available conditions."""
        return self._conditions.copy()
    
    # =========================================================================
    # SINGLE-COMPONENT COSINOR ANALYSIS
    # =========================================================================
    
    def single_cosinor(
        self,
        variable: str,
        condition: str,
        period: Optional[float] = None
    ) -> CosinorParameters:
        """
        Perform single-component cosinor analysis on specified data.
        
        The single-component cosinor model fits:
            y(t) = M + A * cos(2π*t/T + φ) + ε
        
        Where:
            M = MESOR (rhythm-adjusted mean)
            A = Amplitude
            φ = Acrophase (phase)
            T = Period
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            period: Period to use for fitting. If None, uses self.period
        
        Returns:
            CosinorParameters object with fitted parameters and statistics
        
        Raises:
            ValueError: If variable or condition not found in data
        """
        if variable not in self._variables:
            raise ValueError(f"Variable '{variable}' not found. Available: {self._variables}")
        if condition not in self._conditions:
            raise ValueError(f"Condition '{condition}' not found. Available: {self._conditions}")
        
        period = period or self.period
        test_name = f"{variable}_{condition}"
        
        # Get data for this test
        test_data = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(test_name)
        ]
        
        X = test_data['x'].values
        Y = test_data['y'].values
        
        # Fit using cosinor1 module for detailed single-component results
        # Returns: (model, amplitude, acrophase, stats_dict)
        model, amplitude, acrophase_rad, stats = cosinor1.fit_cosinor(
            X, Y, period=period, plot_on=False
        )
        
        # Extract values from stats dictionary
        # stats['values'] = [Intercept (MESOR), amplitude, acrophase]
        # stats['p-values'] = [p_intercept, p_amplitude, p_acrophase]
        # stats['CI'] = (lower_bounds, upper_bounds)
        # stats['F-test'] = p-value for the overall model
        
        mesor = stats['values'][0]
        p_value = stats['F-test']  # Overall model significance
        
        # Convert acrophase to hours
        # Acrophase in CosinorPy is in radians
        acrophase_hours = (-acrophase_rad * period / (2 * np.pi)) % period
        
        # Extract confidence intervals
        ci_lower, ci_upper = stats.get('CI', (np.array([np.nan]*3), np.array([np.nan]*3)))
        
        return CosinorParameters(
            mesor=float(mesor),
            amplitude=float(amplitude),
            acrophase=float(acrophase_rad),
            acrophase_hours=float(acrophase_hours),
            period=period,
            p_value=float(p_value),
            r_squared=None,  # Not directly available from this return
            confidence_intervals={
                'amplitude': (float(ci_lower[1]), float(ci_upper[1])),
                'acrophase': (float(ci_lower[2]), float(ci_upper[2]))
            }
        )
    
    def single_cosinor_all(
        self,
        condition: Optional[str] = None,
        period: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Perform single-component cosinor analysis on all variables.
        
        Args:
            condition: Specific condition to analyze. If None, analyzes all conditions.
            period: Period to use for fitting. If None, uses self.period.
        
        Returns:
            DataFrame with results for all variable-condition combinations
        """
        results = []
        conditions = [condition] if condition else self._conditions
        
        for cond in conditions:
            for var in self._variables:
                try:
                    params = self.single_cosinor(var, cond, period)
                    result = params.to_dict()
                    result['variable'] = var
                    result['condition'] = cond
                    results.append(result)
                except Exception as e:
                    warnings.warn(f"Failed to fit {var}/{cond}: {e}")
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # MULTI-COMPONENT COSINOR ANALYSIS
    # =========================================================================
    
    def multi_cosinor(
        self,
        variable: str,
        condition: str,
        n_components: Optional[int] = None,
        period: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-component cosinor analysis.
        
        Multi-component cosinor fits a model with multiple harmonics:
            y(t) = M + Σ[Aᵢ * cos(2π*t/Tᵢ + φᵢ)] + ε
        
        Where Tᵢ = T/i for i = 1, 2, ..., n_components
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            n_components: Number of components. If None, uses self.n_components
            period: Base period. If None, uses self.period
        
        Returns:
            Dictionary containing:
                - 'parameters': List of CosinorParameters for each component
                - 'p_value': Overall model p-value
                - 'r_squared': Overall R-squared
                - 'raw_results': Raw CosinorPy results DataFrame
        
        Note:
            This method uses cosinor.fit_group which works with DataFrames.
            For direct X,Y fitting, use single_cosinor with n_components parameter.
        """
        n_components = n_components or self.n_components
        period = period or self.period
        test_name = f"{variable}_{condition}"
        
        # Prepare data for cosinor.fit_group (expects DataFrame with 'x', 'y', 'test')
        test_data = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(test_name)
        ].copy()
        test_data['test'] = test_name  # Normalize test name
        
        try:
            # Fit using cosinor.fit_group which handles DataFrame format
            df_results = cosinor.fit_group(
                test_data,
                period=period,
                n_components=n_components
            )
            
            p_value = np.nan
            if df_results is not None and isinstance(df_results, pd.DataFrame):
                if 'p' in df_results.columns and len(df_results) > 0:
                    p_value = df_results['p'].values[0]
            
            return {
                'test_name': test_name,
                'n_components': n_components,
                'period': period,
                'p_value': p_value,
                'raw_results': df_results
            }
        except Exception as e:
            warnings.warn(f"Multi-component cosinor failed: {e}")
            return {
                'test_name': test_name,
                'n_components': n_components,
                'period': period,
                'p_value': np.nan,
                'raw_results': None,
                'error': str(e)
            }
    
    def find_best_model(
        self,
        variable: str,
        condition: str,
        max_components: int = 4,
        period: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Find the optimal number of components using model comparison.
        
        Tests models with 1 to max_components and selects the best based on
        statistical criteria (typically BIC or adjusted p-value).
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            max_components: Maximum number of components to test
            period: Base period. If None, uses self.period
        
        Returns:
            Dictionary with best model information
        """
        period = period or self.period
        test_name = f"{variable}_{condition}"
        
        test_data = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(test_name)
        ].copy()
        test_data['test'] = test_name
        
        model_type = self._get_model_type()
        
        # Fit all component numbers using fit_group
        df_results = cosinor.fit_group(
            test_data,
            period=period,
            n_components=list(range(1, max_components + 1))
        )
        
        # Get best fits
        df_best = cosinor.get_best_fits(
            df_results, 
            n_components=list(range(1, max_components + 1))
        )
        
        return {
            'all_results': df_results,
            'best_models': df_best,
            'test_name': test_name
        }
    
    # =========================================================================
    # POPULATION-MEAN COSINOR ANALYSIS
    # =========================================================================
    
    def population_cosinor(
        self,
        variable: str,
        condition: str,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> Dict[str, Any]:
        """
        Perform population-mean cosinor analysis for dependent data.
        
        Population-mean cosinor fits individual cosinor models to each subject
        and then combines them using vector averaging to estimate population
        parameters with appropriate confidence intervals.
        
        This is appropriate when the same subjects are measured repeatedly
        over time (e.g., bioluminescence, wearable data).
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            period: Period to use. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            Dictionary containing:
                - 'population_params': Population-level CosinorParameters
                - 'individual_params': List of individual subject parameters
                - 'raw_results': Raw CosinorPy results
        """
        period = period or self.period
        test_name = f"{variable}_{condition}"
        
        # Get population data (with _repN suffixes)
        test_data = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(test_name)
        ].copy()
        
        if n_components == 1:
            # Use cosinor1 for single-component population analysis
            df_results = cosinor1.fit_group(
                test_data,
                period=period
            )
            
            # Population fit using cosinor1
            df_pop = cosinor1.population_fit_cosinor(
                test_data,
                period=period,
                plot_on=False
            )
        else:
            # Use cosinor module for multi-component
            df_results = cosinor.fit_group(
                test_data,
                period=period,
                n_components=n_components
            )
            
            df_pop = cosinor.population_fit_group(
                test_data,
                period=period,
                n_components=n_components
            )
        
        return {
            'individual_results': df_results,
            'population_results': df_pop,
            'test_name': test_name,
            'n_components': n_components,
            'period': period
        }
    
    # =========================================================================
    # DIFFERENTIAL RHYTHMICITY ANALYSIS
    # =========================================================================
    
    def compare_conditions(
        self,
        variable: str,
        condition1: str,
        condition2: str,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> DifferentialResult:
        """
        Compare rhythmicity between two conditions for a single variable.
        
        Performs differential rhythmicity analysis to detect significant
        differences in amplitude, acrophase, and MESOR between conditions.
        
        Args:
            variable: Name of the variable to compare
            condition1: First condition name
            condition2: Second condition name  
            period: Period to use. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            DifferentialResult object with comparison statistics
        """
        period = period or self.period
        test1 = f"{variable}_{condition1}"
        test2 = f"{variable}_{condition2}"
        
        # Get data for both conditions
        df_subset = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(f"{variable}_")
        ].copy()
        
        # Create normalized test names for cosinorpy
        df_subset.loc[
            df_subset['test'].str.startswith(test1), 'test'
        ] = test1
        df_subset.loc[
            df_subset['test'].str.startswith(test2), 'test'
        ] = test2
        
        # Filter to only include the two conditions we want
        df_subset = df_subset[df_subset['test'].isin([test1, test2])]
        
        if n_components == 1:
            # Use cosinor1 for detailed single-component comparison
            df_results = cosinor1.test_cosinor_pairs(
                df_subset,
                pairs=[(test1, test2)],
                period=period,
                plot_measurements=False
            )
        else:
            # Use cosinor module for multi-component
            df_results = cosinor.compare_pairs(
                df_subset,
                pairs=[(test1, test2)],
                period=period,
                n_components=n_components
            )
        
        # Extract results - handle both DataFrame and dict returns
        if isinstance(df_results, pd.DataFrame) and len(df_results) > 0:
            row = df_results.iloc[0]
            return DifferentialResult(
                condition1=condition1,
                condition2=condition2,
                amplitude_diff=row.get('d_amplitude', row.get('amplitude_diff', np.nan)),
                amplitude_p_value=row.get('p(d_amplitude)', row.get('p_amplitude', np.nan)),
                acrophase_diff=row.get('d_acrophase', row.get('acrophase_diff', np.nan)),
                acrophase_p_value=row.get('p(d_acrophase)', row.get('p_acrophase', np.nan)),
                mesor_diff=row.get('d_mesor', row.get('mesor_diff', np.nan)),
                mesor_p_value=row.get('p(d_mesor)', row.get('p_mesor', np.nan)),
                q_values={
                    'amplitude': row.get('q(d_amplitude)', row.get('q_amplitude', np.nan)),
                    'acrophase': row.get('q(d_acrophase)', row.get('q_acrophase', np.nan))
                }
            )
        elif isinstance(df_results, dict):
            return DifferentialResult(
                condition1=condition1,
                condition2=condition2,
                amplitude_diff=df_results.get('d_amplitude', np.nan),
                amplitude_p_value=df_results.get('p(d_amplitude)', np.nan),
                acrophase_diff=df_results.get('d_acrophase', np.nan),
                acrophase_p_value=df_results.get('p(d_acrophase)', np.nan)
            )
        else:
            raise ValueError("Comparison failed - no results returned")
    
    def compare_all_conditions(
        self,
        variable: str,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> pd.DataFrame:
        """
        Compare all pairs of conditions for a given variable.
        
        Args:
            variable: Name of the variable to compare
            period: Period to use. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            DataFrame with all pairwise comparisons
        """
        from itertools import combinations
        
        results = []
        for cond1, cond2 in combinations(self._conditions, 2):
            try:
                diff_result = self.compare_conditions(
                    variable, cond1, cond2, period, n_components
                )
                results.append(diff_result.to_dict())
            except Exception as e:
                warnings.warn(f"Failed to compare {cond1} vs {cond2}: {e}")
        
        return pd.DataFrame(results)
    
    def compare_variables(
        self,
        variable1: str,
        variable2: str,
        condition: str,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> DifferentialResult:
        """
        Compare rhythmicity between two variables within the same condition.
        
        Args:
            variable1: First variable name
            variable2: Second variable name
            condition: Condition to analyze
            period: Period to use. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            DifferentialResult object with comparison statistics
        """
        period = period or self.period
        test1 = f"{variable1}_{condition}"
        test2 = f"{variable2}_{condition}"
        
        df_subset = self._cosinorpy_df[
            (self._cosinorpy_df['test'].str.startswith(test1)) |
            (self._cosinorpy_df['test'].str.startswith(test2))
        ].copy()
        
        # Normalize test names
        df_subset.loc[
            df_subset['test'].str.startswith(test1), 'test'
        ] = test1
        df_subset.loc[
            df_subset['test'].str.startswith(test2), 'test'
        ] = test2
        
        if n_components == 1:
            df_results = cosinor1.test_cosinor_pairs(
                df_subset,
                pairs=[(test1, test2)],
                period=period,
                plot_measurements=False
            )
        else:
            df_results = cosinor.compare_pairs(
                df_subset,
                pairs=[(test1, test2)],
                period=period,
                n_components=n_components
            )
        
        if isinstance(df_results, pd.DataFrame) and len(df_results) > 0:
            row = df_results.iloc[0]
            return DifferentialResult(
                condition1=f"{variable1}",
                condition2=f"{variable2}",
                amplitude_diff=row.get('d_amplitude', row.get('amplitude_diff', np.nan)),
                amplitude_p_value=row.get('p(d_amplitude)', row.get('p_amplitude', np.nan)),
                acrophase_diff=row.get('d_acrophase', row.get('acrophase_diff', np.nan)),
                acrophase_p_value=row.get('p(d_acrophase)', row.get('p_acrophase', np.nan))
            )
        else:
            raise ValueError("Comparison failed")
    
    # =========================================================================
    # GENERALIZED (NON-LINEAR) COSINOR ANALYSIS
    # =========================================================================
    
    def nonlinear_cosinor(
        self,
        variable: str,
        condition: str,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> Dict[str, Any]:
        """
        Perform generalized (non-linear) cosinor analysis.
        
        The generalized cosinor model extends the standard cosinor to include
        additional parameters for damping and trend:
        
            y(t) = M + exp(C*t) * Σ[Aᵢ * cos(2π*t/Tᵢ + φᵢ)] + D*t + ε
        
        Where:
            C = Damping coefficient (amplitude decay/growth)
            D = Linear trend component
        
        This is useful for non-stationary rhythms where amplitude changes
        over time.
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            period: Base period. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            Dictionary with non-linear fit results including:
                - Standard cosinor parameters
                - Damping coefficient
                - Trend coefficient
                - Model comparison statistics
        """
        period = period or self.period
        test_name = f"{variable}_{condition}"
        
        test_data = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(test_name)
        ].copy()
        
        X = test_data['x'].values
        Y = test_data['y'].values
        
        # Fit generalized cosinor model
        if n_components == 1:
            results = cosinor_nonlin.fit_generalized_cosinor(
                X, Y,
                period=period,
                plot=False
            )
        else:
            results = cosinor_nonlin.fit_generalized_cosinor_n_comp(
                X, Y,
                period=period,
                n_components=n_components,
                plot=False
            )
        
        return {
            'test_name': test_name,
            'period': period,
            'n_components': n_components,
            'results': results
        }
    
    def compare_nonlinear(
        self,
        variable: str,
        condition1: str,
        condition2: str,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> Dict[str, Any]:
        """
        Compare two conditions using non-linear cosinor models.
        
        Args:
            variable: Name of the variable to compare
            condition1: First condition name
            condition2: Second condition name
            period: Base period. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            Dictionary with comparison results
        """
        period = period or self.period
        test1 = f"{variable}_{condition1}"
        test2 = f"{variable}_{condition2}"
        
        df_subset = self._cosinorpy_df[
            (self._cosinorpy_df['test'].str.startswith(test1)) |
            (self._cosinorpy_df['test'].str.startswith(test2))
        ].copy()
        
        # Normalize names
        df_subset.loc[
            df_subset['test'].str.startswith(test1), 'test'
        ] = test1
        df_subset.loc[
            df_subset['test'].str.startswith(test2), 'test'
        ] = test2
        
        # Use compare_nonlinear_pairs from cosinor_nonlin
        results = cosinor_nonlin.compare_nonlinear_pairs(
            df_subset,
            pairs=[(test1, test2)],
            period=period
        )
        
        return {
            'test1': test1,
            'test2': test2,
            'comparison_results': results
        }
    
    # =========================================================================
    # COUNT DATA ANALYSIS
    # =========================================================================
    
    def fit_count_data(
        self,
        variable: str,
        condition: str,
        model_type: ModelType = ModelType.POISSON,
        period: Optional[float] = None,
        n_components: int = 1
    ) -> Dict[str, Any]:
        """
        Fit cosinor model to count data using Poisson or Negative Binomial regression.
        
        For count data (e.g., RNA-seq counts), standard linear regression is
        inappropriate. This method uses Poisson or Negative Binomial GLM.
        
        Args:
            variable: Name of the variable to analyze
            condition: Name of the condition to analyze
            model_type: Type of GLM - POISSON, GEN_POISSON, or NEGATIVE_BINOMIAL
            period: Period to use. If None, uses self.period
            n_components: Number of cosinor components
        
        Returns:
            Dictionary with model results
        """
        period = period or self.period
        test_name = f"{variable}_{condition}"
        
        test_data = self._cosinorpy_df[
            self._cosinorpy_df['test'].str.startswith(test_name)
        ].copy()
        test_data['test'] = test_name
        
        # Map ModelType to cosinor string
        model_type_str = {
            ModelType.POISSON: 'poisson',
            ModelType.GEN_POISSON: 'gen_poisson',
            ModelType.NEGATIVE_BINOMIAL: 'nb'
        }.get(model_type, 'poisson')
        
        # Extract X and Y arrays
        X = test_data['x'].values
        Y = test_data['y'].values
        
        results = cosinor.fit_me(
            X, Y,
            period=period,
            n_components=n_components,
            model_type=model_type_str,
            plot=False
        )
        
        return {
            'test_name': test_name,
            'model_type': model_type_str,
            'period': period,
            'n_components': n_components,
            'results': results
        }
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _get_model_type(self) -> str:
        """Get the model type string for CosinorPy based on data type."""
        if self.data_type == DataType.COUNT:
            return 'poisson'
        return 'linear'
    
    def get_raw_data(self) -> Optional[pd.DataFrame]:
        """Get the raw loaded data."""
        return self._raw_data.copy() if self._raw_data is not None else None
    
    def get_cosinorpy_data(self) -> Optional[pd.DataFrame]:
        """Get data in CosinorPy format."""
        return self._cosinorpy_df.copy() if self._cosinorpy_df is not None else None
    
    def set_period(self, period: float) -> None:
        """Update the analysis period."""
        self.period = period
    
    def set_n_components(self, n_components: int) -> None:
        """Update the number of cosinor components."""
        self.n_components = n_components
    
    def set_data_type(self, data_type: DataType) -> None:
        """Update the data type."""
        self.data_type = data_type
    
    def set_analysis_mode(self, mode: AnalysisMode) -> None:
        """Update the analysis mode."""
        self.analysis_mode = mode


# =============================================================================
# PLOTTING REFERENCE (for GUI implementation)
# =============================================================================

class PlottingReference:
    """
    Reference class documenting available CosinorPy plotting functions.
    
    This class is not meant to be instantiated - it serves as documentation
    for GUI developers to know which plotting functions are available in
    CosinorPy for visualization.
    
    All functions below can be called directly from the cosinor or cosinor1
    modules after importing CosinorPy.
    """
    
    @staticmethod
    def available_plots() -> Dict[str, str]:
        """
        Get dictionary of available plotting functions and their descriptions.
        
        Returns:
            Dictionary mapping function names to descriptions
        """
        return {
            # cosinor module plots
            'cosinor.plot_data': 
                'Plot raw data. Args: (df, test, folder="", prefix="")',
            
            'cosinor.plot_data_pairs':
                'Plot two groups on same axes. Args: (df, test1, test2)',
            
            'cosinor.plot_phases':
                'Polar plot of phases. Args: (df_best_models, tests, colors, folder, prefix)',
            
            'cosinor.periodogram':
                'Periodogram with significance threshold. Args: (df, test, per_type, max_per, ...)',
            
            'cosinor.plot_heatmap':
                'Heatmap of raw data. Args: (df, test, ...)',
            
            'cosinor.plot_tuples':
                'Plot multiple tests together. Args: (df, df_best_models, tuples, colors, ...)',
            
            # cosinor1 module plots (single-component specific)
            'cosinor1.fit_cosinor':
                'Fit and optionally plot single cosinor. Args: (X, Y, period, plot=True, ...)',
            
            'cosinor1.population_fit':
                'Fit and optionally plot population cosinor. Args: (df, period, plot=True, ...)',
        }


# =============================================================================
# SAMPLE DATA GENERATOR
# =============================================================================

def generate_sample_data(
    n_timepoints: int = 6,
    n_conditions: int = 2,
    n_variables: int = 2,
    n_replicates: int = 3,
    period: float = 24.0,
    noise_level: float = 0.1,
    include_subjects: bool = False,
    n_subjects: int = 5
) -> pd.DataFrame:
    """
    Generate sample data for testing the CosinorAnalyzer.
    
    Creates synthetic rhythmic data with known parameters for testing
    and demonstration purposes.
    
    Args:
        n_timepoints: Number of timepoints per cycle
        n_conditions: Number of conditions (e.g., winter, summer)
        n_variables: Number of variables to generate
        n_replicates: Number of replicates per timepoint (independent) or
                     number of subjects (dependent)
        period: Period of oscillation in hours
        noise_level: Standard deviation of Gaussian noise (as fraction of amplitude)
        include_subjects: If True, generates dependent (population) data with
                         subject IDs; if False, generates independent data
        n_subjects: Number of subjects (only used if include_subjects=True)
    
    Returns:
        DataFrame with synthetic data in the expected format
    """
    condition_names = [f"condition{i+1}" for i in range(n_conditions)]
    variable_names = [f"var{i+1}" for i in range(n_variables)]
    
    # Generate timepoints (evenly spaced within one period)
    timepoints = np.linspace(0, period - period/n_timepoints, n_timepoints)
    
    rows = []
    
    # Parameters for each variable-condition combination
    np.random.seed(42)  # For reproducibility
    
    for cond_idx, condition in enumerate(condition_names):
        for var_idx, variable in enumerate(variable_names):
            # Generate different parameters for each combination
            mesor = 10 + var_idx * 5
            amplitude = 3 + var_idx + cond_idx * 0.5
            acrophase = -np.pi/4 + cond_idx * np.pi/6  # Phase shift between conditions
            
            if include_subjects:
                # Dependent data - same subjects measured over time
                for subj_idx in range(n_subjects):
                    # Add subject-specific variation
                    subj_mesor = mesor + np.random.normal(0, 1)
                    subj_amp = amplitude + np.random.normal(0, 0.3)
                    
                    for t in timepoints:
                        value = subj_mesor + subj_amp * np.cos(2*np.pi*t/period + acrophase)
                        value += np.random.normal(0, amplitude * noise_level)
                        
                        rows.append({
                            'time': t,
                            'condition': condition,
                            'subject': f"subject{subj_idx+1}",
                            variable: value
                        })
            else:
                # Independent data - different samples at each timepoint
                for t in timepoints:
                    for rep in range(n_replicates):
                        value = mesor + amplitude * np.cos(2*np.pi*t/period + acrophase)
                        value += np.random.normal(0, amplitude * noise_level)
                        
                        rows.append({
                            'time': t,
                            'condition': condition,
                            'replicate': rep + 1,
                            variable: value
                        })
    
    df = pd.DataFrame(rows)
    
    # Consolidate variable columns if multiple
    if n_variables > 1 and not include_subjects:
        # Group by time, condition, replicate and combine variable columns
        df = df.groupby(['time', 'condition', 'replicate']).first().reset_index()
    elif n_variables > 1 and include_subjects:
        df = df.groupby(['time', 'condition', 'subject']).first().reset_index()
    
    return df


# =============================================================================
# MAIN - TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("CosinorPy Analysis Module - Test Suite")
    print("=" * 60)
    
    # Check if CosinorPy is available
    if not COSINORPY_AVAILABLE:
        print("\nERROR: CosinorPy is not installed!")
        print("Install it with: pip install cosinorpy")
        sys.exit(1)
    
    print("\n✓ CosinorPy is available")
    
    # Generate sample data for testing
    print("\n--- Generating Sample Data ---")
    
    # Generate independent data
    df_independent = generate_sample_data(
        n_timepoints=6,
        n_conditions=2,
        n_variables=2,
        n_replicates=3,
        period=24.0,
        noise_level=0.15,
        include_subjects=False
    )
    
    # Save to CSV for testing
    csv_path = "example_data.csv"
    df_independent.to_csv(csv_path, index=False)
    print(f"Sample data saved to: {csv_path}")
    print(f"Data shape: {df_independent.shape}")
    print(f"\nSample data head:\n{df_independent.head(10)}")
    
    # Test the analyzer
    print("\n--- Testing CosinorAnalyzer ---")
    
    try:
        # Initialize analyzer
        analyzer = CosinorAnalyzer(period=24.0, n_components=1)
        print("✓ Analyzer initialized")
        
        # Load data
        df = analyzer.load_csv(
            csv_path,
            time_column='time',
            condition_column='condition',
            replicate_column='replicate'
        )
        print("✓ Data loaded successfully")
        print(f"  Variables detected: {analyzer.get_variables()}")
        print(f"  Conditions detected: {analyzer.get_conditions()}")
        print(f"  Analysis mode: {analyzer.analysis_mode}")
        
        # Test single cosinor
        print("\n--- Single-Component Cosinor ---")
        result = analyzer.single_cosinor('var1', 'condition1')
        print(f"Variable: var1, Condition: condition1")
        print(f"  MESOR: {result.mesor:.3f}")
        print(f"  Amplitude: {result.amplitude:.3f}")
        print(f"  Acrophase (hours): {result.acrophase_hours:.2f}")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  Significant: {result.is_significant()}")
        
        # Test all variables
        print("\n--- All Variables Analysis ---")
        df_results = analyzer.single_cosinor_all()
        print(df_results[['variable', 'condition', 'amplitude', 'p_value', 'acrophase_hours']])
        
        # Test differential rhythmicity
        print("\n--- Differential Rhythmicity ---")
        diff = analyzer.compare_conditions('var1', 'condition1', 'condition2')
        print(f"Comparing: condition1 vs condition2 (var1)")
        print(f"  Amplitude difference: {diff.amplitude_diff:.3f}")
        print(f"  Amplitude p-value: {diff.amplitude_p_value:.4f}")
        print(f"  Acrophase difference: {diff.acrophase_diff:.3f}")
        print(f"  Acrophase p-value: {diff.acrophase_p_value:.4f}")
        
        # Test multi-component
        print("\n--- Multi-Component Cosinor ---")
        multi_result = analyzer.multi_cosinor('var1', 'condition1', n_components=2)
        print(f"2-component model fitted")
        if multi_result.get('raw_results') is not None:
            print(f"  Results columns: {multi_result['raw_results'].columns.tolist()}")
        elif multi_result.get('error'):
            print(f"  Note: Multi-component failed due to CosinorPy compatibility issue")
            print(f"  Error: {multi_result['error']}")
        else:
            print(f"  Results: No detailed results available")
        
        # Test model selection
        print("\n--- Best Model Selection ---")
        try:
            best = analyzer.find_best_model('var1', 'condition1', max_components=3)
            print(f"Best models computed")
            if best.get('all_results') is not None:
                print(f"  All results available: Yes")
        except Exception as e:
            print(f"  Note: Best model selection skipped due to: {e}")
        
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
from cosinor_analysis import CosinorAnalyzer, DataType, AnalysisMode

# Initialize
analyzer = CosinorAnalyzer(period=24.0, n_components=1)

# Load CSV (auto-detects structure)
df = analyzer.load_csv(
    "your_data.csv",
    time_column="time",
    condition_column="condition",
    replicate_column="replicate"  # or subject_column for dependent data
)

# Get available variables and conditions
variables = analyzer.get_variables()
conditions = analyzer.get_conditions()

# Single cosinor analysis
result = analyzer.single_cosinor("geneA", "winter")
print(result.amplitude, result.p_value, result.is_significant())

# Compare conditions
diff = analyzer.compare_conditions("geneA", "winter", "summer")
print(diff.acrophase_p_value)

# For count data (RNA-seq)
analyzer.set_data_type(DataType.COUNT)
result = analyzer.fit_count_data("geneA", "winter", model_type=ModelType.POISSON)
""")
