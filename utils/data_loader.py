"""
Data Loader Module
==================

Handles loading and validation of user CSV data for circadian analysis.
Provides automatic column detection and data structure validation.

Expected CSV Format:
    time,condition,replicate,variable1,variable2,...
    0,control,1,10.2,5.3
    0,control,2,10.5,5.1
    4,control,1,12.1,6.2
    ...
"""

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class DataSourceType(Enum):
    """Enumeration for data source types."""
    USER_CSV = "user_csv"
    ROSBASH_H5 = "rosbash_h5"


class ColumnRole(Enum):
    """Enumeration for column roles in the dataset."""
    TIME = "time"
    CONDITION = "condition"
    REPLICATE = "replicate"
    SUBJECT = "subject"
    VARIABLE = "variable"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a detected column."""
    name: str
    role: ColumnRole
    dtype: str
    n_unique: int
    sample_values: List[Any]
    is_numeric: bool


@dataclass
class DatasetInfo:
    """Summary information about loaded dataset."""
    source_type: DataSourceType
    filepath: str
    n_rows: int
    n_columns: int
    time_column: Optional[str]
    condition_column: Optional[str]
    replicate_column: Optional[str]
    subject_column: Optional[str]
    variable_columns: List[str]
    conditions: List[str]
    timepoints: List[float]
    n_replicates_per_timepoint: Dict[str, int]
    has_missing_values: bool
    column_info: List[ColumnInfo]


class CircadianDataLoader:
    """
    Data loader for circadian rhythm analysis.
    
    Handles loading CSV files, automatic column detection, and data validation.
    """
    
    # Common column name patterns for auto-detection
    TIME_PATTERNS = ['time', 'zeit', 'zt', 'ct', 'hour', 'hours', 't', 'timepoint']
    CONDITION_PATTERNS = ['condition', 'group', 'treatment', 'genotype', 'sample', 'cond']
    REPLICATE_PATTERNS = ['replicate', 'rep', 'replica', 'n', 'repeat']
    SUBJECT_PATTERNS = ['subject', 'subj', 'animal', 'mouse', 'fly', 'individual', 'id']
    
    def __init__(self):
        """Initialize the data loader."""
        self._raw_data: Optional[pd.DataFrame] = None
        self._dataset_info: Optional[DatasetInfo] = None
        self._filepath: Optional[str] = None
        
        # Column mappings (can be overridden by user)
        self._time_col: Optional[str] = None
        self._condition_col: Optional[str] = None
        self._replicate_col: Optional[str] = None
        self._subject_col: Optional[str] = None
        self._variable_cols: List[str] = []
    
    def load_csv(
        self,
        filepath: str,
        separator: str = ',',
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        Load a CSV file and perform initial analysis.
        
        Args:
            filepath: Path to the CSV file
            separator: Column separator
            encoding: File encoding
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        try:
            self._raw_data = pd.read_csv(
                filepath,
                sep=separator,
                encoding=encoding
            )
            self._filepath = filepath
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
        
        # Auto-detect columns
        self._auto_detect_columns()
        
        # Generate dataset info
        self._generate_dataset_info()
        
        return self._raw_data.copy()
    
    def load_dataframe(self, df: pd.DataFrame, name: str = "dataframe") -> pd.DataFrame:
        """
        Load data from an existing DataFrame.
        
        Args:
            df: pandas DataFrame
            name: Name identifier for the dataset
        
        Returns:
            Copy of the DataFrame
        """
        self._raw_data = df.copy()
        self._filepath = name
        
        self._auto_detect_columns()
        self._generate_dataset_info()
        
        return self._raw_data.copy()
    
    def _auto_detect_columns(self) -> None:
        """Automatically detect column roles based on names and content."""
        if self._raw_data is None:
            return
        
        columns = self._raw_data.columns.tolist()
        
        # Reset detections
        self._time_col = None
        self._condition_col = None
        self._replicate_col = None
        self._subject_col = None
        self._variable_cols = []
        
        for col in columns:
            col_lower = col.lower().strip()
            
            # Check for time column
            if any(pattern in col_lower for pattern in self.TIME_PATTERNS):
                if self._time_col is None and pd.api.types.is_numeric_dtype(self._raw_data[col]):
                    self._time_col = col
                    continue
            
            # Check for condition column
            if any(pattern in col_lower for pattern in self.CONDITION_PATTERNS):
                if self._condition_col is None:
                    self._condition_col = col
                    continue
            
            # Check for replicate column
            if any(pattern in col_lower for pattern in self.REPLICATE_PATTERNS):
                if self._replicate_col is None:
                    self._replicate_col = col
                    continue
            
            # Check for subject column
            if any(pattern in col_lower for pattern in self.SUBJECT_PATTERNS):
                if self._subject_col is None:
                    self._subject_col = col
                    continue
        
        # All remaining numeric columns are considered variables
        assigned_cols = {
            self._time_col, self._condition_col,
            self._replicate_col, self._subject_col
        }
        
        for col in columns:
            if col not in assigned_cols:
                if pd.api.types.is_numeric_dtype(self._raw_data[col]):
                    self._variable_cols.append(col)
    
    def _generate_dataset_info(self) -> None:
        """Generate summary information about the dataset."""
        if self._raw_data is None:
            return
        
        df = self._raw_data
        
        # Generate column info
        column_info = []
        for col in df.columns:
            role = self._get_column_role(col)
            info = ColumnInfo(
                name=col,
                role=role,
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique(),
                sample_values=df[col].dropna().head(5).tolist(),
                is_numeric=pd.api.types.is_numeric_dtype(df[col])
            )
            column_info.append(info)
        
        # Get conditions and timepoints
        conditions = []
        if self._condition_col:
            conditions = df[self._condition_col].unique().tolist()
        
        timepoints = []
        if self._time_col:
            timepoints = sorted(df[self._time_col].unique().tolist())
        
        # Count replicates per timepoint
        n_reps = {}
        if self._time_col and self._condition_col:
            for cond in conditions:
                cond_data = df[df[self._condition_col] == cond]
                for t in timepoints:
                    key = f"{cond}_{t}"
                    n_reps[key] = len(cond_data[cond_data[self._time_col] == t])
        
        self._dataset_info = DatasetInfo(
            source_type=DataSourceType.USER_CSV,
            filepath=self._filepath or "",
            n_rows=len(df),
            n_columns=len(df.columns),
            time_column=self._time_col,
            condition_column=self._condition_col,
            replicate_column=self._replicate_col,
            subject_column=self._subject_col,
            variable_columns=self._variable_cols.copy(),
            conditions=conditions,
            timepoints=timepoints,
            n_replicates_per_timepoint=n_reps,
            has_missing_values=df.isnull().any().any(),
            column_info=column_info
        )
    
    def _get_column_role(self, col: str) -> ColumnRole:
        """Get the role of a specific column."""
        if col == self._time_col:
            return ColumnRole.TIME
        elif col == self._condition_col:
            return ColumnRole.CONDITION
        elif col == self._replicate_col:
            return ColumnRole.REPLICATE
        elif col == self._subject_col:
            return ColumnRole.SUBJECT
        elif col in self._variable_cols:
            return ColumnRole.VARIABLE
        else:
            return ColumnRole.UNKNOWN
    
    # =========================================================================
    # COLUMN ASSIGNMENT (for GUI override)
    # =========================================================================
    
    def set_time_column(self, col: str) -> None:
        """Manually set the time column."""
        if col in self._raw_data.columns:
            self._time_col = col
            self._generate_dataset_info()
    
    def set_condition_column(self, col: str) -> None:
        """Manually set the condition column."""
        if col in self._raw_data.columns:
            self._condition_col = col
            self._generate_dataset_info()
    
    def set_replicate_column(self, col: Optional[str]) -> None:
        """Manually set the replicate column (can be None)."""
        if col is None or col in self._raw_data.columns:
            self._replicate_col = col
            self._generate_dataset_info()
    
    def set_subject_column(self, col: Optional[str]) -> None:
        """Manually set the subject column (can be None)."""
        if col is None or col in self._raw_data.columns:
            self._subject_col = col
            self._generate_dataset_info()
    
    def set_variable_columns(self, cols: List[str]) -> None:
        """Manually set the variable columns."""
        valid_cols = [c for c in cols if c in self._raw_data.columns]
        self._variable_cols = valid_cols
        self._generate_dataset_info()
    
    # =========================================================================
    # GETTERS
    # =========================================================================
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the loaded data."""
        return self._raw_data.copy() if self._raw_data is not None else None
    
    def get_dataset_info(self) -> Optional[DatasetInfo]:
        """Get dataset summary information."""
        return self._dataset_info
    
    def get_columns(self) -> List[str]:
        """Get list of all columns."""
        if self._raw_data is None:
            return []
        return self._raw_data.columns.tolist()
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns."""
        if self._raw_data is None:
            return []
        return [
            col for col in self._raw_data.columns
            if pd.api.types.is_numeric_dtype(self._raw_data[col])
        ]
    
    def get_time_column(self) -> Optional[str]:
        """Get the detected/assigned time column."""
        return self._time_col
    
    def get_condition_column(self) -> Optional[str]:
        """Get the detected/assigned condition column."""
        return self._condition_col
    
    def get_variable_columns(self) -> List[str]:
        """Get the detected/assigned variable columns."""
        return self._variable_cols.copy()
    
    def get_conditions(self) -> List[str]:
        """Get list of unique conditions."""
        if self._raw_data is None or self._condition_col is None:
            return []
        return self._raw_data[self._condition_col].unique().tolist()
    
    def get_timepoints(self) -> List[float]:
        """Get list of unique timepoints."""
        if self._raw_data is None or self._time_col is None:
            return []
        return sorted(self._raw_data[self._time_col].unique().tolist())
    
    # =========================================================================
    # DATA EXTRACTION FOR ANALYSIS
    # =========================================================================
    
    def get_data_for_analysis(
        self,
        variable: str,
        condition: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract time and value arrays for analysis.
        
        Args:
            variable: Variable column name
            condition: Optional condition filter
        
        Returns:
            Tuple of (time_array, value_array)
        """
        if self._raw_data is None:
            raise ValueError("No data loaded")
        
        if variable not in self._variable_cols:
            raise ValueError(f"Variable '{variable}' not found")
        
        df = self._raw_data
        
        if condition and self._condition_col:
            df = df[df[self._condition_col] == condition]
        
        times = df[self._time_col].values
        values = df[variable].values
        
        # Remove NaN values
        mask = ~(np.isnan(times) | np.isnan(values))
        
        return times[mask], values[mask]
    
    def get_grouped_data(
        self,
        variable: str,
        condition: Optional[str] = None,
        aggregate: str = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data grouped by timepoint with optional aggregation.
        
        Args:
            variable: Variable column name
            condition: Optional condition filter
            aggregate: Aggregation method ('mean', 'median', 'none')
        
        Returns:
            Tuple of (timepoints, values, sem_or_std)
        """
        if self._raw_data is None:
            raise ValueError("No data loaded")
        
        df = self._raw_data
        
        if condition and self._condition_col:
            df = df[df[self._condition_col] == condition]
        
        if aggregate == 'none':
            return self.get_data_for_analysis(variable, condition) + (None,)
        
        grouped = df.groupby(self._time_col)[variable]
        
        if aggregate == 'mean':
            values = grouped.mean()
            errors = grouped.sem()
        elif aggregate == 'median':
            values = grouped.median()
            errors = grouped.std()
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        return values.index.values, values.values, errors.values
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def validate_for_analysis(self) -> Tuple[bool, List[str]]:
        """
        Validate that data is ready for analysis.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if self._raw_data is None:
            issues.append("No data loaded")
            return False, issues
        
        if self._time_col is None:
            issues.append("Time column not detected/assigned")
        
        if self._condition_col is None:
            issues.append("Condition column not detected/assigned")
        
        if not self._variable_cols:
            issues.append("No variable columns detected/assigned")
        
        # Check for sufficient data points
        if self._time_col and len(self._raw_data[self._time_col].unique()) < 4:
            issues.append("Insufficient timepoints (minimum 4 recommended)")
        
        # Check for missing values in key columns
        if self._time_col and self._raw_data[self._time_col].isnull().any():
            issues.append("Missing values in time column")
        
        if self._condition_col and self._raw_data[self._condition_col].isnull().any():
            issues.append("Missing values in condition column")
        
        return len(issues) == 0, issues
    
    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of the data."""
        if self._raw_data is None:
            return pd.DataFrame()
        return self._raw_data.head(n_rows)
