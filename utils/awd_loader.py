"""
AWD Data Loader Module
======================

Handles loading and processing of Actimetrics ClockLab (.awd)
files for circadian analysis (typically rodents).

AWD File Format:
    ASCII text file.
    Header (first ~5-7 lines): Metadata (ID, Start Date, Start Time, etc.)
    Body: Single column of integers representing wheel revolutions per bin.

Output Format (ChronoScope compatible):
    time,condition,subject,activity
    0.0,WildType,Mouse01,45
    0.1,WildType,Mouse01,52
    ...
"""

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import re

from .data_loader import DatasetInfo, DataSourceType, ColumnInfo, ColumnRole


@dataclass
class AWDConfig:
    """Configuration for AWD data processing."""

    # Binning settings
    # AWD files typically have a native bin size (e.g., 6 or 10 min).
    # target_bin_size_minutes allows re-binning if needed.
    target_bin_size_minutes: Optional[int] = None 

    # Time reference
    lights_on_hour: int = 7  # Hour when lights turn on (ZT0)
    lights_on_minute: int = 0

    # Date range (None = use all data)
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

    # Metadata
    condition_name: str = "Condition"  # Name for the condition column
    subject_id: Optional[str] = None   # Auto-detected from header/filename if None


@dataclass
class AWDSummary:
    """Summary information about a loaded AWD file."""
    filepath: str
    subject_id: str
    native_bin_size_minutes: int
    start_datetime: datetime
    end_datetime: datetime
    total_days: float
    total_rows: int
    total_revolutions: int
    missing_data_bins: int


class AWDDataLoader:
    """
    Loader for individual Actimetrics ClockLab (.awd) files.
    Parses the AWD format and converts it to a ChronoScope-compatible DataFrame.
    """

    def __init__(self):
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._filepath: Optional[str] = None
        self._config: AWDConfig = AWDConfig()
        self._summary: Optional[AWDSummary] = None
        
        self._header_subject_id: str = ""
        self._header_start_dt: Optional[datetime] = None
        self._native_bin_min: int = 6  # Standard fallback for rodent wheels

        # ChronoScope compatibility
        self._time_col: str = "time"
        self._condition_col: str = "condition"
        self._subject_col: str = "subject" 
        self._variable_cols: List[str] = ["activity"]

    def load_awd(self, filepath: str) -> pd.DataFrame:
        """Loads an AWD file from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"AWD file not found: {filepath}")

        self._filepath = filepath

        try:
            # AWD files are sometimes saved with different encodings
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = [line.strip() for line in f.readlines()]
            
            if len(lines) < 10:
                raise ValueError("File is too short to be a valid AWD file")

            # 1. Parse Header
            self._header_subject_id = lines[0] if lines[0] else path.stem
            
            if self._config.subject_id is None:
                self._config.subject_id = self._header_subject_id

            # Parse Start Date and Time (Lines 2 and 3)
            date_str = lines[1]
            time_str = lines[2]
            self._header_start_dt = self._parse_awd_datetime(date_str, time_str)
            
            if not self._header_start_dt:
                raise ValueError(f"Could not parse start date/time: {date_str} {time_str}")

            # Locate the start of the numeric data stream robustly
            start_line = -1
            for i in range(3, min(20, len(lines)-2)):
                if (re.match(r'^-?\d+$', lines[i]) and 
                    re.match(r'^-?\d+$', lines[i+1]) and 
                    re.match(r'^-?\d+$', lines[i+2])):
                    start_line = i
                    break
            
            if start_line == -1:
                raise ValueError("Could not locate the start of the numeric data stream.")

            data_str = lines[start_line:]
            
            # Extract values, handling ClockLab's error codes (often negative numbers)
            raw_values = []
            missing_count = 0
            for val in data_str:
                try:
                    # 'M' is sometimes appended to marked data in ClockLab, strip non-digits
                    clean_val = re.sub(r'[^\d\-]', '', val)
                    v = int(clean_val)
                    if v < 0:
                        raw_values.append(np.nan)
                        missing_count += 1
                    else:
                        raw_values.append(v)
                except ValueError:
                    raw_values.append(np.nan)
                    missing_count += 1

            # Build Raw DataFrame
            df = pd.DataFrame({'raw_activity': raw_values})
            
            # Use target bin size if set, otherwise fallback to native default
            self._native_bin_min = self._config.target_bin_size_minutes if self._config.target_bin_size_minutes else 6 
            
            timestamps = [self._header_start_dt + timedelta(minutes=i * self._native_bin_min) for i in range(len(df))]
            df['datetime'] = timestamps
            
            self._raw_data = df
            self._generate_summary(missing_count)

            return self._raw_data.copy()

        except Exception as e:
            raise ValueError(f"Failed to load AWD file {filepath}: {e}")

    def _parse_awd_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        """Parses ClockLab specific date and time strings."""
        try:
            date_parts = date_str.split('-')
            if len(date_parts) == 3:
                day = int(date_parts[0])
                month_str = date_parts[1]
                year_part = int(date_parts[2])
                
                months = {
                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                }
                month = months.get(month_str.lower()[:3], 1)
                
                if year_part < 100:
                    year = 2000 + year_part if year_part < 50 else 1900 + year_part
                else:
                    year = year_part

                time_parts = time_str.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2]) if len(time_parts) > 2 else 0

                return datetime(year, month, day, hour, minute, second)
        except Exception:
            pass
        return None

    def configure(self, config: AWDConfig) -> None:
        self._config = config
        if self._config.subject_id is None:
             self._config.subject_id = self._header_subject_id
        if self._raw_data is not None:
             self._generate_summary(self._summary.missing_data_bins if self._summary else 0)

    def _generate_summary(self, missing_count: int) -> None:
        if self._raw_data is None:
            return

        end_dt = self._raw_data['datetime'].max()
        total_days = (end_dt - self._header_start_dt).total_seconds() / 86400

        self._summary = AWDSummary(
            filepath=self._filepath or "",
            subject_id=self._config.subject_id or "Unknown",
            native_bin_size_minutes=self._native_bin_min,
            start_datetime=self._header_start_dt,
            end_datetime=end_dt,
            total_days=total_days,
            total_rows=len(self._raw_data),
            total_revolutions=int(self._raw_data['raw_activity'].sum(skipna=True)),
            missing_data_bins=missing_count
        )

    def process(self) -> pd.DataFrame:
        if self._raw_data is None:
            raise ValueError("No AWD data loaded. Call load_awd() first.")

        df = self._raw_data.copy()

        if self._config.start_datetime:
            df = df[df['datetime'] >= self._config.start_datetime]
        if self._config.end_datetime:
            df = df[df['datetime'] <= self._config.end_datetime]

        if len(df) == 0:
            raise ValueError("No data remaining after date filtering")

        df['zt'] = df['datetime'].apply(self._datetime_to_zt)

        target_bin = self._config.target_bin_size_minutes or self._native_bin_min
        
        if target_bin != self._native_bin_min:
            df['time_bin'] = (df['zt'] * 60 // target_bin) * target_bin / 60
            grouped = df.groupby('time_bin')['raw_activity'].sum(min_count=1).reset_index()
            grouped.rename(columns={'time_bin': 'time', 'raw_activity': 'activity'}, inplace=True)
            output_df = grouped
        else:
            df.rename(columns={'zt': 'time', 'raw_activity': 'activity'}, inplace=True)
            output_df = df[['time', 'activity']].copy()

        output_df['activity'] = output_df['activity'].fillna(0).astype(int)
        output_df['condition'] = self._config.condition_name
        output_df['subject'] = self._config.subject_id

        self._processed_data = output_df[['time', 'condition', 'subject', 'activity']]
        return self._processed_data.copy()

    def _datetime_to_zt(self, dt: datetime) -> float:
        hours_since_midnight = dt.hour + dt.minute / 60 + dt.second / 3600
        lights_on = self._config.lights_on_hour + self._config.lights_on_minute / 60
        zt = hours_since_midnight - lights_on

        if zt < 0:
            zt += 24

        if self._raw_data is not None and 'datetime' in self._raw_data.columns:
            first_dt = self._raw_data['datetime'].min()
            days_elapsed = (dt.date() - first_dt.date()).days
            zt += days_elapsed * 24

        return round(zt, 4)

    # =========================================================================
    # ChronoScope Compatibility Interface
    # =========================================================================

    def get_data(self) -> Optional[pd.DataFrame]:
        if self._processed_data is None and self._raw_data is not None:
            self.process()
        return self._processed_data.copy() if self._processed_data is not None else None

    def get_dataset_info(self) -> Optional[DatasetInfo]:
        if self._processed_data is None:
            return None

        df = self._processed_data
        column_info = []
        for col in df.columns:
            role = ColumnRole.UNKNOWN
            if col == 'time': role = ColumnRole.TIME
            elif col == 'condition': role = ColumnRole.CONDITION
            elif col == 'subject': role = ColumnRole.SUBJECT
            elif col == 'activity': role = ColumnRole.VARIABLE

            info = ColumnInfo(
                name=col,
                role=role,
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique(),
                sample_values=df[col].head(5).tolist(),
                is_numeric=pd.api.types.is_numeric_dtype(df[col])
            )
            column_info.append(info)

        return DatasetInfo(
            source_type=DataSourceType.USER_CSV,
            filepath=self._filepath or "",
            n_rows=len(df),
            n_columns=len(df.columns),
            time_column='time',
            condition_column='condition',
            replicate_column=None,
            subject_column='subject',
            variable_columns=['activity'],
            conditions=df['condition'].unique().tolist(),
            timepoints=sorted(df['time'].unique().tolist()),
            n_replicates_per_timepoint={},
            has_missing_values=df.isnull().any().any(),
            column_info=column_info
        )

    def get_summary(self) -> Optional[AWDSummary]:
        return self._summary
        
    def get_time_column(self) -> str: return self._time_col
    def get_condition_column(self) -> str: return self._condition_col
    def get_variable_columns(self) -> List[str]: return self._variable_cols.copy()
    def get_conditions(self) -> List[str]: 
        return self._processed_data['condition'].unique().tolist() if self._processed_data is not None else [self._config.condition_name]
    def get_timepoints(self) -> List[float]:
        return sorted(self._processed_data['time'].unique().tolist()) if self._processed_data is not None else []
    def get_columns(self) -> List[str]:
        return self._processed_data.columns.tolist() if self._processed_data is not None else ['time', 'condition', 'subject', 'activity']

    def validate_for_analysis(self) -> Tuple[bool, List[str]]:
        issues = []
        if self._raw_data is None:
            issues.append("No AWD data loaded")
            return False, issues

        if self._processed_data is None:
            try:
                self.process()
            except Exception as e:
                issues.append(f"Failed to process data: {e}")
                return False, issues

        if len(self._processed_data) == 0:
            issues.append("No data after processing")
        if len(self._processed_data['time'].unique()) < 4:
            issues.append("Insufficient timepoints")

        return len(issues) == 0, issues

    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        if self._processed_data is None and self._raw_data is not None:
            try: self.process()
            except: return pd.DataFrame()
        return self._processed_data.head(n_rows) if self._processed_data is not None else pd.DataFrame()


@dataclass
class AWDFileEntry:
    """Entry representing a single AWD file in a batch load."""
    filepath: str
    condition_name: str
    loader: Optional[AWDDataLoader] = None
    summary: Optional[AWDSummary] = None
    is_loaded: bool = False
    error_message: Optional[str] = None


class MultiAWDDataLoader:
    """
    Loader for multiple AWD files.
    Combines individual animal files into a single ChronoScope DataFrame.
    """

    def __init__(self):
        self._files: List[AWDFileEntry] = []
        self._shared_config: AWDConfig = AWDConfig()
        self._combined_data: Optional[pd.DataFrame] = None

        self._time_col: str = "time"
        self._condition_col: str = "condition"
        self._subject_col: str = "subject"
        self._variable_cols: List[str] = ["activity"]

    def add_file(self, filepath: str, condition_name: Optional[str] = None) -> int:
        """Adds an AWD file and loads it into the batch."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"AWD file not found: {filepath}")

        # Default condition name if none provided
        if condition_name is None:
            condition_name = "Condition1"

        loader = AWDDataLoader()
        entry = AWDFileEntry(
            filepath=str(path),
            condition_name=condition_name,
            loader=loader
        )

        try:
            loader.load_awd(str(path))
            
            # Apply shared config but respect individual condition name
            config = AWDConfig(**self._shared_config.__dict__)
            config.condition_name = condition_name
            loader.configure(config)
            
            entry.summary = loader.get_summary()
            entry.is_loaded = True
        except Exception as e:
            entry.error_message = str(e)
            entry.is_loaded = False

        self._files.append(entry)
        self._combined_data = None 
        return len(self._files) - 1

    def remove_file(self, index: int) -> None:
        if 0 <= index < len(self._files):
            del self._files[index]
            self._combined_data = None

    def get_files(self) -> List[AWDFileEntry]:
        return self._files.copy()

    def get_file_count(self) -> int:
        return len(self._files)

    def update_condition_name(self, index: int, condition_name: str) -> None:
        if 0 <= index < len(self._files):
            self._files[index].condition_name = condition_name
            if self._files[index].loader:
                self._files[index].loader._config.condition_name = condition_name
            self._combined_data = None

    def set_shared_config(self, config: AWDConfig) -> None:
        self._shared_config = config
        for entry in self._files:
            if entry.loader:
                ind_condition = entry.condition_name
                new_config = AWDConfig(**config.__dict__)
                new_config.condition_name = ind_condition
                entry.loader.configure(new_config)
                entry.summary = entry.loader.get_summary()
        self._combined_data = None

    def process(self) -> pd.DataFrame:
        """Processes and concatenates all loaded AWD files."""
        if not self._files:
            raise ValueError("No AWD files loaded")

        all_dfs = []
        for entry in self._files:
            if not entry.is_loaded or entry.loader is None:
                continue

            try:
                df = entry.loader.process()
                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {entry.filepath}: {e}")
                continue

        if not all_dfs:
            raise ValueError("No data could be processed from any file")

        self._combined_data = pd.concat(all_dfs, ignore_index=True)
        self._combined_data = self._combined_data.sort_values(
            ['time', 'condition', 'subject']
        ).reset_index(drop=True)

        return self._combined_data.copy()

    # =========================================================================
    # ChronoScope Compatibility Interface
    # =========================================================================

    def get_data(self) -> Optional[pd.DataFrame]:
        if self._combined_data is None and self._files:
            try: self.process()
            except: pass
        return self._combined_data.copy() if self._combined_data is not None else None

    def get_dataset_info(self) -> Optional[DatasetInfo]:
        if self._combined_data is None:
            try: self.process()
            except: return None
        if self._combined_data is None: return None

        df = self._combined_data
        column_info = []
        for col in df.columns:
            role = ColumnRole.UNKNOWN
            if col == 'time': role = ColumnRole.TIME
            elif col == 'condition': role = ColumnRole.CONDITION
            elif col == 'subject': role = ColumnRole.SUBJECT
            elif col == 'activity': role = ColumnRole.VARIABLE

            info = ColumnInfo(
                name=col,
                role=role,
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique(),
                sample_values=df[col].head(5).tolist(),
                is_numeric=pd.api.types.is_numeric_dtype(df[col])
            )
            column_info.append(info)

        filepaths = [e.filepath for e in self._files if e.is_loaded]
        
        return DatasetInfo(
            source_type=DataSourceType.USER_CSV,
            filepath="; ".join(filepaths) if filepaths else "",
            n_rows=len(df),
            n_columns=len(df.columns),
            time_column='time',
            condition_column='condition',
            replicate_column=None,
            subject_column='subject',
            variable_columns=['activity'],
            conditions=df['condition'].unique().tolist(),
            timepoints=sorted(df['time'].unique().tolist()),
            n_replicates_per_timepoint={},
            has_missing_values=df.isnull().any().any(),
            column_info=column_info
        )

    def validate_for_analysis(self) -> Tuple[bool, List[str]]:
        issues = []
        if not self._files:
            issues.append("No files loaded")
            return False, issues

        loaded = sum(1 for e in self._files if e.is_loaded)
        if loaded == 0:
            issues.append("No files successfully loaded")
            return False, issues

        if self._combined_data is None:
            try: self.process()
            except Exception as e:
                issues.append(f"Failed to process batch: {e}")
                return False, issues

        if len(self._combined_data) == 0:
            issues.append("No data after processing")
            
        return len(issues) == 0, issues

    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        if self._combined_data is None:
            try: self.process()
            except: return pd.DataFrame()
        return self._combined_data.head(n_rows) if self._combined_data is not None else pd.DataFrame()

    def get_summary(self) -> Dict[str, Any]:
        """Provides a high-level summary of the entire batch."""
        total_rows = 0
        total_missing = 0
        
        for entry in self._files:
            if entry.summary:
                total_rows += entry.summary.total_rows
                total_missing += entry.summary.missing_data_bins
                
        return {
            'total_files_loaded': sum(1 for e in self._files if e.is_loaded),
            'total_conditions': len(set(e.condition_name for e in self._files if e.is_loaded)),
            'total_data_points': total_rows,
            'missing_data_points': total_missing
        }