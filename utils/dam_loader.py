"""
DAM Data Loader Module
======================

Handles loading and processing of TriKinetics DAM (Drosophila Activity Monitor)
files for circadian analysis.

DAM File Format:
    Tab-separated values with columns:
    - Index (int)
    - Date (e.g., "4 Jul 25")
    - Time (e.g., "16:28:00")
    - Metadata columns (monitor info)
    - Status code (e.g., "Ct")
    - 32 activity channels (counts per minute)

Output Format (ChronoScope compatible):
    time,condition,replicate,activity
    0.0,Monitor51,ch01,45
    0.5,Monitor51,ch01,52
    ...
"""

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from .data_loader import DatasetInfo, DataSourceType, ColumnInfo, ColumnRole


@dataclass
class DAMConfig:
    """Configuration for DAM data processing."""

    # Binning settings
    bin_size_minutes: int = 30  # Options: 1, 5, 15, 30, 60

    # Time reference
    lights_on_hour: int = 8  # Hour when lights turn on (ZT0)
    lights_on_minute: int = 0

    # Date range (None = use all data)
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

    # Dead fly filtering
    exclude_dead: bool = True
    death_threshold_hours: float = 6.0  # Hours of zero activity = dead

    # Channel selection (None = all 32 channels)
    channels: Optional[List[int]] = None  # 1-indexed: [1, 2, 3, ..., 32]

    # Metadata
    condition_name: str = "Monitor"  # Name for the condition column


@dataclass
class DAMChannelInfo:
    """Information about a single DAM channel."""
    channel_number: int
    total_activity: int
    is_alive: bool
    death_time: Optional[datetime] = None
    mean_activity_per_bin: float = 0.0


@dataclass
class DAMSummary:
    """Summary information about loaded DAM data."""
    filepath: str
    monitor_number: Optional[int]
    total_channels: int
    live_channels: int
    dead_channels: int
    start_datetime: datetime
    end_datetime: datetime
    total_days: float
    total_rows: int
    timepoints_after_binning: int
    channel_info: List[DAMChannelInfo]


class DAMDataLoader:
    """
    Loader for TriKinetics DAM (Drosophila Activity Monitor) files.

    Parses the proprietary DAM format and converts it to a ChronoScope-compatible
    DataFrame for circadian rhythm analysis.
    """

    # Number of activity channels in standard DAM monitors
    N_CHANNELS = 32

    # Column indices in DAM file format
    COL_INDEX = 0
    COL_DATE = 1
    COL_TIME = 2
    COL_STATUS = 7  # Usually contains "Ct" or similar
    COL_FIRST_CHANNEL = 9  # Columns 9-40 contain channel data (0-indexed: 9 to 40)

    def __init__(self):
        """Initialize the DAM data loader."""
        self._raw_data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._filepath: Optional[str] = None
        self._config: DAMConfig = DAMConfig()
        self._summary: Optional[DAMSummary] = None
        self._channel_alive: Dict[int, bool] = {}

        # For ChronoScope compatibility
        # DAM data is DEPENDENT: same fly (subject) measured repeatedly over time
        self._time_col: str = "time"
        self._condition_col: str = "condition"
        self._subject_col: str = "subject"  # Each channel is a subject (repeated measures)
        self._replicate_col: Optional[str] = None  # No replicate for DAM (subject IS the identifier)
        self._variable_cols: List[str] = ["activity"]

    def load_dam(self, filepath: str) -> pd.DataFrame:
        """
        Load a DAM monitor file.

        Args:
            filepath: Path to the DAM .txt file

        Returns:
            Raw DataFrame with parsed data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"DAM file not found: {filepath}")

        try:
            # Read tab-separated file without headers
            self._raw_data = pd.read_csv(
                filepath,
                sep='\t',
                header=None,
                encoding='utf-8',
                on_bad_lines='skip'
            )
            self._filepath = filepath

            # Extract monitor number from filename if possible
            monitor_num = self._extract_monitor_number(path.stem)
            if monitor_num and self._config.condition_name == "Monitor":
                self._config.condition_name = f"Monitor{monitor_num}"

            # Parse datetime from columns
            self._parse_datetime()

            # Generate initial summary
            self._generate_summary()

            return self._raw_data.copy()

        except Exception as e:
            raise ValueError(f"Failed to load DAM file: {e}")

    def _extract_monitor_number(self, filename: str) -> Optional[int]:
        """Extract monitor number from filename like 'Monitor51'."""
        import re
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def _parse_datetime(self) -> None:
        """Parse date and time columns into datetime objects."""
        if self._raw_data is None:
            return

        def parse_row_datetime(row):
            try:
                # Date format: "4 Jul 25" or "14 Jul 25"
                date_str = str(row[self.COL_DATE]).strip()
                time_str = str(row[self.COL_TIME]).strip()

                # Parse date - handle both "4 Jul 25" and "04 Jul 25"
                date_parts = date_str.split()
                if len(date_parts) == 3:
                    day = int(date_parts[0])
                    month_str = date_parts[1]
                    year_short = int(date_parts[2])

                    # Convert month abbreviation
                    months = {
                        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                    }
                    month = months.get(month_str.lower()[:3], 1)

                    # Convert 2-digit year to 4-digit
                    year = 2000 + year_short if year_short < 100 else year_short

                    # Parse time - format "16:28:00"
                    time_parts = time_str.split(':')
                    hour = int(time_parts[0])
                    minute = int(time_parts[1])
                    second = int(time_parts[2]) if len(time_parts) > 2 else 0

                    return datetime(year, month, day, hour, minute, second)
            except Exception:
                return None
            return None

        self._raw_data['datetime'] = self._raw_data.apply(parse_row_datetime, axis=1)

        # Remove rows with invalid datetime
        valid_mask = self._raw_data['datetime'].notna()
        self._raw_data = self._raw_data[valid_mask].copy()

        # Sort by datetime
        self._raw_data = self._raw_data.sort_values('datetime').reset_index(drop=True)

    def configure(self, config: DAMConfig) -> None:
        """
        Apply configuration settings.

        Args:
            config: DAMConfig object with processing settings
        """
        self._config = config

        # Recalculate summary with new config
        if self._raw_data is not None:
            self._generate_summary()

    def _detect_dead_flies(self) -> Dict[int, Tuple[bool, Optional[datetime]]]:
        """
        Detect dead flies based on consecutive hours of zero activity.

        Returns:
            Dict mapping channel number to (is_alive, death_time)
        """
        if self._raw_data is None:
            return {}

        threshold_minutes = self._config.death_threshold_hours * 60
        results = {}

        for ch in range(1, self.N_CHANNELS + 1):
            col_idx = self.COL_FIRST_CHANNEL + ch - 1
            if col_idx >= len(self._raw_data.columns):
                continue

            # Get activity data with timestamps
            activity = self._raw_data[col_idx].values
            timestamps = self._raw_data['datetime'].values

            # Find consecutive zeros
            is_alive = True
            death_time = None
            zero_start = None
            zero_duration = 0

            for i, (act, ts) in enumerate(zip(activity, timestamps)):
                if pd.isna(act) or act == 0:
                    if zero_start is None:
                        zero_start = ts
                    if i > 0:
                        try:
                            delta = (pd.Timestamp(ts) - pd.Timestamp(timestamps[i-1])).total_seconds() / 60
                            zero_duration += delta
                        except:
                            zero_duration += 1  # Assume 1 minute if calculation fails
                else:
                    zero_start = None
                    zero_duration = 0

                if zero_duration >= threshold_minutes:
                    is_alive = False
                    death_time = pd.Timestamp(zero_start).to_pydatetime() if zero_start else None
                    break

            results[ch] = (is_alive, death_time)

        return results

    def _generate_summary(self) -> None:
        """Generate summary information about the DAM data."""
        if self._raw_data is None:
            return

        # Detect dead flies
        death_info = self._detect_dead_flies()
        self._channel_alive = {ch: info[0] for ch, info in death_info.items()}

        # Build channel info
        channel_info_list = []
        for ch in range(1, self.N_CHANNELS + 1):
            col_idx = self.COL_FIRST_CHANNEL + ch - 1
            if col_idx >= len(self._raw_data.columns):
                continue

            total_act = self._raw_data[col_idx].sum()
            is_alive, death_time = death_info.get(ch, (True, None))

            info = DAMChannelInfo(
                channel_number=ch,
                total_activity=int(total_act),
                is_alive=is_alive,
                death_time=death_time,
                mean_activity_per_bin=total_act / len(self._raw_data) if len(self._raw_data) > 0 else 0
            )
            channel_info_list.append(info)

        # Calculate timepoints after binning
        if 'datetime' in self._raw_data.columns:
            start_dt = self._raw_data['datetime'].min()
            end_dt = self._raw_data['datetime'].max()

            # Apply date filter if configured
            if self._config.start_datetime:
                start_dt = max(start_dt, self._config.start_datetime)
            if self._config.end_datetime:
                end_dt = min(end_dt, self._config.end_datetime)

            total_minutes = (end_dt - start_dt).total_seconds() / 60
            timepoints = int(total_minutes / self._config.bin_size_minutes)
        else:
            start_dt = datetime.now()
            end_dt = datetime.now()
            timepoints = 0

        live_count = sum(1 for info in channel_info_list if info.is_alive)
        dead_count = len(channel_info_list) - live_count

        self._summary = DAMSummary(
            filepath=self._filepath or "",
            monitor_number=self._extract_monitor_number(Path(self._filepath).stem if self._filepath else ""),
            total_channels=len(channel_info_list),
            live_channels=live_count,
            dead_channels=dead_count,
            start_datetime=start_dt,
            end_datetime=end_dt,
            total_days=(end_dt - start_dt).total_seconds() / 86400,
            total_rows=len(self._raw_data),
            timepoints_after_binning=timepoints,
            channel_info=channel_info_list
        )

    def get_summary(self) -> Optional[DAMSummary]:
        """Get summary information about the loaded data."""
        return self._summary

    def get_channel_summary(self) -> Dict[str, Any]:
        """Get channel summary for UI preview."""
        if self._summary is None:
            return {}

        return {
            'total_channels': self._summary.total_channels,
            'live_channels': self._summary.live_channels,
            'dead_channels': self._summary.dead_channels,
            'start_date': self._summary.start_datetime.strftime('%Y-%m-%d') if self._summary.start_datetime else '',
            'end_date': self._summary.end_datetime.strftime('%Y-%m-%d') if self._summary.end_datetime else '',
            'total_days': round(self._summary.total_days, 1),
            'timepoints_after_binning': self._summary.timepoints_after_binning,
            'bin_size_minutes': self._config.bin_size_minutes
        }

    def process(self) -> pd.DataFrame:
        """
        Process raw DAM data into ChronoScope-compatible format.

        Returns:
            DataFrame with columns: time, condition, replicate, activity
        """
        if self._raw_data is None:
            raise ValueError("No DAM data loaded. Call load_dam() first.")

        # Determine which channels to include
        if self._config.channels:
            channels = self._config.channels
        else:
            channels = list(range(1, self.N_CHANNELS + 1))

        # Filter dead flies if configured
        if self._config.exclude_dead:
            channels = [ch for ch in channels if self._channel_alive.get(ch, True)]

        # Filter by date range
        df = self._raw_data.copy()
        if self._config.start_datetime:
            df = df[df['datetime'] >= self._config.start_datetime]
        if self._config.end_datetime:
            df = df[df['datetime'] <= self._config.end_datetime]

        if len(df) == 0:
            raise ValueError("No data remaining after date filtering")

        # Calculate ZT time for each row
        df['zt'] = df['datetime'].apply(self._datetime_to_zt)

        # Apply binning
        bin_minutes = self._config.bin_size_minutes
        df['time_bin'] = (df['zt'] * 60 // bin_minutes) * bin_minutes / 60

        # Build output DataFrame
        rows = []

        for time_bin in sorted(df['time_bin'].unique()):
            bin_data = df[df['time_bin'] == time_bin]

            for ch in channels:
                col_idx = self.COL_FIRST_CHANNEL + ch - 1
                if col_idx >= len(bin_data.columns):
                    continue

                # Sum activity in this bin
                activity = bin_data[col_idx].sum()

                rows.append({
                    'time': time_bin,
                    'condition': self._config.condition_name,
                    'subject': f"ch{ch:02d}",  # Each channel is a subject (same fly measured repeatedly)
                    'activity': int(activity)
                })

        self._processed_data = pd.DataFrame(rows)

        return self._processed_data.copy()

    def _datetime_to_zt(self, dt: datetime) -> float:
        """
        Convert datetime to ZT (Zeitgeber Time) hours.

        ZT0 is defined by lights_on_hour in config.

        Args:
            dt: datetime object

        Returns:
            ZT time in decimal hours (0.0 - 24.0+)
        """
        # Calculate hours since midnight
        hours_since_midnight = dt.hour + dt.minute / 60 + dt.second / 3600

        # Calculate ZT (relative to lights on)
        lights_on = self._config.lights_on_hour + self._config.lights_on_minute / 60
        zt = hours_since_midnight - lights_on

        # Handle negative values (before lights on)
        if zt < 0:
            zt += 24

        # Add full days
        # Get start of first day
        if self._raw_data is not None and 'datetime' in self._raw_data.columns:
            first_dt = self._raw_data['datetime'].min()
            days_elapsed = (dt.date() - first_dt.date()).days
            zt += days_elapsed * 24

        return round(zt, 4)

    # =========================================================================
    # ChronoScope Compatibility Interface
    # =========================================================================

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the processed data (ChronoScope compatible)."""
        if self._processed_data is None:
            if self._raw_data is not None:
                self.process()
        return self._processed_data.copy() if self._processed_data is not None else None

    def get_dataset_info(self) -> Optional[DatasetInfo]:
        """Get dataset info in ChronoScope format."""
        if self._processed_data is None:
            return None

        df = self._processed_data

        # Build column info
        column_info = []
        for col in df.columns:
            role = ColumnRole.UNKNOWN
            if col == 'time':
                role = ColumnRole.TIME
            elif col == 'condition':
                role = ColumnRole.CONDITION
            elif col == 'subject':
                role = ColumnRole.SUBJECT  # Each channel is a subject (repeated measures)
            elif col == 'activity':
                role = ColumnRole.VARIABLE

            info = ColumnInfo(
                name=col,
                role=role,
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique(),
                sample_values=df[col].head(5).tolist(),
                is_numeric=pd.api.types.is_numeric_dtype(df[col])
            )
            column_info.append(info)

        conditions = df['condition'].unique().tolist()
        timepoints = sorted(df['time'].unique().tolist())

        # DAM data is DEPENDENT: same fly measured repeatedly over time
        return DatasetInfo(
            source_type=DataSourceType.USER_CSV,  # Treat as CSV for compatibility
            filepath=self._filepath or "",
            n_rows=len(df),
            n_columns=len(df.columns),
            time_column='time',
            condition_column='condition',
            replicate_column=None,  # No replicate column for DAM
            subject_column='subject',  # Each channel is a subject (DEPENDENT data)
            variable_columns=['activity'],
            conditions=conditions,
            timepoints=timepoints,
            n_replicates_per_timepoint={},
            has_missing_values=df.isnull().any().any(),
            column_info=column_info
        )

    def get_time_column(self) -> str:
        """Get the time column name."""
        return self._time_col

    def get_condition_column(self) -> str:
        """Get the condition column name."""
        return self._condition_col

    def get_variable_columns(self) -> List[str]:
        """Get the variable column names."""
        return self._variable_cols.copy()

    def get_conditions(self) -> List[str]:
        """Get list of conditions."""
        if self._processed_data is None:
            return [self._config.condition_name]
        return self._processed_data['condition'].unique().tolist()

    def get_timepoints(self) -> List[float]:
        """Get list of timepoints."""
        if self._processed_data is None:
            return []
        return sorted(self._processed_data['time'].unique().tolist())

    def get_columns(self) -> List[str]:
        """Get all column names."""
        if self._processed_data is None:
            return ['time', 'condition', 'replicate', 'activity']
        return self._processed_data.columns.tolist()

    def validate_for_analysis(self) -> Tuple[bool, List[str]]:
        """
        Validate that data is ready for analysis.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if self._raw_data is None:
            issues.append("No DAM data loaded")
            return False, issues

        if self._processed_data is None:
            try:
                self.process()
            except Exception as e:
                issues.append(f"Failed to process data: {e}")
                return False, issues

        if len(self._processed_data) == 0:
            issues.append("No data after processing")

        # Check for live channels
        if self._config.exclude_dead:
            live_count = sum(1 for alive in self._channel_alive.values() if alive)
            if live_count == 0:
                issues.append("No live channels found with current death threshold")

        # Check minimum timepoints
        if len(self._processed_data['time'].unique()) < 4:
            issues.append("Insufficient timepoints (minimum 4 recommended)")

        return len(issues) == 0, issues

    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of processed data."""
        if self._processed_data is None:
            if self._raw_data is not None:
                try:
                    self.process()
                except:
                    return pd.DataFrame()

        if self._processed_data is None:
            return pd.DataFrame()

        return self._processed_data.head(n_rows)

    def get_available_dates(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range available in the loaded data."""
        if self._raw_data is None or 'datetime' not in self._raw_data.columns:
            return None, None

        return (
            self._raw_data['datetime'].min(),
            self._raw_data['datetime'].max()
        )

    def get_channel_count(self) -> int:
        """Get number of channels that will be included in analysis."""
        if self._config.channels:
            channels = self._config.channels
        else:
            channels = list(range(1, self.N_CHANNELS + 1))

        if self._config.exclude_dead:
            channels = [ch for ch in channels if self._channel_alive.get(ch, True)]

        return len(channels)


@dataclass
class MonitorEntry:
    """Entry representing a single DAM monitor in a multi-monitor setup."""
    filepath: str
    condition_name: str
    loader: Optional[DAMDataLoader] = None
    summary: Optional[DAMSummary] = None
    is_loaded: bool = False
    error_message: Optional[str] = None


class MultiDAMDataLoader:
    """
    Loader for multiple DAM monitor files.

    Combines data from multiple monitors into a single DataFrame with
    unique subject identifiers (e.g., M51_ch05, M52_ch03).
    """

    def __init__(self):
        """Initialize the multi-DAM loader."""
        self._monitors: List[MonitorEntry] = []
        self._shared_config: DAMConfig = DAMConfig()
        self._combined_data: Optional[pd.DataFrame] = None

        # For ChronoScope compatibility
        self._time_col: str = "time"
        self._condition_col: str = "condition"
        self._subject_col: str = "subject"
        self._replicate_col: Optional[str] = None
        self._variable_cols: List[str] = ["activity"]

    def add_monitor(self, filepath: str, condition_name: Optional[str] = None) -> int:
        """
        Add a monitor file to the loader.

        Args:
            filepath: Path to the DAM .txt file
            condition_name: Optional condition name (auto-detected if not provided)

        Returns:
            Index of the added monitor
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"DAM file not found: {filepath}")

        # Auto-detect condition name from filename
        if condition_name is None:
            import re
            match = re.search(r'(\d+)', path.stem)
            if match:
                condition_name = f"Monitor{match.group(1)}"
            else:
                condition_name = path.stem

        # Create loader and load file
        loader = DAMDataLoader()
        entry = MonitorEntry(
            filepath=str(path),
            condition_name=condition_name,
            loader=loader
        )

        try:
            loader.load_dam(str(path))
            loader._config.condition_name = condition_name
            entry.summary = loader.get_summary()
            entry.is_loaded = True
        except Exception as e:
            entry.error_message = str(e)
            entry.is_loaded = False

        self._monitors.append(entry)
        self._combined_data = None  # Invalidate combined data

        return len(self._monitors) - 1

    def remove_monitor(self, index: int) -> None:
        """Remove a monitor by index."""
        if 0 <= index < len(self._monitors):
            del self._monitors[index]
            self._combined_data = None

    def get_monitors(self) -> List[MonitorEntry]:
        """Get list of all monitor entries."""
        return self._monitors.copy()

    def get_monitor_count(self) -> int:
        """Get number of monitors."""
        return len(self._monitors)

    def update_condition_name(self, index: int, condition_name: str) -> None:
        """Update the condition name for a monitor."""
        if 0 <= index < len(self._monitors):
            self._monitors[index].condition_name = condition_name
            if self._monitors[index].loader:
                self._monitors[index].loader._config.condition_name = condition_name
            self._combined_data = None

    def set_shared_config(self, config: DAMConfig) -> None:
        """
        Set shared configuration for all monitors.

        Note: condition_name from shared config is ignored; each monitor
        keeps its individual condition name.
        """
        self._shared_config = config

        # Apply config to all loaders (except condition_name)
        for entry in self._monitors:
            if entry.loader:
                individual_condition = entry.condition_name
                entry.loader.configure(config)
                entry.loader._config.condition_name = individual_condition
                entry.summary = entry.loader.get_summary()

        self._combined_data = None

    def get_shared_config(self) -> DAMConfig:
        """Get the shared configuration."""
        return self._shared_config

    def process(self) -> pd.DataFrame:
        """
        Process and combine all monitors into a single DataFrame.

        Subject identifiers are prefixed with monitor identifier to ensure
        uniqueness (e.g., M51_ch05, M52_ch03).

        Returns:
            Combined DataFrame with columns: time, condition, subject, activity
        """
        if not self._monitors:
            raise ValueError("No monitors loaded")

        all_dfs = []

        for entry in self._monitors:
            if not entry.is_loaded or entry.loader is None:
                continue

            # Process this monitor's data
            try:
                df = entry.loader.process()

                # Create unique subject IDs by prefixing with monitor identifier
                monitor_prefix = self._get_monitor_prefix(entry)
                df['subject'] = monitor_prefix + '_' + df['subject'].astype(str)

                # Ensure condition is set correctly
                df['condition'] = entry.condition_name

                all_dfs.append(df)
            except Exception as e:
                print(f"Error processing {entry.filepath}: {e}")
                continue

        if not all_dfs:
            raise ValueError("No data could be processed from any monitor")

        self._combined_data = pd.concat(all_dfs, ignore_index=True)

        # Sort by time and condition
        self._combined_data = self._combined_data.sort_values(
            ['time', 'condition', 'subject']
        ).reset_index(drop=True)

        return self._combined_data.copy()

    def _get_monitor_prefix(self, entry: MonitorEntry) -> str:
        """Get a short prefix for subject IDs from monitor file."""
        import re
        path = Path(entry.filepath)
        match = re.search(r'(\d+)', path.stem)
        if match:
            return f"M{match.group(1)}"
        return path.stem[:8]  # Use first 8 chars of filename

    # =========================================================================
    # ChronoScope Compatibility Interface
    # =========================================================================

    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the combined processed data."""
        if self._combined_data is None:
            if self._monitors:
                try:
                    self.process()
                except:
                    pass
        return self._combined_data.copy() if self._combined_data is not None else None

    def get_dataset_info(self) -> Optional[DatasetInfo]:
        """Get dataset info in ChronoScope format."""
        if self._combined_data is None:
            try:
                self.process()
            except:
                return None

        if self._combined_data is None:
            return None

        df = self._combined_data

        # Build column info
        column_info = []
        for col in df.columns:
            role = ColumnRole.UNKNOWN
            if col == 'time':
                role = ColumnRole.TIME
            elif col == 'condition':
                role = ColumnRole.CONDITION
            elif col == 'subject':
                role = ColumnRole.SUBJECT
            elif col == 'activity':
                role = ColumnRole.VARIABLE

            info = ColumnInfo(
                name=col,
                role=role,
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique(),
                sample_values=df[col].head(5).tolist(),
                is_numeric=pd.api.types.is_numeric_dtype(df[col])
            )
            column_info.append(info)

        conditions = df['condition'].unique().tolist()
        timepoints = sorted(df['time'].unique().tolist())

        # Get combined filepath info
        filepaths = [e.filepath for e in self._monitors if e.is_loaded]
        filepath_str = "; ".join(filepaths) if filepaths else ""

        return DatasetInfo(
            source_type=DataSourceType.USER_CSV,
            filepath=filepath_str,
            n_rows=len(df),
            n_columns=len(df.columns),
            time_column='time',
            condition_column='condition',
            replicate_column=None,
            subject_column='subject',
            variable_columns=['activity'],
            conditions=conditions,
            timepoints=timepoints,
            n_replicates_per_timepoint={},
            has_missing_values=df.isnull().any().any(),
            column_info=column_info
        )

    def get_time_column(self) -> str:
        """Get the time column name."""
        return self._time_col

    def get_condition_column(self) -> str:
        """Get the condition column name."""
        return self._condition_col

    def get_variable_columns(self) -> List[str]:
        """Get the variable column names."""
        return self._variable_cols.copy()

    def get_conditions(self) -> List[str]:
        """Get list of conditions."""
        return list(dict.fromkeys(e.condition_name for e in self._monitors if e.is_loaded))

    def get_timepoints(self) -> List[float]:
        """Get list of timepoints."""
        if self._combined_data is None:
            return []
        return sorted(self._combined_data['time'].unique().tolist())

    def get_columns(self) -> List[str]:
        """Get all column names."""
        return ['time', 'condition', 'subject', 'activity']

    def validate_for_analysis(self) -> Tuple[bool, List[str]]:
        """Validate that data is ready for analysis."""
        issues = []

        if not self._monitors:
            issues.append("No monitors loaded")
            return False, issues

        loaded_count = sum(1 for e in self._monitors if e.is_loaded)
        if loaded_count == 0:
            issues.append("No monitors successfully loaded")
            return False, issues

        # Try to process
        if self._combined_data is None:
            try:
                self.process()
            except Exception as e:
                issues.append(f"Failed to process data: {e}")
                return False, issues

        if len(self._combined_data) == 0:
            issues.append("No data after processing")

        if len(self._combined_data['time'].unique()) < 4:
            issues.append("Insufficient timepoints (minimum 4 recommended)")

        return len(issues) == 0, issues

    def get_preview(self, n_rows: int = 10) -> pd.DataFrame:
        """Get preview of combined data."""
        if self._combined_data is None:
            try:
                self.process()
            except:
                return pd.DataFrame()

        return self._combined_data.head(n_rows) if self._combined_data is not None else pd.DataFrame()

    def get_channel_count(self) -> int:
        """Get total number of channels across all monitors."""
        total = 0
        for entry in self._monitors:
            if entry.is_loaded and entry.loader:
                total += entry.loader.get_channel_count()
        return total

    def get_channel_summary(self) -> Dict[str, Any]:
        """Get combined channel summary."""
        total_channels = 0
        live_channels = 0
        dead_channels = 0

        for entry in self._monitors:
            if entry.summary:
                total_channels += entry.summary.total_channels
                live_channels += entry.summary.live_channels
                dead_channels += entry.summary.dead_channels

        # Get date range from first monitor (assume all have same range)
        start_date = ""
        end_date = ""
        timepoints = 0

        if self._monitors and self._monitors[0].summary:
            s = self._monitors[0].summary
            start_date = s.start_datetime.strftime('%Y-%m-%d')
            end_date = s.end_datetime.strftime('%Y-%m-%d')
            timepoints = s.timepoints_after_binning

        return {
            'total_monitors': len(self._monitors),
            'total_channels': total_channels,
            'live_channels': live_channels,
            'dead_channels': dead_channels,
            'start_date': start_date,
            'end_date': end_date,
            'timepoints_after_binning': timepoints,
            'bin_size_minutes': self._shared_config.bin_size_minutes
        }
