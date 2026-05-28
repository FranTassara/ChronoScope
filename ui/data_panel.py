"""
Data Panel
==========

Panel for loading and configuring data sources (CSV or Rosbash dataset).
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QSplitter, QFrame, QLineEdit, QMessageBox, QProgressBar,
    QDoubleSpinBox, QDateEdit, QTimeEdit, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QThread, QDate, QTime
from PySide6.QtGui import QFont

import pandas as pd

from utils.data_loader import CircadianDataLoader, DatasetInfo
from utils.rosbash_loader import RosbashDataLoader, get_available_clock_genes
from utils.dam_loader import DAMDataLoader, DAMConfig, MultiDAMDataLoader, MonitorEntry
from utils.awd_loader import AWDDataLoader, AWDConfig, MultiAWDDataLoader, AWDFileEntry


class DataLoadWorker(QThread):
    """Worker thread for loading data without blocking UI."""
    finished = Signal(bool, str)  # success, message
    progress = Signal(int)
    
    def __init__(self, loader_type: str, filepath: str, **kwargs):
        super().__init__()
        self.loader_type = loader_type
        self.filepath = filepath
        self.kwargs = kwargs
        self.result = None
    
    def run(self):
        try:
            self.progress.emit(50)
            
            if self.loader_type == 'csv':
                loader = CircadianDataLoader()
                loader.load_csv(self.filepath, **self.kwargs)
                self.result = loader
            elif self.loader_type == 'rosbash':
                loader = RosbashDataLoader(self.filepath)
                self.result = loader
            
            self.progress.emit(100)
            self.finished.emit(True, "Data loaded successfully")
            
        except Exception as e:
            self.finished.emit(False, str(e))


class DataPanel(QWidget):
    """
    Panel for data loading and configuration.
    
    Supports two data sources:
    1. User CSV files with circadian expression data
    2. Preprocessed Rosbash scRNA-seq dataset (HDF5)
    
    Signals:
        data_loaded: Emitted when data is successfully loaded
        data_cleared: Emitted when data is cleared
    """
    
    data_loaded = Signal(object, str)  # loader object, source type
    data_cleared = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._csv_loader: Optional[CircadianDataLoader] = None
        self._rosbash_loader: Optional[RosbashDataLoader] = None
        self._dam_loader: Optional[MultiDAMDataLoader] = None
        self._awd_loader: Optional[MultiAWDDataLoader] = None
        self._current_source: Optional[str] = None

        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Data source selection
        source_group = self._create_source_selection()
        layout.addWidget(source_group)
        
        # Stacked panels for each source type
        self._csv_panel = self._create_csv_panel()
        self._rosbash_panel = self._create_rosbash_panel()
        self._dam_panel = self._create_dam_panel()
        self._awd_panel = self._create_awd_panel()

        # Initially show CSV panel
        self._rosbash_panel.setVisible(False)
        self._dam_panel.setVisible(False)
        self._awd_panel.setVisible(False)

        layout.addWidget(self._csv_panel)
        layout.addWidget(self._rosbash_panel)
        layout.addWidget(self._dam_panel)
        layout.addWidget(self._awd_panel)
        
        # Data preview
        preview_group = self._create_preview_panel()
        layout.addWidget(preview_group)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        # Status label
        self._status_label = QLabel("No data loaded")
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._status_label)
        
        layout.addStretch()
    
    def _create_source_selection(self) -> QGroupBox:
        """Create data source selection group."""
        group = QGroupBox("Data Source")
        layout = QHBoxLayout(group)
        
        self._source_combo = QComboBox()
        self._source_combo.addItems([
            "User CSV File",
            "Rosbash scRNA-seq Dataset",
            "DAM Monitor File",
            "Running Wheel (.awd)"
        ])
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        
        layout.addWidget(QLabel("Source:"))
        layout.addWidget(self._source_combo, 1)
        
        return group
    
    def _create_csv_panel(self) -> QGroupBox:
        """Create CSV configuration panel."""
        group = QGroupBox("CSV Data Configuration")
        layout = QVBoxLayout(group)
        
        # File selection
        file_layout = QHBoxLayout()
        self._csv_path_edit = QLineEdit()
        self._csv_path_edit.setReadOnly(True)
        self._csv_path_edit.setPlaceholderText("Select a CSV file...")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_csv)
        
        file_layout.addWidget(self._csv_path_edit, 1)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)
        
        # Column mapping
        mapping_layout = QHBoxLayout()
        
        # Time column
        time_layout = QVBoxLayout()
        time_layout.addWidget(QLabel("Time Column:"))
        self._time_col_combo = QComboBox()
        self._time_col_combo.setMinimumWidth(120)
        time_layout.addWidget(self._time_col_combo)
        mapping_layout.addLayout(time_layout)
        
        # Condition column
        cond_layout = QVBoxLayout()
        cond_layout.addWidget(QLabel("Condition Column:"))
        self._cond_col_combo = QComboBox()
        self._cond_col_combo.setMinimumWidth(120)
        cond_layout.addWidget(self._cond_col_combo)
        mapping_layout.addLayout(cond_layout)
        
        # Subject column (optional)
        subj_layout = QVBoxLayout()
        subj_layout.addWidget(QLabel("Subject (optional):"))
        self._subj_col_combo = QComboBox()
        self._subj_col_combo.addItem("(None)")
        self._subj_col_combo.setMinimumWidth(120)
        self._subj_col_combo.setToolTip("For DEPENDENT data: Same individuals measured repeatedly over time")
        subj_layout.addWidget(self._subj_col_combo)
        mapping_layout.addLayout(subj_layout)

        layout.addLayout(mapping_layout)

        # Load button
        self._csv_load_btn = QPushButton("Load CSV Data")
        self._csv_load_btn.clicked.connect(self._load_csv_data)
        self._csv_load_btn.setEnabled(False)
        layout.addWidget(self._csv_load_btn)
        
        return group
    
    def _create_rosbash_panel(self) -> QGroupBox:
        """Create Rosbash dataset configuration panel."""
        group = QGroupBox("Rosbash scRNA-seq Dataset Configuration")
        layout = QVBoxLayout(group)
        
        # File selection
        file_layout = QHBoxLayout()
        self._h5_path_edit = QLineEdit()
        self._h5_path_edit.setReadOnly(True)
        self._h5_path_edit.setPlaceholderText("Select the preprocessed HDF5 file...")
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_h5)
        
        file_layout.addWidget(self._h5_path_edit, 1)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # Load button
        self._h5_load_btn = QPushButton("Load Rosbash Dataset")
        self._h5_load_btn.clicked.connect(self._load_rosbash_data)
        self._h5_load_btn.setEnabled(False)
        layout.addWidget(self._h5_load_btn)
        
        return group

    def _create_dam_panel(self) -> QGroupBox:
        """Create DAM Monitor file configuration panel with multi-monitor support."""
        group = QGroupBox("DAM Data Configuration")
        layout = QVBoxLayout(group)

        # =====================================================================
        # Monitor List Section
        # =====================================================================
        monitors_group = QGroupBox("DAM Files")
        monitors_layout = QVBoxLayout(monitors_group)

        # Buttons for add/remove
        btn_layout = QHBoxLayout()
        self._dam_add_btn = QPushButton("+ Add Monitor")
        self._dam_add_btn.clicked.connect(self._add_dam_monitor)
        self._dam_remove_btn = QPushButton("- Remove Selected")
        self._dam_remove_btn.clicked.connect(self._remove_dam_monitor)
        self._dam_remove_btn.setEnabled(False)
        btn_layout.addWidget(self._dam_add_btn)
        btn_layout.addWidget(self._dam_remove_btn)
        btn_layout.addStretch()
        monitors_layout.addLayout(btn_layout)

        # Table for monitors
        self._dam_monitors_table = QTableWidget()
        self._dam_monitors_table.setColumnCount(4)
        self._dam_monitors_table.setHorizontalHeaderLabels(["File", "Condition", "Live", "Status"])
        self._dam_monitors_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._dam_monitors_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._dam_monitors_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._dam_monitors_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._dam_monitors_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._dam_monitors_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._dam_monitors_table.setMinimumHeight(120)
        self._dam_monitors_table.itemSelectionChanged.connect(self._on_dam_selection_changed)
        self._dam_monitors_table.cellChanged.connect(self._on_dam_condition_edited)
        monitors_layout.addWidget(self._dam_monitors_table)

        layout.addWidget(monitors_group)

        # =====================================================================
        # Shared Settings Section
        # =====================================================================
        # Time Settings
        time_group = QGroupBox("Time Settings (shared)")
        time_layout = QHBoxLayout(time_group)

        # Binning
        bin_layout = QVBoxLayout()
        bin_layout.addWidget(QLabel("Binning (min):"))
        self._dam_bin_spin = QSpinBox()
        self._dam_bin_spin.setRange(1, 1440)
        self._dam_bin_spin.setValue(30)
        self._dam_bin_spin.setSuffix(" min")
        self._dam_bin_spin.valueChanged.connect(self._update_dam_preview)
        bin_layout.addWidget(self._dam_bin_spin)
        time_layout.addLayout(bin_layout)

        # Lights ON time (ZT0)
        zt_layout = QVBoxLayout()
        zt_layout.addWidget(QLabel("Lights ON (ZT0):"))
        self._dam_lights_on = QTimeEdit()
        self._dam_lights_on.setTime(QTime(8, 0))
        self._dam_lights_on.setDisplayFormat("HH:mm")
        zt_layout.addWidget(self._dam_lights_on)
        time_layout.addLayout(zt_layout)

        layout.addWidget(time_group)

        # Date Range
        date_group = QGroupBox("Date Range (shared)")
        date_layout = QVBoxLayout(date_group)

        self._dam_use_all_dates = QCheckBox("Use all available dates")
        self._dam_use_all_dates.setChecked(True)
        self._dam_use_all_dates.stateChanged.connect(self._on_dam_date_checkbox_changed)
        date_layout.addWidget(self._dam_use_all_dates)

        date_range_layout = QHBoxLayout()
        date_range_layout.addWidget(QLabel("Start:"))
        self._dam_start_date = QDateEdit()
        self._dam_start_date.setCalendarPopup(True)
        self._dam_start_date.setEnabled(False)
        date_range_layout.addWidget(self._dam_start_date)

        date_range_layout.addWidget(QLabel("End:"))
        self._dam_end_date = QDateEdit()
        self._dam_end_date.setCalendarPopup(True)
        self._dam_end_date.setEnabled(False)
        date_range_layout.addWidget(self._dam_end_date)

        date_layout.addLayout(date_range_layout)
        layout.addWidget(date_group)

        # Dead Fly Filter
        death_group = QGroupBox("Dead Fly Filter (shared)")
        death_layout = QVBoxLayout(death_group)

        self._dam_exclude_dead = QCheckBox("Exclude dead flies")
        self._dam_exclude_dead.setChecked(True)
        self._dam_exclude_dead.stateChanged.connect(self._update_dam_preview)
        death_layout.addWidget(self._dam_exclude_dead)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Death threshold:"))
        self._dam_death_threshold = QDoubleSpinBox()
        self._dam_death_threshold.setRange(1.0, 48.0)
        self._dam_death_threshold.setValue(6.0)
        self._dam_death_threshold.setSuffix(" hours")
        self._dam_death_threshold.setToolTip("Hours of zero activity to consider fly dead")
        self._dam_death_threshold.valueChanged.connect(self._update_dam_preview)
        threshold_layout.addWidget(self._dam_death_threshold)
        threshold_layout.addStretch()

        death_layout.addLayout(threshold_layout)
        layout.addWidget(death_group)

        # =====================================================================
        # Summary Section
        # =====================================================================
        preview_group = QGroupBox("Data Summary")
        preview_layout = QVBoxLayout(preview_group)
        self._dam_preview_label = QLabel("Add DAM files to see summary...")
        self._dam_preview_label.setStyleSheet("color: gray; font-style: italic;")
        preview_layout.addWidget(self._dam_preview_label)
        layout.addWidget(preview_group)

        # Load button
        self._dam_load_btn = QPushButton("Load DAM Data")
        self._dam_load_btn.clicked.connect(self._load_dam_data)
        self._dam_load_btn.setEnabled(False)
        layout.addWidget(self._dam_load_btn)

        # Initialize the multi-loader
        self._dam_loader = MultiDAMDataLoader()

        return group

    def _create_awd_panel(self) -> QGroupBox:
        """Create AWD running wheel file configuration panel with multi-file support."""
        group = QGroupBox("Running Wheel (AWD) Data Configuration")
        layout = QVBoxLayout(group)

        # =====================================================================
        # File List Section
        # =====================================================================
        files_group = QGroupBox("AWD Files (one file per animal)")
        files_layout = QVBoxLayout(files_group)

        btn_layout = QHBoxLayout()
        self._awd_add_btn = QPushButton("+ Add File(s)")
        self._awd_add_btn.clicked.connect(self._add_awd_file)
        self._awd_remove_btn = QPushButton("- Remove Selected")
        self._awd_remove_btn.clicked.connect(self._remove_awd_file)
        self._awd_remove_btn.setEnabled(False)
        btn_layout.addWidget(self._awd_add_btn)
        btn_layout.addWidget(self._awd_remove_btn)
        btn_layout.addStretch()
        files_layout.addLayout(btn_layout)

        self._awd_files_table = QTableWidget()
        self._awd_files_table.setColumnCount(5)
        self._awd_files_table.setHorizontalHeaderLabels(
            ["File", "Subject ID", "Condition", "Days", "Status"]
        )
        self._awd_files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._awd_files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._awd_files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._awd_files_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._awd_files_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self._awd_files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._awd_files_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._awd_files_table.setMinimumHeight(120)
        self._awd_files_table.itemSelectionChanged.connect(self._on_awd_selection_changed)
        self._awd_files_table.cellChanged.connect(self._on_awd_cell_edited)
        files_layout.addWidget(self._awd_files_table)

        layout.addWidget(files_group)

        # =====================================================================
        # Time Settings Section
        # =====================================================================
        time_group = QGroupBox("Time Settings (shared)")
        time_layout = QHBoxLayout(time_group)

        bin_layout = QVBoxLayout()
        bin_layout.addWidget(QLabel("Bin size (min):"))
        self._awd_bin_spin = QSpinBox()
        self._awd_bin_spin.setRange(1, 1440)
        self._awd_bin_spin.setValue(6)
        self._awd_bin_spin.setSuffix(" min")
        self._awd_bin_spin.setToolTip(
            "Native AWD bin is typically 6 min. Set a larger value to re-bin."
        )
        self._awd_bin_spin.valueChanged.connect(self._update_awd_preview)
        bin_layout.addWidget(self._awd_bin_spin)
        time_layout.addLayout(bin_layout)

        zt_layout = QVBoxLayout()
        zt_layout.addWidget(QLabel("Lights ON (ZT0):"))
        self._awd_lights_on = QTimeEdit()
        self._awd_lights_on.setTime(QTime(7, 0))
        self._awd_lights_on.setDisplayFormat("HH:mm")
        zt_layout.addWidget(self._awd_lights_on)
        time_layout.addLayout(zt_layout)

        layout.addWidget(time_group)

        # =====================================================================
        # Date Range Section
        # =====================================================================
        date_group = QGroupBox("Date Range (shared)")
        date_layout = QVBoxLayout(date_group)

        self._awd_use_all_dates = QCheckBox("Use all available dates")
        self._awd_use_all_dates.setChecked(True)
        self._awd_use_all_dates.stateChanged.connect(self._on_awd_date_checkbox_changed)
        date_layout.addWidget(self._awd_use_all_dates)

        date_range_layout = QHBoxLayout()
        date_range_layout.addWidget(QLabel("Start:"))
        self._awd_start_date = QDateEdit()
        self._awd_start_date.setCalendarPopup(True)
        self._awd_start_date.setEnabled(False)
        date_range_layout.addWidget(self._awd_start_date)

        date_range_layout.addWidget(QLabel("End:"))
        self._awd_end_date = QDateEdit()
        self._awd_end_date.setCalendarPopup(True)
        self._awd_end_date.setEnabled(False)
        date_range_layout.addWidget(self._awd_end_date)

        date_layout.addLayout(date_range_layout)
        layout.addWidget(date_group)

        # =====================================================================
        # Summary Section
        # =====================================================================
        preview_group = QGroupBox("Data Summary")
        preview_layout = QVBoxLayout(preview_group)
        self._awd_preview_label = QLabel("Add AWD files to see summary...")
        self._awd_preview_label.setStyleSheet("color: gray; font-style: italic;")
        preview_layout.addWidget(self._awd_preview_label)
        layout.addWidget(preview_group)

        # Load button
        self._awd_load_btn = QPushButton("Load Running Wheel Data")
        self._awd_load_btn.clicked.connect(self._load_awd_data)
        self._awd_load_btn.setEnabled(False)
        layout.addWidget(self._awd_load_btn)

        # Initialize the multi-loader
        self._awd_loader = MultiAWDDataLoader()

        return group

    def _add_dam_monitor(self):
        """Add a new DAM monitor file to the list."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DAM Monitor File(s)",
            "",
            "DAM Files (*.txt);;All Files (*)"
        )

        if not filepaths:
            return

        for filepath in filepaths:
            try:
                # Add to multi-loader
                index = self._dam_loader.add_monitor(filepath)
                monitors = self._dam_loader.get_monitors()
                entry = monitors[index]

                # Add row to table
                row = self._dam_monitors_table.rowCount()
                self._dam_monitors_table.insertRow(row)

                # File name (read-only)
                file_item = QTableWidgetItem(Path(filepath).name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                file_item.setToolTip(filepath)
                self._dam_monitors_table.setItem(row, 0, file_item)

                # Condition name (editable)
                cond_item = QTableWidgetItem(entry.condition_name)
                self._dam_monitors_table.setItem(row, 1, cond_item)

                # Live channels
                live_count = entry.summary.live_channels if entry.summary else 0
                live_item = QTableWidgetItem(str(live_count))
                live_item.setFlags(live_item.flags() & ~Qt.ItemIsEditable)
                self._dam_monitors_table.setItem(row, 2, live_item)

                # Status
                if entry.is_loaded:
                    status_item = QTableWidgetItem("✓ OK")
                    status_item.setForeground(Qt.darkGreen)
                else:
                    status_item = QTableWidgetItem("✗ Error")
                    status_item.setForeground(Qt.red)
                    status_item.setToolTip(entry.error_message or "Unknown error")
                status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
                self._dam_monitors_table.setItem(row, 3, status_item)

            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load {Path(filepath).name}: {e}")

        # Update UI state
        self._update_dam_ui_state()
        self._update_dam_preview()

    def _remove_dam_monitor(self):
        """Remove the selected monitor from the list."""
        selected_rows = self._dam_monitors_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()

        # Remove from loader
        self._dam_loader.remove_monitor(row)

        # Remove from table
        self._dam_monitors_table.removeRow(row)

        # Update UI state
        self._update_dam_ui_state()
        self._update_dam_preview()

    def _on_dam_selection_changed(self):
        """Handle selection change in monitors table."""
        has_selection = len(self._dam_monitors_table.selectionModel().selectedRows()) > 0
        self._dam_remove_btn.setEnabled(has_selection)

    def _on_dam_condition_edited(self, row: int, column: int):
        """Handle condition name edit in table."""
        if column != 1:  # Only condition column is editable
            return

        item = self._dam_monitors_table.item(row, column)
        if item:
            new_name = item.text().strip()
            if new_name:
                self._dam_loader.update_condition_name(row, new_name)

    def _update_dam_ui_state(self):
        """Update UI elements based on current state."""
        monitor_count = self._dam_loader.get_monitor_count()
        self._dam_load_btn.setEnabled(monitor_count > 0)
        self._dam_remove_btn.setEnabled(
            len(self._dam_monitors_table.selectionModel().selectedRows()) > 0
        )

        # Update date range from first monitor if available
        monitors = self._dam_loader.get_monitors()
        if monitors and monitors[0].is_loaded and monitors[0].loader:
            start_dt, end_dt = monitors[0].loader.get_available_dates()
            if start_dt and end_dt:
                self._dam_start_date.setDate(QDate(start_dt.year, start_dt.month, start_dt.day))
                self._dam_end_date.setDate(QDate(end_dt.year, end_dt.month, end_dt.day))

    def _on_dam_date_checkbox_changed(self, state: int):
        """Handle 'use all dates' checkbox change."""
        use_all = state == Qt.Checked
        self._dam_start_date.setEnabled(not use_all)
        self._dam_end_date.setEnabled(not use_all)

    def _update_dam_preview(self):
        """Update DAM preview summary."""
        if self._dam_loader is None or self._dam_loader.get_monitor_count() == 0:
            self._dam_preview_label.setText("Add DAM files to see summary...")
            self._dam_preview_label.setStyleSheet("color: gray; font-style: italic;")
            return

        try:
            # Build config from UI and apply to all monitors
            config = self._build_dam_config()
            self._dam_loader.set_shared_config(config)

            # Refresh table with updated live counts
            monitors = self._dam_loader.get_monitors()
            for row, entry in enumerate(monitors):
                if row < self._dam_monitors_table.rowCount():
                    live_count = entry.summary.live_channels if entry.summary else 0
                    live_item = self._dam_monitors_table.item(row, 2)
                    if live_item:
                        live_item.setText(str(live_count))

            # Get combined summary
            summary = self._dam_loader.get_channel_summary()

            # Build preview text
            n_monitors = summary.get('total_monitors', 0)
            conditions = [e.condition_name for e in monitors if e.is_loaded]
            conditions_str = ", ".join(conditions) if conditions else "-"

            preview_text = (
                f"Monitors: {n_monitors} | Conditions: {conditions_str}\n"
                f"Total Channels: {summary.get('total_channels', 0)} | "
                f"Live: {summary.get('live_channels', 0)} | "
                f"Dead: {summary.get('dead_channels', 0)}\n"
                f"Date range: {summary.get('start_date', '')} to {summary.get('end_date', '')}\n"
                f"Timepoints after binning: {summary.get('timepoints_after_binning', 0)}"
            )

            self._dam_preview_label.setText(preview_text)
            self._dam_preview_label.setStyleSheet("color: black;")

        except Exception as e:
            self._dam_preview_label.setText(f"Error: {e}")
            self._dam_preview_label.setStyleSheet("color: red;")

    def _build_dam_config(self) -> DAMConfig:
        """Build DAMConfig from UI settings (shared config for all monitors)."""
        # Parse binning
        bin_minutes = self._dam_bin_spin.value()

        # Parse lights on time
        lights_on_time = self._dam_lights_on.time()

        # Parse date range
        start_dt = None
        end_dt = None
        if not self._dam_use_all_dates.isChecked():
            start_qdate = self._dam_start_date.date()
            end_qdate = self._dam_end_date.date()
            start_dt = datetime(start_qdate.year(), start_qdate.month(), start_qdate.day())
            end_dt = datetime(end_qdate.year(), end_qdate.month(), end_qdate.day(), 23, 59, 59)

        return DAMConfig(
            bin_size_minutes=bin_minutes,
            lights_on_hour=lights_on_time.hour(),
            lights_on_minute=lights_on_time.minute(),
            start_datetime=start_dt,
            end_datetime=end_dt,
            exclude_dead=self._dam_exclude_dead.isChecked(),
            death_threshold_hours=self._dam_death_threshold.value()
        )

    def _load_dam_data(self):
        """Load DAM data with current configuration."""
        if self._dam_loader is None or self._dam_loader.get_monitor_count() == 0:
            return

        try:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(30)

            # Apply shared config to all monitors
            config = self._build_dam_config()
            self._dam_loader.set_shared_config(config)

            self._progress_bar.setValue(50)

            # Process all monitors and combine data
            self._dam_loader.process()

            self._progress_bar.setValue(70)

            # Validate
            is_valid, issues = self._dam_loader.validate_for_analysis()
            if not is_valid:
                QMessageBox.warning(
                    self, "Validation Issues",
                    "Data loaded with issues:\n- " + "\n- ".join(issues)
                )

            self._current_source = 'dam'

            # Update info labels
            info = self._dam_loader.get_dataset_info()
            n_channels = self._dam_loader.get_channel_count()
            n_monitors = self._dam_loader.get_monitor_count()
            conditions = self._dam_loader.get_conditions()

            self._rows_label.setText(f"Rows: {info.n_rows}")
            self._cols_label.setText(f"Channels: {n_channels}")
            self._conditions_label.setText(f"Conditions: {len(conditions)}")
            self._timepoints_label.setText(f"Timepoints: {len(info.timepoints)}")

            # DAM data is DEPENDENT: same fly (subject) measured repeatedly over time
            self._analysis_type_label.setText("Analysis Type: DEPENDENT")
            self._analysis_type_label.setStyleSheet("font-weight: bold; color: blue;")

            self._progress_bar.setValue(100)
            self._status_label.setText(f"✓ DAM loaded: {n_monitors} monitors, {len(conditions)} conditions")
            self._status_label.setStyleSheet("color: green;")

            # Update preview table
            self._update_preview_table(self._dam_loader.get_preview(10))

            # Emit signal
            self.data_loaded.emit(self._dam_loader, 'dam')

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load DAM data: {e}")
            self._status_label.setText(f"✗ Load failed: {e}")
            self._status_label.setStyleSheet("color: red;")

        finally:
            self._progress_bar.setVisible(False)

    # =========================================================================
    # AWD EVENT HANDLERS & HELPERS
    # =========================================================================

    def _add_awd_file(self):
        """Add one or more AWD animal files to the list."""
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select AWD File(s)",
            "",
            "AWD Files (*.awd);;All Files (*)"
        )

        if not filepaths:
            return

        for filepath in filepaths:
            try:
                index = self._awd_loader.add_file(filepath)
                files = self._awd_loader.get_files()
                entry = files[index]

                self._awd_files_table.blockSignals(True)

                row = self._awd_files_table.rowCount()
                self._awd_files_table.insertRow(row)

                # File name (read-only)
                file_item = QTableWidgetItem(Path(filepath).name)
                file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
                file_item.setToolTip(filepath)
                self._awd_files_table.setItem(row, 0, file_item)

                # Subject ID (editable — auto-detected from header)
                subject_id = (
                    entry.loader._config.subject_id
                    if entry.loader and entry.loader._config.subject_id
                    else Path(filepath).stem
                )
                self._awd_files_table.setItem(row, 1, QTableWidgetItem(subject_id))

                # Condition (editable — user-assigned group)
                self._awd_files_table.setItem(row, 2, QTableWidgetItem(entry.condition_name))

                # Days (read-only)
                days_str = f"{entry.summary.total_days:.1f}" if entry.summary else ""
                days_item = QTableWidgetItem(days_str)
                days_item.setFlags(days_item.flags() & ~Qt.ItemIsEditable)
                self._awd_files_table.setItem(row, 3, days_item)

                # Status (read-only)
                if entry.is_loaded:
                    status_item = QTableWidgetItem("✓ OK")
                    status_item.setForeground(Qt.darkGreen)
                else:
                    status_item = QTableWidgetItem("✗ Error")
                    status_item.setForeground(Qt.red)
                    status_item.setToolTip(entry.error_message or "Unknown error")
                status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
                self._awd_files_table.setItem(row, 4, status_item)

                self._awd_files_table.blockSignals(False)

            except Exception as e:
                self._awd_files_table.blockSignals(False)
                QMessageBox.warning(
                    self, "Load Error", f"Could not load {Path(filepath).name}: {e}"
                )

        self._update_awd_ui_state()
        self._update_awd_preview()

    def _remove_awd_file(self):
        """Remove the selected AWD file from the list."""
        selected_rows = self._awd_files_table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        self._awd_loader.remove_file(row)
        self._awd_files_table.removeRow(row)

        self._update_awd_ui_state()
        self._update_awd_preview()

    def _on_awd_selection_changed(self):
        """Handle selection change in AWD files table."""
        has_selection = len(self._awd_files_table.selectionModel().selectedRows()) > 0
        self._awd_remove_btn.setEnabled(has_selection)

    def _on_awd_cell_edited(self, row: int, column: int):
        """Handle Subject ID or Condition edits in the AWD table."""
        if column not in (1, 2):
            return

        item = self._awd_files_table.item(row, column)
        if not item:
            return

        new_value = item.text().strip()
        if not new_value:
            return

        if column == 1:  # Subject ID
            self._awd_loader.update_subject_id(row, new_value)
        elif column == 2:  # Condition
            self._awd_loader.update_condition_name(row, new_value)

    def _update_awd_ui_state(self):
        """Update AWD UI elements based on current loader state."""
        file_count = self._awd_loader.get_file_count()
        self._awd_load_btn.setEnabled(file_count > 0)
        self._awd_remove_btn.setEnabled(
            len(self._awd_files_table.selectionModel().selectedRows()) > 0
        )

        # Auto-populate date range from the first loaded file
        files = self._awd_loader.get_files()
        if files and files[0].is_loaded and files[0].loader:
            start_dt, end_dt = files[0].loader.get_available_dates()
            if start_dt and end_dt:
                self._awd_start_date.setDate(
                    QDate(start_dt.year, start_dt.month, start_dt.day)
                )
                self._awd_end_date.setDate(
                    QDate(end_dt.year, end_dt.month, end_dt.day)
                )

    def _on_awd_date_checkbox_changed(self, state: int):
        """Handle 'use all dates' checkbox change."""
        use_all = state == Qt.Checked
        self._awd_start_date.setEnabled(not use_all)
        self._awd_end_date.setEnabled(not use_all)

    def _update_awd_preview(self):
        """Update the AWD data summary label."""
        if self._awd_loader is None or self._awd_loader.get_file_count() == 0:
            self._awd_preview_label.setText("Add AWD files to see summary...")
            self._awd_preview_label.setStyleSheet("color: gray; font-style: italic;")
            return

        try:
            config = self._build_awd_config()
            self._awd_loader.set_shared_config(config)

            files = self._awd_loader.get_files()
            summary = self._awd_loader.get_summary()

            n_loaded = summary.get('total_files_loaded', 0)
            # Preserve insertion order, deduplicate conditions
            conditions = list(dict.fromkeys(
                e.condition_name for e in files if e.is_loaded
            ))
            conditions_str = ", ".join(conditions) if conditions else "-"
            total_missing = summary.get('missing_data_points', 0)

            date_range_str = "-"
            if files and files[0].summary:
                s = files[0].summary
                date_range_str = (
                    f"{s.start_datetime.strftime('%Y-%m-%d')} → "
                    f"{s.end_datetime.strftime('%Y-%m-%d')}"
                )

            preview_text = (
                f"Animals: {n_loaded} | Conditions: {conditions_str}\n"
                f"Date range (first file): {date_range_str}\n"
                f"Missing bins (total): {total_missing}"
            )

            self._awd_preview_label.setText(preview_text)
            self._awd_preview_label.setStyleSheet("color: black;")

        except Exception as e:
            self._awd_preview_label.setText(f"Error: {e}")
            self._awd_preview_label.setStyleSheet("color: red;")

    def _build_awd_config(self) -> AWDConfig:
        """Build AWDConfig from current UI settings."""
        lights_on_time = self._awd_lights_on.time()

        start_dt = None
        end_dt = None
        if not self._awd_use_all_dates.isChecked():
            start_qdate = self._awd_start_date.date()
            end_qdate = self._awd_end_date.date()
            start_dt = datetime(
                start_qdate.year(), start_qdate.month(), start_qdate.day()
            )
            end_dt = datetime(
                end_qdate.year(), end_qdate.month(), end_qdate.day(), 23, 59, 59
            )

        return AWDConfig(
            target_bin_size_minutes=self._awd_bin_spin.value(),
            lights_on_hour=lights_on_time.hour(),
            lights_on_minute=lights_on_time.minute(),
            start_datetime=start_dt,
            end_datetime=end_dt,
        )

    def _load_awd_data(self):
        """Load and process all AWD files with current configuration."""
        if self._awd_loader is None or self._awd_loader.get_file_count() == 0:
            return

        try:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(30)

            config = self._build_awd_config()
            self._awd_loader.set_shared_config(config)

            self._progress_bar.setValue(50)

            self._awd_loader.process()

            self._progress_bar.setValue(70)

            is_valid, issues = self._awd_loader.validate_for_analysis()
            if not is_valid:
                QMessageBox.warning(
                    self, "Validation Issues",
                    "Data loaded with issues:\n- " + "\n- ".join(issues)
                )

            self._current_source = 'awd'

            info = self._awd_loader.get_dataset_info()
            n_animals = sum(1 for e in self._awd_loader.get_files() if e.is_loaded)
            conditions = self._awd_loader.get_conditions()

            self._rows_label.setText(f"Rows: {info.n_rows}")
            self._cols_label.setText(f"Animals: {n_animals}")
            self._conditions_label.setText(f"Conditions: {len(conditions)}")
            self._timepoints_label.setText(f"Timepoints: {len(info.timepoints)}")

            self._analysis_type_label.setText("Analysis Type: DEPENDENT")
            self._analysis_type_label.setStyleSheet("font-weight: bold; color: blue;")

            self._progress_bar.setValue(100)
            self._status_label.setText(
                f"✓ AWD loaded: {n_animals} animals, {len(conditions)} conditions"
            )
            self._status_label.setStyleSheet("color: green;")

            self._update_preview_table(self._awd_loader.get_preview(10))

            self.data_loaded.emit(self._awd_loader, 'awd')

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load AWD data: {e}")
            self._status_label.setText(f"✗ Load failed: {e}")
            self._status_label.setStyleSheet("color: red;")

        finally:
            self._progress_bar.setVisible(False)

    def _create_preview_panel(self) -> QGroupBox:
        """Create data preview panel."""
        group = QGroupBox("Data Preview")
        layout = QVBoxLayout(group)
        
        self._preview_table = QTableWidget()
        # self._preview_table.setMaximumHeight(200)  # Commented to allow dynamic resizing
        self._preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self._preview_table)
        
        # Info labels
        info_layout = QHBoxLayout()
        self._rows_label = QLabel("Rows: -")
        self._cols_label = QLabel("Columns: -")
        self._conditions_label = QLabel("Conditions: -")
        self._timepoints_label = QLabel("Timepoints: -")
        self._analysis_type_label = QLabel("Analysis Type: -")
        self._analysis_type_label.setStyleSheet("font-weight: bold;")

        info_layout.addWidget(self._rows_label)
        info_layout.addWidget(self._cols_label)
        info_layout.addWidget(self._conditions_label)
        info_layout.addWidget(self._timepoints_label)
        info_layout.addWidget(self._analysis_type_label)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        return group
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_source_changed(self, index: int):
        """Handle data source selection change."""
        self._csv_panel.setVisible(index == 0)
        self._rosbash_panel.setVisible(index == 1)
        self._dam_panel.setVisible(index == 2)
        self._awd_panel.setVisible(index == 3)
    
    def _browse_csv(self):
        """Open file dialog to select CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)"
        )
        
        if filepath:
            self._csv_path_edit.setText(filepath)
            self._preview_csv(filepath)
            self._csv_load_btn.setEnabled(True)
    
    def _browse_h5(self):
        """Open file dialog to select HDF5 file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Rosbash HDF5 File",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )
        
        if filepath:
            self._h5_path_edit.setText(filepath)
            self._preview_rosbash(filepath)
            self._h5_load_btn.setEnabled(True)
    
    def _preview_csv(self, filepath: str):
        """Preview CSV file and populate column combos."""
        try:
            # Quick load for preview
            df = pd.read_csv(filepath, nrows=100)
            
            # Update column combos
            columns = df.columns.tolist()
            
            self._time_col_combo.clear()
            self._cond_col_combo.clear()
            self._cond_col_combo.addItem("(None)")
            self._subj_col_combo.clear()
            self._subj_col_combo.addItem("(None)")

            for col in columns:
                self._time_col_combo.addItem(col)
                self._cond_col_combo.addItem(col)
                self._subj_col_combo.addItem(col)

            # Auto-select likely columns
            loader = CircadianDataLoader()
            loader.load_dataframe(df, "preview")

            if loader.get_time_column():
                idx = self._time_col_combo.findText(loader.get_time_column())
                if idx >= 0:
                    self._time_col_combo.setCurrentIndex(idx)

            cond_col = loader._condition_col
            if cond_col:
                idx = self._cond_col_combo.findText(cond_col)
                if idx >= 0:
                    self._cond_col_combo.setCurrentIndex(idx)

            # Auto-select subject column
            subj_col = loader._subject_col
            if subj_col:
                idx = self._subj_col_combo.findText(subj_col)
                if idx >= 0:
                    self._subj_col_combo.setCurrentIndex(idx)

            # Update preview table
            self._update_preview_table(df.head(10))
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not preview file: {e}")
    
    def _preview_rosbash(self, filepath: str):
        """Preview Rosbash HDF5 file."""
        try:
            loader = RosbashDataLoader(filepath)

            # Update info labels
            info = loader.get_dataset_info()
            self._rows_label.setText(f"Cells: {info.n_cells}")
            self._cols_label.setText(f"Genes: {info.n_genes}")
            self._conditions_label.setText(f"Conditions: {', '.join(info.conditions)}")
            self._timepoints_label.setText(f"Clusters: {len(info.clusters)}")

            # Rosbash dataset is always independent (different cells)
            self._analysis_type_label.setText("Analysis Type: INDEPENDENT")
            self._analysis_type_label.setStyleSheet("font-weight: bold; color: green;")

            # Store loader temporarily
            self._rosbash_loader = loader

        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not preview file: {e}")
    
    def _update_preview_table(self, df: pd.DataFrame):
        """Update preview table with data."""
        self._preview_table.clear()
        self._preview_table.setRowCount(len(df))
        self._preview_table.setColumnCount(len(df.columns))
        self._preview_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                value = row[col]
                if isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                self._preview_table.setItem(i, j, QTableWidgetItem(text))
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def _load_csv_data(self):
        """Load CSV data with selected configuration."""
        filepath = self._csv_path_edit.text()
        if not filepath:
            return
        
        try:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(30)
            
            loader = CircadianDataLoader()
            loader.load_csv(filepath)
            
            # Apply user column selections
            loader.set_time_column(self._time_col_combo.currentText())

            cond_col = self._cond_col_combo.currentText()
            if cond_col != "(None)":
                loader.set_condition_column(cond_col)

            subj_col = self._subj_col_combo.currentText()
            if subj_col != "(None)":
                loader.set_subject_column(subj_col)
            
            self._progress_bar.setValue(70)
            
            # Validate
            is_valid, issues = loader.validate_for_analysis()
            if not is_valid:
                QMessageBox.warning(
                    self, "Validation Issues",
                    "Data loaded with issues:\n- " + "\n- ".join(issues)
                )
            
            self._csv_loader = loader
            self._current_source = 'csv'
            
            # Update info labels
            info = loader.get_dataset_info()
            self._rows_label.setText(f"Rows: {info.n_rows}")
            self._cols_label.setText(f"Variables: {len(info.variable_columns)}")
            self._conditions_label.setText(f"Conditions: {len(info.conditions)}")
            self._timepoints_label.setText(f"Timepoints: {len(info.timepoints)}")

            # DEBUG: Print all column info
            print(f"[DEBUG ANALYSIS TYPE]")
            print(f"  time_column: {info.time_column}")
            print(f"  condition_column: {info.condition_column}")
            print(f"  replicate_column: {info.replicate_column}")
            print(f"  subject_column: {info.subject_column}")
            print(f"  variable_columns: {info.variable_columns}")

            # Determine analysis type based on what's actually loaded in the loader
            # NOT what's selected in the combo box
            if info.subject_column is not None:
                print(f"  -> Setting DEPENDENT (subject_column = {info.subject_column})")
                self._analysis_type_label.setText("Analysis Type: DEPENDENT")
                self._analysis_type_label.setStyleSheet("font-weight: bold; color: blue;")
            else:
                print(f"  -> Setting INDEPENDENT (subject_column is None)")
                self._analysis_type_label.setText("Analysis Type: INDEPENDENT")
                self._analysis_type_label.setStyleSheet("font-weight: bold; color: green;")
            
            self._progress_bar.setValue(100)
            self._status_label.setText(f"✓ CSV loaded: {Path(filepath).name}")
            self._status_label.setStyleSheet("color: green;")
            
            # Update preview
            self._update_preview_table(loader.get_preview(10))
            
            # Emit signal
            self.data_loaded.emit(loader, 'csv')
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load CSV: {e}")
            self._status_label.setText(f"✗ Load failed: {e}")
            self._status_label.setStyleSheet("color: red;")
        
        finally:
            self._progress_bar.setVisible(False)
    
    def _load_rosbash_data(self):
        """Load Rosbash dataset."""
        if self._rosbash_loader is None:
            return

        try:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(50)

            self._current_source = 'rosbash'

            self._progress_bar.setValue(100)
            info = self._rosbash_loader.get_dataset_info()
            self._status_label.setText(
                f"✓ Rosbash loaded: {info.n_genes} genes available"
            )
            self._status_label.setStyleSheet("color: green;")

            # Emit signal
            self.data_loaded.emit(self._rosbash_loader, 'rosbash')

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load dataset: {e}")
            self._status_label.setText(f"✗ Load failed: {e}")
            self._status_label.setStyleSheet("color: red;")

        finally:
            self._progress_bar.setVisible(False)
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_current_loader(self):
        """Get the current data loader."""
        if self._current_source == 'csv':
            return self._csv_loader
        elif self._current_source == 'rosbash':
            return self._rosbash_loader
        elif self._current_source == 'dam':
            return self._dam_loader
        elif self._current_source == 'awd':
            return self._awd_loader
        return None
    
    def get_current_source_type(self) -> Optional[str]:
        """Get the current data source type."""
        return self._current_source
    
    def get_available_variables(self) -> List[str]:
        """Get list of available variables/genes."""
        if self._current_source == 'csv' and self._csv_loader:
            return self._csv_loader.get_variable_columns()
        elif self._current_source == 'rosbash' and self._rosbash_loader:
            return self._rosbash_loader.get_gene_names()
        elif self._current_source == 'dam' and self._dam_loader:
            return self._dam_loader.get_variable_columns()
        elif self._current_source == 'awd' and self._awd_loader:
            return self._awd_loader.get_variable_columns()
        return []
    
    def get_available_conditions(self) -> List[str]:
        """Get list of available conditions."""
        if self._current_source == 'csv' and self._csv_loader:
            return self._csv_loader.get_conditions()
        elif self._current_source == 'rosbash' and self._rosbash_loader:
            info = self._rosbash_loader.get_dataset_info()
            return info.conditions
        elif self._current_source == 'dam' and self._dam_loader:
            return self._dam_loader.get_conditions()
        elif self._current_source == 'awd' and self._awd_loader:
            return self._awd_loader.get_conditions()
        return []

    def get_available_clusters(self) -> List[str]:
        """Get list of available clusters (for Rosbash data)."""
        if self._current_source == 'rosbash' and self._rosbash_loader:
            return self._rosbash_loader.get_clusters()
        return []
    
    def clear_data(self):
        """Clear all loaded data."""
        self._csv_loader = None
        self._rosbash_loader = None
        self._dam_loader = MultiDAMDataLoader()
        self._awd_loader = MultiAWDDataLoader()
        self._current_source = None

        self._dam_monitors_table.setRowCount(0)
        self._update_dam_ui_state()

        self._awd_files_table.setRowCount(0)
        self._update_awd_ui_state()

        self._preview_table.clear()
        self._status_label.setText("No data loaded")
        self._status_label.setStyleSheet("color: gray; font-style: italic;")

        self.data_cleared.emit()

    def set_source(self, source_type: str):
        """Set the current source type programmatically."""
        if source_type == 'csv':
            self._source_combo.setCurrentIndex(0)
        elif source_type == 'rosbash':
            self._source_combo.setCurrentIndex(1)
        elif source_type == 'dam':
            self._source_combo.setCurrentIndex(2)
        elif source_type == 'awd':
            self._source_combo.setCurrentIndex(3)
