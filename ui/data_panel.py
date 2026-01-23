"""
Data Panel
==========

Panel for loading and configuring data sources (CSV or Rosbash dataset).
"""

from typing import Optional, List, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QSpinBox, QCheckBox,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QSplitter, QFrame, QLineEdit, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

import pandas as pd

from utils.data_loader import CircadianDataLoader, DatasetInfo
from utils.rosbash_loader import RosbashDataLoader, get_available_clock_genes


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
        
        # Initially show CSV panel
        self._rosbash_panel.setVisible(False)
        
        layout.addWidget(self._csv_panel)
        layout.addWidget(self._rosbash_panel)
        
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
            "Rosbash scRNA-seq Dataset"
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
        
        # Replicate column (optional)
        rep_layout = QVBoxLayout()
        rep_layout.addWidget(QLabel("Replicate (optional):"))
        self._rep_col_combo = QComboBox()
        self._rep_col_combo.addItem("(None)")
        self._rep_col_combo.setMinimumWidth(120)
        self._rep_col_combo.setToolTip("For INDEPENDENT data: Different individuals at each timepoint")
        rep_layout.addWidget(self._rep_col_combo)
        mapping_layout.addLayout(rep_layout)

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
        if index == 0:  # CSV
            self._csv_panel.setVisible(True)
            self._rosbash_panel.setVisible(False)
        else:  # Rosbash
            self._csv_panel.setVisible(False)
            self._rosbash_panel.setVisible(True)
    
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
            self._rep_col_combo.clear()
            self._rep_col_combo.addItem("(None)")
            self._subj_col_combo.clear()
            self._subj_col_combo.addItem("(None)")

            for col in columns:
                self._time_col_combo.addItem(col)
                self._cond_col_combo.addItem(col)
                self._rep_col_combo.addItem(col)
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

            # Auto-select replicate column
            rep_col = loader._replicate_col
            if rep_col:
                idx = self._rep_col_combo.findText(rep_col)
                if idx >= 0:
                    self._rep_col_combo.setCurrentIndex(idx)

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

            rep_col = self._rep_col_combo.currentText()
            if rep_col != "(None)":
                loader.set_replicate_column(rep_col)

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
        return []
    
    def get_available_conditions(self) -> List[str]:
        """Get list of available conditions."""
        if self._current_source == 'csv' and self._csv_loader:
            return self._csv_loader.get_conditions()
        elif self._current_source == 'rosbash' and self._rosbash_loader:
            info = self._rosbash_loader.get_dataset_info()
            return info.conditions
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
        self._current_source = None
        
        self._preview_table.clear()
        self._status_label.setText("No data loaded")
        self._status_label.setStyleSheet("color: gray; font-style: italic;")
        
        self.data_cleared.emit()
