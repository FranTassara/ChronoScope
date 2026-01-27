"""
Main Window
===========

Main application window for CircaScope.
Integrates data loading, analysis configuration, and results visualization.
"""

from typing import Optional
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QMenuBar, QMenu,
    QMessageBox, QFileDialog, QApplication, QLabel,
    QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, QSettings, QSize
from PySide6.QtGui import QAction, QIcon, QKeySequence

from ui.data_panel import DataPanel
from ui.analysis_panel import AnalysisPanel
from ui.results_panel import ResultsPanel


class MainWindow(QMainWindow):
    """
    Main application window for CircaScope.
    
    Features a tab-based interface with:
    1. Data tab - Load and configure data
    2. Analysis tab - Configure and run analyses
    3. Results tab - View results and visualizations
    4. Export tab - Export data and figures
    """
    
    APP_NAME = "CircaScope"
    APP_VERSION = "1.0.0"
    
    def __init__(self):
        super().__init__()
        
        self._settings = QSettings("CircaScope", "CircaScope")
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_settings()
    
    def _setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle(f"{self.APP_NAME} v{self.APP_VERSION}")
        self.setMinimumSize(1000, 700)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Main tab widget
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        
        # Data panel
        self._data_panel = DataPanel()
        self._tabs.addTab(self._data_panel, "📁 Data")
        
        # Analysis panel
        self._analysis_panel = AnalysisPanel()
        self._tabs.addTab(self._analysis_panel, "🔬 Analysis")
        
        # Results panel
        self._results_panel = ResultsPanel()
        self._tabs.addTab(self._results_panel, "📊 Results")
        
        layout.addWidget(self._tabs)
    
    def _setup_menu(self):
        """Setup application menu."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        open_csv = QAction("Open CSV...", self)
        open_csv.setShortcut(QKeySequence.Open)
        open_csv.triggered.connect(self._open_csv)
        file_menu.addAction(open_csv)
        
        open_h5 = QAction("Open Rosbash Dataset...", self)
        open_h5.triggered.connect(self._open_h5)
        file_menu.addAction(open_h5)

        open_dam = QAction("Open DAM Monitor File...", self)
        open_dam.triggered.connect(self._open_dam)
        file_menu.addAction(open_dam)

        file_menu.addSeparator()
        
        export_results = QAction("Export Results...", self)
        export_results.setShortcut(QKeySequence("Ctrl+E"))
        export_results.triggered.connect(self._export_results)
        file_menu.addAction(export_results)
        
        file_menu.addSeparator()
        
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        self._run_action = QAction("Run Analysis", self)
        self._run_action.setShortcut(QKeySequence("Ctrl+R"))
        self._run_action.triggered.connect(self._run_analysis)
        self._run_action.setEnabled(False)  # Disabled until data is loaded
        analysis_menu.addAction(self._run_action)
        
        clear_action = QAction("Clear Results", self)
        clear_action.triggered.connect(self._clear_results)
        analysis_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        data_tab = QAction("Data Tab", self)
        data_tab.setShortcut(QKeySequence("Ctrl+1"))
        data_tab.triggered.connect(lambda: self._tabs.setCurrentIndex(0))
        view_menu.addAction(data_tab)
        
        analysis_tab = QAction("Analysis Tab", self)
        analysis_tab.setShortcut(QKeySequence("Ctrl+2"))
        analysis_tab.triggered.connect(lambda: self._tabs.setCurrentIndex(1))
        view_menu.addAction(analysis_tab)
        
        results_tab = QAction("Results Tab", self)
        results_tab.setShortcut(QKeySequence("Ctrl+3"))
        results_tab.triggered.connect(lambda: self._tabs.setCurrentIndex(2))
        view_menu.addAction(results_tab)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("About CircaScope", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self._show_docs)
        help_menu.addAction(docs_action)
        
    def _setup_statusbar(self):
        """Setup status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        
        # Status label
        self._status_label = QLabel("Ready")
        self._statusbar.addWidget(self._status_label, 1)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._statusbar.addPermanentWidget(self._progress_bar)
        
        # Data status
        self._data_status = QLabel("No data loaded")
        self._data_status.setStyleSheet("color: gray;")
        self._statusbar.addPermanentWidget(self._data_status)
    
    def _connect_signals(self):
        """Connect signals between panels."""
        # Data panel signals
        self._data_panel.data_loaded.connect(self._on_data_loaded)
        self._data_panel.data_cleared.connect(self._on_data_cleared)
        
        # Analysis panel signals
        self._analysis_panel.analysis_started.connect(self._on_analysis_started)
        self._analysis_panel.analysis_completed.connect(self._on_analysis_completed)
        self._analysis_panel.analysis_error.connect(self._on_analysis_error)
    
    def _restore_settings(self):
        """Restore application settings."""
        geometry = self._settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self._settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def _save_settings(self):
        """Save application settings."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
    
    def closeEvent(self, event):
        """Handle window close event."""
        self._save_settings()
        event.accept()
    
    # =========================================================================
    # MENU ACTIONS
    # =========================================================================
    
    def _open_csv(self):
        """Open CSV file dialog."""
        self._tabs.setCurrentIndex(0)
        # Trigger the data panel's browse
        self._data_panel._browse_csv()
    
    def _open_h5(self):
        """Open HDF5 file dialog."""
        self._tabs.setCurrentIndex(0)
        self._data_panel._source_combo.setCurrentIndex(1)
        self._data_panel._browse_h5()

    def _open_dam(self):
        """Open DAM monitor file dialog."""
        self._tabs.setCurrentIndex(0)
        self._data_panel.set_source('dam')
        self._data_panel._browse_dam()

    def _run_analysis(self):
        """Trigger analysis run."""
        self._tabs.setCurrentIndex(1)
        self._analysis_panel._run_analysis()
    
    def _clear_results(self):
        """Clear analysis results."""
        self._results_panel.clear_results()
        self._status_label.setText("Results cleared")
    
    def _export_results(self):
        """Open export dialog."""
        self._tabs.setCurrentIndex(2)
        self._results_panel._export_table('csv')
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            f"About {self.APP_NAME}",
            f"""<h2>{self.APP_NAME} v{self.APP_VERSION}</h2>
            <p>A comprehensive desktop application for circadian rhythm analysis.</p>
            <p>Features:</p>
            <ul>
                <li>CosinorPy integration for cosinor analysis</li>
                <li>CircaCompare for differential rhythmicity</li>
                <li>JTK Cycle, Lomb-Scargle, and more</li>
                <li>Support for Rosbash scRNA-seq dataset</li>
            </ul>
            <p><b>Author:</b> Francisco Tassara</p>
            <p><b>License:</b> MIT</p>
            <p>For research use. Please cite if used in publications.</p>
            """
        )
    
    def _show_docs(self):
        """Show documentation."""
        QMessageBox.information(
            self,
            "Documentation",
            """CircaScope Documentation

Data Input:
- CSV files with columns: time, condition, [variables]
- Rosbash scRNA-seq HDF5 dataset

Analysis Methods:
1. CosinorPy: Single/Multi-component cosinor, population mean
2. CircaCompare: Robust cosinor fitting, group comparison
3. Rhythm Analysis: JTK, Lomb-Scargle, Wavelet, etc.

For detailed documentation, visit:
https://github.com/FranTassara/circascope
            """
        )
    
    # =========================================================================
    # SIGNAL HANDLERS
    # =========================================================================
    
    def _on_data_loaded(self, loader, source_type: str):
        """Handle data loaded signal."""
        self._analysis_panel.set_data(loader, source_type)
        self._run_action.setEnabled(True)

        if source_type == 'csv':
            info = loader.get_dataset_info()
            status = f"CSV: {info.n_rows} rows, {len(info.variable_columns)} variables"
        elif source_type == 'dam':
            n_channels = loader.get_channel_count()
            info = loader.get_dataset_info()
            status = f"DAM: {n_channels} channels, {len(info.timepoints)} timepoints"
        else:
            info = loader.get_dataset_info()
            status = f"Rosbash: {info.n_genes} genes, {info.n_cells} cells"

        self._data_status.setText(status)
        self._data_status.setStyleSheet("color: green;")
        self._status_label.setText("Data loaded successfully")

        # Switch to analysis tab
        self._tabs.setCurrentIndex(1)
    
    def _on_data_cleared(self):
        """Handle data cleared signal."""
        self._analysis_panel.clear_data()
        self._run_action.setEnabled(False)
        self._data_status.setText("No data loaded")
        self._data_status.setStyleSheet("color: gray;")
        self._status_label.setText("Data cleared")
    
    def _on_analysis_started(self):
        """Handle analysis started signal."""
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._status_label.setText("Running analysis...")
        self._run_action.setEnabled(False)
    
    def _on_analysis_completed(self, results):
        """Handle analysis completed signal."""
        print(f"[DEBUG] MainWindow._on_analysis_completed() called")
        print(f"[DEBUG] Received {len(results) if results else 0} results")

        self._progress_bar.setVisible(False)
        self._run_action.setEnabled(True)

        if results:
            print(f"[DEBUG] Calling results_panel.add_results() with {len(results)} results")
            self._results_panel.add_results(results)
            self._status_label.setText(f"Analysis complete: {len(results)} results")

            # Switch to results tab
            print(f"[DEBUG] Switching to results tab (index 2)")
            self._tabs.setCurrentIndex(2)
        else:
            print(f"[DEBUG] No results received")
            self._status_label.setText("Analysis complete: No results")
    
    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error signal."""
        self._progress_bar.setVisible(False)
        self._run_action.setEnabled(True)
        self._status_label.setText(f"Analysis error: {error_msg}")
        
        QMessageBox.critical(self, "Analysis Error", error_msg)


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("CircaScope")
    app.setOrganizationName("CircaScope")
    
    # Set style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
