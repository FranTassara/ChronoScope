"""
Results Panel
=============

Panel for displaying analysis results with tables and visualizations.
"""

from typing import Optional, List, Dict, Any
import math

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QTabWidget, QFrame, QFileDialog,
    QMessageBox, QScrollArea, QSizePolicy, QMenu, QToolButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

import pandas as pd
import numpy as np

# Matplotlib imports for embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas widget for embedding plots."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
    
    def clear(self):
        """Clear the current plot."""
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.draw()
    
    def plot_cosinor_fit(
        self,
        times: np.ndarray,
        values: np.ndarray,
        mesor: float,
        amplitude: float,
        acrophase_rad: float,
        period: float = 24.0,
        title: str = "",
        condition: str = ""
    ):
        """Plot raw data with cosinor fit overlay."""
        self.clear()
        ax = self.axes
        
        # Plot raw data
        ax.scatter(times, values, alpha=0.6, label='Data', color='steelblue')
        
        # Plot fit curve
        t_fit = np.linspace(0, period, 200)
        y_fit = mesor + amplitude * np.cos(2 * np.pi * t_fit / period - acrophase_rad)
        ax.plot(t_fit, y_fit, 'r-', linewidth=2, label='Cosinor Fit')
        
        # Add horizontal line at MESOR
        ax.axhline(y=mesor, color='gray', linestyle='--', alpha=0.5, label=f'MESOR={mesor:.2f}')
        
        # Labels
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Expression')
        ax.set_title(f'{title} - {condition}' if condition else title)
        ax.legend(loc='upper right')
        ax.set_xlim(0, period)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_comparison(
        self,
        result: Dict[str, Any],
        title: str = ""
    ):
        """Plot comparison between two conditions."""
        self.clear()
        ax = self.axes
        
        period = result.get('period', 24.0)
        t_fit = np.linspace(0, period, 200)
        
        # Group 0
        mesor_g0 = result.get('mesor_g0', 0)
        amp_g0 = result.get('amplitude_g0', 0)
        acr_g0 = result.get('acrophase_g0', 0)
        y_g0 = mesor_g0 + amp_g0 * np.cos(2 * np.pi * t_fit / period - acr_g0)
        
        # Group 1
        mesor_g1 = result.get('mesor_g1', 0)
        amp_g1 = result.get('amplitude_g1', 0)
        acr_g1 = result.get('acrophase_g1', 0)
        y_g1 = mesor_g1 + amp_g1 * np.cos(2 * np.pi * t_fit / period - acr_g1)
        
        cond1 = result.get('condition1', 'Group 0')
        cond2 = result.get('condition2', 'Group 1')
        
        ax.plot(t_fit, y_g0, '-', linewidth=2, label=cond1, color='steelblue')
        ax.plot(t_fit, y_g1, '-', linewidth=2, label=cond2, color='coral')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Expression')
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(0, period)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_polar_acrophase(
        self,
        acrophases_hours: List[float],
        labels: List[str],
        period: float = 24.0,
        title: str = "Acrophase Distribution"
    ):
        """Plot acrophases on a polar plot."""
        self.fig.clear()
        ax = self.fig.add_subplot(111, projection='polar')

        # Filter out None values and convert hours to radians
        valid_data = [(h, l) for h, l in zip(acrophases_hours, labels) if h is not None]
        if not valid_data:
            # No valid data to plot
            ax.text(0.5, 0.5, 'No acrophase data available',
                   ha='center', va='center', transform=ax.transAxes)
            self.draw()
            return

        valid_hours, valid_labels = zip(*valid_data)
        thetas = [2 * np.pi * h / period for h in valid_hours]
        
        # Plot each point
        colors = plt.cm.tab10(np.linspace(0, 1, len(thetas)))

        for theta, label, color in zip(thetas, valid_labels, colors):
            ax.scatter(theta, 1, s=100, c=[color], label=label, zorder=5)
            ax.annotate(label, (theta, 1.15), ha='center', fontsize=8)
        
        # Configure polar plot
        ax.set_theta_zero_location('N')  # 0 at top (ZT0)
        ax.set_theta_direction(-1)  # Clockwise
        
        # Set ticks for 24-hour clock
        ax.set_xticks(np.linspace(0, 2*np.pi, 9)[:-1])
        ax.set_xticklabels([f'ZT{int(h)}' for h in np.linspace(0, 24, 9)[:-1]])
        
        ax.set_ylim(0, 1.3)
        ax.set_yticks([])
        ax.set_title(title, y=1.08)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_bar_parameters(
        self,
        results: List[Dict],
        parameter: str = 'amplitude',
        title: str = ""
    ):
        """Plot bar chart of a parameter across conditions."""
        self.clear()
        ax = self.axes
        
        # Extract data, converting None to 0
        labels = [f"{r.get('variable', '')}_{r.get('condition', '')}" for r in results]
        values = [r.get(parameter) if r.get(parameter) is not None else 0 for r in results]
        
        # Check for confidence intervals
        ci_low = [r.get(f'{parameter}_ci', (0, 0))[0] if r.get(f'{parameter}_ci') else 0 for r in results]
        ci_high = [r.get(f'{parameter}_ci', (0, 0))[1] if r.get(f'{parameter}_ci') else 0 for r in results]
        
        errors = None
        if any(ci_low) or any(ci_high):
            errors = [[v - l for v, l in zip(values, ci_low)],
                      [h - v for v, h in zip(values, ci_high)]]
        
        x = np.arange(len(labels))
        bars = ax.bar(x, values, yerr=errors, capsize=3, color='steelblue', alpha=0.7)
        
        ax.set_ylabel(parameter.replace('_', ' ').title())
        ax.set_title(title or f'{parameter.title()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_periodogram(
        self,
        periods: np.ndarray,
        power: np.ndarray,
        dominant_period: float,
        title: str = "Periodogram"
    ):
        """Plot Lomb-Scargle or other periodogram."""
        self.clear()
        ax = self.axes
        
        ax.plot(periods, power, 'b-', linewidth=1)
        ax.axvline(x=dominant_period, color='red', linestyle='--',
                   label=f'Peak: {dominant_period:.1f}h')
        
        ax.set_xlabel('Period (hours)')
        ax.set_ylabel('Power')
        ax.set_title(title)
        ax.legend()
        
        self.fig.tight_layout()
        self.draw()


class ResultsPanel(QWidget):
    """
    Panel for displaying and exporting analysis results.
    
    Features:
    - Summary table with expandable details
    - Multiple plot types (cosinor fit, polar, bar, periodogram)
    - Export to CSV/Excel and image formats
    
    Signals:
        export_requested: Emitted when user wants to export
    """
    
    export_requested = Signal(str)  # format type
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._results: List[Dict[str, Any]] = []
        self._current_result_index: int = -1
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        
        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # Main splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Results table
        table_group = self._create_results_table()
        splitter.addWidget(table_group)
        
        # Visualization tabs
        viz_group = self._create_visualization_tabs()
        splitter.addWidget(viz_group)
        
        splitter.setSizes([300, 400])
        layout.addWidget(splitter)
    
    def _create_toolbar(self) -> QFrame:
        """Create toolbar with export options."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Results info
        self._results_label = QLabel("No results")
        layout.addWidget(self._results_label)
        
        layout.addStretch()
        
        # Export button with menu
        export_btn = QToolButton()
        export_btn.setText("Export ▼")
        export_btn.setPopupMode(QToolButton.InstantPopup)
        
        export_menu = QMenu(export_btn)
        export_menu.addAction("Export Table (CSV)", lambda: self._export_table('csv'))
        export_menu.addAction("Export Table (Excel)", lambda: self._export_table('xlsx'))
        export_menu.addSeparator()
        export_menu.addAction("Export Current Plot (PNG)", lambda: self._export_plot('png'))
        export_menu.addAction("Export Current Plot (SVG)", lambda: self._export_plot('svg'))
        export_menu.addAction("Export Current Plot (PDF)", lambda: self._export_plot('pdf'))
        export_menu.addSeparator()
        export_menu.addAction("Export All Plots", self._export_all_plots)
        
        export_btn.setMenu(export_menu)
        layout.addWidget(export_btn)
        
        # Clear button
        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(clear_btn)
        
        return frame
    
    def _create_results_table(self) -> QGroupBox:
        """Create results table group."""
        group = QGroupBox("Results Summary")
        layout = QVBoxLayout(group)
        
        # Filter row
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        
        self._significance_filter = QComboBox()
        self._significance_filter.addItems(["All", "Significant (p<0.05)", "Non-significant"])
        self._significance_filter.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._significance_filter)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Table
        self._results_table = QTableWidget()
        self._results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._results_table.setSelectionMode(QTableWidget.SingleSelection)
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._results_table.itemSelectionChanged.connect(self._on_result_selected)
        layout.addWidget(self._results_table)
        
        return group
    
    def _create_visualization_tabs(self) -> QGroupBox:
        """Create visualization tabs."""
        group = QGroupBox("Visualization")
        layout = QVBoxLayout(group)
        
        self._viz_tabs = QTabWidget()
        
        # Cosinor fit plot
        self._fit_canvas = PlotCanvas(self, width=6, height=4)
        fit_widget = QWidget()
        fit_layout = QVBoxLayout(fit_widget)
        fit_toolbar = NavigationToolbar(self._fit_canvas, self)
        fit_layout.addWidget(fit_toolbar)
        fit_layout.addWidget(self._fit_canvas)
        self._viz_tabs.addTab(fit_widget, "Cosinor Fit")
        
        # Polar plot
        self._polar_canvas = PlotCanvas(self, width=5, height=5)
        polar_widget = QWidget()
        polar_layout = QVBoxLayout(polar_widget)
        polar_toolbar = NavigationToolbar(self._polar_canvas, self)
        polar_layout.addWidget(polar_toolbar)
        polar_layout.addWidget(self._polar_canvas)
        self._viz_tabs.addTab(polar_widget, "Phase Plot")
        
        # Bar chart
        self._bar_canvas = PlotCanvas(self, width=6, height=4)
        bar_widget = QWidget()
        bar_layout = QVBoxLayout(bar_widget)
        
        # Parameter selector for bar chart
        bar_ctrl_layout = QHBoxLayout()
        bar_ctrl_layout.addWidget(QLabel("Parameter:"))
        self._bar_param_combo = QComboBox()
        self._bar_param_combo.addItems(['amplitude', 'mesor', 'acrophase_hours', 'p_value'])
        self._bar_param_combo.currentTextChanged.connect(self._update_bar_plot)
        bar_ctrl_layout.addWidget(self._bar_param_combo)
        bar_ctrl_layout.addStretch()
        bar_layout.addLayout(bar_ctrl_layout)
        
        bar_toolbar = NavigationToolbar(self._bar_canvas, self)
        bar_layout.addWidget(bar_toolbar)
        bar_layout.addWidget(self._bar_canvas)
        self._viz_tabs.addTab(bar_widget, "Parameter Comparison")
        
        # Periodogram (for Lomb-Scargle etc.)
        self._period_canvas = PlotCanvas(self, width=6, height=4)
        period_widget = QWidget()
        period_layout = QVBoxLayout(period_widget)
        period_toolbar = NavigationToolbar(self._period_canvas, self)
        period_layout.addWidget(period_toolbar)
        period_layout.addWidget(self._period_canvas)
        self._viz_tabs.addTab(period_widget, "Periodogram")
        
        layout.addWidget(self._viz_tabs)
        
        return group
    
    # =========================================================================
    # RESULTS HANDLING
    # =========================================================================
    
    def set_results(self, results: List[Dict[str, Any]]):
        """Set analysis results."""
        self._results = results
        self._update_table()
        self._update_plots()
        
        n_sig = sum(1 for r in results if r.get('p_value') is not None and r.get('p_value') < 0.05)
        self._results_label.setText(
            f"{len(results)} results ({n_sig} significant)"
        )
    
    def add_results(self, results: List[Dict[str, Any]]):
        """Add new results to existing."""
        print(f"[DEBUG] ResultsPanel.add_results() called with {len(results)} results")
        if results:
            print(f"[DEBUG] First result keys: {list(results[0].keys())[:10]}...")
            print(f"[DEBUG] First result variable: {results[0].get('variable')}")
            print(f"[DEBUG] First result condition: {results[0].get('condition')}")

        self._results.extend(results)
        self._update_table()
        self._update_plots()

        n_sig = sum(1 for r in self._results if r.get('p_value') is not None and r.get('p_value') < 0.05)
        self._results_label.setText(
            f"{len(self._results)} results ({n_sig} significant)"
        )
        print(f"[DEBUG] ResultsPanel updated: {len(self._results)} total results")
    
    def clear_results(self):
        """Clear all results."""
        self._results = []
        self._results_table.setRowCount(0)
        self._fit_canvas.clear()
        self._polar_canvas.clear()
        self._bar_canvas.clear()
        self._period_canvas.clear()
        self._results_label.setText("No results")
        self._current_result_index = -1
    
    def _update_table(self):
        """Update results table."""
        if not self._results:
            self._results_table.setRowCount(0)
            return
        
        # Determine columns
        columns = ['variable', 'condition', 'method', 'mesor', 'amplitude',
                   'acrophase_hours', 'period', 'p_value', 'significant']
        
        self._results_table.setColumnCount(len(columns))
        self._results_table.setHorizontalHeaderLabels(
            [c.replace('_', ' ').title() for c in columns]
        )
        
        # Apply filter
        filtered = self._get_filtered_results()
        self._results_table.setRowCount(len(filtered))
        
        for i, result in enumerate(filtered):
            for j, col in enumerate(columns):
                if col == 'significant':
                    p_val = result.get('p_value')
                    value = 'Yes' if (p_val is not None and p_val < 0.05) else ('No' if p_val is not None else 'N/A')
                else:
                    value = result.get(col, '-')
                    if isinstance(value, float):
                        if col == 'p_value':
                            value = f'{value:.2e}' if value < 0.001 else f'{value:.4f}'
                        else:
                            value = f'{value:.3f}'
                    elif value is None:
                        value = 'N/A'

                item = QTableWidgetItem(str(value))

                # Color code significance
                if col == 'p_value' or col == 'significant':
                    p_val = result.get('p_value')
                    if p_val is not None and p_val < 0.05:
                        item.setBackground(Qt.green)
                
                self._results_table.setItem(i, j, item)
    
    def _get_filtered_results(self) -> List[Dict]:
        """Get filtered results based on current filter."""
        filter_idx = self._significance_filter.currentIndex()

        if filter_idx == 0:  # All
            return self._results
        elif filter_idx == 1:  # Significant
            return [r for r in self._results if r.get('p_value') is not None and r.get('p_value') < 0.05]
        else:  # Non-significant
            return [r for r in self._results if r.get('p_value') is not None and r.get('p_value') >= 0.05]
    
    def _apply_filter(self):
        """Apply significance filter."""
        self._update_table()
    
    def _on_result_selected(self):
        """Handle result selection in table."""
        rows = self._results_table.selectedIndexes()
        if not rows:
            return
        
        row = rows[0].row()
        filtered = self._get_filtered_results()
        
        if 0 <= row < len(filtered):
            self._current_result_index = row
            self._update_fit_plot(filtered[row])
    
    def _update_plots(self):
        """Update all plots."""
        if not self._results:
            return
        
        # Update bar plot
        self._update_bar_plot()
        
        # Update polar plot with all acrophases
        acrophases = [r.get('acrophase_hours') for r in self._results if 'acrophase_hours' in r]
        labels = [f"{r.get('variable', '')}_{r.get('condition', '')}" for r in self._results if 'acrophase_hours' in r]
        
        if acrophases:
            self._polar_canvas.plot_polar_acrophase(acrophases, labels)
        
        # Update fit plot with first result
        if self._results:
            self._update_fit_plot(self._results[0])
    
    def _update_fit_plot(self, result: Dict):
        """Update cosinor fit plot for selected result."""
        mesor = result.get('mesor', 0)
        amplitude = result.get('amplitude', 0)
        acrophase_rad = result.get('acrophase', result.get('acrophase_rad', 0))
        period = result.get('period', 24.0)

        # Get times and values from result, or generate synthetic data
        times_data = result.get('times')
        values_data = result.get('values')

        if times_data is not None and values_data is not None:
            # Convert from list to numpy array if needed
            times = np.array(times_data) if not isinstance(times_data, np.ndarray) else times_data
            values = np.array(values_data) if not isinstance(values_data, np.ndarray) else values_data
        else:
            # Generate synthetic data points if not available
            times = np.linspace(0, period, 24)
            values = mesor + amplitude * np.cos(
                2 * np.pi * times / period - acrophase_rad) + np.random.normal(0, amplitude * 0.1, len(times))

        variable = result.get('variable', '')
        condition = result.get('condition', '')

        self._fit_canvas.plot_cosinor_fit(
            times, values, mesor, amplitude, acrophase_rad,
            period, title=variable, condition=condition
        )
    
    def _update_bar_plot(self):
        """Update bar parameter plot."""
        if not self._results:
            return
        
        param = self._bar_param_combo.currentText()
        self._bar_canvas.plot_bar_parameters(
            self._results, parameter=param,
            title=f'{param.replace("_", " ").title()} Across Conditions'
        )
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def _export_table(self, format: str):
        """Export results table to file."""
        if not self._results:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self._results)
        
        # File dialog
        if format == 'csv':
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "circascope_results.csv",
                "CSV Files (*.csv)"
            )
        else:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "circascope_results.xlsx",
                "Excel Files (*.xlsx)"
            )
        
        if filepath:
            try:
                if format == 'csv':
                    df.to_csv(filepath, index=False)
                else:
                    df.to_excel(filepath, index=False)
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"Results exported to {filepath}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def _export_plot(self, format: str):
        """Export current plot to file."""
        # Get current canvas
        tab_idx = self._viz_tabs.currentIndex()
        canvases = [self._fit_canvas, self._polar_canvas, self._bar_canvas, self._period_canvas]
        canvas = canvases[tab_idx]
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"circascope_plot.{format}",
            f"{format.upper()} Files (*.{format})"
        )
        
        if filepath:
            try:
                canvas.fig.savefig(filepath, format=format, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Export Complete", f"Plot exported to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def _export_all_plots(self):
        """Export all plots to a directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        
        if directory:
            try:
                from pathlib import Path
                dir_path = Path(directory)
                
                self._fit_canvas.fig.savefig(dir_path / "cosinor_fit.png", dpi=300)
                self._polar_canvas.fig.savefig(dir_path / "phase_plot.png", dpi=300)
                self._bar_canvas.fig.savefig(dir_path / "parameter_comparison.png", dpi=300)
                self._period_canvas.fig.savefig(dir_path / "periodogram.png", dpi=300)
                
                QMessageBox.information(
                    self, "Export Complete",
                    f"All plots exported to {directory}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        return pd.DataFrame(self._results) if self._results else pd.DataFrame()
    
    def get_current_figure(self) -> Figure:
        """Get current matplotlib figure."""
        tab_idx = self._viz_tabs.currentIndex()
        canvases = [self._fit_canvas, self._polar_canvas, self._bar_canvas, self._period_canvas]
        return canvases[tab_idx].fig
