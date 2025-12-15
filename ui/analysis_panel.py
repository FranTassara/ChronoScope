"""
Analysis Panel
==============

Panel for configuring and executing circadian rhythm analyses.
Supports multiple analysis methods from CosinorPy, CircaCompare, and RhythmAnalysis modules.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QStackedWidget, QFormLayout, QFrame, QMessageBox,
    QProgressBar, QScrollArea, QSizePolicy, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QThread

import pandas as pd
import numpy as np


class AnalysisCategory(Enum):
    """Categories of analysis types."""
    SINGLE_GROUP = "Single Group Analysis"
    GROUP_COMPARISON = "Group Comparison"
    BATCH_ANALYSIS = "Batch Analysis"


class AnalysisMethod(Enum):
    """Available analysis methods."""
    # CosinorPy methods
    COSINORPY_SINGLE = "CosinorPy: Single Cosinor"
    COSINORPY_MULTI = "CosinorPy: Multi-Component"
    COSINORPY_POPULATION = "CosinorPy: Population Mean"
    COSINORPY_COMPARE = "CosinorPy: Compare Conditions"
    COSINORPY_COUNT = "CosinorPy: Count Data"
    COSINORPY_NONLINEAR = "CosinorPy: Nonlinear Cosinor"

    # CircaCompare methods
    CIRCACOMPARE_SINGLE = "CircaCompare: Single Fit"
    CIRCACOMPARE_COMPARE = "CircaCompare: Compare Groups"

    # Rhythm Analysis methods
    RHYTHM_JTK = "JTK Cycle"
    RHYTHM_AR_JTK = "AR-JTK Cycle"
    RHYTHM_COSINE_KENDALL = "Cosine-Kendall"
    RHYTHM_COSINOR = "Cosinor (OLS)"
    RHYTHM_HARMONIC = "Harmonic Cosinor"
    RHYTHM_F24 = "Fourier F24"
    RHYTHM_LOMB = "Lomb-Scargle"
    RHYTHM_CWT = "Wavelet (CWT)"
    RHYTHM_LME = "Linear Mixed Effects"


@dataclass
class AnalysisConfig:
    """Configuration for an analysis run."""
    method: AnalysisMethod
    variables: List[str]
    conditions: List[str]
    parameters: Dict[str, Any]
    compare_conditions: Optional[tuple] = None  # (condition1, condition2)


class AnalysisWorker(QThread):
    """Worker thread for running analyses."""
    finished = Signal(bool, object, str)  # success, results, message
    progress = Signal(int, str)  # percentage, status message

    def __init__(self, config: AnalysisConfig, loader, source_type: str):
        super().__init__()
        self.config = config
        self.loader = loader
        self.source_type = source_type
        self.results = []

    def run(self):
        try:
            # Import analysis engine
            from core.analysis_engine import AnalysisEngine, AnalysisType

            engine = AnalysisEngine()

            # Get the full dataset
            if self.source_type == 'csv':
                data = self.loader.get_data()
                time_col = self.loader.get_time_column()
                condition_col = self.loader._condition_col or 'condition'
            else:  # rosbash
                # For Rosbash, we need to prepare data differently
                # This is a simplified placeholder - you may need to adjust
                data = None  # Will be handled in _run_single_analysis
                time_col = 'time'
                condition_col = 'condition'

            # Map UI method to engine AnalysisType
            analysis_type = self._map_method_to_type(self.config.method)

            total = len(self.config.variables) * len(self.config.conditions)
            current = 0

            for var in self.config.variables:
                for cond in self.config.conditions:
                    self.progress.emit(
                        int(current / total * 100),
                        f"Analyzing {var} in {cond}..."
                    )

                    if self.source_type == 'csv':
                        result = engine.run_analysis(
                            data, var, cond, analysis_type,
                            time_col=time_col,
                            condition_col=condition_col,
                            parameters=self.config.parameters
                        )
                    else:  # rosbash
                        # For Rosbash, need special handling
                        result = self._run_rosbash_analysis(var, cond, analysis_type)

                    if result and result.success:
                        self.results.append(result.to_dict())
                    elif result:
                        # Include failed results with error message
                        self.results.append(result.to_dict())

                    current += 1

            self.progress.emit(100, "Complete")
            self.finished.emit(True, self.results, "Analysis completed successfully")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, None, str(e))

    def _map_method_to_type(self, method: AnalysisMethod):
        """Map UI AnalysisMethod to engine AnalysisType."""
        from core.analysis_engine import AnalysisType

        mapping = {
            AnalysisMethod.COSINORPY_SINGLE: AnalysisType.COSINORPY_SINGLE,
            AnalysisMethod.COSINORPY_MULTI: AnalysisType.COSINORPY_MULTI,
            AnalysisMethod.COSINORPY_POPULATION: AnalysisType.COSINORPY_POPULATION,
            AnalysisMethod.COSINORPY_COMPARE: AnalysisType.COSINORPY_COMPARE,
            AnalysisMethod.COSINORPY_COUNT: AnalysisType.COSINORPY_COUNT,
            AnalysisMethod.COSINORPY_NONLINEAR: AnalysisType.COSINORPY_NONLINEAR,
            AnalysisMethod.CIRCACOMPARE_SINGLE: AnalysisType.CIRCACOMPARE_SINGLE,
            AnalysisMethod.CIRCACOMPARE_COMPARE: AnalysisType.CIRCACOMPARE_COMPARE,
            AnalysisMethod.RHYTHM_JTK: AnalysisType.JTK,
            AnalysisMethod.RHYTHM_AR_JTK: AnalysisType.AR_JTK,
            AnalysisMethod.RHYTHM_COSINE_KENDALL: AnalysisType.COSINE_KENDALL,
            AnalysisMethod.RHYTHM_COSINOR: AnalysisType.COSINOR_OLS,
            AnalysisMethod.RHYTHM_HARMONIC: AnalysisType.HARMONIC_COSINOR,
            AnalysisMethod.RHYTHM_F24: AnalysisType.FOURIER_F24,
            AnalysisMethod.RHYTHM_LOMB: AnalysisType.LOMB_SCARGLE,
            AnalysisMethod.RHYTHM_CWT: AnalysisType.CWT,
            AnalysisMethod.RHYTHM_LME: AnalysisType.LME,
        }
        return mapping.get(method, AnalysisType.COSINORPY_SINGLE)

    def _run_rosbash_analysis(self, gene: str, condition: str, analysis_type):
        """Run analysis on Rosbash data (placeholder for now)."""
        from core.analysis_engine import AnalysisResult

        # TODO: Implement Rosbash data analysis
        # This would require extracting the gene expression data from the HDF5
        # and formatting it properly for the analysis engine
        return AnalysisResult(
            variable=gene,
            condition=condition,
            method=analysis_type.value,
            success=False,
            message="Rosbash analysis not yet implemented"
        )


class ParameterWidget(QWidget):
    """Base widget for analysis parameters."""
    
    value_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def get_value(self) -> Any:
        raise NotImplementedError
    
    def set_value(self, value: Any):
        raise NotImplementedError


class AnalysisPanel(QWidget):
    """
    Panel for configuring and running circadian analyses.
    
    Dynamically shows relevant parameters based on selected method.
    
    Signals:
        analysis_started: Emitted when analysis begins
        analysis_completed: Emitted with results when analysis finishes
        analysis_error: Emitted with error message on failure
    """
    
    analysis_started = Signal()
    analysis_completed = Signal(list)  # List of result dicts
    analysis_error = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)

        self._loader = None
        self._source_type: Optional[str] = None
        self._available_variables: List[str] = []
        self._available_conditions: List[str] = []
        self._all_genes: List[str] = []

        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Method selection
        method_group = self._create_method_selection()
        layout.addWidget(method_group)
        
        # Data selection
        data_group = self._create_data_selection()
        layout.addWidget(data_group)
        
        # Parameter configuration (stacked widget for different methods)
        params_group = self._create_parameters_section()
        layout.addWidget(params_group)
        
        # Run controls
        run_group = self._create_run_controls()
        layout.addWidget(run_group)
        
        layout.addStretch()
    
    def _create_method_selection(self) -> QGroupBox:
        """Create method selection group."""
        group = QGroupBox("Analysis Method")
        layout = QVBoxLayout(group)
        
        # Module selection
        module_layout = QHBoxLayout()
        module_layout.addWidget(QLabel("Module:"))
        
        self._module_combo = QComboBox()
        self._module_combo.addItems([
            "CosinorPy",
            "CircaCompare",
            "Rhythm Analysis"
        ])
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)
        module_layout.addWidget(self._module_combo, 1)
        layout.addLayout(module_layout)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self._method_combo = QComboBox()
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self._method_combo, 1)
        layout.addLayout(method_layout)
        
        # Method description
        self._method_desc = QLabel("")
        self._method_desc.setWordWrap(True)
        self._method_desc.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._method_desc)
        
        # Initialize methods
        self._on_module_changed(0)
        
        return group
    
    def _create_data_selection(self) -> QGroupBox:
        """Create data/variable selection group."""
        group = QGroupBox("Data Selection")
        layout = QVBoxLayout(group)

        # Main horizontal layout for variables/genes and conditions
        main_h_layout = QHBoxLayout()

        # Variables/Genes
        var_layout = QVBoxLayout()
        self._var_label = QLabel("Variables:")
        var_layout.addWidget(self._var_label)

        # Search box for genes (visible only for Rosbash)
        self._gene_search = QLineEdit()
        self._gene_search.setPlaceholderText("Search genes...")
        self._gene_search.textChanged.connect(self._filter_genes)
        self._gene_search.setVisible(False)
        var_layout.addWidget(self._gene_search)

        self._var_list = QListWidget()
        self._var_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._var_list.setMaximumHeight(120)
        var_layout.addWidget(self._var_list)

        var_btn_layout = QHBoxLayout()
        select_all = QPushButton("All")
        select_all.clicked.connect(lambda: self._select_all_items(self._var_list))
        clear_sel = QPushButton("None")
        clear_sel.clicked.connect(lambda: self._clear_selection(self._var_list))

        # Add clock genes button (for Rosbash)
        self._clock_genes_btn = QPushButton("Clock Genes")
        self._clock_genes_btn.clicked.connect(self._add_clock_genes)
        self._clock_genes_btn.setVisible(False)

        var_btn_layout.addWidget(select_all)
        var_btn_layout.addWidget(clear_sel)
        var_btn_layout.addWidget(self._clock_genes_btn)
        var_layout.addLayout(var_btn_layout)

        main_h_layout.addLayout(var_layout)

        # Conditions
        cond_layout = QVBoxLayout()
        cond_layout.addWidget(QLabel("Conditions:"))
        self._cond_list = QListWidget()
        self._cond_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._cond_list.setMaximumHeight(120)
        cond_layout.addWidget(self._cond_list)

        cond_btn_layout = QHBoxLayout()
        select_all_c = QPushButton("All")
        select_all_c.clicked.connect(lambda: self._select_all_items(self._cond_list))
        clear_sel_c = QPushButton("None")
        clear_sel_c.clicked.connect(lambda: self._clear_selection(self._cond_list))
        cond_btn_layout.addWidget(select_all_c)
        cond_btn_layout.addWidget(clear_sel_c)
        cond_layout.addLayout(cond_btn_layout)

        main_h_layout.addLayout(cond_layout)

        layout.addLayout(main_h_layout)

        # Rosbash-specific: Cluster selection (hidden for CSV)
        self._cluster_frame = QFrame()
        cluster_layout = QVBoxLayout(self._cluster_frame)
        cluster_layout.setContentsMargins(0, 0, 0, 0)
        cluster_layout.addWidget(QLabel("Clusters:"))
        self._cluster_list = QListWidget()
        self._cluster_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._cluster_list.setMaximumHeight(100)
        cluster_layout.addWidget(self._cluster_list)

        cluster_btn_layout = QHBoxLayout()
        select_all_cl = QPushButton("All")
        select_all_cl.clicked.connect(lambda: self._select_all_items(self._cluster_list))
        clear_sel_cl = QPushButton("None")
        clear_sel_cl.clicked.connect(lambda: self._clear_selection(self._cluster_list))
        cluster_btn_layout.addWidget(select_all_cl)
        cluster_btn_layout.addWidget(clear_sel_cl)
        cluster_layout.addLayout(cluster_btn_layout)

        self._cluster_frame.setVisible(False)
        layout.addWidget(self._cluster_frame)
        
        # Comparison selection (for comparison methods)
        self._compare_frame = QFrame()
        compare_layout = QVBoxLayout(self._compare_frame)
        compare_layout.addWidget(QLabel("Compare:"))
        
        compare_h = QHBoxLayout()
        self._cond1_combo = QComboBox()
        compare_h.addWidget(self._cond1_combo)
        compare_h.addWidget(QLabel("vs"))
        self._cond2_combo = QComboBox()
        compare_h.addWidget(self._cond2_combo)
        compare_layout.addLayout(compare_h)
        
        self._compare_frame.setVisible(False)
        layout.addWidget(self._compare_frame)
        
        return group
    
    def _create_parameters_section(self) -> QGroupBox:
        """Create parameters configuration section."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout(group)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(250)
        
        self._params_widget = QWidget()
        self._params_layout = QFormLayout(self._params_widget)
        scroll.setWidget(self._params_widget)
        
        layout.addWidget(scroll)
        
        # Initialize with default parameters
        self._setup_default_parameters()
        
        return group
    
    def _create_run_controls(self) -> QGroupBox:
        """Create run controls group."""
        group = QGroupBox("Run Analysis")
        layout = QVBoxLayout(group)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: gray;")
        layout.addWidget(self._status_label)
        
        # Run button
        btn_layout = QHBoxLayout()
        
        self._run_btn = QPushButton("▶ Run Analysis")
        self._run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self._run_btn.clicked.connect(self._run_analysis)
        self._run_btn.setEnabled(False)
        btn_layout.addWidget(self._run_btn)
        
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_analysis)
        btn_layout.addWidget(self._cancel_btn)
        
        layout.addLayout(btn_layout)
        
        return group
    
    def _setup_default_parameters(self):
        """Setup default parameter widgets."""
        # Clear existing
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Period
        self._period_spin = QDoubleSpinBox()
        self._period_spin.setRange(1, 72)
        self._period_spin.setValue(24.0)
        self._period_spin.setSuffix(" hours")
        self._params_layout.addRow("Period:", self._period_spin)
        
        # Number of components
        self._components_spin = QSpinBox()
        self._components_spin.setRange(1, 6)
        self._components_spin.setValue(1)
        self._params_layout.addRow("Components:", self._components_spin)
        
        # Period range (for methods that search)
        period_range_widget = QWidget()
        pr_layout = QHBoxLayout(period_range_widget)
        pr_layout.setContentsMargins(0, 0, 0, 0)
        
        self._period_min_spin = QDoubleSpinBox()
        self._period_min_spin.setRange(1, 48)
        self._period_min_spin.setValue(20.0)
        pr_layout.addWidget(self._period_min_spin)
        
        pr_layout.addWidget(QLabel("to"))
        
        self._period_max_spin = QDoubleSpinBox()
        self._period_max_spin.setRange(1, 48)
        self._period_max_spin.setValue(28.0)
        pr_layout.addWidget(self._period_max_spin)
        
        pr_layout.addWidget(QLabel("hours"))
        self._params_layout.addRow("Period Range:", period_range_widget)
        
        # Loss function (for CircaCompare)
        self._loss_combo = QComboBox()
        self._loss_combo.addItems(['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'])
        self._params_layout.addRow("Loss Function:", self._loss_combo)
        
        # F-scale
        self._fscale_spin = QDoubleSpinBox()
        self._fscale_spin.setRange(0.1, 10.0)
        self._fscale_spin.setValue(1.0)
        self._fscale_spin.setSingleStep(0.1)
        self._params_layout.addRow("F-Scale:", self._fscale_spin)

        # Max iterations (for CircaCompare)
        self._max_iterations_spin = QSpinBox()
        self._max_iterations_spin.setRange(100, 2000)
        self._max_iterations_spin.setValue(500)
        self._max_iterations_spin.setSingleStep(100)
        self._params_layout.addRow("Max Iterations:", self._max_iterations_spin)

        # Harmonics
        self._harmonics_spin = QSpinBox()
        self._harmonics_spin.setRange(1, 6)
        self._harmonics_spin.setValue(2)
        self._params_layout.addRow("Harmonics:", self._harmonics_spin)
        
        # Permutations (for F24)
        self._permutations_spin = QSpinBox()
        self._permutations_spin.setRange(100, 10000)
        self._permutations_spin.setValue(1000)
        self._permutations_spin.setSingleStep(100)
        self._params_layout.addRow("Permutations:", self._permutations_spin)
        
        # Update visibility based on current method
        self._update_parameter_visibility()
    
    def _update_parameter_visibility(self):
        """Show/hide parameters based on selected method."""
        # Check if params layout exists (may be called during initialization)
        if not hasattr(self, '_params_layout') or self._params_layout is None:
            return

        method_text = self._method_combo.currentText()
        module_text = self._module_combo.currentText()

        # Default: hide all optional params
        self._hide_param("Components:")
        self._hide_param("Period Range:")
        self._hide_param("Loss Function:")
        self._hide_param("F-Scale:")
        self._hide_param("Max Iterations:")
        self._hide_param("Harmonics:")
        self._hide_param("Permutations:")

        # Show relevant params based on method
        if "Multi-Component" in method_text:
            self._show_param("Components:")

        # Show CircaCompare params when CircaCompare module is selected
        if module_text == "CircaCompare":
            self._show_param("Loss Function:")
            self._show_param("F-Scale:")
            self._show_param("Max Iterations:")
        
        if "Harmonic" in method_text:
            self._show_param("Harmonics:")
        
        if "JTK" in method_text or "Lomb" in method_text or "OLS" in method_text:
            self._show_param("Period Range:")
        
        if "F24" in method_text:
            self._show_param("Permutations:")
        
        # Show comparison frame for comparison methods
        is_comparison = "Compare" in method_text
        self._compare_frame.setVisible(is_comparison)
        self._cond_list.setVisible(not is_comparison)
    
    def _show_param(self, label: str):
        """Show a parameter row."""
        for i in range(self._params_layout.rowCount()):
            item = self._params_layout.itemAt(i, QFormLayout.LabelRole)
            if item and item.widget() and item.widget().text() == label:
                item.widget().setVisible(True)
                field = self._params_layout.itemAt(i, QFormLayout.FieldRole)
                if field and field.widget():
                    field.widget().setVisible(True)
    
    def _hide_param(self, label: str):
        """Hide a parameter row."""
        for i in range(self._params_layout.rowCount()):
            item = self._params_layout.itemAt(i, QFormLayout.LabelRole)
            if item and item.widget() and item.widget().text() == label:
                item.widget().setVisible(False)
                field = self._params_layout.itemAt(i, QFormLayout.FieldRole)
                if field and field.widget():
                    field.widget().setVisible(False)
    
    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================
    
    def _on_module_changed(self, index: int):
        """Handle module selection change."""
        self._method_combo.clear()
        
        if index == 0:  # CosinorPy
            methods = [
                ("Single Cosinor", "Fit single-component cosinor model"),
                ("Multi-Component", "Fit multi-harmonic cosinor model"),
                ("Population Mean", "For dependent/longitudinal data"),
                ("Compare Conditions", "Differential rhythmicity analysis"),
                ("Count Data (Poisson)", "For RNA-seq counts")
            ]
        elif index == 1:  # CircaCompare
            methods = [
                ("Single Fit", "Robust cosinor fitting"),
                ("Compare Groups", "Compare parameters between groups")
            ]
        else:  # Rhythm Analysis
            methods = [
                ("JTK Cycle", "Nonparametric rhythm detection"),
                ("AR-JTK", "JTK with autoregressive correction"),
                ("Cosinor (OLS)", "Parametric cosinor with period search"),
                ("Harmonic Cosinor", "Multi-modal rhythm detection"),
                ("Fourier F24", "Effect size measure (requires 2 replicates)"),
                ("Lomb-Scargle", "For unevenly sampled data"),
                ("Wavelet (CWT)", "Time-frequency analysis"),
                ("Linear Mixed Effects", "Hierarchical modeling")
            ]
        
        for name, desc in methods:
            self._method_combo.addItem(name, desc)
        
        self._on_method_changed(0)
    
    def _on_method_changed(self, index: int):
        """Handle method selection change."""
        desc = self._method_combo.currentData()
        self._method_desc.setText(desc or "")
        self._update_parameter_visibility()
    
    def _select_all_items(self, list_widget: QListWidget):
        """Select all items in a list widget."""
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(True)
    
    def _clear_selection(self, list_widget: QListWidget):
        """Clear selection in a list widget."""
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(False)

    def _filter_genes(self, text: str):
        """Filter gene list based on search text (for Rosbash data)."""
        if not hasattr(self, '_all_genes') or not self._all_genes:
            return

        self._var_list.clear()

        if text:
            matches = [g for g in self._all_genes if text.lower() in g.lower()][:200]
        else:
            matches = self._all_genes[:500]

        for gene in matches:
            self._var_list.addItem(gene)

    def _add_clock_genes(self):
        """Add known clock genes to selection (for Rosbash data)."""
        # Import here to avoid circular dependency
        from utils.rosbash_loader import get_available_clock_genes

        clock_genes = get_available_clock_genes()

        for i in range(self._var_list.count()):
            item = self._var_list.item(i)
            if item.text() in clock_genes:
                item.setSelected(True)
    
    def _run_analysis(self):
        """Start the analysis."""
        # Validate selections
        selected_vars = [
            self._var_list.item(i).text()
            for i in range(self._var_list.count())
            if self._var_list.item(i).isSelected()
        ]
        
        if not selected_vars:
            QMessageBox.warning(self, "No Variables", "Please select at least one variable.")
            return
        
        # Get conditions
        method_text = self._method_combo.currentText()
        is_comparison = "Compare" in method_text
        
        if is_comparison:
            cond1 = self._cond1_combo.currentText()
            cond2 = self._cond2_combo.currentText()
            if cond1 == cond2:
                QMessageBox.warning(self, "Same Conditions", "Please select different conditions to compare.")
                return
            selected_conds = [cond1]
            compare_conditions = (cond1, cond2)
        else:
            selected_conds = [
                self._cond_list.item(i).text()
                for i in range(self._cond_list.count())
                if self._cond_list.item(i).isSelected()
            ]
            compare_conditions = None
            
            if not selected_conds:
                QMessageBox.warning(self, "No Conditions", "Please select at least one condition.")
                return
        
        # Build configuration
        config = AnalysisConfig(
            method=self._get_current_method_enum(),
            variables=selected_vars,
            conditions=selected_conds,
            parameters=self._get_current_parameters(),
            compare_conditions=compare_conditions
        )
        
        # Start worker
        self._worker = AnalysisWorker(config, self._loader, self._source_type)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        
        self._run_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        
        self.analysis_started.emit()
        self._worker.start()
    
    def _cancel_analysis(self):
        """Cancel running analysis."""
        if hasattr(self, '_worker') and self._worker.isRunning():
            self._worker.terminate()
            self._on_finished(False, None, "Analysis cancelled")
    
    def _on_progress(self, percent: int, message: str):
        """Handle progress updates."""
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)
    
    def _on_finished(self, success: bool, results, message: str):
        """Handle analysis completion."""
        self._run_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress_bar.setVisible(False)
        
        if success:
            self._status_label.setText(f"✓ {message}")
            self._status_label.setStyleSheet("color: green;")
            self.analysis_completed.emit(results)
        else:
            self._status_label.setText(f"✗ {message}")
            self._status_label.setStyleSheet("color: red;")
            self.analysis_error.emit(message)
    
    def _get_current_method_enum(self) -> AnalysisMethod:
        """Get the current method as enum."""
        module = self._module_combo.currentIndex()
        method = self._method_combo.currentText()
        
        # Map to enum (simplified)
        mapping = {
            (0, "Single Cosinor"): AnalysisMethod.COSINORPY_SINGLE,
            (0, "Multi-Component"): AnalysisMethod.COSINORPY_MULTI,
            (0, "Population Mean"): AnalysisMethod.COSINORPY_POPULATION,
            (0, "Compare Conditions"): AnalysisMethod.COSINORPY_COMPARE,
            (0, "Count Data (Poisson)"): AnalysisMethod.COSINORPY_COUNT,
            (1, "Single Fit"): AnalysisMethod.CIRCACOMPARE_SINGLE,
            (1, "Compare Groups"): AnalysisMethod.CIRCACOMPARE_COMPARE,
            (2, "JTK Cycle"): AnalysisMethod.RHYTHM_JTK,
            (2, "AR-JTK"): AnalysisMethod.RHYTHM_AR_JTK,
            (2, "Cosinor (OLS)"): AnalysisMethod.RHYTHM_COSINOR,
            (2, "Harmonic Cosinor"): AnalysisMethod.RHYTHM_HARMONIC,
            (2, "Fourier F24"): AnalysisMethod.RHYTHM_F24,
            (2, "Lomb-Scargle"): AnalysisMethod.RHYTHM_LOMB,
            (2, "Wavelet (CWT)"): AnalysisMethod.RHYTHM_CWT,
            (2, "Linear Mixed Effects"): AnalysisMethod.RHYTHM_LME,
        }
        
        return mapping.get((module, method), AnalysisMethod.COSINORPY_SINGLE)
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return {
            'period': self._period_spin.value(),
            'n_components': self._components_spin.value(),
            'period_range': (self._period_min_spin.value(), self._period_max_spin.value()),
            'loss': self._loss_combo.currentText(),
            'f_scale': self._fscale_spin.value(),
            'max_iterations': self._max_iterations_spin.value(),
            'n_harmonics': self._harmonics_spin.value(),
            'n_permutations': self._permutations_spin.value()
        }
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def set_data(self, loader, source_type: str):
        """Set the data loader for analysis."""
        self._loader = loader
        self._source_type = source_type

        # Update variable list
        self._var_list.clear()

        if source_type == 'csv':
            # CSV: Show all available variables
            self._var_label.setText("Variables:")
            self._gene_search.setVisible(False)
            self._clock_genes_btn.setVisible(False)
            self._cluster_frame.setVisible(False)

            variables = loader.get_variable_columns()
            for var in variables:
                item = QListWidgetItem(var)
                self._var_list.addItem(item)

        else:  # rosbash
            # Rosbash: Show all genes with search functionality
            self._var_label.setText("Genes:")
            self._gene_search.setVisible(True)
            self._clock_genes_btn.setVisible(True)
            self._cluster_frame.setVisible(True)

            # Store all genes for filtering
            self._all_genes = loader.get_gene_names()
            # Show first 500 genes initially
            for gene in self._all_genes[:500]:
                self._var_list.addItem(gene)

            # Populate cluster list
            self._cluster_list.clear()
            clusters = loader.get_clusters()
            for cluster in clusters:
                item = QListWidgetItem(cluster)
                item.setSelected(True)  # Select all by default
                self._cluster_list.addItem(item)

        # Update condition list
        self._cond_list.clear()
        self._cond1_combo.clear()
        self._cond2_combo.clear()

        if source_type == 'csv':
            conditions = loader.get_conditions()
        else:  # rosbash
            info = loader.get_dataset_info()
            conditions = info.conditions

        for cond in conditions:
            item = QListWidgetItem(cond)
            item.setSelected(True)  # Select all by default
            self._cond_list.addItem(item)
            self._cond1_combo.addItem(cond)
            self._cond2_combo.addItem(cond)

        # Enable run button
        self._run_btn.setEnabled(True)
        self._available_variables = self._all_genes if source_type == 'rosbash' else variables
        self._available_conditions = conditions
    
    def clear_data(self):
        """Clear loaded data."""
        self._loader = None
        self._source_type = None
        self._var_list.clear()
        self._cond_list.clear()
        self._cond1_combo.clear()
        self._cond2_combo.clear()
        self._cluster_list.clear()
        self._gene_search.clear()
        self._gene_search.setVisible(False)
        self._clock_genes_btn.setVisible(False)
        self._cluster_frame.setVisible(False)
        self._var_label.setText("Variables:")
        self._run_btn.setEnabled(False)
        if hasattr(self, '_all_genes'):
            self._all_genes = []
