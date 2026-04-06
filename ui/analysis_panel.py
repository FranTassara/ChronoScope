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
    # CosinorPy methods - New Refactored
    COSINORPY_PERIODOGRAM = "CosinorPy: Periodogram Analysis"
    COSINORPY_INDEPENDENT = "CosinorPy: Cosinor (Independent Data)"
    COSINORPY_DEPENDENT = "CosinorPy: Cosinor (Dependent Data)"
    COSINORPY_COMPARE_INDEPENDENT = "CosinorPy: Compare Conditions (Independent)"
    COSINORPY_COMPARE_DEPENDENT = "CosinorPy: Compare Conditions (Dependent)"
    COSINORPY_NONLINEAR_INDEPENDENT = "CosinorPy: Nonlinear (Independent Data)"
    COSINORPY_NONLINEAR_DEPENDENT = "CosinorPy: Nonlinear (Dependent Data)"
    COSINORPY_NONLINEAR_COMPARE_INDEPENDENT = "CosinorPy: Nonlinear Compare (Independent)"
    COSINORPY_NONLINEAR_COMPARE_DEPENDENT = "CosinorPy: Nonlinear Compare (Dependent)"

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
    RHYTHM_SPECTRAL = "Spectral Analysis (Periodogram)"
    RHYTHM_CWT = "Wavelet (CWT)"
    RHYTHM_LME = "Linear Mixed Effects"

    # AI Meta-Classifier
    CONSENSUS_AI = "Consensus Rhythmicity Score (AI)"

    # Visualization methods (primarily for DAM data)
    VISUALIZATION_ACTIVITY_PROFILE = "Visualization: Activity Profile"


@dataclass
class AnalysisConfig:
    """Configuration for an analysis run."""
    method: AnalysisMethod
    variables: List[str]
    conditions: List[str]
    parameters: Dict[str, Any]
    compare_conditions: Optional[tuple] = None  # (condition1, condition2)
    clusters: Optional[List[str]] = None  # For Rosbash data: list of selected clusters


class AnalysisWorker(QThread):
    """Worker thread for running analyses."""
    finished = Signal(bool, object, str)  # success, results, message
    progress = Signal(int, str)  # percentage, status message
    period_detected = Signal(float, float)  # detected_period, p_value
    components_detected = Signal(int, float)  # detected_n_components, p_value

    def __init__(self, config: AnalysisConfig, loader, source_type: str):
        super().__init__()
        self.config = config
        self.loader = loader
        self.source_type = source_type
        self.results = []
        self.detected_period = None
        self.detected_n_components = None

    def run(self):
        try:
            # Handle visualization methods separately (they don't use the analysis engine)
            if self.config.method == AnalysisMethod.VISUALIZATION_ACTIVITY_PROFILE:
                self._run_activity_profile_visualization()
                return

            # Import analysis engine
            from core.analysis_engine import AnalysisEngine, AnalysisType

            engine = AnalysisEngine()

            # Get the full dataset
            if self.source_type == 'csv' or self.source_type == 'dam':
                data = self.loader.get_data()
                time_col = self.loader.get_time_column()
                condition_col = self.loader._condition_col if hasattr(self.loader, '_condition_col') else 'condition'
            else:  # rosbash
                # For Rosbash, we need to prepare data differently
                # This is a simplified placeholder - you may need to adjust
                data = None  # Will be handled in _run_single_analysis
                time_col = 'time'
                condition_col = 'condition'

            # TODO: Auto-period detection needs to be reimplemented for refactored architecture
            # The old periodogram method signature is incompatible with the new refactored version
            # Auto-detect optimal period if requested (only for CosinorPy methods)
            # if self.config.parameters.get('auto_period', False) and self.source_type == 'csv':
            #     self.progress.emit(10, "Auto-detecting optimal period...")
            #     # ... old code commented out ...

            # Map UI method to engine AnalysisType
            analysis_type = self._map_method_to_type(self.config.method)

            # Calculate total number of analyses
            # For Rosbash with clusters: variables * conditions * clusters
            # For CSV or Rosbash without clusters: variables * conditions
            if self.source_type == 'rosbash' and self.config.clusters:
                total = len(self.config.variables) * len(self.config.conditions) * len(self.config.clusters)
            else:
                total = len(self.config.variables) * len(self.config.conditions)
            current = 0

            # Check if this is a pairwise comparison analysis (2 conditions)
            if self.config.compare_conditions:
                # This is a pairwise comparison - use run_comparison
                cond1, cond2 = self.config.compare_conditions

                # Calculate total: For Rosbash with clusters, analyze each cluster separately
                if self.source_type == 'rosbash' and self.config.clusters:
                    total = len(self.config.variables) * len(self.config.clusters)
                else:
                    total = len(self.config.variables)
                current = 0

                for var in self.config.variables:
                    # For Rosbash data, iterate over selected clusters
                    clusters_to_analyze = self.config.clusters if (self.source_type == 'rosbash' and self.config.clusters) else [None]

                    for cluster in clusters_to_analyze:
                        # Update progress message
                        if cluster:
                            progress_msg = f"Comparing {var} (cluster: {cluster}): {cond1} vs {cond2}..."
                        else:
                            progress_msg = f"Comparing {var}: {cond1} vs {cond2}..."

                        self.progress.emit(
                            int(current / total * 100),
                            progress_msg
                        )

                        if self.source_type == 'csv' or self.source_type == 'dam':
                            # Get CSV/DAM file path from loader for saving plots
                            csv_path = getattr(self.loader, '_filepath', None)
                            result = engine.run_comparison(
                                data, var, cond1, cond2, analysis_type,
                                time_col=time_col,
                                condition_col=condition_col,
                                parameters=self.config.parameters,
                                data_file_path=csv_path
                            )
                        else:  # rosbash
                            # For Rosbash, compare conditions within specific cluster
                            result = self._run_rosbash_comparison(var, cond1, cond2, cluster, analysis_type)

                        if isinstance(result, list):
                            for r in result:
                                self.results.append(r.to_dict())
                        elif result:
                            self.results.append(result.to_dict())

                        current += 1

            # Special handling for CircaCompare Compare Groups (all pairs)
            elif analysis_type == AnalysisType.CIRCACOMPARE_COMPARE and not self.config.compare_conditions:
                from itertools import combinations
                all_conditions = self.config.conditions
                condition_pairs = list(combinations(all_conditions, 2))

                if self.source_type == 'rosbash' and self.config.clusters:
                    total = len(self.config.variables) * len(self.config.clusters) * len(condition_pairs)
                else:
                    total = len(self.config.variables) * len(condition_pairs)
                current = 0

                for var in self.config.variables:
                    clusters_to_analyze = self.config.clusters if (self.source_type == 'rosbash' and self.config.clusters) else [None]

                    for cluster in clusters_to_analyze:
                        for cond1, cond2 in condition_pairs:
                            if cluster:
                                progress_msg = f"Comparing {var} (cluster: {cluster}): {cond1} vs {cond2}..."
                            else:
                                progress_msg = f"Comparing {var}: {cond1} vs {cond2}..."

                            self.progress.emit(
                                int(current / total * 100),
                                progress_msg
                            )

                            if self.source_type == 'csv' or self.source_type == 'dam':
                                csv_path = getattr(self.loader, '_filepath', None)
                                result = engine.run_comparison(
                                    data, var, cond1, cond2, analysis_type,
                                    time_col=time_col,
                                    condition_col=condition_col,
                                    parameters=self.config.parameters,
                                    data_file_path=csv_path
                                )
                            else:  # rosbash
                                result = self._run_rosbash_comparison(var, cond1, cond2, cluster, analysis_type)

                            if isinstance(result, list):
                                for r in result:
                                    self.results.append(r.to_dict())
                            elif result:
                                self.results.append(result.to_dict())

                            current += 1

                self.progress.emit(100, "Complete")

            # Special handling for Compare Conditions (all pairs) - Independent and Dependent
            elif analysis_type in (AnalysisType.COSINORPY_COMPARE_INDEPENDENT,
                                  AnalysisType.COSINORPY_COMPARE_DEPENDENT,
                                  AnalysisType.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT,
                                  AnalysisType.COSINORPY_NONLINEAR_COMPARE_DEPENDENT):
                # Compare all conditions - run ONCE per variable (or per variable-cluster for Rosbash)
                # For Rosbash with clusters: variables * clusters
                if self.source_type == 'rosbash' and self.config.clusters:
                    total = len(self.config.variables) * len(self.config.clusters)
                else:
                    total = len(self.config.variables)
                current = 0

                for var in self.config.variables:
                    # For Rosbash data, iterate over selected clusters
                    clusters_to_analyze = self.config.clusters if (self.source_type == 'rosbash' and self.config.clusters) else [None]

                    for cluster in clusters_to_analyze:
                        # Update progress message
                        if cluster:
                            progress_msg = f"Comparing all conditions for {var} (cluster: {cluster})..."
                        else:
                            progress_msg = f"Comparing all conditions for {var}..."

                        self.progress.emit(
                            int(current / total * 100),
                            progress_msg
                        )

                        if self.source_type == 'csv' or self.source_type == 'dam':
                            # Get CSV/DAM file path from loader for saving plots
                            csv_path = getattr(self.loader, '_filepath', None)

                            # Use first condition as placeholder (engine will use all conditions)
                            result = engine.run_analysis(
                                data, var, self.config.conditions[0], analysis_type,
                                time_col=time_col,
                                condition_col=condition_col,
                                parameters=self.config.parameters,
                                data_file_path=csv_path
                            )
                        else:  # rosbash
                            # For Rosbash, prepare data with all conditions for specific cluster
                            result = self._run_rosbash_compare_all(var, self.config.conditions, cluster, analysis_type)

                        # Handle both single result and list of results
                        if result:
                            if isinstance(result, list):
                                # Multiple comparison results (multiple pairs)
                                for res in result:
                                    self.results.append(res.to_dict() if hasattr(res, 'to_dict') else res)
                            else:
                                # Single result
                                self.results.append(result.to_dict() if hasattr(result, 'to_dict') else result)

                        current += 1

                self.progress.emit(100, "Complete")

            # Special handling for Periodogram - run once for all selected variables/conditions
            elif analysis_type == AnalysisType.COSINORPY_PERIODOGRAM:
                self.progress.emit(10, "Generating periodograms...")

                if self.source_type == 'csv' or self.source_type == 'dam':
                    # Get CSV/DAM file path from loader for saving plots
                    csv_path = getattr(self.loader, '_filepath', None)

                    # Call periodogram with ALL selected variables and conditions
                    result = engine.run_analysis(
                        data,
                        variable=None,  # Signal to use all selected variables
                        condition=None,  # Signal to use all selected conditions
                        analysis_type=analysis_type,
                        time_col=time_col,
                        condition_col=condition_col,
                        parameters={
                            **self.config.parameters,
                            'selected_variables': self.config.variables,
                            'selected_conditions': self.config.conditions
                        },
                        data_file_path=csv_path
                    )

                    if result and result.success:
                        self.results.append(result.to_dict())
                    elif result:
                        self.results.append(result.to_dict())
                else:
                    # Rosbash not supported for periodogram
                    pass

                self.progress.emit(100, "Complete")

            else:
                # Regular single analysis
                for var in self.config.variables:
                    for cond in self.config.conditions:
                        # For Rosbash data, iterate over selected clusters
                        clusters_to_analyze = self.config.clusters if (self.source_type == 'rosbash' and self.config.clusters) else [None]

                        for cluster in clusters_to_analyze:
                            # Update progress message
                            if cluster:
                                progress_msg = f"Analyzing {var} in {cond} (cluster: {cluster})..."
                            else:
                                progress_msg = f"Analyzing {var} in {cond}..."

                            self.progress.emit(
                                int(current / total * 100),
                                progress_msg
                            )

                            if self.source_type == 'csv' or self.source_type == 'dam':
                                # Get CSV/DAM file path from loader for saving plots
                                csv_path = getattr(self.loader, '_filepath', None)
                                result = engine.run_analysis(
                                    data, var, cond, analysis_type,
                                    time_col=time_col,
                                    condition_col=condition_col,
                                    parameters=self.config.parameters,
                                    data_file_path=csv_path
                                )
                            else:  # rosbash
                                # For Rosbash, need special handling with cluster
                                result = self._run_rosbash_analysis(var, cond, cluster, analysis_type)

                            # Handle both single result and list of results (for multiple periods)
                            if result:
                                if isinstance(result, list):
                                    # Multiple results (e.g., multiple periods tested)
                                    for res in result:
                                        if res.success:
                                            self.results.append(res.to_dict())
                                else:
                                    # Single result
                                    if result.success:
                                        self.results.append(result.to_dict())

                                        # Check if auto-components was used and emit signal
                                        if self.config.parameters.get('auto_components', False) and result.n_components is not None:
                                            self.detected_n_components = result.n_components
                                            self.components_detected.emit(result.n_components, result.p_value or 0.0)
                                            print(f"[DEBUG Worker] Auto-components detected: {result.n_components} components, p-value: {result.p_value}")
                                    else:
                                        print(f"[ERROR Worker] Analysis failed for {var}/{cond}: {getattr(result, 'message', 'unknown error')}")

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
            # CosinorPy - New Refactored Methods
            AnalysisMethod.COSINORPY_PERIODOGRAM: AnalysisType.COSINORPY_PERIODOGRAM,
            AnalysisMethod.COSINORPY_INDEPENDENT: AnalysisType.COSINORPY_INDEPENDENT,
            AnalysisMethod.COSINORPY_DEPENDENT: AnalysisType.COSINORPY_DEPENDENT,
            AnalysisMethod.COSINORPY_COMPARE_INDEPENDENT: AnalysisType.COSINORPY_COMPARE_INDEPENDENT,
            AnalysisMethod.COSINORPY_COMPARE_DEPENDENT: AnalysisType.COSINORPY_COMPARE_DEPENDENT,
            AnalysisMethod.COSINORPY_NONLINEAR_INDEPENDENT: AnalysisType.COSINORPY_NONLINEAR_INDEPENDENT,
            AnalysisMethod.COSINORPY_NONLINEAR_DEPENDENT: AnalysisType.COSINORPY_NONLINEAR_DEPENDENT,
            AnalysisMethod.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT: AnalysisType.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT,
            AnalysisMethod.COSINORPY_NONLINEAR_COMPARE_DEPENDENT: AnalysisType.COSINORPY_NONLINEAR_COMPARE_DEPENDENT,
            # CircaCompare
            AnalysisMethod.CIRCACOMPARE_SINGLE: AnalysisType.CIRCACOMPARE_SINGLE,
            AnalysisMethod.CIRCACOMPARE_COMPARE: AnalysisType.CIRCACOMPARE_COMPARE,
            # Rhythm Analysis
            AnalysisMethod.RHYTHM_JTK: AnalysisType.JTK,
            AnalysisMethod.RHYTHM_AR_JTK: AnalysisType.AR_JTK,
            AnalysisMethod.RHYTHM_COSINE_KENDALL: AnalysisType.COSINE_KENDALL,
            AnalysisMethod.RHYTHM_COSINOR: AnalysisType.COSINOR_OLS,
            AnalysisMethod.RHYTHM_HARMONIC: AnalysisType.HARMONIC_COSINOR,
            AnalysisMethod.RHYTHM_F24: AnalysisType.FOURIER_F24,
            AnalysisMethod.RHYTHM_LOMB: AnalysisType.LOMB_SCARGLE,
            AnalysisMethod.RHYTHM_SPECTRAL: AnalysisType.SPECTRAL_ANALYSIS,
            AnalysisMethod.RHYTHM_CWT: AnalysisType.CWT,
            AnalysisMethod.RHYTHM_LME: AnalysisType.LME,
            # AI Consensus
            AnalysisMethod.CONSENSUS_AI: AnalysisType.CONSENSUS_AI,
        }
        return mapping.get(method, AnalysisType.COSINORPY_PERIODOGRAM)

    def _run_rosbash_analysis(self, gene: str, condition: str, cluster: Optional[str], analysis_type):
        """Run analysis on Rosbash data."""
        from core.analysis_engine import AnalysisEngine, AnalysisResult

        try:
            # Check if gene exists in dataset
            if not self.loader.gene_exists(gene):
                return AnalysisResult(
                    variable=gene,
                    condition=condition,
                    method=analysis_type.value,
                    success=False,
                    message=f"Gene '{gene}' not found in dataset"
                )

            # Get gene expression data from Rosbash loader for specific cluster
            try:
                df = self.loader.get_gene_expression_df(
                    gene=gene,
                    condition=condition,
                    cluster=cluster,  # Use specified cluster
                    use_log1p=False  # Use TP10K normalized data (as in Ma et al. 2021)
                )
            except Exception as e:
                return AnalysisResult(
                    variable=gene,
                    condition=condition,
                    method=analysis_type.value,
                    success=False,
                    message=f"Failed to extract gene expression: {str(e)}"
                )

            # Run analysis using the engine
            engine = AnalysisEngine()

            # Include cluster in variable name if specified
            variable_name = f"{gene} [{cluster}]" if cluster else gene

            result = engine.run_analysis(
                data=df,
                variable=gene,  # Gene name (column in df) - must match column name
                condition=condition,
                analysis_type=analysis_type,
                time_col='time',
                condition_col='condition',
                parameters=self.config.parameters,
                data_file_path=None  # Rosbash doesn't have a file path
            )

            # Update result variable name to include cluster
            if result and cluster:
                result.variable = variable_name

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=gene,
                condition=condition,
                method=analysis_type.value,
                success=False,
                message=f"Analysis error: {str(e)}"
            )

    def _run_rosbash_comparison(self, gene: str, cond1: str, cond2: str, cluster: Optional[str], analysis_type):
        """Run pairwise comparison analysis on Rosbash data."""
        from core.analysis_engine import AnalysisEngine, ComparisonResult

        try:
            # Check if gene exists
            if not self.loader.gene_exists(gene):
                return ComparisonResult(
                    variable=gene,
                    condition1=cond1,
                    condition2=cond2,
                    method=analysis_type.value,
                    success=False,
                    message=f"Gene '{gene}' not found in dataset"
                )

            # Get data for both conditions combined for specific cluster
            try:
                df = self.loader.prepare_for_circacompare(
                    gene=gene,
                    condition1=cond1,
                    condition2=cond2,
                    cluster=cluster  # Use specified cluster
                )
            except Exception as e:
                return ComparisonResult(
                    variable=gene,
                    condition1=cond1,
                    condition2=cond2,
                    method=analysis_type.value,
                    success=False,
                    message=f"Failed to extract gene expression: {str(e)}"
                )

            # Run comparison using the engine
            engine = AnalysisEngine()

            # Include cluster in variable name if specified
            variable_name = f"{gene} [{cluster}]" if cluster else gene

            result = engine.run_comparison(
                data=df,
                variable=gene,  # Gene name (column in df) - must match column name
                condition1=cond1,
                condition2=cond2,
                analysis_type=analysis_type,
                time_col='time',
                condition_col='condition',
                parameters=self.config.parameters,
                data_file_path=None
            )

            # Update result variable name to include cluster
            if result and cluster:
                result.variable = variable_name

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return ComparisonResult(
                variable=gene,
                condition1=cond1,
                condition2=cond2,
                method=analysis_type.value,
                success=False,
                message=f"Comparison error: {str(e)}"
            )

    def _run_rosbash_compare_all(self, gene: str, conditions: List[str], cluster: Optional[str], analysis_type):
        """Run compare-all analysis on Rosbash data."""
        from core.analysis_engine import AnalysisEngine, AnalysisResult

        try:
            # Check if gene exists
            if not self.loader.gene_exists(gene):
                return AnalysisResult(
                    variable=gene,
                    condition='all',
                    method=analysis_type.value,
                    success=False,
                    message=f"Gene '{gene}' not found in dataset"
                )

            # Get data for all conditions combined for specific cluster
            try:
                # Combine data from all conditions
                dfs = []
                for cond in conditions:
                    df_cond = self.loader.get_gene_expression_df(
                        gene=gene,
                        condition=cond,
                        cluster=cluster,  # Use specified cluster
                        use_log1p=False  # Use TP10K normalized data (as in Ma et al. 2021)
                    )
                    dfs.append(df_cond)

                df = pd.concat(dfs, ignore_index=True)
            except Exception as e:
                return AnalysisResult(
                    variable=gene,
                    condition='all',
                    method=analysis_type.value,
                    success=False,
                    message=f"Failed to extract gene expression: {str(e)}"
                )

            # Run analysis using the engine (it will handle multiple conditions)
            engine = AnalysisEngine()

            # Include cluster in variable name if specified
            variable_name = f"{gene} [{cluster}]" if cluster else gene

            result = engine.run_analysis(
                data=df,
                variable=gene,  # Gene name (column in df) - must match column name
                condition=conditions[0],  # Placeholder, engine will use all
                analysis_type=analysis_type,
                time_col='time',
                condition_col='condition',
                parameters=self.config.parameters,
                data_file_path=None
            )

            # Update result variable name(s) to include cluster
            if cluster:
                if result:
                    if isinstance(result, list):
                        for res in result:
                            if hasattr(res, 'variable'):
                                res.variable = variable_name
                    elif hasattr(result, 'variable'):
                        result.variable = variable_name

            return result

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AnalysisResult(
                variable=gene,
                condition='all',
                method=analysis_type.value,
                success=False,
                message=f"Compare-all error: {str(e)}"
            )

    def _run_activity_profile_visualization(self):
        """Generate Activity Profile visualization for DAM data.

        Creates a heatmap showing activity for each day (Y-axis) across ZT times (X-axis).
        Each cell is the mean activity across all subjects at that day and ZT time.
        """
        try:
            self.progress.emit(10, "Preparing activity profile data...")

            # Get the data
            data = self.loader.get_data()
            if data is None or data.empty:
                self.finished.emit(False, None, "No data available for visualization")
                return

            self.progress.emit(30, "Calculating ZT times and days...")

            # Convert time to ZT (mod 24) and calculate day number
            df = data.copy()
            df['zt'] = (df['time'] % 24).round(2)
            df['day'] = (df['time'] // 24).astype(int) + 1  # Day 1, 2, 3, ...

            # Calculate number of days in the data
            n_days = df['day'].max()

            # Get unique conditions
            conditions = df['condition'].unique().tolist()

            # Get unique subjects per condition for sample size info
            n_subjects = {}
            if 'subject' in df.columns:
                for cond in conditions:
                    n_subjects[cond] = df[df['condition'] == cond]['subject'].nunique()

            self.progress.emit(50, "Computing mean activity by day and ZT...")

            # Calculate mean for each condition, day, and ZT time
            # This averages across all subjects FOR EACH DAY separately
            profile_data = {}
            for cond in conditions:
                cond_df = df[df['condition'] == cond]
                # Group by day and ZT time, calculate mean across subjects
                grouped = cond_df.groupby(['day', 'zt'])['activity'].mean().reset_index()
                grouped.columns = ['day', 'zt', 'mean']

                # Pivot to create matrix: rows=days, columns=ZT times
                pivot = grouped.pivot(index='day', columns='zt', values='mean')

                # Fill any NaN values with 0
                pivot = pivot.fillna(0)

                profile_data[cond] = {
                    'days': pivot.index.tolist(),
                    'zt_times': pivot.columns.tolist(),
                    'activity_matrix': pivot.values.tolist()  # 2D array: [day][zt]
                }

            self.progress.emit(60, "Computing actogram data...")

            # ===== ACTOGRAM DATA (double-plotted) =====
            # For actogram, we need 48h on X-axis (current day + next day)
            actogram_data = {}
            for cond in conditions:
                cond_df = df[df['condition'] == cond]
                # Group by day and ZT, average across subjects
                grouped = cond_df.groupby(['day', 'zt'])['activity'].mean().reset_index()
                # Column is 'activity' after reset_index()

                # Create double-plotted data: each row has 48h (day N: 0-24h, day N+1: 0-24h)
                days = sorted(grouped['day'].unique())
                zt_times = sorted(grouped['zt'].unique())
                double_plot_matrix = []

                for i, day in enumerate(days[:-1]):  # Exclude last day (no next day)
                    day_data = grouped[grouped['day'] == day].set_index('zt')['activity']
                    next_day_data = grouped[grouped['day'] == days[i + 1]].set_index('zt')['activity']

                    # Combine: first 24h from current day, next 24h from next day
                    row = []
                    for zt in zt_times:
                        row.append(day_data.get(zt, 0))
                    for zt in zt_times:
                        row.append(next_day_data.get(zt, 0))
                    double_plot_matrix.append(row)

                actogram_data[cond] = {
                    'days': days[:-1],  # Days for which we have double-plot
                    'zt_times': zt_times,
                    'matrix': double_plot_matrix  # 2D: [day][48h bins]
                }

            self.progress.emit(70, "Computing total daily activity...")

            # ===== TOTAL ACTIVITY PER DAY =====
            total_activity_data = {}
            for cond in conditions:
                cond_df = df[df['condition'] == cond]
                # Sum activity per day (across all subjects and timepoints)
                daily_totals = cond_df.groupby('day')['activity'].sum().reset_index()
                # Also get per-subject totals for SEM
                if 'subject' in cond_df.columns:
                    subject_daily = cond_df.groupby(['day', 'subject'])['activity'].sum().reset_index()
                    daily_stats = subject_daily.groupby('day')['activity'].agg(['mean', 'sem']).reset_index()
                    daily_stats['sem'] = daily_stats['sem'].fillna(0)
                else:
                    daily_stats = daily_totals.copy()
                    daily_stats['mean'] = daily_stats['activity']
                    daily_stats['sem'] = 0

                total_activity_data[cond] = {
                    'days': daily_stats['day'].tolist(),
                    'mean': daily_stats['mean'].tolist(),
                    'sem': daily_stats['sem'].tolist() if 'sem' in daily_stats.columns else [0] * len(daily_stats)
                }

            self.progress.emit(80, "Computing activity onset/offset times...")

            # ===== ACTIVITY ONSET AND OFFSET PER DAY =====
            # Onset: first time bin where activity exceeds threshold (mean + 0.5*std of that day)
            # Offset: last time bin where activity exceeds threshold
            onset_data = {}
            for cond in conditions:
                cond_df = df[df['condition'] == cond]
                days_list = []
                onset_times = []
                offset_times = []

                for day in sorted(cond_df['day'].unique()):
                    day_df = cond_df[cond_df['day'] == day]
                    # Average across subjects for each ZT
                    day_profile = day_df.groupby('zt')['activity'].mean().sort_index()

                    if len(day_profile) > 0:
                        threshold = day_profile.mean() + 0.5 * day_profile.std()
                        # Find bins where activity > threshold
                        above_threshold = day_profile[day_profile > threshold]
                        if len(above_threshold) > 0:
                            onset_zt = above_threshold.index[0]   # First
                            offset_zt = above_threshold.index[-1]  # Last
                        else:
                            onset_zt = None
                            offset_zt = None
                    else:
                        onset_zt = None
                        offset_zt = None

                    days_list.append(day)
                    onset_times.append(onset_zt)
                    offset_times.append(offset_zt)

                onset_data[cond] = {
                    'days': days_list,
                    'onset_times': onset_times,
                    'offset_times': offset_times
                }

            self.progress.emit(90, "Preparing visualization result...")

            # Build info message
            subjects_info = ", ".join([f"{cond}: n={n_subjects.get(cond, '?')}" for cond in conditions])
            message = f"Activity profile: {n_days} days, {len(conditions)} condition(s) ({subjects_info})"

            # Create result dictionary with data for plotting
            result = {
                'type': 'activity_profile',
                'success': True,
                'variable': 'activity',
                'conditions': conditions,
                'profile_data': profile_data,
                'actogram_data': actogram_data,
                'total_activity_data': total_activity_data,
                'onset_data': onset_data,
                'n_days': n_days,
                'n_subjects': n_subjects,
                'method': 'Activity Profile',
                'message': message
            }

            self.results.append(result)

            self.progress.emit(100, "Complete")
            self.finished.emit(True, self.results, f"Activity profile: {n_days} days, {len(conditions)} conditions")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished.emit(False, None, f"Visualization error: {str(e)}")


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

        # Data type info panel - REMOVED to save space
        # self._data_info_frame = self._create_data_info_panel()
        # layout.addWidget(self._data_info_frame)

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

        # Module and Method selection side by side
        selection_layout = QHBoxLayout()

        # Module selection (left side)
        module_layout = QVBoxLayout()
        module_layout.addWidget(QLabel("Module:"))

        self._module_combo = QComboBox()
        self._module_combo.addItems([
            "CosinorPy",
            "CircaCompare",
            "Rhythm Analysis",
            "AI Consensus"
        ])
        self._module_combo.currentIndexChanged.connect(self._on_module_changed)
        module_layout.addWidget(self._module_combo)

        selection_layout.addLayout(module_layout)

        # Method selection (right side)
        method_layout = QVBoxLayout()
        method_layout.addWidget(QLabel("Method:"))

        self._method_combo = QComboBox()
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_layout.addWidget(self._method_combo)

        selection_layout.addLayout(method_layout)

        layout.addLayout(selection_layout)

        # Method description (below both dropdowns)
        self._method_desc = QLabel("")
        self._method_desc.setWordWrap(True)
        self._method_desc.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self._method_desc)

        # Initialize methods
        self._on_module_changed(0)

        return group

    def _create_data_info_panel(self) -> QFrame:
        """Create informational panel about data type and method recommendations."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        frame.setStyleSheet("""
            QFrame {
                background-color: #E8F5E9;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Title
        title = QLabel("📊 Data Type Information")
        title.setStyleSheet("font-weight: bold; color: #2E7D32;")
        layout.addWidget(title)

        # Data type detection label
        self._data_type_label = QLabel("No data loaded")
        self._data_type_label.setWordWrap(True)
        self._data_type_label.setStyleSheet("color: #1B5E20;")
        layout.addWidget(self._data_type_label)

        # Method recommendation label
        self._method_recommendation_label = QLabel("")
        self._method_recommendation_label.setWordWrap(True)
        self._method_recommendation_label.setStyleSheet("color: #1B5E20; font-style: italic;")
        layout.addWidget(self._method_recommendation_label)

        frame.setVisible(False)  # Hidden by default until data is loaded
        return frame

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
        # self._var_list.setMaximumHeight(120)  # Commented to allow dynamic resizing
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

        main_h_layout.addLayout(var_layout, 1)  # stretch factor 1

        # Conditions
        cond_layout = QVBoxLayout()
        cond_layout.addWidget(QLabel("Conditions:"))
        self._cond_list = QListWidget()
        self._cond_list.setSelectionMode(QAbstractItemView.MultiSelection)
        # self._cond_list.setMaximumHeight(120)  # Commented to allow dynamic resizing
        cond_layout.addWidget(self._cond_list)

        cond_btn_layout = QHBoxLayout()
        select_all_c = QPushButton("All")
        select_all_c.clicked.connect(lambda: self._select_all_items(self._cond_list))
        clear_sel_c = QPushButton("None")
        clear_sel_c.clicked.connect(lambda: self._clear_selection(self._cond_list))
        cond_btn_layout.addWidget(select_all_c)
        cond_btn_layout.addWidget(clear_sel_c)
        cond_layout.addLayout(cond_btn_layout)

        main_h_layout.addLayout(cond_layout, 1)  # stretch factor 1

        # Rosbash-specific: Cluster selection (hidden for CSV)
        # Now added to the horizontal layout instead of below
        cluster_layout = QVBoxLayout()
        cluster_layout.addWidget(QLabel("Clusters:"))
        self._cluster_list = QListWidget()
        self._cluster_list.setSelectionMode(QAbstractItemView.MultiSelection)
        # self._cluster_list.setMaximumHeight(120)  # Commented to allow dynamic resizing
        cluster_layout.addWidget(self._cluster_list)

        cluster_btn_layout = QHBoxLayout()
        select_all_cl = QPushButton("All")
        select_all_cl.clicked.connect(lambda: self._select_all_items(self._cluster_list))
        clear_sel_cl = QPushButton("None")
        clear_sel_cl.clicked.connect(lambda: self._clear_selection(self._cluster_list))
        cluster_btn_layout.addWidget(select_all_cl)
        cluster_btn_layout.addWidget(clear_sel_cl)
        cluster_layout.addLayout(cluster_btn_layout)

        # Create a container widget for the cluster layout so we can hide/show it
        self._cluster_container = QWidget()
        self._cluster_container.setLayout(cluster_layout)
        self._cluster_container.setVisible(False)

        main_h_layout.addWidget(self._cluster_container, 1)  # stretch factor 1

        layout.addLayout(main_h_layout)
        
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

        # Rosbash comparison type selection (conditions vs clusters)
        self._rosbash_compare_type_frame = QFrame()
        rosbash_compare_layout = QVBoxLayout(self._rosbash_compare_type_frame)
        rosbash_compare_layout.setContentsMargins(0, 5, 0, 5)

        rosbash_label = QLabel("Comparison Type:")
        rosbash_label.setStyleSheet("font-weight: bold;")
        rosbash_compare_layout.addWidget(rosbash_label)

        self._rosbash_compare_type_combo = QComboBox()
        self._rosbash_compare_type_combo.addItems([
            "Compare Conditions (LD vs DD)"
            # "Compare Clusters" - Coming in future version
        ])
        self._rosbash_compare_type_combo.setToolTip(
            "Compare Conditions: Compares LD vs DD within each selected cluster\n\n"
            "Note: Cluster comparison (e.g., LNd vs DN1a) will be available in a future update"
        )
        rosbash_compare_layout.addWidget(self._rosbash_compare_type_combo)

        self._rosbash_compare_type_frame.setVisible(False)
        layout.addWidget(self._rosbash_compare_type_frame)

        return group
    
    def _create_parameters_section(self) -> QGroupBox:
        """Create parameters configuration section."""
        group = QGroupBox("Analysis Parameters")
        layout = QVBoxLayout(group)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # scroll.setMaximumHeight(250)  # Commented to allow dynamic resizing
        
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

        # Number of components (comma-separated list)
        self._components_edit = QLineEdit()
        self._components_edit.setText("1")
        self._components_edit.setPlaceholderText("e.g., 1 or 1,2,3")
        self._components_edit.setToolTip(
            "Components to test (comma-separated).\n\n"
            "• Single component: Enter '1'\n"
            "• Multiple components: Enter '1,2,3' to test models with 1, 2, and 3 components\n"
            "• CosinorPy will select the best model based on the chosen criterium"
        )
        # Connect signal to update comparison parameters when components change
        self._components_edit.textChanged.connect(self._on_components_changed)

        components_label = QLabel("Components:")
        components_label.setToolTip(
            "Components to test (comma-separated).\n\n"
            "• Single component: Enter '1'\n"
            "• Multiple components: Enter '1,2,3' to test models with 1, 2, and 3 components\n"
            "• CosinorPy will select the best model based on the chosen criterium"
        )
        self._params_layout.addRow(components_label, self._components_edit)

        # Period range
        # Note: If min == max, this is a single period. If min != max, it's a range.
        period_range_widget = QWidget()
        pr_layout = QHBoxLayout(period_range_widget)
        pr_layout.setContentsMargins(0, 0, 0, 0)

        self._period_min_spin = QDoubleSpinBox()
        self._period_min_spin.setRange(1, 72)
        self._period_min_spin.setValue(24.0)
        self._period_min_spin.setSuffix(" h")
        self._period_min_spin.valueChanged.connect(self._on_period_changed)
        pr_layout.addWidget(self._period_min_spin)

        pr_layout.addWidget(QLabel("to"))

        self._period_max_spin = QDoubleSpinBox()
        self._period_max_spin.setRange(1, 72)
        self._period_max_spin.setValue(24.0)
        self._period_max_spin.setSuffix(" h")
        self._period_max_spin.valueChanged.connect(self._on_period_changed)
        pr_layout.addWidget(self._period_max_spin)

        pr_layout.addWidget(QLabel("Step:"))

        self._period_step_spin = QDoubleSpinBox()
        self._period_step_spin.setRange(0.1, 10.0)
        self._period_step_spin.setValue(1.0)
        self._period_step_spin.setSingleStep(0.1)
        self._period_step_spin.setDecimals(1)
        self._period_step_spin.setSuffix(" h")
        self._period_step_spin.setToolTip(
            "Step size for period range.\n\n"
            "• Step = 1.0: Test every hour (e.g., 20, 21, 22, ...)\n"
            "• Step = 2.0: Test every 2 hours (e.g., 20, 22, 24, ...)\n"
            "• Step = 0.5: Test every half hour (e.g., 20.0, 20.5, 21.0, ...)\n"
            "• Larger steps = faster analysis, smaller steps = higher resolution"
        )
        self._period_step_spin.valueChanged.connect(self._on_period_changed)
        pr_layout.addWidget(self._period_step_spin)

        period_range_label = QLabel("Period:")
        period_range_label.setToolTip(
            "Period to test (in hours).\n\n"
            "• Single period: Set both values equal (e.g., 24 to 24)\n"
            "• Multiple periods: Set different values (e.g., 20 to 28)\n"
            "  → CosinorPy will test all periods in range and select best fit"
        )
        self._params_layout.addRow(period_range_label, period_range_widget)

        # Period per condition (for Independent Models with exactly 2 conditions)
        # Create widgets for period1 and period2
        period_per_cond_widget = QWidget()
        ppc_layout = QVBoxLayout(period_per_cond_widget)
        ppc_layout.setContentsMargins(0, 0, 0, 0)
        ppc_layout.setSpacing(4)

        # Period for condition 1
        period1_row = QWidget()
        p1_layout = QHBoxLayout(period1_row)
        p1_layout.setContentsMargins(0, 0, 0, 0)
        self._period_cond1_label = QLabel("Period for Condition 1:")
        self._period_cond1_spin = QDoubleSpinBox()
        self._period_cond1_spin.setRange(1, 72)
        self._period_cond1_spin.setValue(24.0)
        self._period_cond1_spin.setSuffix(" h")
        p1_layout.addWidget(self._period_cond1_label)
        p1_layout.addWidget(self._period_cond1_spin)
        ppc_layout.addWidget(period1_row)

        # Period for condition 2
        period2_row = QWidget()
        p2_layout = QHBoxLayout(period2_row)
        p2_layout.setContentsMargins(0, 0, 0, 0)
        self._period_cond2_label = QLabel("Period for Condition 2:")
        self._period_cond2_spin = QDoubleSpinBox()
        self._period_cond2_spin.setRange(1, 72)
        self._period_cond2_spin.setValue(24.0)
        self._period_cond2_spin.setSuffix(" h")
        p2_layout.addWidget(self._period_cond2_label)
        p2_layout.addWidget(self._period_cond2_spin)
        ppc_layout.addWidget(period2_row)

        self._period_per_cond_label = QLabel("Periods:")
        self._params_layout.addRow(self._period_per_cond_label, period_per_cond_widget)

        # Info message for >2 conditions
        self._period_multi_cond_info = QLabel(
            "ℹ️ Using default period (24h) for all conditions.\n"
            "Period selection is only available when comparing exactly 2 conditions."
        )
        self._period_multi_cond_info.setWordWrap(True)
        self._period_multi_cond_info.setStyleSheet(
            "color: #555; font-style: italic; font-size: 9px; "
            "background-color: #f0f0f0; padding: 6px; border-radius: 3px;"
        )
        self._params_layout.addRow("", self._period_multi_cond_info)

        # Loss function (for CircaCompare)
        self._loss_combo = QComboBox()
        self._loss_combo.addItems(['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'])
        self._loss_combo.setToolTip(
            "<b>Loss Function</b><br><br>"
            "Determines how residuals (differences between observed and predicted values) "
            "are penalized during optimization. Different loss functions provide varying "
            "degrees of robustness to outliers:<br><br>"
            "<b>• linear:</b> Standard least squares. No outlier protection. "
            "Best when data has no outliers.<br><br>"
            "<b>• soft_l1:</b> Smooth approximation to L1 (absolute value) loss. "
            "Moderate robustness to outliers.<br><br>"
            "<b>• huber:</b> (Recommended) Combines squared loss for small residuals "
            "and linear loss for large residuals. Good balance between efficiency "
            "and robustness.<br><br>"
            "<b>• cauchy:</b> Heavy-tailed loss function. Strong outlier resistance "
            "but may underweight valid extreme values.<br><br>"
            "<b>• arctan:</b> Very strong outlier resistance. Bounded influence "
            "for extreme values. Use when data contains severe outliers."
        )
        self._params_layout.addRow("Loss Function:", self._loss_combo)

        # F-scale
        self._fscale_spin = QDoubleSpinBox()
        self._fscale_spin.setRange(0.1, 10.0)
        self._fscale_spin.setValue(1.0)
        self._fscale_spin.setSingleStep(0.1)
        self._fscale_spin.setToolTip(
            "<b>F-Scale (Soft Margin)</b><br><br>"
            "Controls the transition point between inlier and outlier treatment "
            "for robust loss functions (soft_l1, huber, cauchy, arctan).<br><br>"
            "• <b>Lower values (0.1-0.5):</b> More aggressive outlier rejection. "
            "Residuals are treated as outliers at smaller magnitudes.<br><br>"
            "• <b>Default value (1.0):</b> Standard threshold. Recommended for "
            "most datasets.<br><br>"
            "• <b>Higher values (2.0-10.0):</b> More tolerant of large residuals. "
            "Useful when data has high natural variability.<br><br>"
            "<i>Note: Has no effect when using 'linear' loss function.</i>"
        )
        self._params_layout.addRow("F-Scale:", self._fscale_spin)

        # Max iterations (for CircaCompare)
        self._max_iterations_spin = QSpinBox()
        self._max_iterations_spin.setRange(100, 2000)
        self._max_iterations_spin.setValue(500)
        self._max_iterations_spin.setSingleStep(100)
        self._max_iterations_spin.setToolTip(
            "<b>Maximum Iterations</b><br><br>"
            "Sets the maximum number of optimization iterations allowed for "
            "the curve fitting algorithm to converge to a solution.<br><br>"
            "• <b>Lower values (100-300):</b> Faster computation but may not "
            "converge for complex or noisy data.<br><br>"
            "• <b>Default value (500):</b> Sufficient for most datasets. "
            "Good balance between speed and accuracy.<br><br>"
            "• <b>Higher values (1000-2000):</b> Use when fitting fails to "
            "converge or for very noisy data. Increases computation time.<br><br>"
            "<i>If the fit doesn't converge, try increasing this value or "
            "adjusting the loss function and f-scale parameters.</i>"
        )
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

        # Amplification (for Nonlinear Cosinor)
        self._amplification_spin = QDoubleSpinBox()
        self._amplification_spin.setRange(-0.2, 0.2)
        self._amplification_spin.setValue(0.0)
        self._amplification_spin.setSingleStep(0.01)
        self._amplification_spin.setDecimals(3)
        self._amplification_spin.setSpecialValueText("Auto")
        self._amplification_spin.setToolTip(
            "Amplification coefficient for non-linear cosinor:\n"
            "- Negative values: damped oscillation (amplitude decreases over time)\n"
            "- Positive values: forced oscillation (amplitude increases over time)\n"
            "- Zero (Auto): automatically estimated from data"
        )
        self._params_layout.addRow("Amplification:", self._amplification_spin)

        # Linear component (for Nonlinear Cosinor)
        self._lin_comp_spin = QDoubleSpinBox()
        self._lin_comp_spin.setRange(-2.0, 2.0)
        self._lin_comp_spin.setValue(0.0)
        self._lin_comp_spin.setSingleStep(0.1)
        self._lin_comp_spin.setDecimals(3)
        self._lin_comp_spin.setSpecialValueText("Auto")
        self._lin_comp_spin.setToolTip(
            "Linear trend component for non-linear cosinor:\n"
            "- Positive: increasing baseline trend\n"
            "- Negative: decreasing baseline trend\n"
            "- Zero (Auto): automatically estimated from data"
        )
        self._params_layout.addRow("Linear Component:", self._lin_comp_spin)

        # Use dependent model (for Nonlinear Cosinor comparison)
        self._use_dependent_model_check = QCheckBox("Use dependent (shared period) model")
        self._use_dependent_model_check.setChecked(False)
        self._use_dependent_model_check.setToolTip(
            "For independent data comparison:\n"
            "- Unchecked: Independent model (each group has its own period)\n"
            "- Checked: Dependent model (both groups share the same period)"
        )
        self._params_layout.addRow("", self._use_dependent_model_check)
        self._use_dependent_model_check.toggled.connect(lambda _: self._update_parameter_visibility())

        # Periodogram type (for Periodogram)
        self._per_type_combo = QComboBox()
        self._per_type_combo.addItems(['per', 'welch', 'lombscargle'])
        self._per_type_combo.setToolTip(
            "Type of periodogram:\n"
            "- per: Standard FFT periodogram\n"
            "- welch: Welch's method (averaged periodogram)\n"
            "- lombscargle: Lomb-Scargle (for unevenly sampled data)"
        )
        self._params_layout.addRow("Periodogram Type:", self._per_type_combo)

        # Max period (for Periodogram)
        self._max_per_spin = QDoubleSpinBox()
        self._max_per_spin.setRange(12.0, 1000.0)
        self._max_per_spin.setValue(240.0)
        self._max_per_spin.setSingleStep(10.0)
        self._max_per_spin.setDecimals(1)
        self._max_per_spin.setSuffix(" h")
        self._max_per_spin.setToolTip(
            "Maximum period to consider in periodogram analysis (in hours)"
        )
        self._params_layout.addRow("Max Period:", self._max_per_spin)

        # Prominent peaks checkbox (for CosinorPy Periodogram)
        self._prominent_check = QCheckBox("Find prominent peaks")
        self._prominent_check.setChecked(False)
        self._prominent_check.setToolTip(
            "Identify and label prominent peaks above significance threshold"
        )
        self._params_layout.addRow("", self._prominent_check)

        # Model Type (for CosinorPy - count data analysis)
        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(['Normal', 'Poisson', 'Negative Binomial'])
        self._model_type_combo.setToolTip(
            "Model type for count data:\n"
            "- Normal: For continuous data (default)\n"
            "- Poisson: For count data (RNA-seq)\n"
            "- Negative Binomial: For overdispersed count data"
        )
        self._params_layout.addRow("Model Type:", self._model_type_combo)

        # Criterium (for CosinorPy - best period selection)
        self._criterium_label = QLabel("Criterium (Best Period):")
        self._criterium_combo = QComboBox()
        self._criterium_combo.addItems(['RSS', 'AIC', 'BIC', 'Log-Likelihood'])
        self._criterium_combo.setToolTip(
            "Criterion for selecting best period:\n"
            "- RSS: Residual Sum of Squares (default)\n"
            "- AIC: Akaike Information Criterion\n"
            "- BIC: Bayesian Information Criterion\n"
            "- Log-Likelihood: Maximum likelihood"
        )
        self._params_layout.addRow(self._criterium_label, self._criterium_combo)

        # Comparison Type (for single component comparison methods)
        self._comparison_type_combo = QComboBox()
        self._comparison_type_combo.addItems(['Pooled Model', 'Independent Models'])
        self._comparison_type_combo.currentTextChanged.connect(self._update_compare_conditions_parameters)
        self._comparison_type_combo.setToolTip(
            "Statistical approach for comparing conditions:\n\n"
            "• Pooled Model: Single model with group interaction\n"
            "  - Same period for both groups\n"
            "  - Assumes common variance\n"
            "  - Higher statistical power\n"
            "  - Recommended for similar groups\n\n"
            "• Independent Models: Separate models per condition\n"
            "  - Different periods allowed\n"
            "  - No variance assumptions\n"
            "  - More robust for heterogeneous groups"
        )
        self._params_layout.addRow("Comparison Type:", self._comparison_type_combo)

        # Comparison Method (for multi-component comparison)
        self._comparison_method_combo = QComboBox()
        self._comparison_method_combo.addItems(['Independent', 'LimoRhyde'])
        self._comparison_method_combo.currentTextChanged.connect(self._update_limo_analysis_options)
        self._comparison_method_combo.currentTextChanged.connect(self._update_compare_conditions_parameters)
        self._comparison_method_combo.setToolTip(
            "Method for multi-component comparison:\n\n"
            "• Independent: Compare separate models\n"
            "  - Analysis: CI or Bootstrap\n"
            "  - Flexible, robust\n\n"
            "• LimoRhyde: LimoRhyde-style comparison\n"
            "  - Analysis: None, CI1, Bootstrap1, CI2, Bootstrap2\n"
            "  - Additional F-tests and p-values"
        )
        self._params_layout.addRow("Comparison Method:", self._comparison_method_combo)

        # Analysis Method (for CosinorPy - CI calculation method)
        self._analysis_method_label = QLabel("Analysis Method:")
        self._analysis_method_combo = QComboBox()
        # Default options for independent data
        self._analysis_method_combo.addItems(['CI', 'Bootstrap', 'Sampling'])
        self._analysis_method_combo.setToolTip(
            "Method for calculating confidence intervals:\n"
            "- CI: Analytical confidence intervals (fast, for independent data)\n"
            "- Bootstrap: Bootstrap resampling (more robust)\n"
            "- Sampling: Parameter sampling (for dependent data)"
        )
        self._analysis_method_combo.currentTextChanged.connect(self._update_cosinor_independent_params_visibility)
        self._params_layout.addRow(self._analysis_method_label, self._analysis_method_combo)

        # Bootstrap size (for Bootstrap analysis method)
        self._bootstrap_size_label = QLabel("Bootstrap Size:")
        self._bootstrap_size_spin = QSpinBox()
        self._bootstrap_size_spin.setRange(50, 10000)
        self._bootstrap_size_spin.setValue(1000)
        self._bootstrap_size_spin.setSingleStep(50)
        self._bootstrap_size_spin.setToolTip(
            "Number of bootstrap iterations (higher = more accurate but slower)"
        )
        self._params_layout.addRow(self._bootstrap_size_label, self._bootstrap_size_spin)

        # Parameters to Compare (multi-select via checkboxes)
        params_widget = QWidget()
        params_layout = QHBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)

        self._param_amplitude_check = QCheckBox("Amplitude")
        self._param_amplitude_check.setChecked(True)
        self._param_acrophase_check = QCheckBox("Acrophase")
        self._param_acrophase_check.setChecked(True)
        self._param_mesor_check = QCheckBox("MESOR")
        self._param_mesor_check.setChecked(True)

        params_layout.addWidget(self._param_amplitude_check)
        params_layout.addWidget(self._param_acrophase_check)
        params_layout.addWidget(self._param_mesor_check)
        params_layout.addStretch()

        params_label = QLabel("Parameters to Compare:")
        params_label.setToolTip(
            "Select which rhythm parameters to compare between conditions.\n"
            "At least one parameter must be selected."
        )
        self._params_layout.addRow(params_label, params_widget)

        # Include Linear Component (checkbox)
        self._include_lin_comp_check = QCheckBox("Include linear component (trend)")
        self._include_lin_comp_check.setChecked(False)
        self._include_lin_comp_check.setToolTip(
            "Include linear trend component in comparison.\n"
            "Only available for Independent comparison method."
        )
        self._params_layout.addRow("", self._include_lin_comp_check)

        # Save CosinorPy plots checkbox (for CosinorPy methods - excludes Periodogram)
        self._save_cosinorpy_plots_check = QCheckBox("Save CosinorPy plots to folder")
        self._save_cosinorpy_plots_check.setChecked(False)
        self._save_cosinorpy_plots_check.setToolTip(
            "Save CosinorPy-generated plots to output folder.\n\n"
            "When checked, calls cosinor.plot_df_models() to generate and save\n"
            "diagnostic plots showing model fits, residuals, and predictions.\n"
            "Plots are saved to disk without displaying in GUI."
        )
        self._params_layout.addRow("", self._save_cosinorpy_plots_check)

        # =====================================================================
        # RHYTHM ANALYSIS - SPECIFIC PARAMETERS
        # =====================================================================

        # Asymmetry (for JTK and AR-JTK)
        self._asymmetry_spin = QDoubleSpinBox()
        self._asymmetry_spin.setRange(0.0, 1.0)
        self._asymmetry_spin.setValue(0.5)
        self._asymmetry_spin.setSingleStep(0.1)
        self._asymmetry_spin.setDecimals(1)
        self._asymmetry_spin.setToolTip(
            "Waveform asymmetry (0.0 to 1.0):\n"
            "- 0.5: Symmetric waveform (default)\n"
            "- <0.5: Peak earlier in cycle\n"
            "- >0.5: Peak later in cycle"
        )
        self._params_layout.addRow("Asymmetry:", self._asymmetry_spin)

        # AR Order/Lag (for AR-JTK)
        self._ar_lag_spin = QSpinBox()
        self._ar_lag_spin.setRange(1, 3)
        self._ar_lag_spin.setValue(1)
        self._ar_lag_spin.setToolTip(
            "Autoregressive lag order:\n"
            "- 1: Standard (default)\n"
            "- 2-3: For very noisy data"
        )
        self._params_layout.addRow("AR Lag:", self._ar_lag_spin)

        # Pre-whiten (for AR-JTK)
        self._prewhiten_check = QCheckBox("Force pre-whitening")
        self._prewhiten_check.setChecked(False)
        self._prewhiten_check.setToolTip(
            "Force or disable data pre-whitening before analysis"
        )
        self._params_layout.addRow("", self._prewhiten_check)

        # Resolution/Interval (for Cosine-Kendall)
        self._resolution_spin = QDoubleSpinBox()
        self._resolution_spin.setRange(0.1, 10.0)
        self._resolution_spin.setValue(0.5)
        self._resolution_spin.setSingleStep(0.1)
        self._resolution_spin.setDecimals(1)
        self._resolution_spin.setSuffix(" h")
        self._resolution_spin.setToolTip(
            "Lag sweep step size for acrophase search.\n"
            "Smaller values = finer resolution (more accurate acrophase, slower).\n"
            "Default 0.5h."
        )
        self._params_layout.addRow("Resolution:", self._resolution_spin)

        # Search Mode (for Cosinor OLS)
        self._search_mode_combo = QComboBox()
        self._search_mode_combo.addItems(['Fixed Period', 'Optimize Period'])
        self._search_mode_combo.setToolTip(
            "Period search mode:\n"
            "- Fixed Period: Use single specified period (e.g., 24h)\n"
            "- Optimize Period: Search for best period within range"
        )
        self._search_mode_combo.currentTextChanged.connect(self._on_search_mode_changed)
        self._params_layout.addRow("Search Mode:", self._search_mode_combo)

        # Target Period (for Fourier F24)
        self._target_period_spin = QDoubleSpinBox()
        self._target_period_spin.setRange(1, 72)
        self._target_period_spin.setValue(24.0)
        self._target_period_spin.setSuffix(" h")
        self._target_period_spin.setToolTip(
            "Target period to test (typically 24h for circadian rhythms)"
        )
        self._params_layout.addRow("Target Period:", self._target_period_spin)

        # N Periods (for Lomb-Scargle oversampling)
        self._n_periods_spin = QSpinBox()
        self._n_periods_spin.setRange(100, 10000)
        self._n_periods_spin.setValue(1000)
        self._n_periods_spin.setSingleStep(100)
        self._n_periods_spin.setToolTip(
            "Number of periods to evaluate in the specified range.\n"
            "Higher values = more precise peak detection but slower."
        )
        self._params_layout.addRow("Oversampling:", self._n_periods_spin)

        # Significance Threshold (for Lomb-Scargle)
        self._alpha_combo = QComboBox()
        self._alpha_combo.addItems(['0.05', '0.01', '0.001'])
        self._alpha_combo.setToolTip(
            "Significance threshold (alpha level) for rhythm detection"
        )
        self._params_layout.addRow("Significance Level:", self._alpha_combo)

        # Wavelet Type (for CWT)
        self._wavelet_combo = QComboBox()
        self._wavelet_combo.addItems(['Morlet (cmor1.5-1.0)', 'Ricker (Mexican Hat)'])
        self._wavelet_combo.setToolTip(
            "Wavelet type for CWT analysis:\n"
            "- Morlet: Standard for biological rhythms (default)\n"
            "- Ricker (Mexican Hat): Alternative wavelet"
        )
        self._params_layout.addRow("Wavelet Type:", self._wavelet_combo)

        # Sampling Interval (for CWT)
        self._sampling_interval_spin = QDoubleSpinBox()
        self._sampling_interval_spin.setRange(0.1, 24.0)
        self._sampling_interval_spin.setValue(0.0)  # 0 = auto-detect
        self._sampling_interval_spin.setSingleStep(0.1)
        self._sampling_interval_spin.setDecimals(1)
        self._sampling_interval_spin.setSuffix(" h")
        self._sampling_interval_spin.setSpecialValueText("Auto-detect")
        self._sampling_interval_spin.setToolTip(
            "Sampling interval in hours:\n"
            "- Auto-detect: Automatically calculate from data (default)\n"
            "- Manual: Specify if auto-detection fails"
        )
        self._params_layout.addRow("Sampling Interval:", self._sampling_interval_spin)

        # Detrending (for Spectral Analysis)
        self._detrending_check = QCheckBox("Detrend data (remove mean/linear trend)")
        self._detrending_check.setChecked(True)
        self._detrending_check.setToolTip(
            "Remove mean or linear trend before spectral analysis.\n"
            "Important for FFT-based methods."
        )
        self._params_layout.addRow("", self._detrending_check)

        # LME Parameters
        # Dependent Variable (for LME)
        self._dependent_var_combo = QComboBox()
        self._dependent_var_combo.setToolTip(
            "Dependent variable for Linear Mixed Effects model"
        )
        self._params_layout.addRow("Dependent Variable:", self._dependent_var_combo)

        # Fixed Effects (for LME) - Using QListWidget for multi-select
        self._fixed_effects_list = QListWidget()
        self._fixed_effects_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._fixed_effects_list.setMaximumHeight(80)
        self._fixed_effects_list.setToolTip(
            "Select fixed effects (e.g., Genotype, Treatment).\n"
            "Hold Ctrl to select multiple."
        )
        self._params_layout.addRow("Fixed Effects:", self._fixed_effects_list)

        # Random Effect (for LME)
        self._random_effect_combo = QComboBox()
        self._random_effect_combo.setToolTip(
            "Grouping variable for random effects in LME model.\n"
            "Use this to account for repeated measures (e.g., subject ID,\n"
            "animal ID, replicate, day). The model will estimate random\n"
            "intercepts for each group."
        )
        self._params_layout.addRow("Random Effect:", self._random_effect_combo)

        # Update visibility based on current method
        self._update_parameter_visibility()

    def _update_analysis_method_options(self, method_text: str):
        """Update Analysis Method combo options based on selected analysis method."""
        # Store current selection
        current_selection = self._analysis_method_combo.currentText()

        # Determine if this is a dependent data method
        is_dependent = 'Dependent' in method_text

        # Clear and repopulate
        self._analysis_method_combo.clear()

        if is_dependent:
            # For dependent data: only Sampling and Bootstrap
            # CosinorPy population methods accept: 'sampling' or 'bootstrap'
            self._analysis_method_combo.addItems(['Sampling', 'Bootstrap'])
            # Set default to Sampling for dependent data
            if current_selection in ['Sampling', 'Bootstrap']:
                self._analysis_method_combo.setCurrentText(current_selection)
            else:
                self._analysis_method_combo.setCurrentText('Sampling')
        else:
            # For independent data: only CI and Bootstrap
            # CosinorPy independent methods accept: 'CI' or 'bootstrap' (NOT sampling)
            self._analysis_method_combo.addItems(['CI', 'Bootstrap'])
            # Restore previous selection if valid
            if current_selection in ['CI', 'Bootstrap']:
                self._analysis_method_combo.setCurrentText(current_selection)
            else:
                self._analysis_method_combo.setCurrentText('CI')

        # Also update criterium options: population_fit_group does not return log-likelihood,
        # so AIC/BIC/Log-Likelihood are only valid for independent data.
        self._update_criterium_options(method_text)

    def _update_criterium_options(self, method_text: str):
        """Update Criterium combo to only show options available in the current data type."""
        current_selection = self._criterium_combo.currentText()
        is_dependent = 'Dependent' in method_text

        self._criterium_combo.clear()

        if is_dependent:
            # population_fit_group does not output log-likelihood, so AIC/BIC/Log-Likelihood
            # cannot be computed. Only RSS is valid.
            self._criterium_combo.addItems(['RSS'])
            self._criterium_combo.setCurrentText('RSS')
        else:
            # fit_group outputs log-likelihood, so all criteria are valid.
            self._criterium_combo.addItems(['RSS', 'AIC', 'BIC', 'Log-Likelihood'])
            if current_selection in ['RSS', 'AIC', 'BIC', 'Log-Likelihood']:
                self._criterium_combo.setCurrentText(current_selection)
            else:
                self._criterium_combo.setCurrentText('RSS')

    def _update_limo_analysis_options(self, comparison_method: str):
        """Update Analysis Method options when Comparison Method changes (for Compare Conditions)."""
        # Only update if we're in a Compare Conditions method
        method_text = self._method_combo.currentText()
        if 'Compare Conditions' not in method_text:
            return

        current_selection = self._analysis_method_combo.currentText()
        self._analysis_method_combo.clear()

        if comparison_method == 'LimoRhyde':
            # LimoRhyde options
            self._analysis_method_combo.addItems(['None', 'CI1', 'Bootstrap1', 'CI2', 'Bootstrap2'])
            if current_selection in ['None', 'CI1', 'Bootstrap1', 'CI2', 'Bootstrap2']:
                self._analysis_method_combo.setCurrentText(current_selection)
            else:
                self._analysis_method_combo.setCurrentText('None')
        else:
            # Independent options (CI or Bootstrap)
            self._analysis_method_combo.addItems(['CI', 'Bootstrap'])
            if current_selection in ['CI', 'Bootstrap']:
                self._analysis_method_combo.setCurrentText(current_selection)
            else:
                self._analysis_method_combo.setCurrentText('CI')

    def _on_components_changed(self):
        """Handle changes to the components text field."""
        # Only update if we're in Compare Conditions method
        method_text = self._method_combo.currentText()
        if "Compare Conditions" in method_text:
            self._update_compare_conditions_parameters()
        # Also update visibility for Cosinor Independent
        elif method_text == "Cosinor (Independent Data)":
            self._update_cosinor_independent_params_visibility()
        # Also update visibility for Cosinor Dependent
        elif method_text == "Cosinor (Dependent Data)":
            self._update_cosinor_dependent_params_visibility()
        # Also update visibility for Nonlinear methods
        elif "Nonlinear" in method_text:
            self._update_nonlinear_independent_params_visibility()

    def _on_period_changed(self):
        """Handle changes to period spinboxes."""
        method_text = self._method_combo.currentText()
        if method_text == "Cosinor (Independent Data)":
            self._update_cosinor_independent_params_visibility()

    def _on_search_mode_changed(self):
        """Handle changes to Cosinor OLS search mode."""
        method_text = self._method_combo.currentText()
        if method_text == "Cosinor (OLS)":
            self._update_cosinor_ols_period_visibility()

    def _update_cosinor_ols_period_visibility(self):
        """Update period widget visibility based on search mode for Cosinor OLS."""
        search_mode = self._search_mode_combo.currentText()

        # Get the period widget (it's a composite widget with min, max, step)
        # We need to find it in the layout
        for i in range(self._params_layout.rowCount()):
            label_item = self._params_layout.itemAt(i, QFormLayout.LabelRole)
            if label_item and label_item.widget():
                label_widget = label_item.widget()
                if label_widget.text() == "Period:":
                    field_item = self._params_layout.itemAt(i, QFormLayout.FieldRole)
                    if field_item and field_item.widget():
                        period_widget = field_item.widget()
                        # The period widget is a QWidget containing HBoxLayout with min, "to", max, "Step:", step
                        layout = period_widget.layout()
                        if layout:
                            if search_mode == "Fixed Period":
                                # Show only the first spinbox (min), hide the rest
                                # Items: 0=min, 1="to", 2=max, 3="Step:", 4=step
                                layout.itemAt(0).widget().setVisible(True)  # min spinbox (used as fixed period)
                                layout.itemAt(1).widget().setVisible(False) # "to" label
                                layout.itemAt(2).widget().setVisible(False) # max spinbox
                                layout.itemAt(3).widget().setVisible(False) # "Step:" label
                                layout.itemAt(4).widget().setVisible(False) # step spinbox
                                # Update tooltip for clarity
                                layout.itemAt(0).widget().setToolTip("Fixed period for cosinor analysis (e.g., 24h)")
                            else:  # "Optimize Period"
                                # Show all widgets (min, "to", max, "Step:", step)
                                layout.itemAt(0).widget().setVisible(True)
                                layout.itemAt(1).widget().setVisible(True)
                                layout.itemAt(2).widget().setVisible(True)
                                layout.itemAt(3).widget().setVisible(True)
                                layout.itemAt(4).widget().setVisible(True)
                                # Restore original tooltip
                                layout.itemAt(0).widget().setToolTip("")
                    break

    def _update_cosinor_independent_params_visibility(self):
        """Update parameter visibility for Cosinor (Independent Data) based on n_components and period."""
        # Only apply when in Cosinor Independent mode
        method_text = self._method_combo.currentText()
        if method_text != "Cosinor (Independent Data)":
            return

        # Parse n_components
        try:
            n_components = self._parse_components(self._components_edit.text())
        except:
            # If parsing fails, show all parameters
            n_components = []

        # Determine if we have multiple periods
        period_min = self._period_min_spin.value()
        period_max = self._period_max_spin.value()
        has_multiple_periods = (period_min != period_max)

        # Determine if we have component > 1 (either single N>1 or multiple components)
        has_component_gt_one = (
            (len(n_components) == 1 and n_components[0] > 1) or
            len(n_components) > 1
        )

        # Logic for Independent Data:
        # - Criterium (Best Period): visible only when has_multiple_periods
        # - Analysis Method: visible only when has_component_gt_one
        # - Bootstrap Size: visible only when Analysis Method = Bootstrap AND has_component_gt_one

        # Show/hide Criterium (Best Period)
        self._criterium_label.setVisible(has_multiple_periods)
        self._criterium_combo.setVisible(has_multiple_periods)

        # Show/hide Analysis Method
        self._analysis_method_label.setVisible(has_component_gt_one)
        self._analysis_method_combo.setVisible(has_component_gt_one)

        # Show/hide Bootstrap Size (only if Analysis Method is Bootstrap AND has_component_gt_one)
        analysis_method = self._analysis_method_combo.currentText()
        show_bootstrap = has_component_gt_one and analysis_method == 'Bootstrap'
        self._bootstrap_size_label.setVisible(show_bootstrap)
        self._bootstrap_size_spin.setVisible(show_bootstrap)

    def _update_cosinor_dependent_params_visibility(self):
        """Update parameter visibility for Cosinor (Dependent Data) based on n_components."""
        # Only apply when in Cosinor Dependent mode
        method_text = self._method_combo.currentText()
        if method_text != "Cosinor (Dependent Data)":
            return

        # Parse n_components
        try:
            n_components = self._parse_components(self._components_edit.text())
        except:
            # If parsing fails, show all parameters
            n_components = []

        # Determine if we have component > 1 (either single N>1 or multiple components)
        has_component_gt_one = (
            (len(n_components) == 1 and n_components[0] > 1) or
            len(n_components) > 1
        )

        # Logic for Dependent Data:
        # - Criterium: visible only when has_component_gt_one (for get_best_models_population)
        # - Analysis Method: visible only when has_component_gt_one (for analyse_best_models_population)
        # - Bootstrap Size: visible only when Analysis Method = Bootstrap AND has_component_gt_one

        # Show/hide Criterium
        self._criterium_label.setVisible(has_component_gt_one)
        self._criterium_combo.setVisible(has_component_gt_one)

        # Show/hide Analysis Method
        self._analysis_method_label.setVisible(has_component_gt_one)
        self._analysis_method_combo.setVisible(has_component_gt_one)

        # Show/hide Bootstrap Size (only if Analysis Method is Bootstrap AND has_component_gt_one)
        analysis_method = self._analysis_method_combo.currentText()
        show_bootstrap = has_component_gt_one and (analysis_method == 'Bootstrap' or analysis_method == 'Sampling')
        # Actually, only show bootstrap size if method is Bootstrap
        show_bootstrap = has_component_gt_one and analysis_method == 'Bootstrap'
        self._bootstrap_size_label.setVisible(show_bootstrap)
        self._bootstrap_size_spin.setVisible(show_bootstrap)

    def _update_nonlinear_independent_params_visibility(self):
        """
        Update parameter visibility for Nonlinear methods based on n_components.

        For nonlinear analysis:
        - amplification and lin_comp are OUTPUTS (results), not inputs
        - Model Type doesn't apply (always uses generalized cosinor model)
        - Criterium doesn't apply (best model selected by p-value automatically)
        - Analysis Method doesn't apply (no CI vs Bootstrap option)
        - Bootstrap Size: only needed for Independent methods with n_components > 1
          (for amplitude/acrophase stats via bootstrap_generalized_cosinor_n_comp_group).
          Dependent/population methods never use bootstrap — stats are derived from
          between-replicate variance, regardless of n_components.
        """
        method_text = self._method_combo.currentText()

        # Only apply to Nonlinear methods
        if "Nonlinear" not in method_text:
            return

        # Dependent/population nonlinear methods never use bootstrap
        is_dependent = "Dependent" in method_text
        if is_dependent:
            self._bootstrap_size_label.setVisible(False)
            self._bootstrap_size_spin.setVisible(False)
            return

        # Parse n_components
        try:
            n_components = self._parse_components(self._components_edit.text())
        except:
            n_components = [1]

        # Determine if we have component > 1 (either single N>1 or multiple components for auto-selection)
        has_component_gt_one = (
            (len(n_components) == 1 and n_components[0] > 1) or
            len(n_components) > 1
        )

        # Bootstrap Size: only show when n_components > 1
        # For 1-component models, stats are calculated analytically (no bootstrap needed)
        self._bootstrap_size_label.setVisible(has_component_gt_one)
        self._bootstrap_size_spin.setVisible(has_component_gt_one)

    def _update_nonlinear_independent_period_visibility(self):
        """
        Update period widget visibility for Method 8 (Nonlinear Compare Independent).

        When use_dependent_model is unchecked (independent model): allow different periods
        per condition. Show per-condition spinboxes when exactly 2 conditions are loaded.
        When checked (shared period): show a single period spinbox for both conditions.
        For 3+ conditions in independent mode: fall back to single shared period
        (CosinorPy compare_pairs uses global period1/period2, not per-pair periods).
        """
        use_dep = self._use_dependent_model_check.isChecked()

        if not use_dep:
            # Independent: attempt to show per-condition periods
            n_conditions = 0
            conditions = []
            if hasattr(self, '_loader') and self._loader and self._source_type in ('csv', 'dam'):
                try:
                    dataset_info = self._loader.get_dataset_info()
                    n_conditions = len(dataset_info.conditions) if dataset_info.conditions else 0
                    conditions = list(dataset_info.conditions) if dataset_info.conditions else []
                except Exception:
                    pass

            if n_conditions == 2:
                self._hide_param("Period:")
                self._show_param("Periods:")
                self._period_cond1_label.setText(f"Period for {conditions[0]}:")
                self._period_cond2_label.setText(f"Period for {conditions[1]}:")
                self._period_multi_cond_info.setVisible(False)
            else:
                # 3+ conditions or unknown: shared period only
                self._show_param("Period:")
                self._hide_param("Periods:")
                self._period_multi_cond_info.setVisible(False)
        else:
            # Dependent/shared period
            self._show_param("Period:")
            self._hide_param("Periods:")
            self._period_multi_cond_info.setVisible(False)

    def _update_compare_conditions_parameters(self):
        """Update parameter visibility based on n_components and comparison method for Compare Conditions."""
        method_text = self._method_combo.currentText()
        if "Compare Conditions" not in method_text:
            return

        # Determine if this is dependent or independent
        is_dependent = "Dependent" in method_text

        # Parse n_components from the text field
        components_text = self._components_edit.text().strip()
        try:
            # Parse comma-separated values
            components = [int(x.strip()) for x in components_text.split(',') if x.strip()]
            is_single_component = len(components) == 1 and components[0] == 1
        except ValueError:
            # If parsing fails, assume multi-component
            is_single_component = False

        # Hide all conditional parameters first
        self._hide_param("Comparison Type:")
        self._hide_param("Comparison Method:")
        self._hide_param("Analysis Method:")
        self._hide_param("Parameters to Compare:")
        self._hide_param("Bootstrap Size:")
        self._hide_checkbox(self._include_lin_comp_check)

        # Determine period widget visibility for Independent Models
        # For Independent data with "Independent Models" comparison type and exactly 2 conditions:
        # show per-condition periods
        show_per_condition_periods = False
        show_multi_cond_info = False

        if not is_dependent and is_single_component:
            # Get comparison type
            comparison_type = self._comparison_type_combo.currentText()

            if comparison_type == "Independent Models":
                # Count unique conditions in data
                if hasattr(self, '_loader') and self._loader and self._source_type in ('csv', 'dam'):
                    dataset_info = self._loader.get_dataset_info()
                    n_conditions = len(dataset_info.conditions) if dataset_info.conditions else 0

                    if n_conditions == 2:
                        # Exactly 2 conditions: show per-condition periods
                        show_per_condition_periods = True
                        # Update condition labels
                        conditions = dataset_info.conditions
                        self._period_cond1_label.setText(f"Period for {conditions[0]}:")
                        self._period_cond2_label.setText(f"Period for {conditions[1]}:")
                    elif n_conditions > 2:
                        # More than 2 conditions: show info message
                        show_multi_cond_info = True

        if is_dependent:
            # Dependent data: simpler UI (no Comparison Type or Method)
            if not is_single_component:
                # Multi-component dependent: show analysis options
                self._show_param("Analysis Method:")
                self._show_param("Parameters to Compare:")
                self._show_checkbox(self._include_lin_comp_check)

                # Update Analysis Method combo to show CI and Permutation for dependent
                current_selection = self._analysis_method_combo.currentText()
                self._analysis_method_combo.clear()
                self._analysis_method_combo.addItems(['CI', 'Permutation'])
                # Restore previous selection if valid
                if current_selection in ['CI', 'Permutation']:
                    self._analysis_method_combo.setCurrentText(current_selection)
                else:
                    self._analysis_method_combo.setCurrentText('CI')
        else:
            # Independent data: existing complex logic
            if is_single_component:
                # Single component: Show only Comparison Type
                # Note: single component methods (test_cosinor_pairs, test_cosinor_pairs_independent)
                # do NOT use bootstrap_size, Analysis Method, or Parameters to Compare
                self._show_param("Comparison Type:")
            else:
                # Multi-component: Show Comparison Method and conditional parameters
                self._show_param("Comparison Method:")
                self._show_param("Bootstrap Size:")  # Only for multi-component

                # Get current comparison method
                comparison_method = self._comparison_method_combo.currentText()

                if comparison_method == 'Independent':
                    # Independent method: Show Analysis Method, Parameters to Compare, and Lin Comp
                    self._show_param("Analysis Method:")
                    self._show_param("Parameters to Compare:")
                    self._show_checkbox(self._include_lin_comp_check)
                else:
                    # LimoRhyde method: Show Analysis Method and Parameters to Compare (no Lin Comp)
                    self._show_param("Analysis Method:")
                    self._show_param("Parameters to Compare:")

        # Update period widget visibility ONLY for Independent Models special cases
        # For all other cases, ensure standard Period spinboxes are visible
        if show_per_condition_periods:
            # Show per-condition periods, hide standard period spinboxes
            self._hide_param("Period:")
            self._show_param("Periods:")
            self._period_multi_cond_info.setVisible(False)
        elif show_multi_cond_info:
            # Hide both period widgets, show info message
            self._hide_param("Period:")
            self._hide_param("Periods:")
            self._period_multi_cond_info.setVisible(True)
        else:
            # Default case: show standard Period, hide Periods and info
            self._show_param("Period:")
            self._hide_param("Periods:")
            self._period_multi_cond_info.setVisible(False)

    def _update_parameter_visibility(self):
        """Show/hide parameters based on selected method."""
        # Check if params layout exists (may be called during initialization)
        if not hasattr(self, '_params_layout') or self._params_layout is None:
            return

        method_text = self._method_combo.currentText()
        module_text = self._module_combo.currentText()
        print(f"[DEBUG _update_parameter_visibility] method_text: '{method_text}', module_text: '{module_text}'")

        # Update Analysis Method options based on data type (independent vs dependent)
        if module_text == "CosinorPy":
            self._update_analysis_method_options(method_text)

        # Default: hide ALL parameters
        self._hide_param("Components:")
        self._hide_param("Period:")
        self._hide_param("Periods:")
        self._hide_param("Loss Function:")
        self._hide_param("F-Scale:")
        self._hide_param("Max Iterations:")
        self._hide_param("Harmonics:")
        self._hide_param("Permutations:")
        self._hide_param("Amplification:")
        self._hide_param("Linear Component:")
        self._hide_param("Periodogram Type:")
        self._hide_param("Max Period:")
        self._hide_param("Model Type:")
        # Note: Criterium label changed to "Criterium (Best Period):"
        if hasattr(self, '_criterium_label'):
            self._criterium_label.setVisible(False)
            self._criterium_combo.setVisible(False)
        # Note: Analysis Method and Bootstrap Size have custom labels
        if hasattr(self, '_analysis_method_label'):
            self._analysis_method_label.setVisible(False)
            self._analysis_method_combo.setVisible(False)
        if hasattr(self, '_bootstrap_size_label'):
            self._bootstrap_size_label.setVisible(False)
            self._bootstrap_size_spin.setVisible(False)
        self._hide_param("Comparison Type:")
        self._hide_param("Comparison Method:")
        self._hide_param("Parameters to Compare:")
        self._hide_checkbox(self._prominent_check)
        self._hide_checkbox(self._use_dependent_model_check)
        self._hide_checkbox(self._save_cosinorpy_plots_check)
        self._hide_checkbox(self._include_lin_comp_check)
        self._period_multi_cond_info.setVisible(False)
        # Hide Rhythm Analysis specific parameters
        self._hide_param("Asymmetry:")
        self._hide_param("AR Lag:")
        self._hide_checkbox(self._prewhiten_check)
        self._hide_param("Resolution:")
        self._hide_param("Search Mode:")
        self._hide_param("Target Period:")
        self._hide_param("Oversampling:")
        self._hide_param("Significance Level:")
        self._hide_param("Wavelet Type:")
        self._hide_param("Sampling Interval:")
        self._hide_checkbox(self._detrending_check)
        self._hide_param("Dependent Variable:")
        self._hide_param("Fixed Effects:")
        self._hide_param("Random Effect:")

        # Always restore the full period widget (min/to/max/Step/step) so that
        # _update_cosinor_ols_period_visibility is the only thing that can hide
        # sub-items, and only when Cosinor OLS is active.
        self._restore_period_widget()

        # =====================================================================
        # COSINORPY - NEW REFACTORED METHODS
        # =====================================================================
        if module_text == "CosinorPy":

            # Method 1: Periodogram Analysis
            # No parameters needed - periodogram_df() only takes df and folder
            if method_text == "Periodogram Analysis":
                pass  # No parameters to show

            # Method 2: Cosinor (Independent Data)
            elif method_text == "Cosinor (Independent Data)":
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_param("Model Type:")
                self._show_checkbox(self._save_cosinorpy_plots_check)
                # Conditionally show Criterium, Analysis Method, Bootstrap Size based on n_components and period
                self._update_cosinor_independent_params_visibility()

            # Method 3: Cosinor (Dependent Data)
            elif method_text == "Cosinor (Dependent Data)":
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_param("Model Type:")
                self._show_checkbox(self._save_cosinorpy_plots_check)
                # Conditionally show Criterium, Analysis Method, Bootstrap Size based on n_components
                self._update_cosinor_dependent_params_visibility()

            # Method 4: Compare Conditions (Independent)
            elif method_text == "Compare Conditions (Independent)":
                # Always show basic parameters
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_checkbox(self._save_cosinorpy_plots_check)

                # Update conditional parameters based on current n_components value
                self._update_compare_conditions_parameters()

            # Method 5: Compare Conditions (Dependent)
            elif method_text == "Compare Conditions (Dependent)":
                # Always show basic parameters
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_checkbox(self._save_cosinorpy_plots_check)

                # Update conditional parameters based on current n_components value
                self._update_compare_conditions_parameters()

            # Method 6: Nonlinear (Independent Data)
            # Model: Y = A + B·exp(C·t)·cos(2π·t/P + φ) + D·t
            # Parameters C (amplification) and D (lin_comp) are OUTPUTS, not inputs
            # Bootstrap is only needed for n_components > 1
            elif method_text == "Nonlinear (Independent Data)":
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_checkbox(self._save_cosinorpy_plots_check)
                # Show bootstrap size only if n_components might be > 1
                self._update_nonlinear_independent_params_visibility()

            # Method 7: Nonlinear (Dependent Data)
            # Same model as Independent, but for population/longitudinal data
            elif method_text == "Nonlinear (Dependent Data)":
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_checkbox(self._save_cosinorpy_plots_check)
                # Show bootstrap size only if n_components might be > 1
                self._update_nonlinear_independent_params_visibility()

            # Method 8: Nonlinear Compare (Independent)
            # Compare conditions using nonlinear model
            elif method_text == "Nonlinear Compare (Independent)":
                self._show_param("Components:")
                self._show_checkbox(self._use_dependent_model_check)
                self._show_checkbox(self._save_cosinorpy_plots_check)
                # Show bootstrap size only if n_components might be > 1
                self._update_nonlinear_independent_params_visibility()
                # Period widget visibility depends on use_dependent_model state
                self._update_nonlinear_independent_period_visibility()

            # Method 9: Nonlinear Compare (Dependent)
            # Compare conditions using nonlinear model for population data
            elif method_text == "Nonlinear Compare (Dependent)":
                self._show_param("Period:")
                self._show_param("Components:")
                self._show_checkbox(self._save_cosinorpy_plots_check)
                # Show bootstrap size only if n_components might be > 1
                self._update_nonlinear_independent_params_visibility()

        # =====================================================================
        # CIRCACOMPARE
        # =====================================================================
        elif module_text == "CircaCompare":
            self._show_param("Period:")
            self._show_param("Loss Function:")
            self._show_param("F-Scale:")
            self._show_param("Max Iterations:")

        # =====================================================================
        # RHYTHM ANALYSIS
        # =====================================================================
        elif module_text == "Rhythm Analysis":

            # 1. JTK Cycle (Python-JTK)
            if method_text == "JTK Cycle":
                self._show_param("Period:")  # Period Range (Min-Max-Step)
                self._show_param("Asymmetry:")

            # 2. AR-JTK
            elif method_text == "AR-JTK":
                self._show_param("Period:")  # Period Range (Min-Max-Step)
                self._show_param("Asymmetry:")
                self._show_param("AR Lag:")
                self._show_checkbox(self._prewhiten_check)

            # 3. Cosine-Kendall
            elif method_text == "Cosine-Kendall":
                self._show_param("Period:")  # Period Range (Min-Max-Step)
                self._show_param("Resolution:")

            # 4. Cosinor (OLS)
            elif method_text == "Cosinor (OLS)":
                self._show_param("Search Mode:")
                self._show_param("Period:")  # Period Range (for optimization mode)
                # Update period widget visibility based on search mode
                self._update_cosinor_ols_period_visibility()

            # 5. Harmonic Cosinor
            elif method_text == "Harmonic Cosinor":
                self._show_param("Period:")  # Fixed Period (single value expected)
                self._show_param("Harmonics:")

            # 6. Fourier F24
            elif method_text == "Fourier F24":
                self._show_param("Target Period:")
                self._show_param("Permutations:")

            # 7. Lomb-Scargle
            elif method_text == "Lomb-Scargle":
                self._show_param("Period:")  # Period Range (Min-Max), Step hidden
                self._show_param("Oversampling:")
                self._show_param("Significance Level:")
                self._period_min_spin.setValue(18.0)
                self._period_max_spin.setValue(32.0)
                self._hide_period_step()

            # 8. Wavelet (CWT)
            elif method_text == "Wavelet (CWT)":
                self._show_param("Period:")  # Period Range (Min-Max), Step hidden
                self._show_param("Wavelet Type:")
                self._show_param("Sampling Interval:")
                self._hide_period_step()

            # 9. Spectral Analysis (Periodogram)
            elif method_text == "Spectral Analysis (Periodogram)":
                self._show_param("Periodogram Type:")
                self._show_param("Max Period:")
                self._show_checkbox(self._detrending_check)
                self._show_checkbox(self._prominent_check)

            # 10. Linear Mixed Effects (Cosinor-based)
            elif method_text == "Linear Mixed Effects":
                self._show_param("Period:")  # Target period for cosinor transformation
                self._show_param("Random Effect:")  # Grouping variable (subject ID, replicate, etc.)

        # =====================================================================
        # AI CONSENSUS
        # =====================================================================
        elif module_text == "AI Consensus":
            # No parameters needed - the meta-classifier uses defaults for all sub-methods
            pass

        # =====================================================================
        # VISUALIZATION (DAM data)
        # =====================================================================
        elif module_text == "Visualization":
            # Activity Profile doesn't require any parameters
            # All parameters are hidden by default
            pass

        # =====================================================================
        # COMPARISON FRAME VISIBILITY
        # =====================================================================
        # Show comparison frame ONLY for pairwise comparison methods (user selects 2 specific conditions)
        # NOT for "Compare All", "Compare Conditions", or "Nonlinear Compare" (which use all conditions automatically)
        is_nonlinear_compare = "Nonlinear Compare" in method_text
        is_circacompare_compare = "Compare Groups" in method_text
        is_pairwise_comparison = "Compare" in method_text and "Compare All" not in method_text and "Compare Conditions" not in method_text and not is_nonlinear_compare and not is_circacompare_compare
        is_compare_all_or_conditions = "Compare All" in method_text or "Compare Conditions" in method_text or is_nonlinear_compare or is_circacompare_compare

        self._compare_frame.setVisible(is_pairwise_comparison)

        # For pairwise comparison methods, hide condition list (use dropdowns instead)
        # For Compare All/Conditions/Nonlinear Compare, show condition list but make it read-only (user shouldn't select)
        if is_compare_all_or_conditions:
            self._cond_list.setVisible(True)
            # Disable selection for Compare All/Conditions/Nonlinear Compare (all conditions will be used automatically)
            self._cond_list.setSelectionMode(QAbstractItemView.NoSelection)
        elif is_pairwise_comparison:
            self._cond_list.setVisible(False)
        else:
            # For normal single-group methods, show condition list with selection enabled
            self._cond_list.setVisible(True)
            self._cond_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # =====================================================================
        # ROSBASH COMPARISON TYPE VISIBILITY
        # =====================================================================
        # Show comparison type selector for Rosbash data with comparison methods
        is_rosbash = self._source_type == 'rosbash'
        is_comparison_method = (is_pairwise_comparison or is_compare_all_or_conditions)

        self._rosbash_compare_type_frame.setVisible(is_rosbash and is_comparison_method)

    def _restore_period_widget(self):
        """Restore all sub-items of the Period: widget to visible.

        _update_cosinor_ols_period_visibility hides sub-items (to / max / Step / step)
        when in Fixed Period mode.  Those items stay hidden when the user switches to
        another method.  Call this at the start of _update_parameter_visibility so
        every method that shows the Period: row gets the full range widget.
        """
        for i in range(self._params_layout.rowCount()):
            label_item = self._params_layout.itemAt(i, QFormLayout.LabelRole)
            if label_item and label_item.widget() and label_item.widget().text() == "Period:":
                field_item = self._params_layout.itemAt(i, QFormLayout.FieldRole)
                if field_item and field_item.widget():
                    layout = field_item.widget().layout()
                    if layout:
                        for j in range(layout.count()):
                            child = layout.itemAt(j)
                            if child and child.widget():
                                child.widget().setVisible(True)
                break

    def _hide_period_step(self):
        """Hide the Step sub-widgets from the Period row (irrelevant for Lomb-Scargle)."""
        for i in range(self._params_layout.rowCount()):
            label_item = self._params_layout.itemAt(i, QFormLayout.LabelRole)
            if label_item and label_item.widget() and label_item.widget().text() == "Period:":
                field_item = self._params_layout.itemAt(i, QFormLayout.FieldRole)
                if field_item and field_item.widget():
                    layout = field_item.widget().layout()
                    if layout and layout.count() >= 5:
                        layout.itemAt(3).widget().setVisible(False)  # "Step:" label
                        layout.itemAt(4).widget().setVisible(False)  # step spinbox
                break

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

    def _show_checkbox(self, checkbox: QCheckBox):
        """Show a checkbox widget."""
        for i in range(self._params_layout.rowCount()):
            field = self._params_layout.itemAt(i, QFormLayout.FieldRole)
            if field and field.widget() == checkbox:
                checkbox.setVisible(True)
                break

    def _hide_checkbox(self, checkbox: QCheckBox):
        """Hide a checkbox widget."""
        for i in range(self._params_layout.rowCount()):
            field = self._params_layout.itemAt(i, QFormLayout.FieldRole)
            if field and field.widget() == checkbox:
                checkbox.setVisible(False)
                break

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def _parse_components(self, text: str) -> list:
        """
        Parse comma-separated components string to list of integers.

        Args:
            text: String like "1" or "1,2,3"

        Returns:
            List of integers, e.g., [1] or [1, 2, 3]
            Returns [1] as default fallback on parse error.
        """
        try:
            # Remove whitespace and split by comma
            parts = [p.strip() for p in text.split(',') if p.strip()]
            # Convert to integers
            components = [int(p) for p in parts]
            # Validate range (1-6 is typical for cosinor analysis)
            if not all(1 <= c <= 6 for c in components):
                print(f"[WARNING] Components out of range (1-6): {components}, using default [1]")
                return [1]
            return components
        except (ValueError, AttributeError) as e:
            print(f"[WARNING] Failed to parse components '{text}': {e}, using default [1]")
            return [1]

    def _parse_period(self, text: str):
        """
        Parse period string to either float or list of floats.

        Args:
            text: String like "24" or "24, 12, 48" or "23-25" (range format)

        Returns:
            Float for single period, or list of floats for multiple periods.
            Returns 24.0 as default fallback on parse error.
        """
        try:
            # Check if it's a range format (e.g., "23-25")
            if '-' in text and ',' not in text:
                # Range format: generate list
                parts = text.split('-')
                if len(parts) == 2:
                    start = float(parts[0].strip())
                    end = float(parts[1].strip())
                    # Generate list with step=1.0
                    periods = []
                    current = start
                    while current <= end:
                        periods.append(current)
                        current += 1.0
                    return periods if len(periods) > 1 else periods[0]

            # Check if comma-separated (multiple periods)
            if ',' in text:
                # Multiple periods: return list
                parts = [p.strip() for p in text.split(',') if p.strip()]
                periods = [float(p) for p in parts]
                return periods if len(periods) > 1 else periods[0]

            # Single period: return float
            return float(text.strip())

        except (ValueError, AttributeError) as e:
            print(f"[WARNING] Failed to parse period '{text}': {e}, using default 24.0")
            return 24.0

    # =========================================================================
    # ORIGINAL EVENT HANDLERS
    # =========================================================================
    
    def _on_module_changed(self, index: int):
        """Handle module selection change."""
        self._method_combo.clear()

        # Use module name instead of index (indices shift when modules are removed)
        module_name = self._module_combo.currentText()

        # Check if subject column exists (for dependent data methods)
        has_subject_col = False
        if hasattr(self, '_loader') and self._loader and self._source_type in ('csv', 'dam'):
            dataset_info = self._loader.get_dataset_info()
            has_subject_col = dataset_info.subject_column is not None

        if module_name == "CosinorPy":  # CosinorPy - NEW REFACTORED METHODS
            methods = [
                ("Periodogram Analysis", "Spectral analysis to identify dominant periods"),
                ("Cosinor (Independent Data)", "Fit cosinor model to independent data"),
                ("Compare Conditions (Independent)", "Compare conditions for independent data"),
                ("Nonlinear (Independent Data)", "Nonlinear cosinor with damping/forcing (independent)"),
                ("Nonlinear Compare (Independent)", "Compare nonlinear parameters (independent data)")
            ]

            # Only add dependent methods if subject column exists
            if has_subject_col:
                methods.insert(2, ("Cosinor (Dependent Data)", "Fit cosinor model to dependent/population data"))
                methods.insert(4, ("Compare Conditions (Dependent)", "Compare conditions for dependent/population data"))
                methods.insert(6, ("Nonlinear (Dependent Data)", "Nonlinear cosinor with damping/forcing (dependent)"))
                methods.append(("Nonlinear Compare (Dependent)", "Compare nonlinear parameters (dependent data)"))

        elif module_name == "CircaCompare":  # CircaCompare
            methods = [
                ("Single Fit", "Robust cosinor fitting - Parsons, Rex, et al. \"CircaCompare: a method to estimate and statistically support differences in mesor, amplitude and phase, between circadian rhythms.\" Bioinformatics 36.4 (2020): 1208-1212."),
                ("Compare Groups", "Compare parameters between groups - Parsons, Rex, et al. \"CircaCompare: a method to estimate and statistically support differences in mesor, amplitude and phase, between circadian rhythms.\" Bioinformatics 36.4 (2020): 1208-1212.")
            ]
        elif module_name == "Rhythm Analysis":  # Rhythm Analysis
            methods = [
                ("JTK Cycle",
                 "Nonparametric rhythm detection using Kendall's tau correlation with triangle waveforms. "
                 "Robust to outliers and non-normal distributions. Uses Benjamini-Hochberg correction for multiple testing. "
                 "Suitable for: Independent data (cross-sectional designs)."),
                ("AR-JTK",
                 "JTK Cycle with autoregressive noise correction. Detects and corrects for autocorrelation in residuals "
                 "using Ljung-Box test and AR prewhitening. Better statistical power for time series with temporal dependencies. "
                 "Suitable for: Dependent data (longitudinal/repeated measures)."),
                ("Cosine-Kendall",
                 "Nonparametric rhythm detection using Kendall's tau with cosine templates. Similar to JTK but assumes "
                 "symmetric waveforms. Robust to outliers. "
                 "Suitable for: Independent data (cross-sectional designs)."),
                ("Cosinor (OLS)",
                 "Parametric cosinor analysis using ordinary least squares regression. Fits y = M + A*cos(wt - phi). "
                 "Provides MESOR, amplitude, acrophase with confidence intervals and F-test significance. Can search for optimal period. "
                 "Assumes independence between observations. "
                 "Suitable for: Independent data (cross-sectional), or pre-averaged time series."),
                ("Harmonic Cosinor",
                 "Extended cosinor with multiple harmonics (up to 4) for complex or asymmetric waveforms. Uses OLS regression "
                 "with F-tests for overall significance and incremental harmonic contribution. Assumes independence between observations. "
                 "Suitable for: Independent data (cross-sectional), or pre-averaged time series."),
                ("Fourier F24",
                 "Effect size measure (F24 score) comparing circadian signal power to noise. Based on Wijnen et al. (2006). "
                 "Requires at least 2 biological replicates per time point to estimate noise variance. "
                 "Suitable for: Independent data with replicates at each timepoint."),
                ("Lomb-Scargle",
                 "Periodogram method designed for unevenly sampled or missing data. Detects dominant period without "
                 "requiring interpolation. Reports False Alarm Probability (FAP) for significance. "
                 "Suitable for: Independent data with irregular sampling, or single time series (exploratory)."),
                ("Spectral Analysis (Periodogram)",
                 "FFT-based power spectral density analysis with interactive visualization. Identifies dominant frequencies "
                 "and harmonics. Automatically interpolates non-uniform data. "
                 "Suitable for: Dependent data (longitudinal time series)."),
                ("Wavelet (CWT)",
                 "Continuous Wavelet Transform for time-frequency analysis. Shows how rhythm period and amplitude change "
                 "over time. Detects non-stationary rhythms. Generates scalogram visualization. "
                 "Suitable for: Dependent data (longitudinal time series)."),
                ("Linear Mixed Effects",
                 "Cosinor-based mixed effects model: y ~ cos(wt) + sin(wt) + (1|random_effect). Accounts for individual "
                 "variability and hierarchical data structure. Uses likelihood ratio test for rhythm significance. "
                 "Suitable for: Dependent data with repeated measures and grouping factors (e.g., individual subjects).")
            ]
        elif module_name == "AI Consensus":  # AI Consensus
            methods = [
                ("Consensus Rhythmicity Score",
                 "AI-powered meta-classifier that combines evidence from JTK_CYCLE, Cosinor, "
                 "Lomb-Scargle, Fourier F24, and Harmonic Cosinor into a single rhythmicity "
                 "probability score (0-1). Uses a pre-trained Random Forest model trained "
                 "on 3,299 instances (1,600 synthetic + 1,699 real biological) with "
                 "known ground truth.")
            ]
        elif module_name == "Visualization":  # Visualization (available for DAM data)
            methods = [
                ("Activity Profile", "Daily activity profile with mean ± SEM by condition")
            ]
        else:
            methods = []

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
        is_compare_all = "Compare All" in method_text
        is_compare_conditions = "Compare Conditions" in method_text
        is_nonlinear_compare = "Nonlinear Compare" in method_text
        is_circacompare_compare = "Compare Groups" in method_text
        # Pairwise uses 2 specific condition dropdowns; Compare All/Conditions/Nonlinear Compare/CircaCompare Compare Groups use ALL conditions
        is_pairwise_comparison = "Compare" in method_text and not is_compare_all and not is_compare_conditions and not is_nonlinear_compare and not is_circacompare_compare

        if is_compare_all or is_compare_conditions or is_nonlinear_compare or is_circacompare_compare:
            # For "Compare All" and "Compare Conditions", automatically use ALL available conditions
            selected_conds = [
                self._cond_list.item(i).text()
                for i in range(self._cond_list.count())
            ]
            compare_conditions = None  # Not needed for Compare All/Conditions

            if len(selected_conds) < 2:
                if is_compare_conditions:
                    comparison_name = "Compare Conditions"
                elif is_nonlinear_compare:
                    comparison_name = "Nonlinear Compare"
                else:
                    comparison_name = "Compare All"
                QMessageBox.warning(self, "Not Enough Conditions",
                                  f"{comparison_name} requires at least 2 conditions in your data.")
                return

        elif is_pairwise_comparison:
            # For pairwise comparison (2 conditions)
            cond1 = self._cond1_combo.currentText()
            cond2 = self._cond2_combo.currentText()
            if cond1 == cond2:
                QMessageBox.warning(self, "Same Conditions", "Please select different conditions to compare.")
                return
            selected_conds = [cond1]
            compare_conditions = (cond1, cond2)
        else:
            # For single analysis methods
            selected_conds = [
                self._cond_list.item(i).text()
                for i in range(self._cond_list.count())
                if self._cond_list.item(i).isSelected()
            ]
            compare_conditions = None

            if not selected_conds:
                QMessageBox.warning(self, "No Conditions", "Please select at least one condition.")
                return

        # Get selected clusters (for Rosbash data only)
        selected_clusters = None
        rosbash_compare_type = None
        if self._source_type == 'rosbash':
            selected_clusters = [
                self._cluster_list.item(i).text()
                for i in range(self._cluster_list.count())
                if self._cluster_list.item(i).isSelected()
            ]
            if not selected_clusters:
                QMessageBox.warning(self, "No Clusters", "Please select at least one cluster for Rosbash data.")
                return

            # Get comparison type if visible (for comparison methods)
            if self._rosbash_compare_type_frame.isVisible():
                rosbash_compare_type = self._rosbash_compare_type_combo.currentText()

        # Build configuration
        params = self._get_current_parameters()
        if rosbash_compare_type:
            params['rosbash_compare_type'] = rosbash_compare_type

        config = AnalysisConfig(
            method=self._get_current_method_enum(),
            variables=selected_vars,
            conditions=selected_conds,
            parameters=params,
            compare_conditions=compare_conditions,
            clusters=selected_clusters
        )
        
        # Start worker
        self._worker = AnalysisWorker(config, self._loader, self._source_type)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.period_detected.connect(self._on_period_detected)
        self._worker.components_detected.connect(self._on_components_detected)

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

    def _on_period_detected(self, period: float, p_value: float):
        """Handle period detection results."""
        # Update the detected period label with green checkmark
        self._detected_period_label.setText(
            f"  ✓ <b>Using auto-detected period: {period:.2f} hours</b> (p={p_value:.4f})\n"
            f"  This period will be shown in all result tables."
        )
        self._detected_period_label.setStyleSheet(
            "color: #2E7D32; font-style: italic; font-size: 10px; "
            "background-color: #E8F5E9; padding: 4px; border-radius: 3px;"
        )
        self._detected_period_label.setVisible(True)

        # Update the period spinbox to show the detected value (read-only)
        self._period_spin.setValue(period)

    def _on_components_detected(self, n_components: int, p_value: float):
        """Handle auto-selected components detection results."""
        # Update the detected components label with green checkmark
        self._detected_components_label.setText(
            f"  ✓ <b>Selected {n_components} component{'s' if n_components > 1 else ''}</b> (p={p_value:.4e})"
        )
        self._detected_components_label.setStyleSheet(
            "color: #2E7D32; font-style: italic; font-size: 10px; "
            "background-color: #E8F5E9; padding: 4px; border-radius: 3px;"
        )
        self._detected_components_label.setVisible(True)

        print(f"[DEBUG] _on_components_detected called: {n_components} components, p={p_value}")

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
        module_name = self._module_combo.currentText()
        method = self._method_combo.currentText()

        # Map (module_name, method_name) to enum
        mapping = {
            # CosinorPy - New Refactored Methods
            ("CosinorPy", "Periodogram Analysis"): AnalysisMethod.COSINORPY_PERIODOGRAM,
            ("CosinorPy", "Cosinor (Independent Data)"): AnalysisMethod.COSINORPY_INDEPENDENT,
            ("CosinorPy", "Cosinor (Dependent Data)"): AnalysisMethod.COSINORPY_DEPENDENT,
            ("CosinorPy", "Compare Conditions (Independent)"): AnalysisMethod.COSINORPY_COMPARE_INDEPENDENT,
            ("CosinorPy", "Compare Conditions (Dependent)"): AnalysisMethod.COSINORPY_COMPARE_DEPENDENT,
            ("CosinorPy", "Nonlinear (Independent Data)"): AnalysisMethod.COSINORPY_NONLINEAR_INDEPENDENT,
            ("CosinorPy", "Nonlinear (Dependent Data)"): AnalysisMethod.COSINORPY_NONLINEAR_DEPENDENT,
            ("CosinorPy", "Nonlinear Compare (Independent)"): AnalysisMethod.COSINORPY_NONLINEAR_COMPARE_INDEPENDENT,
            ("CosinorPy", "Nonlinear Compare (Dependent)"): AnalysisMethod.COSINORPY_NONLINEAR_COMPARE_DEPENDENT,
            # CircaCompare
            ("CircaCompare", "Single Fit"): AnalysisMethod.CIRCACOMPARE_SINGLE,
            ("CircaCompare", "Compare Groups"): AnalysisMethod.CIRCACOMPARE_COMPARE,
            # Rhythm Analysis
            ("Rhythm Analysis", "JTK Cycle"): AnalysisMethod.RHYTHM_JTK,
            ("Rhythm Analysis", "AR-JTK"): AnalysisMethod.RHYTHM_AR_JTK,
            ("Rhythm Analysis", "Cosine-Kendall"): AnalysisMethod.RHYTHM_COSINE_KENDALL,
            ("Rhythm Analysis", "Cosinor (OLS)"): AnalysisMethod.RHYTHM_COSINOR,
            ("Rhythm Analysis", "Harmonic Cosinor"): AnalysisMethod.RHYTHM_HARMONIC,
            ("Rhythm Analysis", "Fourier F24"): AnalysisMethod.RHYTHM_F24,
            ("Rhythm Analysis", "Lomb-Scargle"): AnalysisMethod.RHYTHM_LOMB,
            ("Rhythm Analysis", "Spectral Analysis (Periodogram)"): AnalysisMethod.RHYTHM_SPECTRAL,
            ("Rhythm Analysis", "Wavelet (CWT)"): AnalysisMethod.RHYTHM_CWT,
            ("Rhythm Analysis", "Linear Mixed Effects"): AnalysisMethod.RHYTHM_LME,
            # AI Consensus
            ("AI Consensus", "Consensus Rhythmicity Score"): AnalysisMethod.CONSENSUS_AI,
            # Visualization
            ("Visualization", "Activity Profile"): AnalysisMethod.VISUALIZATION_ACTIVITY_PROFILE,
        }

        return mapping.get((module_name, method), AnalysisMethod.COSINORPY_PERIODOGRAM)
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        # Build period from spinboxes
        period_min = self._period_min_spin.value()
        period_max = self._period_max_spin.value()
        period_step = self._period_step_spin.value()

        # If min == max, single period; otherwise generate range
        if period_min == period_max:
            period = period_min
        else:
            # Generate list of periods from min to max with step
            period = []
            current = period_min
            while current <= period_max:
                period.append(current)
                current += period_step

        # Get selected fixed effects for LME (if applicable)
        fixed_effects = [
            self._fixed_effects_list.item(i).text()
            for i in range(self._fixed_effects_list.count())
            if self._fixed_effects_list.item(i).isSelected()
        ]

        params = {
            'n_components': self._parse_components(self._components_edit.text()),
            'period': period,
            'period_range': (period_min, period_max),
            'period_step': period_step,
            'loss': self._loss_combo.currentText(),
            'f_scale': self._fscale_spin.value(),
            'max_iterations': self._max_iterations_spin.value(),
            'n_harmonics': self._harmonics_spin.value(),
            'n_permutations': self._permutations_spin.value(),
            # Nonlinear cosinor parameters
            'amplification': None if self._amplification_spin.value() == 0.0 else self._amplification_spin.value(),
            'lin_comp': None if self._lin_comp_spin.value() == 0.0 else self._lin_comp_spin.value(),
            'use_dependent_model': self._use_dependent_model_check.isChecked(),
            # Periodogram parameters
            'per_type': self._per_type_combo.currentText(),
            'max_per': self._max_per_spin.value(),
            'prominent': self._prominent_check.isChecked(),
            # CosinorPy new parameters
            'model_type': self._model_type_combo.currentText(),
            'criterium': self._criterium_combo.currentText(),
            'analysis_method': self._analysis_method_combo.currentText(),
            'bootstrap_size': self._bootstrap_size_spin.value(),
            'save_cosinorpy_plots': self._save_cosinorpy_plots_check.isChecked(),
            # Compare Conditions parameters
            'comparison_type': self._comparison_type_combo.currentText(),
            'comparison_method': self._comparison_method_combo.currentText(),
            'parameters_to_compare': self._get_selected_parameters_to_compare(),
            'include_lin_comp': self._include_lin_comp_check.isChecked(),
            # Period per condition (for Independent Models with 2 conditions)
            'period1': self._period_cond1_spin.value(),
            'period2': self._period_cond2_spin.value(),
            # Rhythm Analysis specific parameters
            'asymmetry': self._asymmetry_spin.value(),
            'ar_lag': self._ar_lag_spin.value(),
            'prewhiten': self._prewhiten_check.isChecked(),
            'resolution': self._resolution_spin.value(),
            'search_mode': self._search_mode_combo.currentText(),
            'target_period': self._target_period_spin.value(),
            'n_periods': self._n_periods_spin.value(),
            'alpha': float(self._alpha_combo.currentText()),
            'wavelet': 'cmor1.5-1.0' if 'Morlet' in self._wavelet_combo.currentText() else 'mexh',
            'sampling_interval': None if self._sampling_interval_spin.value() == 0.0 else self._sampling_interval_spin.value(),
            'detrending': self._detrending_check.isChecked(),
            # LME parameters
            'dependent_variable': self._dependent_var_combo.currentText(),
            'fixed_effects': fixed_effects,
            'random_effect': self._random_effect_combo.currentText()
        }
        return params

    def _get_selected_parameters_to_compare(self) -> list:
        """Get list of selected parameters to compare."""
        params = []
        if self._param_amplitude_check.isChecked():
            params.append('amplitude')
        if self._param_acrophase_check.isChecked():
            params.append('acrophase')
        if self._param_mesor_check.isChecked():
            params.append('mesor')
        return params if params else ['amplitude', 'acrophase', 'mesor']  # Default to all if none selected

    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def set_data(self, loader, source_type: str):
        """Set the data loader for analysis."""
        self._loader = loader
        self._source_type = source_type

        # Update variable list
        self._var_list.clear()

        if source_type == 'csv' or source_type == 'dam':
            # CSV/DAM: Show all available variables
            if source_type == 'dam':
                self._var_label.setText("Activity:")
            else:
                self._var_label.setText("Variables:")
            self._gene_search.setVisible(False)
            self._clock_genes_btn.setVisible(False)
            self._cluster_container.setVisible(False)

            variables = loader.get_variable_columns()
            for var in variables:
                item = QListWidgetItem(var)
                self._var_list.addItem(item)

            # Add/remove Visualization module based on source type
            self._update_module_combo_for_source(source_type)

            # Refresh method list based on whether subject column exists
            # This ensures dependent methods are only shown when appropriate
            current_module_index = self._module_combo.currentIndex()
            self._on_module_changed(current_module_index)

        elif source_type == 'rosbash':
            # Rosbash: Show all genes with search functionality
            self._var_label.setText("Genes:")
            self._gene_search.setVisible(True)
            self._clock_genes_btn.setVisible(True)
            self._cluster_container.setVisible(True)

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

            # Update module combo (remove Visualization for rosbash)
            self._update_module_combo_for_source(source_type)

        # Update condition list
        self._cond_list.clear()
        self._cond1_combo.clear()
        self._cond2_combo.clear()

        if source_type == 'csv' or source_type == 'dam':
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

        # Populate LME comboboxes with available columns (for CSV/DAM data)
        if source_type == 'csv' or source_type == 'dam':
            self._populate_lme_columns(loader)

        # Detect and display data type information
        self._detect_and_display_data_type()
    
    def _populate_lme_columns(self, loader):
        """Populate LME parameter comboboxes with available columns from the dataset."""
        try:
            # Get all columns from the dataset
            data = loader.get_data()
            all_columns = data.columns.tolist()

            # Filter out time and condition columns
            time_col = loader.get_time_column()
            condition_col = loader._condition_col if hasattr(loader, '_condition_col') and loader._condition_col else 'condition'

            # Get variable columns (numeric columns that are not time/condition)
            variable_cols = loader.get_variable_columns()

            # Get potential grouping columns (non-numeric or categorical)
            grouping_cols = [col for col in all_columns if col not in variable_cols and col not in [time_col, condition_col]]

            # Populate dependent variable combobox (typically numeric variables)
            self._dependent_var_combo.clear()
            for var in variable_cols:
                self._dependent_var_combo.addItem(var)

            # Populate fixed effects list (can include time, condition, and other grouping variables)
            self._fixed_effects_list.clear()
            potential_fixed_effects = [time_col, condition_col] + grouping_cols
            for effect in potential_fixed_effects:
                if effect:  # Skip None values
                    self._fixed_effects_list.addItem(effect)

            # Populate random effect combobox (typically grouping variables like Subject, Replicate, etc.)
            self._random_effect_combo.clear()
            for col in grouping_cols:
                self._random_effect_combo.addItem(col)

            # If there's a 'subject' column, set it as default for random effect
            if 'subject' in [c.lower() for c in grouping_cols]:
                idx = next(i for i, c in enumerate(grouping_cols) if c.lower() == 'subject')
                self._random_effect_combo.setCurrentIndex(idx)

        except Exception as e:
            print(f"[WARNING] Failed to populate LME columns: {e}")

    def _detect_and_display_data_type(self):
        """Detect data type and display appropriate recommendations."""
        if self._loader is None or self._source_type != 'csv':
            # self._data_info_frame.setVisible(False)
            return

        try:
            # Use the loader's dataset info which has already correctly detected columns
            info = self._loader.get_dataset_info()

            # Determine data type: subject column present → DEPENDENT, otherwise → INDEPENDENT
            if info.subject_column is not None:
                data_type = "DEPENDENT (Longitudinal)"
                description = "✓ Same subjects measured at multiple timepoints"
                recommendation = "Recommended methods: <b>Population Mean</b>"
                color_bg = "#E8F5E9"
                color_border = "#4CAF50"
            else:
                data_type = "INDEPENDENT"
                description = "✓ Different subjects at each timepoint"
                recommendation = "Recommended methods: <b>Single Cosinor</b>, <b>Multi-Component</b>"
                color_bg = "#E3F2FD"
                color_border = "#2196F3"

            # Update labels - DISABLED (panel removed)
            # self._data_type_label.setText(f"<b>Data Type:</b> {data_type}<br>{description}")
            # self._method_recommendation_label.setText(f"💡 {recommendation}")

            # Update frame colors
            # self._data_info_frame.setStyleSheet(f"""
            #     QFrame {{
            #         background-color: {color_bg};
            #         border: 1px solid {color_border};
            #         border-radius: 4px;
            #         padding: 8px;
            #     }}
            # """)

            # self._data_info_frame.setVisible(True)
            pass

        except Exception as e:
            print(f"[DEBUG] Error detecting data type: {e}")
            # self._data_info_frame.setVisible(False)

    def _update_module_combo_for_source(self, source_type: str):
        """Add or remove modules based on data source type."""
        # --- AI Consensus: hide for DAM data (model trained on gene expression) ---
        ai_consensus_idx = None
        for i in range(self._module_combo.count()):
            if self._module_combo.itemText(i) == "AI Consensus":
                ai_consensus_idx = i
                break

        if source_type == 'dam':
            if ai_consensus_idx is not None:
                self._module_combo.removeItem(ai_consensus_idx)
        else:
            if ai_consensus_idx is None:
                # Re-insert AI Consensus at index 3 (after Rhythm Analysis)
                insert_pos = min(3, self._module_combo.count())
                self._module_combo.insertItem(insert_pos, "AI Consensus")

        # --- Visualization: show only for DAM data ---
        has_visualization = False
        for i in range(self._module_combo.count()):
            if self._module_combo.itemText(i) == "Visualization":
                has_visualization = True
                break

        if source_type == 'dam':
            if not has_visualization:
                self._module_combo.addItem("Visualization")
        else:
            if has_visualization:
                for i in range(self._module_combo.count()):
                    if self._module_combo.itemText(i) == "Visualization":
                        self._module_combo.removeItem(i)
                        break

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
        self._cluster_container.setVisible(False)
        self._var_label.setText("Variables:")
        self._run_btn.setEnabled(False)
        if hasattr(self, '_all_genes'):
            self._all_genes = []
