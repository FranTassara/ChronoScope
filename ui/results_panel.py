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
from PySide6.QtGui import QAction, QColor

import pandas as pd
import numpy as np

# Matplotlib imports for embedding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# =============================================================================
# METHOD GROUPS FOR TAB VISIBILITY
# =============================================================================

# Methods that show Cosinor Fit, Polar Acrophase, Parameter Bars
COSINOR_METHODS = {
    'cosinorpy_independent',
    'cosinorpy_dependent',
    'cosinorpy_nonlinear_independent',
    'cosinorpy_nonlinear_dependent',
    'cosinorpy_single',
    'cosinorpy_multi',
    'cosinorpy_population',
    'cosinorpy_nonlinear',
    'jtk',
    'ar_jtk',
    'cosine_kendall',
    'cosinor_ols',
    'harmonic_cosinor',
}

# JTK-family methods (nonparametric Kendall-based)
JTK_METHODS = {'jtk', 'ar_jtk', 'cosine_kendall'}

# Methods that show Periodogram, Polar (if acrophase), Parameter Bars
PERIODOGRAM_METHODS = {
    'lomb_scargle',
    'spectral_analysis',
    'fourier_f24',
}

# Methods that show only text summary (Scalogram/CWT results)
SCALOGRAM_METHODS = {
    'cwt',
}

# AI Meta-Classifier methods
META_CLASSIFIER_METHODS = {
    'consensus_ai',
}

# Methods that show only text summary (currently none - LME now uses cosinor plot)
TEXT_ONLY_METHODS = set()  # Empty - all methods now have visual output

# CosinorPy periodogram - saves plots to directory, no visualization in tabs
COSINORPY_PERIODOGRAM_METHOD = 'cosinorpy_periodogram'

# Comparison methods
COMPARISON_METHODS = {
    'cosinorpy_compare',
    'cosinorpy_compare_pooled',
    'cosinorpy_compare_independent_models',
    'cosinorpy_compare_multi',
    'cosinorpy_compare_limorhyde',
    'cosinorpy_compare_independent',
    'cosinorpy_compare_dependent',
    'cosinorpy_compare_dependent_multi',
    'cosinorpy_nonlinear_compare_independent',
    'cosinorpy_nonlinear_compare_dependent',
    'cosinorpy_compare_all',
    'cosinorpy_compare_all_limo',
    'cosinorpy_limorhyde',
    'cosinorpy_compare_nonlinear',
}


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
        condition: str = "",
        n_components: int = 1
    ):
        """Plot raw data with cosinor fit overlay."""
        self.clear()
        ax = self.axes

        # Determine time range from raw data
        t_min = np.min(times) if len(times) > 0 else 0
        t_max = np.max(times) if len(times) > 0 else period

        # Plot raw data
        ax.scatter(times, values, alpha=0.6, label='Data', color='steelblue')

        # Plot fit curve covering the full data range
        n_points = max(200, int((t_max - t_min) / period * 200))
        t_fit = np.linspace(t_min, t_max, n_points)

        if n_components > 1:
            # Multi-component: reconstruct the full harmonic fit using OLS
            try:
                import statsmodels.api as sm
                from CosinorPy.cosinor import generate_independents
                X_raw = generate_independents(times, n_components=n_components, period=period)
                model = sm.OLS(values, X_raw).fit()
                X_dense = generate_independents(t_fit, n_components=n_components, period=period)
                y_fit = model.predict(X_dense)
            except Exception:
                # Fallback to single-component approximation
                y_fit = mesor + amplitude * np.cos(2 * np.pi * t_fit / period - acrophase_rad)
        else:
            # Single component: standard formula
            y_fit = mesor + amplitude * np.cos(2 * np.pi * t_fit / period - acrophase_rad)

        ax.plot(t_fit, y_fit, 'r-', linewidth=2, label='Cosinor Fit')

        # Add horizontal line at MESOR
        ax.axhline(y=mesor, color='gray', linestyle='--', alpha=0.5, label=f'MESOR={mesor:.2f}')

        # Labels
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Expression')
        ax.set_title(f'{title} - {condition}' if condition else title)
        ax.legend(loc='upper right')
        ax.set_xlim(t_min, t_max)

        self.fig.tight_layout()
        self.draw()
    
    def plot_comparison(
        self,
        result: Dict[str, Any],
        title: str = ""
    ):
        """Plot comparison between two conditions with raw data."""
        self.clear()
        ax = self.axes

        period = result.get('period', 24.0)

        # Get raw data for plotting (if available)
        times_g0 = result.get('times_g0')
        values_g0 = result.get('values_g0')
        times_g1 = result.get('times_g1')
        values_g1 = result.get('values_g1')

        cond1 = result.get('condition1', 'Group 0')
        cond2 = result.get('condition2', 'Group 1')

        # Determine time range from raw data
        t_min, t_max = 0, period
        if times_g0 is not None and values_g0 is not None:
            times_g0 = np.array(times_g0) if not isinstance(times_g0, np.ndarray) else times_g0
            values_g0 = np.array(values_g0) if not isinstance(values_g0, np.ndarray) else values_g0
            t_min = min(t_min, np.min(times_g0))
            t_max = max(t_max, np.max(times_g0))

        if times_g1 is not None and values_g1 is not None:
            times_g1 = np.array(times_g1) if not isinstance(times_g1, np.ndarray) else times_g1
            values_g1 = np.array(values_g1) if not isinstance(values_g1, np.ndarray) else values_g1
            t_min = min(t_min, np.min(times_g1))
            t_max = max(t_max, np.max(times_g1))

        # Plot raw data points for group 0 if available
        if times_g0 is not None and values_g0 is not None:
            ax.scatter(times_g0, values_g0, alpha=0.5, s=30, color='steelblue',
                      label=f'{cond1} data', zorder=1)

        # Plot raw data points for group 1 if available
        if times_g1 is not None and values_g1 is not None:
            ax.scatter(times_g1, values_g1, alpha=0.5, s=30, color='coral',
                      label=f'{cond2} data', zorder=1)

        # Generate time points for smooth curves covering the full data range
        n_points = max(200, int((t_max - t_min) / period * 200))
        t_fit = np.linspace(t_min, t_max, n_points)

        # Group 0 fit curve
        # Formula: y = mesor + amplitude * cos(2π * t / period - acrophase)
        mesor_g0 = result.get('mesor_g0', 0)
        amp_g0 = result.get('amplitude_g0', 0)
        acr_g0 = result.get('acrophase_g0', 0)
        y_g0 = mesor_g0 + amp_g0 * np.cos(2 * np.pi * t_fit / period - acr_g0)

        # Group 1 fit curve
        mesor_g1 = result.get('mesor_g1', 0)
        amp_g1 = result.get('amplitude_g1', 0)
        acr_g1 = result.get('acrophase_g1', 0)
        y_g1 = mesor_g1 + amp_g1 * np.cos(2 * np.pi * t_fit / period - acr_g1)

        ax.plot(t_fit, y_g0, '-', linewidth=2, label=f'{cond1} fit', color='steelblue', zorder=2)
        ax.plot(t_fit, y_g1, '-', linewidth=2, label=f'{cond2} fit', color='coral', zorder=2)

        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Expression')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(t_min, t_max)

        self.fig.tight_layout()
        self.draw()
    
    def plot_polar_acrophase(
        self,
        acrophases_hours: List[float],
        labels: List[str],
        period: float = 24.0,
        title: str = "Acrophase Distribution"
    ):
        """Plot acrophases on a polar plot with legend on the side."""
        self.fig.clear()

        # Create subplot with space for legend on the right
        ax = self.fig.add_subplot(111, projection='polar')

        # Filter out None values and convert hours to radians
        valid_data = [(h, l) for h, l in zip(acrophases_hours, labels) if h is not None]
        if not valid_data:
            ax.text(0.5, 0.5, 'No acrophase data available',
                   ha='center', va='center', transform=ax.transAxes)
            self.draw()
            return

        valid_hours, valid_labels = zip(*valid_data)
        thetas = [2 * np.pi * h / period for h in valid_hours]

        # Plot each point with label for legend
        colors = plt.cm.tab10(np.linspace(0, 1, len(thetas)))

        for theta, label, color in zip(thetas, valid_labels, colors):
            ax.scatter(theta, 1, s=100, c=[color], label=label, zorder=5)

        # Configure polar plot
        ax.set_theta_zero_location('N')  # 0 at top (ZT0)
        ax.set_theta_direction(-1)  # Clockwise

        # Set ticks for 24-hour clock
        ax.set_xticks(np.linspace(0, 2*np.pi, 9)[:-1])
        ax.set_xticklabels([f'ZT{int(h)}' for h in np.linspace(0, 24, 9)[:-1]])

        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.set_title(title, y=1.08)

        # Add legend outside the plot on the right side
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.15, 0.5),
            fontsize=8,
            framealpha=0.9
        )

        # Adjust layout to make room for the legend
        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.75)
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
        # Only use CI if it exists and is valid (not None or (0,0))
        errors = None
        has_any_ci = any(r.get(f'{parameter}_ci') is not None for r in results)

        if has_any_ci:
            lower_errors = []
            upper_errors = []

            for r, v in zip(results, values):
                ci = r.get(f'{parameter}_ci')
                if ci is not None and isinstance(ci, (tuple, list)) and len(ci) == 2:
                    # Valid CI exists
                    ci_low, ci_high = ci
                    # Calculate error bars: distance from value to CI bounds
                    # Ensure errors are non-negative (matplotlib requirement)
                    lower_errors.append(max(0, v - ci_low))
                    upper_errors.append(max(0, ci_high - v))
                else:
                    # No CI for this result - use 0 error
                    lower_errors.append(0)
                    upper_errors.append(0)

            errors = [lower_errors, upper_errors]
        
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

    def plot_activity_profile(
        self,
        profile_data: dict,
        conditions: list,
        title: str = "Activity Profile"
    ):
        """
        Plot daily activity profile as heatmap(s) - one per condition.

        Each heatmap shows:
        - X-axis: ZT time (0-24 hours)
        - Y-axis: Day number (1, 2, 3, ...)
        - Color: Mean activity (averaged across all subjects for that day/ZT)

        Args:
            profile_data: Dict mapping condition name to dict with:
                - 'days': list of day numbers
                - 'zt_times': list of ZT timepoints
                - 'activity_matrix': 2D list [day][zt] of mean activity values
            conditions: List of condition names
            title: Plot title
        """
        self.fig.clear()

        n_conditions = len(conditions)

        # Find global min/max for consistent color scale across conditions
        all_values = []
        for cond in conditions:
            if cond in profile_data:
                matrix = profile_data[cond]['activity_matrix']
                all_values.extend([v for row in matrix for v in row if v is not None])

        if all_values:
            vmin, vmax = min(all_values), max(all_values)
        else:
            vmin, vmax = 0, 1

        # Plot each condition
        for idx, cond in enumerate(conditions):
            # Create subplot with space for light/dark bar at top
            ax = self.fig.add_subplot(n_conditions, 1, idx + 1)

            if cond not in profile_data:
                ax.text(0.5, 0.5, f'No data for {cond}', ha='center', va='center')
                continue

            data = profile_data[cond]
            days = data['days']
            zt_times = data['zt_times']
            matrix = np.array(data['activity_matrix'])

            # Extend Y range to add space for light/dark bar above day 1
            min_day = min(days)
            max_day = max(days)
            y_min = min_day - 1.5  # Extra space above for light/dark bar

            # Create heatmap
            im = ax.imshow(
                matrix,
                aspect='auto',
                cmap='YlOrRd',  # Yellow-Orange-Red colormap
                vmin=vmin,
                vmax=vmax,
                extent=[min(zt_times), max(zt_times), max_day + 0.5, min_day - 0.5],
                interpolation='nearest'
            )

            # Add light/dark phase indicator ABOVE the data (in the extra space)
            bar_y_bottom = min_day - 1.3
            bar_y_top = min_day - 0.7
            ax.fill_between([0, 12], bar_y_bottom, bar_y_top, color='yellow', alpha=0.9)
            ax.fill_between([12, 24], bar_y_bottom, bar_y_top, color='gray', alpha=0.9)
            ax.text(6, (bar_y_bottom + bar_y_top) / 2, 'Light', ha='center', va='center', fontsize=8)
            ax.text(18, (bar_y_bottom + bar_y_top) / 2, 'Dark', ha='center', va='center', fontsize=8, color='white')

            # Configure axes
            ax.set_xlim(0, 24)
            ax.set_ylim(max_day + 0.5, y_min)
            ax.set_xlabel('ZT Time (hours)')
            ax.set_ylabel('Day')
            ax.set_title(f'{cond}')
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xticklabels(['ZT0', 'ZT6', 'ZT12', 'ZT18', 'ZT24'])

            # Set y-ticks to show day numbers (only actual days, not the bar area)
            if len(days) <= 20:
                ax.set_yticks(days)
            else:
                ax.set_yticks([d for d in days if d % 5 == 0 or d == 1])

            # Add colorbar
            cbar = self.fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Activity')

        # Add overall title
        self.fig.suptitle(title, fontsize=12, fontweight='bold')
        self.fig.tight_layout()
        self.draw()

    def plot_activity_profile_averaged(
        self,
        profile_data: dict,
        conditions: list,
        title: str = "Daily Activity Profile (Average)"
    ):
        """
        Plot daily activity profile averaged across all days with mean ± SEM.

        Args:
            profile_data: Dict mapping condition name to dict with:
                - 'days': list of day numbers
                - 'zt_times': list of ZT timepoints
                - 'activity_matrix': 2D list [day][zt] of mean activity values
            conditions: List of condition names
            title: Plot title
        """
        self.clear()
        ax = self.axes

        # Color palette for conditions
        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

        # Add light/dark phase background
        ax.axvspan(0, 12, alpha=0.15, color='yellow', label='_Light')
        ax.axvspan(12, 24, alpha=0.15, color='gray', label='_Dark')

        # Plot each condition
        for i, cond in enumerate(conditions):
            if cond not in profile_data:
                continue

            data = profile_data[cond]
            zt_times = np.array(data['zt_times'])
            matrix = np.array(data['activity_matrix'])  # [days][zt]

            # Calculate mean and SEM across days (axis=0)
            mean = np.nanmean(matrix, axis=0)
            sem = np.nanstd(matrix, axis=0) / np.sqrt(matrix.shape[0])

            color = colors[i]

            # Plot mean line
            ax.plot(zt_times, mean, '-', color=color, linewidth=2, label=cond)

            # Plot SEM shading
            ax.fill_between(zt_times, mean - sem, mean + sem, color=color, alpha=0.3)

        # Configure axes
        ax.set_xlabel('ZT Time (hours)')
        ax.set_ylabel('Activity (mean ± SEM)')
        ax.set_title(title)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['ZT0', 'ZT6', 'ZT12', 'ZT18', 'ZT24'])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    def plot_actogram(
        self,
        actogram_data: dict,
        conditions: list,
        title: str = "Actogram (Double-Plotted)",
        lighting_phases: dict = None,
    ):
        """
        Plot double-plotted actogram.

        Background per row reflects the lighting phase of that day:
          - LD rows : alternating light-yellow / gray (12:12 LD cycle)
          - DD rows : uniform gray
          - LL rows : uniform light yellow
          - No phases defined : standard 12:12 LD shading for all rows

        Args:
            actogram_data   : condition → {days, zt_times, matrix}
            conditions      : list of condition names
            lighting_phases : optional dict {phase: (start_day, end_day)}
            title           : plot title
        """
        self.fig.clear()

        # Build a day→phase lookup from lighting_phases
        def _day_phase(day):
            if not lighting_phases:
                return 'LD'
            for phase_name, phase_range in lighting_phases.items():
                if phase_range is None:
                    continue
                p_start, p_end = phase_range
                if p_start is None:
                    continue
                p_end_eff = p_end if p_end is not None else 99999
                if p_start <= day <= p_end_eff:
                    return phase_name
            return 'LD'  # default if day falls outside all defined ranges

        n_conditions = len(conditions)

        for idx, cond in enumerate(conditions):
            ax = self.fig.add_subplot(n_conditions, 1, idx + 1)

            if cond not in actogram_data:
                ax.text(0.5, 0.5, f'No data for {cond}', ha='center', va='center')
                continue

            data = actogram_data[cond]
            days = data['days']
            matrix = np.array(data['matrix'])

            if len(matrix) == 0:
                ax.text(0.5, 0.5, f'No data for {cond}', ha='center', va='center')
                continue

            n_days = len(days)
            n_bins = len(matrix[0]) if len(matrix) > 0 else 0
            bin_width = 48.0 / n_bins if n_bins > 0 else 1
            max_val = np.max(matrix) if np.max(matrix) > 0 else 1

            # --- Row backgrounds (drawn first, zorder=0) ---
            for i, day in enumerate(days):
                y_bot = n_days - i - 1
                y_top = n_days - i
                phase = _day_phase(day)

                if phase == 'DD':
                    # Uniform gray across the full 48h
                    ax.axhspan(y_bot, y_top, color='#bbbbbb', alpha=0.35, zorder=0)

                elif phase == 'LL':
                    # Uniform light yellow across the full 48h
                    ax.axhspan(y_bot, y_top, color='#fff9c4', alpha=0.55, zorder=0)

                else:
                    # LD: alternating light (yellow) / dark (gray) every 12h
                    # xmin/xmax are axis fractions (0–1 maps to 0–48 h)
                    ax.axhspan(y_bot, y_top, xmin=0/48,  xmax=12/48, color='#fff176', alpha=0.45, zorder=0)
                    ax.axhspan(y_bot, y_top, xmin=12/48, xmax=24/48, color='#9e9e9e', alpha=0.30, zorder=0)
                    ax.axhspan(y_bot, y_top, xmin=24/48, xmax=36/48, color='#fff176', alpha=0.45, zorder=0)
                    ax.axhspan(y_bot, y_top, xmin=36/48, xmax=48/48, color='#9e9e9e', alpha=0.30, zorder=0)

            # --- Activity bars ---
            for i, day in enumerate(days):
                row = matrix[i]
                for j, val in enumerate(row):
                    if val > 0:
                        x_pos = j * bin_width
                        height = 0.8 * (val / max_val)
                        ax.bar(x_pos, height, width=bin_width, bottom=n_days - i - 1,
                               color='black', align='edge', linewidth=0, zorder=2)

            # --- Axes ---
            ax.set_xlim(0, 48)
            ax.set_ylim(0, n_days)
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Day')
            ax.set_title(f'{cond}')
            ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
            ax.set_xticklabels(['0', '6', '12', '18', '24', '30', '36', '42', '48'])

            if n_days <= 20:
                ax.set_yticks([n_days - d for d in days])
                ax.set_yticklabels([str(d) for d in days])
            else:
                tick_days = [d for d in days if d % 5 == 0 or d == 1]
                ax.set_yticks([n_days - d for d in tick_days])
                ax.set_yticklabels([str(d) for d in tick_days])

        self.fig.suptitle(title, fontsize=12, fontweight='bold')
        self.fig.tight_layout()
        self.draw()

    def plot_total_activity(
        self,
        total_activity_data: dict,
        conditions: list,
        title: str = "Total Daily Activity"
    ):
        """
        Plot total activity per day for each condition.

        Args:
            total_activity_data: Dict mapping condition to dict with 'days', 'mean', 'sem'
            conditions: List of condition names
            title: Plot title
        """
        self.clear()
        ax = self.axes

        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

        for i, cond in enumerate(conditions):
            if cond not in total_activity_data:
                continue

            data = total_activity_data[cond]
            days = np.array(data['days'])
            mean = np.array(data['mean'])
            sem = np.array(data['sem'])

            color = colors[i]

            # Plot line with error band
            ax.plot(days, mean, '-o', color=color, linewidth=2, markersize=4, label=cond)
            ax.fill_between(days, mean - sem, mean + sem, color=color, alpha=0.3)

        ax.set_xlabel('Day')
        ax.set_ylabel('Total Activity (mean ± SEM)')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Set x-ticks to integers
        all_days = []
        for cond in conditions:
            if cond in total_activity_data:
                all_days.extend(total_activity_data[cond]['days'])
        if all_days:
            ax.set_xlim(min(all_days) - 0.5, max(all_days) + 0.5)

        self.fig.tight_layout()
        self.draw()

    def plot_activity_onset(
        self,
        onset_data: dict,
        conditions: list,
        title: str = "Activity Onset/Offset"
    ):
        """
        Plot activity onset and offset times for each day.

        Args:
            onset_data: Dict mapping condition to dict with 'days', 'onset_times', 'offset_times'
            conditions: List of condition names
            title: Plot title
        """
        self.clear()
        ax = self.axes

        colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

        # Add light/dark phase background
        ax.axhspan(0, 12, alpha=0.15, color='yellow')
        ax.axhspan(12, 24, alpha=0.15, color='gray')

        for i, cond in enumerate(conditions):
            if cond not in onset_data:
                continue

            data = onset_data[cond]
            days = data['days']
            onset_times = data['onset_times']
            offset_times = data.get('offset_times', [None] * len(days))

            color = colors[i]

            # Plot onset (filled circles)
            valid_onset = [(d, o) for d, o in zip(days, onset_times) if o is not None]
            if valid_onset:
                onset_days, onset_vals = zip(*valid_onset)
                ax.scatter(onset_days, onset_vals, color=color, s=50, marker='o',
                          label=f'{cond} onset', alpha=0.8)
                ax.plot(onset_days, onset_vals, '-', color=color, linewidth=1, alpha=0.4)

            # Plot offset (open triangles)
            valid_offset = [(d, o) for d, o in zip(days, offset_times) if o is not None]
            if valid_offset:
                offset_days, offset_vals = zip(*valid_offset)
                ax.scatter(offset_days, offset_vals, color=color, s=50, marker='^',
                          facecolors='none', edgecolors=color, linewidths=1.5,
                          label=f'{cond} offset', alpha=0.8)
                ax.plot(offset_days, offset_vals, '--', color=color, linewidth=1, alpha=0.4)

        ax.set_xlabel('Day')
        ax.set_ylabel('Time (ZT)')
        ax.set_title(title)
        ax.set_ylim(0, 24)
        ax.set_yticks([0, 6, 12, 18, 24])
        ax.set_yticklabels(['ZT0', 'ZT6', 'ZT12', 'ZT18', 'ZT24'])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    # ------------------------------------------------------------------
    # Chi-square periodogram (Sokolove-Bushell)
    # ------------------------------------------------------------------

    def plot_chi_square_periodogram(
        self,
        chi_sq_data: dict,
        conditions: list,
        title: str = "Chi-square Periodogram (Sokolove-Bushell)"
    ):
        """Plot Qp vs period for each condition, with significance line and τ annotation."""
        self.fig.clear()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        ax = self.fig.add_subplot(111)
        any_plotted = False

        for idx, cond in enumerate(conditions):
            if cond not in chi_sq_data:
                continue
            res = chi_sq_data[cond]
            periods = res.get('periods')
            qp = res.get('qp')
            tau = res.get('tau')
            tau_qp = res.get('tau_qp')
            significance = res.get('significance')

            if periods is None or len(periods) == 0:
                continue

            color = colors[idx % len(colors)]
            ax.plot(periods, qp, color=color, linewidth=1.5, label=cond)

            if tau is not None and tau_qp is not None:
                ax.axvline(tau, color=color, linestyle='--', linewidth=1, alpha=0.7)
                ax.annotate(
                    f'τ={tau:.1f}h',
                    xy=(tau, tau_qp),
                    xytext=(tau + 0.3, tau_qp),
                    color=color,
                    fontsize=8,
                    va='center',
                )

            # Draw significance threshold once (first condition that has it)
            if significance is not None and not any_plotted:
                ax.axhline(significance, color='red', linestyle=':', linewidth=1.0,
                           label='p=0.05 threshold')

            phase_used = res.get('phase_used', 'all')
            if phase_used != 'all' and idx == 0:
                ax.set_title(f"{title}\n(computed on {phase_used} phase)", fontsize=10)

            any_plotted = True

        if not any_plotted:
            ax.text(0.5, 0.5, 'No periodogram data available',
                    ha='center', va='center', transform=ax.transAxes)
        else:
            if ax.get_title() == '':
                ax.set_title(title, fontsize=10)
            ax.set_xlabel('Period (h)')
            ax.set_ylabel('Qp statistic')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.draw()

    # ------------------------------------------------------------------
    # Behavioral metrics: IS, IV, α, ρ
    # ------------------------------------------------------------------

    def plot_behavioral_metrics(
        self,
        is_iv_data: dict,
        alpha_rho_data: dict,
        conditions: list,
        title: str = "Behavioral Metrics"
    ):
        """Bar chart of IS, IV, α (mean), ρ (mean) per condition plus onset/offset SD."""
        self.fig.clear()

        # Collect values
        is_vals, iv_vals, alpha_vals, rho_vals = [], [], [], []
        onset_sd_vals, offset_sd_vals = [], []
        valid_conds = []

        for cond in conditions:
            iv_entry = is_iv_data.get(cond, {})
            ar_entry = alpha_rho_data.get(cond, {})
            IS = iv_entry.get('IS')
            IV = iv_entry.get('IV')
            alpha_m = ar_entry.get('alpha_mean')
            rho_m = ar_entry.get('rho_mean')
            if any(v is not None for v in [IS, IV, alpha_m, rho_m]):
                valid_conds.append(cond)
                is_vals.append(IS if IS is not None else float('nan'))
                iv_vals.append(IV if IV is not None else float('nan'))
                alpha_vals.append(alpha_m if alpha_m is not None else float('nan'))
                rho_vals.append(rho_m if rho_m is not None else float('nan'))
                onset_sd_vals.append(ar_entry.get('onset_sd', 0.0))
                offset_sd_vals.append(ar_entry.get('offset_sd', 0.0))

        if not valid_conds:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No behavioral metrics available',
                    ha='center', va='center', transform=ax.transAxes)
            self.fig.tight_layout()
            self.draw()
            return

        import numpy as np
        n_conds = len(valid_conds)
        x = np.arange(n_conds)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        bar_colors = [colors[i % len(colors)] for i in range(n_conds)]

        # 2×2 layout: IS | IV | α (with SD) | ρ (with SD)
        axes = [self.fig.add_subplot(2, 3, i + 1) for i in range(6)]
        metrics = [
            (axes[0], is_vals, 'IS', 'Interdaily Stability', None, (0, 1)),
            (axes[1], iv_vals, 'IV', 'Intradaily Variability', None, None),
            (axes[2], alpha_vals, 'α (h)', 'Active phase duration', None, (0, 24)),
            (axes[3], rho_vals, 'ρ (h)', 'Rest phase duration', None, (0, 24)),
            (axes[4], onset_sd_vals, 'Onset SD (h)', 'Onset variability', None, None),
            (axes[5], offset_sd_vals, 'Offset SD (h)', 'Offset variability', None, None),
        ]

        for ax, vals, ylabel, subplot_title, errs, ylim in metrics:
            bars = ax.bar(x, vals, color=bar_colors, width=0.6)
            ax.set_title(subplot_title, fontsize=9, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(valid_conds, fontsize=7, rotation=15, ha='right')
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, axis='y', alpha=0.3)
            # Annotate values
            for bar, val in zip(bars, vals):
                import math
                if val is not None and not math.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        self.fig.suptitle(title, fontsize=11, fontweight='bold')
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
        group = QGroupBox("Visualization & Analysis")
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
        
        # Parameter selector for bar chart (hidden when tab is used as Actogram)
        self._bar_param_container = QWidget()
        bar_ctrl_layout = QHBoxLayout(self._bar_param_container)
        bar_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        bar_ctrl_layout.addWidget(QLabel("Parameter:"))
        self._bar_param_combo = QComboBox()
        self._bar_param_combo.addItems(['amplitude', 'mesor', 'acrophase_hours', 'p_value'])
        self._bar_param_combo.currentTextChanged.connect(self._update_bar_plot)
        bar_ctrl_layout.addWidget(self._bar_param_combo)
        bar_ctrl_layout.addStretch()
        bar_layout.addWidget(self._bar_param_container)
        
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

        # Onset plot (for Activity Profile - Activity Onset times)
        self._onset_canvas = PlotCanvas(self, width=6, height=4)
        onset_widget = QWidget()
        onset_layout = QVBoxLayout(onset_widget)
        onset_toolbar = NavigationToolbar(self._onset_canvas, self)
        onset_layout.addWidget(onset_toolbar)
        onset_layout.addWidget(self._onset_canvas)
        self._viz_tabs.addTab(onset_widget, "Activity Onset")

        # Chi-square periodogram (Sokolove-Bushell) - Visualization module only
        self._chisq_canvas = PlotCanvas(self, width=6, height=4)
        chisq_widget = QWidget()
        chisq_layout = QVBoxLayout(chisq_widget)
        chisq_toolbar = NavigationToolbar(self._chisq_canvas, self)
        chisq_layout.addWidget(chisq_toolbar)
        chisq_layout.addWidget(self._chisq_canvas)
        self._viz_tabs.addTab(chisq_widget, "Chi-sq Periodogram")

        # Behavioral metrics: IS, IV, α, ρ - Visualization module only
        self._metrics_canvas = PlotCanvas(self, width=6, height=4)
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_toolbar = NavigationToolbar(self._metrics_canvas, self)
        metrics_layout.addWidget(metrics_toolbar)
        metrics_layout.addWidget(self._metrics_canvas)
        self._viz_tabs.addTab(metrics_widget, "Behavioral Metrics")

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

        self._results_label.setText(self._build_results_summary(results))
    
    def add_results(self, results: List[Dict[str, Any]]):
        """Add new results to existing."""
        self._results.extend(results)
        self._update_table()
        self._update_plots()

        self._results_label.setText(self._build_results_summary(self._results))
    
    def _build_results_summary(self, results: List[Dict[str, Any]]) -> str:
        """Build summary text for results label."""
        if not results:
            return "No results"
        is_consensus = any(r.get('method') == 'consensus_ai' for r in results)
        if is_consensus:
            n_rhythmic = sum(1 for r in results if r.get('r_squared') is not None and r.get('r_squared') >= 0.7)
            return f"{len(results)} results ({n_rhythmic} rhythmic)"
        n_sig = sum(1 for r in results if r.get('p_value') is not None and r.get('p_value') < 0.05)
        return f"{len(results)} results ({n_sig} significant)"

    def clear_results(self):
        """Clear all results."""
        self._results = []
        self._results_table.setRowCount(0)
        self._fit_canvas.clear()
        self._polar_canvas.clear()
        self._bar_canvas.clear()
        self._period_canvas.clear()
        self._onset_canvas.clear()
        self._chisq_canvas.clear()
        self._metrics_canvas.clear()
        self._results_label.setText("No results")
        self._current_result_index = -1
    
    def _update_table(self):
        """Update results table."""
        if not self._results:
            self._results_table.setRowCount(0)
            return

        # Check if we have comparison results
        is_comparison = 'condition1' in self._results[0] and 'condition2' in self._results[0]

        # Check if we have nonlinear cosinor results
        has_nonlinear = any(r.get('amplification') is not None or r.get('lin_comp') is not None for r in self._results)

        # Check if we have nonlinear comparison results
        has_nonlinear_comparison = any(r.get('amplification_diff') is not None or r.get('lin_comp_diff') is not None for r in self._results)

        # Check if we have periodogram results (Spectral Analysis or Lomb-Scargle)
        has_lomb_scargle = any(r.get('method') == 'lomb_scargle' for r in self._results)
        has_spectral = any(r.get('method') == 'spectral_analysis' for r in self._results)
        has_periodogram = has_lomb_scargle or has_spectral  # kept for plot routing

        # Check if we have Fourier F24 results (separate from generic periodogram)
        has_f24 = any(r.get('method') == 'fourier_f24' for r in self._results)

        # Check if we have CosinorPy periodogram (just shows message)
        has_cosinorpy_periodogram = any(r.get('method') == 'cosinorpy_periodogram' for r in self._results)

        # Check if we have CircaCompare Single Fit results
        has_circacompare_single = any(r.get('method') == 'circacompare_single' for r in self._results)

        # Check if we have LME results
        has_lme = any(r.get('method') == 'lme' for r in self._results)

        # Check if we have CWT (Wavelet) results
        has_cwt = any(r.get('method') == 'cwt' for r in self._results)

        # Check if we have Consensus AI results
        has_consensus_ai = any(r.get('method') == 'consensus_ai' for r in self._results)

        # Check if we have JTK-family results
        has_jtk = any(r.get('method') in JTK_METHODS for r in self._results)

        # Check if we have RhythmCount results
        RC_METHODS = {'rhythmcount_single', 'rhythmcount_all_models', 'rhythmcount_best_model',
                      'rhythmcount_parameter_cis', 'rhythmcount_compare_groups'}
        has_rhythmcount_cis = any(r.get('method') == 'rhythmcount_parameter_cis' for r in self._results)
        has_rhythmcount = any(r.get('method') in RC_METHODS for r in self._results)

        # Determine columns based on result type
        if is_comparison:
            has_circacompare_compare = any(r.get('method') == 'circacompare_compare' for r in self._results)

            if has_circacompare_compare:
                columns = ['variable', 'condition1', 'condition2', 'method', 'period',
                           'mesor_g0', 'mesor_g1', 'mesor_diff', 'mesor_diff_ci', 'sig_mesor',
                           'amplitude_g0', 'amplitude_g1', 'amplitude_diff', 'amplitude_diff_ci', 'sig_amplitude',
                           'acrophase_g0_hours', 'acrophase_g1_hours', 'acrophase_diff_hours', 'acrophase_diff_ci', 'sig_acrophase']
                headers = ['Variable', 'Cond1', 'Cond2', 'Method', 'Period (h)',
                           'MESOR-1', 'MESOR-2', 'MESOR-Diff', 'CI (MESOR-Diff)', 'sig (MESOR)',
                           'Amp-1', 'Amp-2', 'Amp-Diff', 'CI (Amp-Diff)', 'sig (Amp)',
                           'Acro-1 (h)', 'Acro-2 (h)', 'Acro-Diff (h)', 'CI (Acro-Diff, rad)', 'sig (Acro)']
            else:
                # CosinorPy / generic comparison table
                columns = ['variable', 'condition1', 'condition2', 'method', 'n_components', 'period',
                          'p1', 'q1', 'p2', 'q2',  # Population-specific p/q values (for dependent multi-component)
                          'amplitude_g0', 'amplitude_g1', 'amplitude_diff', 'p_amplitude', 'q_amplitude', 'amplitude_diff_ci',
                          'acrophase_g0', 'acrophase_g1', 'acrophase_diff', 'p_acrophase', 'q_acrophase', 'acrophase_diff_ci',
                          'mesor_g0', 'mesor_g1', 'mesor_diff', 'p_mesor', 'q_mesor', 'mesor_diff_ci',
                          'me', 'resid_se', 'aic', 'bic']
                headers = ['Variable', 'Cond1', 'Cond2', 'Method', 'Components', 'Period (h)',
                          'p-Cond1', 'q-Cond1', 'p-Cond2', 'q-Cond2',
                          'Amp-1', 'Amp-2', 'Amp-Diff', 'p-Amp', 'q-Amp', 'CI-Amp',
                          'Acro-1', 'Acro-2', 'Acro-Diff', 'p-Acro', 'q-Acro', 'CI-Acro',
                          'MESOR-1', 'MESOR-2', 'MESOR-Diff', 'p-MESOR', 'q-MESOR', 'CI-MESOR',
                          'ME', 'Resid-SE', 'AIC', 'BIC']

                # Add nonlinear comparison columns if present
                if has_nonlinear_comparison:
                    columns.extend(['amplification_g0', 'amplification_g1', 'amplification_diff', 'p_amplification', 'q_amplification', 'amplification_diff_ci',
                                   'lin_comp_g0', 'lin_comp_g1', 'lin_comp_diff', 'p_lin_comp', 'q_lin_comp', 'lin_comp_diff_ci'])
                    headers.extend(['Amplif-1', 'Amplif-2', 'Amplif-Diff', 'p-Amplif', 'q-Amplif', 'CI-Amplif',
                                   'LinComp-1', 'LinComp-2', 'LinComp-Diff', 'p-LinComp', 'q-LinComp', 'CI-LinComp'])
        else:
            # Check if this is an Activity Profile visualization
            is_activity_profile = any(r.get('type') == 'activity_profile' for r in self._results)
            if is_activity_profile:
                columns = [
                    'condition', 'n_subjects', 'n_days',
                    'tau_h', 'tau_significant', 'phase_used',
                    'IS', 'IV',
                    'alpha_mean_h', 'alpha_sd_h',
                    'rho_mean_h', 'rho_sd_h',
                    'onset_sd_h', 'offset_sd_h',
                ]
                headers = [
                    'Condition', 'n', 'Days',
                    'τ (h)', 'τ sig. (p<0.05)', 'Phase (τ)',
                    'IS', 'IV',
                    'α mean (h)', 'α SD (h)',
                    'ρ mean (h)', 'ρ SD (h)',
                    'Onset SD (h)', 'Offset SD (h)',
                ]
            # For CosinorPy periodogram, show just message
            elif has_cosinorpy_periodogram:
                columns = ['variable', 'condition', 'method', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Status']
            # For Fourier F24 (effect size, no p-value)
            elif has_f24:
                columns = ['variable', 'condition', 'method', 'period', 'dominant_period', 'power', 'dominant_power', 'target_power', 'correlation_r', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Target Period (h)', 'Dominant Period (h)', 'F24 Score', 'Dominant Power', 'Target Power', 'Correlation r', 'Notes']
            # For Spectral Analysis (Periodogram)
            elif has_spectral:
                columns = ['variable', 'condition', 'method', 'dominant_period', 'power', 'threshold', 'significant_peaks', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Dominant Period (h)', 'Max Power', 'Threshold (Refinetti)', 'Significant Peaks (h)', 'Notes']
            # For Lomb-Scargle
            elif has_lomb_scargle:
                columns = ['variable', 'condition', 'method', 'dominant_period', 'power', 'p_value', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Dominant Period (h)', 'Dominant Power', 'FAP', 'Notes']
            # For CircaCompare Single Fit
            elif has_circacompare_single:
                columns = ['variable', 'condition', 'method', 'period',
                           'mesor', 'se_mesor', 'mesor_ci',
                           'amplitude', 'se_amplitude', 'amplitude_ci',
                           'acrophase_hours', 'se_acrophase', 'acrophase_ci']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)',
                           'MESOR', 'SE (MESOR)', 'CI (MESOR)',
                           'Amplitude', 'SE (Amplitude)', 'CI (Amplitude)',
                           'Acrophase (h)', 'SE (Acrophase)', 'CI (Acrophase, rad)']
            # For LME results
            elif has_lme:
                columns = ['variable', 'condition', 'method', 'period', 'mesor', 'amplitude',
                           'acrophase_hours', 'p_value', 'r_squared', 'aic', 'bic',
                           'random_effect_var', 'residual_var', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)', 'MESOR', 'Amplitude',
                           'Acrophase (h)', 'p (LRT)', 'R² (marginal)', 'AIC', 'BIC',
                           'Var (random)', 'Var (residual)', 'Notes']
                if any(r.get('best_model') is not None for r in self._results):
                    columns.append('best_model')
                    headers.append('Best Period')
            # For CWT (Wavelet) results
            elif has_cwt:
                columns = ['variable', 'condition', 'method', 'period', 'power', 'period_variation', 'amplitude_modulations']
                headers = ['Variable', 'Condition', 'Method', 'Dominant Period (h)', 'Mean Power', 'Period Variation (h)', 'Amp. Modulations']
            # For Consensus AI results
            elif has_consensus_ai:
                columns = ['variable', 'condition', 'method', 'ai_probability', 'ai_classification']
                headers = ['Variable', 'Condition', 'Method', 'Probability', 'Classification']
            # For JTK-family methods (JTK, AR-JTK, Cosine-Kendall)
            elif has_jtk:
                columns = ['variable', 'condition', 'method', 'period', 'amplitude',
                           'acrophase_hours', 'tau', 'lag', 'asymmetry',
                           'raw_p_value', 'p_value', 'bonf_p_value', 'n_tests', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)', 'Amplitude',
                           'Acrophase (h)', 'Tau', 'Lag (h)', 'Asymmetry',
                           'p (raw)', 'p (BH adj)', 'p (Bonf adj)', 'N tests', 'Notes']
                if any(r.get('best_model') is not None for r in self._results):
                    columns.append('best_model')
                    headers.append('Best Fit')
            # For RhythmCount: Parameter Confidence Intervals
            elif has_rhythmcount_cis:
                columns = ['variable', 'condition', 'method', 'period', 'n_components',
                           'amplitude_ci', 'mesor_ci', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)', 'Components',
                           'CI (Amplitude)', 'CI (MESOR)', 'Notes']
            # For RhythmCount: Fit Single / All Models / Best Model
            elif has_rhythmcount:
                columns = ['variable', 'condition', 'method', 'period', 'n_components',
                           'mesor', 'amplitude', 'p_value', 'aic', 'bic',
                           'rss', 'log_likelihood', 'r_squared', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)', 'Components',
                           'MESOR', 'Amplitude', 'p (LRT)', 'AIC', 'BIC',
                           'RSS', 'Log-Likelihood', 'McFadden R²', 'Notes']
            # For Harmonic Cosinor
            elif any(r.get('method') == 'harmonic_cosinor' for r in self._results):
                columns = ['variable', 'condition', 'method', 'period', 'n_components',
                           'amplitude', 'acrophase_hours',
                           'trough_times', 'peak_times',
                           'p_value', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)', 'Harmonics',
                           'Primary Amplitude (H1)', 'Primary Acrophase H1 (h)',
                           'All Amplitudes', 'All Acrophases (h)',
                           'p (F-test)', 'Notes']
            # For Cosinor OLS
            elif any(r.get('method') == 'cosinor_ols' for r in self._results):
                columns = ['variable', 'condition', 'method', 'period', 'mesor', 'amplitude',
                           'acrophase_hours', 'acrophase',
                           'p_value', 'bonf_p_value',
                           'p_amplitude', 'p_acrophase',
                           'amplitude_ci', 'acrophase_ci', 'message']
                headers = ['Variable', 'Condition', 'Method', 'Period (h)', 'MESOR', 'Amplitude',
                           'Acrophase (h)', 'Acrophase (rad)',
                           'p (raw)', 'p (Bonf adj)',
                           'p(Amplitude)', 'p(Acrophase)',
                           'CI(Amplitude)', 'CI(Acrophase)', 'Notes']
            else:
                # Basic identification
                columns = ['variable', 'condition', 'method', 'n_components']
                headers = ['Variable', 'Condition', 'Method', 'Components']

                # Model parameters
                columns.extend(['mesor', 'amplitude', 'acrophase', 'acrophase_hours', 'period'])
                headers.extend(['MESOR', 'Amplitude', 'Acrophase (rad)', 'Acrophase (h)', 'Period (h)'])

                # Basic statistics (from fit_group)
                columns.extend(['p_value', 'q_value', 'p_reject', 'q_reject'])
                headers.extend(['p', 'q', 'p_reject', 'q_reject'])

                # Fit quality metrics
                columns.extend(['rss', 'r_squared', 'r_squared_adj', 'log_likelihood', 'aic', 'bic', 'me', 'resid_se'])
                headers.extend(['RSS', 'R²', 'R²_adj', 'Log-Likelihood', 'AIC', 'BIC', 'ME', 'Resid-SE'])

                # Confidence intervals (from analyse_best_models)
                columns.extend(['amplitude_ci', 'acrophase_ci', 'mesor_ci'])
                headers.extend(['CI(Amplitude)', 'CI(Acrophase)', 'CI(MESOR)'])

                # p-values for parameters (from analyse_best_models)
                columns.extend(['p_amplitude', 'p_acrophase', 'p_mesor'])
                headers.extend(['p(Amplitude)', 'p(Acrophase)', 'p(MESOR)'])

                # q-values for parameters (from analyse_best_models)
                columns.extend(['q_amplitude', 'q_acrophase', 'q_mesor'])
                headers.extend(['q(Amplitude)', 'q(Acrophase)', 'q(MESOR)'])

                # Other
                columns.extend(['peak_times', 'trough_times'])
                headers.extend(['Peak Times (h)', 'Trough Times (h)'])

                # Add nonlinear columns if present
                if has_nonlinear:
                    columns.extend([
                        'amplification', 'p_amplification', 'q_amplification', 'amplification_ci',
                        'lin_comp', 'p_lin_comp', 'q_lin_comp', 'lin_comp_ci'
                    ])
                    headers.extend([
                        'Amplification (C)', 'p(Amplification)', 'q(Amplification)', 'CI(Amplification)',
                        'Lin. Trend (D)', 'p(Lin. Trend)', 'q(Lin. Trend)', 'CI(Lin. Trend)'
                    ])

                columns.append('significant')
                headers.append('Significant')

                # Add best_model column if any result has it
                if any(r.get('best_model') is not None for r in self._results):
                    columns.append('best_model')
                    headers.append('Best Model')

        self._results_table.setColumnCount(len(columns))
        self._results_table.setHorizontalHeaderLabels(headers)

        # ---------------------------------------------------------------
        # Activity Profile: build one flat row per condition
        # ---------------------------------------------------------------
        if is_activity_profile:
            result = self._results[0]
            conditions_list = result.get('conditions', [])
            n_subjects_map = result.get('n_subjects', {})
            n_days_total = result.get('n_days', 0)
            chi_sq_data = result.get('chi_sq_data', {})
            is_iv_data = result.get('is_iv_data', {})
            alpha_rho_data = result.get('alpha_rho_data', {})

            self._results_table.setRowCount(len(conditions_list))
            for i, cond in enumerate(conditions_list):
                chi = chi_sq_data.get(cond, {})
                iiv = is_iv_data.get(cond, {})
                ar = alpha_rho_data.get(cond, {})

                # τ significance: compare tau_qp against threshold
                tau_qp = chi.get('tau_qp')
                sig_thresh = chi.get('significance')
                if tau_qp is not None and sig_thresh is not None:
                    tau_sig = 'Yes' if tau_qp > sig_thresh else 'No'
                else:
                    tau_sig = 'N/A'

                row_data = {
                    'condition': cond,
                    'n_subjects': n_subjects_map.get(cond, 'N/A'),
                    'n_days': n_days_total,
                    'tau_h': chi.get('tau'),
                    'tau_significant': tau_sig,
                    'phase_used': chi.get('phase_used', 'N/A'),
                    'IS': iiv.get('IS'),
                    'IV': iiv.get('IV'),
                    'alpha_mean_h': ar.get('alpha_mean'),
                    'alpha_sd_h': ar.get('alpha_sd'),
                    'rho_mean_h': ar.get('rho_mean'),
                    'rho_sd_h': ar.get('rho_sd'),
                    'onset_sd_h': ar.get('onset_sd'),
                    'offset_sd_h': ar.get('offset_sd'),
                }

                for j, col in enumerate(columns):
                    val = row_data.get(col, 'N/A')
                    if isinstance(val, float):
                        import math as _math
                        cell = 'N/A' if _math.isnan(val) else f'{val:.3f}'
                    elif val is None:
                        cell = 'N/A'
                    else:
                        cell = str(val)

                    item = QTableWidgetItem(cell)
                    # Highlight τ significance
                    if col == 'tau_significant':
                        if cell == 'Yes':
                            item.setBackground(QColor(144, 238, 144))  # light green
                        elif cell == 'No':
                            item.setBackground(QColor(255, 180, 180))  # light red
                    self._results_table.setItem(i, j, item)

            # Hide all-N/A columns
            for j in range(len(columns)):
                all_na = all(
                    (self._results_table.item(i, j) is None or
                     self._results_table.item(i, j).text() in ('N/A', '-', ''))
                    for i in range(len(conditions_list))
                )
                self._results_table.setColumnHidden(j, all_na)
            return

        # Apply filter
        filtered = self._get_filtered_results()
        self._results_table.setRowCount(len(filtered))

        for i, result in enumerate(filtered):
            for j, col in enumerate(columns):
                if col == 'ai_probability':
                    # Consensus AI: probability is stored in r_squared
                    prob = result.get('r_squared')
                    if prob is not None and not (isinstance(prob, float) and math.isnan(prob)):
                        value = f'{prob:.3f}'
                    else:
                        value = 'N/A'
                elif col == 'ai_classification':
                    # Consensus AI: extract classification from message JSON
                    msg = result.get('message', '')
                    value = 'N/A'
                    if msg:
                        try:
                            import json as _json_mod
                            details = _json_mod.loads(msg)
                            value = details.get('classification', 'N/A')
                        except (ValueError, TypeError):
                            pass
                elif col == 'significant':
                    p_val = result.get('p_value')
                    value = 'Yes' if (p_val is not None and p_val < 0.05) else ('No' if p_val is not None else 'N/A')
                elif col == 'best_model':
                    # Handle best_model column
                    value = result.get('best_model', '')
                    if value is None:
                        value = ''
                else:
                    value = result.get(col, '-')
                    if isinstance(value, float):
                        # Check for NaN first
                        if math.isnan(value):
                            value = 'N/A'
                        elif col.startswith('p_') or col.startswith('q_') or col == 'p_value':
                            value = f'{value:.2e}' if value < 0.001 else f'{value:.4f}'
                        else:
                            value = f'{value:.3f}'
                    elif isinstance(value, tuple) and len(value) == 2:
                        # Format confidence intervals as [lower, upper]
                        if value[0] is not None and value[1] is not None:
                            value = f'[{value[0]:.3f}, {value[1]:.3f}]'
                        else:
                            value = 'N/A'
                    elif isinstance(value, list):
                        # Format lists as comma-separated values
                        if value:
                            # Check if numeric list or string list
                            if all(isinstance(v, (int, float)) for v in value):
                                value = ', '.join([f'{v:.2f}' for v in value])
                            else:
                                value = ', '.join([str(v) for v in value])
                        else:
                            value = 'N/A'
                    elif value is None:
                        value = 'N/A'

                item = QTableWidgetItem(str(value))

                # Color code CircaCompare CI-based significance
                if col.startswith('sig_'):
                    if value == 'Yes':
                        item.setBackground(Qt.green)

                # Color code p-values and q-values
                elif col.startswith('p_') or col.startswith('q_') or col == 'p_value' or col == 'significant':
                    # For comparison, check the specific p-value or q-value column
                    if col.startswith('p_') or col.startswith('q_'):
                        p_val = result.get(col)
                    else:
                        p_val = result.get('p_value')

                    if p_val is not None and p_val < 0.05:
                        item.setBackground(Qt.green)

                # Highlight best model
                elif col == 'best_model':
                    best_model_value = result.get('best_model', '')
                    if best_model_value and 'Yes' in best_model_value:
                        item.setBackground(Qt.yellow)
                        item.setForeground(Qt.black)

                # Color code AI classification
                elif col == 'ai_classification' or col == 'ai_probability':
                    text = str(value)
                    prob = result.get('r_squared')
                    if prob is not None and not (isinstance(prob, float) and math.isnan(prob)):
                        if prob >= 0.7:
                            item.setBackground(QColor(144, 238, 144))  # light green
                        elif prob >= 0.3:
                            item.setBackground(QColor(255, 255, 150))  # light yellow
                            item.setForeground(Qt.black)
                        else:
                            item.setBackground(QColor(255, 180, 180))  # light red

                self._results_table.setItem(i, j, item)

        # Hide columns that only contain N/A values
        for j in range(len(columns)):
            all_na = True
            for i in range(len(filtered)):
                item = self._results_table.item(i, j)
                if item and item.text() not in ['N/A', '-', '']:
                    all_na = False
                    break

            # Hide column if all values are N/A
            if all_na:
                self._results_table.setColumnHidden(j, True)
            else:
                self._results_table.setColumnHidden(j, False)

    def _get_filtered_results(self) -> List[Dict]:
        """Get filtered results based on current filter."""
        filter_idx = self._significance_filter.currentIndex()

        if filter_idx == 0:  # All
            return self._results

        is_consensus = any(r.get('method') == 'consensus_ai' for r in self._results)
        if is_consensus:
            # For consensus AI, filter by probability >= 0.7 (Rhythmic)
            if filter_idx == 1:  # Significant / Rhythmic
                return [r for r in self._results if r.get('r_squared') is not None and r.get('r_squared') >= 0.7]
            else:  # Non-significant / Not Rhythmic
                return [r for r in self._results if r.get('r_squared') is not None and r.get('r_squared') < 0.7]
        else:
            if filter_idx == 1:  # Significant
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
    
    def _configure_tabs_for_method(self, method: str, is_comparison: bool = False):
        """Configure tab visibility and names based on the analysis method.

        Args:
            method: The analysis method identifier
            is_comparison: Whether this is a comparison result
        """
        # Default: hide all tabs first; restore the parameter selector
        for i in range(self._viz_tabs.count()):
            self._viz_tabs.setTabVisible(i, False)
        self._bar_param_container.setVisible(True)

        # Activity Profile - all 5 specialized tabs
        # (handled separately in _update_plots before this is called)

        # CosinorPy Periodogram - no tabs (plots saved to directory)
        if method == COSINORPY_PERIODOGRAM_METHOD:
            # All tabs hidden, show message in first tab
            self._viz_tabs.setTabVisible(0, True)
            self._viz_tabs.setTabText(0, "Info")
            return

        # Cosinor-based methods (including comparisons)
        if method in COSINOR_METHODS or method in COMPARISON_METHODS or is_comparison:
            # Show: Cosinor Fit, Phase Plot, Parameter Comparison
            self._viz_tabs.setTabVisible(0, True)
            self._viz_tabs.setTabVisible(1, True)
            self._viz_tabs.setTabVisible(2, True)
            self._viz_tabs.setTabText(0, "Cosinor Fit")
            self._viz_tabs.setTabText(1, "Phase Plot")
            self._viz_tabs.setTabText(2, "Parameter Comparison")
            return

        # Periodogram-based methods
        if method in PERIODOGRAM_METHODS:
            # Show: Periodogram, Phase Plot (if acrophase), Parameter Comparison
            self._viz_tabs.setTabVisible(0, True)  # Periodogram in first tab
            self._viz_tabs.setTabVisible(1, True)  # Phase Plot
            self._viz_tabs.setTabVisible(2, True)  # Parameter Comparison
            self._viz_tabs.setTabText(0, "Periodogram")
            self._viz_tabs.setTabText(1, "Phase Plot")
            self._viz_tabs.setTabText(2, "Parameter Comparison")
            return

        # CWT/Scalogram methods
        if method in SCALOGRAM_METHODS:
            # Show: Scalogram only
            self._viz_tabs.setTabVisible(0, True)
            self._viz_tabs.setTabText(0, "Scalogram")
            return

        # AI Meta-Classifier methods
        if method in META_CLASSIFIER_METHODS:
            self._viz_tabs.setTabVisible(0, True)
            self._viz_tabs.setTabVisible(1, True)
            self._viz_tabs.setTabVisible(2, True)
            self._viz_tabs.setTabVisible(3, True)
            self._viz_tabs.setTabText(0, "Data Overview")
            self._viz_tabs.setTabText(1, "Probability Score")
            self._viz_tabs.setTabText(2, "Method Radar")
            self._viz_tabs.setTabText(3, "Feature Importance")
            return

        # LME and other text-only methods
        if method in TEXT_ONLY_METHODS:
            # Show: Summary only
            self._viz_tabs.setTabVisible(0, True)
            self._viz_tabs.setTabText(0, "Summary")
            return

        # Default fallback: show first 3 tabs with default names
        self._viz_tabs.setTabVisible(0, True)
        self._viz_tabs.setTabVisible(1, True)
        self._viz_tabs.setTabVisible(2, True)
        self._viz_tabs.setTabText(0, "Cosinor Fit")
        self._viz_tabs.setTabText(1, "Phase Plot")
        self._viz_tabs.setTabText(2, "Parameter Comparison")

    def _update_plots(self):
        """Update all plots based on the analysis method."""
        if not self._results:
            return

        # Get the method from first result
        method = self._results[0].get('method', '')

        # Check if this is a comparison result
        is_comparison = 'condition1' in self._results[0] and 'condition2' in self._results[0]

        # =====================================================================
        # ACTIVITY PROFILE - Special case with 7 custom tabs
        # =====================================================================
        if self._results[0].get('type') == 'activity_profile':
            result = self._results[0]
            profile_data = result.get('profile_data', {})
            actogram_data = result.get('actogram_data', {})
            total_activity_data = result.get('total_activity_data', {})
            onset_data = result.get('onset_data', {})
            chi_sq_data = result.get('chi_sq_data', {})
            is_iv_data = result.get('is_iv_data', {})
            alpha_rho_data = result.get('alpha_rho_data', {})
            lighting_phases = result.get('lighting_phases', None)
            conditions = result.get('conditions', [])
            n_days = result.get('n_days', 1)

            # Show tabs 0-5 for Activity Profile; keep tab 6 (Behavioral Metrics) hidden
            for i in range(self._viz_tabs.count()):
                self._viz_tabs.setTabVisible(i, i < 6)
            self._viz_tabs.setTabText(0, "Heatmap")
            self._viz_tabs.setTabText(1, "Daily Average")
            self._viz_tabs.setTabText(2, "Actogram")
            self._viz_tabs.setTabText(3, "Total Activity")
            self._viz_tabs.setTabText(4, "Onset/Offset")
            self._viz_tabs.setTabText(5, "Chi-sq Periodogram")

            self._bar_param_container.setVisible(False)

            self._fit_canvas.plot_activity_profile(
                profile_data=profile_data,
                conditions=conditions,
                title=f"Activity Heatmap ({n_days} days)"
            )
            self._polar_canvas.plot_activity_profile_averaged(
                profile_data=profile_data,
                conditions=conditions,
                title=f"Daily Activity Profile (mean ± SEM, n={n_days} days)"
            )
            self._bar_canvas.plot_actogram(
                actogram_data=actogram_data,
                conditions=conditions,
                title="Actogram (Double-Plotted)",
                lighting_phases=lighting_phases,
            )
            self._period_canvas.plot_total_activity(
                total_activity_data=total_activity_data,
                conditions=conditions,
                title="Total Daily Activity"
            )
            self._onset_canvas.plot_activity_onset(
                onset_data=onset_data,
                conditions=conditions,
                title="Activity Onset / Offset"
            )
            self._chisq_canvas.plot_chi_square_periodogram(
                chi_sq_data=chi_sq_data,
                conditions=conditions,
                title="Chi-square Periodogram (Sokolove-Bushell)"
            )
            return

        # =====================================================================
        # Configure tabs based on method
        # =====================================================================
        self._configure_tabs_for_method(method, is_comparison)

        # =====================================================================
        # COSINORPY PERIODOGRAM - No plots (saved to directory)
        # =====================================================================
        if method == COSINORPY_PERIODOGRAM_METHOD:
            self._fit_canvas.clear()
            ax = self._fit_canvas.axes
            message = self._results[0].get('message', 'Periodogram plots saved to directory')
            ax.text(0.5, 0.5, message,
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, wrap=True,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self._fit_canvas.draw()
            return

        # =====================================================================
        # PERIODOGRAM METHODS (Lomb-Scargle, Spectral, F24)
        # =====================================================================
        if method in PERIODOGRAM_METHODS:
            # Plot periodogram in first tab
            self._update_fit_plot(self._results[0])

            # Update polar plot if we have acrophase data
            acrophases = [r.get('acrophase_hours') for r in self._results
                         if r.get('acrophase_hours') is not None]
            labels = [f"{r.get('variable', '')}_{r.get('condition', '')}"
                     for r in self._results if r.get('acrophase_hours') is not None]
            if acrophases:
                self._polar_canvas.plot_polar_acrophase(acrophases, labels)
            else:
                self._polar_canvas.clear()
                ax = self._polar_canvas.axes
                ax.text(0.5, 0.5, 'No acrophase data available',
                       ha='center', va='center', transform=ax.transAxes)
                self._polar_canvas.draw()

            # Update bar plot
            self._update_bar_plot()
            return

        # =====================================================================
        # AI META-CLASSIFIER METHODS
        # =====================================================================
        if method in META_CLASSIFIER_METHODS:
            self._update_meta_classifier_plots(self._results[0])
            return

        # =====================================================================
        # CWT/SCALOGRAM METHODS
        # =====================================================================
        if method in SCALOGRAM_METHODS:
            self._update_fit_plot(self._results[0])
            return

        # =====================================================================
        # LME/TEXT-ONLY METHODS
        # =====================================================================
        if method in TEXT_ONLY_METHODS:
            self._update_fit_plot(self._results[0])
            return

        # =====================================================================
        # COSINOR METHODS (including comparisons)
        # =====================================================================
        # Update bar plot
        self._update_bar_plot()

        # Update polar plot with all acrophases
        if is_comparison:
            acrophases = []
            labels = []
            for r in self._results:
                if r.get('acrophase_g0') is not None:
                    acro_hours = (r.get('acrophase_g0') * 24.0) / (2 * np.pi)
                    acrophases.append(acro_hours)
                    labels.append(f"{r.get('variable', '')}_{r.get('condition1', '')}")
                if r.get('acrophase_g1') is not None:
                    acro_hours = (r.get('acrophase_g1') * 24.0) / (2 * np.pi)
                    acrophases.append(acro_hours)
                    labels.append(f"{r.get('variable', '')}_{r.get('condition2', '')}")
        else:
            acrophases = [r.get('acrophase_hours') for r in self._results
                         if r.get('acrophase_hours') is not None]
            labels = [f"{r.get('variable', '')}_{r.get('condition', '')}"
                     for r in self._results if r.get('acrophase_hours') is not None]

        if acrophases:
            self._polar_canvas.plot_polar_acrophase(acrophases, labels)

        # Update fit plot with first result
        if self._results:
            self._update_fit_plot(self._results[0])
    
    def _update_fit_plot(self, result: Dict):
        """Update plot for selected result based on analysis method."""
        # Check if this is a comparison result
        is_comparison = 'condition1' in result and 'condition2' in result

        if is_comparison:
            self._plot_comparison_fit(result)
            return

        method = result.get('method', '')

        # Periodogram methods
        if method in PERIODOGRAM_METHODS:
            self._plot_periodogram_result(result)
            return

        # Scalogram/CWT methods
        if method in SCALOGRAM_METHODS:
            self._plot_scalogram_result(result)
            return

        # AI Meta-Classifier methods
        if method in META_CLASSIFIER_METHODS:
            self._update_meta_classifier_plots(result)
            return

        # Text-only methods (LME)
        if method in TEXT_ONLY_METHODS:
            self._plot_lme_result(result)
            return

        # Default: Cosinor fit for all cosinor-based methods
        self._plot_cosinor_result(result)

    def _plot_cosinor_result(self, result: Dict):
        """Plot cosinor fit for rhythm methods."""
        method = result.get('method', '')
        period = result.get('period', 24.0)
        variable = result.get('variable', '')
        condition = result.get('condition', '')

        # Get times and values from result
        times_data = result.get('times')
        values_data = result.get('values')

        if times_data is not None and values_data is not None:
            times = np.array(times_data) if not isinstance(times_data, np.ndarray) else times_data
            values = np.array(values_data) if not isinstance(values_data, np.ndarray) else values_data
        else:
            times = None
            values = None

        # Check if this is Harmonic Cosinor with a pre-computed fit model
        fit_model = result.get('fit_model')
        if fit_model is not None and 't_grid_full' in fit_model and 'model_wave_full' in fit_model:
            # Use the pre-computed harmonic cosinor fit
            self._plot_harmonic_cosinor_result(result, times, values)
            return

        # Standard cosinor plotting for other methods
        mesor = result.get('mesor')
        amplitude = result.get('amplitude')

        # Get acrophase for plotting - prefer hours-based conversion for consistent sign convention.
        # CosinorPy stores acrophase as a negative phase angle (acr) in the model cos(2πt/T + acr),
        # but our plot formula uses cos(2πt/T - acrophase_rad), which requires acrophase_rad > 0
        # (i.e., 2π * peak_hours / T). Using acrophase_hours avoids the sign mismatch.
        acrophase_hours = result.get('acrophase_hours')
        if acrophase_hours is not None:
            acrophase_rad = 2 * np.pi * acrophase_hours / period
        else:
            acrophase_rad = result.get('acrophase', result.get('acrophase_rad')) or 0

        # Calculate MESOR from data if not provided (for JTK, AR-JTK, Cosine-Kendall, etc.)
        if mesor is None:
            if values is not None and len(values) > 0:
                mesor = float(np.mean(values))
            else:
                mesor = 0

        # Handle amplitude - if None, estimate from data range
        if amplitude is None:
            if values is not None and len(values) > 0:
                amplitude = (np.max(values) - np.min(values)) / 2
            else:
                amplitude = 0

        # Generate synthetic data if not available
        if times is None or values is None:
            times = np.linspace(0, period, 24)
            values = mesor + amplitude * np.cos(
                2 * np.pi * times / period - acrophase_rad) + np.random.normal(0, amplitude * 0.1, len(times))

        n_components = result.get('n_components') or 1
        if isinstance(n_components, list):
            n_components = n_components[0]

        self._fit_canvas.plot_cosinor_fit(
            times, values, mesor, amplitude, acrophase_rad,
            period, title=variable, condition=condition,
            n_components=int(n_components)
        )

    def _plot_harmonic_cosinor_result(self, result: Dict, times: np.ndarray, values: np.ndarray):
        """Plot harmonic cosinor fit using pre-computed model wave."""
        self._fit_canvas.clear()
        ax = self._fit_canvas.axes

        fit_model = result.get('fit_model', {})
        period = result.get('period', 24.0)
        variable = result.get('variable', '')
        condition = result.get('condition', '')

        # Get the pre-computed fit curve
        t_fit = fit_model.get('t_grid_full')
        y_fit = fit_model.get('model_wave_full')
        mesor = fit_model.get('mesor', 0)

        # Plot raw data if available
        if times is not None and values is not None:
            ax.scatter(times, values, alpha=0.6, label='Data', color='steelblue')

        # Plot the harmonic cosinor fit
        if t_fit is not None and y_fit is not None:
            ax.plot(t_fit, y_fit, 'r-', linewidth=2, label='Harmonic Cosinor Fit')

        # Add horizontal line at MESOR
        if mesor != 0:
            ax.axhline(y=mesor, color='gray', linestyle='--', alpha=0.5, label=f'MESOR={mesor:.2f}')

        # Labels
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Expression')
        title_str = f'{variable} - {condition}' if condition else variable
        ax.set_title(f'Harmonic Cosinor: {title_str}')
        ax.legend(loc='upper right')

        # Set x-axis limits based on data range
        if times is not None and len(times) > 0:
            ax.set_xlim(times.min(), times.max())
        elif t_fit is not None and len(t_fit) > 0:
            ax.set_xlim(t_fit.min(), t_fit.max())

        self._fit_canvas.fig.tight_layout()
        self._fit_canvas.draw()

    def _plot_periodogram_result(self, result: Dict):
        """Plot periodogram for Lomb-Scargle, Spectral Analysis, or Fourier F24."""
        self._fit_canvas.clear()
        ax = self._fit_canvas.axes

        method = result.get('method', '')
        variable = result.get('variable', '')
        condition = result.get('condition', '')

        # Get periodogram data
        periods = result.get('periods')
        power_spectrum = result.get('power_spectrum')

        # For Fourier F24, convert frequencies to periods if periods not available
        if periods is None and power_spectrum is not None:
            frequencies = result.get('frequencies')
            if frequencies is not None:
                # Convert frequencies to periods, handling zero frequency
                frequencies = np.array(frequencies)
                valid_mask = frequencies > 0
                if valid_mask.any():
                    periods = np.zeros_like(frequencies)
                    periods[valid_mask] = 1.0 / frequencies[valid_mask]
                    # Filter to reasonable period range (exclude very long periods)
                    reasonable_mask = (periods > 0) & (periods < 100)
                    if reasonable_mask.any():
                        periods = periods[reasonable_mask]
                        power_spectrum = np.array(power_spectrum)[reasonable_mask]

        if periods is not None and power_spectrum is not None:
            # Plot power spectrum
            ax.plot(periods, power_spectrum, 'b-', linewidth=2, label='Power')

            # Mark dominant period
            dominant_period = result.get('dominant_period')
            if dominant_period is not None:
                ax.axvline(x=dominant_period, color='green', linestyle='--',
                          linewidth=2, label=f'Peak: {dominant_period:.2f}h')

            # Add significance threshold if available
            if method == 'lomb_scargle':
                fap = result.get('p_value')  # False Alarm Probability
                if fap is not None and fap < 1.0:
                    # Estimate threshold from FAP (simplified)
                    # For Lomb-Scargle, power threshold ≈ -ln(FAP)
                    threshold = -np.log(max(fap, 1e-10))
                    ax.axhline(y=threshold, color='red', linestyle='--',
                              linewidth=1, label=f'Threshold (FAP={fap:.3f})')

            elif method == 'spectral_analysis':
                threshold = result.get('threshold')
                if threshold is not None:
                    ax.axhline(y=threshold, color='red', linestyle='--',
                              linewidth=1, label=f'Threshold (p=0.05)')

            ax.set_xlabel('Period (hours)', fontsize=10)
            ax.set_ylabel('Power', fontsize=10)
            ax.set_title(f'{method.upper()}: {variable} - {condition}', fontsize=11, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            # No periodogram data available
            ax.text(0.5, 0.5, f'No periodogram data available\n({method})',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        self._fit_canvas.draw()

    def _plot_scalogram_result(self, result: Dict):
        """Plot scalogram for Wavelet (CWT) analysis."""
        self._fit_canvas.clear()
        ax = self._fit_canvas.axes

        variable = result.get('variable', '')
        condition = result.get('condition', '')
        dominant_period = result.get('dominant_period', result.get('period'))
        mean_power = result.get('power')

        # Get scalogram data
        power_matrix = result.get('scalogram_power')
        times = result.get('scalogram_times')
        periods = result.get('scalogram_periods')

        if power_matrix is not None and times is not None and periods is not None:
            # Convert to numpy arrays if needed
            power_matrix = np.array(power_matrix) if not isinstance(power_matrix, np.ndarray) else power_matrix
            times = np.array(times) if not isinstance(times, np.ndarray) else times
            periods = np.array(periods) if not isinstance(periods, np.ndarray) else periods

            # Normalize power for better visualization (log scale often works better)
            power_normalized = np.log10(power_matrix + 1e-10)

            # Plot the scalogram as a 2D heatmap
            # Use pcolormesh for better performance with large arrays
            im = ax.pcolormesh(times, periods, power_normalized, shading='auto', cmap='viridis')

            # Add colorbar
            cbar = self._fit_canvas.fig.colorbar(im, ax=ax, label='Log₁₀(Power)')

            # Mark the dominant period with a horizontal line
            if dominant_period is not None and not np.isnan(dominant_period):
                ax.axhline(y=dominant_period, color='red', linestyle='--', linewidth=1.5,
                          label=f'Dominant: {dominant_period:.1f}h')
                ax.legend(loc='upper right', fontsize=8)

            # Labels
            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_ylabel('Period (hours)', fontsize=10)
            title = f'Scalogram: {variable}'
            if condition:
                title += f' - {condition}'
            ax.set_title(title, fontsize=11, fontweight='bold')

        else:
            # Fallback: show text summary if no scalogram data
            message = f"Wavelet (CWT) Analysis\n\n"
            message += f"Variable: {variable}\n"
            message += f"Condition: {condition}\n\n"
            if dominant_period is not None:
                message += f"Dominant Period: {dominant_period:.2f} h\n"
            if mean_power is not None:
                message += f"Mean Power: {mean_power:.4f}\n"
            message += "\n(Scalogram data not available)"

            ax.text(0.5, 0.5, message,
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        self._fit_canvas.fig.tight_layout()
        self._fit_canvas.draw()

    def _plot_lme_result(self, result: Dict):
        """Plot Linear Mixed Effects results (text summary)."""
        self._fit_canvas.clear()
        ax = self._fit_canvas.axes

        variable = result.get('variable', '')
        condition = result.get('condition', '')

        # Show text summary of LME results
        message = f"Linear Mixed Effects Model\n\n"
        message += f"Variable: {variable}\n"
        message += f"Condition: {condition}\n\n"
        message += "See results table for detailed statistics"

        ax.text(0.5, 0.5, message,
               ha='center', va='center', transform=ax.transAxes,
               fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        self._fit_canvas.draw()

    # =========================================================================
    # AI META-CLASSIFIER VISUALIZATIONS
    # =========================================================================

    def _update_meta_classifier_plots(self, result: Dict):
        """Update all four meta-classifier visualization tabs."""
        import json as _json

        # Parse the JSON message containing detailed results
        try:
            details = _json.loads(result.get('message', '{}'))
        except (ValueError, TypeError):
            details = {}

        probability = details.get('probability', result.get('r_squared', 0))
        classification = details.get('classification', 'Unknown')
        method_results = details.get('method_results', {})
        feature_importances = details.get('feature_importances', {})

        # Tab 0: Data Overview (Mean ± SEM)
        self._plot_data_overview(result, details)

        # Tab 1: Probability Gauge
        self._plot_probability_gauge(probability, classification, result)

        # Tab 2: Radar Chart
        self._plot_method_radar(method_results)

        # Tab 3: Feature Importance
        self._plot_feature_importance(feature_importances)

    def _plot_data_overview(self, result: Dict, details: Dict):
        """Plot Mean +/- SEM of the time series data connected with lines."""
        self._fit_canvas.clear()
        fig = self._fit_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        variable = result.get('variable', '')
        condition = result.get('condition', '')

        # Get mean values (already averaged per timepoint in analysis engine)
        times_data = result.get('times')
        values_data = result.get('values')

        if times_data is None or values_data is None:
            ax.text(0.5, 0.5, 'No data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.axis('off')
            self._fit_canvas.draw()
            return

        times = np.array(times_data) if not isinstance(times_data, np.ndarray) else times_data
        mean_values = np.array(values_data) if not isinstance(values_data, np.ndarray) else values_data

        # Get SEM from details JSON
        sem_values = details.get('sem_values')
        has_sem = sem_values is not None and any(s > 0 for s in sem_values)
        if sem_values is not None:
            sem_values = np.array(sem_values)

        # Sort by time
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        mean_values = mean_values[sort_idx]
        if sem_values is not None:
            sem_values = sem_values[sort_idx]

        # Plot mean line with markers
        ax.plot(times, mean_values, 'o-', color='steelblue', linewidth=2,
                markersize=5, label='Mean', zorder=3)

        # Plot SEM shading if available
        if has_sem:
            ax.fill_between(times, mean_values - sem_values, mean_values + sem_values,
                            color='steelblue', alpha=0.2, label='SEM', zorder=2)

        # Labels
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel(variable, fontsize=10)
        ax.set_title(f'{variable} - {condition}', fontsize=12, fontweight='bold')

        if has_sem:
            ax.legend(fontsize=8, loc='best')

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._fit_canvas.draw()

    def _plot_probability_gauge(self, probability: float, classification: str, result: Dict):
        """Plot a semicircular gauge showing rhythmicity probability."""
        self._polar_canvas.clear()
        fig = self._polar_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        if probability is None:
            probability = 0.0

        # Draw semicircular gauge background arcs
        theta_arrhythmic = np.linspace(np.pi, np.pi * 0.7, 50)
        theta_borderline = np.linspace(np.pi * 0.7, np.pi * 0.3, 50)
        theta_rhythmic = np.linspace(np.pi * 0.3, 0, 50)

        for theta_range, color, label in [
            (theta_arrhythmic, '#ff4444', 'Arrhythmic'),
            (theta_borderline, '#ffaa00', 'Borderline'),
            (theta_rhythmic, '#44bb44', 'Rhythmic'),
        ]:
            x_outer = 1.0 * np.cos(theta_range)
            y_outer = 1.0 * np.sin(theta_range)
            x_inner = 0.6 * np.cos(theta_range)
            y_inner = 0.6 * np.sin(theta_range)
            ax.fill(
                np.concatenate([x_outer, x_inner[::-1]]),
                np.concatenate([y_outer, y_inner[::-1]]),
                color=color, alpha=0.3
            )

        # Needle
        needle_angle = np.pi * (1 - probability)
        ax.plot([0, 0.85 * np.cos(needle_angle)],
                [0, 0.85 * np.sin(needle_angle)],
                'k-', linewidth=3, solid_capstyle='round')
        ax.plot(0, 0, 'ko', markersize=8, zorder=5)

        # Score text
        color_map = {'Rhythmic': '#2d8a2d', 'Borderline': '#cc8800', 'Arrhythmic': '#cc2222'}
        ax.text(0, -0.1, f'{probability:.1%}',
                ha='center', va='top', fontsize=28, fontweight='bold')
        ax.text(0, -0.25, classification,
                ha='center', va='top', fontsize=16,
                color=color_map.get(classification, 'black'))

        # Variable info
        variable = result.get('variable', '')
        condition = result.get('condition', '')
        ax.text(0, 1.15, f'{variable} - {condition}',
                ha='center', va='bottom', fontsize=11, fontstyle='italic')

        # Zone labels
        ax.text(-0.95, 0.05, 'Arrhythmic', ha='center', fontsize=7, color='#cc2222')
        ax.text(0, 1.05, 'Borderline', ha='center', fontsize=7, color='#cc8800')
        ax.text(0.95, 0.05, 'Rhythmic', ha='center', fontsize=7, color='#2d8a2d')

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.4, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Consensus Rhythmicity Score', fontsize=13, fontweight='bold', pad=5)

        fig.tight_layout()
        self._polar_canvas.draw()

    def _plot_method_radar(self, method_results: Dict[str, float]):
        """Plot radar/spider chart of method contributions."""
        self._bar_canvas.clear()
        fig = self._bar_canvas.fig
        fig.clear()

        if not method_results:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No method results available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.axis('off')
            self._bar_canvas.draw()
            return

        labels = list(method_results.keys())
        values = [max(0.0, min(1.0, method_results[l])) for l in labels]
        N = len(labels)

        # Compute angles
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_closed = values + [values[0]]
        angles_closed = angles + [angles[0]]

        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles_closed, values_closed, 'o-', linewidth=2, color='steelblue', markersize=5)
        ax.fill(angles_closed, values_closed, alpha=0.2, color='steelblue')

        # Configure grid
        ax.set_thetagrids(np.degrees(angles), labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7)
        ax.set_title('Method Contribution Scores', fontsize=12, fontweight='bold', pad=20)

        fig.tight_layout()
        self._bar_canvas.draw()

    def _plot_feature_importance(self, feature_importances: Dict[str, float]):
        """Plot horizontal bar chart of feature importances from the Random Forest."""
        self._period_canvas.clear()
        fig = self._period_canvas.fig
        fig.clear()
        ax = fig.add_subplot(111)

        if not feature_importances:
            ax.text(0.5, 0.5, 'No feature importance data available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.axis('off')
            self._period_canvas.draw()
            return

        # Sort by importance, show top 15
        sorted_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        sorted_items = sorted_items[:15]

        _FEATURE_LABELS = {
            'f24_score':           'Fourier F24 Score',
            'cosinor_p_value':     'Cosinor p-value',
            'cosinor_r_squared':   'Cosinor R²',
            'jtk_p_value':         'JTK p-value',
            'cosinor_amplitude':   'Cosinor Amplitude',
            'ls_power':            'Lomb-Scargle Power',
            'amplitude_relative':  'Relative Amplitude',
            'period_dev_24h':      'Period Deviation from 24h',
            'method_agreement':    'Method Agreement',
            'harmonic_p_value':    'Harmonic p-value',
            'ls_p_value':          'Lomb-Scargle FAP',
            'ls_dominant_period':  'Lomb-Scargle Period',
            'log_min_p_value':     'log\u2081\u2080(min p-value)',
            'harmonic_r_squared':  'Harmonic R²',
            'period_concordance':  'Period Concordance',
            'cosinor_period':      'Cosinor Period',
            'jtk_tau':             'JTK \u03c4',
            'jtk_period':          'JTK Period',
        }
        names = [_FEATURE_LABELS.get(item[0], item[0].replace('_', ' ').title()) for item in sorted_items]
        values = [item[1] for item in sorted_items]

        y_pos = range(len(names))
        bars = ax.barh(y_pos, values, color='steelblue', alpha=0.85, edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=10)
        ax.set_title('Random Forest Feature Importances', fontsize=12, fontweight='bold')

        fig.tight_layout()
        self._period_canvas.draw()

    def _plot_comparison_fit(self, result: Dict):
        """Plot cosinor fits for both conditions in a comparison with raw data."""
        variable = result.get('variable', '')
        condition1 = result.get('condition1', '')
        condition2 = result.get('condition2', '')
        period = result.get('period', 24.0)

        # Get parameters for both groups
        mesor_g0 = result.get('mesor_g0')
        amplitude_g0 = result.get('amplitude_g0')
        acrophase_g0 = result.get('acrophase_g0')

        mesor_g1 = result.get('mesor_g1')
        amplitude_g1 = result.get('amplitude_g1')
        acrophase_g1 = result.get('acrophase_g1')

        # Get raw data for plotting (if available)
        times_g0 = result.get('times_g0')
        values_g0 = result.get('values_g0')
        times_g1 = result.get('times_g1')
        values_g1 = result.get('values_g1')

        # Handle nan/None values for all parameters
        def safe_value(val, default=0):
            """Convert None or nan to default value."""
            if val is None:
                return default
            if isinstance(val, float) and np.isnan(val):
                return default
            return val

        mesor_g0 = safe_value(mesor_g0, 0)
        amplitude_g0 = safe_value(amplitude_g0, 0)
        acrophase_g0 = safe_value(acrophase_g0, 0)

        mesor_g1 = safe_value(mesor_g1, 0)
        amplitude_g1 = safe_value(amplitude_g1, 0)
        acrophase_g1 = safe_value(acrophase_g1, 0)

        # CosinorPy stores acrophase as a negative phase angle (acr) in the model
        # cos(2πt/T + acr), but the plot formula uses cos(2πt/T - acrophase_rad),
        # which requires the positive peak-time convention.
        # Convert: acrophase_positive = (-acr * T / 2π) % T * (2π / T)
        # CircaCompare already uses the positive convention — no conversion needed.
        method = result.get('method', '')
        if method.startswith('cosinorpy_'):
            acrophase_g0 = (-acrophase_g0 * period / (2 * np.pi)) % period * (2 * np.pi / period)
            acrophase_g1 = (-acrophase_g1 * period / (2 * np.pi)) % period * (2 * np.pi / period)

        # Check if we have valid data to plot
        if amplitude_g0 == 0 and amplitude_g1 == 0:
            # No valid amplitude data, clear and show message
            self._fit_canvas.clear()
            ax = self._fit_canvas.axes
            ax.text(0.5, 0.5, 'No comparison fit data available',
                    ha='center', va='center', transform=ax.transAxes)
            self._fit_canvas.draw()
            return

        # Clear canvas and get axis
        self._fit_canvas.clear()
        ax = self._fit_canvas.axes

        # Convert raw data arrays and determine time range for fit curve
        t_min, t_max = 0, period
        if times_g0 is not None and values_g0 is not None:
            times_g0 = np.array(times_g0) if not isinstance(times_g0, np.ndarray) else times_g0
            values_g0 = np.array(values_g0) if not isinstance(values_g0, np.ndarray) else values_g0
            t_min = min(t_min, np.min(times_g0))
            t_max = max(t_max, np.max(times_g0))

        if times_g1 is not None and values_g1 is not None:
            times_g1 = np.array(times_g1) if not isinstance(times_g1, np.ndarray) else times_g1
            values_g1 = np.array(values_g1) if not isinstance(values_g1, np.ndarray) else values_g1
            t_min = min(t_min, np.min(times_g1))
            t_max = max(t_max, np.max(times_g1))

        # Plot raw data points for condition 1 (group 0) if available
        if times_g0 is not None and values_g0 is not None:
            ax.scatter(times_g0, values_g0, alpha=0.5, s=30, color='steelblue',
                      label=f'{condition1} data', zorder=1)

        # Plot raw data points for condition 2 (group 1) if available
        if times_g1 is not None and values_g1 is not None:
            ax.scatter(times_g1, values_g1, alpha=0.5, s=30, color='orangered',
                      label=f'{condition2} data', zorder=1)

        # Generate time points for smooth curves covering the full data range
        n_points = max(200, int((t_max - t_min) / period * 200))
        t_fit = np.linspace(t_min, t_max, n_points)

        # Plot fit curve for condition 1 (group 0)
        y_fit_g0 = mesor_g0 + amplitude_g0 * np.cos(2 * np.pi * t_fit / period - acrophase_g0)
        ax.plot(t_fit, y_fit_g0, '-', linewidth=2.5, label=f'{condition1} fit', color='steelblue', zorder=2)

        # Plot fit curve for condition 2 (group 1)
        y_fit_g1 = mesor_g1 + amplitude_g1 * np.cos(2 * np.pi * t_fit / period - acrophase_g1)
        ax.plot(t_fit, y_fit_g1, '-', linewidth=2.5, label=f'{condition2} fit', color='orangered', zorder=2)

        # Add horizontal lines at MESORs if they are not zero
        if mesor_g0 != 0:
            ax.axhline(y=mesor_g0, color='steelblue', linestyle='--', alpha=0.3, zorder=0)
        if mesor_g1 != 0:
            ax.axhline(y=mesor_g1, color='orangered', linestyle='--', alpha=0.3, zorder=0)

        # Labels and legend
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Expression')
        ax.set_title(f'{variable} - Comparison: {condition1} vs {condition2}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        self._fit_canvas.fig.tight_layout()
        self._fit_canvas.draw()

    def _update_periodogram_plot(self, result: Dict):
        """Update periodogram plot for selected result."""
        periods = result.get('periods')
        power = result.get('power_spectrum')
        dominant_period = result.get('dominant_period')
        threshold = result.get('threshold')

        if periods is None or power is None:
            return

        # Clear and plot
        self._period_canvas.clear()
        ax = self._period_canvas.axes

        # Plot power spectrum
        ax.plot(periods, power, 'b-', linewidth=1, label='Power')

        # Plot significance threshold
        if threshold is not None:
            ax.axhline(y=threshold, color='red', linestyle='--',
                      linewidth=1, label=f'Threshold (p=0.05)')

        # Mark dominant period
        if dominant_period is not None:
            ax.axvline(x=dominant_period, color='green', linestyle='--',
                      linewidth=2, label=f'Peak: {dominant_period:.1f}h')

        ax.set_xlabel('Period (hours)')
        ax.set_ylabel('Power')

        variable = result.get('variable', '')
        condition = result.get('condition', '')
        ax.set_title(f'Periodogram - {variable} ({condition})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._period_canvas.fig.tight_layout()
        self._period_canvas.draw()

    def _update_bar_plot(self):
        """Update bar parameter plot."""
        if not self._results:
            return

        # Check if these are comparison results
        is_comparison = 'condition1' in self._results[0] and 'condition2' in self._results[0]

        param = self._bar_param_combo.currentText()

        if is_comparison:
            # For comparisons, create a modified results list with both groups
            bar_results = []
            for r in self._results:
                # Add group 0 result
                bar_results.append({
                    'variable': r.get('variable', ''),
                    'condition': r.get('condition1', ''),
                    param: r.get(f'{param}_g0')
                })
                # Add group 1 result
                bar_results.append({
                    'variable': r.get('variable', ''),
                    'condition': r.get('condition2', ''),
                    param: r.get(f'{param}_g1')
                })
            self._bar_canvas.plot_bar_parameters(
                bar_results, parameter=param,
                title=f'{param.replace("_", " ").title()} Comparison'
            )
        else:
            # Regular single analysis
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
                self, "Save Results", "ChronoScope_results.csv",
                "CSV Files (*.csv)"
            )
        else:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Results", "ChronoScope_results.xlsx",
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
        canvases = [self._fit_canvas, self._polar_canvas, self._bar_canvas, self._period_canvas, self._onset_canvas]
        canvas = canvases[tab_idx]
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", f"ChronoScope_plot.{format}",
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
                self._onset_canvas.fig.savefig(dir_path / "activity_onset.png", dpi=300)
                
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
        canvases = [self._fit_canvas, self._polar_canvas, self._bar_canvas, self._period_canvas, self._onset_canvas]
        return canvases[tab_idx].fig
