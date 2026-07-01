"""
Plot Settings Dialog
=====================

Dialog letting the user customize figure size, colors, fonts, and line
width used by the plots in the Results panel.
"""

import copy

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QDialogButtonBox,
    QColorDialog
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor

from ui.plot_style import PlotStyle, DEFAULTS

CONDITION_PALETTES = ["tab10", "Set1", "Set2", "Dark2", "Paired"]
HEATMAP_CMAPS = ["YlOrRd", "viridis", "plasma", "coolwarm", "magma"]


class _ColorButton(QPushButton):
    """A small button that shows a color swatch and opens a QColorDialog on click."""

    color_changed = Signal(str)

    def __init__(self, color_hex: str, parent=None):
        super().__init__(parent)
        self._color = color_hex
        self.setFixedWidth(60)
        self.clicked.connect(self._pick_color)
        self._update_swatch()

    def _update_swatch(self):
        self.setStyleSheet(f"background-color: {self._color};")
        self.setText(self._color)

    def _pick_color(self):
        chosen = QColorDialog.getColor(QColor(self._color), self, "Select color")
        if chosen.isValid():
            self._color = chosen.name()
            self._update_swatch()
            self.color_changed.emit(self._color)

    def color(self) -> str:
        return self._color

    def set_color(self, color_hex: str):
        self._color = color_hex
        self._update_swatch()


class PlotSettingsDialog(QDialog):
    """
    Dialog for customizing PlotStyle (figure size, DPI, colors, fonts, line width).

    Signals:
        style_changed: Emitted with the new PlotStyle whenever Apply or OK is clicked.
    """

    style_changed = Signal(object)  # PlotStyle

    def __init__(self, style: PlotStyle, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")
        self._style = copy.copy(style)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Figure group ---
        fig_group = QGroupBox("Figure")
        fig_form = QFormLayout(fig_group)

        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(2.0, 20.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setValue(self._style.fig_width)
        fig_form.addRow("Width (in):", self._width_spin)

        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(2.0, 20.0)
        self._height_spin.setSingleStep(0.5)
        self._height_spin.setValue(self._style.fig_height)
        fig_form.addRow("Height (in):", self._height_spin)

        self._screen_dpi_spin = QSpinBox()
        self._screen_dpi_spin.setRange(50, 300)
        self._screen_dpi_spin.setValue(self._style.screen_dpi)
        fig_form.addRow("Screen DPI:", self._screen_dpi_spin)

        self._export_dpi_spin = QSpinBox()
        self._export_dpi_spin.setRange(50, 1200)
        self._export_dpi_spin.setValue(self._style.export_dpi)
        fig_form.addRow("Export DPI:", self._export_dpi_spin)

        layout.addWidget(fig_group)

        # --- Colors group ---
        color_group = QGroupBox("Colors")
        color_form = QFormLayout(color_group)

        self._primary_btn = _ColorButton(self._style.primary_color)
        color_form.addRow("Primary color:", self._primary_btn)

        self._secondary_btn = _ColorButton(self._style.secondary_color)
        color_form.addRow("Secondary color:", self._secondary_btn)

        self._palette_combo = QComboBox()
        self._palette_combo.addItems(CONDITION_PALETTES)
        self._palette_combo.setCurrentText(self._style.condition_palette)
        color_form.addRow("Condition palette:", self._palette_combo)

        self._heatmap_combo = QComboBox()
        self._heatmap_combo.addItems(HEATMAP_CMAPS)
        self._heatmap_combo.setCurrentText(self._style.heatmap_cmap)
        color_form.addRow("Heatmap colormap:", self._heatmap_combo)

        layout.addWidget(color_group)

        # --- Text & lines group ---
        text_group = QGroupBox("Text && Lines")
        text_form = QFormLayout(text_group)

        self._font_spin = QSpinBox()
        self._font_spin.setRange(6, 20)
        self._font_spin.setValue(self._style.base_font_size)
        text_form.addRow("Base font size:", self._font_spin)

        self._line_width_spin = QDoubleSpinBox()
        self._line_width_spin.setRange(0.5, 6.0)
        self._line_width_spin.setSingleStep(0.5)
        self._line_width_spin.setValue(self._style.line_width)
        text_form.addRow("Line width:", self._line_width_spin)

        layout.addWidget(text_group)

        # --- Buttons ---
        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        button_row.addWidget(reset_btn)
        button_row.addStretch()
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_style(self) -> PlotStyle:
        """Build a PlotStyle from the current widget values."""
        return PlotStyle(
            fig_width=self._width_spin.value(),
            fig_height=self._height_spin.value(),
            screen_dpi=self._screen_dpi_spin.value(),
            export_dpi=self._export_dpi_spin.value(),
            primary_color=self._primary_btn.color(),
            secondary_color=self._secondary_btn.color(),
            condition_palette=self._palette_combo.currentText(),
            heatmap_cmap=self._heatmap_combo.currentText(),
            base_font_size=self._font_spin.value(),
            line_width=self._line_width_spin.value(),
        )

    def _reset_defaults(self):
        d = DEFAULTS
        self._width_spin.setValue(d.fig_width)
        self._height_spin.setValue(d.fig_height)
        self._screen_dpi_spin.setValue(d.screen_dpi)
        self._export_dpi_spin.setValue(d.export_dpi)
        self._primary_btn.set_color(d.primary_color)
        self._secondary_btn.set_color(d.secondary_color)
        self._palette_combo.setCurrentText(d.condition_palette)
        self._heatmap_combo.setCurrentText(d.heatmap_cmap)
        self._font_spin.setValue(d.base_font_size)
        self._line_width_spin.setValue(d.line_width)

    def _on_apply(self):
        self.style_changed.emit(self.get_style())

    def _on_ok(self):
        self.style_changed.emit(self.get_style())
        self.accept()
