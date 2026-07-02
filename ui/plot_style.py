"""
Plot Style
==========

User-customizable styling for matplotlib plots in the Results panel.
"""

from dataclasses import dataclass
from typing import List

from PySide6.QtCore import QSettings
import matplotlib.pyplot as plt

_SETTINGS_PREFIX = "PlotStyle/"


def _to_bool(value) -> bool:
    """QSettings may hand back bools as the literal string 'true'/'false' depending on backend."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


@dataclass
class PlotStyle:
    """Aesthetic settings applied to every PlotCanvas plot."""

    fig_width: float = 6.0
    fig_height: float = 4.0
    screen_dpi: int = 100
    export_dpi: int = 300
    primary_color: str = "#4682B4"    # steelblue - data/fit for condition 1
    secondary_color: str = "#FF7F50"  # coral - data/fit for condition 2
    condition_palette: str = "tab10"  # qualitative colormap for multi-condition plots
    heatmap_cmap: str = "YlOrRd"      # activity profile heatmap + CWT scalogram
    base_font_size: int = 9
    line_width: float = 2.0
    show_legend: bool = True

    def get_condition_colors(self, n: int) -> List[str]:
        """Resolve n distinct colors for condition-based plots.

        For 1-2 conditions, uses primary_color/secondary_color so that
        e.g. the phase plot matches the cosinor fit plot's colors.
        For 3+ conditions, falls back to the qualitative condition_palette
        colormap since there's no fixed color assigned beyond the first two.
        """
        if n <= 0:
            return []
        if n == 1:
            return [self.primary_color]
        if n == 2:
            return [self.primary_color, self.secondary_color]
        cmap = plt.get_cmap(self.condition_palette)
        return [cmap(x / (n - 1)) for x in range(n)]

    def font_title(self) -> int:
        return self.base_font_size + 3

    def font_axis(self) -> int:
        return self.base_font_size

    def font_legend(self) -> int:
        return max(6, self.base_font_size - 1)

    def font_tick(self) -> int:
        return max(6, self.base_font_size - 1)

    @classmethod
    def load(cls, settings: QSettings) -> "PlotStyle":
        """Load a PlotStyle from QSettings, falling back to defaults for missing keys."""
        defaults = cls()
        return cls(
            fig_width=float(settings.value(_SETTINGS_PREFIX + "fig_width", defaults.fig_width)),
            fig_height=float(settings.value(_SETTINGS_PREFIX + "fig_height", defaults.fig_height)),
            screen_dpi=int(settings.value(_SETTINGS_PREFIX + "screen_dpi", defaults.screen_dpi)),
            export_dpi=int(settings.value(_SETTINGS_PREFIX + "export_dpi", defaults.export_dpi)),
            primary_color=str(settings.value(_SETTINGS_PREFIX + "primary_color", defaults.primary_color)),
            secondary_color=str(settings.value(_SETTINGS_PREFIX + "secondary_color", defaults.secondary_color)),
            condition_palette=str(settings.value(_SETTINGS_PREFIX + "condition_palette", defaults.condition_palette)),
            heatmap_cmap=str(settings.value(_SETTINGS_PREFIX + "heatmap_cmap", defaults.heatmap_cmap)),
            base_font_size=int(settings.value(_SETTINGS_PREFIX + "base_font_size", defaults.base_font_size)),
            line_width=float(settings.value(_SETTINGS_PREFIX + "line_width", defaults.line_width)),
            show_legend=_to_bool(settings.value(_SETTINGS_PREFIX + "show_legend", defaults.show_legend)),
        )

    def save(self, settings: QSettings) -> None:
        """Persist this PlotStyle to QSettings."""
        settings.setValue(_SETTINGS_PREFIX + "fig_width", self.fig_width)
        settings.setValue(_SETTINGS_PREFIX + "fig_height", self.fig_height)
        settings.setValue(_SETTINGS_PREFIX + "screen_dpi", self.screen_dpi)
        settings.setValue(_SETTINGS_PREFIX + "export_dpi", self.export_dpi)
        settings.setValue(_SETTINGS_PREFIX + "primary_color", self.primary_color)
        settings.setValue(_SETTINGS_PREFIX + "secondary_color", self.secondary_color)
        settings.setValue(_SETTINGS_PREFIX + "condition_palette", self.condition_palette)
        settings.setValue(_SETTINGS_PREFIX + "heatmap_cmap", self.heatmap_cmap)
        settings.setValue(_SETTINGS_PREFIX + "base_font_size", self.base_font_size)
        settings.setValue(_SETTINGS_PREFIX + "line_width", self.line_width)
        settings.setValue(_SETTINGS_PREFIX + "show_legend", self.show_legend)


DEFAULTS = PlotStyle()
