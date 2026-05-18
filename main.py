#!/usr/bin/env python3
"""
ChronoScope - Circadian Rhythm Analysis Application
===================================================

A comprehensive desktop application for circadian rhythm analysis
in biological data, with support for gene expression, protein levels,
and locomotor activity data.

Features:
- CosinorPy integration for cosinor-based rhythmometry
- CircaCompare for differential rhythmicity analysis
- RhythmCount for circadian rhythmicity analysis of count data
- Multiple rhythm detection methods (JTKs, Lomb-Scargle, Wavelet, etc.)
- Support for Rosbash scRNA-seq circadian neuron dataset
- Interactive visualizations and export capabilities

Author: Francisco Tassara
License: MIT
"""

import sys
import os

# Ensure the application directory is in the path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# NumPy 2.0 compatibility patch
# CosinorPy uses deprecated numpy types, so we need to add them back
import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'complex'):
    np.complex = np.complex128


def check_dependencies():
    """Check that all required dependencies are installed."""
    missing = []
    
    try:
        import PySide6
    except ImportError:
        missing.append("PySide6")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """Main application entry point."""
    # Check dependencies first
    check_dependencies()
    
    # Import after dependency check
    from PySide6.QtWidgets import QApplication, QStyleFactory
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    from ui.main_window import MainWindow
    
    # Create application
    app = QApplication(sys.argv)
    
    # Application metadata
    app.setApplicationName("ChronoScope")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("LabCeriani")
    
    # Set application style
    # Try Windows style first for better dropdown rendering, fallback to Fusion
    available_styles = [s.lower() for s in QStyleFactory.keys()]
    if "windowsvista" in available_styles:
        app.setStyle("windowsvista")
    elif "windows" in available_styles:
        app.setStyle("Windows")
    else:
        app.setStyle("Fusion")
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Apply light theme stylesheet
    stylesheet = """
    QMainWindow {
        background-color: #f5f5f5;
    }
    
    QGroupBox {
        font-weight: bold;
        border: 1px solid #cccccc;
        border-radius: 5px;
        margin-top: 1ex;
        padding-top: 10px;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }
    
    QTabWidget::pane {
        border: 1px solid #cccccc;
        border-radius: 5px;
        background: white;
    }
    
    QTabBar::tab {
        background: #e0e0e0;
        border: 1px solid #cccccc;
        border-bottom: none;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        padding: 8px 16px;
        margin-right: 2px;
    }
    
    QTabBar::tab:selected {
        background: white;
        border-bottom: 1px solid white;
    }
    
    QTabBar::tab:hover:!selected {
        background: #f0f0f0;
    }
    
    QPushButton {
        background-color: #e0e0e0;
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 6px 12px;
        min-width: 80px;
    }
    
    QPushButton:hover {
        background-color: #d0d0d0;
    }
    
    QPushButton:pressed {
        background-color: #c0c0c0;
    }
    
    QPushButton:disabled {
        background-color: #f0f0f0;
        color: #999999;
    }
    
    QComboBox {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px 8px;
        background: white;
    }

    QComboBox:hover {
        border-color: #999999;
    }

    QComboBox::drop-down {
        border: none;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid #666666;
        margin-right: 5px;
    }

    QComboBox:on {
        border-color: #4a90d9;
    }

    QComboBox QAbstractItemView {
        border: 1px solid #cccccc;
        background-color: white;
        selection-background-color: #4a90d9;
        selection-color: white;
        outline: none;
        show-decoration-selected: 1;
    }

    QComboBox QAbstractItemView::item {
        padding: 5px 10px;
        min-height: 20px;
        color: #000000;
        background-color: transparent;
        border: none;
    }

    QComboBox QAbstractItemView::item:hover {
        background-color: #b3d9ff;
        color: #000000;
        border: none;
    }

    QComboBox QAbstractItemView::item:selected {
        background-color: #4a90d9;
        color: #ffffff;
        border: none;
    }

    QComboBox QAbstractItemView::item:selected:hover {
        background-color: #3a7bc8;
        color: #ffffff;
    }
    
    QSpinBox, QDoubleSpinBox {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px;
        background: white;
    }
    
    QLineEdit {
        border: 1px solid #cccccc;
        border-radius: 4px;
        padding: 4px 8px;
        background: white;
    }
    
    QLineEdit:focus {
        border-color: #4a90d9;
    }
    
    QTableWidget {
        border: 1px solid #cccccc;
        border-radius: 4px;
        background: white;
        gridline-color: #e0e0e0;
    }
    
    QTableWidget::item:selected {
        background-color: #4a90d9;
        color: white;
    }
    
    QHeaderView::section {
        background-color: #f0f0f0;
        border: none;
        border-right: 1px solid #cccccc;
        border-bottom: 1px solid #cccccc;
        padding: 6px;
        font-weight: bold;
    }
    
    QListWidget {
        border: 1px solid #cccccc;
        border-radius: 4px;
        background: white;
    }
    
    QListWidget::item:selected {
        background-color: #4a90d9;
        color: white;
    }
    
    QProgressBar {
        border: 1px solid #cccccc;
        border-radius: 4px;
        text-align: center;
        background: white;
    }
    
    QProgressBar::chunk {
        background-color: #4a90d9;
        border-radius: 3px;
    }
    
    QStatusBar {
        background: #f0f0f0;
        border-top: 1px solid #cccccc;
    }
    
    QToolBar {
        background: #f5f5f5;
        border-bottom: 1px solid #cccccc;
        spacing: 5px;
        padding: 5px;
    }
    
    QScrollArea {
        border: none;
    }
    """
    
    app.setStyleSheet(stylesheet)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
