"""
Export Module
=============

Handles exporting analysis results to CSV, Excel, and image formats.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

# Matplotlib for figure export
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Excel export
try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except ImportError:
    XLSXWRITER_AVAILABLE = False


class ResultExporter:
    """
    Handles exporting analysis results to various formats.
    """
    
    SUPPORTED_TABLE_FORMATS = ['csv', 'xlsx', 'tsv']
    SUPPORTED_FIGURE_FORMATS = ['png', 'svg', 'pdf', 'jpg', 'tiff']
    
    def __init__(self):
        """Initialize the exporter."""
        self._check_dependencies()
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check which export formats are available."""
        return {
            'csv': True,
            'tsv': True,
            'xlsx': OPENPYXL_AVAILABLE or XLSXWRITER_AVAILABLE,
            'png': MATPLOTLIB_AVAILABLE,
            'svg': MATPLOTLIB_AVAILABLE,
            'pdf': MATPLOTLIB_AVAILABLE,
            'jpg': MATPLOTLIB_AVAILABLE,
            'tiff': MATPLOTLIB_AVAILABLE
        }
    
    # =========================================================================
    # TABLE EXPORT
    # =========================================================================
    
    def export_dataframe(
        self,
        df: pd.DataFrame,
        filepath: str,
        format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Export a DataFrame to file.
        
        Args:
            df: DataFrame to export
            filepath: Output file path
            format: File format ('csv', 'xlsx', 'tsv'). Auto-detected if None.
            **kwargs: Additional arguments passed to pandas export function
        
        Returns:
            Path to exported file
        """
        filepath = Path(filepath)
        
        # Auto-detect format from extension
        if format is None:
            format = filepath.suffix.lstrip('.').lower()
        
        if format not in self.SUPPORTED_TABLE_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_TABLE_FORMATS}")
        
        if format == 'csv':
            df.to_csv(filepath, index=kwargs.pop('index', False), **kwargs)
        
        elif format == 'tsv':
            df.to_csv(filepath, sep='\t', index=kwargs.pop('index', False), **kwargs)
        
        elif format == 'xlsx':
            if not (OPENPYXL_AVAILABLE or XLSXWRITER_AVAILABLE):
                raise ImportError("Excel export requires openpyxl or xlsxwriter")
            
            engine = 'openpyxl' if OPENPYXL_AVAILABLE else 'xlsxwriter'
            df.to_excel(filepath, index=kwargs.pop('index', False), engine=engine, **kwargs)
        
        return str(filepath)
    
    def export_results_dict(
        self,
        results: Dict[str, Any],
        filepath: str,
        format: str = 'csv'
    ) -> str:
        """
        Export a results dictionary to file.
        
        Args:
            results: Dictionary with results
            filepath: Output file path
            format: File format
        
        Returns:
            Path to exported file
        """
        # Convert dict to DataFrame
        if isinstance(results, dict):
            # Handle nested dicts
            flat_results = self._flatten_dict(results)
            df = pd.DataFrame([flat_results])
        else:
            df = pd.DataFrame([results])
        
        return self.export_dataframe(df, filepath, format)
    
    def export_multiple_results(
        self,
        results_list: List[Dict[str, Any]],
        filepath: str,
        format: str = 'csv'
    ) -> str:
        """
        Export multiple results to a single file.
        
        Args:
            results_list: List of result dictionaries
            filepath: Output file path
            format: File format
        
        Returns:
            Path to exported file
        """
        # Flatten each result
        flat_results = [self._flatten_dict(r) for r in results_list]
        df = pd.DataFrame(flat_results)
        
        return self.export_dataframe(df, filepath, format)
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '_'
    ) -> Dict[str, Any]:
        """Flatten nested dictionaries."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple, np.ndarray)):
                # Convert arrays to string representation for simple export
                if len(v) <= 5:
                    items.append((new_key, str(list(v))))
                else:
                    items.append((new_key, f"[{len(v)} items]"))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    # =========================================================================
    # FIGURE EXPORT
    # =========================================================================
    
    def export_figure(
        self,
        fig: 'Figure',
        filepath: str,
        format: Optional[str] = None,
        dpi: int = 300,
        transparent: bool = False,
        **kwargs
    ) -> str:
        """
        Export a matplotlib Figure to file.
        
        Args:
            fig: Matplotlib Figure object
            filepath: Output file path
            format: File format. Auto-detected if None.
            dpi: Resolution for raster formats
            transparent: Use transparent background
            **kwargs: Additional arguments for savefig
        
        Returns:
            Path to exported file
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for figure export")
        
        filepath = Path(filepath)
        
        # Auto-detect format
        if format is None:
            format = filepath.suffix.lstrip('.').lower()
        
        if format not in self.SUPPORTED_FIGURE_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FIGURE_FORMATS}")
        
        # Ensure correct extension
        if not filepath.suffix:
            filepath = filepath.with_suffix(f'.{format}')
        
        fig.savefig(
            filepath,
            format=format,
            dpi=dpi,
            transparent=transparent,
            bbox_inches='tight',
            **kwargs
        )
        
        return str(filepath)
    
    def export_all_figures(
        self,
        figures: Dict[str, 'Figure'],
        output_dir: str,
        format: str = 'png',
        dpi: int = 300
    ) -> List[str]:
        """
        Export multiple figures to a directory.
        
        Args:
            figures: Dictionary mapping names to Figure objects
            output_dir: Output directory path
            format: File format for all figures
            dpi: Resolution
        
        Returns:
            List of exported file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = []
        for name, fig in figures.items():
            # Sanitize filename
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in name)
            filepath = output_dir / f"{safe_name}.{format}"
            
            exported.append(self.export_figure(fig, filepath, format, dpi))
        
        return exported
    
    # =========================================================================
    # COMPREHENSIVE EXPORT
    # =========================================================================
    
    def export_analysis_report(
        self,
        results_df: pd.DataFrame,
        figures: Optional[Dict[str, 'Figure']] = None,
        output_dir: str = '.',
        base_name: str = 'ChronoScope_results',
        table_format: str = 'xlsx',
        figure_format: str = 'png',
        dpi: int = 300
    ) -> Dict[str, List[str]]:
        """
        Export complete analysis results including tables and figures.
        
        Args:
            results_df: DataFrame with analysis results
            figures: Dictionary of figures to export
            output_dir: Output directory
            base_name: Base name for output files
            table_format: Format for table export
            figure_format: Format for figure export
            dpi: Resolution for figures
        
        Returns:
            Dictionary with 'tables' and 'figures' keys containing exported paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {'tables': [], 'figures': []}
        
        # Export results table
        table_path = output_dir / f"{base_name}.{table_format}"
        exported['tables'].append(
            self.export_dataframe(results_df, table_path, table_format)
        )
        
        # Export figures
        if figures:
            fig_dir = output_dir / f"{base_name}_figures"
            fig_dir.mkdir(exist_ok=True)
            
            exported['figures'] = self.export_all_figures(
                figures, fig_dir, figure_format, dpi
            )
        
        return exported


class ExcelReportWriter:
    """
    Creates formatted Excel reports with multiple sheets.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the Excel writer.
        
        Args:
            filepath: Output Excel file path
        """
        if not (OPENPYXL_AVAILABLE or XLSXWRITER_AVAILABLE):
            raise ImportError("Excel export requires openpyxl or xlsxwriter")
        
        self.filepath = filepath
        self.engine = 'openpyxl' if OPENPYXL_AVAILABLE else 'xlsxwriter'
        self._writer = None
        self._sheets = []
    
    def __enter__(self):
        self._writer = pd.ExcelWriter(self.filepath, engine=self.engine)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._writer:
            self._writer.close()
    
    def add_sheet(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        index: bool = False
    ) -> None:
        """
        Add a DataFrame as a new sheet.
        
        Args:
            df: DataFrame to add
            sheet_name: Name for the sheet (max 31 chars)
            index: Include DataFrame index
        """
        # Excel sheet names are limited to 31 characters
        sheet_name = sheet_name[:31]
        
        df.to_excel(self._writer, sheet_name=sheet_name, index=index)
        self._sheets.append(sheet_name)
    
    def add_summary_sheet(
        self,
        summary_data: Dict[str, Any],
        sheet_name: str = 'Summary'
    ) -> None:
        """
        Add a summary information sheet.
        
        Args:
            summary_data: Dictionary with summary information
            sheet_name: Name for the sheet
        """
        # Convert to two-column DataFrame
        rows = []
        for key, value in summary_data.items():
            if isinstance(value, (list, np.ndarray)):
                value = ', '.join(str(v) for v in value[:10])
                if len(value) > 10:
                    value += '...'
            rows.append({'Parameter': key, 'Value': str(value)})
        
        df = pd.DataFrame(rows)
        self.add_sheet(df, sheet_name)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_export_csv(df: pd.DataFrame, filepath: str) -> str:
    """Quick export DataFrame to CSV."""
    exporter = ResultExporter()
    return exporter.export_dataframe(df, filepath, 'csv')


def quick_export_excel(df: pd.DataFrame, filepath: str) -> str:
    """Quick export DataFrame to Excel."""
    exporter = ResultExporter()
    return exporter.export_dataframe(df, filepath, 'xlsx')


def quick_export_figure(fig: 'Figure', filepath: str, dpi: int = 300) -> str:
    """Quick export matplotlib figure."""
    exporter = ResultExporter()
    return exporter.export_figure(fig, filepath, dpi=dpi)
