# ChronoScope

<p align="center">
  <strong>A Comprehensive Desktop Application for Circadian Rhythm Analysis</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#methods">Methods</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

ChronoScope is a cross-platform desktop application designed for comprehensive circadian rhythm analysis in biological data. It provides an intuitive graphical interface for analyzing gene expression, protein levels, and locomotor activity data using state-of-the-art statistical methods.

### Key Advantages

- **Desktop Application**: No internet required, process data locally
- **Multiple Analysis Methods**: CosinorPy, CircaCompare, JTK Cycle, and more
- **Group Comparison**: Statistical comparison of rhythmicity between conditions
- **scRNA-seq Support**: Built-in support for Rosbash circadian neuron dataset
- **Publication-Ready**: Export figures in PNG, SVG, and PDF formats

## Features

### Data Input
- **CSV Files**: Standard format with time, condition, and variable columns
- **Rosbash Dataset**: Pre-curated scRNA-seq data from Drosophila clock neurons

### Analysis Methods

#### CosinorPy Module
- Single-component cosinor fitting
- Multi-component cosinor (up to 6 harmonics)
- Population-mean cosinor for longitudinal data
- Differential rhythmicity analysis
- Poisson cosinor for count data

#### CircaCompare Module
- Robust cosinor fitting with multiple loss functions
- Statistical comparison of rhythm parameters between groups
- Confidence intervals for MESOR, amplitude, and acrophase

#### Rhythm Analysis Module
- **JTK Cycle**: Nonparametric rhythm detection
- **AR-JTK**: Autoregressive-corrected JTK
- **Lomb-Scargle**: For unevenly sampled data
- **Wavelet (CWT)**: Time-frequency analysis
- **Fourier F24**: Effect size measure
- **Harmonic Cosinor**: Multi-modal rhythms
- **Linear Mixed Effects**: Hierarchical modeling

### Visualizations
- Cosinor fit plots with data overlay
- Polar phase plots (acrophase distribution)
- Parameter comparison bar charts
- Periodogram plots

### Export
- Results tables: CSV, Excel
- Figures: PNG, SVG, PDF
- Batch export of all visualizations

## Installation

### Requirements
- Python 3.9 or higher
- PySide6 (Qt for Python)
- See `requirements.txt` for complete list

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ChronoScope.git
cd ChronoScope

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Using pip (coming soon)

```bash
pip install chronoscope
chronoscope  # Run the application
```

## Usage

### Quick Start

1. **Load Data**: 
   - Click "Browse..." to select a CSV file
   - Or select "Rosbash scRNA-seq Dataset" and load the HDF5 file

2. **Configure Columns**:
   - Select time, condition, and variable columns
   - The application auto-detects common column names

3. **Run Analysis**:
   - Choose analysis method (CosinorPy, CircaCompare, etc.)
   - Set parameters (period, components, etc.)
   - Click "Run Analysis"

4. **View Results**:
   - Summary table with all results
   - Interactive visualizations
   - Export to CSV/Excel or images

### CSV Format

```csv
time,condition,replicate,gene1,gene2,gene3
0,control,1,10.2,5.3,8.1
0,control,2,10.5,5.1,8.3
4,control,1,12.1,6.2,9.2
4,control,2,11.8,6.4,9.0
...
```

**Required columns:**
- `time`: Numeric time values (hours)
- `condition`: Group/treatment identifier
- Variable columns: Numeric expression values

**Optional columns:**
- `replicate`: Replicate identifier
- `subject`: For longitudinal data (population-mean cosinor)

### Rosbash Dataset

The application supports the preprocessed Rosbash circadian neuron dataset:

> Ma D, Przybylski D, Bhinder T, et al. "A transcriptomic taxonomy of Drosophila circadian neurons around the clock." eLife 2021;10:e63056.

The HDF5 file should be preprocessed using the included `process_rosbash_dataset.py` script.

## Methods

### Cosinor Analysis

The cosinor model fits:
```
Y(t) = M + A × cos(2πt/τ - φ) + ε
```

Where:
- **M** (MESOR): Midline Estimating Statistic Of Rhythm
- **A** (Amplitude): Half the peak-to-trough difference
- **φ** (Acrophase): Time of peak expression
- **τ** (Period): Fixed or estimated oscillation period

### CircaCompare

Compares rhythm parameters between groups using robust regression with selectable loss functions:
- `linear`: Standard least squares
- `soft_l1`: Smooth L1 loss
- `huber`: Huber loss (robust to outliers)
- `cauchy`: Cauchy loss (very robust)
- `arctan`: Arctan loss

### JTK Cycle

Nonparametric algorithm that:
1. Tests for rhythmicity using Kendall's τ
2. Estimates period and phase
3. Does not assume sinusoidal waveform

### Lomb-Scargle Periodogram

Spectral analysis method ideal for:
- Unevenly sampled time series
- Period estimation
- Detecting multiple periodicities

## Output Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| MESOR | Rhythm-adjusted mean | Same as input |
| Amplitude | Peak-to-trough / 2 | Same as input |
| Acrophase | Time of peak | Hours |
| Period | Oscillation period | Hours |
| p-value | Statistical significance | - |
| R² | Goodness of fit | 0-1 |

## Citation

If you use ChronoScope in your research, please cite:

```bibtex
@software{chronoscope2024,
  author = {Tassara, Francisco},
  title = {ChronoScope: A Desktop Application for Circadian Rhythm Analysis},
  year = {2024},
  url = {https://github.com/yourusername/ChronoScope}
}
```

Also cite the underlying methods:

**CosinorPy:**
> Moškon M. (2020). CosinorPy: A Python Package for Cosinor-based Rhythmometry. 
> *Bioinformatics*, 36(22-23), 5507-5508.

**CircaCompare:**
> Parsons R, et al. (2020). CircaCompare: A method to estimate and statistically 
> support differences in mesor, amplitude and phase, between circadian rhythms.
> *Bioinformatics*, 36(4), 1208-1212.

**JTK_CYCLE:**
> Hughes ME, et al. (2010). JTK_CYCLE: An efficient nonparametric algorithm for 
> detecting rhythmic components in genome-scale data sets. *Journal of Biological 
> Rhythms*, 25(5), 372-380.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ChronoScope/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ChronoScope/discussions)

---

<p align="center">
  Made with ❤️ for the circadian biology community
</p>
