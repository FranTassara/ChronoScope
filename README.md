# ChronoScope

<p align="center">
  <strong>A Desktop Application for Multi-Method Circadian Rhythm Analysis</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#methods">Methods</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

Identifying rhythmic patterns in biological data requires choosing among a growing number of statistical methods — each with different assumptions, sensitivities, and failure modes. ChronoScope addresses this challenge by providing a unified, cross-platform desktop application that integrates seven established rhythm-detection algorithms alongside a novel machine-learning consensus score (CRS-AI), all accessible through a graphical interface that requires no programming experience.

ChronoScope is designed for circadian biologists working with gene expression, protein abundance, locomotor activity, count data, or any uniformly sampled time series. It handles data from CSV files, Drosophila activity monitors (DAM systems, TriKinetics), and the Rosbash single-cell RNA-seq circadian neuron dataset, and exports publication-ready figures in vector and raster formats.

### Key Features

- **No coding required**: point-and-click interface built with PySide6 (Qt)
- **Five integrated analysis modules**: CosinorPy, CircaCompare, RhythmCount, Rhythm Analysis Suite, and CRS-AI
- **Count-data support**: GLM-based cosinor fitting with five count distributions (Poisson, NB, ZIP, ZINB, Generalized Poisson)
- **Differential rhythmicity**: statistical comparison of cosinor parameters between conditions
- **CRS-AI**: Random Forest consensus score integrating outputs from multiple methods
- **DAM Monitor support**: native parsing of TriKinetics locomotor activity files
- **scRNA-seq support**: integrated loader for the Rosbash *Drosophila* clock neuron dataset
- **Publication-ready output**: PNG, SVG, PDF figures; CSV and Excel result tables

---

## Analysis Modules

### 1. CosinorPy Module

Wraps the CosinorPy library (Moškon 2020) with a graphical interface for:

| Analysis | Description |
|---|---|
| Single-component cosinor | Fits M + A·cos(2πt/τ − φ) to a single group |
| Multi-component cosinor | Up to 6 harmonics for non-sinusoidal waveforms |
| Population-mean cosinor | Longitudinal/repeated-measures designs |
| Differential rhythmicity | Tests MESOR, amplitude, and acrophase differences between two groups |
| Poisson cosinor | Cosinor extension for count data (sequencing read counts) |

### 2. CircaCompare Module

Implements the CircaCompare framework (Parsons et al. 2020) for robust comparison of rhythm parameters between conditions. Supports five loss functions for outlier-resilient fitting:

`linear` · `soft_l1` · `huber` · `cauchy` · `arctan`

Outputs confidence intervals for MESOR, amplitude, and acrophase differences.

### 3. RhythmCount Module

Wraps the RhythmCount library (Velikajne et al. 2022) to fit cosinor models within a generalized linear model (GLM) framework for **count-valued time series** — RNA-seq read counts, neuronal spike counts, locomotor activity event tallies, or any non-negative discrete data where the Gaussian residual assumption of standard cosinor is violated.

The expected count at each time point is modeled as:

$$\log(\mu_i) = \beta_0 + \sum_{k=1}^{N} \left[ \beta_{2k-1} \sin\!\left(\frac{2\pi k t_i}{\tau}\right) + \beta_{2k} \cos\!\left(\frac{2\pi k t_i}{\tau}\right) \right]$$

Five count distributions are supported:

| Distribution | When to use |
|---|---|
| Poisson | Equidispersed counts (variance ≈ mean) |
| Generalized Poisson | Flexible dispersion (sub- or over-dispersed) |
| Negative Binomial | Over-dispersed counts (variance > mean); typical in RNA-seq |
| Zero-Inflated Poisson | Excess structural zeros + Poisson counts |
| Zero-Inflated Negative Binomial | Excess structural zeros + overdispersion |

**Five analysis methods** are available:

| Method | Description |
|---|---|
| Fit Single Model | Fits a user-specified distribution and harmonic complexity |
| Fit All Models | Grid search over all distribution × harmonic combinations |
| Fit Best Model | Automatic two-stage model selection (AIC / BIC / Vuong / F-test) |
| Parameter CIs | Bootstrap confidence intervals for amplitude, MESOR, and acrophase |
| Group Comparison | Independent best-model selection and CI estimation per group |

Goodness of fit is assessed via the likelihood ratio test (LLR p-value), AIC, BIC, and McFadden's pseudo-R². Amplitude, MESOR, and acrophase are extracted from the predicted curve after fitting.

### 4. Rhythm Analysis Suite

Seven algorithms available for period detection and rhythmicity testing:

| Method | Type | Best suited for |
|---|---|---|
| JTK Cycle | Nonparametric | Genome-scale datasets; no waveform assumption |
| AR-JTK | Nonparametric | Autoregressive-corrected JTK for autocorrelated data |
| Lomb–Scargle | Spectral | Unevenly sampled or missing time points |
| Wavelet (CWT) | Time–frequency | Non-stationary rhythms; temporal changes in amplitude |
| Fourier F24 | Effect size | Quantifying 24 h power relative to total variance |
| Harmonic cosinor | Parametric | Multi-modal waveforms (up to 4 harmonics) |
| Linear mixed effects | Hierarchical | Nested or longitudinal designs |

### 5. CRS-AI (Consensus Rhythmicity Score)

A Random Forest classifier trained on synthetic and real circadian time series that aggregates feature vectors extracted from JTK Cycle, single-component cosinor, and Lomb–Scargle into a single probability score (0–1). CRS-AI provides an integrated assessment that is more robust than any individual method alone, particularly for short or noisy time series.

The model was trained on synthetic oscillations spanning a range of amplitudes, noise levels, and sampling densities, and validated on two public datasets: GSE11923 (mouse liver, hourly × 48 h) and the Rosbash *Drosophila* circadian neuron scRNA-seq dataset.

---

## Installation

### Requirements

- Python 3.9 or higher
- See `requirements.txt` for the complete dependency list

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ChronoScope.git
cd ChronoScope

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the application
python main.py
```

> **Note:** ChronoScope patches deprecated NumPy type aliases at startup to maintain compatibility with CosinorPy on NumPy ≥ 2.0.

---

## Usage

### Quick Start

1. **Load data** — click *Browse…* to select a CSV file, or choose *DAM Monitor* / *Rosbash scRNA-seq* from the dataset selector.
2. **Map columns** — assign time, condition, and variable columns. The application auto-detects common column names.
3. **Select a module** — CosinorPy, CircaCompare, Rhythm Analysis, or CRS-AI.
4. **Run analysis** — set parameters (period, number of harmonics, etc.) and click *Run Analysis*.
5. **Export results** — summary tables (CSV / Excel) and figures (PNG / SVG / PDF) via the *Export* panel.

### CSV Format

```
time,condition,replicate,gene1,gene2,gene3
0,control,1,10.2,5.3,8.1
0,control,2,10.5,5.1,8.3
4,control,1,12.1,6.2,9.2
4,control,2,11.8,6.4,9.0
...
```

**Required columns:**
- `time` — numeric time in hours
- `condition` — group or treatment label
- One or more numeric variable columns

**Optional columns:**
- `replicate` — replicate identifier
- `subject` — subject/animal ID (required for population-mean cosinor)

### DAM Monitor Files

Load raw TriKinetics `.txt` monitor files directly. ChronoScope extracts activity counts per beam break, bins data to the requested resolution, and structures it for downstream analysis.

### Rosbash scRNA-seq Dataset

Preprocessed HDF5 file derived from:

> Ma D, Przybylski D, Bhinder T, et al. A transcriptomic taxonomy of *Drosophila* circadian neurons around the clock. *eLife* 2021;10:e63056.

Use the included `Rosbash_data/process_rosbash_dataset.py` script to generate the HDF5 file from the raw GEO data.

---

## Methods

### Cosinor Model

$$Y(t) = M + A \cos\!\left(\frac{2\pi t}{\tau} - \varphi\right) + \varepsilon$$

| Symbol | Parameter | Description |
|--------|-----------|-------------|
| M | MESOR | Rhythm-adjusted mean (Midline Estimating Statistic Of Rhythm) |
| A | Amplitude | Half the peak-to-trough difference |
| φ | Acrophase | Time of peak in radians; converted to hours for output |
| τ | Period | Fixed (default 24 h) or estimated |
| ε | Residual | Assumed i.i.d. Gaussian |

Statistical significance is assessed via the zero-amplitude test (F-test on A = 0).

### JTK Cycle

A nonparametric procedure that:
1. Ranks the time series and computes Kendall's τ against a reference waveform template for each candidate period
2. Corrects for multiple period tests using the Bonferroni–Dunn method
3. Returns estimated period, phase, and amplitude without assuming a sinusoidal shape

### Lomb–Scargle Periodogram

Evaluates spectral power at each candidate frequency using a least-squares projection, making it valid for unevenly sampled or gapped time series. Period is estimated at the frequency with maximum power; a *p*-value is derived from the false alarm probability.

### CRS-AI Feature Vector

For each time series, ChronoScope extracts 18 features from JTK Cycle, cosinor, and Lomb–Scargle (p-values, effect sizes, estimated periods, R², method agreement index, and relative amplitude). A pre-trained Random Forest classifier (100 trees, trained on ≥ 10 000 synthetic instances) maps this feature vector to a rhythmicity probability score.

---

## Output Parameters

| Parameter | Description | Units |
|-----------|-------------|-------|
| MESOR | Rhythm-adjusted mean | Same as input |
| Amplitude | Half peak-to-trough difference | Same as input |
| Acrophase | Time of peak | Hours |
| Period | Oscillation period | Hours |
| p-value | Test of zero amplitude (or equivalent) | — |
| R² | Goodness of fit | 0–1 |
| CRS score | Consensus rhythmicity probability | 0–1 |

---

## Citation

If you use ChronoScope in your research, please cite:

```bibtex
@article{tassara2025chronoscope,
  author  = {Tassara, Francisco},
  title   = {ChronoScope: A Desktop Application for Multi-Method Circadian Rhythm Analysis},
  journal = {Journal of Biological Rhythms},
  year    = {2025},
  doi     = {TODO}
}
```

Please also cite the underlying methods used in your analysis:

**CosinorPy:**
> Moškon M. (2020). CosinorPy: a Python package for cosinor-based rhythmometry.
> *Source Code for Biology and Medicine*, 15(1), 1–10.

**CircaCompare:**
> Parsons R, et al. (2020). CircaCompare: a method to estimate and statistically support
> differences in mesor, amplitude and phase, between circadian rhythms.
> *Bioinformatics*, 36(4), 1208–1212.

**RhythmCount:**
> Velikajne N, et al. (2022). RhythmCount: an R/Python package for circadian
> rhythmicity analysis of count data. DOI: to be confirmed upon publication.

**JTK_CYCLE:**
> Hughes ME, et al. (2010). JTK_CYCLE: an efficient nonparametric algorithm for
> detecting rhythmic components in genome-scale datasets.
> *Journal of Biological Rhythms*, 25(5), 372–380.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PySide6 | ≥ 6.5 | GUI framework |
| NumPy | ≥ 1.24 | Numerical computation |
| pandas | ≥ 1.5 | Data handling |
| SciPy | ≥ 1.10 | Statistical tests, Lomb–Scargle |
| matplotlib | ≥ 3.7 | Visualizations |
| scikit-learn | ≥ 1.2 | CRS-AI Random Forest |
| CosinorPy | ≥ 1.1 | Cosinor rhythmometry |

---

<p align="center">
  Developed for the circadian biology community · Francisco Tassara
</p>
