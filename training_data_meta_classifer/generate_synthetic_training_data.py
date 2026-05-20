"""
Generate Expanded Synthetic Training Data for Consensus Rhythmicity Score
==========================================================================

v1.2: Balanced 50/50 classes, outlier contamination, expanded arrhythmic diversity

Generates a diverse set of synthetic time series with known ground truth
labels (rhythmic / non-rhythmic) for training the meta-classifier.

Signal types:
  Rhythmic (10 types): clean cosine, medium/low SNR, damped, square, sawtooth,
    amplitude-modulated, multi-harmonic (2,3), borderline
  Non-rhythmic (36 types): white/pink/brown/bimodal noise, autocorrelated (7 rho),
    trends (linear+/-, quadratic), step, piecewise constant, exponential decay,
    non-circadian periods (17, including near-circadian 21/22/26/27h),
    fast-damped (2)

Outlier contamination: 15% of all instances receive 1-2 random spike outliers
to teach the model robustness to data artifacts (e.g. sensor errors, pipette
failures). This forces the model to learn that Cosinor OLS R² degrades under
outliers while JTK (rank-based) remains robust.

Author: Francisco Tassara
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.signal import square, sawtooth


# Fraction of instances to contaminate with spike outliers
OUTLIER_FRACTION = 0.15


# =============================================================================
# SIGNAL GENERATORS - RHYTHMIC
# =============================================================================

def _cosine_signal(time: np.ndarray, period: float, amplitude: float,
                   phase: float, mesor: float) -> np.ndarray:
    """Pure cosine signal."""
    return mesor + amplitude * np.cos(2 * np.pi * time / period - phase)


def _damped_signal(time: np.ndarray, period: float, amplitude: float,
                   phase: float, mesor: float, decay_rate: float = 0.02) -> np.ndarray:
    """Damped oscillation: amplitude decays exponentially."""
    envelope = np.exp(-decay_rate * time)
    return mesor + amplitude * envelope * np.cos(2 * np.pi * time / period - phase)


def _square_wave_signal(time: np.ndarray, period: float, amplitude: float,
                        phase: float, mesor: float) -> np.ndarray:
    """Square wave approximation."""
    return mesor + amplitude * square(2 * np.pi * time / period - phase)


def _sawtooth_signal(time: np.ndarray, period: float, amplitude: float,
                     phase: float, mesor: float) -> np.ndarray:
    """Sawtooth wave."""
    return mesor + amplitude * sawtooth(2 * np.pi * time / period - phase)


def _amplitude_modulated_signal(time: np.ndarray, period: float, amplitude: float,
                                phase: float, mesor: float,
                                mod_period: float = 120.0) -> np.ndarray:
    """Cosine with slow amplitude modulation envelope."""
    envelope = 0.5 + 0.5 * np.cos(2 * np.pi * time / mod_period)
    return mesor + amplitude * envelope * np.cos(2 * np.pi * time / period - phase)


def _multi_harmonic_signal(time: np.ndarray, period: float, amplitude: float,
                           phase: float, mesor: float, n_harmonics: int = 2) -> np.ndarray:
    """Multi-harmonic cosine (decreasing amplitude per harmonic)."""
    signal = np.full_like(time, mesor, dtype=float)
    for k in range(1, n_harmonics + 1):
        amp_k = amplitude / k
        signal += amp_k * np.cos(2 * np.pi * k * time / period - phase)
    return signal


# =============================================================================
# SIGNAL GENERATORS - NON-RHYTHMIC
# =============================================================================

def _white_noise(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """Pure white noise around mesor."""
    return mesor + np.random.normal(0, noise_std, size=len(time))


def _pink_noise(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """1/f pink noise around mesor (spectral method)."""
    n = len(time)
    white = np.fft.rfft(np.random.normal(0, 1, n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0  # avoid div by zero
    pink_filter = 1.0 / np.sqrt(np.maximum(freqs, 1e-10))
    pink_spectrum = white * pink_filter
    pink = np.fft.irfft(pink_spectrum, n=n)
    std = np.std(pink)
    if std > 0:
        pink = pink / std * noise_std
    return mesor + pink


def _brown_noise(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """Brownian motion / random walk."""
    n = len(time)
    increments = np.random.normal(0, 1, n)
    brown = np.cumsum(increments)
    brown = brown - np.mean(brown)
    std = np.std(brown)
    if std > 0:
        brown = brown / std * noise_std
    return mesor + brown


def _bimodal_noise(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """Bimodal noise: mixture of two Gaussians with separated means."""
    n = len(time)
    component = np.random.binomial(1, 0.5, n)
    signal = np.where(
        component == 0,
        np.random.normal(-1.5, noise_std * 0.5, n),
        np.random.normal(1.5, noise_std * 0.5, n)
    )
    return mesor + signal


def _autocorrelated_noise(time: np.ndarray, mesor: float, noise_std: float,
                          rho: float = 0.5) -> np.ndarray:
    """AR(1) autocorrelated noise (can look rhythmic but isn't)."""
    n = len(time)
    signal = np.zeros(n)
    signal[0] = np.random.normal(0, noise_std)
    for i in range(1, n):
        signal[i] = rho * signal[i - 1] + np.random.normal(0, noise_std * np.sqrt(1 - rho ** 2))
    return mesor + signal


def _trend_noise(time: np.ndarray, mesor: float, noise_std: float,
                 slope: float = 0.05) -> np.ndarray:
    """Linear trend plus noise."""
    trend = slope * (time - time[0])
    return mesor + trend + np.random.normal(0, noise_std, size=len(time))


def _quadratic_trend(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """Quadratic (parabolic) trend plus noise."""
    t_span = max(time[-1] - time[0], 1.0)
    t_norm = (time - time[0]) / t_span  # normalize to [0, 1]
    trend = 4.0 * (t_norm - 0.5) ** 2  # U-shaped, range [0, 1]
    return mesor + trend * 3.0 + np.random.normal(0, noise_std, len(time))


def _step_signal(time: np.ndarray, mesor: float, noise_std: float,
                 step_time: float = None) -> np.ndarray:
    """Step function (not oscillatory)."""
    if step_time is None:
        step_time = (time[-1] - time[0]) / 2 + time[0]
    signal = np.where(time >= step_time, mesor + 2.0, mesor - 2.0)
    return signal + np.random.normal(0, noise_std, size=len(time))


def _piecewise_constant(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """Multiple constant levels (piecewise step function)."""
    n = len(time)
    n_segments = min(3, max(2, n // 2))
    signal = np.zeros(n)
    boundaries = np.linspace(0, n, n_segments + 1, dtype=int)
    for seg in range(n_segments):
        start, end = boundaries[seg], boundaries[seg + 1]
        level = np.random.uniform(-2, 2)
        signal[start:end] = level
    return mesor + signal + np.random.normal(0, noise_std, n)


def _exponential_decay_monotonic(time: np.ndarray, mesor: float,
                                 noise_std: float) -> np.ndarray:
    """Monotonic exponential decay (no oscillation)."""
    initial_elevation = 5.0
    decay_rate = 0.05
    signal = mesor + initial_elevation * np.exp(-decay_rate * (time - time[0]))
    return signal + np.random.normal(0, noise_std, len(time))


# =============================================================================
# OUTLIER INJECTION
# =============================================================================

def _inject_outliers(values: np.ndarray, n_outliers: int = None,
                     magnitude_range: Tuple[float, float] = (3.0, 5.0)) -> np.ndarray:
    """
    Inject random spike outliers into a time series.

    Simulates experimental artifacts: sensor spikes, pipette failures, etc.
    Spikes are 3-5 standard deviations from the signal, randomly positive
    or negative.

    Args:
        values: time series values
        n_outliers: number of outliers (default: random 1-2)
        magnitude_range: spike magnitude range in signal standard deviations

    Returns:
        contaminated values (copy)
    """
    contaminated = values.copy()

    if n_outliers is None:
        n_outliers = np.random.randint(1, 3)  # 1 or 2 outliers

    n = len(values)
    if n_outliers >= n:
        n_outliers = max(1, n // 3)

    outlier_indices = np.random.choice(n, size=n_outliers, replace=False)
    data_std = np.std(values)
    if data_std < 0.01:
        data_std = 1.0  # fallback for near-constant signals

    for idx in outlier_indices:
        spike_magnitude = np.random.uniform(*magnitude_range) * data_std
        spike_sign = np.random.choice([-1, 1])
        contaminated[idx] += spike_sign * spike_magnitude

    return contaminated


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_instances(seed: int = 42) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Generate a diverse set of synthetic time series instances for training.

    Returns:
        Tuple of:
        - List of metadata dicts (one per instance) with keys:
          'instance_id', 'signal_type', 'is_rhythmic', 'n_timepoints',
          'n_replicates', 'sampling_hours', 'snr', 'has_outliers'
        - DataFrame with all instances in ChronoScope format
          (columns: time, condition, replicate, value)
    """
    np.random.seed(seed)

    metadata_list = []
    all_dataframes = []
    instance_id = 0

    # Parameter sweeps
    timepoint_configs = [
        (6, 4.0),    # 6 timepoints, 4h sampling (20h span, ~1 day, Rosbash-like)
        (12, 4.0),   # 12 timepoints, 4h sampling (2 days)
        (24, 2.0),   # 24 timepoints, 2h sampling (2 days)
        (48, 2.0),   # 48 timepoints, 2h sampling (4 days)
    ]

    replicate_counts = [1, 3]

    # =========================================================================
    # RHYTHMIC SIGNALS (label = 1) — 800 instances
    # =========================================================================

    rhythmic_configs = [
        # (name, generator_func, extra_kwargs, amplitude, mesor, noise_stds)

        # Clean circadian
        ('circadian_clean', _cosine_signal, {},
         2.0, 10.0, [0.2, 0.5]),  # SNR ~10, ~4

        # Medium noise circadian
        ('circadian_medium', _cosine_signal, {},
         2.0, 10.0, [0.67, 1.0]),  # SNR ~3, ~2

        # Low SNR circadian
        ('circadian_low_snr', _cosine_signal, {},
         1.5, 10.0, [1.5, 2.0]),  # SNR ~1, ~0.75

        # Damped oscillation
        ('damped_oscillation', _damped_signal, {'decay_rate': 0.015},
         2.5, 10.0, [0.4, 0.8]),

        # Square wave
        ('square_wave', _square_wave_signal, {},
         1.5, 10.0, [0.5, 1.0]),

        # Sawtooth wave
        ('sawtooth_wave', _sawtooth_signal, {},
         1.5, 10.0, [0.5, 1.0]),

        # Amplitude modulated
        ('amplitude_modulated', _amplitude_modulated_signal, {'mod_period': 120.0},
         2.5, 12.0, [0.4, 0.8]),

        # Multi-harmonic 2
        ('multi_harmonic_2', _multi_harmonic_signal, {'n_harmonics': 2},
         2.0, 15.0, [0.4, 0.8]),

        # Multi-harmonic 3
        ('multi_harmonic_3', _multi_harmonic_signal, {'n_harmonics': 3},
         2.5, 18.0, [0.5, 1.0]),

        # Borderline: very low amplitude
        ('borderline_rhythm', _cosine_signal, {},
         0.5, 10.0, [0.4, 0.6]),
    ]

    # Periods to test for rhythmic signals
    circadian_periods = [23.0, 23.5, 24.0, 24.5, 25.0]

    for name, gen_func, extra_kwargs, amplitude, mesor, noise_stds in rhythmic_configs:
        for n_tp, sampling_h in timepoint_configs:
            for n_rep in replicate_counts:
                for noise_std in noise_stds:
                    for period in circadian_periods:
                        phase = np.random.uniform(0, 2 * np.pi)
                        time = np.arange(0, n_tp * sampling_h, sampling_h)

                        # Decide outlier contamination for this instance
                        has_outliers = np.random.random() < OUTLIER_FRACTION

                        rows = []
                        for rep in range(1, n_rep + 1):
                            signal = gen_func(time, period=period, amplitude=amplitude,
                                              phase=phase, mesor=mesor, **extra_kwargs)
                            noise = np.random.normal(0, noise_std, size=len(time))
                            values = signal + noise

                            if has_outliers:
                                values = _inject_outliers(values)

                            for i, t in enumerate(time):
                                rows.append({
                                    'time': t,
                                    'condition': 'control',
                                    'replicate': f'rep{rep}',
                                    f'var_{instance_id}': values[i]
                                })

                        df = pd.DataFrame(rows)
                        all_dataframes.append(df)

                        snr = amplitude / noise_std if noise_std > 0 else 999
                        metadata_list.append({
                            'instance_id': instance_id,
                            'variable': f'var_{instance_id}',
                            'signal_type': name,
                            'is_rhythmic': 1,
                            'n_timepoints': n_tp,
                            'n_replicates': n_rep,
                            'sampling_hours': sampling_h,
                            'snr': snr,
                            'period': period,
                            'has_outliers': has_outliers,
                        })
                        instance_id += 1

    # =========================================================================
    # NON-RHYTHMIC PURE NOISE (label = 0) — 440 instances
    # =========================================================================

    non_rhythmic_configs = [
        # (name, generator_func, extra_kwargs, noise_stds)

        # === Pure noise types ===
        ('white_noise', _white_noise, {},
         [0.2, 0.5, 1.0, 2.0, 5.0]),
        ('pink_noise', _pink_noise, {},
         [0.3, 0.5, 1.0, 2.0, 5.0]),
        ('brown_noise', _brown_noise, {},
         [0.5, 1.0, 2.0]),
        ('bimodal_noise', _bimodal_noise, {},
         [0.5, 1.0, 2.0]),

        # === Autocorrelated noise (various rho) ===
        ('autocorr_rho02', _autocorrelated_noise, {'rho': 0.2},
         [0.5, 1.0, 2.0]),
        ('autocorr_rho03', _autocorrelated_noise, {'rho': 0.3},
         [0.5, 1.0, 2.0]),
        ('autocorr_rho04', _autocorrelated_noise, {'rho': 0.4},
         [0.5, 1.0, 2.0]),
        ('autocorr_rho06', _autocorrelated_noise, {'rho': 0.6},
         [0.5, 1.0, 2.0]),
        ('autocorr_rho07', _autocorrelated_noise, {'rho': 0.7},
         [0.5, 1.0, 2.0]),
        ('autocorr_rho08', _autocorrelated_noise, {'rho': 0.8},
         [0.5, 1.0, 2.0]),
        ('autocorr_rho095', _autocorrelated_noise, {'rho': 0.95},
         [0.5, 1.0, 2.0]),

        # === Trends ===
        ('trend_linear_pos', _trend_noise, {'slope': 0.05},
         [0.3, 0.5, 1.0]),
        ('trend_linear_neg', _trend_noise, {'slope': -0.05},
         [0.3, 0.5, 1.0]),
        ('trend_quadratic', _quadratic_trend, {},
         [0.3, 0.5, 1.0]),

        # === Step / piecewise ===
        ('step_function', _step_signal, {},
         [0.3, 0.5, 1.0]),
        ('piecewise_constant', _piecewise_constant, {},
         [0.3, 0.5, 1.0]),

        # === Monotonic decay ===
        ('exponential_decay', _exponential_decay_monotonic, {},
         [0.3, 0.5, 1.0]),
    ]

    for name, gen_func, extra_kwargs, noise_stds in non_rhythmic_configs:
        for n_tp, sampling_h in timepoint_configs:
            for n_rep in replicate_counts:
                for noise_std in noise_stds:
                    time = np.arange(0, n_tp * sampling_h, sampling_h)
                    has_outliers = np.random.random() < OUTLIER_FRACTION

                    rows = []
                    for rep in range(1, n_rep + 1):
                        values = gen_func(time, mesor=10.0, noise_std=noise_std,
                                          **extra_kwargs)
                        if has_outliers:
                            values = _inject_outliers(values)

                        for i, t in enumerate(time):
                            rows.append({
                                'time': t,
                                'condition': 'control',
                                'replicate': f'rep{rep}',
                                f'var_{instance_id}': values[i]
                            })

                    df = pd.DataFrame(rows)
                    all_dataframes.append(df)

                    metadata_list.append({
                        'instance_id': instance_id,
                        'variable': f'var_{instance_id}',
                        'signal_type': name,
                        'is_rhythmic': 0,
                        'n_timepoints': n_tp,
                        'n_replicates': n_rep,
                        'sampling_hours': sampling_h,
                        'snr': 0.0,
                        'period': 0.0,
                        'has_outliers': has_outliers,
                    })
                    instance_id += 1

    # =========================================================================
    # NON-CIRCADIAN RHYTHMS (label = 0) — 312 instances
    # Oscillatory but at wrong periods (not 22-26h circadian window)
    # =========================================================================

    # NOTE: periods close to 24h (21, 22, 26, 27) are deliberately included as
    # NON-rhythmic. Rationale: prevents the model from learning a trivial gap
    # via period_dev_24h. In the synthetic v1.x set, rhythmic instances had
    # periods in [23, 25] while non-rhythmic oscillators were all >=4h away
    # from 24h. That made period_dev_24h a near-perfect discriminator on
    # synthetic data, which inflates CV and biases the model away from
    # learning the harder features (amplitude regularity, phase coherence).
    # In real biology, ~21h and ~27h oscillations are NOT circadian (e.g.,
    # tidal, infradian, or non-biological artifacts); the model should
    # learn that period proximity to 24h is necessary but not sufficient.
    non_circadian_periods = [
        4.0, 5.0, 6.0, 8.0, 10.0, 12.0,   # ultradian
        14.0, 16.0, 18.0, 20.0,             # sub-circadian
        21.0, 22.0,                          # near-circadian (NOT rhythmic)
        26.0, 27.0,                          # near-circadian (NOT rhythmic)
        30.0, 36.0, 48.0,                    # infradian
    ]
    non_circadian_noise_stds = [0.3, 0.6, 1.0]

    for period in non_circadian_periods:
        amplitude = 1.5 if period <= 12.0 else 2.0
        mesor = 10.0

        for noise_std in non_circadian_noise_stds:
            for n_tp, sampling_h in timepoint_configs:
                for n_rep in replicate_counts:
                    phase = np.random.uniform(0, 2 * np.pi)
                    time = np.arange(0, n_tp * sampling_h, sampling_h)
                    has_outliers = np.random.random() < OUTLIER_FRACTION

                    rows = []
                    for rep in range(1, n_rep + 1):
                        signal = _cosine_signal(time, period=period, amplitude=amplitude,
                                                phase=phase, mesor=mesor)
                        noise = np.random.normal(0, noise_std, size=len(time))
                        values = signal + noise

                        if has_outliers:
                            values = _inject_outliers(values)

                        for i, t in enumerate(time):
                            rows.append({
                                'time': t,
                                'condition': 'control',
                                'replicate': f'rep{rep}',
                                f'var_{instance_id}': values[i]
                            })

                    df = pd.DataFrame(rows)
                    all_dataframes.append(df)

                    metadata_list.append({
                        'instance_id': instance_id,
                        'variable': f'var_{instance_id}',
                        'signal_type': f'non_circadian_{period:.0f}h',
                        'is_rhythmic': 0,
                        'n_timepoints': n_tp,
                        'n_replicates': n_rep,
                        'sampling_hours': sampling_h,
                        'snr': amplitude / noise_std if noise_std > 0 else 999,
                        'period': period,
                        'has_outliers': has_outliers,
                    })
                    instance_id += 1

    # =========================================================================
    # FAST-DAMPED OSCILLATIONS (label = 0) — 48 instances
    # Start rhythmic-looking but die within 1-2 cycles (not sustained)
    # =========================================================================

    fast_damped_configs = [
        # (name, decay_rate, noise_stds)
        ('fast_damped_moderate', 0.15, [0.3, 0.5, 1.0]),
        ('fast_damped_extreme', 0.30, [0.3, 0.5, 1.0]),
    ]

    for name, decay_rate, noise_stds in fast_damped_configs:
        for noise_std in noise_stds:
            for n_tp, sampling_h in timepoint_configs:
                for n_rep in replicate_counts:
                    phase = np.random.uniform(0, 2 * np.pi)
                    time = np.arange(0, n_tp * sampling_h, sampling_h)
                    has_outliers = np.random.random() < OUTLIER_FRACTION

                    rows = []
                    for rep in range(1, n_rep + 1):
                        signal = _damped_signal(time, period=24.0, amplitude=2.5,
                                                phase=phase, mesor=10.0,
                                                decay_rate=decay_rate)
                        noise = np.random.normal(0, noise_std, size=len(time))
                        values = signal + noise

                        if has_outliers:
                            values = _inject_outliers(values)

                        for i, t in enumerate(time):
                            rows.append({
                                'time': t,
                                'condition': 'control',
                                'replicate': f'rep{rep}',
                                f'var_{instance_id}': values[i]
                            })

                    df = pd.DataFrame(rows)
                    all_dataframes.append(df)

                    metadata_list.append({
                        'instance_id': instance_id,
                        'variable': f'var_{instance_id}',
                        'signal_type': name,
                        'is_rhythmic': 0,
                        'n_timepoints': n_tp,
                        'n_replicates': n_rep,
                        'sampling_hours': sampling_h,
                        'snr': 2.5 / noise_std if noise_std > 0 else 999,
                        'period': 24.0,
                        'has_outliers': has_outliers,
                    })
                    instance_id += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================

    n_rhythmic = sum(1 for m in metadata_list if m['is_rhythmic'] == 1)
    n_non_rhythmic = sum(1 for m in metadata_list if m['is_rhythmic'] == 0)
    n_outliers = sum(1 for m in metadata_list if m['has_outliers'])

    print(f"Generated {instance_id} instances total")
    print(f"  Rhythmic:     {n_rhythmic}")
    print(f"  Non-rhythmic: {n_non_rhythmic}")
    print(f"  Balance:      {n_rhythmic / instance_id * 100:.1f}% / {n_non_rhythmic / instance_id * 100:.1f}%")
    print(f"  With outliers: {n_outliers} ({n_outliers / instance_id * 100:.1f}%)")

    return metadata_list, all_dataframes


if __name__ == '__main__':
    metadata, dataframes = generate_training_instances(seed=42)

    # Print summary
    from collections import Counter
    types = Counter(m['signal_type'] for m in metadata)
    print("\nSignal type counts:")
    for t, c in sorted(types.items()):
        label = "rhythmic" if any(m['is_rhythmic'] for m in metadata if m['signal_type'] == t) else "non-rhythmic"
        print(f"  {t}: {c} ({label})")

    # Outlier stats
    n_outlier_rhythmic = sum(1 for m in metadata if m['is_rhythmic'] == 1 and m['has_outliers'])
    n_outlier_non = sum(1 for m in metadata if m['is_rhythmic'] == 0 and m['has_outliers'])
    print(f"\nOutlier contamination:")
    print(f"  Rhythmic instances with outliers: {n_outlier_rhythmic}")
    print(f"  Non-rhythmic instances with outliers: {n_outlier_non}")
