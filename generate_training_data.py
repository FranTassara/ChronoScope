"""
Generate Expanded Synthetic Training Data for Consensus Rhythmicity Score
==========================================================================

Generates a diverse set of synthetic time series with known ground truth
labels (rhythmic / non-rhythmic) for training the meta-classifier.

Covers many scenarios that the original generate_synthetic_data.py does not:
- Damped oscillations
- Non-sinusoidal waveforms (square, sawtooth)
- Autocorrelated noise
- Variable SNR levels
- Different sampling densities
- Borderline cases

Author: Francisco Tassara
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from scipy.signal import square, sawtooth


# =============================================================================
# SIGNAL GENERATORS
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


def _white_noise(time: np.ndarray, mesor: float, noise_std: float) -> np.ndarray:
    """Pure white noise around mesor."""
    return mesor + np.random.normal(0, noise_std, size=len(time))


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


def _step_signal(time: np.ndarray, mesor: float, noise_std: float,
                 step_time: float = None) -> np.ndarray:
    """Step function (not oscillatory)."""
    if step_time is None:
        step_time = (time[-1] - time[0]) / 2 + time[0]
    signal = np.where(time >= step_time, mesor + 2.0, mesor - 2.0)
    return signal + np.random.normal(0, noise_std, size=len(time))


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
          'n_replicates', 'sampling_hours', 'snr'
        - DataFrame with all instances in CircaScope format
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
    # RHYTHMIC SIGNALS (label = 1)
    # =========================================================================

    rhythmic_configs = [
        # (name, generator_func, extra_kwargs, amplitude, mesor, noise_std_factor, periods)
        # noise_std_factor is relative to amplitude: noise_std = amplitude / snr

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
                        # Random phase
                        phase = np.random.uniform(0, 2 * np.pi)

                        time = np.arange(0, n_tp * sampling_h, sampling_h)
                        rows = []

                        for rep in range(1, n_rep + 1):
                            signal = gen_func(time, period=period, amplitude=amplitude,
                                              phase=phase, mesor=mesor, **extra_kwargs)
                            noise = np.random.normal(0, noise_std, size=len(time))
                            values = signal + noise

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
                        })
                        instance_id += 1

    # =========================================================================
    # NON-RHYTHMIC SIGNALS (label = 0)
    # =========================================================================

    non_rhythmic_configs = [
        # (name, generator_func, extra_kwargs, noise_stds)
        ('white_noise', _white_noise, {}, [0.5, 1.0, 2.0]),
        ('autocorr_rho03', _autocorrelated_noise, {'rho': 0.3}, [0.5, 1.0]),
        ('autocorr_rho06', _autocorrelated_noise, {'rho': 0.6}, [0.5, 1.0]),
        ('autocorr_rho08', _autocorrelated_noise, {'rho': 0.8}, [0.5, 1.0]),
        ('trend_linear', _trend_noise, {'slope': 0.05}, [0.5, 1.0]),
        ('step_function', _step_signal, {}, [0.5, 1.0]),
    ]

    for name, gen_func, extra_kwargs, noise_stds in non_rhythmic_configs:
        for n_tp, sampling_h in timepoint_configs:
            for n_rep in replicate_counts:
                for noise_std in noise_stds:
                    time = np.arange(0, n_tp * sampling_h, sampling_h)
                    rows = []

                    for rep in range(1, n_rep + 1):
                        values = gen_func(time, mesor=10.0, noise_std=noise_std,
                                          **extra_kwargs)
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
                    })
                    instance_id += 1

    # Non-circadian rhythms (rhythmic but NOT circadian -> label = 0)
    non_circadian_configs = [
        ('ultradian_8h', 8.0, 1.5, 10.0, [0.3, 0.6]),
        ('ultradian_12h', 12.0, 1.8, 12.0, [0.3, 0.6]),
        ('infradian_48h', 48.0, 2.0, 15.0, [0.4, 0.8]),
    ]

    for name, period, amplitude, mesor, noise_stds in non_circadian_configs:
        for n_tp, sampling_h in timepoint_configs:
            for n_rep in replicate_counts:
                for noise_std in noise_stds:
                    phase = np.random.uniform(0, 2 * np.pi)
                    time = np.arange(0, n_tp * sampling_h, sampling_h)
                    rows = []

                    for rep in range(1, n_rep + 1):
                        signal = _cosine_signal(time, period=period, amplitude=amplitude,
                                                phase=phase, mesor=mesor)
                        noise = np.random.normal(0, noise_std, size=len(time))
                        values = signal + noise

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
                        'snr': amplitude / noise_std if noise_std > 0 else 999,
                        'period': period,
                    })
                    instance_id += 1

    print(f"Generated {instance_id} instances total")
    print(f"  Rhythmic: {sum(1 for m in metadata_list if m['is_rhythmic'] == 1)}")
    print(f"  Non-rhythmic: {sum(1 for m in metadata_list if m['is_rhythmic'] == 0)}")

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
