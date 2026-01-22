"""
Generate synthetic circadian data with multiple variables, periods, and components
for testing CircaScope CosinorPy methods.
"""

import numpy as np
import pandas as pd
from typing import Optional

def generate_cosinor_signal(
    time: np.ndarray,
    period: float,
    n_components: int,
    amplitude: float,
    phase: float,
    mesor: float,
    noise_level: float = 0.2
) -> np.ndarray:
    """
    Generate a cosinor signal with multiple harmonic components.

    Args:
        time: Time points
        period: Main period (hours)
        n_components: Number of harmonic components
        amplitude: Amplitude of main component
        phase: Phase shift (radians)
        mesor: MESOR (Midline Estimating Statistic Of Rhythm)
        noise_level: Standard deviation of Gaussian noise

    Returns:
        Signal values
    """
    signal = np.zeros_like(time)

    for k in range(1, n_components + 1):
        # Each harmonic component with decreasing amplitude
        amp_k = amplitude / k
        signal += amp_k * np.cos(2 * np.pi * k * time / period + phase)

    signal += mesor

    # Add Gaussian noise
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, size=len(time))

    return signal


def create_synthetic_dataset(
    time_points: int = 48,
    time_step: float = 2.0,
    n_replicates: int = 3,
    data_type: str = 'independent',  # 'independent' or 'dependent'
    n_subjects: int = 5,  # Only used for dependent data
    seed: int = 42
) -> pd.DataFrame:
    """
    Create a synthetic dataset with multiple variables showing different rhythmic patterns.

    Variables:
    - circadian_pure: Clean 24h rhythm, 1 component
    - circadian_noisy: 24h rhythm with more noise, 1 component
    - ultradian_12h: 12h rhythm, 1 component
    - ultradian_8h: 8h rhythm, 1 component
    - infradian_48h: 48h rhythm, 1 component
    - multi_harmonic_2: 24h rhythm with 2 components
    - multi_harmonic_3: 24h rhythm with 3 components
    - arrhythmic: No rhythm (just noise)

    Args:
        data_type: 'independent' (with replicates) or 'dependent' (with subjects)
        n_subjects: Number of subjects (for dependent data)

    Returns:
        DataFrame with columns: time, condition, replicate/subject, var1, var2, ...
    """
    np.random.seed(seed)

    # Create time points
    time = np.arange(0, time_points * time_step, time_step)

    data_rows = []

    # Two conditions: control and treatment
    conditions = ['control', 'treatment']

    for condition in conditions:
        # Phase shift for treatment
        phase_shift = 0 if condition == 'control' else np.pi / 4

        if data_type == 'independent':
            # Independent data: multiple replicates
            for rep in range(1, n_replicates + 1):
                for t in time:
                    row = {
                        'time': t,
                        'condition': condition,
                        'replicate': f'rep{rep}',
                    }
                    row.update(_generate_variables(t, phase_shift))
                    data_rows.append(row)
        else:
            # Dependent data: multiple subjects
            for subject_id in range(1, n_subjects + 1):
                # Add inter-subject variability
                subject_mesor_shift = np.random.normal(0, 1.0)
                subject_amplitude_factor = np.random.uniform(0.8, 1.2)

                for t in time:
                    row = {
                        'time': t,
                        'condition': condition,
                        'subject': f'subject{subject_id}',
                    }
                    row.update(_generate_variables(
                        t, phase_shift,
                        mesor_shift=subject_mesor_shift,
                        amplitude_factor=subject_amplitude_factor
                    ))
                    data_rows.append(row)

    df = pd.DataFrame(data_rows)
    return df


def _generate_variables(
    t: float,
    phase_shift: float,
    mesor_shift: float = 0.0,
    amplitude_factor: float = 1.0
) -> dict:
    """Generate all variables for a single time point."""
    row = {}

    # Variable 1: Pure circadian (24h, 1 component)
    row['circadian_pure'] = generate_cosinor_signal(
        np.array([t]), period=24, n_components=1,
        amplitude=2.0 * amplitude_factor, phase=phase_shift,
        mesor=10.0 + mesor_shift, noise_level=0.3
    )[0]

    # Variable 2: Noisy circadian (24h, 1 component, more noise)
    row['circadian_noisy'] = generate_cosinor_signal(
        np.array([t]), period=24, n_components=1,
        amplitude=1.5 * amplitude_factor, phase=phase_shift + np.pi/6,
        mesor=8.0 + mesor_shift, noise_level=0.8
    )[0]

    # Variable 3: Ultradian 12h rhythm
    row['ultradian_12h'] = generate_cosinor_signal(
        np.array([t]), period=12, n_components=1,
        amplitude=1.8 * amplitude_factor, phase=phase_shift,
        mesor=15.0 + mesor_shift, noise_level=0.4
    )[0]

    # Variable 4: Ultradian 8h rhythm
    row['ultradian_8h'] = generate_cosinor_signal(
        np.array([t]), period=8, n_components=1,
        amplitude=1.2 * amplitude_factor, phase=phase_shift,
        mesor=12.0 + mesor_shift, noise_level=0.5
    )[0]

    # Variable 5: Infradian 48h rhythm
    row['infradian_48h'] = generate_cosinor_signal(
        np.array([t]), period=48, n_components=1,
        amplitude=2.5 * amplitude_factor, phase=phase_shift,
        mesor=20.0 + mesor_shift, noise_level=0.4
    )[0]

    # Variable 6: Multi-harmonic with 2 components
    row['multi_harmonic_2'] = generate_cosinor_signal(
        np.array([t]), period=24, n_components=2,
        amplitude=2.0 * amplitude_factor, phase=phase_shift,
        mesor=18.0 + mesor_shift, noise_level=0.5
    )[0]

    # Variable 7: Multi-harmonic with 3 components
    row['multi_harmonic_3'] = generate_cosinor_signal(
        np.array([t]), period=24, n_components=3,
        amplitude=2.5 * amplitude_factor, phase=phase_shift,
        mesor=22.0 + mesor_shift, noise_level=0.6
    )[0]

    # Variable 8: Arrhythmic (pure noise)
    row['arrhythmic'] = np.random.normal(14.0 + mesor_shift, 1.0)

    return row


if __name__ == '__main__':
    # Generate INDEPENDENT dataset
    print("=" * 70)
    print("Generating INDEPENDENT synthetic circadian dataset...")
    print("=" * 70)
    df_independent = create_synthetic_dataset(
        time_points=48,      # 48 time points
        time_step=2.0,       # Every 2 hours
        n_replicates=3,      # 3 biological replicates
        data_type='independent',
        seed=42
    )

    # Save to CSV
    output_path_indep = 'examples/synthetic_rhythms_test_data.csv'
    df_independent.to_csv(output_path_indep, index=False)

    print(f"[OK] Independent dataset saved to: {output_path_indep}")
    print(f"  Total rows: {len(df_independent)}")
    print(f"  Time points: {df_independent['time'].nunique()}")
    print(f"  Conditions: {df_independent['condition'].unique().tolist()}")
    print(f"  Replicates: {df_independent['replicate'].nunique()}")
    print(f"  Variables: {[col for col in df_independent.columns if col not in ['time', 'condition', 'replicate']]}")

    # Generate DEPENDENT dataset
    print("\n" + "=" * 70)
    print("Generating DEPENDENT/POPULATION synthetic circadian dataset...")
    print("=" * 70)
    df_dependent = create_synthetic_dataset(
        time_points=48,      # 48 time points
        time_step=2.0,       # Every 2 hours
        n_subjects=5,        # 5 subjects per condition
        data_type='dependent',
        seed=123
    )

    # Save to CSV
    output_path_dep = 'examples/synthetic_rhythms_dependent_test_data.csv'
    df_dependent.to_csv(output_path_dep, index=False)

    print(f"[OK] Dependent dataset saved to: {output_path_dep}")
    print(f"  Total rows: {len(df_dependent)}")
    print(f"  Time points: {df_dependent['time'].nunique()}")
    print(f"  Conditions: {df_dependent['condition'].unique().tolist()}")
    print(f"  Subjects: {df_dependent['subject'].nunique()}")
    print(f"  Variables: {[col for col in df_dependent.columns if col not in ['time', 'condition', 'subject']]}")

    print("\n" + "=" * 70)
    print("Variables description (both datasets):")
    print("=" * 70)
    print("  - circadian_pure: Clean 24h rhythm (1 component)")
    print("  - circadian_noisy: Noisy 24h rhythm (1 component)")
    print("  - ultradian_12h: 12h rhythm (1 component)")
    print("  - ultradian_8h: 8h rhythm (1 component)")
    print("  - infradian_48h: 48h rhythm (1 component)")
    print("  - multi_harmonic_2: 24h rhythm (2 components)")
    print("  - multi_harmonic_3: 24h rhythm (3 components)")
    print("  - arrhythmic: No rhythm (pure noise)")

    print("\n" + "=" * 70)
    print("Independent dataset preview:")
    print("=" * 70)
    print(df_independent.head(10))

    print("\n" + "=" * 70)
    print("Dependent dataset preview:")
    print("=" * 70)
    print(df_dependent.head(10))
