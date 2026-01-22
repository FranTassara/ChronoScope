SYNTHETIC RHYTHMS TEST DATASET
================================

File: synthetic_rhythms_test_data.csv
Generated: 2024 using generate_synthetic_data.py

OVERVIEW
--------
This dataset contains synthetic circadian and ultradian rhythms designed to test
CircaScope's CosinorPy analysis methods. Each variable has known rhythmic parameters,
making it ideal for validating analysis results.

DATASET STRUCTURE
-----------------
- Time points: 48 (0-94 hours, every 2 hours)
- Conditions: 2 (control, treatment)
- Replicates: 3 biological replicates per condition
- Total rows: 288
- Variables: 8 different rhythmic patterns

COLUMNS
-------
- time: Time in hours (0, 2, 4, ..., 94)
- condition: "control" or "treatment"
- replicate: "rep1", "rep2", or "rep3"
- [8 rhythmic variables - see below]

VARIABLES AND EXPECTED PARAMETERS
----------------------------------

1. circadian_pure
   Period: 24 hours
   Components: 1
   Amplitude: ~2.0
   Mesor: ~10.0
   Noise: Low (0.3)
   Description: Clean circadian rhythm with minimal noise
   Use case: Testing basic cosinor analysis with high signal-to-noise ratio

2. circadian_noisy
   Period: 24 hours
   Components: 1
   Amplitude: ~1.5
   Mesor: ~8.0
   Noise: High (0.8)
   Description: Noisy circadian rhythm
   Use case: Testing robustness of analysis methods to noise

3. ultradian_12h
   Period: 12 hours
   Components: 1
   Amplitude: ~1.8
   Mesor: ~15.0
   Noise: Medium (0.4)
   Description: Ultradian rhythm with 12-hour period
   Use case: Testing period detection and multi-period analysis

4. ultradian_8h
   Period: 8 hours
   Components: 1
   Amplitude: ~1.2
   Mesor: ~12.0
   Noise: Medium (0.5)
   Description: Faster ultradian rhythm with 8-hour period
   Use case: Testing analysis of shorter periods

5. infradian_48h
   Period: 48 hours
   Components: 1
   Amplitude: ~2.5
   Mesor: ~20.0
   Noise: Medium (0.4)
   Description: Infradian rhythm with 48-hour period
   Use case: Testing detection of longer periods

6. multi_harmonic_2
   Period: 24 hours (main)
   Components: 2
   Amplitude: ~2.0 (first), ~1.0 (second)
   Mesor: ~18.0
   Noise: Medium (0.5)
   Description: Circadian rhythm with 2 harmonic components (24h + 12h)
   Use case: Testing multi-component cosinor analysis

7. multi_harmonic_3
   Period: 24 hours (main)
   Components: 3
   Amplitude: ~2.5 (first), ~1.25 (second), ~0.83 (third)
   Mesor: ~22.0
   Noise: Medium-high (0.6)
   Description: Complex rhythm with 3 harmonic components (24h + 12h + 8h)
   Use case: Testing multi-component model selection (RSS/AIC/BIC)

8. arrhythmic
   Period: None
   Components: 0
   Amplitude: 0
   Mesor: ~14.0
   Noise: Pure noise (1.0)
   Description: Non-rhythmic variable (negative control)
   Use case: Testing that analysis correctly identifies lack of rhythm

CONDITION DIFFERENCES
---------------------
Treatment condition has a phase shift of π/4 radians (~6 hours) compared to control.
This is useful for testing:
- Comparison methods (Compare Conditions Independent/Dependent)
- Phase difference detection
- Acrophase comparison

RECOMMENDED TESTS
-----------------

Test 1: PERIODOGRAM ANALYSIS
- Variables: All
- Expected results:
  * circadian_pure: Clear peak at 24h
  * circadian_noisy: Peak at 24h (may be less prominent)
  * ultradian_12h: Clear peak at 12h
  * ultradian_8h: Clear peak at 8h
  * infradian_48h: Peak at 48h
  * multi_harmonic_2: Peaks at 24h and 12h
  * multi_harmonic_3: Peaks at 24h, 12h, and 8h
  * arrhythmic: No clear peaks

Test 2: COSINOR (Independent Data)
- Variables: circadian_pure, ultradian_12h
- Period: Auto-detect or specify (24h, 12h)
- Components: [1] or [1,2,3] for model selection
- Expected: Significant p-values, good fit (high R²)

Test 3: COSINOR (Dependent Data)
- Variables: circadian_pure, multi_harmonic_2
- Period: 24h
- Components: 1 for circadian_pure, [1,2,3] for multi_harmonic_2
- Expected: ME and resid_SE values available

Test 4: COMPARE CONDITIONS
- Variables: circadian_pure
- Conditions: control vs treatment
- Expected: Significant acrophase difference (~6 hours phase shift)
- Expected: No significant amplitude difference

Test 5: MULTI-COMPONENT ANALYSIS
- Variable: multi_harmonic_3
- Period: 24h
- Components: [1, 2, 3]
- Criterium: Try RSS, AIC, BIC
- Expected: Best model should be n_components=3
- Expected: AIC/BIC should select 3 components over 1 or 2

NOTES
-----
- All rhythms are generated using cosine functions with specified parameters
- Noise is Gaussian with specified standard deviation
- Treatment phase shift allows testing of comparison methods
- Dataset includes both positive controls (rhythmic) and negative control (arrhythmic)
- Multi-component variables allow testing of model selection criteria

USAGE EXAMPLE
-------------
1. Load dataset in CircaScope
2. Select "Periodogram Analysis" to identify dominant periods
3. Use "Cosinor (Independent Data)" with auto-period and auto-components
4. Compare control vs treatment conditions for circadian_pure
5. Test multi-component analysis on multi_harmonic_3

For questions or issues, see generate_synthetic_data.py for implementation details.
