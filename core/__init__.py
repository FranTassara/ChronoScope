"""
CircaScope Core Analysis Modules
================================

This package contains the analysis engines for circadian rhythm analysis:
- cosinor_analysis: CosinorPy wrapper for cosinor-based rhythmometry
- circacompare_analysis: CircaCompare implementation for differential rhythmicity
- rhythm_analysis: Additional methods (JTK, Lomb-Scargle, CWT, etc.)
- analysis_engine: Unified interface for all analysis methods
"""

from .cosinor_analysis import (
    CosinorAnalyzer,
    CosinorParameters,
    DataType,
    ModelType,
    AnalysisMethod,
    Criterium,
    COSINORPY_AVAILABLE
)

from .circacompare_analysis import (
    CircaCompareAnalyzer,
    CircaSingleResult,
    CircaCompareResult
)

from .rhythm_analysis import (
    RhythmAnalyzer,
    JTKResult,
    CosinorResult,
    HarmonicCosinorResult,
    FourierF24Result,
    LombScargleResult,
    CWTResult,
    LMEResult,
    AnalysisMethod,
    DefaultPeriodRanges
)

from .analysis_engine import (
    AnalysisEngine,
    AnalysisType,
    AnalysisResult,
    ComparisonResult
)

from .meta_classifier import ConsensusClassifier, SKLEARN_AVAILABLE
from .feature_extraction import extract_features, FEATURE_NAMES

__all__ = [
    # Analysis Engine (main interface)
    'AnalysisEngine',
    'AnalysisType',
    'AnalysisResult',
    'ComparisonResult',
    # CosinorPy module (refactored)
    'CosinorAnalyzer',
    'CosinorParameters',
    'DataType',
    'ModelType',
    'AnalysisMethod',
    'Criterium',
    'COSINORPY_AVAILABLE',
    # CircaCompare module
    'CircaCompareAnalyzer',
    'CircaSingleResult',
    'CircaCompareResult',
    # Rhythm Analysis module
    'RhythmAnalyzer',
    'JTKResult',
    'CosinorResult',
    'HarmonicCosinorResult',
    'FourierF24Result',
    'LombScargleResult',
    'CWTResult',
    'LMEResult',
    'DefaultPeriodRanges',
    # AI Meta-Classifier
    'ConsensusClassifier',
    'SKLEARN_AVAILABLE',
    'extract_features',
    'FEATURE_NAMES',
]
