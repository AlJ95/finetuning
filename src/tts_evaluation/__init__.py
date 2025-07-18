"""
TTS-Evaluation-Paket für deutsche Sprache.

Dieses Paket bietet umfassende Evaluation-Metriken für Text-to-Speech-Modelle
mit Fokus auf deutsche Sprache und erweiterte Funktionalitäten.
"""

from .core import (
    TTSEvaluator,
    AudioQualityMetrics,
    PhonemeAccuracyMetrics,
    MOSApproximation,
    TTSEvaluationResults
)

from .utils import (
    load_audio_file,
    evaluate_audio_files
)

__version__ = "1.0.0"

__all__ = [
    "TTSEvaluator",
    "AudioQualityMetrics", 
    "PhonemeAccuracyMetrics",
    "MOSApproximation",
    "TTSEvaluationResults",
    "load_audio_file",
    "evaluate_audio_files"
]