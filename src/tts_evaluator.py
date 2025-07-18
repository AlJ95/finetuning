"""
TTS-spezifische Evaluation-Metriken für deutsche Sprache.

DEPRECATED: Diese Datei ist veraltet. Verwenden Sie stattdessen das modulare
src.tts_evaluation Paket für bessere Wartbarkeit und erweiterte Funktionalitäten.

Backward-Compatibility-Layer für bestehenden Code.
"""

# Backward Compatibility - Import aus dem neuen modularen System
from src.tts_evaluation import (
    TTSEvaluator,
    AudioQualityMetrics,
    PhonemeAccuracyMetrics,
    MOSApproximation,
    TTSEvaluationResults,
    load_audio_file,
    evaluate_audio_files
)

# Für Kompatibilität mit alten Imports
from src.tts_evaluation.phoneme_analyzer import GERMAN_PHONEMES
from src.tts_evaluation.phoneme_analyzer import WHISPERX_AVAILABLE
from src.tts_evaluation.mos_predictor import UTMOS_AVAILABLE, SPEECHMETRICS_AVAILABLE

# Warnung für Benutzer
import warnings
warnings.warn(
    "src.tts_evaluator ist veraltet. Verwenden Sie 'from src.tts_evaluation import TTSEvaluator' für neue Projekte.",
    DeprecationWarning,
    stacklevel=2
)