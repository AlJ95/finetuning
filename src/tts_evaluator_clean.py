"""
TTS-spezifische Evaluation-Metriken für deutsche Sprache.

Implementiert MOS-Approximation, Phonem-Accuracy, Audio-Qualitäts-Metriken
und Report-Generierung für TTS-Modell-Evaluation mit erweiterten Funktionalitäten.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime
import logging

# Audio quality metrics
from pesq import pesq
from pystoi import stoi
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

# Advanced evaluation libraries
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    
try:
    from utmos22 import UTMOS22
    UTMOS_AVAILABLE = True
except ImportError:
    UTMOS_AVAILABLE = False
    
try:
    import speechmetrics
    SPEECHMETRICS_AVAILABLE = True
except ImportError:
    SPEECHMETRICS_AVAILABLE = False

# German phoneme mapping
GERMAN_PHONEMES = {
    'a': ['a', 'ah', 'aa'],
    'e': ['e', 'eh', 'ee'],
    'i': ['i', 'ih', 'ii'],
    'o': ['o', 'oh', 'oo'],
    'u': ['u', 'uh', 'uu'],
    'ä': ['ae', 'aeh'],
    'ö': ['oe', 'oeh'],
    'ü': ['ue', 'ueh'],
    'ß': ['ss'],
    'ch': ['ch', 'x'],
    'sch': ['sh'],
    'th': ['t'],
    'ph': ['f'],
    'ck': ['k'],
    'ng': ['ng'],
    'nk': ['nk']
}


@dataclass
class AudioQualityMetrics:
    """Audio-Qualitäts-Metriken für TTS-Evaluation."""
    pesq_score: float
    stoi_score: float
    mel_spectral_distance: float
    signal_to_noise_ratio: float
    spectral_centroid_mean: float
    spectral_rolloff_mean: float
    zero_crossing_rate: float
    mfcc_similarity: float


@dataclass
class PhonemeAccuracyMetrics:
    """Deutsche Phonem-Accuracy Metriken."""
    overall_accuracy: float
    vowel_accuracy: float
    consonant_accuracy: float
    umlaut_accuracy: float
    phoneme_confusion_matrix: Dict[str, Dict[str, int]]
    problematic_phonemes: List[str]


@dataclass
class MOSApproximation:
    """MOS (Mean Opinion Score) Approximation."""
    predicted_mos: float
    confidence_interval: Tuple[float, float]
    quality_category: str  # Excellent, Good, Fair, Poor, Bad
    contributing_factors: Dict[str, float]


@dataclass
class TTSEvaluationResults:
    """Vollständige TTS-Evaluation-Ergebnisse."""
    timestamp: str
    model_name: str
    dataset_name: str
    audio_quality: AudioQualityMetrics
    phoneme_accuracy: PhonemeAccuracyMetrics
    mos_approximation: MOSApproximation
    inference_time_ms: float
    audio_duration_s: float
    text_length: int
    overall_score: float


class TTSEvaluator:
    """
    TTS-spezifische Evaluation-Metriken für deutsche Sprache.
    
    Implementiert MOS-Approximation, Phonem-Accuracy-Messung,
    Audio-Qualitäts-Metriken und Report-Generierung mit erweiterten Funktionalitäten.
    """
    
    def __init__(self, sample_rate: int = 16000, log_level: str = "INFO", use_advanced_metrics: bool = True):
        """
        Initialisiert den TTS-Evaluator.
        
        Args:
            sample_rate: Standard-Sampling-Rate für Audio-Verarbeitung
            log_level: Logging-Level
            use_advanced_metrics: Ob erweiterte Metriken (WhisperX, UTMOS, etc.) verwendet werden sollen
        """
        self.sample_rate = sample_rate
        self.use_advanced_metrics = use_advanced_metrics
        self.logger = self._setup_logging(log_level)
        
        # MOS-Approximation Gewichtungen (basierend auf TTS-Forschung)
        self.mos_weights = {
            'pesq': 0.3,
            'stoi': 0.25,
            'spectral_quality': 0.2,
            'naturalness': 0.15,
            'pronunciation': 0.1
        }
        
        # Qualitäts-Kategorien für MOS
        self.mos_categories = {
            (4.0, 5.0): "Excellent",
            (3.0, 4.0): "Good", 
            (2.0, 3.0): "Fair",
            (1.0, 2.0): "Poor",
            (0.0, 1.0): "Bad"
        }
        
        # Initialisiere erweiterte Modelle
        self._whisperx_model = None
        self._whisperx_align_model = None
        self._whisperx_metadata = None
        self._utmos_model = None
        self._init_advanced_models()
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging für TTS-Evaluator."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_advanced_models(self):
        """Initialisiert erweiterte Modelle für bessere Evaluation."""
        if not self.use_advanced_metrics:
            return
            
        # WhisperX Model initialisieren (Lazy Loading)
        if WHISPERX_AVAILABLE:
            try:
                self.logger.info("WhisperX verfügbar für Phonem-Alignment")
            except Exception as e:
                self.logger.warning(f"WhisperX-Initialisierung fehlgeschlagen: {e}")
                
        # UTMOS Model initialisieren (Lazy Loading)
        if UTMOS_AVAILABLE:
            try:
                self.logger.info("UTMOS verfügbar für MOS-Vorhersage")
            except Exception as e:
                self.logger.warning(f"UTMOS-Initialisierung fehlgeschlagen: {e}")

    def _load_whisperx_models(self):
        """Lädt WhisperX-Modelle bei Bedarf."""
        if not WHISPERX_AVAILABLE or not self.use_advanced_metrics:
            return False
            
        try:
            if self._whisperx_model is None:
                self.logger.info("Lade WhisperX-Modell...")
                device = "cpu"  # Verwende CPU für Kompatibilität
                self._whisperx_model = whisperx.load_model("large-v2", device=device, language="de")
                
            if self._whisperx_align_model is None:
                self.logger.info("Lade WhisperX-Alignment-Modell...")
                self._whisperx_align_model, self._whisperx_metadata = whisperx.load_align_model(
                    language_code="de", device="cpu"
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der WhisperX-Modelle: {e}")
            return False

    def _load_utmos_model(self):
        """Lädt UTMOS-Modell bei Bedarf."""
        if not UTMOS_AVAILABLE or not self.use_advanced_metrics:
            return False
            
        try:
            if self._utmos_model is None:
                self.logger.info("Lade UTMOS-Modell...")
                self._utmos_model = UTMOS22()
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des UTMOS-Modells: {e}")
            return False