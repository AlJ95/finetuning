"""
Kern-Klassen und Datenstrukturen für TTS-Evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Imports werden in den Methoden gemacht, um zirkuläre Imports zu vermeiden


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
    Hauptklasse für TTS-Evaluation mit erweiterten Funktionalitäten.
    
    Koordiniert die verschiedenen Evaluation-Module und bietet eine
    einheitliche API für die TTS-Modell-Bewertung.
    """
    
    def __init__(self, sample_rate: int = 16000, log_level: str = "INFO", use_advanced_metrics: bool = True):
        """
        Initialisiert den TTS-Evaluator.
        
        Args:
            sample_rate: Standard-Sampling-Rate für Audio-Verarbeitung
            log_level: Logging-Level
            use_advanced_metrics: Ob erweiterte Metriken verwendet werden sollen
        """
        self.sample_rate = sample_rate
        self.use_advanced_metrics = use_advanced_metrics
        self.logger = self._setup_logging(log_level)
        
        # Qualitäts-Kategorien für MOS
        self.mos_categories = {
            (4.0, 5.0): "Excellent",
            (3.0, 4.0): "Good", 
            (2.0, 3.0): "Fair",
            (1.0, 2.0): "Poor",
            (0.0, 1.0): "Bad"
        }
        
        # Initialisiere spezialisierte Module (Lazy Loading)
        self._phoneme_analyzer = None
        self._mos_predictor = None
        self._audio_metrics = None
    
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
    
    @property
    def phoneme_analyzer(self):
        """Lazy Loading für PhonemeAnalyzer."""
        if self._phoneme_analyzer is None:
            from .phoneme_analyzer import PhonemeAnalyzer
            self._phoneme_analyzer = PhonemeAnalyzer(
                sample_rate=self.sample_rate,
                use_advanced_metrics=self.use_advanced_metrics,
                logger=self.logger
            )
        return self._phoneme_analyzer
    
    @property
    def mos_predictor(self):
        """Lazy Loading für MOSPredictor."""
        if self._mos_predictor is None:
            from .mos_predictor import MOSPredictor
            self._mos_predictor = MOSPredictor(
                sample_rate=self.sample_rate,
                use_advanced_metrics=self.use_advanced_metrics,
                logger=self.logger
            )
        return self._mos_predictor
    
    @property
    def audio_metrics(self):
        """Lazy Loading für AudioMetricsCalculator."""
        if self._audio_metrics is None:
            from .audio_metrics import AudioMetricsCalculator
            self._audio_metrics = AudioMetricsCalculator(
                sample_rate=self.sample_rate,
                use_advanced_metrics=self.use_advanced_metrics,
                logger=self.logger
            )
        return self._audio_metrics
    
    def calculate_mos_approximation(
        self, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        text: Optional[str] = None
    ) -> MOSApproximation:
        """
        Berechnet MOS (Mean Opinion Score) Approximation.
        
        Args:
            generated_audio: Generiertes Audio-Signal
            reference_audio: Referenz-Audio (optional)
            text: Original-Text (optional)
            
        Returns:
            MOSApproximation mit Vorhersage und Konfidenzintervall
        """
        return self.mos_predictor.calculate_mos_approximation(
            generated_audio, reference_audio, text
        )
    
    def measure_phoneme_accuracy(
        self, 
        text: str, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> PhonemeAccuracyMetrics:
        """
        Misst deutsche Phonem-Accuracy.
        
        Args:
            text: Original-Text
            generated_audio: Generiertes Audio
            reference_audio: Referenz-Audio (optional)
            
        Returns:
            PhonemeAccuracyMetrics mit detaillierter Analyse
        """
        return self.phoneme_analyzer.measure_phoneme_accuracy(
            text, generated_audio, reference_audio
        )
    
    def calculate_audio_quality_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> AudioQualityMetrics:
        """
        Berechnet umfassende Audio-Qualitäts-Metriken.
        
        Args:
            generated_audio: Generiertes Audio-Signal
            reference_audio: Referenz-Audio (optional)
            
        Returns:
            AudioQualityMetrics mit allen Qualitäts-Metriken
        """
        return self.audio_metrics.calculate_audio_quality_metrics(
            generated_audio, reference_audio
        )
    
    def evaluate_tts_model(
        self,
        model_name: str,
        dataset_name: str,
        generated_audio: np.ndarray,
        text: str,
        reference_audio: Optional[np.ndarray] = None,
        inference_time_ms: float = 0.0
    ) -> TTSEvaluationResults:
        """
        Führt vollständige TTS-Modell-Evaluation durch.
        
        Args:
            model_name: Name des evaluierten Modells
            dataset_name: Name des verwendeten Datensatzes
            generated_audio: Generiertes Audio-Signal
            text: Original-Text
            reference_audio: Referenz-Audio (optional)
            inference_time_ms: Inferenz-Zeit in Millisekunden
            
        Returns:
            TTSEvaluationResults mit allen Metriken
        """
        self.logger.info(f"Starte vollständige TTS-Evaluation für Modell: {model_name}")
        
        # Audio-Qualitäts-Metriken
        audio_quality = self.calculate_audio_quality_metrics(generated_audio, reference_audio)
        
        # Phonem-Accuracy-Metriken
        phoneme_accuracy = self.measure_phoneme_accuracy(text, generated_audio, reference_audio)
        
        # MOS-Approximation
        mos_approximation = self.calculate_mos_approximation(generated_audio, reference_audio, text)
        
        # Audio-Eigenschaften
        audio_duration_s = len(generated_audio) / self.sample_rate
        text_length = len(text)
        
        # Overall-Score (gewichteter Durchschnitt der Hauptmetriken)
        overall_score = self._calculate_overall_score(
            audio_quality, phoneme_accuracy, mos_approximation
        )
        
        return TTSEvaluationResults(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            dataset_name=dataset_name,
            audio_quality=audio_quality,
            phoneme_accuracy=phoneme_accuracy,
            mos_approximation=mos_approximation,
            inference_time_ms=inference_time_ms,
            audio_duration_s=audio_duration_s,
            text_length=text_length,
            overall_score=overall_score
        )
    
    def _calculate_overall_score(
        self,
        audio_quality: AudioQualityMetrics,
        phoneme_accuracy: PhonemeAccuracyMetrics,
        mos_approximation: MOSApproximation
    ) -> float:
        """Berechnet Overall-Score aus allen Metriken."""
        # Gewichtungen für Overall-Score
        weights = {
            'mos': 0.4,
            'audio_quality': 0.3,
            'phoneme_accuracy': 0.3
        }
        
        # MOS-Score (bereits 1-5, normalisiert auf 0-1)
        mos_normalized = (mos_approximation.predicted_mos - 1.0) / 4.0
        
        # Audio-Quality-Score (Durchschnitt relevanter Metriken)
        audio_score = (
            (audio_quality.pesq_score / 4.5) * 0.3 +  # PESQ normalisiert
            audio_quality.stoi_score * 0.3 +  # STOI bereits 0-1
            (1.0 - min(audio_quality.mel_spectral_distance / 10.0, 1.0)) * 0.2 +  # Mel-Distanz invertiert
            min(audio_quality.signal_to_noise_ratio / 30.0, 1.0) * 0.2  # SNR normalisiert
        )
        
        # Phoneme-Accuracy-Score
        phoneme_score = phoneme_accuracy.overall_accuracy
        
        # Gewichteter Overall-Score
        overall = (
            weights['mos'] * mos_normalized +
            weights['audio_quality'] * audio_score +
            weights['phoneme_accuracy'] * phoneme_score
        )
        
        return max(0.0, min(1.0, overall))
    
    def generate_evaluation_report(
        self, 
        results: TTSEvaluationResults,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generiert detaillierten Evaluation-Report.
        
        Args:
            results: TTS-Evaluation-Ergebnisse
            output_path: Pfad für Report-Speicherung (optional)
            
        Returns:
            Formatierter Report als String
        """
        from .utils import generate_report
        return generate_report(results, output_path, self.logger)
    
    def save_results_json(self, results: TTSEvaluationResults, output_path: str) -> None:
        """Speichert Evaluation-Ergebnisse als JSON."""
        from .utils import save_results_json
        save_results_json(results, output_path, self.logger)