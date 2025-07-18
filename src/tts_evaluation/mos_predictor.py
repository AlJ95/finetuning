"""
MOS-Vorhersage-Modul für TTS-Evaluation.

Implementiert erweiterte MOS-Approximation mit UTMOS und anderen
vortrainierten Modellen sowie Fallback auf heuristische Methoden.
"""

import numpy as np
from typing import Optional, Dict, Tuple
import logging

from .core import MOSApproximation

# Advanced evaluation libraries
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


class MOSPredictor:
    """
    Spezialisierte Klasse für MOS (Mean Opinion Score) Vorhersage.
    
    Verwendet UTMOS22 oder andere vortrainierte Modelle für präzise
    MOS-Vorhersagen oder fällt auf heuristische Methoden zurück.
    """
    
    def __init__(self, sample_rate: int = 16000, use_advanced_metrics: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialisiert den MOS-Predictor.
        
        Args:
            sample_rate: Audio-Sampling-Rate
            use_advanced_metrics: Ob erweiterte Metriken verwendet werden sollen
            logger: Logger-Instanz
        """
        self.sample_rate = sample_rate
        self.use_advanced_metrics = use_advanced_metrics
        self.logger = logger or logging.getLogger(__name__)
        
        # MOS-Approximation Gewichtungen (für Fallback)
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
        
        # UTMOS-Modell (Lazy Loading)
        self._utmos_model = None
    
    def calculate_mos_approximation(
        self, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        text: Optional[str] = None
    ) -> MOSApproximation:
        """
        Berechnet MOS (Mean Opinion Score) Approximation mit erweiterten Methoden.
        
        Args:
            generated_audio: Generiertes Audio-Signal
            reference_audio: Referenz-Audio (optional)
            text: Original-Text (optional)
            
        Returns:
            MOSApproximation mit Vorhersage und Konfidenzintervall
        """
        if self.use_advanced_metrics and (UTMOS_AVAILABLE or SPEECHMETRICS_AVAILABLE):
            return self._calculate_advanced_mos_approximation(generated_audio, reference_audio, text)
        else:
            return self._calculate_basic_mos_approximation(generated_audio, reference_audio, text)
    
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
    
    def _calculate_mos_with_utmos(self, audio: np.ndarray, text: str) -> float:
        """Verwendet UTMOS22 für präzise MOS-Vorhersage."""
        if not self._load_utmos_model():
            self.logger.warning("UTMOS nicht verfügbar, verwende Fallback-Methode")
            return 0.0
            
        try:
            # Audio für UTMOS vorbereiten
            if len(audio) == 0:
                return 2.5  # Fallback für leeres Audio
                
            # UTMOS erwartet Audio im Bereich [-1, 1]
            audio_normalized = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # MOS-Score berechnen
            mos_score = self._utmos_model.score(
                wav=audio_normalized, 
                sr=self.sample_rate,
                text=text
            )
            
            # Sicherstellen, dass Score im gültigen Bereich liegt
            return max(1.0, min(5.0, float(mos_score)))
            
        except Exception as e:
            self.logger.error(f"UTMOS-MOS-Berechnung fehlgeschlagen: {e}")
            return 2.5  # Fallback-Wert
    
    def _calculate_advanced_mos_approximation(
        self, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        text: Optional[str] = None
    ) -> MOSApproximation:
        """Berechnet erweiterte MOS-Approximation mit UTMOS."""
        self.logger.info("Berechne erweiterte MOS-Approximation mit UTMOS...")
        
        try:
            # UTMOS-basierte MOS-Vorhersage
            utmos_score = 0.0
            if text and self.use_advanced_metrics:
                utmos_score = self._calculate_mos_with_utmos(generated_audio, text)
            
            # Basis-Qualitäts-Metriken berechnen
            from .audio_metrics import AudioMetricsCalculator
            audio_metrics = AudioMetricsCalculator(
                sample_rate=self.sample_rate,
                use_advanced_metrics=False,  # Verwende Basic-Metriken für MOS-Komponenten
                logger=self.logger
            )
            quality_metrics = audio_metrics.calculate_audio_quality_metrics(generated_audio, reference_audio)
            
            # Erweiterte MOS-Komponenten
            mos_components = {
                'utmos': utmos_score / 5.0 if utmos_score > 0 else 0.0,  # Normalisiert auf 0-1
                'pesq': min(quality_metrics.pesq_score / 4.5, 1.0),
                'stoi': quality_metrics.stoi_score,
                'spectral_quality': self._calculate_spectral_quality(generated_audio),
                'naturalness': self._calculate_naturalness_score(generated_audio),
                'pronunciation': self._calculate_pronunciation_score(generated_audio, text) if text else 0.8
            }
            
            # Erweiterte Gewichtungen mit UTMOS
            if utmos_score > 0:
                weights = {
                    'utmos': 0.4,      # UTMOS als Hauptkomponente
                    'pesq': 0.2,
                    'stoi': 0.15,
                    'spectral_quality': 0.1,
                    'naturalness': 0.1,
                    'pronunciation': 0.05
                }
            else:
                # Fallback auf ursprüngliche Gewichtungen
                weights = self.mos_weights
                # Entferne UTMOS-Komponente
                mos_components = {k: v for k, v in mos_components.items() if k != 'utmos'}
            
            # Gewichteter MOS-Score (1-5 Skala)
            weighted_score = sum(
                weights.get(key, 0) * value 
                for key, value in mos_components.items()
            )
            predicted_mos = 1.0 + (weighted_score * 4.0)
            
            # Erweiterte Konfidenzintervall-Berechnung
            confidence_range = self._calculate_advanced_confidence_interval(mos_components, utmos_score)
            confidence_interval = (
                max(1.0, predicted_mos - confidence_range),
                min(5.0, predicted_mos + confidence_range)
            )
            
            # Qualitäts-Kategorie bestimmen
            quality_category = self._get_quality_category(predicted_mos)
            
            return MOSApproximation(
                predicted_mos=predicted_mos,
                confidence_interval=confidence_interval,
                quality_category=quality_category,
                contributing_factors=mos_components
            )
            
        except Exception as e:
            self.logger.error(f"Erweiterte MOS-Berechnung fehlgeschlagen: {e}")
            # Fallback auf ursprüngliche Methode
            return self._calculate_basic_mos_approximation(generated_audio, reference_audio, text)
    
    def _calculate_basic_mos_approximation(
        self, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        text: Optional[str] = None
    ) -> MOSApproximation:
        """Fallback-Methode für MOS-Approximation ohne erweiterte Modelle."""
        try:
            # Basis-Qualitäts-Metriken berechnen
            from .audio_metrics import AudioMetricsCalculator
            audio_metrics = AudioMetricsCalculator(
                sample_rate=self.sample_rate,
                use_advanced_metrics=False,
                logger=self.logger
            )
            quality_metrics = audio_metrics.calculate_audio_quality_metrics(generated_audio, reference_audio)
            
            mos_components = {
                'pesq': min(quality_metrics.pesq_score / 4.5, 1.0),
                'stoi': quality_metrics.stoi_score,
                'spectral_quality': self._calculate_spectral_quality(generated_audio),
                'naturalness': self._calculate_naturalness_score(generated_audio),
                'pronunciation': self._calculate_pronunciation_score(generated_audio, text) if text else 0.8
            }
            
            weighted_score = sum(
                self.mos_weights[key] * value 
                for key, value in mos_components.items()
            )
            predicted_mos = 1.0 + (weighted_score * 4.0)
            
            confidence_range = self._calculate_confidence_interval(mos_components)
            confidence_interval = (
                max(1.0, predicted_mos - confidence_range),
                min(5.0, predicted_mos + confidence_range)
            )
            
            quality_category = self._get_quality_category(predicted_mos)
            
            return MOSApproximation(
                predicted_mos=predicted_mos,
                confidence_interval=confidence_interval,
                quality_category=quality_category,
                contributing_factors=mos_components
            )
            
        except Exception as e:
            self.logger.error(f"Basis-MOS-Berechnung fehlgeschlagen: {e}")
            # Absolute Fallback
            return MOSApproximation(
                predicted_mos=2.5,
                confidence_interval=(2.0, 3.0),
                quality_category="Fair",
                contributing_factors={}
            )
    
    def _calculate_spectral_quality(self, audio: np.ndarray) -> float:
        """Berechnet spektrale Qualität basierend auf Frequenz-Charakteristika."""
        try:
            import librosa
            
            if len(audio) == 0:
                return 0.7
                
            # Spektrale Features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Qualitäts-Score basierend auf spektralen Eigenschaften
            centroid_score = 1.0 - abs(np.mean(spectral_centroid) - 2000) / 4000
            bandwidth_score = min(np.mean(spectral_bandwidth) / 3000, 1.0)
            rolloff_score = 1.0 - abs(np.mean(spectral_rolloff) - 4000) / 6000
            
            quality_score = (centroid_score * 0.4 + bandwidth_score * 0.3 + rolloff_score * 0.3)
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.warning(f"Spektrale Qualitäts-Berechnung fehlgeschlagen: {e}")
            return 0.7
    
    def _calculate_naturalness_score(self, audio: np.ndarray) -> float:
        """Berechnet Natürlichkeits-Score basierend auf Audio-Charakteristika."""
        try:
            import librosa
            
            if len(audio) == 0:
                return 0.7
                
            # Pitch-Variation
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            pitch_variation = np.std(pitch_values) if len(pitch_values) > 0 else 0
            pitch_score = min(pitch_variation / 50, 1.0)
            
            # Rhythmus-Konsistenz
            onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate)
            
            if len(onset_times) > 1:
                onset_intervals = np.diff(onset_times)
                rhythm_consistency = 1.0 - (np.std(onset_intervals) / np.mean(onset_intervals))
                rhythm_score = max(0.0, min(1.0, rhythm_consistency))
            else:
                rhythm_score = 0.5
            
            # Energie-Verteilung
            energy = librosa.feature.rms(y=audio)[0]
            energy_variation = np.std(energy) / (np.mean(energy) + 1e-10)
            energy_score = min(energy_variation * 2, 1.0)
            
            naturalness = (pitch_score * 0.4 + rhythm_score * 0.3 + energy_score * 0.3)
            return max(0.0, min(1.0, naturalness))
            
        except Exception as e:
            self.logger.warning(f"Natürlichkeits-Score-Berechnung fehlgeschlagen: {e}")
            return 0.7
    
    def _calculate_pronunciation_score(self, audio: np.ndarray, text: Optional[str]) -> float:
        """Berechnet Aussprache-Score (vereinfachte Implementierung)."""
        if not text or len(audio) == 0:
            return 0.8
        
        try:
            import librosa
            
            expected_duration = len(text.split()) * 0.6
            actual_duration = len(audio) / self.sample_rate
            
            duration_ratio = actual_duration / expected_duration if expected_duration > 0 else 1.0
            duration_score = 1.0 - abs(1.0 - duration_ratio)
            duration_score = max(0.0, min(1.0, duration_score))
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
            spectral_consistency = 1.0 - (np.std(mel_spec) / (np.mean(mel_spec) + 1e-10))
            consistency_score = max(0.0, min(1.0, spectral_consistency))
            
            pronunciation_score = (duration_score * 0.6 + consistency_score * 0.4)
            return pronunciation_score
            
        except Exception as e:
            self.logger.warning(f"Aussprache-Score-Berechnung fehlgeschlagen: {e}")
            return 0.7
    
    def _calculate_confidence_interval(self, mos_components: Dict[str, float]) -> float:
        """Berechnet Konfidenzintervall für MOS-Score."""
        values = list(mos_components.values())
        if not values:
            return 0.5
        
        variance = np.var(values)
        confidence_range = np.sqrt(variance) * 0.5
        return min(confidence_range, 1.0)
    
    def _calculate_advanced_confidence_interval(self, mos_components: Dict[str, float], utmos_score: float) -> float:
        """Berechnet erweiterte Konfidenzintervalle unter Berücksichtigung von UTMOS."""
        values = list(mos_components.values())
        if not values:
            return 0.5
        
        base_variance = np.var(values)
        
        # UTMOS-basierte Konfidenz-Anpassung
        if utmos_score > 0:
            utmos_confidence = min(utmos_score / 5.0, 1.0)
            confidence_modifier = 1.0 - (utmos_confidence * 0.3)
        else:
            confidence_modifier = 1.2
        
        confidence_range = np.sqrt(base_variance) * 0.5 * confidence_modifier
        return min(confidence_range, 1.0)
    
    def _get_quality_category(self, mos_score: float) -> str:
        """Bestimmt Qualitäts-Kategorie basierend auf MOS-Score."""
        for (min_score, max_score), category in self.mos_categories.items():
            if min_score <= mos_score < max_score:
                return category
        return "Unknown"