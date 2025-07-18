"""
Audio-Metriken-Modul für TTS-Evaluation.

Implementiert erweiterte Audio-Qualitäts-Metriken mit DTW-Alignment
und anderen fortgeschrittenen Techniken.
"""

import numpy as np
from typing import Optional
import logging

from .core import AudioQualityMetrics

# Audio quality metrics
from pesq import pesq
from pystoi import stoi
from scipy.stats import pearsonr


class AudioMetricsCalculator:
    """
    Spezialisierte Klasse für Audio-Qualitäts-Metriken.
    
    Implementiert PESQ, STOI, Mel-Spektrogramm-Distanz mit DTW,
    MFCC-Ähnlichkeit und andere Audio-Qualitäts-Metriken.
    """
    
    def __init__(self, sample_rate: int = 16000, use_advanced_metrics: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialisiert den Audio-Metriken-Calculator.
        
        Args:
            sample_rate: Audio-Sampling-Rate
            use_advanced_metrics: Ob erweiterte Metriken (DTW) verwendet werden sollen
            logger: Logger-Instanz
        """
        self.sample_rate = sample_rate
        self.use_advanced_metrics = use_advanced_metrics
        self.logger = logger or logging.getLogger(__name__)
    
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
        if self.use_advanced_metrics:
            return self._calculate_advanced_audio_quality_metrics(generated_audio, reference_audio)
        else:
            return self._calculate_basic_audio_quality_metrics(generated_audio, reference_audio)
    
    def _calculate_advanced_audio_quality_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> AudioQualityMetrics:
        """Berechnet erweiterte Audio-Qualitäts-Metriken mit DTW und anderen Verbesserungen."""
        self.logger.info("Berechne erweiterte Audio-Qualitäts-Metriken...")
        
        try:
            # Standard-Metriken (PESQ, STOI) bleiben unverändert
            pesq_score = 0.0
            stoi_score = 0.0
            
            if reference_audio is not None:
                pesq_score = self._calculate_pesq(generated_audio, reference_audio)
                stoi_score = self._calculate_stoi(generated_audio, reference_audio)
            
            # Erweiterte Metriken mit DTW
            mel_spectral_distance = self._calculate_mel_spectral_distance_with_dtw(
                generated_audio, reference_audio
            ) if reference_audio is not None else self._calculate_mel_spectral_distance_basic(generated_audio)
            
            # Standard-Metriken
            snr = self._calculate_snr(generated_audio)
            spectral_centroid_mean, spectral_rolloff_mean, zero_crossing_rate = self._calculate_spectral_features(generated_audio)
            
            # Erweiterte MFCC-Ähnlichkeit mit DTW
            mfcc_similarity = 0.0
            if reference_audio is not None:
                mfcc_similarity = self._calculate_mfcc_similarity_with_dtw(generated_audio, reference_audio)
            
            return AudioQualityMetrics(
                pesq_score=pesq_score,
                stoi_score=stoi_score,
                mel_spectral_distance=mel_spectral_distance,
                signal_to_noise_ratio=snr,
                spectral_centroid_mean=spectral_centroid_mean,
                spectral_rolloff_mean=spectral_rolloff_mean,
                zero_crossing_rate=zero_crossing_rate,
                mfcc_similarity=mfcc_similarity
            )
            
        except Exception as e:
            self.logger.error(f"Erweiterte Audio-Qualitäts-Metriken fehlgeschlagen: {e}")
            # Fallback auf Basic-Methode
            return self._calculate_basic_audio_quality_metrics(generated_audio, reference_audio)
    
    def _calculate_basic_audio_quality_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> AudioQualityMetrics:
        """Fallback-Methode für Audio-Qualitäts-Metriken ohne erweiterte Features."""
        try:
            pesq_score = 0.0
            stoi_score = 0.0
            
            if reference_audio is not None:
                pesq_score = self._calculate_pesq(generated_audio, reference_audio)
                stoi_score = self._calculate_stoi(generated_audio, reference_audio)
            
            mel_spectral_distance = self._calculate_mel_spectral_distance_basic(
                generated_audio, reference_audio
            ) if reference_audio is not None else 1.0
            
            snr = self._calculate_snr(generated_audio)
            spectral_centroid_mean, spectral_rolloff_mean, zero_crossing_rate = self._calculate_spectral_features(generated_audio)
            
            mfcc_similarity = 0.0
            if reference_audio is not None:
                mfcc_similarity = self._calculate_mfcc_similarity_basic(generated_audio, reference_audio)
            
            return AudioQualityMetrics(
                pesq_score=pesq_score,
                stoi_score=stoi_score,
                mel_spectral_distance=mel_spectral_distance,
                signal_to_noise_ratio=snr,
                spectral_centroid_mean=spectral_centroid_mean,
                spectral_rolloff_mean=spectral_rolloff_mean,
                zero_crossing_rate=zero_crossing_rate,
                mfcc_similarity=mfcc_similarity
            )
            
        except Exception as e:
            self.logger.error(f"Basis Audio-Qualitäts-Metriken fehlgeschlagen: {e}")
            # Absolute Fallback-Metriken
            return AudioQualityMetrics(
                pesq_score=2.5,
                stoi_score=0.7,
                mel_spectral_distance=1.0,
                signal_to_noise_ratio=10.0,
                spectral_centroid_mean=2000.0,
                spectral_rolloff_mean=4000.0,
                zero_crossing_rate=0.1,
                mfcc_similarity=0.7
            )
    
    def _calculate_pesq(self, generated_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Berechnet PESQ-Score."""
        try:
            min_len = min(len(generated_audio), len(reference_audio))
            gen_audio_norm = generated_audio[:min_len]
            ref_audio_norm = reference_audio[:min_len]
            
            return pesq(self.sample_rate, ref_audio_norm, gen_audio_norm, 'wb')
        except Exception as e:
            self.logger.warning(f"PESQ-Berechnung fehlgeschlagen: {e}")
            return 2.5
    
    def _calculate_stoi(self, generated_audio: np.ndarray, reference_audio: np.ndarray) -> float:
        """Berechnet STOI-Score."""
        try:
            min_len = min(len(generated_audio), len(reference_audio))
            gen_audio_norm = generated_audio[:min_len]
            ref_audio_norm = reference_audio[:min_len]
            
            stoi_score = stoi(ref_audio_norm, gen_audio_norm, self.sample_rate, extended=False)
            return max(0.0, min(1.0, stoi_score))
        except Exception as e:
            self.logger.warning(f"STOI-Berechnung fehlgeschlagen: {e}")
            return 0.7
    
    def _calculate_mfcc_similarity_with_dtw(
        self, 
        generated_audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> float:
        """Berechnet MFCC-Ähnlichkeit mit DTW-Alignment."""
        try:
            import librosa
            
            # MFCC-Features extrahieren
            mfcc_gen = librosa.feature.mfcc(
                y=generated_audio, sr=self.sample_rate, n_mfcc=13
            )
            mfcc_ref = librosa.feature.mfcc(
                y=reference_audio, sr=self.sample_rate, n_mfcc=13
            )
            
            # DTW-Alignment durchführen
            D, wp = librosa.sequence.dtw(
                X=mfcc_gen, 
                Y=mfcc_ref, 
                metric='cosine'
            )
            
            # Ähnlichkeits-Score aus DTW-Distanz berechnen
            aligned_distances = D[wp[:, 0], wp[:, 1]]
            mean_distance = np.mean(aligned_distances)
            
            # Konvertiere Distanz zu Ähnlichkeit (0-1 Skala)
            similarity = 1.0 / (1.0 + mean_distance)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.warning(f"DTW-MFCC-Ähnlichkeits-Berechnung fehlgeschlagen: {e}")
            # Fallback auf ursprüngliche Methode
            return self._calculate_mfcc_similarity_basic(generated_audio, reference_audio)
    
    def _calculate_mfcc_similarity_basic(
        self, 
        generated_audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> float:
        """Fallback-Methode für MFCC-Ähnlichkeit ohne DTW."""
        try:
            import librosa
            
            # MFCC-Features extrahieren
            mfcc_gen = librosa.feature.mfcc(
                y=generated_audio, sr=self.sample_rate, n_mfcc=13
            )
            mfcc_ref = librosa.feature.mfcc(
                y=reference_audio, sr=self.sample_rate, n_mfcc=13
            )
            
            # Auf gleiche Länge bringen (ursprüngliche Methode)
            min_frames = min(mfcc_gen.shape[1], mfcc_ref.shape[1])
            mfcc_gen_norm = mfcc_gen[:, :min_frames]
            mfcc_ref_norm = mfcc_ref[:, :min_frames]
            
            # Korrelation zwischen MFCC-Vektoren berechnen
            correlations = []
            for i in range(mfcc_gen_norm.shape[0]):
                corr, _ = pearsonr(mfcc_gen_norm[i], mfcc_ref_norm[i])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.5
            
        except Exception as e:
            self.logger.warning(f"MFCC-Ähnlichkeits-Berechnung fehlgeschlagen: {e}")
            return 0.5
    
    def _calculate_mel_spectral_distance_with_dtw(
        self, 
        generated_audio: np.ndarray, 
        reference_audio: np.ndarray
    ) -> float:
        """Berechnet Mel-Spektrogramm-Distanz mit DTW-Alignment."""
        try:
            import librosa
            
            # Mel-Spektrogramme berechnen
            mel_spec_gen = librosa.feature.melspectrogram(
                y=generated_audio, sr=self.sample_rate, n_mels=80
            )
            mel_spec_ref = librosa.feature.melspectrogram(
                y=reference_audio, sr=self.sample_rate, n_mels=80
            )
            
            # Log-Mel-Spektrogramme
            mel_spec_gen_db = librosa.power_to_db(mel_spec_gen, ref=np.max)
            mel_spec_ref_db = librosa.power_to_db(mel_spec_ref, ref=np.max)
            
            # DTW-Alignment durchführen
            D, wp = librosa.sequence.dtw(
                X=mel_spec_gen_db, 
                Y=mel_spec_ref_db, 
                metric='euclidean'
            )
            
            # Mittlere Distanz entlang des optimalen Pfads
            aligned_distances = D[wp[:, 0], wp[:, 1]]
            mean_distance = np.mean(aligned_distances)
            
            return max(0.0, mean_distance)
            
        except Exception as e:
            self.logger.warning(f"DTW-Mel-Spektral-Distanz-Berechnung fehlgeschlagen: {e}")
            # Fallback auf ursprüngliche Methode
            return self._calculate_mel_spectral_distance_basic(generated_audio, reference_audio)
    
    def _calculate_mel_spectral_distance_basic(
        self, 
        generated_audio: np.ndarray, 
        reference_audio: Optional[np.ndarray] = None
    ) -> float:
        """Fallback-Methode für Mel-Spektral-Distanz ohne DTW."""
        try:
            import librosa
            
            # Mel-Spektrogramm für generiertes Audio
            mel_spec_gen = librosa.feature.melspectrogram(
                y=generated_audio, sr=self.sample_rate, n_mels=80
            )
            mel_spec_gen_db = librosa.power_to_db(mel_spec_gen, ref=np.max)
            
            if reference_audio is not None:
                # Mel-Spektrogramm für Referenz-Audio
                mel_spec_ref = librosa.feature.melspectrogram(
                    y=reference_audio, sr=self.sample_rate, n_mels=80
                )
                mel_spec_ref_db = librosa.power_to_db(mel_spec_ref, ref=np.max)
                
                # Spektrogramme auf gleiche Größe bringen
                min_frames = min(mel_spec_gen_db.shape[1], mel_spec_ref_db.shape[1])
                mel_gen_norm = mel_spec_gen_db[:, :min_frames]
                mel_ref_norm = mel_spec_ref_db[:, :min_frames]
                
                # Euklidische Distanz berechnen
                distance = np.mean(np.sqrt(np.sum((mel_gen_norm - mel_ref_norm) ** 2, axis=0)))
                return distance
            else:
                # Ohne Referenz: Spektrale Konsistenz messen
                spectral_variance = np.var(mel_spec_gen_db, axis=1)
                return np.mean(spectral_variance)
                
        except Exception as e:
            self.logger.warning(f"Mel-Spektral-Distanz-Berechnung fehlgeschlagen: {e}")
            return 1.0
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Berechnet Signal-to-Noise Ratio."""
        try:
            if len(audio) == 0:
                return 10.0
                
            # Signal-Power (RMS der oberen 50% der Amplituden)
            sorted_audio = np.sort(np.abs(audio))
            signal_threshold = sorted_audio[int(len(sorted_audio) * 0.5)]
            signal_power = np.mean(audio[np.abs(audio) > signal_threshold] ** 2)
            
            # Noise-Power (RMS der unteren 20% der Amplituden)
            noise_threshold = sorted_audio[int(len(sorted_audio) * 0.2)]
            noise_samples = audio[np.abs(audio) < noise_threshold]
            noise_power = np.mean(noise_samples ** 2) if len(noise_samples) > 0 else 1e-10
            
            # SNR in dB
            snr_db = 10 * np.log10(signal_power / noise_power)
            return max(0.0, snr_db)  # Negative SNR auf 0 begrenzen
            
        except Exception as e:
            self.logger.warning(f"SNR-Berechnung fehlgeschlagen: {e}")
            return 10.0
    
    def _calculate_spectral_features(self, audio: np.ndarray) -> tuple:
        """Berechnet spektrale Features (Centroid, Rolloff, ZCR)."""
        try:
            import librosa
            
            # Spektrale Features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate
            )[0]
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            return (
                np.mean(spectral_centroid),
                np.mean(spectral_rolloff),
                np.mean(zcr)
            )
            
        except Exception as e:
            self.logger.warning(f"Spektrale Features-Berechnung fehlgeschlagen: {e}")
            return (2000.0, 4000.0, 0.1)