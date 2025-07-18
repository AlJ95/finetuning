"""
Tests für das modulare TTS-Evaluation-Paket.

Testet die erweiterten Funktionalitäten mit WhisperX, UTMOS und DTW.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.tts_evaluation import (
    TTSEvaluator,
    AudioQualityMetrics,
    PhonemeAccuracyMetrics,
    MOSApproximation,
    TTSEvaluationResults,
    load_audio_file,
    evaluate_audio_files
)


class TestTTSEvaluationModular:
    """Test-Suite für das modulare TTS-Evaluation-System."""
    
    @pytest.fixture
    def evaluator(self):
        """TTSEvaluator-Instanz für Tests."""
        return TTSEvaluator(sample_rate=16000, log_level="WARNING")
    
    @pytest.fixture
    def evaluator_basic(self):
        """TTSEvaluator-Instanz im Basic-Modus."""
        return TTSEvaluator(sample_rate=16000, log_level="WARNING", use_advanced_metrics=False)
    
    @pytest.fixture
    def sample_audio(self):
        """Synthetisches Audio-Signal für Tests."""
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Mischung aus Sinuswellen (simuliert Sprache)
        audio = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 Note
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 Note
            0.1 * np.random.normal(0, 0.05, len(t))  # Leichtes Rauschen
        )
        
        return audio.astype(np.float32)
    
    @pytest.fixture
    def reference_audio(self):
        """Referenz-Audio für Vergleichstests."""
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Ähnliches aber leicht unterschiedliches Audio
        audio = (
            0.35 * np.sin(2 * np.pi * 450 * t) +
            0.25 * np.sin(2 * np.pi * 900 * t) +
            0.05 * np.random.normal(0, 0.03, len(t))
        )
        
        return audio.astype(np.float32)
    
    def test_evaluator_initialization(self):
        """Test TTSEvaluator-Initialisierung."""
        evaluator = TTSEvaluator(sample_rate=22050, log_level="DEBUG")
        
        assert evaluator.sample_rate == 22050
        assert evaluator.use_advanced_metrics == True
        assert len(evaluator.mos_categories) == 5
    
    def test_evaluator_initialization_basic_mode(self):
        """Test TTSEvaluator-Initialisierung im Basic-Modus."""
        evaluator = TTSEvaluator(sample_rate=16000, use_advanced_metrics=False)
        
        assert evaluator.sample_rate == 16000
        assert evaluator.use_advanced_metrics == False
    
    def test_lazy_loading_modules(self, evaluator):
        """Test Lazy Loading der Module."""
        # Module sollten erst bei Bedarf geladen werden
        assert evaluator._phoneme_analyzer is None
        assert evaluator._mos_predictor is None
        assert evaluator._audio_metrics is None
        
        # Nach Zugriff sollten sie initialisiert sein
        phoneme_analyzer = evaluator.phoneme_analyzer
        assert evaluator._phoneme_analyzer is not None
        assert phoneme_analyzer is evaluator.phoneme_analyzer  # Gleiche Instanz
        
        mos_predictor = evaluator.mos_predictor
        assert evaluator._mos_predictor is not None
        
        audio_metrics = evaluator.audio_metrics
        assert evaluator._audio_metrics is not None
    
    def test_calculate_mos_approximation(self, evaluator, sample_audio, reference_audio):
        """Test MOS-Approximation-Berechnung."""
        mos_result = evaluator.calculate_mos_approximation(
            generated_audio=sample_audio,
            reference_audio=reference_audio,
            text="Hallo Welt"
        )
        
        assert isinstance(mos_result, MOSApproximation)
        assert 1.0 <= mos_result.predicted_mos <= 5.0
        assert len(mos_result.confidence_interval) == 2
        assert mos_result.confidence_interval[0] <= mos_result.confidence_interval[1]
        assert mos_result.quality_category in ["Excellent", "Good", "Fair", "Poor", "Bad"]
        assert isinstance(mos_result.contributing_factors, dict)
    
    def test_calculate_mos_without_reference(self, evaluator, sample_audio):
        """Test MOS-Berechnung ohne Referenz-Audio."""
        mos_result = evaluator.calculate_mos_approximation(
            generated_audio=sample_audio,
            text="Test ohne Referenz"
        )
        
        assert isinstance(mos_result, MOSApproximation)
        assert 1.0 <= mos_result.predicted_mos <= 5.0
    
    def test_measure_phoneme_accuracy(self, evaluator, sample_audio, reference_audio):
        """Test Phonem-Accuracy-Messung."""
        phoneme_result = evaluator.measure_phoneme_accuracy(
            text="Hallo schöne Welt",
            generated_audio=sample_audio,
            reference_audio=reference_audio
        )
        
        assert isinstance(phoneme_result, PhonemeAccuracyMetrics)
        assert 0.0 <= phoneme_result.overall_accuracy <= 1.0
        assert 0.0 <= phoneme_result.vowel_accuracy <= 1.0
        assert 0.0 <= phoneme_result.consonant_accuracy <= 1.0
        assert 0.0 <= phoneme_result.umlaut_accuracy <= 1.0
        assert isinstance(phoneme_result.phoneme_confusion_matrix, dict)
        assert isinstance(phoneme_result.problematic_phonemes, list)
    
    def test_calculate_audio_quality_metrics(self, evaluator, sample_audio, reference_audio):
        """Test Audio-Qualitäts-Metriken-Berechnung."""
        quality_result = evaluator.calculate_audio_quality_metrics(
            generated_audio=sample_audio,
            reference_audio=reference_audio
        )
        
        assert isinstance(quality_result, AudioQualityMetrics)
        assert quality_result.pesq_score >= 0.0
        assert 0.0 <= quality_result.stoi_score <= 1.0
        assert quality_result.mel_spectral_distance >= 0.0
        assert quality_result.signal_to_noise_ratio >= 0.0
        assert quality_result.spectral_centroid_mean > 0.0
        assert quality_result.spectral_rolloff_mean > 0.0
        assert quality_result.zero_crossing_rate >= 0.0
        assert 0.0 <= quality_result.mfcc_similarity <= 1.0
    
    def test_calculate_audio_quality_without_reference(self, evaluator, sample_audio):
        """Test Audio-Qualitäts-Metriken ohne Referenz."""
        quality_result = evaluator.calculate_audio_quality_metrics(
            generated_audio=sample_audio
        )
        
        assert isinstance(quality_result, AudioQualityMetrics)
        assert quality_result.pesq_score == 0.0  # Kein Referenz-Audio
        assert quality_result.stoi_score == 0.0  # Kein Referenz-Audio
        assert quality_result.mfcc_similarity == 0.0  # Kein Referenz-Audio
    
    def test_evaluate_tts_model_complete(self, evaluator, sample_audio, reference_audio):
        """Test vollständige TTS-Modell-Evaluation."""
        results = evaluator.evaluate_tts_model(
            model_name="test-model",
            dataset_name="test-dataset",
            generated_audio=sample_audio,
            text="Dies ist ein Test für deutsche TTS-Evaluation",
            reference_audio=reference_audio,
            inference_time_ms=150.0
        )
        
        assert isinstance(results, TTSEvaluationResults)
        assert results.model_name == "test-model"
        assert results.dataset_name == "test-dataset"
        assert results.inference_time_ms == 150.0
        assert results.audio_duration_s > 0.0
        assert results.text_length > 0
        assert 0.0 <= results.overall_score <= 1.0
        
        # Prüfe Sub-Komponenten
        assert isinstance(results.audio_quality, AudioQualityMetrics)
        assert isinstance(results.phoneme_accuracy, PhonemeAccuracyMetrics)
        assert isinstance(results.mos_approximation, MOSApproximation)
    
    def test_generate_evaluation_report(self, evaluator, sample_audio):
        """Test Evaluation-Report-Generierung."""
        # Erst Evaluation durchführen
        results = evaluator.evaluate_tts_model(
            model_name="test-model",
            dataset_name="test-dataset",
            generated_audio=sample_audio,
            text="Test für Report-Generierung"
        )
        
        # Report generieren
        report = evaluator.generate_evaluation_report(results)
        
        assert isinstance(report, str)
        assert "TTS MODEL EVALUATION REPORT" in report
        assert "test-model" in report
        assert "test-dataset" in report
        assert "MOS APPROXIMATION" in report
        assert "AUDIO QUALITY METRICS" in report
        assert "PHONEME ACCURACY" in report
        assert "RECOMMENDATIONS" in report
    
    def test_save_results_json(self, evaluator, sample_audio):
        """Test JSON-Speicherung der Ergebnisse."""
        results = evaluator.evaluate_tts_model(
            model_name="json-test-model",
            dataset_name="json-test-dataset",
            generated_audio=sample_audio,
            text="JSON Test"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_results.json"
            evaluator.save_results_json(results, str(output_path))
            
            assert output_path.exists()
            
            # JSON-Inhalt prüfen
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            assert loaded_data['model_name'] == "json-test-model"
            assert loaded_data['dataset_name'] == "json-test-dataset"
            assert 'audio_quality' in loaded_data
            assert 'phoneme_accuracy' in loaded_data
            assert 'mos_approximation' in loaded_data
    
    def test_basic_mode_fallback(self, evaluator_basic, sample_audio, reference_audio):
        """Test Fallback auf Basic-Modus wenn erweiterte Metriken deaktiviert sind."""
        # MOS-Approximation sollte auch im Basic-Modus funktionieren
        mos_result = evaluator_basic.calculate_mos_approximation(
            generated_audio=sample_audio,
            text="Test Basic Mode"
        )
        assert isinstance(mos_result, MOSApproximation)
        assert 1.0 <= mos_result.predicted_mos <= 5.0
        
        # Phonem-Accuracy sollte auch im Basic-Modus funktionieren
        phoneme_result = evaluator_basic.measure_phoneme_accuracy(
            text="Test Basic Mode",
            generated_audio=sample_audio
        )
        assert isinstance(phoneme_result, PhonemeAccuracyMetrics)
        
        # Audio-Qualitäts-Metriken sollten auch im Basic-Modus funktionieren
        quality_result = evaluator_basic.calculate_audio_quality_metrics(
            generated_audio=sample_audio,
            reference_audio=reference_audio
        )
        assert isinstance(quality_result, AudioQualityMetrics)
    
    def test_dtw_functionality(self, evaluator, sample_audio, reference_audio):
        """Test DTW-Funktionalität bei erweiterten Metriken."""
        # Test mit verschiedenen Audio-Längen (DTW sollte das handhaben)
        short_audio = sample_audio[:8000]  # Halbe Länge
        
        quality_result = evaluator.calculate_audio_quality_metrics(
            generated_audio=short_audio,
            reference_audio=reference_audio
        )
        
        assert isinstance(quality_result, AudioQualityMetrics)
        assert quality_result.mfcc_similarity >= 0.0
        assert quality_result.mel_spectral_distance >= 0.0
    
    def test_error_handling_invalid_audio(self, evaluator):
        """Test Fehlerbehandlung bei ungültigen Audio-Daten."""
        # Leeres Audio-Array
        empty_audio = np.array([])
        
        mos_result = evaluator.calculate_mos_approximation(empty_audio)
        assert isinstance(mos_result, MOSApproximation)
        # MOS sollte im gültigen Bereich sein, auch bei leerem Audio
        assert 1.0 <= mos_result.predicted_mos <= 5.0
        
        # Audio mit NaN-Werten
        nan_audio = np.array([np.nan, np.nan, np.nan])
        
        quality_result = evaluator.calculate_audio_quality_metrics(nan_audio)
        assert isinstance(quality_result, AudioQualityMetrics)
    
    def test_advanced_metrics_availability(self):
        """Test Verfügbarkeit erweiterte Metriken."""
        from src.tts_evaluation.phoneme_analyzer import WHISPERX_AVAILABLE
        from src.tts_evaluation.mos_predictor import UTMOS_AVAILABLE, SPEECHMETRICS_AVAILABLE
        
        # Diese Tests sollten nicht fehlschlagen, auch wenn die Pakete nicht installiert sind
        assert isinstance(WHISPERX_AVAILABLE, bool)
        assert isinstance(UTMOS_AVAILABLE, bool)
        assert isinstance(SPEECHMETRICS_AVAILABLE, bool)
    
    def test_german_phoneme_mapping(self):
        """Test deutsche Phonem-Mappings."""
        from src.tts_evaluation.phoneme_analyzer import GERMAN_PHONEMES
        
        # Prüfe wichtige deutsche Phoneme
        assert 'ä' in GERMAN_PHONEMES
        assert 'ö' in GERMAN_PHONEMES
        assert 'ü' in GERMAN_PHONEMES
        assert 'ß' in GERMAN_PHONEMES
        assert 'ch' in GERMAN_PHONEMES
        assert 'sch' in GERMAN_PHONEMES
        
        # Prüfe Phonem-Mappings
        assert 'ae' in GERMAN_PHONEMES['ä']
        assert 'oe' in GERMAN_PHONEMES['ö']
        assert 'ue' in GERMAN_PHONEMES['ü']
        assert 'ss' in GERMAN_PHONEMES['ß']


class TestUtilityFunctions:
    """Test-Suite für Utility-Funktionen."""
    
    @patch('src.tts_evaluation.utils.librosa.load')
    def test_load_audio_file(self, mock_librosa_load):
        """Test Audio-Datei-Laden."""
        # Mock librosa.load
        mock_audio = np.random.random(16000).astype(np.float32)
        mock_librosa_load.return_value = (mock_audio, 16000)
        
        audio, sr = load_audio_file(Path("test.wav"), target_sr=16000)
        
        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        mock_librosa_load.assert_called_once()
    
    @patch('src.tts_evaluation.utils.librosa.load')
    def test_load_audio_file_error(self, mock_librosa_load):
        """Test Fehlerbehandlung beim Audio-Laden."""
        mock_librosa_load.side_effect = Exception("Datei nicht gefunden")
        
        with pytest.raises(ValueError, match="Fehler beim Laden der Audio-Datei"):
            load_audio_file(Path("nonexistent.wav"))
    
    @patch('src.tts_evaluation.utils.load_audio_file')
    def test_evaluate_audio_files(self, mock_load_audio):
        """Test Audio-Datei-Evaluation-Convenience-Funktion."""
        # Mock Audio-Laden
        sample_audio = np.random.random(16000).astype(np.float32)
        mock_load_audio.return_value = (sample_audio, 16000)
        
        evaluator = TTSEvaluator(log_level="WARNING")
        
        # Test ohne Referenz-Audio
        results = evaluate_audio_files(
            evaluator=evaluator,
            generated_audio_path=Path("generated.wav"),
            text="Test Audio Evaluation",
            reference_audio_path=None,
            model_name="test-model",
            dataset_name="test-dataset"
        )
        
        assert isinstance(results, TTSEvaluationResults)
        assert results.model_name == "test-model"
        assert results.dataset_name == "test-dataset"
        assert mock_load_audio.call_count == 1  # Nur generiertes Audio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])