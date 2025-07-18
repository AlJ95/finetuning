"""
Tests für TTS-Evaluator-Komponente.

Testet MOS-Approximation, Phonem-Accuracy, Audio-Qualitäts-Metriken
und Report-Generierung mit synthetischen Daten.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.tts_evaluator import (
    TTSEvaluator,
    AudioQualityMetrics,
    PhonemeAccuracyMetrics,
    MOSApproximation,
    TTSEvaluationResults,
    load_audio_file,
    evaluate_audio_files
)


class TestTTSEvaluator:
    """Test-Suite für TTSEvaluator-Klasse."""
    
    @pytest.fixture
    def evaluator(self):
        """TTSEvaluator-Instanz für Tests."""
        return TTSEvaluator(sample_rate=16000, log_level="WARNING")
    
    @pytest.fixture
    def sample_audio(self):
        """Synthetisches Audio-Signal für Tests."""
        # Generiere 1 Sekunde synthetisches Audio (16kHz)
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
        assert evaluator.mos_weights['pesq'] == 0.3
        assert evaluator.mos_weights['stoi'] == 0.25
        assert len(evaluator.mos_categories) == 5
        assert evaluator.use_advanced_metrics == True
    
    def test_evaluator_initialization_basic_mode(self):
        """Test TTSEvaluator-Initialisierung im Basic-Modus."""
        evaluator = TTSEvaluator(sample_rate=16000, use_advanced_metrics=False)
        
        assert evaluator.sample_rate == 16000
        assert evaluator.use_advanced_metrics == False
    
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
            evaluator.save_results_json(results, output_path)
            
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
    
    def test_text_to_phonemes(self, evaluator):
        """Test deutsche Text-zu-Phonem-Konvertierung."""
        # Test verschiedene deutsche Texte
        test_cases = [
            ("hallo", ['h', 'a', 'l', 'l', 'o']),
            ("schön", ['sh', 'oe']),
            ("weiß", ['ss']),  # Test für ß -> ss
            ("mädchen", ['m', 'ae', 'd', 'ch'])
        ]
        
        for text, expected_phonemes in test_cases:
            phonemes = evaluator._text_to_phonemes(text)
            assert isinstance(phonemes, list)
            assert len(phonemes) > 0
            # Prüfe ob wichtige deutsche Phoneme korrekt erkannt werden
            for expected in expected_phonemes:
                if expected in ['sh', 'oe', 'ss', 'ch', 'ae']:  # Spezielle deutsche Phoneme
                    assert expected in phonemes, f"Expected phoneme '{expected}' not found in {phonemes} for text '{text}'"
    
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
    
    def test_german_phoneme_mapping(self, evaluator):
        """Test deutsche Phonem-Mappings."""
        from src.tts_evaluator import GERMAN_PHONEMES
        
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
    
    def test_advanced_metrics_availability(self):
        """Test Verfügbarkeit erweiterte Metriken."""
        from src.tts_evaluator import WHISPERX_AVAILABLE, UTMOS_AVAILABLE, SPEECHMETRICS_AVAILABLE
        
        # Diese Tests sollten nicht fehlschlagen, auch wenn die Pakete nicht installiert sind
        assert isinstance(WHISPERX_AVAILABLE, bool)
        assert isinstance(UTMOS_AVAILABLE, bool)
        assert isinstance(SPEECHMETRICS_AVAILABLE, bool)
    
    def test_basic_mode_fallback(self, sample_audio, reference_audio):
        """Test Fallback auf Basic-Modus wenn erweiterte Metriken deaktiviert sind."""
        evaluator = TTSEvaluator(use_advanced_metrics=False, log_level="WARNING")
        
        # MOS-Approximation sollte auch im Basic-Modus funktionieren
        mos_result = evaluator.calculate_mos_approximation(
            generated_audio=sample_audio,
            text="Test Basic Mode"
        )
        assert isinstance(mos_result, MOSApproximation)
        assert 1.0 <= mos_result.predicted_mos <= 5.0
        
        # Phonem-Accuracy sollte auch im Basic-Modus funktionieren
        phoneme_result = evaluator.measure_phoneme_accuracy(
            text="Test Basic Mode",
            generated_audio=sample_audio
        )
        assert isinstance(phoneme_result, PhonemeAccuracyMetrics)
        
        # Audio-Qualitäts-Metriken sollten auch im Basic-Modus funktionieren
        quality_result = evaluator.calculate_audio_quality_metrics(
            generated_audio=sample_audio,
            reference_audio=reference_audio
        )
        assert isinstance(quality_result, AudioQualityMetrics)
    
    def test_dtw_fallback_behavior(self, sample_audio, reference_audio):
        """Test DTW-Fallback-Verhalten bei Fehlern."""
        evaluator = TTSEvaluator(use_advanced_metrics=True, log_level="WARNING")
        
        # Test mit sehr kurzen Audio-Signalen (könnte DTW-Probleme verursachen)
        short_audio = sample_audio[:100]  # Nur 100 Samples
        short_reference = reference_audio[:50]  # Noch kürzer
        
        # Sollte nicht crashen, sondern graceful fallback verwenden
        quality_result = evaluator.calculate_audio_quality_metrics(
            generated_audio=short_audio,
            reference_audio=short_reference
        )
        assert isinstance(quality_result, AudioQualityMetrics)
        assert quality_result.mfcc_similarity >= 0.0
    
    @patch('src.tts_evaluator.WHISPERX_AVAILABLE', False)
    def test_whisperx_unavailable_fallback(self, sample_audio):
        """Test Fallback wenn WhisperX nicht verfügbar ist."""
        evaluator = TTSEvaluator(use_advanced_metrics=True, log_level="WARNING")
        
        # Sollte auf Basic-Phonem-Accuracy zurückfallen
        phoneme_result = evaluator.measure_phoneme_accuracy(
            text="Test ohne WhisperX",
            generated_audio=sample_audio
        )
        assert isinstance(phoneme_result, PhonemeAccuracyMetrics)
        assert 0.0 <= phoneme_result.overall_accuracy <= 1.0
    
    @patch('src.tts_evaluator.UTMOS_AVAILABLE', False)
    def test_utmos_unavailable_fallback(self, sample_audio):
        """Test Fallback wenn UTMOS nicht verfügbar ist."""
        evaluator = TTSEvaluator(use_advanced_metrics=True, log_level="WARNING")
        
        # Sollte auf Basic-MOS-Approximation zurückfallen
        mos_result = evaluator.calculate_mos_approximation(
            generated_audio=sample_audio,
            text="Test ohne UTMOS"
        )
        assert isinstance(mos_result, MOSApproximation)
        assert 1.0 <= mos_result.predicted_mos <= 5.0


class TestUtilityFunctions:
    """Test-Suite für Utility-Funktionen."""
    
    @patch('librosa.load')
    def test_load_audio_file(self, mock_librosa_load):
        """Test Audio-Datei-Laden."""
        # Mock librosa.load
        mock_audio = np.random.random(16000).astype(np.float32)
        mock_librosa_load.return_value = (mock_audio, 16000)
        
        audio, sr = load_audio_file(Path("test.wav"), target_sr=16000)
        
        assert isinstance(audio, np.ndarray)
        assert sr == 16000
        mock_librosa_load.assert_called_once()
    
    @patch('librosa.load')
    def test_load_audio_file_error(self, mock_librosa_load):
        """Test Fehlerbehandlung beim Audio-Laden."""
        mock_librosa_load.side_effect = Exception("Datei nicht gefunden")
        
        with pytest.raises(ValueError, match="Fehler beim Laden der Audio-Datei"):
            load_audio_file(Path("nonexistent.wav"))
    
    @patch('src.tts_evaluator.load_audio_file')
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