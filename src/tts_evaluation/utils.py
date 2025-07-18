"""
Hilfsfunktionen für TTS-Evaluation.

Enthält Utility-Funktionen für Audio-Laden, Report-Generierung,
JSON-Export und andere gemeinsame Funktionalitäten.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple
import logging
from dataclasses import asdict

from .core import TTSEvaluationResults, TTSEvaluator


def load_audio_file(file_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Lädt Audio-Datei und konvertiert auf Ziel-Sampling-Rate.
    
    Args:
        file_path: Pfad zur Audio-Datei
        target_sr: Ziel-Sampling-Rate
        
    Returns:
        Tuple aus Audio-Array und Sampling-Rate
    """
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Fehler beim Laden der Audio-Datei {file_path}: {e}")


def evaluate_audio_files(
    evaluator: TTSEvaluator,
    generated_audio_path: Path,
    text: str,
    reference_audio_path: Optional[Path] = None,
    model_name: str = "unknown",
    dataset_name: str = "unknown"
) -> TTSEvaluationResults:
    """
    Convenience-Funktion für Audio-Datei-Evaluation.
    
    Args:
        evaluator: TTSEvaluator-Instanz
        generated_audio_path: Pfad zum generierten Audio
        text: Original-Text
        reference_audio_path: Pfad zum Referenz-Audio (optional)
        model_name: Name des Modells
        dataset_name: Name des Datensatzes
        
    Returns:
        TTSEvaluationResults
    """
    # Audio-Dateien laden
    generated_audio, _ = load_audio_file(generated_audio_path, evaluator.sample_rate)
    
    reference_audio = None
    if reference_audio_path and reference_audio_path.exists():
        reference_audio, _ = load_audio_file(reference_audio_path, evaluator.sample_rate)
    
    # Evaluation durchführen
    return evaluator.evaluate_tts_model(
        model_name=model_name,
        dataset_name=dataset_name,
        generated_audio=generated_audio,
        text=text,
        reference_audio=reference_audio
    )


def generate_report(
    results: TTSEvaluationResults,
    output_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Generiert detaillierten Evaluation-Report.
    
    Args:
        results: TTS-Evaluation-Ergebnisse
        output_path: Pfad für Report-Speicherung (optional)
        logger: Logger-Instanz (optional)
        
    Returns:
        Formatierter Report als String
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Generiere TTS-Evaluation-Report...")
    
    report_lines = [
        "=" * 80,
        "TTS MODEL EVALUATION REPORT",
        "=" * 80,
        "",
        f"Timestamp: {results.timestamp}",
        f"Model: {results.model_name}",
        f"Dataset: {results.dataset_name}",
        f"Overall Score: {results.overall_score:.3f} ({results.overall_score * 100:.1f}%)",
        "",
        "=" * 80,
        "MOS APPROXIMATION",
        "=" * 80,
        f"Predicted MOS: {results.mos_approximation.predicted_mos:.2f}/5.0",
        f"Quality Category: {results.mos_approximation.quality_category}",
        f"Confidence Interval: [{results.mos_approximation.confidence_interval[0]:.2f}, {results.mos_approximation.confidence_interval[1]:.2f}]",
        "",
        "Contributing Factors:",
    ]
    
    for factor, value in results.mos_approximation.contributing_factors.items():
        report_lines.append(f"  - {factor.capitalize()}: {value:.3f}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "AUDIO QUALITY METRICS",
        "=" * 80,
        f"PESQ Score: {results.audio_quality.pesq_score:.3f}",
        f"STOI Score: {results.audio_quality.stoi_score:.3f}",
        f"Mel Spectral Distance: {results.audio_quality.mel_spectral_distance:.3f}",
        f"Signal-to-Noise Ratio: {results.audio_quality.signal_to_noise_ratio:.1f} dB",
        f"Spectral Centroid (Mean): {results.audio_quality.spectral_centroid_mean:.1f} Hz",
        f"Spectral Rolloff (Mean): {results.audio_quality.spectral_rolloff_mean:.1f} Hz",
        f"Zero Crossing Rate: {results.audio_quality.zero_crossing_rate:.4f}",
        f"MFCC Similarity: {results.audio_quality.mfcc_similarity:.3f}",
        "",
        "=" * 80,
        "PHONEME ACCURACY (GERMAN)",
        "=" * 80,
        f"Overall Accuracy: {results.phoneme_accuracy.overall_accuracy:.3f} ({results.phoneme_accuracy.overall_accuracy * 100:.1f}%)",
        f"Vowel Accuracy: {results.phoneme_accuracy.vowel_accuracy:.3f} ({results.phoneme_accuracy.vowel_accuracy * 100:.1f}%)",
        f"Consonant Accuracy: {results.phoneme_accuracy.consonant_accuracy:.3f} ({results.phoneme_accuracy.consonant_accuracy * 100:.1f}%)",
        f"Umlaut Accuracy: {results.phoneme_accuracy.umlaut_accuracy:.3f} ({results.phoneme_accuracy.umlaut_accuracy * 100:.1f}%)",
        "",
    ])
    
    if results.phoneme_accuracy.problematic_phonemes:
        report_lines.extend([
            "Problematic Phonemes:",
            "  " + ", ".join(results.phoneme_accuracy.problematic_phonemes),
            ""
        ])
    
    report_lines.extend([
        "=" * 80,
        "PERFORMANCE METRICS",
        "=" * 80,
        f"Inference Time: {results.inference_time_ms:.1f} ms",
        f"Audio Duration: {results.audio_duration_s:.2f} seconds",
        f"Text Length: {results.text_length} characters",
        f"Real-time Factor: {(results.inference_time_ms / 1000) / results.audio_duration_s:.2f}x" if results.audio_duration_s > 0 else "N/A",
        "",
        "=" * 80,
        "RECOMMENDATIONS",
        "=" * 80,
    ])
    
    # Empfehlungen basierend auf Ergebnissen
    recommendations = _generate_recommendations(results)
    for rec in recommendations:
        report_lines.append(f"• {rec}")
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    report_text = "\n".join(report_lines)
    
    # Report speichern falls Pfad angegeben
    if output_path:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report gespeichert: {output_path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Reports: {e}")
    
    return report_text


def save_results_json(
    results: TTSEvaluationResults, 
    output_path: str, 
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Speichert Evaluation-Ergebnisse als JSON.
    
    Args:
        results: TTS-Evaluation-Ergebnisse
        output_path: Pfad für JSON-Speicherung
        logger: Logger-Instanz (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dataclass zu Dict konvertieren
        results_dict = asdict(results)
        
        # Numpy-Typen zu Python-Typen konvertieren für JSON-Serialisierung
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_dict = convert_numpy_types(results_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ergebnisse als JSON gespeichert: {output_path}")
        
    except Exception as e:
        logger.error(f"Fehler beim Speichern der JSON-Ergebnisse: {e}")


def _generate_recommendations(results: TTSEvaluationResults) -> list:
    """Generiert Empfehlungen basierend auf Evaluation-Ergebnissen."""
    recommendations = []
    
    # MOS-basierte Empfehlungen
    if results.mos_approximation.predicted_mos < 3.0:
        recommendations.append("MOS-Score ist niedrig. Überprüfen Sie Trainings-Daten und Hyperparameter.")
    
    # Audio-Qualitäts-Empfehlungen
    if results.audio_quality.pesq_score < 2.0:
        recommendations.append("PESQ-Score ist niedrig. Audio-Qualität verbessern durch bessere Datenvorverarbeitung.")
    
    if results.audio_quality.stoi_score < 0.7:
        recommendations.append("STOI-Score ist niedrig. Sprachverständlichkeit kann durch längeres Training verbessert werden.")
    
    if results.audio_quality.signal_to_noise_ratio < 15.0:
        recommendations.append("SNR ist niedrig. Rauschunterdrückung in der Datenvorverarbeitung implementieren.")
    
    # Phonem-Accuracy-Empfehlungen
    if results.phoneme_accuracy.overall_accuracy < 0.7:
        recommendations.append("Phonem-Accuracy ist niedrig. Mehr deutsche Trainingsdaten oder phonem-spezifisches Training.")
    
    if results.phoneme_accuracy.umlaut_accuracy < 0.6:
        recommendations.append("Umlaut-Accuracy ist niedrig. Spezielle Aufmerksamkeit auf deutsche Umlaute im Training.")
    
    if results.phoneme_accuracy.problematic_phonemes:
        recommendations.append(f"Problematische Phoneme identifiziert: {', '.join(results.phoneme_accuracy.problematic_phonemes)}. Gezieltes Training empfohlen.")
    
    # Performance-Empfehlungen
    rtf = (results.inference_time_ms / 1000) / results.audio_duration_s if results.audio_duration_s > 0 else 0
    if rtf > 1.0:
        recommendations.append("Real-time Factor > 1.0. Modell-Optimierung für schnellere Inferenz empfohlen.")
    
    if not recommendations:
        recommendations.append("Alle Metriken sind im akzeptablen Bereich. Modell ist bereit für Deployment.")
    
    return recommendations