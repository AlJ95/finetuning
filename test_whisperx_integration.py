#!/usr/bin/env python3
"""
Test-Skript für WhisperX-Integration in der TTS-Evaluation.

Demonstriert die erweiterte Phonem-Analyse mit WhisperX
für deutsche TTS-Modelle.
"""

import numpy as np
import librosa
from pathlib import Path
import tempfile
import soundfile as sf

from src.tts_evaluation import TTSEvaluator


def create_synthetic_german_audio(text: str, duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Erstellt synthetisches Audio für deutsche Texte.
    
    Args:
        text: Deutscher Text
        duration: Audio-Dauer in Sekunden
        sample_rate: Sampling-Rate
        
    Returns:
        Synthetisches Audio-Array
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Verschiedene Frequenzen für verschiedene Phoneme
    audio = np.zeros_like(t)
    
    # Basis-Frequenz für Sprache
    base_freq = 150  # Grundfrequenz
    
    # Moduliere Frequenz basierend auf Text-Eigenschaften
    for i, char in enumerate(text.lower()):
        if char.isalpha():
            # Verschiedene Frequenzen für verschiedene Buchstaben
            char_freq = base_freq + (ord(char) - ord('a')) * 10
            
            # Zeitfenster für diesen Buchstaben
            start_idx = int((i / len(text)) * len(t))
            end_idx = int(((i + 1) / len(text)) * len(t))
            
            if end_idx > start_idx:
                # Sinuswelle für diesen Buchstaben
                segment = np.sin(2 * np.pi * char_freq * t[start_idx:end_idx])
                
                # Envelope für natürlicheren Klang
                envelope = np.exp(-3 * np.linspace(0, 1, len(segment)))
                audio[start_idx:end_idx] += segment * envelope * 0.3
    
    # Leichtes Rauschen hinzufügen
    noise = np.random.normal(0, 0.05, len(audio))
    audio += noise
    
    # Normalisieren
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def test_whisperx_phoneme_analysis():
    """Test der erweiterten Phonem-Analyse mit WhisperX."""
    print("🎯 Teste WhisperX-Integration für deutsche Phonem-Analyse...")
    
    # Evaluator mit erweiterten Metriken initialisieren
    evaluator = TTSEvaluator(
        sample_rate=16000, 
        use_advanced_metrics=True,
        log_level="INFO"
    )
    
    # Deutsche Test-Texte mit verschiedenen Phonem-Herausforderungen
    test_cases = [
        "Hallo schöne Welt",
        "Das Mädchen spricht Deutsch",
        "Weiß, grün und blau",
        "Ich möchte Kaffee trinken"
    ]
    
    print(f"📊 Analysiere {len(test_cases)} deutsche Test-Sätze...")
    
    results = []
    
    for i, text in enumerate(test_cases):
        print(f"\n🔍 Test {i+1}: '{text}'")
        
        # Synthetisches Audio generieren
        generated_audio = create_synthetic_german_audio(text, duration=2.0)
        reference_audio = create_synthetic_german_audio(text, duration=2.1)  # Leicht unterschiedlich
        
        print(f"   📈 Audio generiert: {len(generated_audio)} Samples ({len(generated_audio)/16000:.2f}s)")
        
        # Phonem-Analyse durchführen
        try:
            phoneme_metrics = evaluator.measure_phoneme_accuracy(
                text=text,
                generated_audio=generated_audio,
                reference_audio=reference_audio
            )
            
            print(f"   ✅ Phonem-Analyse erfolgreich:")
            print(f"      • Overall Accuracy: {phoneme_metrics.overall_accuracy:.3f}")
            print(f"      • Vowel Accuracy: {phoneme_metrics.vowel_accuracy:.3f}")
            print(f"      • Consonant Accuracy: {phoneme_metrics.consonant_accuracy:.3f}")
            print(f"      • Umlaut Accuracy: {phoneme_metrics.umlaut_accuracy:.3f}")
            
            if phoneme_metrics.problematic_phonemes:
                print(f"      • Problematische Phoneme: {', '.join(phoneme_metrics.problematic_phonemes)}")
            
            results.append({
                'text': text,
                'metrics': phoneme_metrics,
                'success': True
            })
            
        except Exception as e:
            print(f"   ❌ Fehler bei Phonem-Analyse: {e}")
            results.append({
                'text': text,
                'error': str(e),
                'success': False
            })
    
    # Zusammenfassung
    successful_tests = sum(1 for r in results if r['success'])
    print(f"\n📋 Zusammenfassung:")
    print(f"   • Erfolgreiche Tests: {successful_tests}/{len(test_cases)}")
    
    if successful_tests > 0:
        avg_overall = np.mean([r['metrics'].overall_accuracy for r in results if r['success']])
        avg_vowel = np.mean([r['metrics'].vowel_accuracy for r in results if r['success']])
        avg_consonant = np.mean([r['metrics'].consonant_accuracy for r in results if r['success']])
        avg_umlaut = np.mean([r['metrics'].umlaut_accuracy for r in results if r['success']])
        
        print(f"   • Durchschnittliche Overall Accuracy: {avg_overall:.3f}")
        print(f"   • Durchschnittliche Vowel Accuracy: {avg_vowel:.3f}")
        print(f"   • Durchschnittliche Consonant Accuracy: {avg_consonant:.3f}")
        print(f"   • Durchschnittliche Umlaut Accuracy: {avg_umlaut:.3f}")
    
    return results


def test_whisperx_vs_basic_comparison():
    """Vergleicht WhisperX-basierte vs. heuristische Phonem-Analyse."""
    print("\n🔬 Vergleiche WhisperX vs. Heuristische Phonem-Analyse...")
    
    text = "Das deutsche Mädchen spricht schön"
    audio = create_synthetic_german_audio(text, duration=3.0)
    
    # Test mit erweiterten Metriken (WhisperX)
    evaluator_advanced = TTSEvaluator(use_advanced_metrics=True, log_level="WARNING")
    
    # Test mit Basic-Metriken (Heuristik)
    evaluator_basic = TTSEvaluator(use_advanced_metrics=False, log_level="WARNING")
    
    print(f"📝 Test-Text: '{text}'")
    print(f"🎵 Audio-Länge: {len(audio)/16000:.2f}s")
    
    try:
        # WhisperX-Analyse
        print("\n🚀 WhisperX-Analyse...")
        advanced_metrics = evaluator_advanced.measure_phoneme_accuracy(text, audio)
        
        print(f"   • Overall Accuracy: {advanced_metrics.overall_accuracy:.3f}")
        print(f"   • Vowel Accuracy: {advanced_metrics.vowel_accuracy:.3f}")
        print(f"   • Consonant Accuracy: {advanced_metrics.consonant_accuracy:.3f}")
        print(f"   • Umlaut Accuracy: {advanced_metrics.umlaut_accuracy:.3f}")
        
        # Heuristische Analyse
        print("\n🔧 Heuristische Analyse...")
        basic_metrics = evaluator_basic.measure_phoneme_accuracy(text, audio)
        
        print(f"   • Overall Accuracy: {basic_metrics.overall_accuracy:.3f}")
        print(f"   • Vowel Accuracy: {basic_metrics.vowel_accuracy:.3f}")
        print(f"   • Consonant Accuracy: {basic_metrics.consonant_accuracy:.3f}")
        print(f"   • Umlaut Accuracy: {basic_metrics.umlaut_accuracy:.3f}")
        
        # Vergleich
        print("\n📊 Vergleich:")
        print(f"   • Overall Accuracy Differenz: {advanced_metrics.overall_accuracy - basic_metrics.overall_accuracy:+.3f}")
        print(f"   • Vowel Accuracy Differenz: {advanced_metrics.vowel_accuracy - basic_metrics.vowel_accuracy:+.3f}")
        print(f"   • Consonant Accuracy Differenz: {advanced_metrics.consonant_accuracy - basic_metrics.consonant_accuracy:+.3f}")
        print(f"   • Umlaut Accuracy Differenz: {advanced_metrics.umlaut_accuracy - basic_metrics.umlaut_accuracy:+.3f}")
        
        return {
            'advanced': advanced_metrics,
            'basic': basic_metrics,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Fehler beim Vergleich: {e}")
        return {'error': str(e), 'success': False}


def test_complete_evaluation_with_whisperx():
    """Test der vollständigen TTS-Evaluation mit WhisperX."""
    print("\n🎯 Teste vollständige TTS-Evaluation mit WhisperX...")
    
    evaluator = TTSEvaluator(use_advanced_metrics=True, log_level="INFO")
    
    text = "Guten Tag, wie geht es Ihnen heute?"
    generated_audio = create_synthetic_german_audio(text, duration=3.0)
    reference_audio = create_synthetic_german_audio(text, duration=2.8)
    
    print(f"📝 Test-Text: '{text}'")
    print(f"🎵 Generiertes Audio: {len(generated_audio)/16000:.2f}s")
    print(f"🎵 Referenz-Audio: {len(reference_audio)/16000:.2f}s")
    
    try:
        # Vollständige Evaluation
        results = evaluator.evaluate_tts_model(
            model_name="WhisperX-Test-Model",
            dataset_name="German-Synthetic-Test",
            generated_audio=generated_audio,
            text=text,
            reference_audio=reference_audio,
            inference_time_ms=250.0
        )
        
        print(f"\n✅ Vollständige Evaluation erfolgreich:")
        print(f"   • Overall Score: {results.overall_score:.3f}")
        print(f"   • MOS Approximation: {results.mos_approximation.predicted_mos:.2f}/5.0")
        print(f"   • MOS Category: {results.mos_approximation.quality_category}")
        print(f"   • Phoneme Overall Accuracy: {results.phoneme_accuracy.overall_accuracy:.3f}")
        print(f"   • Audio Quality PESQ: {results.audio_quality.pesq_score:.3f}")
        print(f"   • Audio Quality STOI: {results.audio_quality.stoi_score:.3f}")
        print(f"   • Inference Time: {results.inference_time_ms}ms")
        print(f"   • Real-time Factor: {(results.inference_time_ms/1000)/results.audio_duration_s:.2f}x")
        
        # Report generieren
        report = evaluator.generate_evaluation_report(results)
        print(f"\n📄 Report generiert ({len(report)} Zeichen)")
        
        return {'results': results, 'report': report, 'success': True}
        
    except Exception as e:
        print(f"❌ Fehler bei vollständiger Evaluation: {e}")
        return {'error': str(e), 'success': False}


def main():
    """Hauptfunktion für WhisperX-Integration-Tests."""
    print("🎤 WhisperX-Integration Test für deutsche TTS-Evaluation")
    print("=" * 60)
    
    # Test 1: Grundlegende Phonem-Analyse
    phoneme_results = test_whisperx_phoneme_analysis()
    
    # Test 2: Vergleich WhisperX vs. Heuristik
    comparison_results = test_whisperx_vs_basic_comparison()
    
    # Test 3: Vollständige Evaluation
    complete_results = test_complete_evaluation_with_whisperx()
    
    # Gesamtfazit
    print("\n" + "=" * 60)
    print("🏁 Test-Zusammenfassung:")
    
    successful_tests = 0
    total_tests = 3
    
    if any(r['success'] for r in phoneme_results):
        print("   ✅ Phonem-Analyse mit WhisperX: Erfolgreich")
        successful_tests += 1
    else:
        print("   ❌ Phonem-Analyse mit WhisperX: Fehlgeschlagen")
    
    if comparison_results.get('success', False):
        print("   ✅ WhisperX vs. Heuristik Vergleich: Erfolgreich")
        successful_tests += 1
    else:
        print("   ❌ WhisperX vs. Heuristik Vergleich: Fehlgeschlagen")
    
    if complete_results.get('success', False):
        print("   ✅ Vollständige Evaluation: Erfolgreich")
        successful_tests += 1
    else:
        print("   ❌ Vollständige Evaluation: Fehlgeschlagen")
    
    print(f"\n🎯 Gesamtergebnis: {successful_tests}/{total_tests} Tests erfolgreich")
    
    if successful_tests == total_tests:
        print("🎉 Alle Tests erfolgreich! WhisperX-Integration funktioniert korrekt.")
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Überprüfen Sie die Logs für Details.")
    
    return successful_tests == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)