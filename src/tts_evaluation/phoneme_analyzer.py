"""
Phonem-Analyse-Modul für deutsche TTS-Evaluation.

Implementiert erweiterte Phonem-Accuracy-Messung mit WhisperX
und Fallback auf heuristische Methoden.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

from .core import PhonemeAccuracyMetrics

# Advanced evaluation libraries
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

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


class PhonemeAnalyzer:
    """
    Spezialisierte Klasse für deutsche Phonem-Analyse.
    
    Verwendet WhisperX für präzises Forced Alignment oder
    fällt auf heuristische Methoden zurück.
    """
    
    def __init__(self, sample_rate: int = 16000, use_advanced_metrics: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialisiert den Phonem-Analyzer.
        
        Args:
            sample_rate: Audio-Sampling-Rate
            use_advanced_metrics: Ob erweiterte Metriken verwendet werden sollen
            logger: Logger-Instanz
        """
        self.sample_rate = sample_rate
        self.use_advanced_metrics = use_advanced_metrics
        self.logger = logger or logging.getLogger(__name__)
        
        # WhisperX-Modelle (Lazy Loading)
        self._whisperx_model = None
        self._whisperx_align_model = None
        self._whisperx_metadata = None
    
    def measure_phoneme_accuracy(
        self, 
        text: str, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> PhonemeAccuracyMetrics:
        """
        Misst deutsche Phonem-Accuracy mit erweiterten Methoden.
        
        Args:
            text: Original-Text
            generated_audio: Generiertes Audio
            reference_audio: Referenz-Audio (optional)
            
        Returns:
            PhonemeAccuracyMetrics mit detaillierter Analyse
        """
        if self.use_advanced_metrics and WHISPERX_AVAILABLE:
            return self._calculate_advanced_phoneme_accuracy(text, generated_audio, reference_audio)
        else:
            return self._calculate_basic_phoneme_accuracy(text, generated_audio, reference_audio)
    
    def _load_whisperx_models(self):
        """Lädt WhisperX-Modelle bei Bedarf."""
        if not WHISPERX_AVAILABLE or not self.use_advanced_metrics:
            return False
            
        try:
            if self._whisperx_model is None:
                self.logger.info("Lade WhisperX-Modell...")
                device = "cpu"  # Verwende CPU für Kompatibilität
                # Verwende float32 für CPU-Kompatibilität
                self._whisperx_model = whisperx.load_model(
                    "large-v2", 
                    device=device, 
                    language="de",
                    compute_type="float32"  # Explizit float32 für CPU
                )
                
            if self._whisperx_align_model is None:
                self.logger.info("Lade WhisperX-Alignment-Modell...")
                self._whisperx_align_model, self._whisperx_metadata = whisperx.load_align_model(
                    language_code="de", 
                    device="cpu"
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der WhisperX-Modelle: {e}")
            return False
    
    def _align_phonemes_with_whisperx(self, audio: np.ndarray, text: str) -> List[Dict]:
        """Verwendet WhisperX für präzises Phonem-Alignment."""
        if not self._load_whisperx_models():
            self.logger.warning("WhisperX nicht verfügbar, verwende Fallback-Methode")
            return []
            
        try:
            # Transkription durchführen
            result = self._whisperx_model.transcribe(audio, batch_size=16)
            
            # Forced Alignment durchführen
            aligned_result = whisperx.align(
                result["segments"], 
                self._whisperx_align_model, 
                self._whisperx_metadata, 
                audio, 
                device="cpu",
                return_char_alignments=True
            )
            
            # Phonem-Level Informationen extrahieren
            phoneme_alignments = []
            for segment in aligned_result.get("segments", []):
                for word in segment.get("words", []):
                    if "chars" in word:  # Zeichen-Level Alignment verfügbar
                        for char_info in word["chars"]:
                            phoneme_alignments.append({
                                "phoneme": char_info["char"],
                                "start": char_info["start"],
                                "end": char_info["end"],
                                "confidence": char_info.get("score", 1.0)
                            })
                    else:
                        # Fallback auf Wort-Level
                        phoneme_alignments.append({
                            "phoneme": word["word"],
                            "start": word["start"],
                            "end": word["end"],
                            "confidence": word.get("score", 1.0)
                        })
            
            return phoneme_alignments
            
        except Exception as e:
            self.logger.error(f"WhisperX-Alignment fehlgeschlagen: {e}")
            return []
    
    def _extract_graphemes_from_alignment(self, alignment_data: List[Dict]) -> List[str]:
        """Extrahiert Grapheme (Zeichen) aus WhisperX-Alignment-Daten."""
        graphemes = []
        
        for item in alignment_data:
            char = item.get("phoneme", "").strip()
            if char and char.isalpha():
                graphemes.append(char.lower())
                    
        return graphemes
    
    def _text_to_graphemes(self, text: str) -> List[str]:
        """Konvertiert Text zu normalisierten Graphemen für CER-Berechnung."""
        import re
        
        # Text normalisieren: nur deutsche Buchstaben und Umlaute behalten
        normalized_text = re.sub(r'[^a-zäöüß ]', '', text.lower()).replace(' ', '')
        
        # In Liste von Zeichen umwandeln
        return list(normalized_text)
    
    def _calculate_advanced_phoneme_accuracy(
        self, 
        text: str, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> PhonemeAccuracyMetrics:
        """Berechnet erweiterte Phonem-Accuracy mit WhisperX."""
        self.logger.info("Berechne erweiterte Phonem-Accuracy mit WhisperX...")
        
        try:
            # WhisperX-Alignment für generiertes Audio
            generated_alignment = self._align_phonemes_with_whisperx(generated_audio, text)
            recognized_graphemes = self._extract_graphemes_from_alignment(generated_alignment)
            
            # Erwartete Grapheme aus Text extrahieren (Ground Truth)
            expected_graphemes = self._text_to_graphemes(text)
            
            # Falls Referenz-Audio verfügbar, auch dafür Alignment durchführen
            if reference_audio is not None:
                reference_alignment = self._align_phonemes_with_whisperx(reference_audio, text)
                reference_graphemes = self._extract_graphemes_from_alignment(reference_alignment)
                
                # Verwende Referenz-Grapheme als Ground Truth falls verfügbar
                expected_graphemes = reference_graphemes if reference_graphemes else expected_graphemes
            
            # Accuracy-Metriken berechnen (jetzt Graphem-zu-Graphem Vergleich = CER)
            overall_accuracy = self._calculate_phoneme_accuracy_with_alignment(
                expected_graphemes, recognized_graphemes, generated_alignment
            )
            
            # Spezifische Accuracy-Metriken (basierend auf Graphemen)
            vowel_accuracy = self._calculate_vowel_accuracy(expected_graphemes, recognized_graphemes)
            consonant_accuracy = self._calculate_consonant_accuracy(expected_graphemes, recognized_graphemes)
            umlaut_accuracy = self._calculate_umlaut_accuracy(expected_graphemes, recognized_graphemes)
            
            # Erweiterte Confusion Matrix mit Timing-Informationen
            confusion_matrix = self._create_advanced_phoneme_confusion_matrix(
                expected_graphemes, recognized_graphemes, generated_alignment
            )
            
            # Problematische Grapheme identifizieren
            problematic_phonemes = self._identify_problematic_phonemes(confusion_matrix)
            
            return PhonemeAccuracyMetrics(
                overall_accuracy=overall_accuracy,
                vowel_accuracy=vowel_accuracy,
                consonant_accuracy=consonant_accuracy,
                umlaut_accuracy=umlaut_accuracy,
                phoneme_confusion_matrix=confusion_matrix,
                problematic_phonemes=problematic_phonemes
            )
            
        except Exception as e:
            self.logger.error(f"Erweiterte Phonem-Accuracy-Berechnung fehlgeschlagen: {e}")
            # Fallback auf ursprüngliche Methode
            return self._calculate_basic_phoneme_accuracy(text, generated_audio, reference_audio)
    
    def _calculate_basic_phoneme_accuracy(
        self, 
        text: str, 
        generated_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> PhonemeAccuracyMetrics:
        """Fallback-Methode für Phonem-Accuracy ohne erweiterte Modelle."""
        # Vereinfachte Implementierung für Fallback
        expected_phonemes = self._text_to_phonemes(text)
        recognized_phonemes = self._recognize_phonemes_heuristic(generated_audio)
        
        overall_accuracy = self._calculate_phoneme_accuracy(expected_phonemes, recognized_phonemes)
        vowel_accuracy = self._calculate_vowel_accuracy(expected_phonemes, recognized_phonemes)
        consonant_accuracy = self._calculate_consonant_accuracy(expected_phonemes, recognized_phonemes)
        umlaut_accuracy = self._calculate_umlaut_accuracy(expected_phonemes, recognized_phonemes)
        
        confusion_matrix = self._create_phoneme_confusion_matrix(expected_phonemes, recognized_phonemes)
        problematic_phonemes = self._identify_problematic_phonemes(confusion_matrix)
        
        return PhonemeAccuracyMetrics(
            overall_accuracy=overall_accuracy,
            vowel_accuracy=vowel_accuracy,
            consonant_accuracy=consonant_accuracy,
            umlaut_accuracy=umlaut_accuracy,
            phoneme_confusion_matrix=confusion_matrix,
            problematic_phonemes=problematic_phonemes
        )
    
    def _text_to_phonemes(self, text: str) -> List[str]:
        """Konvertiert deutschen Text zu Phonemen (vereinfachte Implementierung)."""
        text_lower = text.lower()
        phonemes = []
        
        i = 0
        while i < len(text_lower):
            # Mehrzeichige Phoneme zuerst prüfen
            if i < len(text_lower) - 2:
                trigram = text_lower[i:i+3]
                if trigram in ['sch']:
                    phonemes.append('sh')
                    i += 3
                    continue
            
            if i < len(text_lower) - 1:
                bigram = text_lower[i:i+2]
                if bigram in ['ch', 'th', 'ph', 'ck', 'ng', 'nk']:
                    phonemes.extend(GERMAN_PHONEMES.get(bigram, [bigram]))
                    i += 2
                    continue
            
            # Einzelzeichen
            char = text_lower[i]
            if char.isalpha():
                phonemes.extend(GERMAN_PHONEMES.get(char, [char]))
            i += 1
        
        return phonemes
    
    def _recognize_phonemes_heuristic(self, audio: np.ndarray) -> List[str]:
        """Heuristische Phonem-Erkennung als Fallback."""
        # Vereinfachte Implementierung
        # In der Praxis würde hier eine komplexere Analyse stattfinden
        import librosa
        
        try:
            if len(audio) == 0:
                return []
                
            # MFCC-Features extrahieren
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            zcr = librosa.feature.zero_crossing_rate(audio)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            
            phonemes = []
            num_frames = mfccs.shape[1]
            segment_length = max(1, num_frames // 10)
            
            for i in range(0, num_frames, segment_length):
                end_idx = min(i + segment_length, num_frames)
                
                segment_zcr = zcr[:, i:end_idx] if zcr.size > 0 else np.array([0.1])
                segment_centroid = spectral_centroid[:, i:end_idx] if spectral_centroid.size > 0 else np.array([2000])
                
                avg_zcr = np.mean(segment_zcr)
                avg_centroid = np.mean(segment_centroid)
                
                # Heuristische Klassifikation
                if avg_zcr > 0.15:  # Konsonant
                    if avg_centroid > 3000:
                        phonemes.append('s')
                    else:
                        phonemes.append('t')
                else:  # Vokal
                    if avg_centroid < 1500:
                        phonemes.append('u')
                    elif avg_centroid > 2500:
                        phonemes.append('i')
                    else:
                        phonemes.append('a')
            
            return phonemes
            
        except Exception as e:
            self.logger.warning(f"Heuristische Phonem-Erkennung fehlgeschlagen: {e}")
            return []
    
    # Hilfsmethoden für Accuracy-Berechnungen
    def _calculate_phoneme_accuracy(self, expected: List[str], recognized: List[str]) -> float:
        """Berechnet Overall-Phonem-Accuracy."""
        if not expected or not recognized:
            return 0.0
        
        min_len = min(len(expected), len(recognized))
        max_len = max(len(expected), len(recognized))
        
        if max_len == 0:
            return 1.0
        
        matches = sum(1 for i in range(min_len) if expected[i] == recognized[i])
        return matches / max_len
    
    def _calculate_vowel_accuracy(self, expected: List[str], recognized: List[str]) -> float:
        """Berechnet Vokal-spezifische Accuracy (für Grapheme)."""
        vowels = ['a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü']
        expected_vowels = [g for g in expected if g in vowels]
        recognized_vowels = [g for g in recognized if g in vowels]
        return self._calculate_phoneme_accuracy(expected_vowels, recognized_vowels)
    
    def _calculate_consonant_accuracy(self, expected: List[str], recognized: List[str]) -> float:
        """Berechnet Konsonanten-spezifische Accuracy (für Grapheme)."""
        vowels = ['a', 'e', 'i', 'o', 'u', 'ä', 'ö', 'ü']
        expected_consonants = [g for g in expected if g not in vowels]
        recognized_consonants = [g for g in recognized if g not in vowels]
        return self._calculate_phoneme_accuracy(expected_consonants, recognized_consonants)
    
    def _calculate_umlaut_accuracy(self, expected: List[str], recognized: List[str]) -> float:
        """Berechnet Umlaut-spezifische Accuracy (für Grapheme)."""
        umlauts = ['ä', 'ö', 'ü']
        expected_umlauts = [g for g in expected if g in umlauts]
        recognized_umlauts = [g for g in recognized if g in umlauts]
        return self._calculate_phoneme_accuracy(expected_umlauts, recognized_umlauts)
    
    def _create_phoneme_confusion_matrix(self, expected: List[str], recognized: List[str]) -> Dict[str, Dict[str, int]]:
        """Erstellt Phonem-Confusion-Matrix."""
        confusion_matrix = {}
        min_len = min(len(expected), len(recognized))
        
        for i in range(min_len):
            exp = expected[i]
            rec = recognized[i]
            
            if exp not in confusion_matrix:
                confusion_matrix[exp] = {}
            if rec not in confusion_matrix[exp]:
                confusion_matrix[exp][rec] = 0
            
            confusion_matrix[exp][rec] += 1
        
        return confusion_matrix
    
    def _identify_problematic_phonemes(self, confusion_matrix: Dict[str, Dict[str, int]]) -> List[str]:
        """Identifiziert problematische Phoneme."""
        problematic = []
        
        for expected_phoneme, recognitions in confusion_matrix.items():
            total_occurrences = sum(recognitions.values())
            correct_recognitions = recognitions.get(expected_phoneme, 0)
            
            if total_occurrences > 0:
                accuracy = correct_recognitions / total_occurrences
                if accuracy < 0.7:
                    problematic.append(expected_phoneme)
        
        return problematic
    
    # Erweiterte Methoden für WhisperX-Integration
    def _calculate_phoneme_accuracy_with_alignment(
        self, expected: List[str], recognized: List[str], alignment_data: List[Dict]
    ) -> float:
        """Berechnet Phonem-Accuracy unter Berücksichtigung von Timing-Informationen."""
        if not expected or not recognized:
            return 0.0
        
        total_weight = 0.0
        weighted_matches = 0.0
        min_len = min(len(expected), len(recognized))
        
        for i in range(min_len):
            confidence = 1.0
            if i < len(alignment_data):
                confidence = alignment_data[i].get("confidence", 1.0)
            
            total_weight += confidence
            
            if expected[i] == recognized[i]:
                weighted_matches += confidence
        
        length_penalty = abs(len(expected) - len(recognized)) / max(len(expected), len(recognized))
        
        if total_weight > 0:
            accuracy = weighted_matches / total_weight
            accuracy = accuracy * (1.0 - length_penalty * 0.1)
            return max(0.0, min(1.0, accuracy))
        
        return 0.0
    
    def _create_advanced_phoneme_confusion_matrix(
        self, expected: List[str], recognized: List[str], alignment_data: List[Dict]
    ) -> Dict[str, Dict[str, int]]:
        """Erstellt erweiterte Phonem-Confusion-Matrix mit Timing-Informationen."""
        confusion_matrix = {}
        min_len = min(len(expected), len(recognized))
        
        for i in range(min_len):
            exp = expected[i]
            rec = recognized[i]
            
            # Confidence-Score berücksichtigen
            confidence = 1.0
            if i < len(alignment_data):
                confidence = alignment_data[i].get("confidence", 1.0)
            
            # Nur hochwertige Alignments verwenden
            if confidence < 0.5:
                continue
                
            if exp not in confusion_matrix:
                confusion_matrix[exp] = {}
            if rec not in confusion_matrix[exp]:
                confusion_matrix[exp][rec] = 0
            
            confusion_matrix[exp][rec] += 1
        
        return confusion_matrix