# TTS-Finetuning Best Practices und Orpheus 3B Spezifika - Recherche Ergebnisse

## 1. Orpheus 3B Model Spezifikationen

### Architektur und Eigenschaften
- **Modellgröße**: 3B Parameter, Llama-basierte Speech-LLM Architektur
- **Sampling Rate**: 24 kHz (wichtig für Dataset-Vorbereitung)
- **Audio Token Ausgabe**: Direkte Audio-Token-Generierung ohne separaten Vocoder
- **Emotionale Unterstützung**: Eingebaute Unterstützung für emotionale Cues wie `<laugh>`, `<sigh>`, `<chuckle>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`
- **Kompatibilität**: Exportierbar via llama.cpp für breite Inference-Engine-Unterstützung

### Verfügbare Modellvarianten
- `unsloth/orpheus-3b-0.1-pretrained` - Basis-Modell
- `unsloth/orpheus-3b-0.1-ft` - Fine-tuned auf 8 professionelle Sprecher
- `canopylabs/orpheus-3b-0.1-ft` - Hugging Face Version
- `canopylabs/orpheus-3b-0.1-pretrained` - Hugging Face Basis-Version

### Deutsche Sprachunterstützung
- **Bestehende Modelle**: Bereits trainierte deutsche Modelle verfügbar:
  - `kadirnar/Orpheus-TTS-De`
  - `kadirnar/Orpheus-TTS-De-Beta` 
  - `kadirnar/Orpheus-Thorsten-De`
- **Verwendete Datasets**: Emilia-DE und Thorsten-Voice TV-44kHz-Full
- **Herausforderung**: Sampling Rate Anpassung von 44kHz auf 24kHz erforderlich

## 2. TTS Fine-tuning Best Practices

### Dataset Qualitätsanforderungen
- **Phonem-Abdeckung**: Gute Abdeckung von Phonemen, Di-Phonemen und Tri-Phonemen
- **Audio-Qualität**: Rauschfrei, konsistente Tonlage und Pitch
- **Längenverteilung**: Gaußsche Verteilung von Clip- und Textlängen
- **Normalisierung**: Vollständig annotiert und normalisiert
- **Sampling Rate**: Konsistent (16-22 kHz für allgemeine TTS, 24 kHz für Orpheus)

### Unsloth-spezifische Konfiguration
- **Quantisierung**: `load_in_4bit=False` für bessere Genauigkeit bei TTS
- **Training-Modus**: LoRA 16-bit empfohlen über QLoRA 4-bit
- **Alternative**: Full Fine-tuning (FFT) für höchste Qualität bei ausreichend VRAM
- **Performance**: 1.5x schneller, 50% weniger Speicher durch Flash Attention 2

### Training-Parameter (Empfohlene Werte)
```python
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none"
)
```

### Vollständiger Training-Workflow
```python
# 1. Model und Dataset laden
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/orpheus-3b-0.1-ft",
    max_seq_length=2048,
    dtype=None,  # Auto-detection
    load_in_4bit=False  # Wichtig für TTS!
)

# 2. Dataset vorbereiten (24kHz für Orpheus)
from datasets import load_dataset, Audio
dataset = load_dataset("your_dataset", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

# 3. Training starten
trainer.train()

# 4. Model speichern (nur LoRA Adapter)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### Audio-Token Verarbeitung
- **Herausforderung**: Audio muss in diskrete Tokens konvertiert werden
- **Orpheus-spezifisch**: Verwendet Audio-Codec für `<custom_token_x>` Sequenzen
- **Unsloth Integration**: Möglicherweise automatische Audio-Tokenisierung
- **Manuelle Verarbeitung**: Falls nötig, Orpheus GitHub encode_audio Funktion verwenden
- **Training-Ansatz**: Text als Input, Audio-Token-IDs als Labels

### Fine-tuning vs. Zero-shot Voice Cloning
- **Zero-shot Limitationen**: 
  - Erfasst nur grundlegende Tonlage und Timbre
  - Verliert Details wie Sprechgeschwindigkeit, Phrasierung, Eigenarten
  - Folgt dem Modell-Stil, nicht dem Sprecher-Stil
- **Fine-tuning Vorteile**:
  - Erfasst vollständige expressive Bandbreite
  - Lernt spezifische Sprechmuster und Prosodie
  - Notwendig für personalisierte oder expressive Stimmen
- **Anwendungsfall**: Zero-shot für einfache Stimmänderung, Fine-tuning für authentische Stimmklonierung

## 3. Deutsche TTS-spezifische Überlegungen

### Phonetische Besonderheiten
- **Umlaute**: ä, ö, ü, ß müssen korrekt behandelt werden
- **Betonung**: Deutsche Wortbetonung unterscheidet sich von Englisch
- **Zusammengesetzte Wörter**: Lange deutsche Komposita erfordern besondere Aufmerksamkeit

### Verfügbare deutsche Datasets
- **Thorsten-Voice**: TV-44kHz-Full (hohe Qualität, aber monoton)
- **Emilia-DE**: Deutscher Subset verfügbar
- **HUI-Audio-Corpus-German**: Hochqualitatives deutsches TTS-Dataset
- **VoxPopuli**: Multilingualer Korpus mit deutschem Subset

### Dataset-Qualitätsbewertung
- **Spektrogramm-Analyse**: Rauschpegel überprüfen
- **Phonem-Verteilung**: Ausgewogene Abdeckung sicherstellen
- **Alignment-Qualität**: Speech-Phoneme Alignment kritisch für TTS-Qualität

## 4. Offene Fragen und weitere Forschungsrichtungen

### Technische Unklarheiten
1. **Audio-Tokenisierung**: Genaue Implementierung der Audio-zu-Token Konvertierung bei Orpheus
2. **Unsloth-Automatisierung**: Inwieweit automatisiert Unsloth die Audio-Verarbeitung?
3. **Optimale Hyperparameter**: Spezifische Parameter für deutsche Sprache

### Evaluationsmetriken
1. **Objektive Metriken**: MOS (Mean Opinion Score), WER (Word Error Rate)
2. **Subjektive Bewertung**: Natürlichkeit, Verständlichkeit, Emotionalität
3. **Deutsche Spezifika**: Umgang mit Umlauten und Komposita

### Praktische Implementierung
1. **Hardware-Anforderungen**: 
   - Single GPU empfohlen (CUDA_VISIBLE_DEVICES=0)
   - Batch Size >1 kann zu Fehlern bei Multi-GPU führen
2. **Training-Zeit**: 
   - Colab T4 GPU: 1-2 Stunden für 3h Audiodaten
   - Unsloth-Optimierung macht Training deutlich schneller
3. **Inference-Optimierung**: 
   - GGUF-Konvertierung für Deployment möglich
   - llama.cpp Kompatibilität durch Orpheus-Architektur

## 5. Nächste Schritte

### Sofortige Maßnahmen
1. **Dataset-Auswahl**: Entscheidung zwischen verfügbaren deutschen Datasets
2. **Modell-Wahl**: Pretrained vs. Fine-tuned Orpheus als Ausgangspunkt
3. **Umgebung-Setup**: Unsloth-Installation und Konfiguration

### Mittelfristige Ziele
1. **Baseline-Training**: Erstes Training mit kleinem Dataset
2. **Evaluations-Pipeline**: Automatisierte Qualitätsbewertung
3. **Hyperparameter-Tuning**: Optimierung für deutsche Sprache

### Langfristige Überlegungen
1. **Multi-Speaker-Support**: Erweiterung auf mehrere deutsche Sprecher
2. **Dialekt-Unterstützung**: Regionale deutsche Varianten
3. **Produktions-Deployment**: Skalierbare Inference-Lösung

## 6. Referenzen und Quellen

- Unsloth TTS Documentation: https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning
- Coqui TTS Fine-tuning Guide: https://docs.coqui.ai/en/latest/finetuning.html
- Hugging Face Audio Course: https://huggingface.co/learn/audio-course/chapter6/fine-tuning
- Orpheus German Discussion: https://github.com/canopyai/Orpheus-TTS/discussions/117
- TTS Dataset Quality Guidelines: https://docs.coqui.ai/en/latest/what_makes_a_good_dataset.html