# German TTS Fine-tuning with Unsloth and Orpheus 3B

Ein spezialisiertes System für das Fine-tuning deutscher Text-to-Speech Modelle mit Unsloth und Orpheus 3B.

## 🎯 Projektziel

Entwicklung eines effizienten Fine-tuning Systems für deutsche TTS-Modelle mit:
- **Unsloth** für memory-efficient Training
- **Orpheus 3B** als Basis-Modell
- **Deutsche Datasets** (Thorsten-Voice, MLS German)
- **VLLM-Kompatibilität** für Production Deployment

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Repository klonen
git clone <repository-url>
cd german-tts-finetuning

# Virtual Environment erstellen
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Daten vorbereiten

```bash
# Thorsten-Voice Dataset (empfohlen für Start)
# Download von: https://www.thorsten-voice.de/en/datasets-2/
# Extrahieren nach: data/thorsten-voice/

# MLS German Dataset (optional, für Multi-Speaker)
# Download von: https://openslr.org/94/
# Extrahieren nach: data/mls-german/
```

### 3. Datenverarbeitung

```python
from pathlib import Path
from src.torsten_voice_processor import TorstenVoiceDataProcessor

# Thorsten-Voice Dataset verarbeiten
processor = TorstenVoiceDataProcessor()
dataset_path = Path("data/thorsten-voice/ThorstenVoice-Dataset_2022.10")
processed_dataset = processor.process_dataset(dataset_path)

print(f"Verarbeitet: {len(processed_dataset)} Audio-Dateien")
```

## 📊 Unterstützte Datasets

### Thorsten-Voice Dataset 2022.10 ✅
- **Format**: LJSpeech (WAV + CSV)
- **Größe**: 1.3 GB, 12.432 Utterances
- **Sprecher**: 1 (Single-Speaker)
- **Sample Rate**: 22kHz
- **Qualität**: Studio-Aufnahmen
- **Status**: ✅ Vollständig unterstützt

### Multilingual LibriSpeech German 🔄
- **Format**: MLS (OPUS + TXT)
- **Größe**: 29.6 GB, 476.805 Utterances
- **Sprecher**: 236 (Multi-Speaker)
- **Sample Rate**: 16kHz
- **Qualität**: Audiobook-Qualität
- **Status**: 🔄 In Entwicklung

## 🏗️ Architektur

### DataProcessor Framework
```
DataProcessor (Abstract Base)
├── TorstenVoiceDataProcessor    ✅ Implementiert
├── MLSGermanDataProcessor       🔄 In Entwicklung
└── CustomDataProcessor          📋 Erweiterbar
```

### Pipeline-Komponenten
```
Data Processing → Unsloth Training → Model Export → VLLM Deployment
      ✅              🔄               🔄            📋
```

## 📁 Projektstruktur

```
german-tts-finetuning/
├── src/                        # Source Code
│   ├── data_processor_base.py  # Abstrakte Basisklasse
│   ├── torsten_voice_processor.py  # Thorsten-Voice Processor
│   └── data_exploration.py     # Datenanalyse Tools
├── tests/                      # Unit Tests
├── docs/                       # Dokumentation
├── data/                       # Trainingsdaten (lokal)
├── models/                     # Trainierte Modelle
└── .kiro/specs/               # Projekt-Spezifikationen
```

## 🔧 Verwendung

### DataProcessor verwenden

```python
from src.torsten_voice_processor import TorstenVoiceDataProcessor
from src.data_processor_base import ProcessingConfig

# Konfiguration anpassen
config = ProcessingConfig(
    min_duration=1.0,           # Minimale Audio-Dauer (Sekunden)
    max_duration=10.0,          # Maximale Audio-Dauer (Sekunden)
    target_sample_rate=22050,   # Ziel-Sample-Rate
    min_snr=15.0,              # Minimum Signal-to-Noise Ratio (dB)
    quality_threshold=0.7,      # Qualitätsschwelle (0.0-1.0)
    batch_size=50              # Batch-Größe für Verarbeitung
)

# Processor erstellen und verwenden
processor = TorstenVoiceDataProcessor(config)
dataset = processor.process_dataset(Path("data/thorsten-voice/"))

# Statistiken anzeigen
stats = processor.get_processing_stats(dataset)
print(f"Verarbeitete Items: {stats['total_items']}")
print(f"Gesamtdauer: {stats['total_duration_hours']:.2f} Stunden")
print(f"Durchschnittliche Qualität: {stats['avg_quality_score']:.3f}")
```

### Qualitätsfilterung

Das System filtert automatisch basierend auf:
- **Audio-Dauer**: 1-10 Sekunden (konfigurierbar)
- **Signal-to-Noise Ratio**: >15 dB für Thorsten-Voice
- **RMS-Energie**: Ausreichende Lautstärke
- **Text-Audio-Alignment**: Optimiert für deutsche Sprache

## 📈 Performance

### Thorsten-Voice Dataset
- **Verarbeitungsgeschwindigkeit**: ~100 Dateien/Minute
- **Qualitätsfilterung**: ~85% der Dateien passieren Filter
- **Memory Usage**: ~2GB für komplettes Dataset
- **Empfohlene Hardware**: 8GB RAM, SSD Storage

## 🧪 Tests

```bash
# Unit Tests ausführen
python -m pytest tests/ -v

# Spezifische Tests
python -m pytest tests/test_torsten_voice_processor.py -v
python -m pytest tests/test_data_processor_base.py -v
```

## 📚 Dokumentation

- **[DataProcessor Usage Guide](docs/data_processor_usage.md)**: Detaillierte Anleitung
- **[Dataset Analysis Report](dataset_analysis_report.md)**: Datenanalyse-Ergebnisse
- **[Research Findings](research_findings.md)**: TTS Fine-tuning Best Practices
- **[Project Status](PROJECT_STATUS.md)**: Aktueller Entwicklungsstand

## 🔄 Entwicklungsstatus

**Aktueller Stand**: 5/15 Tasks abgeschlossen (33%)

### ✅ Abgeschlossen
- [x] Entwicklungsumgebung Setup
- [x] TTS Fine-tuning Recherche
- [x] Dataset-Exploration und -Analyse
- [x] DataProcessor Framework
- [x] TorstenVoiceDataProcessor

### 🔄 In Arbeit
- [ ] MLSGermanDataProcessor
- [ ] Unsloth-Integration
- [ ] TTS-Evaluation-Metriken

### 📋 Geplant
- [ ] Model Persistence für VLLM
- [ ] Pipeline-Orchestrierung
- [ ] Umfassende Tests und Dokumentation

## 🤝 Beitragen

1. **Issues**: Probleme und Feature-Requests über GitHub Issues
2. **Pull Requests**: Code-Beiträge willkommen
3. **Dokumentation**: Verbesserungen der Dokumentation
4. **Tests**: Zusätzliche Test-Cases

## 📄 Lizenz

[Lizenz-Information hier einfügen]

## 🙏 Danksagungen

- **Thorsten Müller** für das Thorsten-Voice Dataset
- **Unsloth Team** für das effiziente Fine-tuning Framework
- **Hugging Face** für Modell-Hosting und Tools
- **OpenSLR** für das MLS German Dataset

---

**Letztes Update**: 2025-01-17  
**Version**: 0.1.0-alpha  
**Status**: In aktiver Entwicklung