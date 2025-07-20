# DVC Pipeline Usage Guide

Diese Anleitung beschreibt die Verwendung der DVC-Pipeline für German TTS Fine-tuning mit Orpheus 3B und VLLM-Kompatibilität.

## Pipeline Übersicht

Die Pipeline besteht aus 5 sequenziellen Stages:

1. **Data Loading** - Lädt deutsche TTS-Datensätze (Thorsten Voice, MLS German)
2. **Preprocessing** - Filtert und bereitet Daten für Training vor
3. **Training** - Fine-tuning mit Unsloth und Orpheus 3B
4. **Evaluation** - Bewertung mit TTS-spezifischen Metriken
5. **Persistence** - VLLM-kompatible Modell-Speicherung

## Schnellstart

### Komplette Pipeline ausführen
```bash
# Vollständige Pipeline
python -m src.pipeline.run_pipeline

# Mit DVC (empfohlen)
dvc repro
```

### Einzelne Stages ausführen
```bash
# Nur Training Stage
python -m src.pipeline.run_pipeline --stage training

# Ab Training Stage
python -m src.pipeline.run_pipeline --from training

# Bis Evaluation Stage
python -m src.pipeline.run_pipeline --to evaluation

# Stages überspringen
python -m src.pipeline.run_pipeline --skip data_loading preprocessing
```

### Pipeline-Konfiguration

Parameter werden in `params.yaml` konfiguriert:

```yaml
# Datasets konfigurieren
data_loading:
  datasets: ["thorsten_voice", "mls_german"]
  data_paths:
    thorsten_voice: "data/raw/thorsten_voice"
    mls_german: "data/raw/mls_german"
  max_samples_per_dataset: 10000

# Training Parameter
training:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  learning_rate: 2e-4
  
# Model Persistence
persistence:
  vllm_compatible: true
  save_method: "merged_16bit"
  huggingface:
    upload_to_hub: false
    repo_name: "username/german-tts-orpheus"
```

## Stage Details

### 1. Data Loading Stage

**Zweck:** Lädt und validiert deutsche TTS-Datensätze

**Eingaben:**
- Rohe Audio-Dateien und Transkripte
- Dataset-Konfiguration aus `params.yaml`

**Ausgaben:**
- `data/processed/{dataset}/samples.pkl` - Verarbeitete Samples
- `data/processed/dataset_info.json` - Dataset-Statistiken

**Einzeln ausführen:**
```bash
python -m src.pipeline.run_pipeline --stage data_loading
```

### 2. Preprocessing Stage

**Zweck:** Qualitätsfilterung und Train/Val-Split

**Eingaben:**
- Verarbeitete Dataset-Samples
- Preprocessing-Parameter

**Ausgaben:**
- `data/preprocessed/train_dataset.pkl` - Training-Daten
- `data/preprocessed/val_dataset.pkl` - Validierungs-Daten
- `data/preprocessed/preprocessing_stats.json` - Preprocessing-Statistiken

**Konfiguration:**
```yaml
preprocessing:
  audio:
    min_duration: 1.0
    max_duration: 10.0
    quality_threshold: 0.7
  text:
    min_length: 10
    max_length: 500
  dataset_split:
    train_ratio: 0.8
    stratify_by_speaker: true
```

### 3. Training Stage

**Zweck:** Fine-tuning mit Unsloth und Orpheus 3B

**Eingaben:**
- Preprocessed Training/Validation Data
- Model und Training Parameter

**Ausgaben:**
- `models/lora_adapters/` - LoRA Adapter
- `models/training_logs/` - Training Logs
- `metrics/training_metrics.json` - Training Metriken

**Memory-Efficient Features:**
- Unsloth Optimierungen (1.5x schneller, 50% weniger VRAM)
- LoRA Fine-tuning
- Gradient Checkpointing

### 4. Evaluation Stage

**Zweck:** Umfassende TTS-Evaluation

**Eingaben:**
- Trainiertes Modell (LoRA Adapters)
- Validierungs-Daten

**Ausgaben:**
- `results/evaluation/detailed_results.json` - Detaillierte Ergebnisse
- `results/evaluation/evaluation_report.md` - Markdown Report
- `metrics/evaluation_metrics.json` - Evaluation Metriken

**Metriken:**
- MOS Score (Mean Opinion Score)
- PESQ (Perceptual Evaluation of Speech Quality)
- STOI (Short-Time Objective Intelligibility)
- Deutsche Phonem-Accuracy
- Inference Speed & Memory Usage

### 5. Persistence Stage

**Zweck:** VLLM-kompatible Modell-Speicherung

**Eingaben:**
- Trainiertes Modell mit LoRA Adapters
- Persistence-Konfiguration

**Ausgaben:**
- `models/vllm_compatible/` - VLLM-Ready Model (merged_16bit)
- `models/deployment_ready/` - Deployment Package
- `models/model_registry.json` - Model Registry

**VLLM Features:**
- Merged 16-bit Model für optimale VLLM Performance
- Automatische Kompatibilitäts-Validierung
- Deployment-ready Package mit Konfiguration
- Optional: Hugging Face Hub Upload

## Monitoring und Debugging

### Pipeline Status verfolgen
```bash
# DVC Pipeline Status
dvc status

# DVC Metriken anzeigen
dvc metrics show

# DVC Plots anzeigen
dvc plots show
```

### Logs und Ergebnisse
- **Pipeline Results:** `results/pipeline_results_*.json`
- **Stage Logs:** Automatisches Logging in jeder Stage
- **Training Logs:** `models/training_logs/`
- **Evaluation Report:** `results/evaluation/evaluation_report.md`

### Debugging einzelner Stages
```bash
# Einzelne Stage mit Debugging
python -c "
from src.pipeline.training_stage import TrainingStage
stage = TrainingStage()
result = stage.run()
print(result)
"
```

## VLLM Deployment

Nach erfolgreichem Pipeline-Durchlauf:

### 1. Modell-Validierung
```bash
# VLLM-Kompatibilität prüfen
python -c "
from src.model_persistence import ModelPersistence
mp = ModelPersistence()
compatible = mp.validate_vllm_compatibility('models/vllm_compatible')
print(f'VLLM Compatible: {compatible}')
"
```

### 2. VLLM Server starten
```bash
# Mit dem trainierten Modell
python -m vllm.entrypoints.openai.api_server \
    --model models/vllm_compatible \
    --dtype float16 \
    --max-model-len 2048
```

### 3. Deployment Package verwenden
```bash
# Deployment-ready Package nutzen
cd models/deployment_ready
cat deployment_config.json  # Konfiguration anzeigen
```

## Troubleshooting

### Häufige Probleme

**1. GPU Memory Issues**
```yaml
# params.yaml anpassen
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  load_in_4bit: true
```

**2. Dataset nicht gefunden**
```yaml
# Pfade in params.yaml korrigieren
data_loading:
  data_paths:
    thorsten_voice: "/absolute/path/to/dataset"
```

**3. VLLM Kompatibilitätsprobleme**
- Checke Model-Files in `models/vllm_compatible/`
- Validiere mit `validate_vllm_compatibility()`
- Logs in `models/training_logs/` prüfen

### Performance Optimierung

**Training Speed:**
- Unsloth Optimierungen aktiviert
- Gradient Checkpointing
- Mixed Precision Training

**Memory Usage:**
- LoRA statt Full Fine-tuning
- 4-bit Quantisierung optional
- Batch Size anpassen

**VLLM Inference:**
- Merged 16-bit Format für beste Performance
- Tensor Parallelism für Multi-GPU
- KV-Cache Optimierung

## Best Practices

1. **Parameter Tuning:** Beginne mit kleinen Datasets zum Testen
2. **Monitoring:** Nutze DVC Plots für Training-Visualisierung
3. **Checkpointing:** Pipeline kann an jedem Stage unterbrochen/fortgesetzt werden
4. **Validation:** Führe Evaluation Stage immer aus vor Deployment
5. **Documentation:** Pipeline generiert automatisch Reports und Dokumentation

## Support

Bei Problemen:
1. Pipeline-Logs in `results/` prüfen
2. Stage-spezifische Debugging-Informationen
3. DVC Status und Metriken validieren
4. Model Registry für Deployment-Status checken
