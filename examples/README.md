# German TTS Training Examples

This directory contains practical examples for training German Text-to-Speech models using the Thorsten-Voice dataset and Orpheus 3B.

## 📁 Files Overview

| File | Description | Complexity |
|------|-------------|------------|
| `quick_start.py` | Minimal 2-step training pipeline | ⭐ Beginner |
| `end_to_end_training.py` | Complete pipeline with all options | ⭐⭐⭐ Advanced |
| `mls_training.py` | Multi-speaker training with MLS German | ⭐⭐ Intermediate |

## 🚀 Quick Start

### Prerequisites
1. Download the Thorsten-Voice dataset:
   ```bash
   # Download from: https://www.thorsten-voice.de/en/datasets-2/
   # Extract to: data/thorsten-voice/
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Training
```bash
# Quick start with defaults
python examples/quick_start.py --dataset-path data/thorsten-voice

# End-to-end with custom settings
python examples/end_to_end_training.py \
    --dataset-path data/thorsten-voice \
    --output-dir my_models \
    --epochs 5 \
    --save-vllm
```

## 📊 Dataset Requirements

### Thorsten-Voice Dataset
- **Format**: LJSpeech (WAV + CSV)
- **Size**: ~1.3 GB, 12,432 utterances
- **Speaker**: Single speaker (Thorsten)
- **Sample Rate**: 22kHz
- **Language**: German

### Expected Directory Structure
```
data/thorsten-voice/
├── ThorstenVoice-Dataset_2022.10/
│   ├── metadata.csv
│   ├── metadata_train.csv
│   ├── metadata_dev.csv
│   └── wav/
│       ├── 00001.wav
│       ├── 00002.wav
│       └── ...
```

## 🎯 Training Examples

### 1. Quick Start (2 minutes)
```python
from examples.quick_start import quick_start

# Train with sensible defaults
results = quick_start("data/thorsten-voice")
```

### 2. Custom Training
```python
from src.torsten_voice_processor import TorstenVoiceDataProcessor
from src.unsloth_trainer import UnslothTrainer, TrainingConfig

# Process dataset
processor = TorstenVoiceDataProcessor()
dataset = processor.process_dataset("data/thorsten-voice")

# Configure training
config = TrainingConfig(
    num_train_epochs=5,
    learning_rate=3e-4,
    max_audio_length=15.0
)

# Train model
trainer = UnslothTrainer(config=config)
results = trainer.start_finetuning(dataset)
```

### 3. Advanced Configuration
```python
from src.data_processor_base import ProcessingConfig
from src.model_persistence import PersistenceConfig

# Custom processing
processing_config = ProcessingConfig(
    min_duration=2.0,
    max_duration=20.0,
    quality_threshold=0.8,
    target_sample_rate=24000
)

# VLLM-compatible saving
persistence_config = PersistenceConfig(
    vllm_compatible=True,
    save_merged_model=True,
    compression_level=4
)
```

## 🔧 Command Line Options

### quick_start.py
```bash
python examples/quick_start.py --help
# Options:
#   --dataset-path    Path to Thorsten-Voice dataset (required)
#   --output-dir      Output directory (default: outputs)
```

### end_to_end_training.py
```bash
python examples/end_to_end_training.py --help
# Options:
#   --dataset-path      Path to dataset (required)
#   --output-dir      Output directory (default: outputs)
#   --epochs          Training epochs (default: 3)
#   --batch-size      Batch size (default: 1)
#   --learning-rate   Learning rate (default: 2e-4)
#   --max-audio-length Maximum audio length (default: 10.0)
#   --min-audio-length Minimum audio length (default: 1.0)
#   --quality-threshold Quality threshold (default: 0.7)
#   --save-vllm       Save VLLM-compatible model
```

## 📈 Expected Results

### Training Metrics
- **Dataset Size**: ~12,000 samples after filtering
- **Training Time**: 2-4 hours on RTX 4090
- **Final Loss**: ~0.5-1.0
- **Memory Usage**: ~8-12 GB GPU memory

### Output Files
```
outputs/
├── lora_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer.json
├── training_logs/
│   └── training_20250117_143022.log
└── vllm_model/ (if --save-vllm used)
    ├── config.json
    ├── model.safetensors
    └── tokenizer.json
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Error: Unsloth not found**
   ```bash
   pip install unsloth
   ```

2. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 1`
   - Enable gradient checkpointing
   - Use 4-bit quantization

3. **Dataset Not Found**
   - Ensure correct path structure
   - Check file permissions
   - Verify dataset download

4. **Audio Processing Errors**
   - Check audio file formats
   - Verify sample rates
   - Check for corrupted files

### Performance Tips
- Use SSD storage for faster I/O
- Increase batch size if GPU memory allows
- Use multiple workers for data loading
- Monitor GPU memory usage

## 🎓 Next Steps

1. **Multi-speaker Training**: Try `mls_training.py` for multi-speaker models
2. **Custom Datasets**: Adapt processors for your own data
3. **Evaluation**: Use the evaluation scripts in `src/tts_evaluation/`
4. **Deployment**: Use VLLM-compatible models for production

## 📚 Additional Resources

- [Thorsten-Voice Dataset](https://www.thorsten-voice.de/en/datasets-2/)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Orpheus 3B Model](https://huggingface.co/unsloth/orpheus-3b-0.1-ft)
- [German TTS Best Practices](docs/research_findings.md)
