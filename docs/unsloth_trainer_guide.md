# Unsloth Trainer Guide

## Overview

The UnslothTrainer provides memory-efficient fine-tuning of the Orpheus 3B model for German TTS using Unsloth optimizations. This guide covers setup, configuration, and usage for German text-to-speech fine-tuning.

## Features

- **Memory-Efficient Training**: Uses Unsloth optimizations for reduced VRAM usage
- **LoRA Fine-tuning**: Parameter-efficient training with Low-Rank Adaptation
- **German TTS Optimization**: Specialized prompt formatting for German speech synthesis
- **VLLM Compatibility**: Export models in VLLM-compatible format
- **Comprehensive Monitoring**: Memory usage tracking and training metrics
- **Audio Quality Filtering**: Automatic filtering by duration and quality scores

## Installation Requirements

```bash
# Install Unsloth (requires specific PyTorch version)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Additional dependencies
pip install transformers datasets trl torch torchaudio
```

**Note**: Unsloth works best with PyTorch 2.3.x. Newer versions may have compatibility issues.

## Quick Start

```python
from src.unsloth_trainer import UnslothTrainer, TrainingConfig
from src.data_processor_base import AudioDataset

# Configure training
config = TrainingConfig(
    model_name="unsloth/orpheus-3b-0.1-ft",
    output_dir="outputs/german_tts",
    num_train_epochs=3,
    learning_rate=2e-4
)

# Initialize trainer
trainer = UnslothTrainer(config)

# Load Orpheus 3B model
model, tokenizer = trainer.load_orpheus_model()

# Prepare your dataset (list of AudioDataset objects)
# dataset = load_your_german_audio_data()

# Start training
# results = trainer.start_finetuning(dataset)
```

## Configuration Options

### TrainingConfig Parameters

#### Model Configuration
- `model_name`: Orpheus 3B model identifier (default: "unsloth/orpheus-3b-0.1-ft")
- `max_seq_length`: Maximum sequence length (default: 2048)
- `dtype`: Data type for model weights (auto-detected)
- `load_in_4bit`: Enable 4-bit quantization (default: False for TTS quality)

#### LoRA Configuration
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA scaling parameter (default: 16)
- `lora_dropout`: Dropout rate for LoRA layers (default: 0.0)
- `target_modules`: Modules to apply LoRA to (default: attention and MLP layers)

#### Training Parameters
- `per_device_train_batch_size`: Batch size per device (default: 1)
- `gradient_accumulation_steps`: Steps to accumulate gradients (default: 4)
- `learning_rate`: Learning rate (default: 2e-4)
- `num_train_epochs`: Number of training epochs (default: 3)
- `optim`: Optimizer type (default: "adamw_8bit")

#### Audio-Specific Parameters
- `target_sample_rate`: Target audio sample rate (default: 24000 Hz)
- `max_audio_length`: Maximum audio duration in seconds (default: 30.0)
- `min_audio_length`: Minimum audio duration in seconds (default: 0.5)

## Memory Optimization Features

### Unsloth Optimizations
- **FastLanguageModel**: Optimized model loading and inference
- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **8-bit AdamW**: Memory-efficient optimizer
- **LoRA**: Parameter-efficient fine-tuning

### Audio Processing Optimizations
- **Length Filtering**: Automatically filters audio by duration to prevent OOM
- **Quality Filtering**: Uses quality scores to select best training samples
- **Batch Size Optimization**: Configured for TTS workloads
- **Memory Monitoring**: Real-time GPU memory tracking

## German TTS Specific Features

### Prompt Formatting
The trainer uses specialized prompt formatting optimized for German TTS:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a high-quality German text-to-speech system. Generate natural, expressive German speech with proper pronunciation and intonation.<|eot_id|><|start_header_id|>user<|end_header_id|>

{german_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Audio Quality Considerations
- Sample rate fixed at 24kHz for Orpheus 3B compatibility
- Duration filtering to maintain consistent training batches
- Quality score integration for sample selection

## Usage Examples

### Basic Training Setup

```python
import logging
from pathlib import Path
from src.unsloth_trainer import UnslothTrainer, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

# Configure for German TTS
config = TrainingConfig(
    model_name="unsloth/orpheus-3b-0.1-ft",
    output_dir="outputs/german_tts_model",
    
    # Training parameters
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    
    # Audio parameters
    target_sample_rate=24000,
    max_audio_length=30.0,
    min_audio_length=0.5,
    
    # LoRA parameters
    r=16,
    lora_alpha=16,
    lora_dropout=0.0
)

# Initialize trainer
trainer = UnslothTrainer(config)

# Check memory stats
memory_stats = trainer.get_memory_stats()
print(f"Available GPU memory: {memory_stats.get('gpu_memory_total_gb', 'N/A')} GB")
```

### Loading and Preparing Data

```python
# Load German datasets
from src.mls_german_processor import MLSGermanDataProcessor
from src.torsten_voice_processor import TorstenVoiceDataProcessor

# Process MLS German dataset
mls_processor = MLSGermanDataProcessor()
mls_data = mls_processor.process_dataset("path/to/mls_german_opus.tar.gz")

# Process Torsten Voice dataset
torsten_processor = TorstenVoiceDataProcessor()
torsten_data = torsten_processor.process_dataset("path/to/torstenvoicedataset2022.10.zip")

# Combine datasets
combined_data = mls_data + torsten_data

# Load model and start training
model, tokenizer = trainer.load_orpheus_model()
results = trainer.start_finetuning(combined_data)

print(f"Training completed in {results.training_time:.2f} seconds")
print(f"Final loss: {results.final_loss:.4f}")
```

### Saving for VLLM Deployment

```python
# Save model in VLLM-compatible format
vllm_path = "outputs/german_tts_vllm"
trainer.save_model_for_vllm(vllm_path, save_method="merged_16bit")

# Validate model loading
is_valid = trainer.validate_model_loading(vllm_path)
print(f"Model validation: {'✓ Passed' if is_valid else '✗ Failed'}")
```

## Integration with Data Processors

The UnslothTrainer integrates seamlessly with the data processors:

```python
# Using with TorstenVoiceDataProcessor
from src.torsten_voice_processor import TorstenVoiceDataProcessor

processor = TorstenVoiceDataProcessor()
dataset = processor.process_dataset("path/to/torsten_dataset.zip")

# Filter high-quality samples
high_quality = [item for item in dataset if item.quality_score > 0.8]

# Train with filtered data
trainer = UnslothTrainer()
trainer.load_orpheus_model()
results = trainer.start_finetuning(high_quality)
```

## Memory Management

### GPU Memory Monitoring

```python
# Check memory before training
initial_stats = trainer.get_memory_stats()
print(f"GPU Memory before: {initial_stats['gpu_memory_allocated_gb']:.2f} GB")

# Training automatically tracks memory usage
results = trainer.start_finetuning(dataset)

# Check final memory stats
final_stats = results.memory_stats
print(f"Peak GPU Memory: {final_stats['memory_peak_gb']:.2f} GB")
```

### Memory Optimization Tips

1. **Batch Size**: Start with batch_size=1 and increase gradually
2. **Sequence Length**: Reduce max_seq_length if experiencing OOM
3. **Audio Length**: Filter very long audio samples
4. **Gradient Accumulation**: Increase steps to simulate larger batches
5. **LoRA Rank**: Lower rank (r=8) uses less memory

## Troubleshooting

### Common Issues

#### PyTorch Version Compatibility
```
WARNING: PyTorch 2.7.1+cu128 may not be compatible with current Unsloth version
```
**Solution**: Downgrade to PyTorch 2.3.x:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

#### Out of Memory Errors
**Solutions**:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Filter longer audio samples
- Enable gradient checkpointing

#### Model Loading Errors
**Check**:
- Unsloth installation is correct
- Model name is valid
- Sufficient disk space for model download
- Internet connection for model download

### Performance Optimization

#### Training Speed
- Use `bf16=True` on supported hardware
- Enable gradient checkpointing
- Optimize batch size for your GPU
- Use 8-bit optimizer

#### Memory Efficiency
- Use LoRA instead of full fine-tuning
- Filter audio by length and quality
- Remove unused dataset columns
- Enable gradient checkpointing

## Best Practices

### Data Preparation
1. **Quality Filtering**: Use quality scores > 0.8
2. **Duration Filtering**: Keep audio between 0.5-30 seconds
3. **Text Cleaning**: Ensure clean German transcripts
4. **Sample Rate**: Standardize to 24kHz

### Training Configuration
1. **Learning Rate**: Start with 2e-4, adjust based on loss curves
2. **LoRA Rank**: Use r=16 for good quality/efficiency balance
3. **Batch Size**: Start small (1) and increase if memory allows
4. **Epochs**: 3-5 epochs typically sufficient for TTS

### Model Validation
1. **Save Checkpoints**: Regular saving during training
2. **Validation Set**: Hold out data for evaluation
3. **Model Testing**: Validate loading before deployment
4. **VLLM Export**: Test VLLM compatibility

## API Reference

### UnslothTrainer Class

#### Methods

- `__init__(config: TrainingConfig)`: Initialize trainer
- `load_orpheus_model() -> Tuple[Model, Tokenizer]`: Load Orpheus 3B
- `prepare_dataset_for_unsloth(dataset: List[AudioDataset]) -> Dataset`: Prepare HF dataset
- `format_training_data(dataset: Dataset) -> Dataset`: Format for TTS training
- `configure_training() -> SFTTrainer`: Setup SFT trainer
- `start_finetuning(dataset: List[AudioDataset]) -> TrainingResults`: Start training
- `save_model_for_vllm(output_path: str, save_method: str) -> str`: Save for VLLM
- `get_memory_stats() -> Dict[str, Any]`: Get memory statistics
- `validate_model_loading(model_path: str) -> bool`: Validate saved model

### TrainingConfig Class

Dataclass containing all training configuration parameters. See Configuration Options section for details.

### TrainingResults Class

Contains training results and metrics:
- `final_loss`: Final training loss
- `training_time`: Total training time in seconds
- `total_steps`: Number of training steps completed
- `model_path`: Path to saved model
- `tokenizer_path`: Path to saved tokenizer
- `training_logs`: List of training log entries
- `memory_stats`: Memory usage statistics

## Integration Examples

### Complete Pipeline Example

```python
#!/usr/bin/env python3
"""Complete German TTS fine-tuning pipeline."""

import logging
from pathlib import Path
from src.unsloth_trainer import UnslothTrainer, TrainingConfig
from src.mls_german_processor import MLSGermanDataProcessor

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configuration
    config = TrainingConfig(
        output_dir="outputs/german_tts_final",
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4
    )
    
    # Process data
    logger.info("Processing German audio data...")
    processor = MLSGermanDataProcessor()
    dataset = processor.process_dataset("data/mls_german_opus.tar.gz")
    
    # Filter high-quality samples
    filtered_data = [item for item in dataset if item.quality_score > 0.8]
    logger.info(f"Using {len(filtered_data)} high-quality samples")
    
    # Initialize trainer
    trainer = UnslothTrainer(config)
    
    # Load model
    logger.info("Loading Orpheus 3B model...")
    model, tokenizer = trainer.load_orpheus_model()
    
    # Start training
    logger.info("Starting fine-tuning...")
    results = trainer.start_finetuning(filtered_data)
    
    # Save for VLLM
    logger.info("Saving model for VLLM...")
    vllm_path = "outputs/german_tts_vllm"
    trainer.save_model_for_vllm(vllm_path)
    
    # Validate
    is_valid = trainer.validate_model_loading(vllm_path)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final loss: {results.final_loss:.4f}")
    logger.info(f"Training time: {results.training_time:.2f}s")
    logger.info(f"Model validation: {'✓' if is_valid else '✗'}")

if __name__ == "__main__":
    main()
```

This guide provides comprehensive coverage of the UnslothTrainer implementation for German TTS fine-tuning with Orpheus 3B.