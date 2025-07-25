# 4. Advanced Configuration

## Before You Start
✔ Completed basic pipeline setup  
✔ Understand German TTS fundamentals  

## Custom Training Options

### Hyperparameter Tuning
```python
from src.unsloth_trainer import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,  # Lower for fine-tuning
    batch_size=2,       # Reduce for German speech clarity
    num_train_epochs=5, # Longer training for German
    use_unsloth_optimizations=True
)
```

### LoRA Configuration
```python
config.target_modules.extend([
    "dense",           # Additional modules for German
    "layer_norm"       # Better accent handling
])
```

## Model Export Options

### VLLM Compatibility
```python
from src.model_persistence import ModelPersistence

persistence = ModelPersistence()
persistence.save_model(
    model,
    "german_tts_vllm",
    save_method="merged_16bit"  # Required for VLLM
)
```

### Quantization
```python
# 4-bit quantization for German TTS
model.save_pretrained_merged(
    "german_tts_quantized",
    tokenizer,
    save_method="merged_4bit"
)
```

## Performance Optimization

### Memory Management
```python
# Enable memory-efficient German training
trainer = UnslothTrainer(
    use_gradient_checkpointing=True,
    enable_xformers=True  # For German long sequences
)
```

## What's Next
→ [Troubleshooting Advanced Issues](5_troubleshooting.md#advanced)  
→ [Back to Pipeline Overview](2_pipeline.md)
