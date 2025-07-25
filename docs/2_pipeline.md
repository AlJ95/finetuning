# 2. Pipeline Overview

## Before You Start
✔ Completed environment setup  
✔ Basic understanding of TTS concepts  

## Pipeline Stages

### 1. Data Loading
```python
from src.torsten_voice_processor import TorstenVoiceDataProcessor
processor = TorstenVoiceDataProcessor()
dataset = processor.load_dataset("path/to/thorsten_voice")
```

### 2. Preprocessing
```python
# German-specific preprocessing
cleaned_data = processor.preprocess(
    dataset,
    umlaut_handling="normalize",  # Special German handling
    min_duration=0.5  
)
```

### 3. Training
```python
from src.unsloth_trainer import UnslothTrainer
trainer = UnslothTrainer()
model = trainer.train(cleaned_data)
```

### 4. Evaluation
```python
from src.tts_evaluator import TTSEvaluator
evaluator = TTSEvaluator(language="de")  # German evaluation
results = evaluator.evaluate(model)
```

### 5. Persistence
```python
from src.model_persistence import ModelPersistence
persistence = ModelPersistence()
persistence.save_model(model, "german_tts_model")
```

## German TTS Pipeline Notes
⚠️ Key considerations:
- Set `language="de"` in all components
- Use `phonemizer[de]` for text preprocessing
- Adjust sample rate to 24kHz for German speech

## What's Next
→ [German TTS Specifics](3_german_tts.md)  
→ [Troubleshooting Pipeline Issues](5_troubleshooting.md#pipeline)
