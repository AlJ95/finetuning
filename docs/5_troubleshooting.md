# 5. Troubleshooting Guide

## Common Issues by Category

### Setup Problems
**Error**: `locale.Error: unsupported locale setting`  
✅ Solution:  
```bash
export LC_ALL=de_DE.UTF-8
export LANG=de_DE.UTF-8
```

### Data Loading
**Error**: `Umlauts not recognized in text`  
✅ Solution: Ensure UTF-8 encoding:  
```python
with open("file.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

### Training Issues
**Error**: `Poor German pronunciation quality`  
✅ Solutions:  
- Increase `num_train_epochs` to ≥5  
- Set `learning_rate=1e-4`  
- Verify phonemizer language is set to 'de'

### Evaluation Problems
**Error**: `Low phoneme accuracy for German`  
✅ Check:  
1. Phoneme alphabet matches German IPA  
2. Audio sample rate is 24kHz  
3. Text preprocessing includes umlaut handling

### VLLM Deployment
**Error**: `Model not loading in VLLM`  
✅ Required:  
```python
model.save_pretrained_merged(
    "model",
    tokenizer,
    save_method="merged_16bit"  # Must use this for VLLM
)
```

## German TTS Specific Solutions

### Umlaut Processing
```python
# Normalization options:
text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
# OR
text = text.encode('utf-8').decode('ascii', 'ignore')
```

### Audio Artifacts
- **Symptom**: Metallic sounding German speech  
- **Fix**: Increase `target_sample_rate=24000` in audio config

## Diagnostic Tools
```bash
# Check German text processing:
python -c "print('Überprüfung'.encode('utf-8'))"

# Verify audio properties:
python test_whisperx_integration.py --check-audio
```

## Getting Help
For unresolved issues:  
📌 [Open GitHub Issue](https://github.com/AlJ95/finetuning/issues)  
📧 Email support: german-tts-support@example.com

## Quick Links
← [Back to Setup Guide](1_setup.md)  
← [Pipeline Overview](2_pipeline.md)
