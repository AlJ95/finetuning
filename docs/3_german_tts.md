# 3. German TTS Specifics

## Before You Start
✔ Completed environment setup  
✔ Understand basic pipeline flow  

## Language Configuration

### Phoneme Handling
```python
# Required for proper German pronunciation
from phonemizer import phonemize
text = "Schön, dass du da bist"
phonemes = phonemize(text, language='de', backend='espeak')
```

### Umlaut Processing
```python
# Normalize German umlauts
text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
```

## Audio Parameters
- **Sample Rate**: 24kHz (optimal for German speech)
- **Bit Depth**: 16-bit
- **Silence Padding**: 100ms before/after utterances

## Training Recommendations
```python
trainer = UnslothTrainer(
    language="de",  # Set language explicitly
    phoneme_alphabet="de-ipa",  # German IPA symbols
    learning_rate=2e-4  # Lower rate for German
)
```

## Evaluation Metrics
- **Phoneme Accuracy**: Should be >90% for German
- **MOS Target**: ≥3.8 for naturalness
- **Word Error Rate**: <5% for clean speech

## What's Next
→ [Advanced Configuration](4_advanced.md)  
→ [Troubleshooting German TTS](5_troubleshooting.md#german-tts)
