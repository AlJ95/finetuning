# German TTS Dataset Analysis Report

**Exploration Date:** 2025-01-17  
**Location:** D:\Trainingsdaten\TTS

## Executive Summary

Two German TTS datasets were analyzed for compatibility with Unsloth TTS training using the Orpheus 3B model:

1. **Thorsten Voice Dataset 2022.10** - Single-speaker, high-quality German TTS dataset
2. **Multilingual LibriSpeech German** - Multi-speaker, large-scale German speech dataset

**Recommendation:** Start with Thorsten Voice Dataset for initial single-speaker TTS training, then experiment with MLS German for multi-speaker capabilities.

## Dataset Comparison

| Metric | Thorsten Voice | MLS German |
|--------|----------------|------------|
| **Size** | 1.3 GB | 29.6 GB |
| **Utterances** | 12,432 | 476,805 |
| **Speakers** | 1 (single-speaker) | 236 (multi-speaker) |
| **Format** | LJSpeech (WAV) | MLS (OPUS) |
| **Sample Rate** | 22kHz | 16kHz |
| **Audio Quality** | High (studio quality) | Good (audiobook quality) |
| **Text Length** | 5-157 chars (avg: 54) | 9-617 chars (avg: 177) |

## Thorsten Voice Dataset Analysis

### Dataset Characteristics
- **Format:** LJSpeech-compatible structure
- **Audio Files:** 12,449 WAV files
- **Total Duration:** ~13 hours (estimated from samples)
- **Sample Rate:** 22kHz (optimal for TTS)
- **Channels:** Mono (TTS-appropriate)
- **Text Quality:** Clean German text with proper punctuation

### Audio Quality Assessment
- **Duration Range:** 1.5 - 7.7 seconds per utterance
- **Average Duration:** 3.8 seconds
- **Format:** WAV (uncompressed, high quality)
- **Consistency:** Single speaker, consistent recording conditions

### Text Analysis
- **Character Distribution:** Proper German language distribution
- **Special Characters:** Includes umlauts (ä, ö, ü) and ß
- **Punctuation:** Standard German punctuation patterns
- **Sample Texts:**
  - "und überzeugen dank feingefühl für den ganz großen leinwand-stoff."
  - "zur schadenshöhe gab es keine angaben."
  - "außerdem können glasscheiben, wände und andere hindernisse das ergebnis beeinflussen."

### Unsloth Compatibility
✅ **Highly Compatible**
- Standard sample rate (22kHz)
- Mono audio format
- Clean metadata structure
- Manageable dataset size for training

**Required Preprocessing:**
1. Parse LJSpeech metadata format (pipe-separated)
2. Load and validate audio files
3. Create train/validation splits
4. Convert to Unsloth format: `{'audio': audio_array, 'text': transcript}`

## MLS German Dataset Analysis

### Dataset Characteristics
- **Format:** Multilingual LibriSpeech structure
- **Audio Files:** 476,805 OPUS files
- **Total Duration:** ~1,900 hours (estimated)
- **Sample Rate:** 16kHz
- **Channels:** Mono
- **Speakers:** 236 unique speakers

### Audio Quality Assessment
- **Duration Range:** 10.4 - 16.6 seconds per utterance
- **Average Duration:** 13.8 seconds
- **Format:** OPUS (compressed, good quality)
- **Variability:** Multi-speaker with varying recording conditions

### Text Analysis
- **Longer Utterances:** Average 177 characters (vs 54 for Thorsten)
- **Content:** Appears to be from audiobooks/literature
- **Language Quality:** Proper German with complex sentence structures
- **Speaker Distribution:** Uneven (some speakers have 100+ utterances, others <10)

### Unsloth Compatibility
✅ **Compatible with Preprocessing**
- Requires OPUS to WAV conversion
- Lower sample rate (16kHz) may need upsampling
- Multi-speaker structure needs special handling

**Required Preprocessing:**
1. Extract and convert OPUS files to WAV
2. Parse MLS transcript format (tab-separated)
3. Implement speaker conditioning (optional)
4. Balance speaker distribution
5. Standardize sample rates

## Quality Assessment

### Thorsten Voice Dataset
**Strengths:**
- Standard TTS sample rate (22kHz)
- Mono audio format suitable for TTS
- Large number of utterances for training
- Single speaker consistency
- Studio-quality recordings

**Concerns:**
- Some very short texts may not be useful
- Limited speaker diversity

**Recommendations:**
- Filter utterances by length (10-200 characters)
- Validate audio-text alignment
- Check for consistent speaker voice

### MLS German Dataset
**Strengths:**
- Multi-speaker dataset (236 speakers)
- Very large dataset for robust training
- Diverse content and speaking styles
- Good coverage of German language patterns

**Concerns:**
- Multi-speaker may require speaker conditioning
- OPUS format needs conversion to WAV
- Uneven speaker distribution
- Lower sample rate (16kHz vs 22kHz)

**Recommendations:**
- Consider single-speaker subset for initial training
- Convert OPUS to WAV format
- Implement speaker ID conditioning
- Balance speakers in training data

## Implementation Recommendations

### Phase 1: Thorsten Voice (Recommended Start)
1. **Advantages:**
   - Simpler single-speaker training
   - High audio quality
   - Standard TTS format
   - Faster training iterations

2. **Implementation Steps:**
   - Parse metadata.csv files
   - Load WAV files with librosa/soundfile
   - Create 80/10/10 train/val/test splits
   - Format for Unsloth TTS training

### Phase 2: MLS German (Advanced Experiments)
1. **Advantages:**
   - Multi-speaker capabilities
   - Large dataset for robust models
   - Diverse speaking styles

2. **Implementation Steps:**
   - Extract and convert OPUS files
   - Parse transcript files
   - Implement speaker embedding/conditioning
   - Balance dataset by speaker

### Unsloth Integration Strategy

**Memory Optimization:**
- Use Unsloth's 4-bit quantization
- Implement batch processing for large datasets
- Leverage gradient checkpointing

**Training Configuration:**
```python
training_config = {
    "model_name": "orpheus-3b",
    "learning_rate": 2e-4,
    "batch_size": 4,  # Adjust based on GPU memory
    "max_seq_length": 2048,
    "num_epochs": 3,
    "use_unsloth_optimizations": True
}
```

**Data Format for Unsloth:**
```python
dataset_format = {
    "audio": np.array,  # Audio waveform
    "text": str,        # German transcript
    "speaker_id": int   # Optional for MLS
}
```

## Technical Specifications

### File Formats
- **Thorsten:** WAV (PCM, 22kHz, mono)
- **MLS:** OPUS (compressed, 16kHz, mono)

### Metadata Formats
- **Thorsten:** CSV with pipe separation: `filename|transcript|normalized_transcript`
- **MLS:** Tab-separated: `audio_id\ttranscript`

### Directory Structures
- **Thorsten:** `ThorstenVoice-Dataset_2022.10/wavs/` + metadata files
- **MLS:** Hierarchical by speaker and chapter: `train/audio/de/[speaker]/[chapter]/`

## Conclusion

Both datasets are suitable for German TTS training with Unsloth and Orpheus 3B:

1. **Start with Thorsten Voice** for proof-of-concept and initial model development
2. **Scale to MLS German** for advanced multi-speaker capabilities
3. **Both datasets are Unsloth-compatible** with appropriate preprocessing
4. **Expected training time:** Thorsten (~2-4 hours), MLS (~20-40 hours) with Unsloth optimizations

The analysis confirms that both datasets meet the requirements for German TTS finetuning and are compatible with the Unsloth framework for efficient training of the Orpheus 3B model.