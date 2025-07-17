# DataProcessor Abstract Base Class Usage Guide

## Overview

The `DataProcessor` abstract base class provides a standardized framework for processing German audio datasets for TTS fine-tuning. It handles audio metadata extraction, quality validation, batch processing, and text-audio alignment.

## Key Features

- **Abstract Interface**: Standardized methods for different dataset formats
- **Quality Validation**: Audio quality metrics and filtering
- **Batch Processing**: Parallel processing with progress tracking
- **Text-Audio Alignment**: Validation of transcript-audio synchronization
- **Configurable**: Flexible configuration for different use cases

## Core Components

### Data Models

```python
@dataclass
class AudioDataset:
    """Data model for audio dataset entries."""
    file_path: Path
    text_transcript: str
    duration: float
    sample_rate: int
    quality_score: float
    metadata: Dict[str, Any]

@dataclass
class QualityMetrics:
    """Quality metrics for audio validation."""
    signal_to_noise_ratio: float
    duration_seconds: float
    sample_rate: int
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float

@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    min_duration: float = 0.5  # seconds
    max_duration: float = 30.0  # seconds
    target_sample_rate: int = 22050
    min_snr: float = 10.0  # dB
    max_workers: int = 4
    batch_size: int = 100
    quality_threshold: float = 0.6
```

### Abstract Methods

When extending `DataProcessor`, you must implement these abstract methods:

#### `load_dataset_metadata(dataset_path: Path) -> List[Dict[str, Any]]`
Load dataset metadata from the specific dataset format.

```python
def load_dataset_metadata(self, dataset_path: Path) -> List[Dict[str, Any]]:
    """Load metadata from your dataset format (CSV, JSON, etc.)"""
    # Implementation specific to your dataset
    pass
```

#### `parse_transcript(metadata: Dict[str, Any]) -> str`
Parse and clean transcript text from dataset-specific metadata.

```python
def parse_transcript(self, metadata: Dict[str, Any]) -> str:
    """Extract and clean transcript from metadata"""
    # Implementation specific to your dataset format
    pass
```

## Usage Example

### 1. Create a Concrete Implementation

```python
from pathlib import Path
from typing import Dict, List, Any
from src.data_processor_base import DataProcessor, ProcessingConfig

class MyDatasetProcessor(DataProcessor):
    def load_dataset_metadata(self, dataset_path: Path) -> List[Dict[str, Any]]:
        # Load from your dataset format (CSV, JSON, directory structure, etc.)
        metadata_list = []
        
        # Example: Load from CSV file
        csv_file = dataset_path / "metadata.csv"
        with open(csv_file, 'r') as f:
            # Parse CSV and create metadata dictionaries
            pass
            
        return metadata_list
    
    def parse_transcript(self, metadata: Dict[str, Any]) -> str:
        # Extract and clean transcript text
        transcript = metadata.get('text', '')
        
        # Apply dataset-specific cleaning
        transcript = transcript.strip()
        transcript = transcript.replace('\n', ' ')
        
        return transcript
```

### 2. Configure Processing Parameters

```python
config = ProcessingConfig(
    min_duration=1.0,        # Minimum audio duration in seconds
    max_duration=15.0,       # Maximum audio duration in seconds
    min_snr=12.0,           # Minimum signal-to-noise ratio in dB
    quality_threshold=0.7,   # Minimum quality score (0.0-1.0)
    max_workers=4,          # Number of parallel workers
    batch_size=100          # Batch size for processing
)
```

### 3. Process Your Dataset

```python
# Create processor instance
processor = MyDatasetProcessor(config)

# Process entire dataset
dataset_path = Path("path/to/your/dataset")
processed_dataset = processor.process_dataset(dataset_path)

# Get processing statistics
stats = processor.get_processing_stats(processed_dataset)
print(f"Processed {stats['total_items']} items")
print(f"Total duration: {stats['total_duration_hours']:.2f} hours")
print(f"Average quality: {stats['avg_quality_score']:.3f}")
```

## Built-in Methods

The base class provides several built-in methods you can use:

### Audio Analysis
- `extract_audio_metadata(audio_file)`: Extract technical metadata from audio files
- `calculate_quality_metrics(audio_data, sample_rate)`: Calculate quality metrics
- `validate_audio_quality(audio_file)`: Validate audio against quality thresholds
- `align_text_audio(text, audio_file)`: Calculate text-audio alignment score

### Processing
- `process_batch(file_batch, metadata_batch)`: Process a batch of files
- `filter_dataset(dataset)`: Filter dataset based on quality thresholds
- `process_dataset(dataset_path)`: Process entire dataset with progress tracking

### Statistics
- `get_processing_stats(dataset)`: Generate comprehensive processing statistics

## Quality Validation

The processor automatically validates audio files based on:

1. **Duration**: Must be within `min_duration` and `max_duration`
2. **Signal-to-Noise Ratio**: Must exceed `min_snr` threshold
3. **Energy Level**: Must have sufficient RMS energy
4. **Spectral Quality**: Must have valid spectral characteristics

## Text-Audio Alignment

The alignment score is calculated based on:
- Characters per second ratio
- Comparison to typical German speech rate (12-15 chars/sec)
- Optimal rate target of 13.5 characters per second

## Error Handling

The processor includes comprehensive error handling:
- Individual file processing errors don't stop batch processing
- Detailed logging of errors and warnings
- Graceful degradation for missing or corrupted files

## Performance Considerations

- Use appropriate `max_workers` based on your system capabilities
- Adjust `batch_size` based on available memory
- Consider using SSD storage for better I/O performance
- Monitor memory usage with large datasets

## Example Output

```
2024-01-15 10:30:15 - MyDatasetProcessor - INFO - Starting dataset processing: /path/to/dataset
2024-01-15 10:30:15 - MyDatasetProcessor - INFO - Found 1000 items in dataset
Processing batches: 100%|██████████| 10/10 [00:45<00:00,  4.5s/batch]
2024-01-15 10:31:00 - MyDatasetProcessor - INFO - Processed 950 valid items
2024-01-15 10:31:00 - MyDatasetProcessor - INFO - Filtered dataset: 850/950 items passed quality threshold
2024-01-15 10:31:00 - MyDatasetProcessor - INFO - Dataset processing complete: 850 final items
```

## Concrete Implementations

### TorstenVoiceDataProcessor

A specialized implementation for the Thorsten-Voice German TTS dataset (LJSpeech format):

```python
from src.torsten_voice_processor import TorstenVoiceDataProcessor

# Create processor with Thorsten-specific defaults
processor = TorstenVoiceDataProcessor()

# Or with custom configuration
config = ProcessingConfig(
    min_duration=1.0,        # Thorsten has some short utterances
    max_duration=10.0,       # Based on analysis: max ~7.7s
    target_sample_rate=22050, # Thorsten's native sample rate
    min_snr=15.0,           # Higher threshold for studio-quality
    quality_threshold=0.7,   # Higher threshold for single-speaker
    batch_size=50           # Smaller batches for detailed processing
)
processor = TorstenVoiceDataProcessor(config)

# Process Thorsten-Voice dataset
dataset_path = Path("path/to/ThorstenVoice-Dataset_2022.10")
processed_dataset = processor.process_dataset(dataset_path)
```

**Features:**
- LJSpeech format parsing (pipe-separated CSV)
- German text normalization and validation
- Enhanced quality validation for studio recordings
- Optimized alignment scoring for German speech patterns
- Thorsten-specific configuration defaults

**Dataset Structure Expected:**
```
ThorstenVoice-Dataset_2022.10/
├── metadata.csv          # Pipe-separated: filename|transcript|normalized_transcript
├── wavs/                 # Audio files directory
│   ├── thorsten_001.wav
│   ├── thorsten_002.wav
│   └── ...
└── README.md
```

## Best Practices

1. **Test with Small Datasets**: Start with a small subset to validate your implementation
2. **Monitor Quality Scores**: Adjust thresholds based on your dataset characteristics
3. **Validate Transcripts**: Ensure transcript parsing handles your dataset format correctly
4. **Check Audio Formats**: Verify all audio files are in supported formats
5. **Resource Management**: Monitor CPU and memory usage during processing
6. **German Text Handling**: For German datasets, ensure proper handling of umlauts (ä, ö, ü, ß)
7. **Sample Rate Consistency**: Verify all audio files have consistent sample rates