"""
Abstract base class for data processing in German TTS fine-tuning pipeline.

This module provides the foundation for processing different German audio datasets
with standardized interfaces for metadata extraction, quality validation, and batch processing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


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


class DataProcessor(ABC):
    """
    Abstract base class for processing German audio datasets.
    
    Provides standardized interfaces for:
    - Audio metadata extraction
    - Quality validation and filtering
    - Batch processing with progress tracking
    - Text-audio alignment validation
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize the data processor with configuration."""
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def load_dataset_metadata(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """
        Load dataset metadata from the specific dataset format.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            List of metadata dictionaries for each audio file
        """
        pass
    
    @abstractmethod
    def parse_transcript(self, metadata: Dict[str, Any]) -> str:
        """
        Parse transcript text from dataset-specific metadata.
        
        Args:
            metadata: Dataset-specific metadata dictionary
            
        Returns:
            Cleaned transcript text
        """
        pass
    
    def extract_audio_metadata(self, audio_file: Path) -> Dict[str, Any]:
        """
        Extract metadata from audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary containing audio metadata
        """
        try:
            import librosa
            import soundfile as sf
            
            # Load audio file info without loading the full audio
            info = sf.info(str(audio_file))
            
            # Load audio for additional analysis
            audio_data, sr = librosa.load(str(audio_file), sr=None)
            
            metadata = {
                'file_path': audio_file,
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
                'frames': info.frames,
                'audio_shape': audio_data.shape,
                'rms_energy': float(np.sqrt(np.mean(audio_data**2))),
                'max_amplitude': float(np.max(np.abs(audio_data))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio_data)[0]))
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {audio_file}: {e}")
            return {}
    
    def calculate_quality_metrics(self, audio_data: np.ndarray, sample_rate: int) -> QualityMetrics:
        """
        Calculate quality metrics for audio data.
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            QualityMetrics object with calculated metrics
        """
        try:
            import librosa
            
            # Calculate RMS energy
            rms_energy = float(np.sqrt(np.mean(audio_data**2)))
            
            # Estimate SNR (simplified approach)
            # Use the ratio of signal power to noise floor estimation
            signal_power = np.mean(audio_data**2)
            noise_floor = np.percentile(audio_data**2, 10)  # Bottom 10% as noise estimate
            snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
            
            # Zero crossing rate
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)[0]))
            
            # Spectral centroid
            spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate)[0]))
            
            return QualityMetrics(
                signal_to_noise_ratio=float(snr),
                duration_seconds=len(audio_data) / sample_rate,
                sample_rate=sample_rate,
                rms_energy=rms_energy,
                zero_crossing_rate=zcr,
                spectral_centroid=spectral_centroid
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {e}")
            # Return default metrics on error
            return QualityMetrics(
                signal_to_noise_ratio=0.0,
                duration_seconds=len(audio_data) / sample_rate,
                sample_rate=sample_rate,
                rms_energy=0.0,
                zero_crossing_rate=0.0,
                spectral_centroid=0.0
            )
    
    def validate_audio_quality(self, audio_file: Path) -> Tuple[bool, QualityMetrics]:
        """
        Validate audio quality against configured thresholds.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        try:
            import librosa
            
            # Load audio
            audio_data, sr = librosa.load(str(audio_file), sr=None)
            
            # Calculate quality metrics
            metrics = self.calculate_quality_metrics(audio_data, sr)
            
            # Validate against thresholds
            is_valid = (
                self.config.min_duration <= metrics.duration_seconds <= self.config.max_duration
                and metrics.signal_to_noise_ratio >= self.config.min_snr
                and metrics.rms_energy > 0.001  # Minimum energy threshold
                and not np.isnan(metrics.spectral_centroid)
            )
            
            return is_valid, metrics
            
        except Exception as e:
            self.logger.error(f"Error validating audio quality for {audio_file}: {e}")
            return False, QualityMetrics(0.0, 0.0, 0, 0.0, 0.0, 0.0)
    
    def align_text_audio(self, text: str, audio_file: Path) -> float:
        """
        Calculate text-audio alignment score.
        
        Args:
            text: Transcript text
            audio_file: Path to audio file
            
        Returns:
            Alignment score (0.0 to 1.0, higher is better)
        """
        try:
            import librosa
            
            # Load audio to get duration
            audio_data, sr = librosa.load(str(audio_file), sr=None)
            duration = len(audio_data) / sr
            
            # Simple heuristic: characters per second
            # Typical German speech: 12-15 characters per second
            chars_per_second = len(text) / duration
            
            # Score based on how close to typical speech rate
            optimal_rate = 13.5  # characters per second
            rate_diff = abs(chars_per_second - optimal_rate)
            
            # Convert to 0-1 score (closer to optimal = higher score)
            alignment_score = max(0.0, 1.0 - (rate_diff / optimal_rate))
            
            return float(alignment_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating text-audio alignment: {e}")
            return 0.0
    
    def process_batch(self, file_batch: List[Path], metadata_batch: List[Dict[str, Any]]) -> List[AudioDataset]:
        """
        Process a batch of audio files with their metadata.
        
        Args:
            file_batch: List of audio file paths
            metadata_batch: List of corresponding metadata dictionaries
            
        Returns:
            List of processed AudioDataset objects
        """
        processed_items = []
        
        for audio_file, metadata in zip(file_batch, metadata_batch):
            try:
                # Extract transcript
                transcript = self.parse_transcript(metadata)
                
                # Validate quality
                is_valid, quality_metrics = self.validate_audio_quality(audio_file)
                
                if not is_valid:
                    self.logger.debug(f"Skipping {audio_file} due to quality issues")
                    continue
                
                # Calculate alignment score
                alignment_score = self.align_text_audio(transcript, audio_file)
                
                # Extract additional metadata
                audio_metadata = self.extract_audio_metadata(audio_file)
                
                # Create AudioDataset object
                dataset_item = AudioDataset(
                    file_path=audio_file,
                    text_transcript=transcript,
                    duration=quality_metrics.duration_seconds,
                    sample_rate=quality_metrics.sample_rate,
                    quality_score=alignment_score,
                    metadata={
                        **audio_metadata,
                        'quality_metrics': quality_metrics,
                        'alignment_score': alignment_score,
                        'original_metadata': metadata
                    }
                )
                
                processed_items.append(dataset_item)
                
            except Exception as e:
                self.logger.error(f"Error processing {audio_file}: {e}")
                continue
        
        return processed_items
    
    def filter_dataset(self, dataset: List[AudioDataset]) -> List[AudioDataset]:
        """
        Filter dataset based on quality thresholds.
        
        Args:
            dataset: List of AudioDataset objects
            
        Returns:
            Filtered list of AudioDataset objects
        """
        filtered_dataset = []
        
        for item in dataset:
            if item.quality_score >= self.config.quality_threshold:
                filtered_dataset.append(item)
            else:
                self.logger.debug(f"Filtered out {item.file_path} (quality: {item.quality_score:.3f})")
        
        self.logger.info(f"Filtered dataset: {len(filtered_dataset)}/{len(dataset)} items passed quality threshold")
        return filtered_dataset
    
    def process_dataset(self, dataset_path: Path) -> List[AudioDataset]:
        """
        Process entire dataset with batch processing and progress tracking.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            List of processed and filtered AudioDataset objects
        """
        self.logger.info(f"Starting dataset processing: {dataset_path}")
        
        # Load dataset metadata
        metadata_list = self.load_dataset_metadata(dataset_path)
        self.logger.info(f"Found {len(metadata_list)} items in dataset")
        
        # Process in batches with parallel execution
        all_processed = []
        
        # Create batches
        batches = []
        for i in range(0, len(metadata_list), self.config.batch_size):
            batch_metadata = metadata_list[i:i + self.config.batch_size]
            batch_files = [Path(item['file_path']) for item in batch_metadata]
            batches.append((batch_files, batch_metadata))
        
        # Process batches with progress bar
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self.process_batch, files, metadata): (files, metadata)
                for files, metadata in batches
            }
            
            # Process completed batches with progress tracking
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in as_completed(future_to_batch):
                    try:
                        batch_results = future.result()
                        all_processed.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {e}")
                        pbar.update(1)
        
        self.logger.info(f"Processed {len(all_processed)} valid items")
        
        # Apply final filtering
        filtered_dataset = self.filter_dataset(all_processed)
        
        self.logger.info(f"Dataset processing complete: {len(filtered_dataset)} final items")
        return filtered_dataset
    
    def get_processing_stats(self, dataset: List[AudioDataset]) -> Dict[str, Any]:
        """
        Generate processing statistics for the dataset.
        
        Args:
            dataset: List of processed AudioDataset objects
            
        Returns:
            Dictionary with processing statistics
        """
        if not dataset:
            return {}
        
        durations = [item.duration for item in dataset]
        quality_scores = [item.quality_score for item in dataset]
        sample_rates = [item.sample_rate for item in dataset]
        
        stats = {
            'total_items': len(dataset),
            'total_duration_hours': sum(durations) / 3600,
            'avg_duration_seconds': np.mean(durations),
            'min_duration_seconds': np.min(durations),
            'max_duration_seconds': np.max(durations),
            'avg_quality_score': np.mean(quality_scores),
            'min_quality_score': np.min(quality_scores),
            'max_quality_score': np.max(quality_scores),
            'sample_rates': list(set(sample_rates)),
            'avg_transcript_length': np.mean([len(item.text_transcript) for item in dataset])
        }
        
        return stats