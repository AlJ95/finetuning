"""
MLSGermanDataProcessor for Multilingual LibriSpeech German dataset.

This processor handles the MLS format with multi-speaker data, OPUS audio files,
and speaker ID extraction for German TTS fine-tuning.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, Counter
import numpy as np

from data_processor_base import DataProcessor, ProcessingConfig, AudioDataset


class MLSGermanDataProcessor(DataProcessor):
    """
    Data processor for Multilingual LibriSpeech German dataset.
    
    Handles:
    - MLS directory structure and metadata parsing
    - Multi-speaker data processing with speaker ID extraction
    - OPUS to WAV conversion for audio processing
    - Speaker-specific quality validation and filtering
    - Balanced speaker sampling for training
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None, 
                 target_speakers: Optional[List[str]] = None,
                 max_samples_per_speaker: Optional[int] = None):
        """
        Initialize MLS German data processor.
        
        Args:
            config: Processing configuration
            target_speakers: List of specific speaker IDs to process (None = all)
            max_samples_per_speaker: Maximum samples per speaker for balanced training
        """
        super().__init__(config)
        self.target_speakers = target_speakers
        self.max_samples_per_speaker = max_samples_per_speaker
        self.speaker_stats = defaultdict(int)
        self.speaker_quality_stats = defaultdict(list)
        
        # MLS-specific configuration
        self.expected_sample_rate = 16000  # MLS uses 16kHz
        self.audio_format = 'opus'
        
    def load_dataset_metadata(self, dataset_path: Path) -> List[Dict[str, Any]]:
        """
        Load MLS dataset metadata from directory structure and transcript files.
        
        MLS structure:
        - mls_german_opus/
          - train/
            - audio/
              - {speaker_id}/
                - {book_id}/
                  - {speaker_id}_{book_id}_{segment_id}.opus
            - transcripts.txt
          - dev/
          - test/
        
        Args:
            dataset_path: Path to MLS dataset root directory
            
        Returns:
            List of metadata dictionaries for each audio file
        """
        self.logger.info(f"Loading MLS German dataset from: {dataset_path}")
        
        metadata_list = []
        
        # Process each split (train, dev, test)
        for split in ['train', 'dev', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                self.logger.warning(f"Split directory not found: {split_path}")
                continue
                
            self.logger.info(f"Processing {split} split...")
            
            # Load transcripts for this split
            transcripts = self._load_transcripts(split_path / 'transcripts.txt')
            self.logger.info(f"Loaded {len(transcripts)} transcripts for {split}")
            
            # Find audio files
            audio_dir = split_path / 'audio'
            if not audio_dir.exists():
                self.logger.warning(f"Audio directory not found: {audio_dir}")
                continue
            
            # Process each speaker directory
            for speaker_dir in audio_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                    
                speaker_id = speaker_dir.name
                
                # Skip if we have target speakers and this isn't one of them
                if self.target_speakers and speaker_id not in self.target_speakers:
                    continue
                
                # Process each book directory for this speaker
                for book_dir in speaker_dir.iterdir():
                    if not book_dir.is_dir():
                        continue
                        
                    book_id = book_dir.name
                    
                    # Process audio files in this book
                    for audio_file in book_dir.glob(f"*.{self.audio_format}"):
                        # Parse filename: {speaker_id}_{book_id}_{segment_id}.opus
                        file_id = audio_file.stem
                        
                        # Get transcript for this file
                        transcript = transcripts.get(file_id, "")
                        if not transcript:
                            self.logger.debug(f"No transcript found for {file_id}")
                            continue
                        
                        # Check speaker sample limit
                        if (self.max_samples_per_speaker and 
                            self.speaker_stats[speaker_id] >= self.max_samples_per_speaker):
                            continue
                        
                        metadata = {
                            'file_path': str(audio_file),
                            'file_id': file_id,
                            'speaker_id': speaker_id,
                            'book_id': book_id,
                            'transcript': transcript,
                            'split': split,
                            'audio_format': self.audio_format,
                            'expected_sample_rate': self.expected_sample_rate
                        }
                        
                        metadata_list.append(metadata)
                        self.speaker_stats[speaker_id] += 1
        
        self.logger.info(f"Loaded {len(metadata_list)} audio files from {len(self.speaker_stats)} speakers")
        self._log_speaker_distribution()
        
        return metadata_list
    
    def _load_transcripts(self, transcript_file: Path) -> Dict[str, str]:
        """
        Load transcripts from MLS transcript file.
        
        Format: {file_id}\t{transcript}
        
        Args:
            transcript_file: Path to transcripts.txt file
            
        Returns:
            Dictionary mapping file_id to transcript text
        """
        transcripts = {}
        
        if not transcript_file.exists():
            self.logger.warning(f"Transcript file not found: {transcript_file}")
            return transcripts
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split on tab
                    parts = line.split('\t', 1)
                    if len(parts) != 2:
                        self.logger.warning(f"Invalid transcript format at line {line_num}: {line}")
                        continue
                    
                    file_id, transcript = parts
                    transcripts[file_id] = transcript.strip()
            
            self.logger.info(f"Loaded {len(transcripts)} transcripts from {transcript_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading transcripts from {transcript_file}: {e}")
        
        return transcripts
    
    def _log_speaker_distribution(self):
        """Log speaker distribution statistics."""
        if not self.speaker_stats:
            return
        
        total_samples = sum(self.speaker_stats.values())
        speaker_counts = Counter(self.speaker_stats.values())
        
        self.logger.info(f"Speaker distribution:")
        self.logger.info(f"  Total speakers: {len(self.speaker_stats)}")
        self.logger.info(f"  Total samples: {total_samples}")
        self.logger.info(f"  Avg samples per speaker: {total_samples / len(self.speaker_stats):.1f}")
        self.logger.info(f"  Min samples: {min(self.speaker_stats.values())}")
        self.logger.info(f"  Max samples: {max(self.speaker_stats.values())}")
        
        # Log top speakers
        top_speakers = sorted(self.speaker_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        self.logger.info(f"  Top 10 speakers: {top_speakers}")
    
    def parse_transcript(self, metadata: Dict[str, Any]) -> str:
        """
        Parse and clean transcript text from MLS metadata.
        
        Args:
            metadata: MLS-specific metadata dictionary
            
        Returns:
            Cleaned transcript text
        """
        transcript = metadata.get('transcript', '')
        
        if not transcript:
            return ""
        
        # MLS-specific text cleaning
        transcript = transcript.strip()
        
        # Remove extra whitespace
        transcript = re.sub(r'\s+', ' ', transcript)
        
        # Handle German-specific characters (should already be correct in MLS)
        # MLS transcripts are already normalized
        
        # Convert to lowercase for consistency (optional - depends on model requirements)
        # transcript = transcript.lower()
        
        return transcript
    
    def extract_speaker_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract speaker-specific metadata from MLS data.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enhanced metadata with speaker information
        """
        speaker_id = metadata.get('speaker_id', 'unknown')
        
        # Add speaker-specific information
        speaker_metadata = {
            'speaker_id': speaker_id,
            'is_multi_speaker': True,
            'speaker_sample_count': self.speaker_stats.get(speaker_id, 0),
            'book_id': metadata.get('book_id', 'unknown'),
            'split': metadata.get('split', 'unknown')
        }
        
        return speaker_metadata
    
    def validate_audio_quality(self, audio_file: Path) -> Tuple[bool, Any]:
        """
        Validate audio quality with MLS-specific considerations.
        
        Args:
            audio_file: Path to OPUS audio file
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        try:
            import librosa
            import soundfile as sf
            
            # Convert OPUS to temporary WAV if needed for processing
            audio_data, sr = librosa.load(str(audio_file), sr=None)
            
            # Validate sample rate
            if sr != self.expected_sample_rate:
                self.logger.debug(f"Unexpected sample rate {sr} for {audio_file} (expected {self.expected_sample_rate})")
            
            # Calculate quality metrics
            quality_metrics = self.calculate_quality_metrics(audio_data, sr)
            
            # MLS-specific validation criteria
            is_valid = (
                # Duration checks
                self.config.min_duration <= quality_metrics.duration_seconds <= self.config.max_duration
                # SNR check
                and quality_metrics.signal_to_noise_ratio >= self.config.min_snr
                # Energy check
                and quality_metrics.rms_energy > 0.001
                # Spectral check
                and not np.isnan(quality_metrics.spectral_centroid)
                # MLS-specific: not too quiet (common issue with some MLS files)
                and quality_metrics.rms_energy > 0.01
            )
            
            # Track speaker-specific quality
            speaker_id = audio_file.parent.parent.name
            self.speaker_quality_stats[speaker_id].append(quality_metrics.signal_to_noise_ratio)
            
            return is_valid, quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating MLS audio quality for {audio_file}: {e}")
            return False, None
    
    def get_speaker_balance_stats(self, dataset: List[AudioDataset]) -> Dict[str, Any]:
        """
        Get speaker balance statistics for the processed dataset.
        
        Args:
            dataset: List of processed AudioDataset objects
            
        Returns:
            Dictionary with speaker balance statistics
        """
        speaker_counts = defaultdict(int)
        speaker_durations = defaultdict(float)
        speaker_quality = defaultdict(list)
        
        for item in dataset:
            speaker_id = item.metadata.get('original_metadata', {}).get('speaker_id', 'unknown')
            speaker_counts[speaker_id] += 1
            speaker_durations[speaker_id] += item.duration
            speaker_quality[speaker_id].append(item.quality_score)
        
        # Calculate statistics
        total_speakers = len(speaker_counts)
        total_samples = sum(speaker_counts.values())
        total_duration = sum(speaker_durations.values())
        
        stats = {
            'total_speakers': total_speakers,
            'total_samples': total_samples,
            'total_duration_hours': total_duration / 3600,
            'avg_samples_per_speaker': total_samples / total_speakers if total_speakers > 0 else 0,
            'avg_duration_per_speaker': total_duration / total_speakers if total_speakers > 0 else 0,
            'speaker_distribution': dict(speaker_counts),
            'speaker_durations': {k: v/3600 for k, v in speaker_durations.items()},  # Convert to hours
            'speaker_avg_quality': {k: sum(v)/len(v) for k, v in speaker_quality.items() if v},
            'balance_coefficient': self._calculate_balance_coefficient(list(speaker_counts.values()))
        }
        
        return stats
    
    def _calculate_balance_coefficient(self, counts: List[int]) -> float:
        """
        Calculate balance coefficient (0 = perfectly balanced, 1 = completely imbalanced).
        
        Args:
            counts: List of sample counts per speaker
            
        Returns:
            Balance coefficient
        """
        if not counts or len(counts) <= 1:
            return 0.0
        
        import numpy as np
        
        # Use coefficient of variation as balance measure
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 1.0
        
        cv = std_count / mean_count
        # Normalize to 0-1 range (CV of 1.0 = completely imbalanced)
        return min(cv, 1.0)
    
    def filter_by_speaker_balance(self, dataset: List[AudioDataset], 
                                 max_samples_per_speaker: int) -> List[AudioDataset]:
        """
        Filter dataset to balance speakers by limiting samples per speaker.
        
        Args:
            dataset: List of AudioDataset objects
            max_samples_per_speaker: Maximum samples to keep per speaker
            
        Returns:
            Balanced dataset
        """
        speaker_samples = defaultdict(list)
        
        # Group by speaker
        for item in dataset:
            speaker_id = item.metadata.get('original_metadata', {}).get('speaker_id', 'unknown')
            speaker_samples[speaker_id].append(item)
        
        # Balance speakers
        balanced_dataset = []
        for speaker_id, samples in speaker_samples.items():
            # Sort by quality score (keep best samples)
            samples.sort(key=lambda x: x.quality_score, reverse=True)
            
            # Take up to max_samples_per_speaker
            selected_samples = samples[:max_samples_per_speaker]
            balanced_dataset.extend(selected_samples)
            
            self.logger.info(f"Speaker {speaker_id}: kept {len(selected_samples)}/{len(samples)} samples")
        
        self.logger.info(f"Balanced dataset: {len(balanced_dataset)} samples from {len(speaker_samples)} speakers")
        return balanced_dataset
    
    def get_processing_stats(self, dataset: List[AudioDataset]) -> Dict[str, Any]:
        """
        Generate MLS-specific processing statistics.
        
        Args:
            dataset: List of processed AudioDataset objects
            
        Returns:
            Dictionary with comprehensive processing statistics
        """
        # Get base stats
        base_stats = super().get_processing_stats(dataset)
        
        # Add MLS-specific stats
        mls_stats = self.get_speaker_balance_stats(dataset)
        
        # Combine statistics
        combined_stats = {
            **base_stats,
            'mls_specific': mls_stats,
            'dataset_type': 'MLS German',
            'multi_speaker': True,
            'audio_format': self.audio_format,
            'expected_sample_rate': self.expected_sample_rate
        }
        
        return combined_stats


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Configure processing for MLS
    config = ProcessingConfig(
        min_duration=2.0,      # MLS has longer utterances
        max_duration=20.0,     # Allow longer clips
        min_snr=8.0,          # Slightly lower SNR threshold for MLS
        quality_threshold=0.5, # Lower threshold due to multi-speaker variability
        max_workers=4,
        batch_size=100
    )
    
    # Create processor with speaker balancing
    processor = MLSGermanDataProcessor(
        config=config,
        target_speakers=None,  # Process all speakers
        max_samples_per_speaker=1000  # Limit for balanced training
    )
    
    print("MLS German DataProcessor created successfully!")
    print(f"Configuration: {config}")
    print(f"Expected sample rate: {processor.expected_sample_rate}Hz")
    print(f"Audio format: {processor.audio_format}")
    
    # Example of how to use:
    # dataset_path = Path("D:/Trainingsdaten/TTS/mls_german_opus")
    # processed_dataset = processor.process_dataset(dataset_path)
    # stats = processor.get_processing_stats(processed_dataset)
    # balanced_dataset = processor.filter_by_speaker_balance(processed_dataset, 500)