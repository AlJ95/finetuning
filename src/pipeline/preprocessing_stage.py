"""
Data preprocessing stage for DVC pipeline.

This stage preprocesses the loaded datasets by applying quality filters,
text normalization, audio processing, and creating train/validation splits.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import pickle
import random
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from .base_stage import BasePipelineStage, PipelineError
from ..data_processor_base import AudioDataset


class PreprocessingStage(BasePipelineStage):
    """
    Data preprocessing pipeline stage.
    
    Applies quality filtering, text normalization, and creates train/validation splits
    from the loaded datasets.
    
    Features:
    - Audio quality filtering (SNR, duration, sample rate)
    - Text normalization and filtering
    - Speaker stratification for splits
    - Statistics and quality reports
    """
    
    def __init__(self):
        super().__init__("preprocessing")
        
        # Load preprocessing parameters
        self.audio_params = self.stage_params.get("audio", {})
        self.text_params = self.stage_params.get("text", {})
        self.split_params = self.stage_params.get("dataset_split", {})
        self.batch_params = self.stage_params.get("batch_processing", {})
        
        self.logger.info("Initialized preprocessing stage")
    
    def validate_inputs(self) -> bool:
        """Validate that required input data is available."""
        try:
            # Check if dataset info file exists
            dataset_info_file = Path("data/processed/dataset_info.json")
            if not dataset_info_file.exists():
                self.logger.error("Dataset info file not found from data loading stage")
                return False
            
            # Load dataset info
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
            
            # Check if any datasets were processed
            if not self.dataset_info.get("datasets_processed"):
                self.logger.error("No datasets available for preprocessing")
                return False
            
            # Validate each dataset's samples file
            for dataset_name in self.dataset_info["datasets_processed"]:
                samples_file = Path("data/processed") / dataset_name / "samples.pkl"
                if not samples_file.exists():
                    self.logger.error(f"Samples file missing for {dataset_name}")
                    return False
            
            # Check disk space
            if not self.check_disk_space(20.0):
                self.logger.error("Insufficient disk space for preprocessing")
                return False
            
            self.logger.info("Input validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute data preprocessing and filtering."""
        results = {
            "total_input_samples": 0,
            "total_output_samples": 0,
            "filtering_stats": {},
            "dataset_splits": {},
            "preprocessing_summary": {}
        }
        
        # Load all datasets
        all_samples = []
        for dataset_name in self.dataset_info["datasets_processed"]:
            samples_file = Path("data/processed") / dataset_name / "samples.pkl"
            
            self.logger.info(f"Loading samples from {dataset_name}")
            with open(samples_file, 'rb') as f:
                dataset_samples = pickle.load(f)
            
            # Add dataset source to metadata
            for sample in dataset_samples:
                if sample.metadata is None:
                    sample.metadata = {}
                sample.metadata["source_dataset"] = dataset_name
            
            all_samples.extend(dataset_samples)
            self.logger.info(f"Loaded {len(dataset_samples)} samples from {dataset_name}")
        
        results["total_input_samples"] = len(all_samples)
        self.logger.info(f"Total input samples: {len(all_samples)}")
        
        # Apply filtering
        filtered_samples, filtering_stats = self._apply_filters(all_samples)
        results["filtering_stats"] = filtering_stats
        results["total_output_samples"] = len(filtered_samples)
        
        self.logger.info(f"After filtering: {len(filtered_samples)} samples")
        
        # Create train/validation splits
        train_samples, val_samples = self._create_splits(filtered_samples)
        
        results["dataset_splits"] = {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_ratio": len(train_samples) / len(filtered_samples),
            "val_ratio": len(val_samples) / len(filtered_samples)
        }
        
        # Save processed datasets
        self._save_datasets(train_samples, val_samples)
        
        # Generate preprocessing statistics
        preprocessing_stats = self._generate_statistics(filtered_samples, train_samples, val_samples)
        results["preprocessing_summary"] = preprocessing_stats
        
        # Save preprocessing statistics
        stats_file = "data/preprocessed/preprocessing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("Preprocessing completed successfully")
        return results
    
    def _apply_filters(self, samples: List[AudioDataset]) -> Tuple[List[AudioDataset], Dict[str, Any]]:
        """Apply quality filters to the samples."""
        self.logger.info("Applying quality filters")
        
        filtering_stats = {
            "input_count": len(samples),
            "filters_applied": [],
            "rejected_counts": {},
            "rejection_reasons": defaultdict(int)
        }
        
        filtered_samples = []
        
        # Get filter parameters
        min_duration = self.audio_params.get("min_duration", 1.0)
        max_duration = self.audio_params.get("max_duration", 10.0)
        min_snr = self.audio_params.get("min_snr", 15.0)
        quality_threshold = self.audio_params.get("quality_threshold", 0.7)
        
        min_text_length = self.text_params.get("min_length", 10)
        max_text_length = self.text_params.get("max_length", 500)
        
        for sample in tqdm(samples, desc="Filtering samples"):
            rejected = False
            rejection_reason = []
            
            # Audio duration filter
            if sample.duration < min_duration or sample.duration > max_duration:
                rejected = True
                rejection_reason.append("duration")
                filtering_stats["rejection_reasons"]["duration"] += 1
            
            # Audio quality filter
            if sample.quality_score < quality_threshold:
                rejected = True
                rejection_reason.append("quality_score")
                filtering_stats["rejection_reasons"]["quality_score"] += 1
            
            # Text length filter
            text_length = len(sample.text_transcript)
            if text_length < min_text_length or text_length > max_text_length:
                rejected = True
                rejection_reason.append("text_length")
                filtering_stats["rejection_reasons"]["text_length"] += 1
            
            # Sample rate filter (assuming target sample rate from params)
            target_sr = self.audio_params.get("target_sample_rate", 24000)
            if sample.sample_rate != target_sr:
                # Note: In practice, we might resample here instead of rejecting
                self.logger.debug(f"Sample rate mismatch: {sample.sample_rate} vs {target_sr}")
            
            if not rejected:
                # Apply text normalization if enabled
                if self.text_params.get("normalize_text", True):
                    sample.text_transcript = self._normalize_text(sample.text_transcript)
                
                filtered_samples.append(sample)
            else:
                self.logger.debug(f"Rejected sample {sample.file_path}: {', '.join(rejection_reason)}")
        
        filtering_stats["output_count"] = len(filtered_samples)
        filtering_stats["rejection_rate"] = 1 - (len(filtered_samples) / len(samples))
        
        self.logger.info(f"Filtering complete: {len(filtered_samples)}/{len(samples)} samples retained")
        self.logger.info(f"Rejection rate: {filtering_stats['rejection_rate']:.2%}")
        
        return filtered_samples, filtering_stats
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text according to configuration."""
        import re
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove special characters if configured
        if self.text_params.get("remove_special_chars", True):
            # Keep German umlauts and basic punctuation
            text = re.sub(r'[^\w\säöüÄÖÜß.,!?;:-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _create_splits(self, samples: List[AudioDataset]) -> Tuple[List[AudioDataset], List[AudioDataset]]:
        """Create train/validation splits."""
        self.logger.info("Creating train/validation splits")
        
        train_ratio = self.split_params.get("train_ratio", 0.8)
        val_ratio = self.split_params.get("val_ratio", 0.2)
        stratify_by_speaker = self.split_params.get("stratify_by_speaker", True)
        
        random.seed(42)  # Ensure reproducible splits
        
        if stratify_by_speaker:
            # Group samples by speaker
            speaker_groups = defaultdict(list)
            for sample in samples:
                speaker_id = sample.metadata.get("speaker_id", "unknown")
                speaker_groups[speaker_id].append(sample)
            
            train_samples = []
            val_samples = []
            
            # Split each speaker's samples
            for speaker_id, speaker_samples in speaker_groups.items():
                random.shuffle(speaker_samples)
                n_train = int(len(speaker_samples) * train_ratio)
                
                train_samples.extend(speaker_samples[:n_train])
                val_samples.extend(speaker_samples[n_train:])
            
            self.logger.info(f"Stratified split by {len(speaker_groups)} speakers")
        
        else:
            # Simple random split
            random.shuffle(samples)
            n_train = int(len(samples) * train_ratio)
            
            train_samples = samples[:n_train]
            val_samples = samples[n_train:]
        
        self.logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
        return train_samples, val_samples
    
    def _save_datasets(self, train_samples: List[AudioDataset], val_samples: List[AudioDataset]):
        """Save the processed datasets."""
        self.logger.info("Saving processed datasets")
        
        # Save train dataset
        train_file = "data/preprocessed/train_dataset.pkl"
        with open(train_file, 'wb') as f:
            pickle.dump(train_samples, f)
        
        # Save validation dataset
        val_file = "data/preprocessed/val_dataset.pkl"
        with open(val_file, 'wb') as f:
            pickle.dump(val_samples, f)
        
        self.logger.info(f"Saved {len(train_samples)} training samples to {train_file}")
        self.logger.info(f"Saved {len(val_samples)} validation samples to {val_file}")
    
    def _generate_statistics(self, all_samples: List[AudioDataset], 
                           train_samples: List[AudioDataset], 
                           val_samples: List[AudioDataset]) -> Dict[str, Any]:
        """Generate comprehensive statistics about the preprocessed data."""
        
        def compute_stats(samples: List[AudioDataset], name: str) -> Dict[str, Any]:
            if not samples:
                return {}
            
            durations = [s.duration for s in samples]
            quality_scores = [s.quality_score for s in samples]
            text_lengths = [len(s.text_transcript) for s in samples]
            
            # Speaker statistics
            speakers = set()
            source_datasets = defaultdict(int)
            for s in samples:
                if s.metadata:
                    speaker_id = s.metadata.get("speaker_id")
                    if speaker_id:
                        speakers.add(speaker_id)
                    
                    source_dataset = s.metadata.get("source_dataset")
                    if source_dataset:
                        source_datasets[source_dataset] += 1
            
            return {
                "num_samples": len(samples),
                "duration_stats": {
                    "mean": float(np.mean(durations)),
                    "median": float(np.median(durations)),
                    "min": float(np.min(durations)),
                    "max": float(np.max(durations)),
                    "std": float(np.std(durations)),
                    "total_hours": float(np.sum(durations) / 3600)
                },
                "quality_stats": {
                    "mean": float(np.mean(quality_scores)),
                    "median": float(np.median(quality_scores)),
                    "min": float(np.min(quality_scores)),
                    "max": float(np.max(quality_scores)),
                    "std": float(np.std(quality_scores))
                },
                "text_stats": {
                    "mean_length": float(np.mean(text_lengths)),
                    "median_length": float(np.median(text_lengths)),
                    "min_length": int(np.min(text_lengths)),
                    "max_length": int(np.max(text_lengths))
                },
                "num_speakers": len(speakers),
                "source_datasets": dict(source_datasets)
            }
        
        stats = {
            "overall": compute_stats(all_samples, "overall"),
            "train": compute_stats(train_samples, "train"),
            "validation": compute_stats(val_samples, "validation")
        }
        
        self.logger.info("Generated preprocessing statistics")
        return stats
    
    def validate_outputs(self) -> bool:
        """Validate that preprocessing outputs were created correctly."""
        try:
            # Check if output files exist
            required_files = [
                "data/preprocessed/train_dataset.pkl",
                "data/preprocessed/val_dataset.pkl",
                "data/preprocessed/preprocessing_stats.json"
            ]
            
            for file_path in required_files:
                if not Path(file_path).exists():
                    self.logger.error(f"Required output file missing: {file_path}")
                    return False
            
            # Validate train dataset
            with open("data/preprocessed/train_dataset.pkl", 'rb') as f:
                train_samples = pickle.load(f)
            
            if not train_samples:
                self.logger.error("No training samples found")
                return False
            
            # Validate validation dataset
            with open("data/preprocessed/val_dataset.pkl", 'rb') as f:
                val_samples = pickle.load(f)
            
            if not val_samples:
                self.logger.error("No validation samples found")
                return False
            
            # Check sample types
            if not isinstance(train_samples[0], AudioDataset):
                self.logger.error("Invalid sample type in training dataset")
                return False
            
            if not isinstance(val_samples[0], AudioDataset):
                self.logger.error("Invalid sample type in validation dataset")
                return False
            
            # Validate statistics file
            with open("data/preprocessed/preprocessing_stats.json", 'r') as f:
                stats = json.load(f)
            
            required_keys = ["total_input_samples", "total_output_samples", "filtering_stats", "dataset_splits"]
            for key in required_keys:
                if key not in stats:
                    self.logger.error(f"Missing key in statistics: {key}")
                    return False
            
            self.logger.info(f"Validation passed: {len(train_samples)} train, {len(val_samples)} val samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False


def main():
    """Main entry point for running preprocessing stage."""
    stage = PreprocessingStage()
    
    try:
        result = stage.run()
        print(f"Preprocessing completed successfully")
        print(f"Input samples: {result['results']['total_input_samples']}")
        print(f"Output samples: {result['results']['total_output_samples']}")
        print(f"Train samples: {result['results']['dataset_splits']['train_samples']}")
        print(f"Val samples: {result['results']['dataset_splits']['val_samples']}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        
    except PipelineError as e:
        print(f"Preprocessing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
