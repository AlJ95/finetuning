"""
Data loading stage for DVC pipeline.

This stage loads the German TTS datasets (Thorsten Voice and MLS German)
and performs initial processing and validation.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import pickle
from tqdm import tqdm

from .base_stage import BasePipelineStage, PipelineError
from ..torsten_voice_processor import TorstenVoiceDataProcessor
from ..mls_german_processor import MLSGermanDataProcessor
from ..data_processor_base import AudioDataset


class DataLoadingStage(BasePipelineStage):
    """
    Data loading pipeline stage.
    
    Loads and initially processes German TTS datasets:
    - Thorsten Voice Dataset
    - MLS German Dataset
    
    Outputs dataset information and processed data for next stage.
    """
    
    def __init__(self):
        super().__init__("data_loading")
        
        # Initialize data processors
        self.processors = {}
        
        # Configure processors based on available datasets
        datasets = self.stage_params.get("datasets", [])
        data_paths = self.stage_params.get("data_paths", {})
        
        if "thorsten_voice" in datasets:
            thorsten_path = data_paths.get("thorsten_voice")
            if thorsten_path and Path(thorsten_path).exists():
                self.processors["thorsten_voice"] = TorstenVoiceDataProcessor(thorsten_path)
                self.logger.info(f"Initialized Thorsten Voice processor: {thorsten_path}")
            else:
                self.logger.warning(f"Thorsten Voice dataset not found at {thorsten_path}")
        
        if "mls_german" in datasets:
            mls_path = data_paths.get("mls_german")
            if mls_path and Path(mls_path).exists():
                self.processors["mls_german"] = MLSGermanDataProcessor(mls_path)
                self.logger.info(f"Initialized MLS German processor: {mls_path}")
            else:
                self.logger.warning(f"MLS German dataset not found at {mls_path}")
        
        if not self.processors:
            self.logger.warning("No dataset processors initialized")
    
    def validate_inputs(self) -> bool:
        """Validate that required datasets are available."""
        try:
            # Check if at least one dataset is available
            if not self.processors:
                self.logger.error("No dataset processors available")
                return False
            
            # Check disk space (estimate 50GB needed for processing)
            if not self.check_disk_space(50.0):
                self.logger.error("Insufficient disk space for data loading")
                return False
            
            # Validate dataset paths and basic structure
            for name, processor in self.processors.items():
                if not processor.validate_dataset_structure():
                    self.logger.error(f"Dataset structure validation failed for {name}")
                    return False
                
                self.logger.info(f"Dataset {name} structure validation passed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute data loading and initial processing."""
        results = {
            "datasets_processed": [],
            "total_samples": 0,
            "dataset_info": {},
            "processing_stats": {}
        }
        
        max_samples = self.stage_params.get("max_samples_per_dataset", -1)
        random_seed = self.stage_params.get("random_seed", 42)
        
        # Process each dataset
        for dataset_name, processor in self.processors.items():
            self.logger.info(f"Processing {dataset_name} dataset")
            
            try:
                # Load and process dataset
                dataset_samples = processor.load_dataset(
                    max_samples=max_samples,
                    random_seed=random_seed
                )
                
                self.logger.info(f"Loaded {len(dataset_samples)} samples from {dataset_name}")
                
                # Get dataset statistics
                stats = processor.get_dataset_statistics()
                
                # Save processed data
                output_dir = Path("data/processed") / dataset_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save samples as pickle for efficient loading
                samples_file = output_dir / "samples.pkl"
                with open(samples_file, 'wb') as f:
                    pickle.dump(dataset_samples, f)
                
                # Save statistics as JSON
                stats_file = output_dir / "statistics.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, default=str)
                
                # Update results
                results["datasets_processed"].append(dataset_name)
                results["total_samples"] += len(dataset_samples)
                results["dataset_info"][dataset_name] = {
                    "num_samples": len(dataset_samples),
                    "samples_file": str(samples_file),
                    "stats_file": str(stats_file),
                    "statistics": stats
                }
                results["processing_stats"][dataset_name] = {
                    "samples_loaded": len(dataset_samples),
                    "max_samples_requested": max_samples,
                    "processing_time": None  # Will be added by base class
                }
                
                self.logger.info(f"Completed processing {dataset_name}: {len(dataset_samples)} samples")
                
            except Exception as e:
                self.logger.error(f"Failed to process {dataset_name}: {e}")
                # Continue with other datasets
                continue
        
        # Save overall dataset information
        dataset_info_file = "data/processed/dataset_info.json"
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Data loading completed. Total samples: {results['total_samples']}")
        
        return results
    
    def validate_outputs(self) -> bool:
        """Validate that outputs were created correctly."""
        try:
            # Check if dataset info file exists
            dataset_info_file = Path("data/processed/dataset_info.json")
            if not dataset_info_file.exists():
                self.logger.error("Dataset info file not created")
                return False
            
            # Load and validate dataset info
            with open(dataset_info_file, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
            
            # Check if any datasets were processed
            if not dataset_info.get("datasets_processed"):
                self.logger.error("No datasets were successfully processed")
                return False
            
            # Validate each processed dataset
            for dataset_name in dataset_info["datasets_processed"]:
                dataset_dir = Path("data/processed") / dataset_name
                
                # Check if samples file exists
                samples_file = dataset_dir / "samples.pkl"
                if not samples_file.exists():
                    self.logger.error(f"Samples file missing for {dataset_name}")
                    return False
                
                # Check if statistics file exists
                stats_file = dataset_dir / "statistics.json"
                if not stats_file.exists():
                    self.logger.error(f"Statistics file missing for {dataset_name}")
                    return False
                
                # Validate sample data can be loaded
                try:
                    with open(samples_file, 'rb') as f:
                        samples = pickle.load(f)
                    
                    if not samples:
                        self.logger.error(f"No samples in {dataset_name}")
                        return False
                    
                    # Check if samples are AudioDataset objects
                    if not isinstance(samples[0], AudioDataset):
                        self.logger.error(f"Invalid sample format in {dataset_name}")
                        return False
                    
                    self.logger.info(f"Validated {len(samples)} samples for {dataset_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to validate samples for {dataset_name}: {e}")
                    return False
            
            self.logger.info("Output validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False


def main():
    """Main entry point for running data loading stage."""
    stage = DataLoadingStage()
    
    try:
        result = stage.run()
        print(f"Data loading completed successfully")
        print(f"Datasets processed: {result['results']['datasets_processed']}")
        print(f"Total samples: {result['results']['total_samples']}")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
        
    except PipelineError as e:
        print(f"Data loading failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
