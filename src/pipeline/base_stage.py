"""
Base pipeline stage for DVC-based TTS fine-tuning pipeline.

This module provides the abstract base class for all pipeline stages.
Each stage implements standardized interfaces for error handling, logging,
progress tracking, and data validation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import json
import time
import traceback
from datetime import datetime
import yaml
import os


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class BasePipelineStage(ABC):
    """
    Abstract base class for all DVC pipeline stages.
    
    Provides common functionality for:
    - Parameter loading and validation
    - Logging setup and management
    - Error handling and recovery
    - Progress tracking
    - Data validation
    - Metrics and artifacts management
    """
    
    def __init__(self, stage_name: str, params_file: str = "params.yaml"):
        """
        Initialize pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            params_file: Path to parameters file
        """
        self.stage_name = stage_name
        self.params_file = params_file
        self.start_time = None
        self.end_time = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Load parameters
        self.params = self._load_parameters()
        
        # Stage-specific parameters
        self.stage_params = self.params.get(stage_name, {})
        
        # Create output directories
        self._create_output_directories()
        
        self.logger.info(f"Initialized {stage_name} pipeline stage")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration for the stage."""
        logger = logging.getLogger(f"pipeline.{self.stage_name}")
        
        if not logger.handlers:
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # File handler
            log_file = logs_dir / f"{self.stage_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)
        
        return logger
    
    def _load_parameters(self) -> Dict[str, Any]:
        """Load parameters from YAML file."""
        try:
            with open(self.params_file, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
            self.logger.info(f"Loaded parameters from {self.params_file}")
            return params
        except Exception as e:
            raise PipelineError(f"Failed to load parameters from {self.params_file}: {e}")
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            "data/processed",
            "data/preprocessed", 
            "models/checkpoints",
            "models/training_logs",
            "models/lora_adapters",
            "models/vllm_compatible",
            "models/deployment_ready",
            "results/evaluation",
            "metrics",
            "plots",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def validate_inputs(self) -> bool:
        """
        Validate stage inputs before execution.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the main logic of the pipeline stage.
        
        Returns:
            Dictionary with execution results and metadata
        """
        pass
    
    @abstractmethod
    def validate_outputs(self) -> bool:
        """
        Validate stage outputs after execution.
        
        Returns:
            True if outputs are valid, False otherwise
        """
        pass
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline stage with error handling.
        
        Returns:
            Dictionary with execution results and metadata
        """
        self.start_time = time.time()
        stage_metadata = {
            "stage_name": self.stage_name,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "error": None
        }
        
        try:
            self.logger.info(f"Starting {self.stage_name} stage")
            
            # Validate inputs
            if not self.validate_inputs():
                raise PipelineError("Input validation failed")
            
            self.logger.info("Input validation passed")
            
            # Execute main logic
            results = self.execute()
            
            # Validate outputs
            if not self.validate_outputs():
                raise PipelineError("Output validation failed")
            
            self.logger.info("Output validation passed")
            
            # Update metadata
            self.end_time = time.time()
            stage_metadata.update({
                "status": "completed",
                "end_time": datetime.now().isoformat(),
                "duration_seconds": self.end_time - self.start_time,
                "results": results
            })
            
            self.logger.info(f"Completed {self.stage_name} stage in {stage_metadata['duration_seconds']:.2f} seconds")
            
            # Save stage metadata
            self._save_stage_metadata(stage_metadata)
            
            return stage_metadata
            
        except Exception as e:
            self.end_time = time.time()
            error_msg = f"Stage {self.stage_name} failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            stage_metadata.update({
                "status": "failed",
                "end_time": datetime.now().isoformat(),
                "duration_seconds": self.end_time - self.start_time if self.start_time else 0,
                "error": {
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            })
            
            # Save error metadata
            self._save_stage_metadata(stage_metadata)
            
            raise PipelineError(error_msg) from e
    
    def _save_stage_metadata(self, metadata: Dict[str, Any]):
        """Save stage execution metadata."""
        metadata_dir = Path("logs") / "stage_metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        metadata_file = metadata_dir / f"{self.stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.debug(f"Saved stage metadata to {metadata_file}")
    
    def save_metrics(self, metrics: Dict[str, Any], metrics_file: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            metrics_file: Path to metrics file
        """
        metrics_path = Path(metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Saved metrics to {metrics_file}")
    
    def save_plots_data(self, plots_data: Dict[str, Any], plots_file: str):
        """
        Save plots data to JSON file for DVC plots.
        
        Args:
            plots_data: Dictionary of plot data
            plots_file: Path to plots file
        """
        plots_path = Path(plots_file)
        plots_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(plots_path, 'w', encoding='utf-8') as f:
            json.dump(plots_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved plots data to {plots_file}")
    
    def load_previous_stage_output(self, file_path: str) -> Any:
        """
        Load output from previous pipeline stage.
        
        Args:
            file_path: Path to the output file
            
        Returns:
            Loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise PipelineError(f"Previous stage output not found: {file_path}")
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif file_path.suffix == '.pkl':
            import pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise PipelineError(f"Unsupported file format: {file_path.suffix}")
    
    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            required_gb: Required disk space in GB
            
        Returns:
            True if sufficient space available
        """
        try:
            import shutil
            free_bytes = shutil.disk_usage(".").free
            free_gb = free_bytes / (1024**3)
            
            self.logger.info(f"Available disk space: {free_gb:.2f} GB")
            
            if free_gb < required_gb:
                self.logger.warning(f"Insufficient disk space. Required: {required_gb} GB, Available: {free_gb:.2f} GB")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {e}")
            return True  # Assume sufficient space if check fails
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        memory_info = {}
        
        try:
            import psutil
            process = psutil.Process()
            memory_info['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
            memory_info['system_memory_percent'] = psutil.virtual_memory().percent
            memory_info['system_memory_available_gb'] = psutil.virtual_memory().available / 1024**3
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
        
        try:
            import torch
            if torch.cuda.is_available():
                memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass
        
        return memory_info


def main():
    """Main entry point for running individual stages."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline.base_stage <stage_name>")
        sys.exit(1)
    
    stage_name = sys.argv[1]
    
    # Import and run the specified stage
    if stage_name == "data_loading":
        from .data_loading_stage import DataLoadingStage
        stage = DataLoadingStage()
    elif stage_name == "preprocessing":
        from .preprocessing_stage import PreprocessingStage  
        stage = PreprocessingStage()
    elif stage_name == "training":
        from .training_stage import TrainingStage
        stage = TrainingStage()
    elif stage_name == "evaluation":
        from .evaluation_stage import EvaluationStage
        stage = EvaluationStage()
    elif stage_name == "persistence":
        from .persistence_stage import PersistenceStage
        stage = PersistenceStage()
    else:
        print(f"Unknown stage: {stage_name}")
        sys.exit(1)
    
    try:
        result = stage.run()
        print(f"Stage {stage_name} completed successfully")
        print(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
        sys.exit(0)
    except Exception as e:
        print(f"Stage {stage_name} failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
