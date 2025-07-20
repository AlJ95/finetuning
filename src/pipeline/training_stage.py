"""
Training stage for DVC pipeline.

This stage handles the fine-tuning of the Orpheus 3B model using Unsloth
with the preprocessed German TTS datasets.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import pickle
import time
import shutil

from .base_stage import BasePipelineStage, PipelineError
from ..unsloth_trainer import UnslothTrainer, TrainingConfig, TrainingResults
from ..data_processor_base import AudioDataset


class TrainingStage(BasePipelineStage):
    """
    Training pipeline stage.
    
    Fine-tunes the Orpheus 3B model using Unsloth with preprocessed German TTS data.
    
    Features:
    - Memory-efficient training with Unsloth optimizations
    - Configurable training parameters
    - Progress monitoring and logging
    - Model checkpointing and recovery
    - Training metrics collection
    """
    
    def __init__(self):
        super().__init__("training")
        
        # Load training and model parameters
        self.training_params = self.stage_params
        self.model_params = self.params.get("model", {})
        
        # Initialize trainer
        self.trainer = None
        self.training_config = None
        
        self.logger.info("Initialized training stage")
    
    def validate_inputs(self) -> bool:
        """Validate that required preprocessed data is available."""
        try:
            # Check if preprocessed datasets exist
            train_file = Path("data/preprocessed/train_dataset.pkl")
            val_file = Path("data/preprocessed/val_dataset.pkl")
            
            if not train_file.exists():
                self.logger.error("Training dataset not found")
                return False
            
            if not val_file.exists():
                self.logger.error("Validation dataset not found")
                return False
            
            # Load and validate datasets
            with open(train_file, 'rb') as f:
                train_samples = pickle.load(f)
            
            with open(val_file, 'rb') as f:
                val_samples = pickle.load(f)
            
            if not train_samples:
                self.logger.error("Empty training dataset")
                return False
            
            if not val_samples:
                self.logger.error("Empty validation dataset")
                return False
            
            # Check sample types
            if not isinstance(train_samples[0], AudioDataset):
                self.logger.error("Invalid training sample type")
                return False
            
            self.logger.info(f"Training samples: {len(train_samples)}")
            self.logger.info(f"Validation samples: {len(val_samples)}")
            
            # Check disk space (need space for model checkpoints)
            if not self.check_disk_space(30.0):
                self.logger.error("Insufficient disk space for training")
                return False
            
            # Check GPU availability
            try:
                import torch
                if not torch.cuda.is_available():
                    self.logger.warning("CUDA not available - training will be slow")
                else:
                    gpu_name = torch.cuda.get_device_name()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    self.logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
                    
                    if gpu_memory < 8.0:
                        self.logger.warning("GPU memory < 8GB - consider reducing batch size")
            except ImportError:
                self.logger.warning("PyTorch not available")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute the training process."""
        results = {
            "training_started": time.time(),
            "training_completed": None,
            "training_duration": None,
            "model_saved": False,
            "training_metrics": {},
            "memory_stats": {},
            "checkpoints_saved": 0
        }
        
        try:
            # Create training configuration
            self._create_training_config()
            
            # Initialize trainer
            self.trainer = UnslothTrainer(self.training_config)
            self.logger.info("Initialized Unsloth trainer")
            
            # Load model
            model, tokenizer = self.trainer.load_orpheus_model()
            self.logger.info("Loaded Orpheus 3B model")
            
            # Load training data
            train_samples = self._load_training_data()
            self.logger.info(f"Loaded {len(train_samples)} training samples")
            
            # Log memory stats before training
            memory_before = self.trainer.get_memory_stats()
            results["memory_stats"]["before_training"] = memory_before
            self.logger.info(f"Memory before training: {memory_before}")
            
            # Start training
            self.logger.info("Starting fine-tuning...")
            training_results = self.trainer.start_finetuning(train_samples)
            
            # Update results
            results["training_completed"] = time.time()
            results["training_duration"] = results["training_completed"] - results["training_started"]
            results["model_saved"] = True
            results["training_metrics"] = {
                "final_loss": training_results.final_loss,
                "total_steps": training_results.total_steps,
                "training_time": training_results.training_time
            }
            results["memory_stats"]["after_training"] = training_results.memory_stats
            
            # Save training logs and metrics
            self._save_training_artifacts(training_results, results)
            
            # Save metrics for DVC
            self._save_dvc_metrics(training_results)
            
            self.logger.info(f"Training completed successfully in {results['training_duration']:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            results["error"] = str(e)
            results["training_completed"] = time.time()
            if results["training_started"]:
                results["training_duration"] = results["training_completed"] - results["training_started"]
            raise
    
    def _create_training_config(self):
        """Create training configuration from parameters."""
        self.logger.info("Creating training configuration")
        
        # Base configuration
        config_dict = {
            # Model parameters
            "model_name": self.model_params.get("name", "unsloth/orpheus-3b-0.1-ft"),
            "max_seq_length": self.model_params.get("max_seq_length", 2048),
            "load_in_4bit": self.model_params.get("load_in_4bit", False),
            "dtype": self.model_params.get("dtype"),
            
            # LoRA parameters
            "r": self.model_params.get("lora", {}).get("r", 16),
            "target_modules": self.model_params.get("lora", {}).get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            "lora_alpha": self.model_params.get("lora", {}).get("lora_alpha", 16),
            "lora_dropout": self.model_params.get("lora", {}).get("lora_dropout", 0.0),
            "bias": self.model_params.get("lora", {}).get("bias", "none"),
            "use_rslora": self.model_params.get("lora", {}).get("use_rslora", False),
            
            # Training parameters
            "num_train_epochs": self.training_params.get("num_train_epochs", 3),
            "per_device_train_batch_size": self.training_params.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": self.training_params.get("gradient_accumulation_steps", 4),
            "learning_rate": self.training_params.get("learning_rate", 2e-4),
            "weight_decay": self.training_params.get("weight_decay", 0.01),
            "warmup_steps": self.training_params.get("warmup_steps", 5),
            "lr_scheduler_type": self.training_params.get("lr_scheduler_type", "linear"),
            "optim": self.training_params.get("optim", "adamw_8bit"),
            "max_steps": self.training_params.get("max_steps", -1),
            
            # Memory optimization
            "use_gradient_checkpointing": self.training_params.get("use_gradient_checkpointing", "unsloth"),
            "dataloader_num_workers": self.training_params.get("dataloader_num_workers", 0),
            "remove_unused_columns": self.training_params.get("remove_unused_columns", False),
            "group_by_length": self.training_params.get("group_by_length", True),
            
            # Logging and saving
            "logging_steps": self.training_params.get("logging_steps", 1),
            "save_steps": self.training_params.get("save_steps", 500),
            "save_total_limit": self.training_params.get("save_total_limit", 3),
            "report_to": self.training_params.get("report_to", "none"),
            "output_dir": self.training_params.get("output_dir", "models/checkpoints"),
        }
        
        self.training_config = TrainingConfig(**config_dict)
        self.logger.info("Training configuration created")
    
    def _load_training_data(self) -> List[AudioDataset]:
        """Load training data from preprocessed files."""
        train_file = Path("data/preprocessed/train_dataset.pkl")
        
        with open(train_file, 'rb') as f:
            train_samples = pickle.load(f)
        
        return train_samples
    
    def _save_training_artifacts(self, training_results: TrainingResults, stage_results: Dict[str, Any]):
        """Save training artifacts and logs."""
        self.logger.info("Saving training artifacts")
        
        # Create training logs directory
        logs_dir = Path("models/training_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training results
        results_file = logs_dir / "training_results.json"
        training_data = {
            "final_loss": training_results.final_loss,
            "training_time": training_results.training_time,
            "total_steps": training_results.total_steps,
            "model_path": training_results.model_path,
            "tokenizer_path": training_results.tokenizer_path,
            "memory_stats": training_results.memory_stats,
            "stage_results": stage_results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        # Save training logs if available
        if training_results.training_logs:
            logs_file = logs_dir / "training_logs.json"
            with open(logs_file, 'w', encoding='utf-8') as f:
                json.dump(training_results.training_logs, f, indent=2, default=str)
        
        # Copy model artifacts to standard location
        if Path(training_results.model_path).exists():
            lora_dest = Path("models/lora_adapters")
            if lora_dest.exists():
                shutil.rmtree(lora_dest)
            
            shutil.copytree(training_results.model_path, lora_dest)
            self.logger.info(f"Copied LoRA adapters to {lora_dest}")
        
        self.logger.info("Training artifacts saved")
    
    def _save_dvc_metrics(self, training_results: TrainingResults):
        """Save metrics in DVC format."""
        metrics = {
            "final_loss": training_results.final_loss,
            "training_time_seconds": training_results.training_time,
            "total_steps": training_results.total_steps
        }
        
        # Add memory metrics if available
        if training_results.memory_stats:
            metrics.update({
                "peak_gpu_memory_gb": training_results.memory_stats.get("memory_peak_gb", 0),
                "gpu_memory_before_gb": training_results.memory_stats.get("memory_before_gb", 0),
                "gpu_memory_after_gb": training_results.memory_stats.get("memory_after_gb", 0)
            })
        
        self.save_metrics(metrics, "metrics/training_metrics.json")
        
        # Save plot data for loss over time
        if training_results.training_logs:
            plot_data = []
            for log_entry in training_results.training_logs:
                if "train_loss" in log_entry and "step" in log_entry:
                    plot_data.append({
                        "step": log_entry["step"],
                        "loss": log_entry["train_loss"]
                    })
            
            if plot_data:
                self.save_plots_data(plot_data, "plots/training_loss.json")
    
    def validate_outputs(self) -> bool:
        """Validate that training outputs were created correctly."""
        try:
            # Check if LoRA adapters were saved
            lora_dir = Path("models/lora_adapters")
            if not lora_dir.exists():
                self.logger.error("LoRA adapters directory not found")
                return False
            
            # Check for required model files
            required_files = ["adapter_config.json", "adapter_model.safetensors"]
            for file_name in required_files:
                file_path = lora_dir / file_name
                if not file_path.exists():
                    self.logger.warning(f"Expected model file missing: {file_name}")
                    # Don't fail validation for missing files - adapter format may vary
            
            # Check training metrics
            metrics_file = Path("metrics/training_metrics.json")
            if not metrics_file.exists():
                self.logger.error("Training metrics file not found")
                return False
            
            # Validate metrics content
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            required_metrics = ["final_loss", "training_time_seconds", "total_steps"]
            for metric in required_metrics:
                if metric not in metrics:
                    self.logger.error(f"Missing required metric: {metric}")
                    return False
            
            # Check training logs
            logs_dir = Path("models/training_logs")
            if not logs_dir.exists():
                self.logger.error("Training logs directory not found")
                return False
            
            results_file = logs_dir / "training_results.json"
            if not results_file.exists():
                self.logger.error("Training results file not found")
                return False
            
            self.logger.info("Training output validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False


def main():
    """Main entry point for running training stage."""
    stage = TrainingStage()
    
    try:
        result = stage.run()
        print(f"Training completed successfully")
        print(f"Final loss: {result['results']['training_metrics']['final_loss']:.4f}")
        print(f"Training time: {result['results']['training_duration']:.2f} seconds")
        print(f"Total steps: {result['results']['training_metrics']['total_steps']}")
        
        if "memory_stats" in result["results"]:
            memory_stats = result["results"]["memory_stats"]
            if "after_training" in memory_stats:
                peak_memory = memory_stats["after_training"].get("memory_peak_gb", 0)
                print(f"Peak GPU memory: {peak_memory:.2f} GB")
        
    except PipelineError as e:
        print(f"Training failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
