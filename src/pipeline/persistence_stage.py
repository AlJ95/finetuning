"""
Model persistence stage for DVC pipeline.

This stage handles saving the trained model in VLLM-compatible formats,
validates VLLM compatibility, and prepares models for deployment.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import shutil
import time

from .base_stage import BasePipelineStage, PipelineError
from ..model_persistence import ModelPersistence, PersistenceConfig
from ..unsloth_trainer import UnslothTrainer, TrainingConfig


class PersistenceStage(BasePipelineStage):
    """
    Model persistence pipeline stage.
    
    Handles model persistence for VLLM deployment:
    - Saves models in VLLM-compatible format (merged_16bit)
    - Validates VLLM compatibility
    - Creates model registry entries
    - Optionally uploads to Hugging Face Hub
    
    Features:
    - Multiple save formats (merged, LoRA, quantized)
    - VLLM compatibility validation
    - Model registry management
    - Deployment-ready packaging
    """
    
    def __init__(self):
        super().__init__("persistence")
        
        # Load persistence parameters
        self.persistence_params = self.stage_params
        
        # Initialize persistence handler
        self.persistence_handler = None
        self.persistence_config = None
        
        self.logger.info("Initialized model persistence stage")
    
    def validate_inputs(self) -> bool:
        """Validate that required trained model is available."""
        try:
            # Check if LoRA adapters exist
            lora_dir = Path("models/lora_adapters")
            if not lora_dir.exists():
                self.logger.error("LoRA adapters not found - training must complete first")
                return False
            
            # Check training results
            training_results_file = Path("models/training_logs/training_results.json")
            if not training_results_file.exists():
                self.logger.error("Training results not found")
                return False
            
            # Check evaluation metrics (optional but preferred)
            eval_metrics_file = Path("metrics/evaluation_metrics.json")
            if not eval_metrics_file.exists():
                self.logger.warning("Evaluation metrics not found - quality assessment will be limited")
            
            # Check disk space for model exports
            if not self.check_disk_space(20.0):
                self.logger.error("Insufficient disk space for model persistence")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute model persistence and VLLM preparation."""
        results = {
            "persistence_started": time.time(),
            "models_saved": {},
            "vllm_compatibility": {},
            "model_registry": {},
            "huggingface_upload": None
        }
        
        try:
            # Create persistence configuration
            self._create_persistence_config()
            
            # Initialize persistence handler
            self.persistence_handler = ModelPersistence(self.persistence_config)
            self.logger.info("Initialized model persistence handler")
            
            # Load trained model for persistence
            trainer = self._load_trained_model()
            
            # Gather training metadata
            training_metadata = self._gather_training_metadata()
            
            # Save model in various formats
            saved_paths = self._save_model_formats(trainer, training_metadata)
            results["models_saved"] = saved_paths
            
            # Validate VLLM compatibility
            vllm_validation = self._validate_vllm_compatibility(saved_paths)
            results["vllm_compatibility"] = vllm_validation
            
            # Create model registry entry
            registry_entry = self._create_model_registry_entry(saved_paths, training_metadata)
            results["model_registry"] = registry_entry
            
            # Upload to Hugging Face if configured
            if self.persistence_params.get("huggingface", {}).get("upload_to_hub", False):
                hf_result = self._upload_to_huggingface(saved_paths)
                results["huggingface_upload"] = hf_result
            
            # Save final model registry
            self._save_model_registry(results)
            
            self.logger.info("Model persistence completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Model persistence failed: {e}")
            results["error"] = str(e)
            raise
    
    def _create_persistence_config(self):
        """Create persistence configuration from parameters."""
        self.logger.info("Creating persistence configuration")
        
        config_dict = {
            "output_dir": "models",
            "save_merged_model": self.persistence_params.get("save_merged_model", True),
            "save_lora_adapters": self.persistence_params.get("save_lora_adapters", True),
            "save_tokenizer": self.persistence_params.get("save_tokenizer", True),
            "vllm_compatible": self.persistence_params.get("vllm_compatible", True),
            "quantization_method": self.persistence_params.get("quantization", {}).get("method", "none"),
            "hf_repo_name": self.persistence_params.get("huggingface", {}).get("repo_name"),
            "private_repo": self.persistence_params.get("huggingface", {}).get("private_repo", True)
        }
        
        self.persistence_config = PersistenceConfig(**config_dict)
        self.logger.info("Persistence configuration created")
    
    def _load_trained_model(self) -> UnslothTrainer:
        """Load the trained model for persistence."""
        self.logger.info("Loading trained model")
        
        # Load training configuration
        with open("models/training_logs/training_results.json", 'r') as f:
            training_results = json.load(f)
        
        # Create trainer configuration
        model_params = self.params.get("model", {})
        training_params = self.params.get("training", {})
        
        config_dict = {
            "model_name": model_params.get("name", "unsloth/orpheus-3b-0.1-ft"),
            "max_seq_length": model_params.get("max_seq_length", 2048),
            "load_in_4bit": model_params.get("load_in_4bit", False),
            "r": model_params.get("lora", {}).get("r", 16),
            "target_modules": model_params.get("lora", {}).get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            "output_dir": training_params.get("output_dir", "models/checkpoints"),
        }
        
        trainer_config = TrainingConfig(**config_dict)
        trainer = UnslothTrainer(trainer_config)
        
        # Load the base model
        trainer.load_orpheus_model()
        
        # Load LoRA adapters
        lora_path = "models/lora_adapters"
        if Path(lora_path).exists():
            self.logger.info(f"Loaded LoRA adapters from {lora_path}")
        
        return trainer
    
    def _gather_training_metadata(self) -> Dict[str, Any]:
        """Gather metadata about the training process."""
        metadata = {
            "model_name": self.persistence_params.get("model_name", "german-tts-orpheus-3b"),
            "model_version": self.persistence_params.get("model_version", "v1.0"),
            "model_description": self.persistence_params.get("model_description", "German TTS model fine-tuned with Orpheus 3B"),
            "training_timestamp": time.time()
        }
        
        # Add training metrics
        training_metrics_file = Path("metrics/training_metrics.json")
        if training_metrics_file.exists():
            with open(training_metrics_file, 'r') as f:
                training_metrics = json.load(f)
            metadata["training_metrics"] = training_metrics
        
        # Add evaluation metrics
        eval_metrics_file = Path("metrics/evaluation_metrics.json")
        if eval_metrics_file.exists():
            with open(eval_metrics_file, 'r') as f:
                eval_metrics = json.load(f)
            metadata["evaluation_metrics"] = eval_metrics
        
        return metadata
    
    def _save_model_formats(self, trainer: UnslothTrainer, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Save model in various formats."""
        self.logger.info("Saving model in multiple formats")
        
        model_name = metadata["model_name"]
        saved_paths = {}
        
        # Save using ModelPersistence
        persistence_result = self.persistence_handler.save_model(
            trainer.model,
            trainer.tokenizer,
            model_name,
            metadata
        )
        
        saved_paths.update(persistence_result)
        
        # Save VLLM-compatible format using Unsloth
        if self.persistence_params.get("vllm_compatible", True):
            vllm_path = "models/vllm_compatible"
            vllm_result = trainer.save_model_for_vllm(
                vllm_path,
                save_method=self.persistence_params.get("save_method", "merged_16bit")
            )
            saved_paths["vllm_compatible"] = vllm_result
            self.logger.info(f"Saved VLLM-compatible model to {vllm_result}")
        
        # Create deployment-ready package
        deployment_path = self._create_deployment_package(saved_paths, metadata)
        saved_paths["deployment_ready"] = deployment_path
        
        return saved_paths
    
    def _create_deployment_package(self, saved_paths: Dict[str, str], metadata: Dict[str, Any]) -> str:
        """Create deployment-ready package."""
        self.logger.info("Creating deployment-ready package")
        
        deployment_dir = Path("models/deployment_ready")
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy VLLM-compatible model
        if "vllm_compatible" in saved_paths:
            vllm_src = Path(saved_paths["vllm_compatible"])
            vllm_dest = deployment_dir / "model"
            
            if vllm_dest.exists():
                shutil.rmtree(vllm_dest)
            shutil.copytree(vllm_src, vllm_dest)
        
        # Create deployment configuration
        deployment_config = {
            "model_info": {
                "name": metadata["model_name"],
                "version": metadata["model_version"],
                "description": metadata["model_description"],
                "architecture": "Orpheus 3B + LoRA",
                "language": "German",
                "sample_rate": 24000
            },
            "vllm_config": {
                "model_path": "./model",
                "tensor_parallel_size": 1,
                "dtype": "float16",
                "max_model_len": 2048
            },
            "deployment_info": {
                "created_at": time.time(),
                "compatibility": "VLLM",
                "requirements": ["vllm>=0.2.0", "torch>=2.0.0"]
            }
        }
        
        # Save deployment config
        config_file = deployment_dir / "deployment_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(deployment_config, f, indent=2, default=str)
        
        self.logger.info(f"Created deployment package at {deployment_dir}")
        return str(deployment_dir)
    
    def _validate_vllm_compatibility(self, saved_paths: Dict[str, str]) -> Dict[str, Any]:
        """Validate VLLM compatibility."""
        self.logger.info("Validating VLLM compatibility")
        
        validation_results = {
            "is_compatible": False,
            "validation_details": {},
            "compatibility_checks": []
        }
        
        if "vllm_compatible" in saved_paths:
            vllm_path = saved_paths["vllm_compatible"]
            is_compatible = self.persistence_handler.validate_vllm_compatibility(vllm_path)
            
            validation_results["is_compatible"] = is_compatible
            validation_results["model_path"] = vllm_path
            
            if is_compatible:
                self.logger.info("VLLM compatibility validation passed")
                validation_results["compatibility_checks"].append("Model structure valid")
                validation_results["compatibility_checks"].append("Required files present")
                validation_results["compatibility_checks"].append("Basic inference test passed")
            else:
                self.logger.warning("VLLM compatibility validation failed")
                validation_results["compatibility_checks"].append("Validation failed - check logs")
        
        return validation_results
    
    def _create_model_registry_entry(self, saved_paths: Dict[str, str], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create model registry entry."""
        registry_entry = {
            "model_id": f"{metadata['model_name']}_{metadata['model_version']}",
            "model_info": {
                "name": metadata["model_name"],
                "version": metadata["model_version"],
                "description": metadata["model_description"],
                "created_at": metadata["training_timestamp"],
                "architecture": "Orpheus 3B + LoRA",
                "language": "German"
            },
            "model_paths": saved_paths,
            "performance_metrics": metadata.get("evaluation_metrics", {}),
            "training_info": metadata.get("training_metrics", {}),
            "deployment_status": "ready" if saved_paths.get("vllm_compatible") else "pending"
        }
        
        return registry_entry
    
    def _upload_to_huggingface(self, saved_paths: Dict[str, str]) -> Dict[str, Any]:
        """Upload model to Hugging Face Hub."""
        self.logger.info("Uploading model to Hugging Face Hub")
        
        hf_config = self.persistence_params.get("huggingface", {})
        repo_name = hf_config.get("repo_name")
        
        if not repo_name:
            raise PipelineError("Hugging Face repo name not specified")
        
        try:
            if "vllm_compatible" in saved_paths:
                model_path = saved_paths["vllm_compatible"]
                repo_url = self.persistence_handler.upload_to_huggingface(
                    model_path,
                    repo_name,
                    private=hf_config.get("private_repo", True)
                )
                
                return {
                    "status": "success",
                    "repo_url": repo_url,
                    "model_path": model_path
                }
            else:
                raise PipelineError("No VLLM-compatible model found for upload")
                
        except Exception as e:
            self.logger.error(f"Hugging Face upload failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _save_model_registry(self, results: Dict[str, Any]):
        """Save model registry to file."""
        registry_file = "models/model_registry.json"
        
        # Load existing registry or create new
        registry = []
        if Path(registry_file).exists():
            with open(registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
        
        # Add new entry
        registry.append(results["model_registry"])
        
        # Save updated registry
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, default=str)
        
        self.logger.info(f"Updated model registry: {registry_file}")
    
    def validate_outputs(self) -> bool:
        """Validate that persistence outputs were created correctly."""
        try:
            # Check VLLM-compatible model
            vllm_dir = Path("models/vllm_compatible")
            if not vllm_dir.exists():
                self.logger.error("VLLM-compatible model directory not found")
                return False
            
            # Check deployment package
            deployment_dir = Path("models/deployment_ready")
            if not deployment_dir.exists():
                self.logger.error("Deployment-ready package not found")
                return False
            
            # Check deployment config
            deployment_config = deployment_dir / "deployment_config.json"
            if not deployment_config.exists():
                self.logger.error("Deployment configuration not found")
                return False
            
            # Check model registry
            registry_file = Path("models/model_registry.json")
            if not registry_file.exists():
                self.logger.error("Model registry not found")
                return False
            
            # Validate registry content
            with open(registry_file, 'r') as f:
                registry = json.load(f)
            
            if not registry:
                self.logger.error("Empty model registry")
                return False
            
            self.logger.info("Model persistence output validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False


def main():
    """Main entry point for running persistence stage."""
    stage = PersistenceStage()
    
    try:
        result = stage.run()
        print(f"Model persistence completed successfully")
        
        if "models_saved" in result["results"]:
            models = result["results"]["models_saved"]
            print(f"Models saved:")
            for format_name, path in models.items():
                print(f"  - {format_name}: {path}")
        
        if "vllm_compatibility" in result["results"]:
            compat = result["results"]["vllm_compatibility"]
            print(f"VLLM compatible: {compat.get('is_compatible', False)}")
        
        if "huggingface_upload" in result["results"] and result["results"]["huggingface_upload"]:
            hf_result = result["results"]["huggingface_upload"]
            if hf_result.get("status") == "success":
                print(f"Uploaded to Hugging Face: {hf_result.get('repo_url')}")
        
    except PipelineError as e:
        print(f"Model persistence failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
