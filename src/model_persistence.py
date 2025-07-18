"""
Model persistence module for VLLM-compatible TTS model export.

This module provides functionality to save trained TTS models in formats
compatible with VLLM deployment, including both merged models and LoRA adapters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json
import os
from datetime import datetime

# Optional imports for Hugging Face Hub
try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("Hugging Face Hub not available. Install with: pip install huggingface_hub")

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    logging.warning("Unsloth not available for model persistence")


@dataclass
class PersistenceConfig:
    """Configuration for model persistence."""
    
    output_dir: str = "models"
    save_merged_model: bool = True
    save_lora_adapters: bool = True
    save_tokenizer: bool = True
    create_subdirs: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # VLLM-specific settings
    vllm_compatible: bool = True
    quantization_method: str = "none"  # none, 4bit, 8bit
    
    # Hugging Face settings
    hf_repo_name: Optional[str] = None
    hf_token: Optional[str] = None
    private_repo: bool = True


class ModelPersistence:
    """
    Handles model persistence for VLLM-compatible TTS model deployment.
    
    Provides functionality to:
    - Save models in VLLM-compatible formats
    - Export LoRA adapters separately
    - Upload to Hugging Face Hub
    - Validate VLLM compatibility
    """
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        """Initialize model persistence handler."""
        self.config = config or PersistenceConfig()
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
    
    def save_model(
        self,
        model: Any,
        tokenizer: Any,
        model_name: str,
        training_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Save model in VLLM-compatible formats.
        
        Args:
            model: The trained model
            tokenizer: The tokenizer
            model_name: Name for the saved model
            training_metadata: Optional training metadata to save
            
        Returns:
            Dictionary with paths to saved model files
        """
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth not available for model persistence")
        
        # Create output directory
        timestamp = datetime.now().strftime(self.config.timestamp_format)
        base_dir = Path(self.config.output_dir) / f"{model_name}_{timestamp}"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        try:
            # Save merged model for VLLM
            if self.config.save_merged_model:
                merged_path = base_dir / "merged_model"
                merged_path.mkdir(exist_ok=True)
                
                self.logger.info("Saving merged model for VLLM...")
                model.save_pretrained_merged(
                    str(merged_path),
                    tokenizer,
                    save_method="merged_16bit"
                )
                saved_paths["merged_model"] = str(merged_path)
                self.logger.info(f"Merged model saved to: {merged_path}")
                
                # Validate VLLM compatibility
                is_compatible = self.validate_vllm_compatibility(str(merged_path))
                saved_paths["vllm_compatible"] = str(is_compatible)
            
            # Save LoRA adapters
            if self.config.save_lora_adapters:
                lora_path = base_dir / "lora_adapters"
                lora_path.mkdir(exist_ok=True)
                
                self.logger.info("Saving LoRA adapters...")
                model.save_pretrained_merged(
                    str(lora_path),
                    tokenizer,
                    save_method="lora"
                )
                saved_paths["lora_adapters"] = str(lora_path)
                self.logger.info(f"LoRA adapters saved to: {lora_path}")
            
            # Save tokenizer separately
            if self.config.save_tokenizer:
                tokenizer_path = base_dir / "tokenizer"
                tokenizer_path.mkdir(exist_ok=True)
                
                tokenizer.save_pretrained(str(tokenizer_path))
                saved_paths["tokenizer"] = str(tokenizer_path)
                self.logger.info(f"Tokenizer saved to: {tokenizer_path}")
            
            # Save training metadata
            if training_metadata:
                metadata_path = base_dir / "training_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(training_metadata, f, indent=2, default=str)
                saved_paths["metadata"] = str(metadata_path)
            
            # Save persistence info
            persistence_info = {
                "model_name": model_name,
                "timestamp": timestamp,
                "config": self.config.__dict__,
                "saved_paths": saved_paths,
                "created_at": datetime.now().isoformat()
            }
            
            info_path = base_dir / "persistence_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(persistence_info, f, indent=2, default=str)
            
            saved_paths["info"] = str(info_path)
            saved_paths["base_dir"] = str(base_dir)
            
            self.logger.info(f"Model persistence completed. Base directory: {base_dir}")
            return saved_paths
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def upload_to_huggingface(
        self,
        model_path: str,
        repo_name: str,
        private: bool = True,
        token: Optional[str] = None
    ) -> str:
        """
        Upload model to Hugging Face Hub.
        
        Args:
            model_path: Local path to the model
            repo_name: Repository name on Hugging Face
            private: Whether to create a private repository
            token: Hugging Face token (optional if set in config)
            
        Returns:
            Repository URL
        """
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("Hugging Face Hub not available")
        
        if not token:
            token = self.config.hf_token
        
        if not token:
            raise ValueError("Hugging Face token required")
        
        try:
            # Create repository
            repo_url = create_repo(
                repo_name,
                token=token,
                private=private,
                exist_ok=True
            )
            
            # Upload model files
            upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                token=token,
                commit_message=f"Upload TTS model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            self.logger.info(f"Model uploaded to Hugging Face: {repo_url}")
            return repo_url
            
        except Exception as e:
            self.logger.error(f"Error uploading to Hugging Face: {e}")
            raise
    
    def validate_vllm_compatibility(self, model_path: str) -> bool:
        """
        Validate that a saved model is compatible with VLLM.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if model is VLLM-compatible
        """
        try:
            self.logger.info(f"Validating VLLM compatibility for: {model_path}")
            
            # Check if model directory exists
            model_dir = Path(model_path)
            if not model_dir.exists():
                self.logger.error(f"Model directory not found: {model_path}")
                return False
            
            # Check for required files
            required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
            missing_files = []
            
            for file_name in required_files:
                file_path = model_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                self.logger.warning(f"Missing required files: {missing_files}")
                return False
            
            # Try to load with transformers (VLLM compatibility check)
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                # Test tokenizer loading
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.logger.info("Tokenizer loaded successfully")
                
                # Test model loading (without loading to GPU)
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_dir),
                    torch_dtype="auto",
                    device_map="cpu"
                )
                self.logger.info("Model loaded successfully")
                
                # Test basic inference
                inputs = tokenizer("Hallo Welt", return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                self.logger.info("Basic inference test passed")
                return True
                
            except Exception as e:
                self.logger.error(f"VLLM compatibility check failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating VLLM compatibility: {e}")
            return False
    
    def load_model_info(self, base_dir: str) -> Dict[str, Any]:
        """
        Load model persistence information.
        
        Args:
            base_dir: Base directory of the saved model
            
        Returns:
            Dictionary with model information
        """
        info_path = Path(base_dir) / "persistence_info.json"
        
        if not info_path.exists():
            raise FileNotFoundError(f"Model info not found: {info_path}")
        
        with open(info_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_saved_models(self) -> list:
        """
        List all saved models in the output directory.
        
        Returns:
            List of model information dictionaries
        """
        output_dir = Path(self.config.output_dir)
        if not output_dir.exists():
            return []
        
        models = []
        for model_dir in output_dir.iterdir():
            if model_dir.is_dir():
                info_path = model_dir / "persistence_info.json"
                if info_path.exists():
                    try:
                        model_info = self.load_model_info(str(model_dir))
                        models.append(model_info)
                    except Exception as e:
                        self.logger.warning(f"Error loading model info for {model_dir}: {e}")
        
        return sorted(models, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def cleanup_old_models(self, keep_last: int = 3) -> int:
        """
        Clean up old model directories, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent models to keep
            
        Returns:
            Number of directories removed
        """
        models = self.list_saved_models()
        if len(models) <= keep_last:
            return 0
        
        models_to_remove = models[keep_last:]
        removed_count = 0
        
        for model_info in models_to_remove:
            base_dir = model_info.get('base_dir')
            if base_dir and Path(base_dir).exists():
                try:
                    import shutil
                    shutil.rmtree(base_dir)
                    self.logger.info(f"Removed old model: {base_dir}")
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Error removing {base_dir}: {e}")
        
        return removed_count


# Integration with UnslothTrainer
def add_persistence_to_trainer(trainer_instance, persistence_config: Optional[PersistenceConfig] = None):
    """
    Add model persistence functionality to an UnslothTrainer instance.
    
    Args:
        trainer_instance: Instance of UnslothTrainer
        persistence_config: Optional persistence configuration
        
    Returns:
        ModelPersistence instance
    """
    persistence = ModelPersistence(persistence_config)
    
    # Add save method to trainer
    def save_trained_model(self, model_name: str, training_metadata: Optional[Dict[str, Any]] = None):
        """Save the trained model with VLLM compatibility."""
        return persistence.save_model(
            self.model,
            self.tokenizer,
            model_name,
            training_metadata
        )
    
    # Bind method to trainer instance
    import types
    trainer_instance.save_trained_model = types.MethodType(save_trained_model, trainer_instance)
    
    return persistence


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = PersistenceConfig(
        output_dir="models",
        save_merged_model=True,
        save_lora_adapters=True,
        vllm_compatible=True
    )
    
    persistence = ModelPersistence(config)
    
    # Example: List saved models
    models = persistence.list_saved_models()
    print(f"Found {len(models)} saved models")
    
    # Example: Cleanup old models
    removed = persistence.cleanup_old_models(keep_last=5)
    print(f"Removed {removed} old model directories")
