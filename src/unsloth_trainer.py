"""
Unsloth integration for Orpheus 3B TTS fine-tuning.

This module provides the UnslothTrainer class for memory-efficient fine-tuning
of the Orpheus 3B model using the Unsloth library with German TTS datasets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
import torch
from datasets import Dataset, Audio
import numpy as np
from tqdm import tqdm

# Unsloth imports with compatibility checking
try:
    import torch
    # Check PyTorch version compatibility
    torch_version = torch.__version__
    major, minor = map(int, torch_version.split('.')[:2])
    
    if major > 2 or (major == 2 and minor > 3):
        logging.warning(f"PyTorch {torch_version} may not be compatible with current Unsloth version. Consider downgrading to PyTorch 2.3.x")
    
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from transformers import TrainingArguments
    from trl import SFTTrainer
    UNSLOTH_AVAILABLE = True
    logging.info(f"Unsloth loaded successfully with PyTorch {torch_version}")
    
except ImportError as e:
    logging.warning(f"Unsloth not available: {e}")
    UNSLOTH_AVAILABLE = False
    # Import fallback dependencies
    try:
        from transformers import TrainingArguments
    except ImportError:
        TrainingArguments = None
except AttributeError as e:
    logging.warning(f"Unsloth compatibility issue: {e}")
    logging.warning("This is likely due to PyTorch version incompatibility. Consider using PyTorch 2.3.x with Unsloth")
    UNSLOTH_AVAILABLE = False
    # Import fallback dependencies
    try:
        from transformers import TrainingArguments
    except ImportError:
        TrainingArguments = None
except Exception as e:
    logging.warning(f"Unexpected error loading Unsloth: {e}")
    UNSLOTH_AVAILABLE = False
    # Import fallback dependencies
    try:
        from transformers import TrainingArguments
    except ImportError:
        TrainingArguments = None

from .data_processor_base import AudioDataset


@dataclass
class TrainingConfig:
    """Configuration for Unsloth TTS training with Orpheus 3B."""
    
    # Model configuration
    model_name: str = "unsloth/orpheus-3b-0.1-ft"
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # Auto-detection
    load_in_4bit: bool = False  # Important for TTS quality
    
    # LoRA configuration
    r: int = 16  # LoRA rank
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None
    
    # Training parameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    num_train_epochs: int = 3
    max_steps: int = -1
    
    # Audio-specific parameters
    target_sample_rate: int = 24000  # Orpheus 3B requirement
    max_audio_length: float = 30.0  # seconds
    min_audio_length: float = 0.5   # seconds
    
    # Output configuration
    output_dir: str = "outputs"
    logging_steps: int = 1
    save_steps: int = 500
    save_total_limit: int = 3
    report_to: str = "none"
    
    # Memory optimization
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    group_by_length: bool = True
    
    def to_training_arguments(self) -> TrainingArguments:
        """Convert to Transformers TrainingArguments."""
        return TrainingArguments(
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.random_state,
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            save_steps=self.save_steps,
            save_total_limit=self.save_total_limit,
            report_to=self.report_to,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            group_by_length=self.group_by_length,
        )


@dataclass
class TrainingResults:
    """Results from TTS training."""
    
    final_loss: float
    training_time: float
    total_steps: int
    model_path: str
    tokenizer_path: str
    training_logs: List[Dict[str, Any]]
    memory_stats: Dict[str, Any]


class UnslothTrainer:
    """
    Unsloth-based trainer for Orpheus 3B TTS fine-tuning.
    
    Provides memory-efficient training with German TTS datasets using
    Unsloth optimizations for faster training and reduced VRAM usage.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize the Unsloth trainer."""
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth is not available. Please install it first.")
        
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Initialize device and memory tracking
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
    
    def load_orpheus_model(self) -> Tuple[Any, Any]:
        """
        Load Orpheus 3B model with Unsloth optimizations.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading Orpheus 3B model: {self.config.model_name}")
        
        try:
            # Load model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            self.logger.info("Model loaded successfully")
            
            # Configure for LoRA fine-tuning
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config.r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.bias,
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
                random_state=self.config.random_state,
                use_rslora=self.config.use_rslora,
                loftq_config=self.config.loftq_config,
            )
            
            self.logger.info("LoRA configuration applied")
            
            # Store references
            self.model = model
            self.tokenizer = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"Error loading Orpheus model: {e}")
            raise
    
    def prepare_dataset_for_unsloth(self, dataset: List[AudioDataset]) -> Dataset:
        """
        Prepare dataset for Unsloth-compatible training.
        
        Args:
            dataset: List of AudioDataset objects
            
        Returns:
            Hugging Face Dataset formatted for TTS training
        """
        self.logger.info(f"Preparing dataset with {len(dataset)} samples")
        
        # Filter by audio length
        filtered_dataset = []
        for item in dataset:
            if (self.config.min_audio_length <= item.duration <= self.config.max_audio_length):
                filtered_dataset.append(item)
            else:
                self.logger.debug(f"Filtered out {item.file_path} (duration: {item.duration:.2f}s)")
        
        self.logger.info(f"After filtering: {len(filtered_dataset)} samples")
        
        # Prepare data for Hugging Face Dataset
        data_dict = {
            'audio': [],
            'text': [],
            'file_path': [],
            'duration': [],
            'quality_score': []
        }
        
        for item in filtered_dataset:
            data_dict['audio'].append(str(item.file_path))
            data_dict['text'].append(item.text_transcript)
            data_dict['file_path'].append(str(item.file_path))
            data_dict['duration'].append(item.duration)
            data_dict['quality_score'].append(item.quality_score)
        
        # Create Hugging Face Dataset
        hf_dataset = Dataset.from_dict(data_dict)
        
        # Cast audio column to Audio feature with target sample rate
        hf_dataset = hf_dataset.cast_column(
            "audio", 
            Audio(sampling_rate=self.config.target_sample_rate)
        )
        
        self.logger.info("Dataset prepared for Unsloth training")
        return hf_dataset
    
    def format_training_data(self, dataset: Dataset) -> Dataset:
        """
        Format dataset for TTS training with proper prompt structure.
        
        Args:
            dataset: Hugging Face Dataset with audio and text
            
        Returns:
            Formatted dataset for training
        """
        def formatting_prompts_func(examples):
            """Format examples for TTS training."""
            texts = []
            
            for text in examples["text"]:
                # Create TTS-specific prompt format optimized for German
                # Include audio duration and quality hints for better training
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a high-quality German text-to-speech system. Generate natural, expressive German speech with proper pronunciation and intonation.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                texts.append(prompt)
            
            return {"text": texts}
        
        # Apply formatting with error handling
        try:
            formatted_dataset = dataset.map(
                formatting_prompts_func,
                batched=True,
                desc="Formatting prompts for German TTS",
                remove_columns=dataset.column_names  # Remove original columns to save memory
            )
            
            self.logger.info("Training data formatted for German TTS")
            return formatted_dataset
            
        except Exception as e:
            self.logger.error(f"Error formatting training data: {e}")
            raise
    
    def configure_training(self):
        """
        Configure SFT trainer for TTS fine-tuning.
        
        Returns:
            Configured SFTTrainer instance
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        # Get training arguments
        training_args = self.config.to_training_arguments()
        
        self.logger.info("Configuring SFT trainer")
        
        # Note: Dataset will be provided during training
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=None,  # Will be set during training
            args=training_args,
            packing=False,  # Important for TTS
            max_seq_length=self.config.max_seq_length,
        )
        
        self.trainer = trainer
        self.logger.info("SFT trainer configured")
        
        return trainer
    
    def start_finetuning(self, dataset: List[AudioDataset]) -> TrainingResults:
        """
        Start fine-tuning process with the provided dataset.
        
        Args:
            dataset: List of AudioDataset objects for training
            
        Returns:
            TrainingResults with training metrics and paths
        """
        self.logger.info("Starting TTS fine-tuning process")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_orpheus_model()
        
        # Prepare dataset
        hf_dataset = self.prepare_dataset_for_unsloth(dataset)
        formatted_dataset = self.format_training_data(hf_dataset)
        
        # Configure trainer if not already done
        if self.trainer is None:
            self.configure_training()
        
        # Set the dataset for training
        self.trainer.train_dataset = formatted_dataset
        
        # Track memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
            self.logger.info(f"GPU memory before training: {memory_before / 1e9:.2f} GB")
        
        # Start training
        import time
        start_time = time.time()
        
        try:
            self.logger.info("Beginning training...")
            trainer_stats = self.trainer.train()
            
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Track memory after training
            memory_stats = {}
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_stats = {
                    'memory_before_gb': memory_before / 1e9,
                    'memory_after_gb': memory_after / 1e9,
                    'memory_peak_gb': torch.cuda.max_memory_allocated() / 1e9
                }
                self.logger.info(f"GPU memory after training: {memory_after / 1e9:.2f} GB")
                self.logger.info(f"Peak GPU memory: {memory_stats['memory_peak_gb']:.2f} GB")
            
            # Save model and tokenizer
            model_path = Path(self.config.output_dir) / "lora_model"
            model_path.mkdir(parents=True, exist_ok=True)
            
            self.model.save_pretrained(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))
            
            self.logger.info(f"Model saved to: {model_path}")
            
            # Create training results
            results = TrainingResults(
                final_loss=trainer_stats.training_loss,
                training_time=training_time,
                total_steps=trainer_stats.global_step,
                model_path=str(model_path),
                tokenizer_path=str(model_path),
                training_logs=trainer_stats.log_history,
                memory_stats=memory_stats
            )
            
            self.logger.info("Fine-tuning completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def save_model_for_vllm(self, output_path: str, save_method: str = "merged_16bit") -> str:
        """
        Save model in VLLM-compatible format.
        
        Args:
            output_path: Path to save the model
            save_method: Unsloth save method ("merged_16bit", "lora", etc.)
            
        Returns:
            Path to saved model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        self.logger.info(f"Saving model for VLLM with method: {save_method}")
        
        try:
            # Use Unsloth's optimized saving
            self.model.save_pretrained_merged(
                output_path,
                self.tokenizer,
                save_method=save_method
            )
            
            self.logger.info(f"Model saved for VLLM at: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving model for VLLM: {e}")
            raise
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_available': True,
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        else:
            stats['gpu_available'] = False
        
        # Add system memory info if available
        try:
            import psutil
            stats.update({
                'system_memory_available_gb': psutil.virtual_memory().available / 1e9,
                'system_memory_total_gb': psutil.virtual_memory().total / 1e9,
                'system_memory_percent': psutil.virtual_memory().percent,
            })
        except ImportError:
            pass
        
        return stats
    
    def validate_model_loading(self, model_path: str) -> bool:
        """
        Validate that a saved model can be loaded correctly.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if model loads successfully, False otherwise
        """
        try:
            self.logger.info(f"Validating model loading from: {model_path}")
            
            # Try to load the model
            test_model, test_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=self.config.max_seq_length,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            # Basic validation - check if model can process a simple input
            test_input = "Hallo, dies ist ein Test."
            inputs = test_tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = test_model(**inputs)
            
            self.logger.info("Model validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
