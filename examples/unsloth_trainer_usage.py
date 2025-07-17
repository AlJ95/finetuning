#!/usr/bin/env python3
"""
Example usage of UnslothTrainer for German TTS fine-tuning with Orpheus 3B.

This example demonstrates how to use the UnslothTrainer class to fine-tune
the Orpheus 3B model with German TTS datasets using Unsloth optimizations.
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.unsloth_trainer import UnslothTrainer, TrainingConfig
from src.mls_german_processor import MLSGermanDataProcessor
from src.torsten_voice_processor import TorstenVoiceDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting Unsloth Trainer example for German TTS fine-tuning")
    
    # Configuration for training
    config = TrainingConfig(
        model_name="unsloth/orpheus-3b-0.1-ft",
        max_seq_length=2048,
        load_in_4bit=False,  # Keep False for TTS quality
        
        # LoRA parameters optimized for TTS
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        
        # Training parameters
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        max_steps=-1,
        
        # Audio parameters
        target_sample_rate=24000,
        max_audio_length=30.0,
        min_audio_length=0.5,
        
        # Output configuration
        output_dir="outputs/orpheus_german_tts",
        logging_steps=10,
        save_steps=500,
    )
    
    # Initialize trainer
    logger.info("Initializing UnslothTrainer...")
    trainer = UnslothTrainer(config)
    
    # Display memory stats
    memory_stats = trainer.get_memory_stats()
    logger.info(f"Memory stats: {memory_stats}")
    
    # Example 1: Load and validate Orpheus model
    logger.info("\n=== Example 1: Loading Orpheus 3B Model ===")
    try:
        model, tokenizer = trainer.load_orpheus_model()
        logger.info("✓ Orpheus 3B model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Tokenizer type: {type(tokenizer)}")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.info("Note: This is expected if Unsloth is not installed or model is not available")
        return
    
    # Example 2: Prepare sample dataset
    logger.info("\n=== Example 2: Dataset Preparation ===")
    
    # Create sample audio datasets (normally you'd load from actual processors)
    from src.data_processor_base import AudioDataset
    
    sample_datasets = [
        AudioDataset(
            file_path=Path("sample1.wav"),
            text_transcript="Hallo, ich bin ein deutsches Text-zu-Sprache-System.",
            duration=3.5,
            sample_rate=24000,
            quality_score=0.9
        ),
        AudioDataset(
            file_path=Path("sample2.wav"),
            text_transcript="Guten Tag, wie geht es Ihnen heute?",
            duration=2.8,
            sample_rate=24000,
            quality_score=0.85
        ),
        AudioDataset(
            file_path=Path("sample3.wav"),
            text_transcript="Das Wetter ist heute sehr schön.",
            duration=2.2,
            sample_rate=24000,
            quality_score=0.92
        )
    ]
    
    try:
        # Prepare dataset for Unsloth
        hf_dataset = trainer.prepare_dataset_for_unsloth(sample_datasets)
        logger.info(f"✓ Dataset prepared with {len(hf_dataset)} samples")
        
        # Format training data
        formatted_dataset = trainer.format_training_data(hf_dataset)
        logger.info("✓ Training data formatted for TTS")
        
    except Exception as e:
        logger.error(f"✗ Dataset preparation failed: {e}")
        return
    
    # Example 3: Configure training
    logger.info("\n=== Example 3: Training Configuration ===")
    try:
        sft_trainer = trainer.configure_training()
        logger.info("✓ SFT trainer configured successfully")
        logger.info(f"Trainer type: {type(sft_trainer)}")
    except Exception as e:
        logger.error(f"✗ Training configuration failed: {e}")
        return
    
    # Example 4: Memory-efficient training (demo - not actual training)
    logger.info("\n=== Example 4: Training Demo ===")
    logger.info("Note: This would start actual training with real datasets")
    logger.info("Training parameters:")
    logger.info(f"  - Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  - Learning rate: {config.learning_rate}")
    logger.info(f"  - Epochs: {config.num_train_epochs}")
    logger.info(f"  - LoRA rank: {config.r}")
    logger.info(f"  - Target sample rate: {config.target_sample_rate} Hz")
    
    # Example 5: Model saving for VLLM
    logger.info("\n=== Example 5: VLLM Model Saving Demo ===")
    try:
        output_path = "outputs/orpheus_vllm_demo"
        # Note: This would normally be called after training
        logger.info(f"Would save model for VLLM at: {output_path}")
        logger.info("Save method: merged_16bit (for VLLM compatibility)")
    except Exception as e:
        logger.error(f"Model saving demo failed: {e}")
    
    # Example 6: Integration with data processors
    logger.info("\n=== Example 6: Data Processor Integration ===")
    
    # Example paths (adjust to your actual data locations)
    torsten_path = Path("D:/Trainingsdaten/TTS/torstenvoicedataset2022.10.zip")
    mls_path = Path("D:/Trainingsdaten/TTS/mls_german_opus.tar.gz")
    
    if torsten_path.exists():
        logger.info("✓ Torsten Voice dataset found")
        logger.info("  - Would use TorstenVoiceDataProcessor")
        logger.info("  - LJSpeech format with German audio")
    else:
        logger.info("ℹ Torsten Voice dataset not found at expected path")
    
    if mls_path.exists():
        logger.info("✓ MLS German dataset found")
        logger.info("  - Would use MLSGermanDataProcessor")
        logger.info("  - Multi-speaker German LibriSpeech format")
    else:
        logger.info("ℹ MLS German dataset not found at expected path")
    
    # Example workflow
    logger.info("\n=== Complete Training Workflow ===")
    logger.info("1. Load datasets with TorstenVoiceDataProcessor or MLSGermanDataProcessor")
    logger.info("2. Filter and validate audio quality")
    logger.info("3. Initialize UnslothTrainer with optimized config")
    logger.info("4. Load Orpheus 3B model with Unsloth optimizations")
    logger.info("5. Prepare datasets for Unsloth-compatible training")
    logger.info("6. Configure SFT trainer with TTS-specific parameters")
    logger.info("7. Start memory-efficient fine-tuning")
    logger.info("8. Save model in VLLM-compatible format")
    logger.info("9. Validate model loading and basic inference")
    
    logger.info("\n=== Memory Optimization Features ===")
    logger.info("✓ Unsloth FastLanguageModel for efficient loading")
    logger.info("✓ LoRA fine-tuning for reduced memory usage")
    logger.info("✓ Gradient checkpointing enabled")
    logger.info("✓ 8-bit AdamW optimizer")
    logger.info("✓ Audio length filtering to prevent OOM")
    logger.info("✓ Batch size optimization for TTS")
    
    logger.info("\nUnsloth Trainer example completed successfully!")


if __name__ == "__main__":
    main()