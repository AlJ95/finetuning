#!/usr/bin/env python3
"""
End-to-End German TTS Training Example

This script demonstrates the complete pipeline for training a German TTS model
using the Thorsten-Voice dataset with Unsloth and Orpheus 3B.

Usage:
    python examples/end_to_end_training.py --dataset-path /path/to/thorsten-voice
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.torsten_voice_processor import TorstenVoiceDataProcessor
from src.unsloth_trainer import UnslothTrainer, TrainingConfig
from src.model_persistence import PersistenceConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='End-to-end German TTS training')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to Thorsten-Voice dataset')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max-audio-length', type=float, default=10.0,
                        help='Maximum audio length in seconds')
    parser.add_argument('--min-audio-length', type=float, default=1.0,
                        help='Minimum audio length in seconds')
    parser.add_argument('--quality-threshold', type=float, default=0.7,
                        help='Quality threshold for filtering')
    parser.add_argument('--save-vllm', action='store_true',
                        help='Save model in VLLM-compatible format')
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    logger = setup_logging()
    
    logger.info("Starting end-to-end German TTS training")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Step 1: Data Processing
    logger.info("Step 1: Processing Thorsten-Voice dataset")
    
    from src.data_processor_base import ProcessingConfig
    
    # Configure data processing
    processing_config = ProcessingConfig(
        min_duration=args.min_audio_length,
        max_duration=args.max_audio_length,
        target_sample_rate=24000,  # Orpheus 3B requirement
        min_snr=15.0,
        quality_threshold=args.quality_threshold,
        batch_size=100,
        max_workers=4
    )
    
    # Create processor and process dataset
    processor = TorstenVoiceDataProcessor(config=processing_config)
    dataset_path = Path(args.dataset_path)
    
    logger.info(f"Processing dataset from: {dataset_path}")
    processed_dataset = processor.process_dataset(dataset_path)
    
    # Get processing statistics
    stats = processor.get_processing_stats(processed_dataset)
    logger.info(f"Processing complete: {stats}")
    
    # Step 2: Model Training
    logger.info("Step 2: Training Orpheus 3B model")
    
    # Configure training
    training_config = TrainingConfig(
        model_name="unsloth/orpheus-3b-0.1-ft",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_audio_length=args.max_audio_length,
        min_audio_length=args.min_audio_length,
        output_dir=args.output_dir,
        save_steps=500,
        logging_steps=1
    )
    
    # Configure persistence for VLLM compatibility
    persistence_config = PersistenceConfig(
        base_model_dir=args.output_dir,
        vllm_compatible=True,
        save_merged_model=True,
        save_lora_adapters=True,
        compression_level=4
    )
    
    # Create trainer
    trainer = UnslothTrainer(
        config=training_config,
        persistence_config=persistence_config
    )
    
    # Start training
    logger.info("Starting training...")
    results = trainer.start_finetuning(processed_dataset)
    
    # Step 3: Save VLLM-compatible model (if requested)
    if args.save_vllm:
        logger.info("Step 3: Saving VLLM-compatible model")
        vllm_path = Path(args.output_dir) / "vllm_model"
        vllm_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model_for_vllm(str(vllm_path), save_method="merged_16bit")
        logger.info(f"VLLM model saved to: {vllm_path}")
    
    # Final summary
    logger.info("Training completed successfully!")
    logger.info(f"Final loss: {results.final_loss:.4f}")
    logger.info(f"Training time: {results.training_time:.2f} seconds")
    logger.info(f"Total steps: {results.total_steps}")
    logger.info(f"Model saved to: {results.model_path}")
    
    if results.memory_stats:
        logger.info(f"Peak GPU memory: {results.memory_stats.get('memory_peak_gb', 0):.2f} GB")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n✅ Training completed successfully!")
        print(f"📊 Final loss: {results.final_loss:.4f}")
        print(f"⏱️  Training time: {results.training_time:.2f}s")
        print(f"🎯 Model path: {results.model_path}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        sys.exit(1)
