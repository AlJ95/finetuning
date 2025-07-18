#!/usr/bin/env python3
"""
Quick Start Guide for German TTS Fine-tuning

This script provides a minimal example to get started with German TTS training
using the Thorsten-Voice dataset.

Usage:
    python examples/quick_start.py --dataset-path /path/to/thorsten-voice
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.torsten_voice_processor import TorstenVoiceDataProcessor
from src.unsloth_trainer import UnslothTrainer, TrainingConfig
from src.data_processor_base import ProcessingConfig


def quick_start(dataset_path: str, output_dir: str = "outputs"):
    """
    Quick start function for German TTS training.
    
    Args:
        dataset_path: Path to Thorsten-Voice dataset
        output_dir: Output directory for models
    
    Returns:
        Training results
    """
    print("🚀 German TTS Quick Start")
    print("=" * 50)
    
    # Step 1: Process dataset with sensible defaults
    print("📊 Processing Thorsten-Voice dataset...")
    
    processing_config = ProcessingConfig(
        min_duration=1.0,
        max_duration=10.0,
        target_sample_rate=24000,
        min_snr=15.0,
        quality_threshold=0.7,
        batch_size=100,
        max_workers=4
    )
    
    processor = TorstenVoiceDataProcessor(config=processing_config)
    dataset = processor.process_dataset(Path(dataset_path))
    
    stats = processor.get_processing_stats(dataset)
    print(f"✅ Processed {stats['total_items']} audio samples")
    print(f"📈 Total duration: {stats['total_duration_hours']:.2f} hours")
    
    # Step 2: Train model with optimized settings
    print("\n🎯 Training Orpheus 3B model...")
    
    training_config = TrainingConfig(
        model_name="unsloth/orpheus-3b-0.1-ft",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        max_audio_length=10.0,
        min_audio_length=1.0,
        output_dir=output_dir,
        save_steps=500,
        logging_steps=1
    )
    
    trainer = UnslothTrainer(config=training_config)
    results = trainer.start_finetuning(dataset)
    
    print("\n✅ Training completed!")
    print(f"📊 Final loss: {results.final_loss:.4f}")
    print(f"⏱️  Training time: {results.training_time:.2f} seconds")
    print(f"🎯 Model saved to: {results.model_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start German TTS training')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to Thorsten-Voice dataset')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    try:
        results = quick_start(args.dataset_path, args.output_dir)
        print("\n🎉 Quick start completed successfully!")
    except Exception as e:
        print(f"\n❌ Quick start failed: {e}")
        sys.exit(1)
