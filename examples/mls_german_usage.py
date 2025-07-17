"""
Example usage of MLSGermanDataProcessor for processing Multilingual LibriSpeech German dataset.

This example demonstrates how to:
1. Configure the processor for MLS German data
2. Process the dataset with multi-speaker support
3. Apply speaker balancing for training
4. Generate comprehensive statistics
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from mls_german_processor import MLSGermanDataProcessor
from data_processor_base import ProcessingConfig


def main():
    """Main example function."""
    print("=== MLS German DataProcessor Usage Example ===\n")
    
    # 1. Configure processing parameters for MLS
    print("1. Configuring processor...")
    config = ProcessingConfig(
        min_duration=2.0,          # MLS has longer utterances than typical TTS
        max_duration=20.0,         # Allow longer clips for better training
        min_snr=8.0,              # Slightly lower SNR threshold for MLS
        quality_threshold=0.5,     # Lower threshold due to multi-speaker variability
        max_workers=4,            # Parallel processing
        batch_size=100            # Process in batches
    )
    
    # 2. Create processor with speaker balancing options
    print("2. Creating MLS German processor...")
    processor = MLSGermanDataProcessor(
        config=config,
        target_speakers=None,      # Process all speakers (or specify list like ['1001', '1002'])
        max_samples_per_speaker=1000  # Limit samples per speaker for balanced training
    )
    
    print(f"   - Expected sample rate: {processor.expected_sample_rate}Hz")
    print(f"   - Audio format: {processor.audio_format}")
    print(f"   - Multi-speaker support: enabled")
    print(f"   - Speaker balancing: {processor.max_samples_per_speaker} samples max per speaker")
    
    # 3. Example dataset path (adjust to your actual path)
    dataset_path = Path("D:/Trainingsdaten/TTS/mls_german_opus")
    
    if not dataset_path.exists():
        print(f"\n⚠️  Dataset path not found: {dataset_path}")
        print("   Please adjust the dataset_path variable to point to your MLS German dataset.")
        print("   Expected structure:")
        print("   mls_german_opus/")
        print("   ├── train/")
        print("   │   ├── audio/")
        print("   │   │   └── {speaker_id}/")
        print("   │   │       └── {book_id}/")
        print("   │   │           └── *.opus")
        print("   │   └── transcripts.txt")
        print("   ├── dev/")
        print("   └── test/")
        return
    
    # 4. Process the dataset
    print(f"\n3. Processing dataset from: {dataset_path}")
    print("   This may take several minutes for the full MLS dataset...")
    
    try:
        # Process dataset (this will take time for the full dataset)
        processed_dataset = processor.process_dataset(dataset_path)
        
        print(f"   ✅ Successfully processed {len(processed_dataset)} audio files")
        
        # 5. Generate comprehensive statistics
        print("\n4. Generating statistics...")
        stats = processor.get_processing_stats(processed_dataset)
        
        print(f"   📊 Dataset Statistics:")
        print(f"      - Total items: {stats['total_items']}")
        print(f"      - Total duration: {stats['total_duration_hours']:.2f} hours")
        print(f"      - Average duration: {stats['avg_duration_seconds']:.2f} seconds")
        print(f"      - Average quality score: {stats['avg_quality_score']:.3f}")
        print(f"      - Sample rates: {stats['sample_rates']}")
        
        # MLS-specific statistics
        mls_stats = stats['mls_specific']
        print(f"   🎤 Speaker Statistics:")
        print(f"      - Total speakers: {mls_stats['total_speakers']}")
        print(f"      - Average samples per speaker: {mls_stats['avg_samples_per_speaker']:.1f}")
        print(f"      - Balance coefficient: {mls_stats['balance_coefficient']:.3f} (0=balanced, 1=imbalanced)")
        
        # Show top speakers
        speaker_dist = mls_stats['speaker_distribution']
        top_speakers = sorted(speaker_dist.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"      - Top 5 speakers: {top_speakers}")
        
        # 6. Apply speaker balancing if needed
        if mls_stats['balance_coefficient'] > 0.3:  # If dataset is imbalanced
            print(f"\n5. Applying speaker balancing...")
            balanced_dataset = processor.filter_by_speaker_balance(processed_dataset, 500)
            
            print(f"   ⚖️  Balanced dataset: {len(balanced_dataset)} samples")
            
            # Generate stats for balanced dataset
            balanced_stats = processor.get_processing_stats(balanced_dataset)
            balanced_mls_stats = balanced_stats['mls_specific']
            
            print(f"      - New balance coefficient: {balanced_mls_stats['balance_coefficient']:.3f}")
            print(f"      - Duration after balancing: {balanced_stats['total_duration_hours']:.2f} hours")
        else:
            print(f"\n5. Dataset is already well-balanced, no filtering needed.")
            balanced_dataset = processed_dataset
        
        # 7. Example: Save dataset information for training
        print(f"\n6. Dataset ready for training!")
        print(f"   - Final dataset size: {len(balanced_dataset)} samples")
        print(f"   - Ready for Unsloth TTS training with Orpheus 3B")
        print(f"   - Multi-speaker conditioning available")
        
        # Example of accessing individual samples
        if balanced_dataset:
            sample = balanced_dataset[0]
            print(f"\n   📝 Sample entry:")
            print(f"      - File: {sample.file_path}")
            print(f"      - Text: {sample.text_transcript[:50]}...")
            print(f"      - Duration: {sample.duration:.2f}s")
            print(f"      - Quality: {sample.quality_score:.3f}")
            print(f"      - Speaker: {sample.metadata.get('original_metadata', {}).get('speaker_id', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ Error processing dataset: {e}")
        print("   Please check that the dataset path is correct and accessible.")


def demonstrate_speaker_filtering():
    """Demonstrate speaker-specific filtering."""
    print("\n=== Speaker Filtering Example ===")
    
    # Create processor for specific speakers only
    target_speakers = ['2422', '4536', '6507']  # High-sample speakers from MLS
    
    processor = MLSGermanDataProcessor(
        target_speakers=target_speakers,
        max_samples_per_speaker=200
    )
    
    print(f"Configured for speakers: {target_speakers}")
    print(f"Max samples per speaker: {processor.max_samples_per_speaker}")
    print("This configuration would process only the specified speakers.")


def demonstrate_quality_filtering():
    """Demonstrate quality-based filtering."""
    print("\n=== Quality Filtering Example ===")
    
    # Create processor with strict quality requirements
    strict_config = ProcessingConfig(
        min_duration=3.0,          # Longer minimum duration
        max_duration=15.0,         # Shorter maximum duration
        min_snr=12.0,             # Higher SNR requirement
        quality_threshold=0.7,     # Higher quality threshold
    )
    
    processor = MLSGermanDataProcessor(config=strict_config)
    
    print("Strict quality configuration:")
    print(f"  - Min duration: {strict_config.min_duration}s")
    print(f"  - Max duration: {strict_config.max_duration}s")
    print(f"  - Min SNR: {strict_config.min_snr}dB")
    print(f"  - Quality threshold: {strict_config.quality_threshold}")
    print("This would result in a smaller but higher-quality dataset.")


if __name__ == "__main__":
    main()
    demonstrate_speaker_filtering()
    demonstrate_quality_filtering()
    
    print("\n=== Usage Complete ===")
    print("The MLSGermanDataProcessor is ready for use with your MLS German dataset!")