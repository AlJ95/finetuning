import sys
sys.path.insert(0, 'src')

try:
    from torsten_voice_processor import TorstenVoiceDataProcessor
    print("Import successful!")
    
    # Test instantiation
    processor = TorstenVoiceDataProcessor()
    print(f"Processor created with sample rate: {processor.config.target_sample_rate}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()