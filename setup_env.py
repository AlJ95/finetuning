#!/usr/bin/env python3
"""
Setup script for German TTS Finetuning environment
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def test_imports():
    """Test if key libraries can be imported"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import librosa
        import soundfile
        print("✓ Audio libraries (librosa, soundfile) imported")
    except ImportError:
        print("✗ Audio libraries not found")
        return False
    
    try:
        import numpy
        import pandas
        print("✓ Data processing libraries imported")
    except ImportError:
        print("✗ Data processing libraries not found")
        return False
    
    return True

def check_directories():
    """Check if project directories exist"""
    dirs = ['data', 'models', 'src', 'tests']
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory '{dir_name}' exists")
        else:
            print(f"✗ Directory '{dir_name}' missing")
            return False
    return True

def main():
    print("German TTS Finetuning Environment Setup Check")
    print("=" * 50)
    
    if not check_python_version():
        sys.exit(1)
    
    if not check_directories():
        sys.exit(1)
    
    if not test_imports():
        print("\nSome libraries are missing. Please install requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n✓ Environment setup complete!")
    print("\nNext steps:")
    print("1. Place your datasets in the 'data/' directory")
    print("2. Run the first task from the implementation plan")

if __name__ == "__main__":
    main()