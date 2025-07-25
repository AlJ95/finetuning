# 1. Environment Setup Guide

## Before You Start
✔ System Requirements:  
- Linux (recommended) or Windows WSL2  
- NVIDIA GPU with ≥16GB VRAM  
- Python 3.10+  

## Installation Steps

### 1. Python Environment
```bash
python -m venv tts-env
source tts-env/bin/activate  # Linux/Mac
tts-env\Scripts\activate     # Windows
```

### 2. Core Dependencies
```bash
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. German TTS Specific Packages
```bash
pip install phonemizer[de]  # German phoneme support
```

### 4. Verify Installation
```bash
python test_import.py  # Should run without errors
```

## German TTS Notes
⚠️ Special considerations:
- Ensure locale is set to de_DE.UTF-8 for proper text processing
- Umlaut handling requires UTF-8 encoding throughout pipeline

## What's Next
→ [Pipeline Overview](2_pipeline.md)  
→ [Troubleshooting Setup](5_troubleshooting.md#setup)
