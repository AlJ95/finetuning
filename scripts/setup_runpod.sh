#!/bin/bash
# German TTS Fine-tuning Setup für Runpod.io
# Dieses Skript richtet die komplette Umgebung für Remote-TTS-Training ein

set -e  # Beende bei Fehlern

echo "=== German TTS Fine-tuning Setup für Runpod.io ==="

# System-Updates
echo "Aktualisiere System..."
apt update && apt upgrade -y

# Installiere System-Abhängigkeiten
echo "Installiere System-Abhängigkeiten..."
apt install -y git htop tmux screen vim wget curl

# CUDA-Check
echo "=== CUDA Status ==="
nvidia-smi

# Python Environment Setup
echo "Erstelle Python Environment..."
python3 -m pip install --upgrade pip

# PyTorch mit CUDA Support
echo "Installiere PyTorch mit CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth Installation (TTS-spezifisch)
echo "Installiere Unsloth für TTS..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# Audio Processing Libraries
echo "Installiere Audio Libraries..."
pip install librosa soundfile scipy numpy pandas

# Hugging Face und Datasets
echo "Installiere Hugging Face Tools..."
pip install datasets transformers tokenizers

# Monitoring und Logging
echo "Installiere Monitoring Tools..."
pip install tensorboard wandb matplotlib seaborn

# Erstelle Verzeichnisstruktur
echo "Erstelle Verzeichnisstruktur..."
mkdir -p /workspace/{data,models,logs,scripts,configs}

# Setze Environment Variablen
echo "export PYTHONPATH=/workspace:$PYTHONPATH" >> ~/.bashrc
echo "cd /workspace" >> ~/.bashrc

# Erstelle tmux Session für persistentes Training
echo "Erstelle tmux Session..."
tmux new-session -d -s tts_training
tmux set-option -g mouse on

# Installations-Check
echo "=== Installations-Check ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA verfügbar: {torch.cuda.is_available()}')"
python -c "import unsloth; print('Unsloth: OK')"
python -c "import librosa; print('Librosa: OK')"

echo "=== Setup abgeschlossen! ==="
echo "Verwende 'tmux attach -t tts_training' für persistente Sessions"
echo "Starte Jupyter mit: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
