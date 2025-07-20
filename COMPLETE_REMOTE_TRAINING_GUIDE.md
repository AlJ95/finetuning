# 🚀 Komplette Remote TTS Training Anleitung

Diese Anleitung bietet eine **vollständige und getestete** Lösung für das Remote-Training deutscher TTS-Modelle auf Runpod.io mit SSH-Zugang.

## 📋 Inhaltsverzeichnis

1. [Vorbereitung](#vorbereitung)
2. [SSH-Setup](#ssh-setup)
3. [Runpod-Instance erstellen](#runpod-instance-erstellen)
4. [Automatisches Setup](#automatisches-setup)
5. [Training durchführen](#training-durchführen)
6. [Fehlerbehebung](#fehlerbehebung)
7. [Monitoring](#monitoring)
8. [Ergebnisse sichern](#ergebnisse-sichern)

---

## 🎯 Vorbereitung

### Lokale Voraussetzungen
```bash
# SSH Client installieren (falls nicht vorhanden)
# Ubuntu/Debian:
sudo apt update && sudo apt install openssh-client

# macOS: SSH ist bereits installiert

# Windows: Git Bash oder WSL verwenden
```

### 1. SSH Key erstellen
```bash
# SSH Key für Runpod erstellen
./scripts/ssh_connect.sh setup-key

# Alternative manuell:
ssh-keygen -t ed25519 -C "runpod-tts" -f ~/.ssh/runpod-tts -N ""
```

### 2. Environment Variablen vorbereiten
```bash
# Hugging Face Token (für Dataset Downloads)
export HF_TOKEN="your_huggingface_token"

# Weights & Biases (optional)
export WANDB_API_KEY="your_wandb_key"
```

---

## 🔐 SSH-Setup

### SSH Key zur Runpod-Instance hinzufügen

1. **Public Key anzeigen:**
```bash
cat ~/.ssh/runpod-tts.pub
```

2. **In Runpod einfügen:**
   - Gehe zu Runpod → Settings → SSH Keys
   - "New SSH Key" → Namen vergeben → Public Key einfügen

### SSH Config für einfachen Zugriff
```bash
# ~/.ssh/config erstellen/anpassen
cat >> ~/.ssh/config << 'EOF'
Host runpod-tts
    HostName ssh.runpod.io
    User YOUR_POD_ID
    IdentityFile ~/.ssh/runpod-tts
    Port 22
EOF
```

---

## 🖥️ Runpod-Instance erstellen

### Empfohlene Konfiguration
```yaml
# Primäre Wahl für deutsches TTS Training
Image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
GPU: RTX 4090 (24GB VRAM) - $0.75/hour
CPU: 8-16 vCPUs
RAM: 32-64GB
Storage: 200GB Container Disk
Ports: 22, 8888, 6006, 8080
```

### Environment Variables setzen
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## ⚡ Automatisches Setup

### 1. Verbindung herstellen
```bash
# Mit Pod ID verbinden
./scripts/ssh_connect.sh connect YOUR_POD_ID

# Oder mit SSH Config
ssh runpod-tts
```

### 2. Automatisches Setup starten
```bash
# Auf der Remote Instance:
cd /workspace
git clone https://github.com/AlJ95/finetuning.git
cd finetuning
chmod +x scripts/*.sh
./scripts/setup_runpod.sh
```

### 3. Setup verifizieren
```bash
# System Check
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import unsloth; print('Unsloth: OK')"
```

---

## 🎯 Training durchführen

### Datensätze herunterladen
```bash
# Alle Datensätze
python scripts/download_datasets.py --datasets all

# Nur Thorsten Voice mit 1000 Samples
python scripts/download_datasets.py --datasets thorsten --thorsten-size 1000

# MLS German mit 5000 Samples
python scripts/download_datasets.py --datasets mls --mls-size 5000
```

### Training starten
```bash
# Standard Training (Thorsten Voice, 3 Epochen)
python scripts/train_remote.py --dataset thorsten --epochs 3 --batch-size 2

# Training mit MLS German
python scripts/train_remote.py --dataset mls_german --epochs 2 --batch-size 1

# Mit Hugging Face Export
python scripts/train_remote.py --dataset thorsten --epochs 3 --push-to-hub --hub-repo username/german-tts
```

### Training in tmux Session
```bash
# Neue Session erstellen
tmux new -s tts_training

# Training starten
python scripts/train_remote.py --dataset thorsten --epochs 5

# Session verlassen (Training läuft weiter)
Ctrl+B, dann D

# Zurück zur Session
tmux attach -t tts_training
```

---

## 🔍 Monitoring

### Echtzeit-Überwachung
```bash
# Kontinuierliche Überwachung
python scripts/monitor_training.py --interval 30

# Einmaliger Check
python scripts/monitor_training.py --once

# Mit TensorBoard
python scripts/monitor_training.py --tensorboard
```

### TensorBoard starten
```bash
# Auf Remote Instance
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# Lokaler Zugriff mit Port-Forwarding
ssh -L 6006:localhost:6006 YOUR_POD_ID@ssh.runpod.io
# Dann: http://localhost:6006
```

---

## 🐛 Fehlerbehebung

### Häufige Probleme und Lösungen

#### 1. SSH Verbindung fehlgeschlagen
```bash
# Problem: Permission denied (publickey)
# Lösung:
chmod 600 ~/.ssh/runpod-tts
ssh-add ~/.ssh/runpod-tts

# Oder explizit Key angeben:
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io
```

#### 2. CUDA Out of Memory
```bash
# Batch-Größe reduzieren
python scripts/train_remote.py --batch-size 1 --gradient-accumulation-steps 4

# Oder Gradient Checkpointing aktivieren
python scripts/train_remote.py --gradient-checkpointing
```

#### 3. Dataset Download fehlgeschlagen
```bash
# Hugging Face Token prüfen
huggingface-cli login

# Manuelles Download als Fallback
python -c "from datasets import load_dataset; print('OK')"
```

#### 4. Git Clone mit SSH Key
```bash
# Falls HTTPS nicht funktioniert, SSH verwenden:
git clone git@github.com:AlJ95/finetuning.git

# SSH Agent starten (falls nötig)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_key
```

#### 5. Container startet nicht
```bash
# Container neu starten
runpodctl stop POD_ID
runpodctl start POD_ID

# Oder neue Instance erstellen
```

---

## 📊 Monitoring & Debugging

### System-Status prüfen
```bash
# GPU-Auslastung
watch -n 1 nvidia-smi

# Festplatten-Platz
df -h

# Prozesse
htop

# Training-Logs
tail -f logs/training.log
```

### Performance-Optimierung
```bash
# Environment Variables für bessere Performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=true
```

---

## 💾 Ergebnisse sichern

### Modelle herunterladen
```bash
# Gesamtes Modell-Verzeichnis
scp -r YOUR_POD_ID@ssh.runpod.io:/workspace/models ./local_backup/

# Einzelne Modelle
scp YOUR_POD_ID@ssh.runpod.io:/workspace/models/thorsten_tts/model.pth ./models/

# Mit rsync (schneller)
rsync -avz -e ssh YOUR_POD_ID@ssh.runpod.io:/workspace/models/ ./models/
```

### Automatisches Backup
```bash
# Backup-Script erstellen
cat > backup_models.sh << 'EOF'
#!/bin/bash
POD_ID=$1
rsync -avz -e ssh $POD_ID@ssh.runpod.io:/workspace/models/ ./models_backup/$(date +%Y%m%d)/
EOF
chmod +x backup_models.sh
```

---

## 🔄 Kompletter Workflow

### Schnellstart (5 Minuten)
```bash
# 1. SSH Key erstellen
./scripts/ssh_connect.sh setup-key

# 2. Runpod Instance erstellen und IP notieren
# 3. Verbinden und Setup
./scripts/ssh_connect.sh setup YOUR_POD_ID

# 4. Training starten
./scripts/ssh_connect.sh train YOUR_POD_ID thorsten 3
```

### Detaillierter Workflow
```bash
# Phase 1: Setup (10 Minuten)
./scripts/ssh_connect.sh setup-key
# Runpod Instance erstellen
./scripts/ssh_connect.sh connect YOUR_POD_ID
# Auf Remote: ./scripts/setup_runpod.sh

# Phase 2: Training (2-4 Stunden)
tmux new -s training
python scripts/download_datasets.py --datasets thorsten --thorsten-size 5000
python scripts/train_remote.py --dataset thorsten --epochs 3 --batch-size 2

# Phase 3: Ergebnisse (5 Minuten)
./scripts/ssh_connect.sh download YOUR_POD_ID /workspace/models ./models/
```

---

## 📈 Kostenübersicht

| Konfiguration | Preis/Std | Thorsten 5k | MLS 10k | Gesamt |
|---------------|-----------|-------------|---------|---------|
| RTX 4090 24GB | $0.75 | ~$3-5 | ~$8-12 | ~$11-17 |
| A100 40GB | $1.20 | ~$5-8 | ~$12-18 | ~$17-26 |
| A100 80GB | $2.00 | ~$8-12 | ~$20-30 | ~$28-42 |

---

## 🆘 Support & Troubleshooting

### Debug-Modus aktivieren
```bash
# Training mit Debug-Output
python scripts/train_remote.py --debug --dataset thorsten --epochs 1

# SSH mit verbose
ssh -v YOUR_POD_ID@ssh.runpod.io
```

### Logs sammeln
```bash
# System-Informationen
python scripts/monitor_training.py --once > system_info.txt

# Training-Logs
cp logs/training.log ./training_debug.log
```

### Kontakt
Bei Problemen:
1. Überprüfe GPU-Auslastung: `nvidia-smi`
2. Prüfe Festplatten-Platz: `df -h`
3. Nutze das Monitoring-Script für detaillierte Informationen
4. Erstelle ein Issue mit den Logs

---

## 📝 Wichtige Befehle zum Kopieren

### SSH Verbindung
```bash
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io
```

### Training starten
```bash
python scripts/train_remote.py --dataset thorsten --epochs 3 --batch-size 2 --learning-rate 5e-5
```

### Monitoring
```bash
python scripts/monitor_training.py --tensorboard --interval 60
```

### Backup
```bash
rsync -avz -e ssh YOUR_POD_ID@ssh.runpod.io:/workspace/models/ ./models_backup/
