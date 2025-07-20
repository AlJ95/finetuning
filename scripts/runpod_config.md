# Runpod.io Base Image und Konfiguration Guide

## ⚠️ WICHTIG: Aktualisierte Version mit PyTorch 2.8

## Empfohlenes Base Image (Neueste Empfehlung)

### Primäre Wahl - PyTorch 2.8 Official Runpod Image
**Image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **CUDA:** 12.8.1 (kompatibel mit PyTorch 2.8)
- **Python:** 3.11
- **PyTorch:** 2.8.0
- **cuDNN:** Entwickler-Version
- **Ubuntu:** 22.04

### Alternative Optionen

1. **Docker Hub:** `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`
2. **Neueste Version:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
3. **Minimal Setup:** `nvidia/cuda:12.8-devel-ubuntu22.04` + manuelles Setup

## Schritt-für-Schritt Setup in Runpod

### 1. Instance Erstellen (Neue Empfehlung)
1. **Login:** runpod.io
2. **Deploy:** Secure Cloud → GPU
3. **Template auswählen:** "PyTorch 2.8 CUDA 12.8" oder Custom Image
4. **Direktes Image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`

### 2. Empfohlene Konfiguration (PyTorch 2.8)
```yaml
GPU: RTX 4090 (24GB VRAM) - $0.75/hour
CPU: 8-16 vCPUs
RAM: 32-64GB
Storage: 200GB Container Disk (empfohlen für Performance)
Image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
Ports: 22, 8888, 6006, 8080
```

### Storage Empfehlung
- **Container Disk (empfohlen):** 200GB für maximale Performance
- **Volume Disk:** Optional für persistente Daten, aber langsamer
- **Mindestens:** 150GB für MLS German Dataset

### 3. Environment Variables (erforderlich)
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
HF_HUB_ENABLE_HF_TRANSFER=1  # Für schnellere Downloads
```

## SSH/SCP Commands für Datei-Transfer

### 1. SSH Verbindung herstellen
```bash
# Standard SSH Verbindung
ssh root@<INSTANCE_IP>

# Mit Port (falls geändert)
ssh -p 22 root@<INSTANCE_IP>

# Mit SSH Key
ssh -i ~/.ssh/id_rsa root@<INSTANCE_IP>
```

### 2. SCP Commands für Datei-Upload
```bash
# Gesamtes Projekt hochladen
scp -r /pfad/zu/finetuning root@<INSTANCE_IP>:/workspace/

# Einzelne Dateien hochladen
scp requirements.txt root@<INSTANCE_IP>:/workspace/
scp scripts/setup_runpod.sh root@<INSTANCE_IP>:/workspace/scripts/
scp -r src/ root@<INSTANCE_IP>:/workspace/

# Mit Port angeben
scp -P 22 -r /pfad/zu/finetuning root@<INSTANCE_IP>:/workspace/

# Mit SSH Key
scp -i ~/.ssh/id_rsa -r /pfad/zu/finetuning root@<INSTANCE_IP>:/workspace/
```

### 3. SCP Commands für Datei-Download
```bash
# Dateien vom Server herunterladen
scp root@<INSTANCE_IP>:/workspace/models/model.pth ./lokaler/ordner/

# Gesamtes Verzeichnis herunterladen
scp -r root@<INSTANCE_IP>:/workspace/models ./lokaler/ordner/

# Mit Port
scp -P 22 root@<INSTANCE_IP>:/workspace/output/* ./lokaler/ordner/

# Mit SSH Key
scp -i ~/.ssh/id_rsa root@<INSTANCE_IP>:/workspace/results/* ./lokaler/ordner/
```

### 4. Praktische Beispiele
```bash
# Beispiel: Gesamtes Projekt hochladen
scp -r /home/user/finetuning root@123.45.67.89:/workspace/

# Beispiel: Nur src und scripts hochladen
scp -r src/ scripts/ root@123.45.67.89:/workspace/

# Beispiel: Modelle herunterladen
scp -r root@123.45.67.89:/workspace/models ./backup/

# Beispiel: Einzelne Datei hochladen
scp requirements.txt root@123.45.67.89:/workspace/
```

## Vollständige Einrichtung nach dem Deploy

### 1. Dateien hochladen via SCP
```bash
# Von lokalem Computer zum Runpod Server
scp -r /pfad/zu/deinem/finetuning root@<INSTANCE_IP>:/workspace/

# Beispiel mit tatsächlichen Pfaden
scp -r /home/jan-albrecht/work/projects/internal/Piloting/Finetuning/TTS/finetuning root@<INSTANCE_IP>:/workspace/
```

### 2. SSH Verbindung herstellen
```bash
# Verbinde dich mit dem Server
ssh root@<INSTANCE_IP>

# Wechsle ins Projektverzeichnis
cd /workspace/finetuning

# Führe das Setup aus
chmod +x scripts/setup_runpod.sh
./scripts/setup_runpod.sh
```

### 3. Alternative: Einzelne Dateien hochladen
```bash
# Nur wichtige Dateien hochladen
scp requirements.txt root@<INSTANCE_IP>:/workspace/
scp scripts/setup_runpod.sh root@<INSTANCE_IP>:/workspace/scripts/
scp -r src/ root@<INSTANCE_IP>:/workspace/
scp -r scripts/ root@<INSTANCE_IP>:/workspace/
```

## Verifizierung nach Start (PyTorch 2.8)

Nach dem Deploy und Datei-Upload:
```bash
# SSH Verbindung
ssh root@<INSTANCE_IP>

# Ins Projektverzeichnis wechseln
cd /workspace/finetuning

# System Check
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Unsloth Installation
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
python -c "import unsloth; print('Unsloth: OK')"
```

## Troubleshooting SSH/SCP

### 1. Verbindungsprobleme
```bash
# Teste Verbindung
ssh -v root@<INSTANCE_IP>

# Port prüfen
telnet <INSTANCE_IP> 22

# Firewall prüfen
sudo ufw status
```

### 2. SCP Fehlerbehebung
```bash
# Mit verbose Output
scp -v -r /pfad/zu/finetuning root@<INSTANCE_IP>:/workspace/

# Falls Permission denied
chmod 600 ~/.ssh/id_rsa
ssh-add ~/.ssh/id_rsa

# Falls Host key problem
ssh-keygen -R <INSTANCE_IP>
```

### 3. Geschwindigkeit optimieren
```bash
# Mit Kompression für schnelleren Transfer
scp -C -r /pfad/zu/finetuning root@<INSTANCE_IP>:/workspace/

# Parallel transfers (rsync Alternative)
rsync -avz -e ssh /pfad/zu/finetuning root@<INSTANCE_IP>:/workspace/
```

## Performance-Optimierung (PyTorch 2.8)

### Environment Variables für bessere Performance
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=true
```

### Pre-Installation Script
```bash
# Führe dies nach dem Deploy aus:
curl -fsSL https://raw.githubusercontent.com/AlJ95/finetuning/main/scripts/setup_runpod.sh | bash

# Oder manuell:
chmod +x scripts/setup_runpod.sh
./scripts/setup_runpod.sh
```

## Migration zu PyTorch 2.8

### Wenn du auf PyTorch 2.8 aktualisierst:
```bash
# Backup wichtiger Daten
cp -r /workspace/models /workspace/models_backup

# Unsloth installieren
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Zusätzliche Pakete
pip install librosa soundfile scipy

# Requirements installieren
pip install -r requirements.txt
```

## Fehlerbehebung bei GPU-Problemen

### Falls `nvidia-smi` nicht funktioniert:
1. **Stelle sicher, dass du eine GPU-Instance ausgewählt hast**
2. **Überprüfe das Base Image** - muss CUDA enthalten
3. **Verwende das korrekte Kommando:** `nvidia-smi` (nicht `nvidia smi`)
4. **Falls nötig, installiere NVIDIA-Treiber:**
   ```bash
   apt-get update && apt-get install -y nvidia-utils-535
