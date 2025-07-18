# Remote TTS Training Scripts für Runpod.io

Diese Sammlung von Skripten ermöglicht das vollständige Remote-Training von deutschen TTS-Modellen auf Runpod.io mit SSH-Zugang.

## Schnellstart

### 1. SSH Setup
```bash
# SSH Key erstellen
./scripts/ssh_connect.sh setup-key

# Verbindung herstellen (ersetze IP durch deine Runpod-Instance)
./scripts/ssh_connect.sh connect jacxa38ckzy1ye-64411204 my-tts-instance
```

### 2. Remote Instance einrichten
```bash
# Automatisches Setup auf Remote Instance
./scripts/ssh_connect.sh setup-remote jacxa38ckzy1ye-64411204
```

### 3. Training durchführen
```bash
# Auf Remote Instance verbinden
./scripts/ssh_connect.sh connect jacxa38ckzy1ye-64411204

# Auf der Remote Instance:
cd /workspace
python scripts/download_datasets.py --datasets thorsten --thorsten-size 1000
python scripts/train_remote.py --dataset thorsten --epochs 3 --batch-size 2
```

## Skript-Übersicht

### `setup_runpod.sh`
Vollautomatisches Setup der Runpod-Instance:
- Installiert alle benötigten Dependencies
- Konfiguriert Python Environment
- Erstellt Verzeichnisstruktur
- Startet tmux Session für persistentes Training

**Verwendung:**
```bash
./scripts/setup_runpod.sh
```

### `download_datasets.py`
Laden deutscher TTS-Datensätze:
- Thorsten Voice Dataset
- MLS German Subset
- Test-Dataset für schnelle Tests

**Verwendung:**
```bash
# Alle Datensätze
python scripts/download_datasets.py --datasets all

# Nur Thorsten Voice mit Subset
python scripts/download_datasets.py --datasets thorsten --thorsten-size 5000

# MLS German mit 10000 Samples
python scripts/download_datasets.py --datasets mls --mls-size 10000
```

### `train_remote.py`
Haupt-Training-Script:
- Unterstützt Thorsten Voice und MLS German
- Konfigurierbare Training-Parameter
- Automatisches Modell-Speichern
- Optionaler Hugging Face Export

**Verwendung:**
```bash
# Standard Training mit Thorsten Voice
python scripts/train_remote.py --dataset thorsten --epochs 3

# Training mit MLS German und kleinerem Batch
python scripts/train_remote.py --dataset mls_german --batch-size 1 --epochs 2

# Mit Hugging Face Export
python scripts/train_remote.py --dataset thorsten --push-to-hub --hub-repo username/german-tts
```

### `monitor_training.py`
Überwachung und Debugging:
- GPU-Auslastung in Echtzeit
- Festplatten- und System-Status
- Training-Fortschritt
- TensorBoard Integration

**Verwendung:**
```bash
# Kontinuierliche Überwachung
python scripts/monitor_training.py --interval 30

# Einmaliger Status-Check
python scripts/monitor_training.py --once

# Mit TensorBoard
python scripts/monitor_training.py --tensorboard
```

### `ssh_connect.sh`
SSH-Verbindungsmanagement:
- Automatisches SSH Key Setup
- Port-Forwarding für Jupyter/TensorBoard
- Datei-Upload/Download
- Konfigurationsmanagement

**Verwendung:**
```bash
# SSH Key erstellen
./scripts/ssh_connect.sh setup-key

# Verbindung mit Port-Forwarding
./scripts/ssh_connect.sh connect-port jacxa38ckzy1ye-64411204

# Dateien hochladen
./scripts/ssh_connect.sh upload jacxa38ckzy1ye-64411204 ./local/path /workspace/remote/path

# Dateien herunterladen
./scripts/ssh_connect.sh download jacxa38ckzy1ye-64411204 /workspace/remote/path ./local/path
```

## Kompletter Workflow

### Phase 1: Setup
```bash
# 1. SSH Key erstellen
./scripts/ssh_connect.sh setup-key

# 2. Runpod Instance erstellen und IP notieren
# 3. Remote Setup durchführen
./scripts/ssh_connect.sh setup-remote <INSTANCE_IP>
```

### Phase 2: Training
```bash
# 1. Verbindung herstellen
./scripts/ssh_connect.sh connect-port <INSTANCE_IP>

# 2. Auf Remote Instance:
tmux new -s tts_training

# 3. Datensätze herunterladen
python scripts/download_datasets.py --datasets thorsten --thorsten-size 5000

# 4. Training starten
python scripts/train_remote.py --dataset thorsten --epochs 3 --batch-size 2

# 5. In anderem Terminal: Monitoring starten
python scripts/monitor_training.py --tensorboard
```

### Phase 3: Ergebnisse
```bash
# Modell herunterladen
./scripts/ssh_connect.sh download <INSTANCE_IP> /workspace/models/thorsten_tts ./models/

# Oder direkt zu Hugging Face exportieren
python scripts/train_remote.py --push-to-hub --hub-repo username/german-tts-orpheus
```

## Konfiguration

### Environment Variablen
```bash
# Auf Remote Instance setzen:
export HF_TOKEN="dein-huggingface-token"
export WANDB_API_KEY="dein-wandb-key"
```

### GPU-Konfiguration
- **Empfohlen:** RTX 4090 (24GB VRAM) - $0.75/h
- **Alternative:** A100 40GB - $1.20/h
- **Storage:** 150GB NVMe empfohlen

### Kostenübersicht
- **Thorsten Voice (5000 Samples):** ~$3-5
- **MLS German (10000 Samples):** ~$8-12
- **Setup-Zeit:** 10-15 Minuten
- **Training-Zeit:** 2-4 Stunden

## Debugging

### Häufige Probleme

1. **CUDA Out of Memory**
   ```bash
   # Batch-Größe reduzieren
   python scripts/train_remote.py --batch-size 1 --gradient-accumulation-steps 4
   ```

2. **SSH Verbindungsabbruch**
   ```bash
   # tmux Session verwenden
   tmux attach -t tts_training
   ```

3. **Dataset Download Fehler**
   ```bash
   # Hugging Face Token setzen
   huggingface-cli login
   ```

### Monitoring URLs
- **TensorBoard:** http://localhost:6006 (nach Port-Forwarding)
- **Jupyter:** http://localhost:8888 (nach Port-Forwarding)

## Tipps für effizientes Remote-Training

1. **tmux Sessions** für persistentes Training
2. **Screen** als Alternative zu tmux
3. **nohup** für lange Läufe: `nohup python train_remote.py &`
4. **rsync** für schnelle Datei-Synchronisation
5. **wandb** für erweitertes Monitoring

## Support und Troubleshooting

Bei Problemen:
1. Überprüfe GPU-Auslastung mit `nvidia-smi`
2. Prüfe Festplatten-Platz mit `df -h`
3. Nutze das Monitoring-Script für detaillierte Informationen
