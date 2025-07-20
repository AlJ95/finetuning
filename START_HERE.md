# 🚀 START HIER - Dein Einstieg ins Remote TTS Training

**Diese Datei ist dein Einstiegspunkt!** Folge der Reihenfolge von oben nach unten.

## 📋 Schritt 1: Vorbereitung (2 Minuten)

### 1.1 SSH Key erstellen
```bash
# Terminal öffnen und ausführen:
./scripts/ssh_connect.sh setup-key

# Falls das Skript nicht existiert, manuell:
ssh-keygen -t ed25519 -C "runpod-tts" -f ~/.ssh/runpod-tts -N ""
```

### 1.2 Runpod Instance erstellen
1. Gehe zu [runpod.io](https://runpod.io)
2. Klicke "Deploy" → "Secure Cloud" → "GPU"
3. **Wichtig:** Diese Einstellungen verwenden:
   - **Image:** `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
   - **GPU:** RTX 4090 (24GB VRAM)
   - **Storage:** 200GB Container Disk
   - **Environment Variables:**
     ```
     HF_TOKEN=dein_huggingface_token
     WANDB_API_KEY=dein_wandb_key
     ```

## 📋 Schritt 2: SSH Key in Runpod einfügen (1 Minute)

```bash
# Public Key anzeigen
cat ~/.ssh/runpod-tts.pub

# Kopiere den Output und füge ihn ein:
# Runpod → Settings → SSH Keys → Add SSH Key
```

## 📋 Schritt 3: Verbindung herstellen (1 Minute)

```bash
# Ersetze YOUR_POD_ID durch deine tatsächliche Pod ID
./scripts/ssh_connect.sh connect YOUR_POD_ID

# Alternative manuell:
ssh -i ~/.ssh/runpod-tts YOUR_POD_ID@ssh.runpod.io
```

## 📋 Schritt 4: Automatisches Setup (5 Minuten)

**Nach der SSH Verbindung auf der Remote Instance:**
```bash
cd /workspace
git clone https://github.com/AlJ95/finetuning.git
cd finetuning
chmod +x scripts/*.sh
./scripts/setup_runpod.sh
```

## 📋 Schritt 5: Training starten (Sobald Setup fertig)

```bash
# Datensätze herunterladen
python scripts/download_datasets.py --datasets thorsten --thorsten-size 1000

# Training starten
python scripts/train_remote.py --dataset thorsten --epochs 3 --batch-size 2
```

## 🎯 Was tun wenn etwas nicht funktioniert?

### Problem: SSH Verbindung
→ Siehe **SSH_TROUBLESHOOTING_GUIDE.md**

### Problem: Dataset Download
→ Siehe **DATASET_FIXES.md**

### Problem: Allgemeines Setup
→ Siehe **COMPLETE_REMOTE_TRAINING_GUIDE.md**

## 🚀 Schnellstart für Experten

```bash
# Alles in einem Befehl (nachdem Runpod Instance erstellt wurde)
./scripts/ssh_connect.sh setup YOUR_POD_ID && ./scripts/ssh_connect.sh train YOUR_POD_ID thorsten 3
```

## 📞 Notfall-Kontakt

**Falls du stecken bleibst:**
1. Prüfe **SSH_TROUBLESHOOTING_GUIDE.md** für SSH-Probleme
2. Prüfe **DATASET_FIXES.md** für Dataset-Probleme
3. Erstelle ein Issue mit dem Befehl: `python scripts/monitor_training.py --once`

---

**✅ Fertig! Du kannst jetzt mit Schritt 1 beginnen.**
