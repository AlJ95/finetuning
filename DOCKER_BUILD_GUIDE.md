# 🐳 Custom Docker Image für TTS Training

Diese Anleitung zeigt, wie du ein **optimiertes Docker Image** erstellst, das die xFormers C++/CUDA Extensions korrekt unterstützt.

## 🎯 Problem und Lösung

**Problem:** Das aktuelle Runpod Image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` hat xFormers Kompatibilitätsprobleme.

**Lösung:** Eigenes Docker Image mit korrekten Versionen und kompiliertem xFormers.

## 📋 Schritt-für-Schritt Anleitung

### 1. Docker Image bauen

```bash
# Docker Image bauen
docker build -f Dockerfile.custom -t tts-training-cuda128:latest .

# Oder für Runpod (mit Docker Hub)
docker build -f Dockerfile.custom -t yourusername/tts-training-cuda128:latest .
docker push yourusername/tts-training-cuda128:latest
```

### 2. Runpod Custom Image verwenden

**In Runpod:**
1. **Deploy** → **Secure Cloud** → **GPU**
2. **Custom Image** auswählen
3. **Image URL:** `yourusername/tts-training-cuda128:latest`
4. **Environment Variables:**
   ```
   HF_TOKEN=your_huggingface_token
   WANDB_API_KEY=your_wandb_key
   ```

### 3. Alternative: Lokales Build und Upload

```bash
# Lokales Build
docker build -f Dockerfile.custom -t tts-training-cuda128:latest .

# Als .tar für Runpod Upload
docker save tts-training-cuda128:latest > tts-training-cuda128.tar
```

## 🔧 Verifizierung nach Build

```bash
# Image testen
docker run --gpus all -it tts-training-cuda128:latest

# Im Container:
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import xformers; print('xFormers: OK')"
python -c "import unsloth; print('Unsloth: OK')"
```

## 📊 Performance Vergleich

| Image | xFormers | CUDA | PyTorch | Größe | Build-Zeit |
|-------|----------|------|---------|-------|------------|
| Runpod Original | ❌ Probleme | 12.8 | 2.8 | ~8GB | - |
| Custom Image | ✅ Kompatibel | 12.8 | 2.8 | ~12GB | ~30min |

## 🚀 Schnellstart mit Custom Image

### Option A: Docker Hub (Empfohlen)
```bash
# Bereits gebautes Image verwenden
# Image: tts-training-cuda128:latest
```

### Option B: Selbst bauen
```bash
# 1. Repository klonen
git clone https://github.com/AlJ95/finetuning.git
cd finetuning

# 2. Image bauen
docker build -f Dockerfile.custom -t tts-training-cuda128:latest .

# 3. In Runpod verwenden
# Custom Image: tts-training-cuda128:latest
```

## 📝 Dockerfile Details

### Key Features:
- **NVIDIA CUDA 12.8** mit Development Tools
- **PyTorch 2.8** mit CUDA 12.8 Support
- **xFormers** von Quelle kompiliert für maximale Kompatibilität
- **Unsloth** mit allen Dependencies
- **Audio Libraries** (librosa, soundfile, scipy)
- **Development Tools** (vim, tmux, htop)

### System Requirements:
- **GPU:** NVIDIA RTX 4090 oder besser
- **CUDA:** 12.8 kompatibel
- **RAM:** 32GB+ empfohlen
- **Storage:** 200GB+ für Datasets

## 🔄 Update Prozess

```bash
# Image aktualisieren
docker pull yourusername/tts-training-cuda128:latest

# Oder neu bauen
docker build --no-cache -f Dockerfile.custom -t tts-training-cuda128:latest .
```

## 🆘 Troubleshooting

### Build Fehler
```bash
# Falls Build fehlschlägt
docker build --no-cache -f Dockerfile.custom -t tts-training-cuda128:latest .

# Spezifische Fehler debuggen
docker build --progress=plain -f Dockerfile.custom -t tts-training-cuda128:latest .
```

### CUDA nicht erkannt
```bash
# NVIDIA Container Toolkit prüfen
docker run --gpus all nvidia/cuda:12.8-devel-ubuntu22.04 nvidia-smi
```

## ✅ Erfolgs-Checkliste

- [ ] Docker Image erfolgreich gebaut
- [ ] xFormers ohne Fehler geladen
- [ ] PyTorch CUDA korrekt
- [ ] Unsloth funktioniert
- [ ] Audio Libraries verfügbar
- [ ] In Runpod getestet

**Das Custom Image löst das xFormers Problem und bietet optimale Performance für TTS Training!**
