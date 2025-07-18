# Runpod.io Base Image und Konfiguration Guide

## Empfohlenes Base Image

### Primäre Wahl
**Image:** `runpod/pytorch:2.1.0-py3.10-cuda12.1-devel`
- **CUDA:** 12.1 (kompatibel mit Unsloth)
- **Python:** 3.10
- **PyTorch:** 2.1.0
- **Ubuntu:** 22.04

### Alternative Optionen

1. **Docker Hub:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
2. **NVIDIA:** `nvidia/cuda:12.1-devel-ubuntu22.04`
3. **Community Template:** "PyTorch 2.1 + CUDA 12.1" (von Runpod)

## Schritt-für-Schritt Setup in Runpod

### 1. Instance Erstellen
1. **Login:** runpod.io
2. **Deploy:** Secure Cloud → GPU
3. **Template auswählen:** "PyTorch 2.1 CUDA 12.1" oder Custom Image

### 2. Konfiguration
```yaml
GPU: RTX 4090 (24GB VRAM) - $0.75/hour
CPU: 8 vCPUs
RAM: 32GB
Storage: 150GB NVMe
Image: runpod/pytorch:2.1.0-py3.10-cuda12.1-devel
Ports: 22, 8888, 6006
```

### 3. Environment Variables (optional)
```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

## Verifizierung nach Start

Nach dem Deploy:
```bash
# SSH Verbindung
./scripts/ssh_connect.sh connect <INSTANCE_IP>

# System Check
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Troubleshooting Base Images

### Falls Standard Image nicht verfügbar:
1. **Fallback:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel`
2. **Minimal:** `nvidia/cuda:12.1-devel-ubuntu22.04` + manuelles Setup
3. **Community:** Suche nach "PyTorch CUDA 12" Templates

### Spezifikationen Checkliste
- [ ] CUDA 12.1+
- [ ] Python 3.9-3.11
- [ ] PyTorch 2.1.0+
- [ ] Ubuntu 20.04/22.04
- [ ] SSH Zugang aktiviert
- [ ] 150GB+ Storage
- [ ] 24GB+ VRAM (RTX 4090/A100)
