# 🛠️ Dataset Download Fixes & Korrekturen

Diese Datei dokumentiert die behobenen Probleme beim Dataset-Download und enthält die korrekten Dataset-Namen.

## ✅ Behobene Probleme

### 1. Thorsten Voice Dataset
**Problem:** Dataset 'thorsten-voice/tcv2-2022.10' existiert nicht
**Lösung:** Korrekter Dataset-Name ist `thorsten/tcv2`

### 2. Test Dataset Directory
**Problem:** `FileNotFoundError: [Errno 2] No such file or directory`
**Lösung:** Verzeichnis wird vorher erstellt mit `mkdir(parents=True, exist_ok=True)`

### 3. Encoding Issues
**Problem:** Unicode Zeichen in Test-Daten
**Lösung:** UTF-8 Encoding explizit gesetzt

## 📋 Korrekte Dataset-Namen

| Dataset Name | Korrekter Hugging Face Name | Status |
|-------------|----------------------------|---------|
| Thorsten Voice | `thorsten/tcv2` | ✅ Fixiert |
| MLS German | `facebook/multilingual_librispeech` | ✅ Funktioniert |
| Test Dataset | Lokal erstellt | ✅ Fixiert |

## 🎯 Verifizierte Download-Befehle

### Thorsten Voice (korrigiert)
```bash
# Korrekter Download
python scripts/download_datasets.py --datasets thorsten --thorsten-size 1000

# Manuelle Überprüfung
python -c "from datasets import load_dataset; ds = load_dataset('thorsten/tcv2', split='train[:10]'); print(len(ds))"
```

### MLS German
```bash
# Standard Download
python scripts/download_datasets.py --datasets mls --mls-size 10000

# Manuelle Überprüfung
python -c "from datasets import load_dataset; ds = load_dataset('facebook/multilingual_librispeech', 'german', split='train[:10]'); print(len(ds))"
```

### Alle Datensätze
```bash
# Kompletter Download
python scripts/download_datasets.py --datasets all --thorsten-size 5000 --mls-size 10000
```

## 🔍 Testbefehle zur Verifizierung

```bash
# Teste Dataset-Existenz
python -c "
from datasets import list_datasets
datasets = list_datasets()
print('thorsten/tcv2' in datasets)  # Sollte True sein
print('facebook/multilingual_librispeech' in datasets)  # Sollte True sein
"

# Teste Download
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('thorsten/tcv2', split='train[:5]')
    print(f'Thorsten Voice: {len(ds)} samples loaded successfully')
except Exception as e:
    print(f'Error: {e}')
"
```

## 📊 Dataset-Größen

| Dataset | Samples | Größe | Download-Zeit |
|---------|---------|--------|---------------|
| Thorsten Voice (full) | ~23k | ~12GB | ~10-15 min |
| Thorsten Voice (1k) | 1,000 | ~500MB | ~2-3 min |
| MLS German (10k) | 10,000 | ~8GB | ~5-8 min |
| Test Dataset | 4 | <1MB | <1 sec |

## 🚀 Schnellstart nach Fixes

```bash
# 1. Dataset Download
python scripts/download_datasets.py --datasets all --thorsten-size 1000 --mls-size 5000

# 2. Verifizierung
python scripts/download_datasets.py --verify-only

# 3. Training starten
python scripts/train_remote.py --dataset thorsten --epochs 3
```

## 📝 Troubleshooting

### Falls Thorsten Voice nicht funktioniert
```bash
# Alternative Datasets
python -c "from datasets import list_datasets; [print(d) for d in list_datasets() if 'thorsten' in d.lower()]"
```

### Falls MLS German nicht funktioniert
```bash
# Teste German Subset
python -c "from datasets import load_dataset; ds = load_dataset('facebook/multilingual_librispeech', 'german', split='train[:1]'); print('OK')"
```

## ✅ Status nach Fixes

- ✅ Thorsten Voice Dataset: Korrekter Name `thorsten/tcv2`
- ✅ MLS German Dataset: Funktioniert mit `facebook/multilingual_librispeech`
- ✅ Test Dataset: Directory wird korrekt erstellt
- ✅ Encoding: UTF-8 für deutsche Zeichen
- ✅ Error Handling: Besseres Error Handling und Fallbacks
