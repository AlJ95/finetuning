# DVC Grundlagen für TTS Fine-tuning

## 1. Installation
```bash
pip install dvc[s3]  # Für AWS S3 Support
```

## 2. Projekt Setup
```bash
dvc init
git commit -m "Initialize DVC"
```

## 3. Pipeline ausführen
```bash
dvc repro  # Führt alle Stages aus
dvc repro training  # Nur Training Stage
```

## 4. Daten versionieren
```bash
dvc add data/raw/dataset.zip
git add data/raw/dataset.zip.dvc
```

## 5. Remote Storage
```bash
dvc remote add -d myremote s3://mybucket/dvc
dvc push  # Daten hochladen
dvc pull  # Daten herunterladen
```

## 6. Experimente
```bash
# Parameter ändern (params.yaml)
dvc repro
dvc exp show  # Ergebnisse vergleichen
```

## 7. Hilfreiche Kommandos
```bash
dvc status  # Änderungen anzeigen
dvc dag  # Pipeline visualisieren
dvc metrics diff  # Metriken vergleichen
