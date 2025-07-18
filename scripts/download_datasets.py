#!/usr/bin/env python3
"""
Deutsche TTS Datensätze für Remote Training herunterladen
Unterstützt Thorsten Voice und MLS German
"""

import os
import sys
from pathlib import Path
from datasets import load_dataset, Audio
import argparse
import logging

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self, base_path="/workspace/data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def download_thorsten_voice(self, subset_size=None):
        """Thorsten Voice Dataset herunterladen"""
        logger.info("Lade Thorsten Voice Dataset...")
        
        try:
            if subset_size:
                dataset = load_dataset(
                    "thorsten-voice/tcv2-2022.10", 
                    split=f"train[:{subset_size}]"
                )
            else:
                dataset = load_dataset("thorsten-voice/tcv2-2022.10", split="train")
            
            # Speichere Dataset lokal
            output_path = self.base_path / "thorsten_voice"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"Thorsten Voice: {len(dataset)} Samples gespeichert")
            return len(dataset)
            
        except Exception as e:
            logger.error(f"Fehler beim Download von Thorsten Voice: {e}")
            return 0
    
    def download_mls_german(self, subset_size=10000):
        """MLS German Subset herunterladen"""
        logger.info("Lade MLS German Dataset...")
        
        try:
            dataset = load_dataset(
                "facebook/multilingual_librispeech", 
                "german", 
                split=f"train[:{subset_size}]"
            )
            
            # Konvertiere zu korrekter Sampling Rate für Orpheus
            dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
            
            # Speichere Dataset lokal
            output_path = self.base_path / "mls_german_subset"
            dataset.save_to_disk(str(output_path))
            
            logger.info(f"MLS German: {len(dataset)} Samples gespeichert")
            return len(dataset)
            
        except Exception as e:
            logger.error(f"Fehler beim Download von MLS German: {e}")
            return 0
    
    def download_test_dataset(self):
        """Kleines Test-Dataset für schnelle Tests"""
        logger.info("Erstelle Test-Dataset...")
        
        # Erstelle ein kleines synthetisches Dataset für Tests
        test_data = [
            {"audio": [0.1, 0.2, 0.3], "text": "Hallo Welt"},
            {"audio": [0.2, 0.3, 0.4], "text": "Guten Tag"},
        ]
        
        output_path = self.base_path / "test_dataset"
        # Speichere als JSON für einfache Verarbeitung
        
        import json
        with open(output_path / "test_data.json", "w") as f:
            json.dump(test_data, f)
        
        logger.info("Test-Dataset erstellt")
        return len(test_data)
    
    def verify_datasets(self):
        """Überprüfe heruntergeladene Datensätze"""
        datasets = ["thorsten_voice", "mls_german_subset", "test_dataset"]
        
        for dataset_name in datasets:
            path = self.base_path / dataset_name
            if path.exists():
                logger.info(f"✅ {dataset_name}: {len(list(path.rglob('*')))} Dateien")
            else:
                logger.warning(f"❌ {dataset_name}: Nicht gefunden")

def main():
    parser = argparse.ArgumentParser(description="Deutsche TTS Datensätze herunterladen")
    parser.add_argument("--datasets", nargs="+", default=["thorsten", "mls", "test"],
                        choices=["thorsten", "mls", "test", "all"],
                        help="Welche Datensätze herunterladen")
    parser.add_argument("--thorsten-size", type=int, help="Subset-Größe für Thorsten")
    parser.add_argument("--mls-size", type=int, default=10000, help="Subset-Größe für MLS")
    parser.add_argument("--base-path", default="/workspace/data", help="Basis-Pfad für Daten")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.base_path)
    
    if "all" in args.datasets or "thorsten" in args.datasets:
        downloader.download_thorsten_voice(args.thorsten_size)
    
    if "all" in args.datasets or "mls" in args.datasets:
        downloader.download_mls_german(args.mls_size)
    
    if "all" in args.datasets or "test" in args.datasets:
        downloader.download_test_dataset()
    
    # Überprüfe Downloads
    downloader.verify_datasets()

if __name__ == "__main__":
    main()
