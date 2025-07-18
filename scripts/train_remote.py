#!/usr/bin/env python3
"""
Remote TTS Training Script für Runpod.io
Trainiert Orpheus 3B mit deutschen Datensätzen
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from datetime import datetime

# Füge src zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent))

from src.unsloth_trainer import UnslothTrainer
from src.torsten_voice_processor import TorstenVoiceProcessor
from src.mls_german_processor import MLSGermanProcessor
from src.model_persistence import ModelPersistence

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RemoteTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Logging
        self.setup_logging()
        
    def setup_logging(self):
        """Erweitertes Logging für Remote Training"""
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        logger.addHandler(file_handler)
        logger.info(f"Training gestartet - Logfile: {log_file}")
    
    def prepare_dataset(self):
        """Datensatz basierend auf Konfiguration vorbereiten"""
        dataset_type = self.config.dataset
        
        if dataset_type == "thorsten":
            processor = TorstenVoiceProcessor(
                data_path=self.config.data_path,
                max_samples=self.config.max_samples
            )
        elif dataset_type == "mls_german":
            processor = MLSGermanProcessor(
                data_path=self.config.data_path,
                max_samples=self.config.max_samples
            )
        else:
            raise ValueError(f"Unbekannter Datensatz: {dataset_type}")
        
        logger.info(f"Verarbeite {dataset_type} Datensatz...")
        dataset = processor.process()
        
        logger.info(f"Dataset Größe: {len(dataset)} Samples")
        return dataset
    
    def train_model(self, dataset):
        """Modell training mit Monitoring"""
        logger.info("Starte Training...")
        
        trainer = UnslothTrainer(
            model_name=self.config.model_name,
            dataset=dataset,
            output_dir=str(self.output_dir / "checkpoints"),
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_epochs=self.config.epochs,
            max_seq_length=self.config.max_seq_length
        )
        
        # Training starten
        training_results = trainer.train()
        
        logger.info("Training abgeschlossen")
        return training_results
    
    def save_and_export(self, training_results):
        """Modell speichern und exportieren"""
        logger.info("Speichere Modell...")
        
        # Lokales Speichern
        final_model_path = self.output_dir / "final_model"
        persistence = ModelPersistence()
        
        persistence.save_model(
            model=training_results.model,
            tokenizer=training_results.tokenizer,
            save_path=str(final_model_path),
            save_method="merged_16bit"  # Für VLLM Kompatibilität
        )
        
        # Optional: Zu Hugging Face exportieren
        if self.config.push_to_hub:
            logger.info("Exportiere zu Hugging Face...")
            persistence.push_to_hub(
                model_path=str(final_model_path),
                repo_name=self.config.hub_repo,
                private=True
            )
        
        logger.info(f"Modell gespeichert unter: {final_model_path}")
        return final_model_path

def main():
    parser = argparse.ArgumentParser(description="Remote TTS Training für Runpod.io")
    
    # Dataset Konfiguration
    parser.add_argument("--dataset", choices=["thorsten", "mls_german"], default="thorsten",
                        help="Welcher Datensatz verwendet werden soll")
    parser.add_argument("--data-path", default="/workspace/data",
                        help="Pfad zu den Datensätzen")
    parser.add_argument("--max-samples", type=int, help="Maximale Anzahl Samples")
    
    # Modell Konfiguration
    parser.add_argument("--model-name", default="unsloth/orpheus-3b-0.1-ft",
                        help="Basis-Modell für Training")
    
    # Training Parameter
    parser.add_argument("--epochs", type=int, default=3, help="Anzahl Trainingsepochen")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch-Größe")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Lernrate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximale Sequenzlänge")
    
    # Output Konfiguration
    parser.add_argument("--output-dir", default="/workspace/models",
                        help="Ausgabeverzeichnis für Modelle")
    
    # Hugging Face Export
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Modell zu Hugging Face exportieren")
    parser.add_argument("--hub-repo", help="Hugging Face Repository Name")
    
    args = parser.parse_args()
    
    # Training starten
    trainer = RemoteTrainer(args)
    
    try:
        # Dataset vorbereiten
        dataset = trainer.prepare_dataset()
        
        # Training durchführen
        results = trainer.train_model(dataset)
        
        # Modell speichern
        final_path = trainer.save_and_export(results)
        
        logger.info(f"Training erfolgreich abgeschlossen! Modell: {final_path}")
        
    except Exception as e:
        logger.error(f"Training fehlgeschlagen: {e}")
        raise

if __name__ == "__main__":
    main()
