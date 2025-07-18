#!/usr/bin/env python3
"""
Monitoring und Debugging Script für Remote TTS Training
Überwacht GPU-Auslastung, Training-Fortschritt und System-Status
"""

import os
import sys
import time
import subprocess
import json
import logging
from datetime import datetime
from pathlib import Path

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, log_dir="/workspace/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.log_dir / "training_metrics.json"
        self.system_log = self.log_dir / "system_monitor.log"
        
    def get_gpu_status(self):
        """GPU-Status abrufen"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpus.append({
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_used": int(parts[2]),
                        "memory_total": int(parts[3]),
                        "utilization": int(parts[4]),
                        "temperature": int(parts[5])
                    })
            return gpus
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des GPU-Status: {e}")
            return []
    
    def get_disk_usage(self):
        """Festplatten-Auslastung abrufen"""
        try:
            result = subprocess.run(
                ["df", "-h", "/workspace"],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                return {
                    "total": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "usage_percent": parts[4]
                }
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Festplatten-Auslastung: {e}")
            return {}
    
    def get_system_load(self):
        """System-Last abrufen"""
        try:
            with open('/proc/loadavg', 'r') as f:
                load_data = f.read().split()
                return {
                    "load_1min": float(load_data[0]),
                    "load_5min": float(load_data[1]),
                    "load_15min": float(load_data[2]),
                    "processes": load_data[3]
                }
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der System-Last: {e}")
            return {}
    
    def monitor_training(self, training_dir="/workspace/models"):
        """Überwache Training-Fortschritt"""
        training_path = Path(training_dir)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "gpu_status": self.get_gpu_status(),
            "disk_usage": self.get_disk_usage(),
            "system_load": self.get_system_load(),
            "training_files": []
        }
        
        # Prüfe auf neue Modelldateien
        if training_path.exists():
            model_files = list(training_path.rglob("*.bin"))
            checkpoint_files = list(training_path.rglob("checkpoint-*"))
            
            metrics["training_files"] = {
                "model_files": len(model_files),
                "checkpoints": len(checkpoint_files),
                "latest_checkpoint": str(max(checkpoint_files, key=lambda x: x.stat().st_mtime)) if checkpoint_files else None
            }
        
        # Speichere Metriken
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        return metrics
    
    def print_status(self, metrics):
        """Status auf Konsole ausgeben"""
        print("\n" + "="*60)
        print(f"Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # GPU Status
        for gpu in metrics["gpu_status"]:
            print(f"GPU {gpu['index']} ({gpu['name']}):")
            print(f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
            print(f"  Utilization: {gpu['utilization']}%")
            print(f"  Temperature: {gpu['temperature']}°C")
        
        # Disk Usage
        disk = metrics["disk_usage"]
        if disk:
            print(f"\nDisk Usage: {disk['used']} / {disk['total']} ({disk['usage_percent']})")
        
        # System Load
        load = metrics["system_load"]
        if load:
            print(f"System Load: {load['load_1min']:.2f}, {load['load_5min']:.2f}, {load['load_15min']:.2f}")
        
        # Training Progress
        files = metrics["training_files"]
        if files:
            print(f"\nTraining Files: {files['model_files']} model files, {files['checkpoints']} checkpoints")
            if files["latest_checkpoint"]:
                print(f"Latest Checkpoint: {files['latest_checkpoint']}")
    
    def start_tensorboard(self, log_dir="/workspace/logs"):
        """TensorBoard starten"""
        try:
            cmd = [
                "tensorboard",
                "--logdir", log_dir,
                "--host", "0.0.0.0",
                "--port", "6006",
                "--reload_interval", "30"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"TensorBoard gestartet auf Port 6006")
            return process
        except Exception as e:
            logger.error(f"Fehler beim Starten von TensorBoard: {e}")
            return None
    
    def continuous_monitoring(self, interval=30):
        """Kontinuierliche Überwachung"""
        logger.info(f"Starte kontinuierliche Überwachung (Intervall: {interval}s)")
        
        while True:
            try:
                metrics = self.monitor_training()
                self.print_status(metrics)
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("Monitoring gestoppt")
                break
            except Exception as e:
                logger.error(f"Fehler in der Überwachung: {e}")
                time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description="Training Monitor für Remote TTS")
    parser.add_argument("--interval", type=int, default=30, help="Überwachungsintervall in Sekunden")
    parser.add_argument("--log-dir", default="/workspace/logs", help="Log-Verzeichnis")
    parser.add_argument("--training-dir", default="/workspace/models", help="Training-Verzeichnis")
    parser.add_argument("--tensorboard", action="store_true", help="TensorBoard starten")
    parser.add_argument("--once", action="store_true", help="Nur einmal ausführen")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_dir)
    
    if args.tensorboard:
        monitor.start_tensorboard(args.log_dir)
    
    if args.once:
        metrics = monitor.monitor_training(args.training_dir)
        monitor.print_status(metrics)
    else:
        monitor.continuous_monitoring(args.interval)

if __name__ == "__main__":
    main()
