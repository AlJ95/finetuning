#!/usr/bin/env python3
"""
Data Exploration Script for German TTS Datasets
Analyzes Thorsten Voice Dataset and MLS German Dataset
"""

import os
import zipfile
import tarfile
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import shutil
from collections import defaultdict
import librosa
import soundfile as sf
import numpy as np

class DatasetExplorer:
    """Explores and analyzes TTS datasets for German language training"""
    
    def __init__(self, data_dir: str = "D:/Trainingsdaten/TTS"):
        self.data_dir = Path(data_dir)
        self.thorsten_zip = self.data_dir / "ThorstenVoice-Dataset_2022.10.zip"
        self.mls_tar = self.data_dir / "mls_german_opus.tar.gz"
        
    def analyze_thorsten_dataset(self) -> Dict[str, Any]:
        """Analyze Thorsten Voice Dataset (LJSpeech format)"""
        print("=== Analyzing Thorsten Voice Dataset ===")
        
        if not self.thorsten_zip.exists():
            return {"error": f"Dataset not found at {self.thorsten_zip}"}
        
        analysis = {
            "dataset_name": "Thorsten Voice Dataset 2022.10",
            "format": "LJSpeech",
            "file_size_mb": self.thorsten_zip.stat().st_size / (1024 * 1024),
            "structure": {},
            "audio_stats": {},
            "text_stats": {},
            "quality_assessment": {},
            "unsloth_compatibility": {}
        }
        
        # Extract and analyze structure
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(self.thorsten_zip, 'r') as zip_ref:
                # Get file list without extracting everything
                file_list = zip_ref.namelist()
                
                # Analyze structure
                analysis["structure"] = self._analyze_file_structure(file_list)
                
                # Extract metadata file and sample audio files for analysis
                metadata_files = [f for f in file_list if f.endswith('.csv') or f.endswith('.txt')]
                audio_files = [f for f in file_list if f.endswith('.wav')][:10]  # Sample first 10
                
                # Extract metadata
                for meta_file in metadata_files:
                    zip_ref.extract(meta_file, temp_dir)
                
                # Extract sample audio files
                for audio_file in audio_files:
                    zip_ref.extract(audio_file, temp_dir)
                
                # Analyze metadata
                analysis["text_stats"] = self._analyze_thorsten_metadata(temp_dir, metadata_files)
                
                # Analyze audio samples
                analysis["audio_stats"] = self._analyze_audio_samples(temp_dir, audio_files)
        
        # Assess quality and compatibility
        analysis["quality_assessment"] = self._assess_thorsten_quality(analysis)
        analysis["unsloth_compatibility"] = self._assess_unsloth_compatibility(analysis, "thorsten")
        
        return analysis
    
    def analyze_mls_dataset(self) -> Dict[str, Any]:
        """Analyze Multilingual LibriSpeech German Dataset"""
        print("=== Analyzing MLS German Dataset ===")
        
        if not self.mls_tar.exists():
            return {"error": f"Dataset not found at {self.mls_tar}"}
        
        analysis = {
            "dataset_name": "Multilingual LibriSpeech German",
            "format": "MLS",
            "file_size_mb": self.mls_tar.stat().st_size / (1024 * 1024),
            "structure": {},
            "audio_stats": {},
            "text_stats": {},
            "quality_assessment": {},
            "unsloth_compatibility": {}
        }
        
        # Extract and analyze structure
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(self.mls_tar, 'r:gz') as tar_ref:
                # Get file list without extracting everything
                file_list = tar_ref.getnames()
                
                # Analyze structure
                analysis["structure"] = self._analyze_file_structure(file_list)
                
                # Extract metadata and sample files
                transcript_files = [f for f in file_list if 'transcripts.txt' in f or '.trans.txt' in f]
                audio_files = [f for f in file_list if f.endswith('.opus') or f.endswith('.flac')][:10]
                
                # Extract metadata
                for trans_file in transcript_files[:3]:  # Sample first 3 transcript files
                    try:
                        tar_ref.extract(trans_file, temp_dir)
                    except:
                        continue
                
                # Extract sample audio files
                for audio_file in audio_files:
                    try:
                        tar_ref.extract(audio_file, temp_dir)
                    except:
                        continue
                
                # Analyze metadata
                analysis["text_stats"] = self._analyze_mls_metadata(temp_dir, transcript_files)
                
                # Analyze audio samples
                analysis["audio_stats"] = self._analyze_audio_samples(temp_dir, audio_files)
        
        # Assess quality and compatibility
        analysis["quality_assessment"] = self._assess_mls_quality(analysis)
        analysis["unsloth_compatibility"] = self._assess_unsloth_compatibility(analysis, "mls")
        
        return analysis
    
    def _analyze_file_structure(self, file_list: List[str]) -> Dict[str, Any]:
        """Analyze file structure of dataset"""
        structure = {
            "total_files": len(file_list),
            "file_types": defaultdict(int),
            "directory_structure": defaultdict(int),
            "audio_files": 0,
            "text_files": 0
        }
        
        for file_path in file_list:
            # Count file types
            if '.' in file_path:
                ext = file_path.split('.')[-1].lower()
                structure["file_types"][ext] += 1
                
                if ext in ['wav', 'flac', 'opus', 'mp3']:
                    structure["audio_files"] += 1
                elif ext in ['txt', 'csv', 'tsv']:
                    structure["text_files"] += 1
            
            # Count directory levels
            depth = file_path.count('/')
            structure["directory_structure"][f"depth_{depth}"] += 1
        
        return dict(structure)
    
    def _analyze_thorsten_metadata(self, temp_dir: str, metadata_files: List[str]) -> Dict[str, Any]:
        """Analyze Thorsten dataset metadata (LJSpeech format)"""
        stats = {
            "total_utterances": 0,
            "text_length_stats": {},
            "character_distribution": defaultdict(int),
            "sample_texts": []
        }
        
        text_lengths = []
        
        for meta_file in metadata_files:
            file_path = Path(temp_dir) / meta_file
            if not file_path.exists():
                continue
                
            try:
                # Try different formats
                if meta_file.endswith('.csv'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f, delimiter='|')
                        for row in reader:
                            if len(row) >= 2:
                                text = row[1] if len(row) > 1 else row[0]
                                text_lengths.append(len(text))
                                stats["total_utterances"] += 1
                                
                                # Sample first 5 texts
                                if len(stats["sample_texts"]) < 5:
                                    stats["sample_texts"].append(text[:100])
                                
                                # Character distribution
                                for char in text.lower():
                                    stats["character_distribution"][char] += 1
                
                elif meta_file.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if '|' in line:
                                parts = line.strip().split('|')
                                if len(parts) >= 2:
                                    text = parts[1]
                                    text_lengths.append(len(text))
                                    stats["total_utterances"] += 1
                                    
                                    if len(stats["sample_texts"]) < 5:
                                        stats["sample_texts"].append(text[:100])
                                    
                                    for char in text.lower():
                                        stats["character_distribution"][char] += 1
            except Exception as e:
                print(f"Error reading {meta_file}: {e}")
        
        if text_lengths:
            stats["text_length_stats"] = {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": np.mean(text_lengths),
                "median": np.median(text_lengths),
                "std": np.std(text_lengths)
            }
        
        return stats
    
    def _analyze_mls_metadata(self, temp_dir: str, transcript_files: List[str]) -> Dict[str, Any]:
        """Analyze MLS dataset metadata"""
        stats = {
            "total_utterances": 0,
            "text_length_stats": {},
            "character_distribution": defaultdict(int),
            "sample_texts": [],
            "speaker_info": defaultdict(int)
        }
        
        text_lengths = []
        
        for trans_file in transcript_files:
            file_path = Path(temp_dir) / trans_file
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if '\t' in line:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                audio_id = parts[0]
                                text = parts[1]
                                
                                # Extract speaker ID from audio_id
                                speaker_id = audio_id.split('_')[0] if '_' in audio_id else 'unknown'
                                stats["speaker_info"][speaker_id] += 1
                                
                                text_lengths.append(len(text))
                                stats["total_utterances"] += 1
                                
                                if len(stats["sample_texts"]) < 5:
                                    stats["sample_texts"].append(text[:100])
                                
                                for char in text.lower():
                                    stats["character_distribution"][char] += 1
            except Exception as e:
                print(f"Error reading {trans_file}: {e}")
        
        if text_lengths:
            stats["text_length_stats"] = {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": np.mean(text_lengths),
                "median": np.median(text_lengths),
                "std": np.std(text_lengths)
            }
        
        stats["unique_speakers"] = len(stats["speaker_info"])
        
        return stats
    
    def _analyze_audio_samples(self, temp_dir: str, audio_files: List[str]) -> Dict[str, Any]:
        """Analyze audio file samples"""
        stats = {
            "sample_count": 0,
            "duration_stats": {},
            "sample_rate_stats": {},
            "channel_stats": {},
            "format_info": defaultdict(int)
        }
        
        durations = []
        sample_rates = []
        channels = []
        
        for audio_file in audio_files:
            file_path = Path(temp_dir) / audio_file
            if not file_path.exists():
                continue
                
            try:
                # Get audio info
                info = sf.info(str(file_path))
                
                durations.append(info.duration)
                sample_rates.append(info.samplerate)
                channels.append(info.channels)
                
                stats["format_info"][info.format] += 1
                stats["sample_count"] += 1
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
        
        if durations:
            stats["duration_stats"] = {
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "mean_seconds": np.mean(durations),
                "median_seconds": np.median(durations),
                "total_hours": sum(durations) / 3600
            }
        
        if sample_rates:
            stats["sample_rate_stats"] = {
                "rates": list(set(sample_rates)),
                "most_common": max(set(sample_rates), key=sample_rates.count)
            }
        
        if channels:
            stats["channel_stats"] = {
                "mono_files": channels.count(1),
                "stereo_files": channels.count(2),
                "other": len(channels) - channels.count(1) - channels.count(2)
            }
        
        return stats
    
    def _assess_thorsten_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of Thorsten dataset for TTS training"""
        quality = {
            "overall_score": "good",
            "strengths": [],
            "concerns": [],
            "recommendations": []
        }
        
        # Check audio quality indicators
        if analysis["audio_stats"].get("sample_rate_stats", {}).get("most_common") == 22050:
            quality["strengths"].append("Standard TTS sample rate (22kHz)")
        
        if analysis["audio_stats"].get("channel_stats", {}).get("mono_files", 0) > 0:
            quality["strengths"].append("Mono audio format suitable for TTS")
        
        # Check text quality
        if analysis["text_stats"].get("total_utterances", 0) > 10000:
            quality["strengths"].append("Large number of utterances for training")
        
        # Check for potential issues
        text_stats = analysis["text_stats"].get("text_length_stats", {})
        if text_stats.get("max", 0) > 500:
            quality["concerns"].append("Some very long texts may need segmentation")
        
        if text_stats.get("min", 0) < 10:
            quality["concerns"].append("Some very short texts may not be useful")
        
        quality["recommendations"] = [
            "Filter utterances by length (10-200 characters)",
            "Validate audio-text alignment",
            "Check for consistent speaker voice"
        ]
        
        return quality
    
    def _assess_mls_quality(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of MLS dataset for TTS training"""
        quality = {
            "overall_score": "good",
            "strengths": [],
            "concerns": [],
            "recommendations": []
        }
        
        # Multi-speaker advantages
        if analysis["text_stats"].get("unique_speakers", 0) > 1:
            quality["strengths"].append(f"Multi-speaker dataset ({analysis['text_stats']['unique_speakers']} speakers)")
        
        if analysis["text_stats"].get("total_utterances", 0) > 50000:
            quality["strengths"].append("Very large dataset for robust training")
        
        # Potential concerns
        quality["concerns"].append("Multi-speaker may require speaker conditioning")
        quality["concerns"].append("OPUS format may need conversion to WAV")
        
        quality["recommendations"] = [
            "Consider single-speaker subset for initial training",
            "Convert OPUS to WAV format",
            "Implement speaker ID conditioning",
            "Balance speakers in training data"
        ]
        
        return quality
    
    def _assess_unsloth_compatibility(self, analysis: Dict[str, Any], dataset_type: str) -> Dict[str, Any]:
        """Assess compatibility with Unsloth TTS training"""
        compatibility = {
            "compatible": True,
            "required_preprocessing": [],
            "format_requirements": [],
            "recommendations": []
        }
        
        if dataset_type == "thorsten":
            compatibility["format_requirements"] = [
                "Convert to Unsloth TTS format: {'audio': audio_array, 'text': transcript}",
                "Ensure consistent sample rate (22kHz recommended)",
                "Normalize audio levels"
            ]
            
            compatibility["required_preprocessing"] = [
                "Parse LJSpeech metadata format",
                "Load and validate audio files",
                "Create train/validation splits"
            ]
            
        elif dataset_type == "mls":
            compatibility["format_requirements"] = [
                "Convert OPUS to WAV format",
                "Handle multi-speaker data structure",
                "Standardize sample rates"
            ]
            
            compatibility["required_preprocessing"] = [
                "Extract and convert audio files",
                "Parse MLS transcript format",
                "Implement speaker conditioning (optional)"
            ]
        
        compatibility["recommendations"] = [
            "Use Unsloth's memory-efficient loading",
            "Implement batch processing for large datasets",
            "Consider data augmentation for robustness"
        ]
        
        return compatibility

def main():
    """Main exploration function"""
    explorer = DatasetExplorer()
    
    print("Starting German TTS Dataset Exploration...")
    print("=" * 50)
    
    # Analyze both datasets
    thorsten_analysis = explorer.analyze_thorsten_dataset()
    mls_analysis = explorer.analyze_mls_dataset()
    
    # Save results
    results = {
        "exploration_date": "2025-01-17",
        "thorsten_voice_dataset": thorsten_analysis,
        "mls_german_dataset": mls_analysis,
        "comparison": {
            "thorsten_size_mb": thorsten_analysis.get("file_size_mb", 0),
            "mls_size_mb": mls_analysis.get("file_size_mb", 0),
            "thorsten_utterances": thorsten_analysis.get("text_stats", {}).get("total_utterances", 0),
            "mls_utterances": mls_analysis.get("text_stats", {}).get("total_utterances", 0),
            "recommendation": "Start with Thorsten Voice for single-speaker TTS, use MLS for multi-speaker experiments"
        }
    }
    
    # Save to JSON file
    with open("data_exploration_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print("\nExploration complete! Results saved to data_exploration_results.json")
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Thorsten Voice Dataset: {thorsten_analysis.get('file_size_mb', 0):.1f} MB, "
          f"{thorsten_analysis.get('text_stats', {}).get('total_utterances', 0)} utterances")
    print(f"MLS German Dataset: {mls_analysis.get('file_size_mb', 0):.1f} MB, "
          f"{mls_analysis.get('text_stats', {}).get('total_utterances', 0)} utterances")

if __name__ == "__main__":
    main()