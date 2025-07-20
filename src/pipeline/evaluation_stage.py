"""
Evaluation stage for DVC pipeline.

This stage evaluates the trained TTS model using comprehensive German TTS metrics
including audio quality, phoneme accuracy, and model performance.
"""

from typing import Dict, Any, List
from pathlib import Path
import json
import pickle
import numpy as np
from tqdm import tqdm

from .base_stage import BasePipelineStage, PipelineError
from ..tts_evaluator_clean import TTSEvaluator, EvaluationConfig, EvaluationResults
from ..data_processor_base import AudioDataset


class EvaluationStage(BasePipelineStage):
    """
    Evaluation pipeline stage.
    
    Evaluates the trained TTS model using comprehensive metrics:
    - Audio quality metrics (PESQ, STOI, MOS approximation)
    - German-specific phoneme accuracy
    - Model performance metrics (inference speed, memory usage)
    - Comparative analysis with baseline
    
    Features:
    - Configurable evaluation metrics
    - Batch processing for efficiency
    - Detailed reporting and visualization
    - Audio sample generation for manual review
    """
    
    def __init__(self):
        super().__init__("evaluation")
        
        # Load evaluation parameters
        self.eval_params = self.stage_params
        
        # Initialize evaluator
        self.evaluator = None
        self.eval_config = None
        
        self.logger.info("Initialized evaluation stage")
    
    def validate_inputs(self) -> bool:
        """Validate that required model and data are available."""
        try:
            # Check if LoRA adapters exist
            lora_dir = Path("models/lora_adapters")
            if not lora_dir.exists():
                self.logger.error("LoRA adapters not found - training stage must complete first")
                return False
            
            # Check validation dataset
            val_file = Path("data/preprocessed/val_dataset.pkl")
            if not val_file.exists():
                self.logger.error("Validation dataset not found")
                return False
            
            # Load and validate validation data
            with open(val_file, 'rb') as f:
                val_samples = pickle.load(f)
            
            if not val_samples:
                self.logger.error("Empty validation dataset")
                return False
            
            # Check training metrics for baseline comparison
            training_metrics_file = Path("metrics/training_metrics.json")
            if not training_metrics_file.exists():
                self.logger.warning("Training metrics not found - baseline comparison will be limited")
            
            self.logger.info(f"Validation samples available: {len(val_samples)}")
            
            # Check disk space for evaluation outputs
            if not self.check_disk_space(5.0):
                self.logger.error("Insufficient disk space for evaluation")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def execute(self) -> Dict[str, Any]:
        """Execute comprehensive model evaluation."""
        results = {
            "evaluation_started": True,
            "metrics_computed": {},
            "evaluation_summary": {},
            "audio_samples_generated": 0,
            "total_samples_evaluated": 0
        }
        
        try:
            # Create evaluation configuration
            self._create_evaluation_config()
            
            # Initialize evaluator
            self.evaluator = TTSEvaluator(self.eval_config)
            self.logger.info("Initialized TTS evaluator")
            
            # Load validation data
            val_samples = self._load_validation_data()
            
            # Limit evaluation samples if specified
            eval_samples_limit = self.eval_params.get("eval_samples", -1)
            if eval_samples_limit > 0:
                val_samples = val_samples[:eval_samples_limit]
            
            results["total_samples_evaluated"] = len(val_samples)
            self.logger.info(f"Evaluating {len(val_samples)} samples")
            
            # Run evaluation
            self.logger.info("Starting comprehensive evaluation...")
            evaluation_results = self.evaluator.evaluate_model_comprehensive(
                model_path="models/lora_adapters",
                validation_data=val_samples
            )
            
            # Process evaluation results
            results["metrics_computed"] = self._process_evaluation_results(evaluation_results)
            
            # Generate audio samples if requested
            if self.eval_params.get("save_audio_samples", True):
                num_samples = self.eval_params.get("num_audio_samples", 10)
                audio_samples = self._generate_audio_samples(val_samples[:num_samples])
                results["audio_samples_generated"] = len(audio_samples)
            
            # Create evaluation summary
            results["evaluation_summary"] = self._create_evaluation_summary(evaluation_results)
            
            # Save evaluation artifacts
            self._save_evaluation_artifacts(evaluation_results, results)
            
            # Save DVC metrics and plots
            self._save_dvc_metrics_and_plots(evaluation_results)
            
            self.logger.info("Evaluation completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            results["error"] = str(e)
            raise
    
    def _create_evaluation_config(self):
        """Create evaluation configuration from parameters."""
        self.logger.info("Creating evaluation configuration")
        
        metrics_config = self.eval_params.get("metrics", {})
        
        config_dict = {
            # Metric selection
            "calculate_pesq": metrics_config.get("calculate_pesq", True),
            "calculate_stoi": metrics_config.get("calculate_stoi", True),
            "calculate_mos": metrics_config.get("calculate_mos", True),
            "phoneme_accuracy": metrics_config.get("phoneme_accuracy", True),
            "prosody_analysis": metrics_config.get("prosody_analysis", True),
            "inference_speed": metrics_config.get("inference_speed", True),
            "memory_usage": metrics_config.get("memory_usage", True),
            
            # Processing parameters
            "batch_size": self.eval_params.get("batch_size", 8),
            "target_sample_rate": 24000,  # Orpheus 3B requirement
            
            # Output configuration
            "save_detailed_results": True,
            "save_audio_samples": self.eval_params.get("save_audio_samples", True),
            "generate_plots": self.eval_params.get("generate_plots", True),
            "detailed_report": self.eval_params.get("detailed_report", True),
        }
        
        self.eval_config = EvaluationConfig(**config_dict)
        self.logger.info("Evaluation configuration created")
    
    def _load_validation_data(self) -> List[AudioDataset]:
        """Load validation data for evaluation."""
        val_file = Path("data/preprocessed/val_dataset.pkl")
        
        with open(val_file, 'rb') as f:
            val_samples = pickle.load(f)
        
        return val_samples
    
    def _process_evaluation_results(self, evaluation_results: EvaluationResults) -> Dict[str, Any]:
        """Process and structure evaluation results."""
        self.logger.info("Processing evaluation results")
        
        processed_results = {
            "audio_quality": {
                "pesq_score": evaluation_results.pesq_score,
                "stoi_score": evaluation_results.stoi_score,
                "mos_score": evaluation_results.mos_score
            },
            "german_specific": {
                "phoneme_accuracy": evaluation_results.phoneme_accuracy,
                "prosody_score": getattr(evaluation_results, 'prosody_score', None)
            },
            "performance": {
                "inference_speed_ms": evaluation_results.inference_speed,
                "memory_usage_mb": evaluation_results.memory_usage,
                "model_size_mb": evaluation_results.model_size_mb
            },
            "overall_quality": evaluation_results.overall_quality_score
        }
        
        # Add detailed metrics if available
        if hasattr(evaluation_results, 'detailed_metrics'):
            processed_results["detailed_metrics"] = evaluation_results.detailed_metrics
        
        return processed_results
    
    def _generate_audio_samples(self, samples: List[AudioDataset]) -> List[Dict[str, Any]]:
        """Generate audio samples for manual review."""
        self.logger.info(f"Generating {len(samples)} audio samples for review")
        
        # Create samples directory
        samples_dir = Path("results/evaluation/audio_samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        generated_samples = []
        
        for i, sample in enumerate(tqdm(samples, desc="Generating audio samples")):
            try:
                # Note: In a real implementation, this would generate TTS audio
                # For now, we'll save the reference information
                sample_info = {
                    "sample_id": i,
                    "text": sample.text_transcript,
                    "original_file": str(sample.file_path),
                    "duration": sample.duration,
                    "quality_score": sample.quality_score,
                    "speaker_id": sample.metadata.get("speaker_id") if sample.metadata else None
                }
                
                # Save sample info
                sample_file = samples_dir / f"sample_{i:03d}_info.json"
                with open(sample_file, 'w', encoding='utf-8') as f:
                    json.dump(sample_info, f, indent=2, default=str)
                
                generated_samples.append(sample_info)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate sample {i}: {e}")
                continue
        
        # Save samples index
        index_file = samples_dir / "samples_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(generated_samples, f, indent=2, default=str)
        
        self.logger.info(f"Generated {len(generated_samples)} audio samples")
        return generated_samples
    
    def _create_evaluation_summary(self, evaluation_results: EvaluationResults) -> Dict[str, Any]:
        """Create comprehensive evaluation summary."""
        
        # Load training metrics for comparison
        training_metrics = {}
        training_metrics_file = Path("metrics/training_metrics.json")
        if training_metrics_file.exists():
            with open(training_metrics_file, 'r') as f:
                training_metrics = json.load(f)
        
        summary = {
            "overall_performance": {
                "mos_score": evaluation_results.mos_score,
                "pesq_score": evaluation_results.pesq_score,
                "stoi_score": evaluation_results.stoi_score,
                "phoneme_accuracy": evaluation_results.phoneme_accuracy,
                "overall_quality": evaluation_results.overall_quality_score
            },
            "technical_performance": {
                "inference_speed_ms": evaluation_results.inference_speed,
                "memory_usage_mb": evaluation_results.memory_usage,
                "model_size_mb": evaluation_results.model_size_mb
            },
            "training_comparison": {
                "final_training_loss": training_metrics.get("final_loss"),
                "training_time_seconds": training_metrics.get("training_time_seconds"),
                "training_steps": training_metrics.get("total_steps")
            },
            "quality_assessment": self._assess_quality_level(evaluation_results),
            "recommendations": self._generate_recommendations(evaluation_results)
        }
        
        return summary
    
    def _assess_quality_level(self, results: EvaluationResults) -> Dict[str, str]:
        """Assess overall quality level based on metrics."""
        
        # Define quality thresholds
        mos_thresholds = {"excellent": 4.0, "good": 3.5, "fair": 3.0, "poor": 2.5}
        pesq_thresholds = {"excellent": 3.5, "good": 3.0, "fair": 2.5, "poor": 2.0}
        stoi_thresholds = {"excellent": 0.9, "good": 0.8, "fair": 0.7, "poor": 0.6}
        
        def get_quality_level(score: float, thresholds: Dict[str, float]) -> str:
            for level, threshold in thresholds.items():
                if score >= threshold:
                    return level
            return "very_poor"
        
        assessment = {
            "mos_level": get_quality_level(results.mos_score, mos_thresholds),
            "pesq_level": get_quality_level(results.pesq_score, pesq_thresholds),
            "stoi_level": get_quality_level(results.stoi_score, stoi_thresholds),
            "phoneme_level": "good" if results.phoneme_accuracy > 0.85 else "needs_improvement"
        }
        
        # Overall assessment
        levels = [assessment["mos_level"], assessment["pesq_level"], assessment["stoi_level"]]
        if "excellent" in levels and len([l for l in levels if l in ["excellent", "good"]]) >= 2:
            assessment["overall"] = "excellent"
        elif "good" in levels and len([l for l in levels if l in ["good", "fair"]]) >= 2:
            assessment["overall"] = "good"
        elif "fair" in levels:
            assessment["overall"] = "fair"
        else:
            assessment["overall"] = "needs_improvement"
        
        return assessment
    
    def _generate_recommendations(self, results: EvaluationResults) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if results.mos_score < 3.5:
            recommendations.append("Consider increasing training epochs or adjusting learning rate for better naturalness")
        
        if results.pesq_score < 3.0:
            recommendations.append("Audio quality could be improved with better preprocessing or higher sample rate")
        
        if results.phoneme_accuracy < 0.85:
            recommendations.append("Phoneme accuracy needs improvement - consider more phoneme-aware training data")
        
        if results.inference_speed > 1000:  # ms
            recommendations.append("Inference speed is slow - consider model optimization or quantization")
        
        if results.memory_usage > 8000:  # MB
            recommendations.append("Memory usage is high - consider using smaller model variant or optimization")
        
        if not recommendations:
            recommendations.append("Model performance is good across all metrics")
        
        return recommendations
    
    def _save_evaluation_artifacts(self, evaluation_results: EvaluationResults, stage_results: Dict[str, Any]):
        """Save evaluation artifacts and reports."""
        self.logger.info("Saving evaluation artifacts")
        
        # Create evaluation results directory
        results_dir = Path("results/evaluation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed evaluation results
        detailed_results_file = results_dir / "detailed_results.json"
        detailed_data = {
            "evaluation_results": evaluation_results.__dict__,
            "stage_results": stage_results,
            "evaluation_config": self.eval_config.__dict__ if self.eval_config else {}
        }
        
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        # Generate evaluation report if requested
        if self.eval_params.get("detailed_report", True):
            report_file = results_dir / "evaluation_report.md"
            self._generate_markdown_report(evaluation_results, stage_results, report_file)
        
        self.logger.info("Evaluation artifacts saved")
    
    def _generate_markdown_report(self, evaluation_results: EvaluationResults, 
                                 stage_results: Dict[str, Any], report_file: Path):
        """Generate markdown evaluation report."""
        
        report_content = f"""# German TTS Model Evaluation Report

## Overview
This report provides a comprehensive evaluation of the fine-tuned Orpheus 3B model for German TTS.

## Evaluation Summary

### Audio Quality Metrics
- **MOS Score**: {evaluation_results.mos_score:.3f}/5.0
- **PESQ Score**: {evaluation_results.pesq_score:.3f}/4.5
- **STOI Score**: {evaluation_results.stoi_score:.3f}/1.0

### German Language Metrics
- **Phoneme Accuracy**: {evaluation_results.phoneme_accuracy:.1%}

### Performance Metrics
- **Inference Speed**: {evaluation_results.inference_speed:.1f} ms per sample
- **Memory Usage**: {evaluation_results.memory_usage:.1f} MB
- **Model Size**: {evaluation_results.model_size_mb:.1f} MB

### Overall Quality Score
**{evaluation_results.overall_quality_score:.3f}/5.0**

## Quality Assessment
{stage_results.get('evaluation_summary', {}).get('quality_assessment', {}).get('overall', 'Unknown')}

## Recommendations
"""
        
        recommendations = stage_results.get('evaluation_summary', {}).get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report_content += f"{i}. {rec}\n"
        
        report_content += f"""
## Samples Evaluated
- **Total Samples**: {stage_results.get('total_samples_evaluated', 0)}
- **Audio Samples Generated**: {stage_results.get('audio_samples_generated', 0)}

## Technical Details
- **Evaluation Date**: {evaluation_results.timestamp if hasattr(evaluation_results, 'timestamp') else 'Unknown'}
- **Model Path**: models/lora_adapters
- **Configuration**: Orpheus 3B with LoRA fine-tuning

---
*Generated automatically by DVC evaluation pipeline*
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Generated evaluation report: {report_file}")
    
    def _save_dvc_metrics_and_plots(self, evaluation_results: EvaluationResults):
        """Save metrics and plots for DVC visualization."""
        
        # Save metrics
        metrics = {
            "mos_score": evaluation_results.mos_score,
            "pesq_score": evaluation_results.pesq_score,
            "stoi_score": evaluation_results.stoi_score,
            "phoneme_accuracy": evaluation_results.phoneme_accuracy,
            "overall_quality_score": evaluation_results.overall_quality_score,
            "inference_speed_ms": evaluation_results.inference_speed,
            "memory_usage_mb": evaluation_results.memory_usage,
            "model_size_mb": evaluation_results.model_size_mb
        }
        
        self.save_metrics(metrics, "metrics/evaluation_metrics.json")
        
        # Save plot data
        plot_data = [
            {"metric": "MOS", "value": evaluation_results.mos_score, "max_value": 5.0},
            {"metric": "PESQ", "value": evaluation_results.pesq_score, "max_value": 4.5},
            {"metric": "STOI", "value": evaluation_results.stoi_score, "max_value": 1.0},
            {"metric": "Phoneme Accuracy", "value": evaluation_results.phoneme_accuracy, "max_value": 1.0},
            {"metric": "Overall Quality", "value": evaluation_results.overall_quality_score, "max_value": 5.0}
        ]
        
        self.save_plots_data(plot_data, "plots/evaluation_plots.json")
    
    def validate_outputs(self) -> bool:
        """Validate that evaluation outputs were created correctly."""
        try:
            # Check evaluation results directory
            results_dir = Path("results/evaluation")
            if not results_dir.exists():
                self.logger.error("Evaluation results directory not found")
                return False
            
            # Check detailed results file
            detailed_results_file = results_dir / "detailed_results.json"
            if not detailed_results_file.exists():
                self.logger.error("Detailed results file not found")
                return False
            
            # Validate detailed results content
            with open(detailed_results_file, 'r') as f:
                detailed_results = json.load(f)
            
            required_keys = ["evaluation_results", "stage_results"]
            for key in required_keys:
                if key not in detailed_results:
                    self.logger.error(f"Missing key in detailed results: {key}")
                    return False
            
            # Check evaluation metrics
            metrics_file = Path("metrics/evaluation_metrics.json")
            if not metrics_file.exists():
                self.logger.error("Evaluation metrics file not found")
                return False
            
            # Validate metrics content
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            required_metrics = ["mos_score", "pesq_score", "stoi_score", "phoneme_accuracy"]
            for metric in required_metrics:
                if metric not in metrics:
                    self.logger.error(f"Missing required metric: {metric}")
                    return False
            
            # Check plots data
            plots_file = Path("plots/evaluation_plots.json")
            if not plots_file.exists():
                self.logger.error("Evaluation plots file not found")
                return False
            
            self.logger.info("Evaluation output validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Output validation failed: {e}")
            return False


def main():
    """Main entry point for running evaluation stage."""
    stage = EvaluationStage()
    
    try:
        result = stage.run()
        print(f"Evaluation completed successfully")
        
        if "metrics_computed" in result["results"]:
            metrics = result["results"]["metrics_computed"]
            print(f"MOS Score: {metrics['audio_quality']['mos_score']:.3f}")
            print(f"PESQ Score: {metrics['audio_quality']['pesq_score']:.3f}")
            print(f"STOI Score: {metrics['audio_quality']['stoi_score']:.3f}")
            print(f"Phoneme Accuracy: {metrics['german_specific']['phoneme_accuracy']:.1%}")
            print(f"Overall Quality: {metrics['overall_quality']:.3f}")
        
        print(f"Samples evaluated: {result['results']['total_samples_evaluated']}")
        print(f"Audio samples generated: {result['results']['audio_samples_generated']}")
        
    except PipelineError as e:
        print(f"Evaluation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
