"""
DVC Pipeline Runner for German TTS Fine-tuning.

This script provides a convenient interface to run the complete DVC pipeline
or individual stages with proper error handling and reporting.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from .base_stage import PipelineError
from .data_loading_stage import DataLoadingStage
from .preprocessing_stage import PreprocessingStage
from .training_stage import TrainingStage
from .evaluation_stage import EvaluationStage
from .persistence_stage import PersistenceStage


class PipelineRunner:
    """
    Main pipeline runner for German TTS fine-tuning.
    
    Coordinates execution of all pipeline stages with error handling,
    progress tracking, and comprehensive reporting.
    """
    
    def __init__(self):
        self.stages = {
            "data_loading": DataLoadingStage,
            "preprocessing": PreprocessingStage,
            "training": TrainingStage,
            "evaluation": EvaluationStage,
            "persistence": PersistenceStage
        }
        
        self.stage_order = [
            "data_loading",
            "preprocessing", 
            "training",
            "evaluation",
            "persistence"
        ]
        
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_complete_pipeline(self, skip_stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete DVC pipeline.
        
        Args:
            skip_stages: List of stage names to skip
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        skip_stages = skip_stages or []
        self.start_time = time.time()
        
        print("🚀 Starting German TTS Fine-tuning Pipeline")
        print("=" * 60)
        
        pipeline_results = {
            "pipeline_status": "running",
            "stages_completed": [],
            "stages_failed": [],
            "stage_results": {},
            "total_duration": None,
            "error": None
        }
        
        try:
            for stage_name in self.stage_order:
                if stage_name in skip_stages:
                    print(f"⏭️  Skipping {stage_name} stage")
                    continue
                
                print(f"\n📋 Running {stage_name} stage...")
                print("-" * 40)
                
                try:
                    stage_class = self.stages[stage_name]
                    stage = stage_class()
                    
                    result = stage.run()
                    pipeline_results["stage_results"][stage_name] = result
                    pipeline_results["stages_completed"].append(stage_name)
                    
                    print(f"✅ {stage_name} completed successfully")
                    self._print_stage_summary(stage_name, result)
                    
                except PipelineError as e:
                    print(f"❌ {stage_name} failed: {e}")
                    pipeline_results["stages_failed"].append(stage_name)
                    pipeline_results["error"] = str(e)
                    pipeline_results["pipeline_status"] = "failed"
                    break
                
                except Exception as e:
                    print(f"💥 {stage_name} crashed: {e}")
                    pipeline_results["stages_failed"].append(stage_name)
                    pipeline_results["error"] = f"Unexpected error in {stage_name}: {str(e)}"
                    pipeline_results["pipeline_status"] = "failed"
                    break
            
            # Pipeline completed successfully
            if not pipeline_results["stages_failed"]:
                pipeline_results["pipeline_status"] = "completed"
                print(f"\n🎉 Pipeline completed successfully!")
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Pipeline interrupted by user")
            pipeline_results["pipeline_status"] = "interrupted"
            pipeline_results["error"] = "User interrupted"
        
        except Exception as e:
            print(f"\n💥 Pipeline failed with unexpected error: {e}")
            pipeline_results["pipeline_status"] = "crashed"
            pipeline_results["error"] = str(e)
        
        finally:
            self.end_time = time.time()
            pipeline_results["total_duration"] = self.end_time - self.start_time
            
            # Save pipeline results
            self._save_pipeline_results(pipeline_results)
            
            # Print final summary
            self._print_pipeline_summary(pipeline_results)
        
        return pipeline_results
    
    def run_stage(self, stage_name: str) -> Dict[str, Any]:
        """
        Run a single pipeline stage.
        
        Args:
            stage_name: Name of the stage to run
            
        Returns:
            Dictionary with stage results
        """
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}. Available: {list(self.stages.keys())}")
        
        print(f"🎯 Running single stage: {stage_name}")
        print("=" * 40)
        
        try:
            stage_class = self.stages[stage_name]
            stage = stage_class()
            
            result = stage.run()
            
            print(f"✅ {stage_name} completed successfully")
            self._print_stage_summary(stage_name, result)
            
            return result
            
        except PipelineError as e:
            print(f"❌ {stage_name} failed: {e}")
            raise
        
        except Exception as e:
            print(f"💥 {stage_name} crashed: {e}")
            raise
    
    def _print_stage_summary(self, stage_name: str, result: Dict[str, Any]):
        """Print summary for a completed stage."""
        duration = result.get("duration_seconds", 0)
        print(f"   ⏱️  Duration: {duration:.2f} seconds")
        
        # Stage-specific summaries
        if stage_name == "data_loading":
            if "results" in result:
                total_samples = result["results"].get("total_samples", 0)
                datasets = result["results"].get("datasets_processed", [])
                print(f"   📊 Loaded {total_samples} samples from {len(datasets)} datasets")
        
        elif stage_name == "preprocessing":
            if "results" in result:
                input_samples = result["results"].get("total_input_samples", 0)
                output_samples = result["results"].get("total_output_samples", 0)
                retention_rate = (output_samples / input_samples * 100) if input_samples > 0 else 0
                print(f"   🔧 Processed {input_samples} → {output_samples} samples ({retention_rate:.1f}% retained)")
        
        elif stage_name == "training":
            if "results" in result:
                final_loss = result["results"].get("training_metrics", {}).get("final_loss", 0)
                total_steps = result["results"].get("training_metrics", {}).get("total_steps", 0)
                print(f"   🎯 Final loss: {final_loss:.4f} after {total_steps} steps")
        
        elif stage_name == "evaluation":
            if "results" in result:
                metrics = result["results"].get("metrics_computed", {})
                if "overall_quality" in metrics:
                    quality = metrics["overall_quality"]
                    print(f"   📈 Overall quality score: {quality:.3f}/5.0")
        
        elif stage_name == "persistence":
            if "results" in result:
                vllm_compat = result["results"].get("vllm_compatibility", {}).get("is_compatible", False)
                models_saved = len(result["results"].get("models_saved", {}))
                print(f"   💾 Saved {models_saved} model formats, VLLM compatible: {vllm_compat}")
    
    def _print_pipeline_summary(self, results: Dict[str, Any]):
        """Print final pipeline summary."""
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)
        
        status = results["pipeline_status"]
        duration = results.get("total_duration", 0)
        
        print(f"Status: {status.upper()}")
        print(f"Total Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        print(f"Stages Completed: {len(results['stages_completed'])}")
        print(f"Stages Failed: {len(results['stages_failed'])}")
        
        if results["stages_completed"]:
            print(f"\n✅ Completed Stages:")
            for stage in results["stages_completed"]:
                print(f"   - {stage}")
        
        if results["stages_failed"]:
            print(f"\n❌ Failed Stages:")
            for stage in results["stages_failed"]:
                print(f"   - {stage}")
        
        if results.get("error"):
            print(f"\n🚨 Error: {results['error']}")
        
        # Show final model info if pipeline completed
        if status == "completed" and "persistence" in results["stage_results"]:
            persistence_results = results["stage_results"]["persistence"]["results"]
            vllm_compat = persistence_results.get("vllm_compatibility", {}).get("is_compatible", False)
            
            print(f"\n🎉 TRAINED MODEL READY!")
            print(f"   VLLM Compatible: {'✅' if vllm_compat else '❌'}")
            print(f"   Model Location: models/vllm_compatible/")
            print(f"   Deployment Package: models/deployment_ready/")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"pipeline_results_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Pipeline results saved to: {results_file}")


def main():
    """Main CLI interface for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="German TTS Fine-tuning Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.run_pipeline                    # Run complete pipeline
  python -m src.pipeline.run_pipeline --stage training   # Run single stage
  python -m src.pipeline.run_pipeline --skip data_loading preprocessing  # Skip stages
  python -m src.pipeline.run_pipeline --from training    # Start from specific stage
        """
    )
    
    parser.add_argument(
        "--stage",
        choices=["data_loading", "preprocessing", "training", "evaluation", "persistence"],
        help="Run a single stage only"
    )
    
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["data_loading", "preprocessing", "training", "evaluation", "persistence"],
        help="Skip specified stages"
    )
    
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=["data_loading", "preprocessing", "training", "evaluation", "persistence"],
        help="Start pipeline from specified stage"
    )
    
    parser.add_argument(
        "--to",
        dest="end_at",
        choices=["data_loading", "preprocessing", "training", "evaluation", "persistence"],
        help="End pipeline at specified stage"
    )
    
    args = parser.parse_args()
    
    try:
        runner = PipelineRunner()
        
        if args.stage:
            # Run single stage
            result = runner.run_stage(args.stage)
            sys.exit(0 if result.get("status") == "completed" else 1)
        
        else:
            # Run complete pipeline with modifications
            skip_stages = args.skip or []
            
            # Handle --from and --to arguments
            if args.start_from or args.end_at:
                stage_order = runner.stage_order
                
                start_idx = 0
                if args.start_from:
                    start_idx = stage_order.index(args.start_from)
                
                end_idx = len(stage_order)
                if args.end_at:
                    end_idx = stage_order.index(args.end_at) + 1
                
                # Skip stages outside the range
                for i, stage in enumerate(stage_order):
                    if i < start_idx or i >= end_idx:
                        skip_stages.append(stage)
            
            result = runner.run_complete_pipeline(skip_stages=skip_stages)
            sys.exit(0 if result["pipeline_status"] == "completed" else 1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    except Exception as e:
        print(f"\n💥 Pipeline runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
