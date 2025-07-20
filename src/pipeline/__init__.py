"""
DVC Pipeline framework for German TTS Fine-tuning.

This package contains all pipeline stages for the German TTS fine-tuning project.
Each stage is designed to be modular, reproducible, and DVC-compatible.
"""

from .base_stage import BasePipelineStage
from .data_loading_stage import DataLoadingStage
from .preprocessing_stage import PreprocessingStage
from .training_stage import TrainingStage
from .evaluation_stage import EvaluationStage
from .persistence_stage import PersistenceStage

__all__ = [
    "BasePipelineStage",
    "DataLoadingStage", 
    "PreprocessingStage",
    "TrainingStage",
    "EvaluationStage",
    "PersistenceStage"
]
