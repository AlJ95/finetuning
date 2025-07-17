"""
Tests for UnslothTrainer class.

This module contains unit tests for the Unsloth integration with Orpheus 3B,
including configuration, model loading, dataset preparation, and training setup.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from src.unsloth_trainer import (
    TrainingConfig, 
    TrainingResults, 
    UnslothTrainer,
    UNSLOTH_AVAILABLE
)
from src.data_processor_base import AudioDataset


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        # Test default values
        assert config.model_name == "unsloth/orpheus-3b-0.1-ft"
        assert config.max_seq_length == 2048
        assert config.load_in_4bit == False
        assert config.r == 16
        assert config.lora_alpha == 16
        assert config.target_sample_rate == 24000
        assert config.per_device_train_batch_size == 1
        assert config.learning_rate == 2e-4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            model_name="custom/model",
            max_seq_length=1024,
            learning_rate=1e-4,
            r=32
        )
        
        assert config.model_name == "custom/model"
        assert config.max_seq_length == 1024
        assert config.learning_rate == 1e-4
        assert config.r == 32
    
    def test_to_training_arguments(self):
        """Test conversion to TrainingArguments."""
        config = TrainingConfig()
        
        # Mock the is_bfloat16_supported function
        with patch('src.unsloth_trainer.is_bfloat16_supported', return_value=True):
            training_args = config.to_training_arguments()
            
            assert training_args.per_device_train_batch_size == 1
            assert training_args.learning_rate == 2e-4
            assert training_args.num_train_epochs == 3
            assert training_args.bf16 == True
            assert training_args.fp16 == False


class TestTrainingResults:
    """Test TrainingResults dataclass."""
    
    def test_training_results_creation(self):
        """Test TrainingResults creation."""
        results = TrainingResults(
            final_loss=0.5,
            training_time=120.0,
            total_steps=100,
            model_path="/path/to/model",
            tokenizer_path="/path/to/tokenizer",
            training_logs=[{"step": 1, "loss": 0.8}],
            memory_stats={"gpu_memory_gb": 8.0}
        )
        
        assert results.final_loss == 0.5
        assert results.training_time == 120.0
        assert results.total_steps == 100
        assert results.model_path == "/path/to/model"
        assert len(results.training_logs) == 1
        assert "gpu_memory_gb" in results.memory_stats


@pytest.mark.skipif(not UNSLOTH_AVAILABLE, reason="Unsloth not available")
class TestUnslothTrainer:
    """Test UnslothTrainer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = TrainingConfig(
            output_dir="test_output",
            num_train_epochs=1,
            max_steps=5
        )
        
        # Create mock audio datasets
        self.mock_datasets = [
            AudioDataset(
                file_path=Path("test1.wav"),
                text_transcript="Hallo Welt",
                duration=2.5,
                sample_rate=24000,
                quality_score=0.9
            ),
            AudioDataset(
                file_path=Path("test2.wav"),
                text_transcript="Guten Tag",
                duration=1.8,
                sample_rate=24000,
                quality_score=0.8
            )
        ]
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = UnslothTrainer(self.config)
        
        assert trainer.config == self.config
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None
    
    def test_trainer_initialization_without_unsloth(self):
        """Test trainer initialization when Unsloth is not available."""
        with patch('src.unsloth_trainer.UNSLOTH_AVAILABLE', False):
            with pytest.raises(ImportError, match="Unsloth is not available"):
                UnslothTrainer(self.config)
    
    @patch('src.unsloth_trainer.FastLanguageModel')
    def test_load_orpheus_model(self, mock_fast_model):
        """Test Orpheus model loading."""
        # Setup mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_fast_model.from_pretrained.return_value = (mock_model, mock_tokenizer)
        mock_fast_model.get_peft_model.return_value = mock_model
        
        trainer = UnslothTrainer(self.config)
        model, tokenizer = trainer.load_orpheus_model()
        
        # Verify model loading was called correctly
        mock_fast_model.from_pretrained.assert_called_once_with(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Verify LoRA configuration was applied
        mock_fast_model.get_peft_model.assert_called_once()
        
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
    
    def test_prepare_dataset_for_unsloth(self):
        """Test dataset preparation for Unsloth."""
        trainer = UnslothTrainer(self.config)
        
        with patch('src.unsloth_trainer.Dataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.cast_column.return_value = mock_dataset
            mock_dataset_class.from_dict.return_value = mock_dataset
            
            result = trainer.prepare_dataset_for_unsloth(self.mock_datasets)
            
            # Verify dataset creation
            mock_dataset_class.from_dict.assert_called_once()
            mock_dataset.cast_column.assert_called_once()
            
            # Check the data dict structure
            call_args = mock_dataset_class.from_dict.call_args[0][0]
            assert 'audio' in call_args
            assert 'text' in call_args
            assert 'duration' in call_args
            assert len(call_args['audio']) == 2  # Both samples should pass filtering


if __name__ == "__main__":
    pytest.main([__file__])