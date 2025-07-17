"""
Tests for the abstract DataProcessor base class.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.data_processor_base import (
    DataProcessor, 
    AudioDataset, 
    QualityMetrics, 
    ProcessingConfig
)


class ConcreteDataProcessor(DataProcessor):
    """Concrete implementation for testing the abstract base class."""
    
    def load_dataset_metadata(self, dataset_path: Path):
        """Mock implementation for testing."""
        return [
            {
                'file_path': 'test_audio_1.wav',
                'transcript': 'Dies ist ein Test.',
                'speaker_id': 'speaker_1'
            },
            {
                'file_path': 'test_audio_2.wav', 
                'transcript': 'Das ist ein weiterer Test.',
                'speaker_id': 'speaker_2'
            }
        ]
    
    def parse_transcript(self, metadata):
        """Mock implementation for testing."""
        return metadata.get('transcript', '')


class TestDataProcessorBase:
    """Test cases for DataProcessor base class functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ProcessingConfig(
            min_duration=1.0,
            max_duration=10.0,
            min_snr=5.0,
            max_workers=2,
            batch_size=2
        )
        self.processor = ConcreteDataProcessor(self.config)
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.config.min_duration == 1.0
        assert self.processor.config.max_duration == 10.0
        assert self.processor.config.min_snr == 5.0
        assert self.processor.logger is not None
    
    def test_default_config(self):
        """Test processor with default configuration."""
        processor = ConcreteDataProcessor()
        assert processor.config.min_duration == 0.5
        assert processor.config.max_duration == 30.0
        assert processor.config.target_sample_rate == 22050
    
    @patch('librosa.load')
    @patch('soundfile.info')
    def test_extract_audio_metadata(self, mock_sf_info, mock_librosa_load):
        """Test audio metadata extraction."""
        # Mock soundfile info
        mock_info = Mock()
        mock_info.duration = 5.0
        mock_info.samplerate = 22050
        mock_info.channels = 1
        mock_info.format = 'WAV'
        mock_info.subtype = 'PCM_16'
        mock_info.frames = 110250
        mock_sf_info.return_value = mock_info
        
        # Mock librosa load
        mock_audio = np.random.randn(110250)
        mock_librosa_load.return_value = (mock_audio, 22050)
        
        # Test metadata extraction
        test_file = Path('test_audio.wav')
        metadata = self.processor.extract_audio_metadata(test_file)
        
        assert metadata['duration'] == 5.0
        assert metadata['sample_rate'] == 22050
        assert metadata['channels'] == 1
        assert 'rms_energy' in metadata
        assert 'zero_crossing_rate' in metadata
    
    def test_calculate_quality_metrics(self):
        """Test quality metrics calculation."""
        # Create test audio data
        sample_rate = 22050
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration)) * 0.1
        
        with patch('librosa.feature.zero_crossing_rate') as mock_zcr, \
             patch('librosa.feature.spectral_centroid') as mock_sc:
            
            mock_zcr.return_value = np.array([[0.05]])
            mock_sc.return_value = np.array([[1000.0]])
            
            metrics = self.processor.calculate_quality_metrics(audio_data, sample_rate)
            
            assert isinstance(metrics, QualityMetrics)
            assert metrics.duration_seconds == pytest.approx(duration, rel=1e-2)
            assert metrics.sample_rate == sample_rate
            assert metrics.rms_energy > 0
            assert metrics.signal_to_noise_ratio != 0
    
    @patch('librosa.load')
    def test_validate_audio_quality(self, mock_librosa_load):
        """Test audio quality validation."""
        # Mock good quality audio
        good_audio = np.random.randn(44100) * 0.1  # 2 seconds at 22050 Hz
        mock_librosa_load.return_value = (good_audio, 22050)
        
        with patch.object(self.processor, 'calculate_quality_metrics') as mock_calc:
            mock_calc.return_value = QualityMetrics(
                signal_to_noise_ratio=15.0,  # Above threshold
                duration_seconds=2.0,  # Within range
                sample_rate=22050,
                rms_energy=0.1,  # Above minimum
                zero_crossing_rate=0.05,
                spectral_centroid=1000.0
            )
            
            is_valid, metrics = self.processor.validate_audio_quality(Path('test.wav'))
            
            assert is_valid is True
            assert metrics.signal_to_noise_ratio == 15.0
    
    @patch('librosa.load')
    def test_validate_audio_quality_invalid(self, mock_librosa_load):
        """Test audio quality validation with invalid audio."""
        # Mock poor quality audio
        poor_audio = np.random.randn(11025) * 0.001  # 0.5 seconds, very quiet
        mock_librosa_load.return_value = (poor_audio, 22050)
        
        with patch.object(self.processor, 'calculate_quality_metrics') as mock_calc:
            mock_calc.return_value = QualityMetrics(
                signal_to_noise_ratio=2.0,  # Below threshold
                duration_seconds=0.5,  # Too short
                sample_rate=22050,
                rms_energy=0.0001,  # Too quiet
                zero_crossing_rate=0.05,
                spectral_centroid=1000.0
            )
            
            is_valid, metrics = self.processor.validate_audio_quality(Path('test.wav'))
            
            assert is_valid is False
    
    @patch('librosa.load')
    def test_align_text_audio(self, mock_librosa_load):
        """Test text-audio alignment calculation."""
        # Mock audio (2 seconds)
        audio_data = np.random.randn(44100)
        mock_librosa_load.return_value = (audio_data, 22050)
        
        # Test with typical German speech rate
        text = "Dies ist ein Test mit etwa dreißig Zeichen."  # ~27 chars
        alignment_score = self.processor.align_text_audio(text, Path('test.wav'))
        
        assert 0.0 <= alignment_score <= 1.0
        assert isinstance(alignment_score, float)
    
    def test_filter_dataset(self):
        """Test dataset filtering based on quality threshold."""
        # Create test dataset with varying quality scores
        dataset = [
            AudioDataset(
                file_path=Path('good1.wav'),
                text_transcript='Test 1',
                duration=2.0,
                sample_rate=22050,
                quality_score=0.8,  # Above threshold
                metadata={}
            ),
            AudioDataset(
                file_path=Path('bad1.wav'),
                text_transcript='Test 2',
                duration=2.0,
                sample_rate=22050,
                quality_score=0.3,  # Below threshold
                metadata={}
            ),
            AudioDataset(
                file_path=Path('good2.wav'),
                text_transcript='Test 3',
                duration=2.0,
                sample_rate=22050,
                quality_score=0.7,  # Above threshold
                metadata={}
            )
        ]
        
        filtered = self.processor.filter_dataset(dataset)
        
        assert len(filtered) == 2
        assert all(item.quality_score >= self.processor.config.quality_threshold for item in filtered)
    
    def test_get_processing_stats(self):
        """Test processing statistics generation."""
        # Create test dataset
        dataset = [
            AudioDataset(
                file_path=Path('test1.wav'),
                text_transcript='Short test',
                duration=2.0,
                sample_rate=22050,
                quality_score=0.8,
                metadata={}
            ),
            AudioDataset(
                file_path=Path('test2.wav'),
                text_transcript='Longer test transcript',
                duration=5.0,
                sample_rate=22050,
                quality_score=0.9,
                metadata={}
            )
        ]
        
        stats = self.processor.get_processing_stats(dataset)
        
        assert stats['total_items'] == 2
        assert stats['total_duration_hours'] == pytest.approx(7.0 / 3600, rel=1e-3)
        assert stats['avg_duration_seconds'] == 3.5
        assert stats['min_duration_seconds'] == 2.0
        assert stats['max_duration_seconds'] == 5.0
        assert stats['avg_quality_score'] == pytest.approx(0.85, rel=1e-3)
        assert 22050 in stats['sample_rates']
    
    def test_get_processing_stats_empty(self):
        """Test processing statistics with empty dataset."""
        stats = self.processor.get_processing_stats([])
        assert stats == {}
    
    @patch.object(ConcreteDataProcessor, 'process_batch')
    def test_process_dataset(self, mock_process_batch):
        """Test full dataset processing."""
        # Mock batch processing results
        mock_process_batch.return_value = [
            AudioDataset(
                file_path=Path('test1.wav'),
                text_transcript='Test 1',
                duration=2.0,
                sample_rate=22050,
                quality_score=0.8,
                metadata={}
            )
        ]
        
        dataset_path = Path('test_dataset')
        result = self.processor.process_dataset(dataset_path)
        
        assert len(result) == 1
        assert mock_process_batch.called
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # This should fail because abstract methods aren't implemented
            DataProcessor()


if __name__ == '__main__':
    pytest.main([__file__])