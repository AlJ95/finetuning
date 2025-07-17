"""
Unit tests for MLSGermanDataProcessor.

Tests the MLS-specific functionality including multi-speaker processing,
OPUS audio handling, and speaker balance filtering.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
import json
import numpy as np

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from mls_german_processor import MLSGermanDataProcessor
from data_processor_base import ProcessingConfig, AudioDataset, QualityMetrics


class TestMLSGermanDataProcessor(unittest.TestCase):
    """Test cases for MLS German data processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ProcessingConfig(
            min_duration=1.0,
            max_duration=15.0,
            min_snr=8.0,
            quality_threshold=0.5,
            max_workers=2,
            batch_size=10
        )
        
        self.processor = MLSGermanDataProcessor(
            config=self.config,
            target_speakers=['1001', '1002'],
            max_samples_per_speaker=100
        )
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertEqual(self.processor.expected_sample_rate, 16000)
        self.assertEqual(self.processor.audio_format, 'opus')
        self.assertEqual(self.processor.target_speakers, ['1001', '1002'])
        self.assertEqual(self.processor.max_samples_per_speaker, 100)
    
    def test_load_transcripts(self):
        """Test transcript loading from MLS format."""
        # Create temporary transcript file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("1001_001_001\tDas ist ein Test.\n")
            f.write("1001_001_002\tNoch ein Test mit längerer Transkription.\n")
            f.write("1002_001_001\tEin dritter Test.\n")
            f.write("\n")  # Empty line
            f.write("invalid_line_without_tab\n")  # Invalid format
            temp_file = f.name
        
        try:
            transcripts = self.processor._load_transcripts(Path(temp_file))
            
            expected = {
                "1001_001_001": "Das ist ein Test.",
                "1001_001_002": "Noch ein Test mit längerer Transkription.",
                "1002_001_001": "Ein dritter Test."
            }
            
            self.assertEqual(transcripts, expected)
            
        finally:
            os.unlink(temp_file)
    
    def test_parse_transcript(self):
        """Test transcript parsing and cleaning."""
        test_cases = [
            {
                'input': {'transcript': '  Das ist ein   Test.  '},
                'expected': 'Das ist ein Test.'
            },
            {
                'input': {'transcript': 'Text\tmit\nWhitespace'},
                'expected': 'Text mit Whitespace'
            },
            {
                'input': {'transcript': ''},
                'expected': ''
            },
            {
                'input': {},
                'expected': ''
            }
        ]
        
        for case in test_cases:
            result = self.processor.parse_transcript(case['input'])
            self.assertEqual(result, case['expected'])
    
    def test_extract_speaker_metadata(self):
        """Test speaker metadata extraction."""
        metadata = {
            'speaker_id': '1001',
            'book_id': '001',
            'split': 'train'
        }
        
        # Mock speaker stats
        self.processor.speaker_stats['1001'] = 50
        
        result = self.processor.extract_speaker_metadata(metadata)
        
        expected = {
            'speaker_id': '1001',
            'is_multi_speaker': True,
            'speaker_sample_count': 50,
            'book_id': '001',
            'split': 'train'
        }
        
        self.assertEqual(result, expected)
    
    @patch('librosa.load')
    def test_validate_audio_quality(self, mock_librosa_load):
        """Test audio quality validation for MLS format."""
        # Mock audio data
        mock_audio_data = np.random.randn(16000 * 5)  # 5 seconds of audio
        mock_librosa_load.return_value = (mock_audio_data, 16000)
        
        # Mock the calculate_quality_metrics method
        mock_quality = QualityMetrics(
            signal_to_noise_ratio=15.0,
            duration_seconds=5.0,
            sample_rate=16000,
            rms_energy=0.05,
            zero_crossing_rate=0.1,
            spectral_centroid=2000.0
        )
        
        with patch.object(self.processor, 'calculate_quality_metrics', return_value=mock_quality):
            # Test valid audio
            audio_file = Path("speaker/book/1001_001_001.opus")
            is_valid, quality_metrics = self.processor.validate_audio_quality(audio_file)
            
            self.assertTrue(is_valid)
            self.assertEqual(quality_metrics, mock_quality)
    
    @patch('librosa.load')
    def test_validate_audio_quality_invalid(self, mock_librosa_load):
        """Test audio quality validation with invalid audio."""
        # Mock low-quality audio data
        mock_audio_data = np.random.randn(16000) * 0.001  # Very quiet audio
        mock_librosa_load.return_value = (mock_audio_data, 16000)
        
        # Mock low-quality metrics
        mock_quality = QualityMetrics(
            signal_to_noise_ratio=5.0,  # Below threshold
            duration_seconds=1.0,
            sample_rate=16000,
            rms_energy=0.001,  # Too quiet
            zero_crossing_rate=0.1,
            spectral_centroid=2000.0
        )
        
        with patch.object(self.processor, 'calculate_quality_metrics', return_value=mock_quality):
            audio_file = Path("speaker/book/1001_001_001.opus")
            is_valid, quality_metrics = self.processor.validate_audio_quality(audio_file)
            
            self.assertFalse(is_valid)
    
    def test_calculate_balance_coefficient(self):
        """Test speaker balance coefficient calculation."""
        # Perfectly balanced
        balanced_counts = [100, 100, 100, 100]
        balance_coeff = self.processor._calculate_balance_coefficient(balanced_counts)
        self.assertAlmostEqual(balance_coeff, 0.0, places=2)
        
        # Highly imbalanced
        imbalanced_counts = [1000, 10, 5, 1]
        balance_coeff = self.processor._calculate_balance_coefficient(imbalanced_counts)
        self.assertGreater(balance_coeff, 0.5)
        
        # Edge cases
        self.assertEqual(self.processor._calculate_balance_coefficient([]), 0.0)
        self.assertEqual(self.processor._calculate_balance_coefficient([100]), 0.0)
    
    def test_filter_by_speaker_balance(self):
        """Test speaker balance filtering."""
        # Create mock dataset with imbalanced speakers
        dataset = []
        
        # Speaker 1001: 5 samples (quality scores: 0.9, 0.8, 0.7, 0.6, 0.5)
        for i, quality in enumerate([0.9, 0.8, 0.7, 0.6, 0.5]):
            item = AudioDataset(
                file_path=Path(f"1001_{i}.opus"),
                text_transcript=f"Text {i}",
                duration=5.0,
                sample_rate=16000,
                quality_score=quality,
                metadata={'original_metadata': {'speaker_id': '1001'}}
            )
            dataset.append(item)
        
        # Speaker 1002: 3 samples (quality scores: 0.8, 0.7, 0.6)
        for i, quality in enumerate([0.8, 0.7, 0.6]):
            item = AudioDataset(
                file_path=Path(f"1002_{i}.opus"),
                text_transcript=f"Text {i}",
                duration=5.0,
                sample_rate=16000,
                quality_score=quality,
                metadata={'original_metadata': {'speaker_id': '1002'}}
            )
            dataset.append(item)
        
        # Filter to max 2 samples per speaker
        balanced_dataset = self.processor.filter_by_speaker_balance(dataset, 2)
        
        # Should have 4 samples total (2 per speaker)
        self.assertEqual(len(balanced_dataset), 4)
        
        # Check that best quality samples were kept
        speaker_1001_samples = [item for item in balanced_dataset 
                               if item.metadata['original_metadata']['speaker_id'] == '1001']
        speaker_1002_samples = [item for item in balanced_dataset 
                               if item.metadata['original_metadata']['speaker_id'] == '1002']
        
        self.assertEqual(len(speaker_1001_samples), 2)
        self.assertEqual(len(speaker_1002_samples), 2)
        
        # Check quality scores (should be the highest ones)
        speaker_1001_qualities = [item.quality_score for item in speaker_1001_samples]
        speaker_1002_qualities = [item.quality_score for item in speaker_1002_samples]
        
        self.assertIn(0.9, speaker_1001_qualities)  # Best quality kept
        self.assertIn(0.8, speaker_1001_qualities)  # Second best kept
        self.assertNotIn(0.5, speaker_1001_qualities)  # Worst quality filtered
        
        self.assertIn(0.8, speaker_1002_qualities)  # Best quality kept
        self.assertIn(0.7, speaker_1002_qualities)  # Second best kept
    
    def test_get_speaker_balance_stats(self):
        """Test speaker balance statistics calculation."""
        # Create mock dataset
        dataset = [
            AudioDataset(
                file_path=Path("1001_001.opus"),
                text_transcript="Text 1",
                duration=5.0,
                sample_rate=16000,
                quality_score=0.8,
                metadata={'original_metadata': {'speaker_id': '1001'}}
            ),
            AudioDataset(
                file_path=Path("1001_002.opus"),
                text_transcript="Text 2",
                duration=3.0,
                sample_rate=16000,
                quality_score=0.7,
                metadata={'original_metadata': {'speaker_id': '1001'}}
            ),
            AudioDataset(
                file_path=Path("1002_001.opus"),
                text_transcript="Text 3",
                duration=4.0,
                sample_rate=16000,
                quality_score=0.9,
                metadata={'original_metadata': {'speaker_id': '1002'}}
            )
        ]
        
        stats = self.processor.get_speaker_balance_stats(dataset)
        
        self.assertEqual(stats['total_speakers'], 2)
        self.assertEqual(stats['total_samples'], 3)
        self.assertAlmostEqual(stats['total_duration_hours'], 12.0 / 3600, places=4)
        self.assertEqual(stats['speaker_distribution']['1001'], 2)
        self.assertEqual(stats['speaker_distribution']['1002'], 1)
        self.assertAlmostEqual(stats['speaker_avg_quality']['1001'], 0.75, places=2)
        self.assertAlmostEqual(stats['speaker_avg_quality']['1002'], 0.9, places=2)
    
    def test_get_processing_stats(self):
        """Test comprehensive processing statistics."""
        # Create mock dataset
        dataset = [
            AudioDataset(
                file_path=Path("test.opus"),
                text_transcript="Test text",
                duration=5.0,
                sample_rate=16000,
                quality_score=0.8,
                metadata={'original_metadata': {'speaker_id': '1001'}}
            )
        ]
        
        stats = self.processor.get_processing_stats(dataset)
        
        # Check that both base and MLS-specific stats are included
        self.assertIn('total_items', stats)
        self.assertIn('mls_specific', stats)
        self.assertEqual(stats['dataset_type'], 'MLS German')
        self.assertTrue(stats['multi_speaker'])
        self.assertEqual(stats['audio_format'], 'opus')
        self.assertEqual(stats['expected_sample_rate'], 16000)


class TestMLSDatasetStructure(unittest.TestCase):
    """Test MLS dataset structure handling."""
    
    def setUp(self):
        """Set up test directory structure."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.temp_dir)
        
        # Create MLS-like directory structure
        self.train_dir = self.dataset_path / 'train'
        self.train_audio_dir = self.train_dir / 'audio'
        self.speaker_dir = self.train_audio_dir / '1001'
        self.book_dir = self.speaker_dir / '001'
        
        # Create directories
        self.book_dir.mkdir(parents=True)
        
        # Create transcript file
        self.transcript_file = self.train_dir / 'transcripts.txt'
        with open(self.transcript_file, 'w', encoding='utf-8') as f:
            f.write("1001_001_001\tDas ist ein Test.\n")
            f.write("1001_001_002\tNoch ein Test.\n")
        
        # Create mock audio files
        (self.book_dir / '1001_001_001.opus').touch()
        (self.book_dir / '1001_001_002.opus').touch()
        
        self.processor = MLSGermanDataProcessor()
    
    def tearDown(self):
        """Clean up test directory."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_dataset_metadata_structure(self):
        """Test loading metadata from MLS directory structure."""
        metadata_list = self.processor.load_dataset_metadata(self.dataset_path)
        
        self.assertEqual(len(metadata_list), 2)
        
        # Check first item
        item = metadata_list[0]
        self.assertIn('file_path', item)
        self.assertIn('speaker_id', item)
        self.assertIn('book_id', item)
        self.assertIn('transcript', item)
        self.assertEqual(item['speaker_id'], '1001')
        self.assertEqual(item['book_id'], '001')
        self.assertEqual(item['split'], 'train')
        self.assertEqual(item['audio_format'], 'opus')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)