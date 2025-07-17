"""
Tests for TorstenVoiceDataProcessor.

This module contains unit tests for the Thorsten-Voice dataset processor,
testing LJSpeech format parsing, German text processing, and quality validation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import csv
import numpy as np

import sys
sys.path.append('src')

from torsten_voice_processor import TorstenVoiceDataProcessor
from data_processor_base import ProcessingConfig


class TestTorstenVoiceDataProcessor(unittest.TestCase):
    """Test cases for TorstenVoiceDataProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = TorstenVoiceDataProcessor()
        
    def test_initialization(self):
        """Test processor initialization with default config."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.config.target_sample_rate, 22050)
        self.assertEqual(self.processor.config.min_snr, 15.0)
        self.assertEqual(self.processor.config.quality_threshold, 0.7)
    
    def test_clean_german_text(self):
        """Test German text cleaning and normalization."""
        # Test basic cleaning
        text = "  Hallo Welt!  "
        cleaned = self.processor._clean_german_text(text)
        self.assertEqual(cleaned, "hallo welt!")
        
        # Test multiple spaces
        text = "Das  ist   ein    Test."
        cleaned = self.processor._clean_german_text(text)
        self.assertEqual(cleaned, "das ist ein test.")
    
    def test_validate_german_text(self):
        """Test German text validation."""
        # Valid German text
        self.assertTrue(self.processor._validate_german_text("Das ist ein guter Test."))
        
        # Invalid text (too short)
        self.assertFalse(self.processor._validate_german_text("Hi"))
        self.assertFalse(self.processor._validate_german_text(""))
        
        # Invalid text (too long)
        long_text = "a" * 250
        self.assertFalse(self.processor._validate_german_text(long_text))
    
    def test_parse_transcript(self):
        """Test transcript parsing from metadata."""
        # Test with normalized transcript
        metadata = {
            'filename': 'test001',
            'transcript': 'Das ist ein Test!',
            'normalized_transcript': 'das ist ein test!'
        }
        result = self.processor.parse_transcript(metadata)
        self.assertEqual(result, 'das ist ein test!')
        
        # Test without normalized transcript
        metadata = {
            'filename': 'test002',
            'transcript': 'Hallo Welt!'
        }
        result = self.processor.parse_transcript(metadata)
        self.assertEqual(result, 'hallo welt!')
    
    @patch('torsten_voice_processor.librosa')
    def test_align_text_audio(self, mock_librosa):
        """Test text-audio alignment calculation."""
        # Mock librosa.load to return fake audio data
        mock_audio_data = np.random.random(44100)  # 1 second at 44.1kHz
        mock_librosa.load.return_value = (mock_audio_data, 22050)
        
        # Test alignment calculation
        test_file = Path("test.wav")
        score = self.processor.align_text_audio("Das ist ein Test", test_file)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        mock_librosa.load.assert_called_once()


if __name__ == '__main__':
    unittest.main()