"""
Robust audio format handling with fallback mechanisms for TTS pipeline.

Provides multi-backend audio loading with automatic format conversion
and comprehensive error handling for various audio file formats.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# Try importing audio libraries with fallbacks
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import audioread
    AUDIOREAD_AVAILABLE = True
except ImportError:
    AUDIOREAD_AVAILABLE = False

class AudioFormatError(Exception):
    """Custom exception for audio format handling errors."""
    pass

class AudioFormatHandler:
    """
    Handles audio loading with multiple fallback mechanisms and format conversion.
    
    Usage:
        handler = AudioFormatHandler()
        audio_data, sample_rate = handler.load_audio("audio.mp3", target_sr=22050)
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file with multiple fallback mechanisms.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None for native)
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            AudioFormatError: If all loading attempts fail
        """
        methods = [
            self._try_librosa,
            self._try_soundfile, 
            self._try_audioread
        ]
        
        last_error = None
        
        for method in methods:
            try:
                audio, sr = method(file_path, target_sr)
                self.logger.info(f"Successfully loaded {file_path} using {method.__name__}")
                return audio, sr
            except Exception as e:
                last_error = e
                self.logger.warning(f"Failed to load {file_path} with {method.__name__}: {str(e)}")
                continue
                
        raise AudioFormatError(f"All audio loading methods failed for {file_path}. Last error: {str(last_error)}")
    
    def _try_librosa(self, file_path: str, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
        """Try loading with librosa (supports many formats but slower)."""
        if not LIBROSA_AVAILABLE:
            raise AudioFormatError("librosa not available")
            
        return librosa.load(file_path, sr=target_sr)
    
    def _try_soundfile(self, file_path: str, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
        """Try loading with soundfile (fast but limited formats)."""
        if not SOUNDFILE_AVAILABLE:
            raise AudioFormatError("soundfile not available")
            
        data, sr = sf.read(file_path)
        
        # Convert to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            
        # Resample if needed
        if target_sr is not None and sr != target_sr:
            if LIBROSA_AVAILABLE:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            else:
                raise AudioFormatError("Resampling requires librosa")
                
        return data, sr
    
    def _try_audioread(self, file_path: str, target_sr: Optional[int]) -> Tuple[np.ndarray, int]:
        """Try loading with audioread (fallback for problematic files)."""
        if not AUDIOREAD_AVAILABLE:
            raise AudioFormatError("audioread not available")
            
        with audioread.audio_open(file_path) as f:
            sr = f.samplerate
            data = []
            for buf in f:
                data.append(np.frombuffer(buf, dtype=np.float32))
                
        audio = np.concatenate(data)
        
        # Resample if needed
        if target_sr is not None and sr != target_sr:
            if LIBROSA_AVAILABLE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            else:
                raise AudioFormatError("Resampling requires librosa")
                
        return audio, sr
    
    def is_audio_file(self, file_path: str) -> bool:
        """
        Check if file is a valid audio file that can be processed.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file is a supported audio format
        """
        try:
            self.load_audio(file_path)
            return True
        except AudioFormatError:
            return False
