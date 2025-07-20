"""
Structured error analysis with recovery suggestions.

Provides comprehensive error classification, analysis and
automated recovery strategy suggestions.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import re

class ErrorAnalyzer:
    """
    Analyzes errors and provides structured recovery suggestions.
    
    Features:
    - Error classification and categorization
    - Automated recovery strategy suggestions
    - Detailed debugging information
    - Error trend analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        self.error_db = []
        
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
    
    def analyze_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze an error and provide recovery suggestions.
        
        Args:
            error: The exception to analyze
            context: Additional context about the error
            
        Returns:
            Dictionary with analysis results:
            {
                'error_type': classified error type,
                'error_message': cleaned error message,
                'recovery_suggestions': list of suggestions,
                'debugging_info': additional debugging info
            }
        """
        analysis = {
            'error_type': self._classify_error(error),
            'error_message': str(error),
            'recovery_suggestions': [],
            'debugging_info': {}
        }
        
        # Add context if provided
        if context:
            analysis['context'] = context
        
        # Get specific suggestions based on error type
        analysis['recovery_suggestions'] = self._get_suggestions(analysis['error_type'])
        
        # Extract additional debugging info
        analysis['debugging_info'] = self._extract_debug_info(error)
        
        # Log the error analysis
        self.error_db.append(analysis)
        self.logger.info(f"Analyzed error: {analysis['error_type']}")
        
        return analysis
    
    def _classify_error(self, error: Exception) -> str:
        """Classify the error type based on exception and message."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # CUDA memory errors
        if 'cuda out of memory' in error_msg:
            return 'CUDA_OOM'
        elif 'cuda error' in error_msg:
            return 'CUDA_ERROR'
            
        # Audio processing errors
        elif 'audio' in error_msg or 'soundfile' in error_msg or 'librosa' in error_msg:
            return 'AUDIO_PROCESSING_ERROR'
            
        # Model compatibility errors
        elif 'compatibility' in error_msg or 'vllm' in error_msg:
            return 'MODEL_COMPATIBILITY_ERROR'
            
        # Training errors
        elif 'training' in error_msg or 'loss' in error_msg or 'gradient' in error_msg:
            return 'TRAINING_ERROR'
            
        # Default classification
        return f'GENERIC_{error_type}'
    
    def _get_suggestions(self, error_type: str) -> List[str]:
        """Get recovery suggestions based on error type."""
        suggestions_map = {
            'CUDA_OOM': [
                "Reduce batch size",
                "Increase gradient accumulation steps",
                "Use mixed precision training",
                "Clear CUDA cache with torch.cuda.empty_cache()",
                "Try model with smaller footprint"
            ],
            'AUDIO_PROCESSING_ERROR': [
                "Check audio file format compatibility",
                "Verify audio file is not corrupted",
                "Try alternative audio loading backend (soundfile, librosa, audioread)",
                "Convert audio to standard format (WAV, 16kHz, mono)"
            ],
            'MODEL_COMPATIBILITY_ERROR': [
                "Verify model architecture is VLLM-compatible",
                "Check tokenizer configuration",
                "Export model with different settings",
                "Validate model with VLLMCompatibilityValidator"
            ],
            'TRAINING_ERROR': [
                "Adjust learning rate",
                "Check training data quality",
                "Verify loss function implementation",
                "Monitor gradient flow"
            ]
        }
        
        # Default suggestions for unknown errors
        default_suggestions = [
            "Check logs for more details",
            "Verify input data format",
            "Retry operation with debug mode enabled"
        ]
        
        return suggestions_map.get(error_type, default_suggestions)
    
    def _extract_debug_info(self, error: Exception) -> Dict[str, Any]:
        """Extract additional debugging information from error."""
        debug_info = {}
        error_msg = str(error)
        
        # Extract CUDA memory stats if available
        if 'cuda' in error_msg.lower():
            debug_info['cuda_related'] = True
            match = re.search(r'(\d+)\s*MB', error_msg)
            if match:
                debug_info['memory_mb'] = int(match.group(1))
        
        # Extract file paths mentioned in error
        path_matches = re.findall(r'(\/[^\s]+)', error_msg)
        if path_matches:
            debug_info['referenced_paths'] = path_matches
            
        return debug_info
    
    def save_error_report(self, output_path: str) -> None:
        """
        Save all analyzed errors to a JSON report.
        
        Args:
            output_path: Path to save the report
        """
        report = {
            'total_errors': len(self.error_db),
            'error_types': self._get_error_type_stats(),
            'errors': self.error_db
        }
        
        Path(output_path).write_text(json.dumps(report, indent=2))
        self.logger.info(f"Saved error report to {output_path}")
    
    def _get_error_type_stats(self) -> Dict[str, int]:
        """Get statistics of error type occurrences."""
        stats = {}
        for error in self.error_db:
            error_type = error['error_type']
            stats[error_type] = stats.get(error_type, 0) + 1
        return stats
    
    def get_most_common_error(self) -> Optional[Dict[str, Any]]:
        """Get the most frequently occurring error analysis."""
        if not self.error_db:
            return None
            
        error_types = self._get_error_type_stats()
        most_common = max(error_types.items(), key=lambda x: x[1])
        
        for error in reversed(self.error_db):
            if error['error_type'] == most_common[0]:
                return error
                
        return None
