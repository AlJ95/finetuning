"""
Intelligent CUDA memory management with automatic recovery strategies.

Provides robust handling of CUDA out-of-memory errors with adaptive
batch sizing and gradient accumulation strategies.
"""

import logging
import torch
from typing import Optional, Dict

class MemoryError(Exception):
    """Custom exception for memory management errors."""
    pass

class MemoryManager:
    """
    Handles CUDA memory issues with intelligent recovery strategies.
    
    Features:
    - Automatic batch size reduction
    - Gradient accumulation adjustment
    - Memory fragmentation handling
    - Proactive memory monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        self.min_batch_size = 1  # Absolute minimum batch size
        
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
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current CUDA memory usage statistics.
        
        Returns:
            Dictionary with memory metrics in GB:
            {
                'allocated': current allocated memory,
                'reserved': current reserved memory,
                'total': total GPU memory,
                'peak_allocated': peak allocated memory,
                'peak_reserved': peak reserved memory
            }
        """
        if not torch.cuda.is_available():
            return {}
            
        return {
            'allocated': torch.cuda.memory_allocated() / (1024 ** 3),
            'reserved': torch.cuda.memory_reserved() / (1024 ** 3),
            'total': torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            'peak_allocated': torch.cuda.max_memory_allocated() / (1024 ** 3),
            'peak_reserved': torch.cuda.max_memory_reserved() / (1024 ** 3)
        }
    
    def clear_cache(self) -> None:
        """Clear CUDA cache and empty memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.logger.info("CUDA cache cleared")
    
    def handle_oom(self, current_batch_size: int, current_accumulation: int = 1) -> Tuple[int, int]:
        """
        Handle CUDA out-of-memory error with recovery strategy.
        
        Args:
            current_batch_size: Current batch size that caused OOM
            current_accumulation: Current gradient accumulation steps
            
        Returns:
            Tuple of (new_batch_size, new_accumulation_steps)
            
        Raises:
            MemoryError: If minimum thresholds are reached
        """
        self.clear_cache()
        
        stats = self.get_memory_stats()
        self.logger.warning(f"CUDA OOM detected. Memory stats: {stats}")
        
        # First try reducing batch size while keeping accumulation
        new_batch_size = max(self.min_batch_size, current_batch_size // 2)
        if new_batch_size != current_batch_size:
            self.logger.info(f"Reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size, current_accumulation
            
        # If batch size is already minimum, adjust gradient accumulation
        new_accumulation = current_accumulation * 2
        self.logger.info(f"Batch size at minimum ({self.min_batch_size}), increasing gradient accumulation to {new_accumulation}")
        
        if new_accumulation > 32:  # Prevent excessive accumulation
            raise MemoryError("Cannot recover from OOM - batch size and accumulation limits reached")
            
        return new_batch_size, new_accumulation
    
    def recommend_batch_size(self, model_memory_estimate: float, safety_factor: float = 0.8) -> Optional[int]:
        """
        Recommend initial batch size based on model memory estimate.
        
        Args:
            model_memory_estimate: Estimated model memory per sample in GB
            safety_factor: Safety margin (0-1)
            
        Returns:
            Recommended batch size or None if GPU unavailable
        """
        if not torch.cuda.is_available():
            return None
            
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        available_memory = total_memory * safety_factor
        
        batch_size = int(available_memory // model_memory_estimate)
        self.logger.info(f"Recommended batch size: {batch_size} (model memory: {model_memory_estimate:.2f}GB/sample, available: {available_memory:.2f}GB)")
        
        return max(self.min_batch_size, batch_size)
    
    def check_memory_usage(self, threshold: float = 0.9) -> bool:
        """
        Check if memory usage is approaching critical levels.
        
        Args:
            threshold: Warning threshold (0-1) of total memory usage
            
        Returns:
            True if memory usage is above threshold
        """
        if not torch.cuda.is_available():
            return False
            
        stats = self.get_memory_stats()
        if not stats:
            return False
            
        usage = stats['allocated'] / stats['total']
        if usage > threshold:
            self.logger.warning(f"High GPU memory usage: {usage:.1%} (threshold: {threshold:.0%})")
            return True
            
        return False
