"""
VLLM compatibility validation with detailed error reporting.

Provides comprehensive validation of model compatibility with VLLM
and generates detailed conversion logs for debugging.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class VLLMCompatibilityError(Exception):
    """Custom exception for VLLM compatibility issues."""
    pass

class VLLMCompatibilityValidator:
    """
    Validates model compatibility with VLLM and provides detailed error reports.
    
    Features:
    - Model architecture validation
    - Tokenizer compatibility checks
    - Export format verification
    - Detailed conversion logs
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()
        self.validation_log = []
        
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
    
    def _log_validation_step(self, step: str, status: str, details: str = ""):
        """Log a validation step with status and details."""
        entry = {
            'step': step,
            'status': status,
            'details': details
        }
        self.validation_log.append(entry)
        self.logger.info(f"{step}: {status} - {details}")
    
    def validate_model(self, model_path: str) -> bool:
        """
        Validate model compatibility with VLLM.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            True if model is VLLM-compatible
        """
        self.validation_log = []  # Reset log
        is_valid = True
        
        try:
            # 1. Check model files exist
            self._log_validation_step("File Check", "Running", "Checking required model files")
            if not self._check_model_files(model_path):
                self._log_validation_step("File Check", "Failed", "Missing required model files")
                is_valid = False
            else:
                self._log_validation_step("File Check", "Passed", "All required files present")
            
            # 2. Validate model architecture
            self._log_validation_step("Architecture Check", "Running", "Validating model architecture")
            arch_valid, arch_details = self._validate_architecture(model_path)
            if not arch_valid:
                self._log_validation_step("Architecture Check", "Failed", arch_details)
                is_valid = False
            else:
                self._log_validation_step("Architecture Check", "Passed", arch_details)
            
            # 3. Validate tokenizer
            self._log_validation_step("Tokenizer Check", "Running", "Validating tokenizer")
            tokenizer_valid, tokenizer_details = self._validate_tokenizer(model_path)
            if not tokenizer_valid:
                self._log_validation_step("Tokenizer Check", "Failed", tokenizer_details)
                is_valid = False
            else:
                self._log_validation_step("Tokenizer Check", "Passed", tokenizer_details)
            
            # 4. Test inference
            self._log_validation_step("Inference Test", "Running", "Testing model inference")
            inference_valid, inference_details = self._test_inference(model_path)
            if not inference_valid:
                self._log_validation_step("Inference Test", "Failed", inference_details)
                is_valid = False
            else:
                self._log_validation_step("Inference Test", "Passed", inference_details)
            
            return is_valid
            
        except Exception as e:
            self._log_validation_step("Validation", "Error", f"Unexpected error: {str(e)}")
            raise VLLMCompatibilityError(f"Validation failed: {str(e)}") from e
    
    def _check_model_files(self, model_path: str) -> bool:
        """Check if required model files exist."""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"
        ]
        
        missing_files = []
        for file in required_files:
            if not (Path(model_path) / file).exists():
                missing_files.append(file)
                
        return len(missing_files) == 0
    
    def _validate_architecture(self, model_path: str) -> Tuple[bool, str]:
        """Validate model architecture is VLLM-compatible."""
        try:
            config = AutoModelForCausalLM.from_pretrained(model_path, config_only=True)
            
            # Check for required architecture components
            required_components = [
                "num_hidden_layers",
                "hidden_size",
                "num_attention_heads"
            ]
            
            missing = []
            for comp in required_components:
                if not hasattr(config, comp):
                    missing.append(comp)
            
            if missing:
                return False, f"Missing architecture components: {', '.join(missing)}"
            
            # Check for known incompatible architectures
            if hasattr(config, "architectures"):
                for arch in config.architectures:
                    if "GPT2" in arch:
                        return False, "GPT2 architecture not fully compatible with VLLM"
            
            return True, f"Valid architecture: {config.model_type}"
            
        except Exception as e:
            return False, f"Error loading model config: {str(e)}"
    
    def _validate_tokenizer(self, model_path: str) -> Tuple[bool, str]:
        """Validate tokenizer compatibility."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Test tokenization
            test_text = "Dies ist ein Test für deutsche Sprache."
            tokens = tokenizer.encode(test_text)
            decoded = tokenizer.decode(tokens)
            
            if decoded.strip() != test_text.strip():
                return False, "Tokenizer roundtrip failed"
                
            return True, "Tokenizer validated successfully"
            
        except Exception as e:
            return False, f"Tokenizer validation failed: {str(e)}"
    
    def _test_inference(self, model_path: str) -> Tuple[bool, str]:
        """Test basic model inference."""
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            input_text = "Hallo, wie geht es dir?"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            if not hasattr(outputs, 'logits'):
                return False, "Model output missing logits"
                
            return True, "Inference test passed"
            
        except Exception as e:
            return False, f"Inference test failed: {str(e)}"
    
    def get_validation_report(self) -> str:
        """Get detailed validation report in JSON format."""
        return json.dumps(self.validation_log, indent=2)
    
    def save_validation_report(self, output_path: str) -> None:
        """Save validation report to file."""
        report = self.get_validation_report()
        Path(output_path).write_text(report)
        self.logger.info(f"Saved validation report to {output_path}")
