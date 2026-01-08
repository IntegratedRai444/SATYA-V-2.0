"""
Base Detector Class
Abstract base class for all detection modules
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract base class for all deepfake detectors.
    """

    def __init__(self, model_path: str, enable_gpu: bool = False):
        """
        Initialize the detector.

        Args:
            model_path: Path to model files
            enable_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.enable_gpu = enable_gpu
        self.device = "cuda" if enable_gpu else "cpu"
        self.models_loaded = False

        logger.info(f"Initializing {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def load_models(self):
        """Load required models for detection."""
        pass

    @abstractmethod
    def analyze(self, data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Analyze media data for deepfakes.

        Args:
            data: Raw media data as bytes
            **kwargs: Additional parameters

        Returns:
            Analysis result dictionary
        """
        pass

    def _create_result(
        self,
        authenticity: str,
        confidence: float,
        key_findings: list,
        **additional_data,
    ) -> Dict[str, Any]:
        """
        Create a standardized result dictionary.

        Args:
            authenticity: 'AUTHENTIC MEDIA' or 'MANIPULATED MEDIA'
            confidence: Confidence score (0-100)
            key_findings: List of finding strings
            **additional_data: Additional result fields

        Returns:
            Standardized result dictionary
        """
        import random
        from datetime import datetime

        result = {
            "success": True,
            "authenticity": authenticity,
            "confidence": confidence,
            "analysis_date": datetime.now().isoformat(),
            "case_id": f"{self.__class__.__name__.lower()}-{int(datetime.now().timestamp())}-{random.randint(1000, 9999)}",
            "key_findings": key_findings,
            "technical_details": {
                "detector": self.__class__.__name__,
                "device": self.device,
                "models_loaded": self.models_loaded,
            },
        }

        # Add any additional data
        result.update(additional_data)

        return result
