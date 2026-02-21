"""
Multimodal Fusion Detector
Combines Image, Video, Audio, and Text NLP detection for comprehensive deepfake analysis
Uses ensemble learning and cross-modal verification for maximum accuracy
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Import all detectors
try:
    from .image_detector import ImageDetector

    IMAGE_DETECTOR_AVAILABLE = True
except:
    IMAGE_DETECTOR_AVAILABLE = False
    logger.warning("Image detector not available")

try:
    from .video_detector import VideoDetector

    VIDEO_DETECTOR_AVAILABLE = True
except:
    VIDEO_DETECTOR_AVAILABLE = False
    logger.warning("Video detector not available")

try:
    from .audio_detector import AudioDetector

    AUDIO_DETECTOR_AVAILABLE = True
except:
    AUDIO_DETECTOR_AVAILABLE = False
    logger.warning("Audio detector not available")

try:
    from .text_nlp_detector import TextNLPDetector

    TEXT_NLP_DETECTOR_AVAILABLE = True
except:
    TEXT_NLP_DETECTOR_AVAILABLE = False
    logger.warning("Text NLP detector not available")


class MultimodalFusionDetector:
    """
    Advanced multimodal fusion detector that combines:
    1. Image analysis (EfficientNet + FaceNet)
    2. Video analysis (temporal + motion)
    3. Audio analysis (Wav2Vec2 + prosody)
    4. Text NLP analysis (BERT/RoBERTa + linguistic)
    5. Cross-modal consistency checking
    6. Ensemble learning for final decision
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize multimodal fusion detector"""
        self.config = config or self._default_config()

        logger.info("🔮 Initializing Multimodal Fusion Detector")

        # Initialize all detectors using singleton pattern
        from services.detector_singleton import DetectorSingleton
        detector_singleton = DetectorSingleton()
        
        self.image_detector = detector_singleton.get_detector('image') if IMAGE_DETECTOR_AVAILABLE else None
        self.video_detector = detector_singleton.get_detector('video') if VIDEO_DETECTOR_AVAILABLE else None
        self.audio_detector = detector_singleton.get_detector('audio') if AUDIO_DETECTOR_AVAILABLE else None
        self.text_nlp_detector = (
            detector_singleton.get_detector('text') if TEXT_NLP_DETECTOR_AVAILABLE else None
        )

        # Fusion weights (can be learned)
        self.fusion_weights = {
            "image": 0.25,
            "video": 0.30,
            "audio": 0.25,
            "text": 0.20,
        }

        # Cross-modal weights
        self.cross_modal_weight = 0.15

        logger.info("✅ Multimodal fusion detector initialized")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "enable_cross_modal": True,
            "enable_ensemble": True,
            "min_modalities": 2,  # Minimum modalities required
            "confidence_threshold": 0.6,
        }

    def detect_multimodal(
        self,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
        transcript: Optional[str] = None,
    ) -> Dict:
        """
        Comprehensive multimodal analysis

        Args:
            image_path: Path to image file
            video_path: Path to video file
            audio_path: Path to audio file
            text: Text content to analyze
            transcript: Video/audio transcript

        Returns:
            Comprehensive multimodal detection results
        """
        try:
            logger.info("🔍 Starting multimodal analysis")

            # Initialize results
            results = {
                "success": True,
                "authenticity_score": 0.0,
                "confidence": 0.0,
                "label": "unknown",
                "modalities_analyzed": [],
                "individual_results": {},
                "cross_modal_analysis": {},
                "fusion_result": {},
                "explanation": "",
                "recommendations": [],
            }

            # 1. Analyze each modality
            modality_scores = {}

            # Image analysis
            if image_path and self.image_detector:
                logger.info("📸 Analyzing image...")
                # Load image and convert to numpy array
                from PIL import Image as PILImage
                image_pil = PILImage.open(image_path).convert("RGB")
                image_array = np.array(image_pil)
                image_result = self.image_detector.analyze(image_array)
                results["individual_results"]["image"] = image_result
                # Extract authenticity score properly
                if isinstance(image_result, dict):
                    modality_scores["image"] = image_result.get("authenticity_score", 0.5)
                else:
                    # Fallback if image_result is not a dict
                    modality_scores["image"] = 0.5
                results["modalities_analyzed"].append("image")

            # Video analysis
            if video_path and self.video_detector:
                logger.info("🎥 Analyzing video...")
                video_result = self.video_detector.detect(video_path)
                results["individual_results"]["video"] = video_result
                modality_scores["video"] = video_result.get("authenticity_score", 0.5)
                results["modalities_analyzed"].append("video")

            # Audio analysis
            if audio_path and self.audio_detector:
                logger.info("🎵 Analyzing audio...")
                audio_result = self.audio_detector.detect(audio_path)
                results["individual_results"]["audio"] = audio_result
                modality_scores["audio"] = audio_result.get("authenticity_score", 0.5)
                results["modalities_analyzed"].append("audio")

            # Text NLP analysis
            if (text or transcript) and self.text_nlp_detector:
                logger.info("🔤 Analyzing text...")
                text_to_analyze = text or transcript
                text_result = self.text_nlp_detector.detect(text_to_analyze)
                results["individual_results"]["text"] = text_result
                modality_scores["text"] = text_result.get("authenticity_score", 0.5)
                results["modalities_analyzed"].append("text")

            # Check minimum modalities
            if len(modality_scores) < self.config["min_modalities"]:
                results["warnings"] = [
                    f"Only {len(modality_scores)} modalities available, need at least {self.config['min_modalities']}"
                ]

            # 2. Cross-modal consistency analysis
            if self.config["enable_cross_modal"] and len(modality_scores) >= 2:
                cross_modal_result = self._analyze_cross_modal_consistency(
                    results["individual_results"]
                )
                results["cross_modal_analysis"] = cross_modal_result

            # 3. Ensemble fusion
            if self.config["enable_ensemble"]:
                fusion_result = self._ensemble_fusion(
                    modality_scores, results.get("cross_modal_analysis", {})
                )
                results["fusion_result"] = fusion_result
                results["authenticity_score"] = fusion_result["score"]
                results["confidence"] = fusion_result["confidence"]
                results["label"] = fusion_result["label"]
            else:
                # Simple weighted average
                final_score = self._weighted_average(modality_scores)
                results["authenticity_score"] = final_score
                results["confidence"] = 1.0 - np.std(list(modality_scores.values()))
                results["label"] = self._determine_label(final_score)

            # 4. Generate explanation
            results["explanation"] = self._generate_multimodal_explanation(results)

            # 5. Generate recommendations
            results["recommendations"] = self._generate_multimodal_recommendations(
                results
            )

            logger.info(
                f"✅ Multimodal analysis complete: {results['label']} ({results['authenticity_score']:.3f})"
            )

            return results

        except Exception as e:
            logger.error(f"❌ Multimodal detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "label": "error",
                "confidence": 0.0,
            }

    def _analyze_cross_modal_consistency(self, individual_results: Dict) -> Dict:
        """Analyze consistency across modalities"""
        try:
            consistency_checks = []

            # Video-Audio consistency
            if "video" in individual_results and "audio" in individual_results:
                video_score = individual_results["video"].get("authenticity_score", 0.5)
                audio_score = individual_results["audio"].get("authenticity_score", 0.5)

                consistency = 1.0 - abs(video_score - audio_score)
                consistency_checks.append(
                    {
                        "modalities": ["video", "audio"],
                        "consistency": float(consistency),
                        "suspicious": consistency < 0.5,
                    }
                )

            # Video-Text consistency (if transcript available)
            if "video" in individual_results and "text" in individual_results:
                video_score = individual_results["video"].get("authenticity_score", 0.5)
                text_score = individual_results["text"].get("authenticity_score", 0.5)

                consistency = 1.0 - abs(video_score - text_score)
                consistency_checks.append(
                    {
                        "modalities": ["video", "text"],
                        "consistency": float(consistency),
                        "suspicious": consistency < 0.5,
                    }
                )

            # Audio-Text consistency
            if "audio" in individual_results and "text" in individual_results:
                audio_score = individual_results["audio"].get("authenticity_score", 0.5)
                text_score = individual_results["text"].get("authenticity_score", 0.5)

                consistency = 1.0 - abs(audio_score - text_score)
                consistency_checks.append(
                    {
                        "modalities": ["audio", "text"],
                        "consistency": float(consistency),
                        "suspicious": consistency < 0.5,
                    }
                )

            # Overall cross-modal consistency
            if consistency_checks:
                avg_consistency = np.mean(
                    [c["consistency"] for c in consistency_checks]
                )
                suspicious_count = sum(1 for c in consistency_checks if c["suspicious"])
            else:
                avg_consistency = 1.0
                suspicious_count = 0

            return {
                "average_consistency": float(avg_consistency),
                "checks": consistency_checks,
                "suspicious_pairs": suspicious_count,
                "consistent": avg_consistency > 0.6,
            }

        except Exception as e:
            logger.error(f"Cross-modal analysis failed: {e}")
            return {"average_consistency": 0.5, "consistent": True}

    def _ensemble_fusion(self, modality_scores: Dict, cross_modal: Dict) -> Dict:
        """Advanced ensemble fusion of all modalities"""
        try:
            # 1. Weighted average of modality scores
            weighted_scores = []
            weights = []

            for modality, score in modality_scores.items():
                if modality in self.fusion_weights:
                    weighted_scores.append(score * self.fusion_weights[modality])
                    weights.append(self.fusion_weights[modality])

            # Normalize weights
            total_weight = sum(weights)
            base_score = (
                sum(weighted_scores) / total_weight if total_weight > 0 else 0.5
            )

            # 2. Apply cross-modal consistency bonus/penalty
            cross_modal_consistency = cross_modal.get("average_consistency", 1.0)
            cross_modal_adjustment = (
                cross_modal_consistency - 0.5
            ) * self.cross_modal_weight

            # 3. Final fused score
            final_score = base_score + cross_modal_adjustment
            final_score = max(0.0, min(1.0, final_score))

            # 4. Calculate confidence
            # Higher confidence if:
            # - Modalities agree (low variance)
            # - Cross-modal consistency is high
            score_variance = np.var(list(modality_scores.values()))
            agreement_confidence = 1.0 - min(1.0, score_variance)
            cross_modal_confidence = cross_modal_consistency

            confidence = agreement_confidence * 0.6 + cross_modal_confidence * 0.4

            # 5. Determine label
            label = self._determine_label(final_score)

            return {
                "score": float(final_score),
                "confidence": float(confidence),
                "label": label,
                "base_score": float(base_score),
                "cross_modal_adjustment": float(cross_modal_adjustment),
                "method": "ensemble_fusion",
            }

        except Exception as e:
            logger.error(f"Ensemble fusion failed: {e}")
            return {"score": 0.5, "confidence": 0.0, "label": "unknown"}

    def _weighted_average(self, modality_scores: Dict) -> float:
        """Simple weighted average of modality scores"""
        try:
            weighted_sum = 0.0
            total_weight = 0.0

            for modality, score in modality_scores.items():
                if modality in self.fusion_weights:
                    weighted_sum += score * self.fusion_weights[modality]
                    total_weight += self.fusion_weights[modality]

            return weighted_sum / total_weight if total_weight > 0 else 0.5
        except:
            return 0.5

    def _determine_label(self, score: float) -> str:
        """Determine label from score"""
        if score >= 0.7:
            return "authentic"
        elif score <= 0.4:
            return "deepfake"
        else:
            return "suspicious"

    def _generate_multimodal_explanation(self, results: Dict) -> str:
        """Generate comprehensive multimodal explanation"""
        label = results.get("label", "unknown")
        score = results.get("authenticity_score", 0)
        confidence = results.get("confidence", 0)
        modalities = results.get("modalities_analyzed", [])

        explanations = []

        # Main verdict
        explanations.append(
            f"Multimodal analysis ({len(modalities)} modalities) indicates: {label.upper()}"
        )
        explanations.append(
            f"Overall score: {score:.2f}, Confidence: {confidence:.2f}."
        )

        # Individual modality results
        for modality in modalities:
            if modality in results.get("individual_results", {}):
                mod_result = results["individual_results"][modality]
                mod_score = mod_result.get("authenticity_score", 0)
                mod_label = mod_result.get("label", "unknown")
                explanations.append(
                    f"{modality.capitalize()}: {mod_label} ({mod_score:.2f})."
                )

        # Cross-modal consistency
        if "cross_modal_analysis" in results:
            cross_modal = results["cross_modal_analysis"]
            if not cross_modal.get("consistent", True):
                explanations.append("WARNING: Cross-modal inconsistencies detected.")

        return " ".join(explanations)

    def _generate_multimodal_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on multimodal analysis"""
        recommendations = []

        label = results.get("label", "unknown")
        modalities = results.get("modalities_analyzed", [])

        if label == "deepfake":
            recommendations.append(
                "STRONG EVIDENCE of deepfake manipulation across multiple modalities"
            )
            recommendations.append("Do not trust this content for verification")
        elif label == "suspicious":
            recommendations.append(
                "Multiple modalities show suspicious characteristics"
            )
            recommendations.append("Recommend additional verification")
        else:
            recommendations.append(
                "Content appears authentic across analyzed modalities"
            )
            recommendations.append("Always verify source and context")

        # Cross-modal warnings
        if "cross_modal_analysis" in results:
            cross_modal = results["cross_modal_analysis"]
            if cross_modal.get("suspicious_pairs", 0) > 0:
                recommendations.append(
                    f"WARNING: {cross_modal['suspicious_pairs']} cross-modal inconsistencies detected"
                )

        # Modality-specific recommendations
        if len(modalities) < 3:
            recommendations.append(
                f"Only {len(modalities)} modalities analyzed - consider analyzing more for higher confidence"
            )

        return recommendations


# Singleton instance
_multimodal_fusion_detector_instance = None


def get_multimodal_fusion_detector() -> MultimodalFusionDetector:
    """Get or create multimodal fusion detector instance"""
    global _multimodal_fusion_detector_instance
    if _multimodal_fusion_detector_instance is None:
        _multimodal_fusion_detector_instance = MultimodalFusionDetector()
    return _multimodal_fusion_detector_instance
