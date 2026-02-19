"""
Fusion Engine
Combines results from multiple detection modalities for unified analysis
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FusionEngine:
    """
    Fusion engine for combining multimodal deepfake detection results.
    """

    def __init__(self):
        """Initialize the fusion engine with default weights."""
        # Default weights for each modality
        self.weights = {"image": 0.35, "video": 0.40, "audio": 0.25}

        logger.info("Fusion engine initialized with weights: " + str(self.weights))

    def set_weights(self, weights: Dict[str, float]):
        """
        Set custom weights for modalities.

        Args:
            weights: Dictionary of modality weights
        """
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}
            logger.info(f"Updated fusion weights: {self.weights}")

    def aggregate_scores(self, results: Dict[str, Dict[str, Any]]) -> float:
        """
        Aggregate confidence scores from multiple modalities using weighted average.

        Args:
            results: Dictionary of modality results

        Returns:
            Aggregated confidence score (0-100)
        """
        if not results:
            return 50.0

        try:
            # Extract confidence scores
            scores = {}
            for modality, result in results.items():
                if "confidence" in result:
                    confidence = result["confidence"]
                    authenticity = result.get("authenticity", "UNCERTAIN")

                    # Convert to authenticity score (0-1, where 1 = authentic)
                    if authenticity == "AUTHENTIC MEDIA":
                        auth_score = confidence / 100.0
                    elif authenticity == "MANIPULATED MEDIA":
                        auth_score = 1.0 - (confidence / 100.0)
                    else:
                        auth_score = 0.5

                    scores[modality] = auth_score

            if not scores:
                return 50.0

            # Calculate weighted average
            weighted_sum = 0.0
            weight_sum = 0.0

            for modality, score in scores.items():
                weight = self.weights.get(modality, 0.33)
                weighted_sum += score * weight
                weight_sum += weight

            if weight_sum > 0:
                aggregated_score = weighted_sum / weight_sum
            else:
                aggregated_score = 0.5

            logger.info(
                f"Aggregated score: {aggregated_score:.3f} from {list(scores.keys())}"
            )

            return aggregated_score

        except Exception as e:
            logger.error(f"Score aggregation error: {e}")
            return 0.5

    def check_cross_modal_consistency(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Advanced cross-modal consistency analysis.

        Args:
            results: Dictionary of modality results

        Returns:
            Dictionary with various consistency metrics
        """
        if len(results) < 2:
            return {
                "overall_consistency": 1.0,
                "authenticity_agreement": 1.0,
                "confidence_alignment": 1.0,
            }

        try:
            # 1. Basic authenticity agreement
            authenticity_agreement = self._calculate_authenticity_agreement(results)

            # 2. Confidence alignment
            confidence_alignment = self._calculate_confidence_alignment(results)

            # 3. Audio-visual synchronization (if both present)
            av_sync_score = 1.0
            if "video" in results and "audio" in results:
                av_sync_score = self._check_audio_visual_sync(
                    results["video"], results["audio"]
                )

            # 4. Temporal consistency across modalities
            temporal_consistency = self._check_temporal_consistency_cross_modal(results)

            # 5. Feature-level consistency
            feature_consistency = self._check_feature_consistency(results)

            # 6. Emotion/expression consistency
            emotion_consistency = self._check_emotion_consistency(results)

            # Combine all consistency metrics
            overall_consistency = (
                authenticity_agreement * 0.25
                + confidence_alignment * 0.2
                + av_sync_score * 0.2
                + temporal_consistency * 0.15
                + feature_consistency * 0.1
                + emotion_consistency * 0.1
            )

            consistency_metrics = {
                "overall_consistency": overall_consistency,
                "authenticity_agreement": authenticity_agreement,
                "confidence_alignment": confidence_alignment,
                "audio_visual_sync": av_sync_score,
                "temporal_consistency": temporal_consistency,
                "feature_consistency": feature_consistency,
                "emotion_consistency": emotion_consistency,
            }

            logger.info(f"Cross-modal consistency metrics: {consistency_metrics}")

            return consistency_metrics

        except Exception as e:
            logger.error(f"Consistency check error: {e}")
            return {
                "overall_consistency": 0.5,
                "authenticity_agreement": 0.5,
                "confidence_alignment": 0.5,
            }

    def _calculate_authenticity_agreement(
        self, results: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate agreement in authenticity labels."""
        authenticities = []

        for modality, result in results.items():
            auth = result.get("authenticity", "UNCERTAIN")
            authenticities.append(auth)

        # Count each type
        authentic_count = sum(1 for a in authenticities if a == "AUTHENTIC MEDIA")
        manipulated_count = sum(1 for a in authenticities if a == "MANIPULATED MEDIA")
        uncertain_count = sum(1 for a in authenticities if a == "UNCERTAIN")

        total = len(authenticities)

        # Calculate agreement ratio
        max_agreement = max(authentic_count, manipulated_count, uncertain_count)
        agreement_ratio = max_agreement / total

        # Penalize uncertainty
        uncertainty_penalty = uncertain_count / total * 0.3

        return max(0.0, agreement_ratio - uncertainty_penalty)

    def _calculate_confidence_alignment(
        self, results: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate alignment of confidence scores."""
        confidences = []

        for modality, result in results.items():
            conf = result.get("confidence", 50.0)
            confidences.append(conf)

        if len(confidences) < 2:
            return 1.0

        # Calculate variance in confidence scores
        conf_std = np.std(confidences)
        conf_range = max(confidences) - min(confidences)

        # Good alignment = low variance and range
        variance_score = max(0, 1 - (conf_std / 30))
        range_score = max(0, 1 - (conf_range / 60))

        return (variance_score + range_score) / 2

    def _check_audio_visual_sync(
        self, video_result: Dict[str, Any], audio_result: Dict[str, Any]
    ) -> float:
        """Check audio-visual synchronization and consistency."""
        try:
            sync_scores = []

            # 1. Confidence correlation
            video_conf = video_result.get("confidence", 50.0)
            audio_conf = audio_result.get("confidence", 50.0)

            # Similar confidence levels suggest consistency
            conf_diff = abs(video_conf - audio_conf)
            conf_sync = max(0, 1 - (conf_diff / 50))
            sync_scores.append(("confidence_sync", conf_sync, 0.3))

            # 2. Authenticity agreement
            video_auth = video_result.get("authenticity", "UNCERTAIN")
            audio_auth = audio_result.get("authenticity", "UNCERTAIN")

            if video_auth == audio_auth:
                auth_sync = 1.0
            elif "UNCERTAIN" in [video_auth, audio_auth]:
                auth_sync = 0.6
            else:
                auth_sync = 0.2  # Conflicting

            sync_scores.append(("authenticity_sync", auth_sync, 0.4))

            # 3. Temporal alignment (if temporal data available)
            temporal_sync = self._check_temporal_alignment(video_result, audio_result)
            sync_scores.append(("temporal_sync", temporal_sync, 0.2))

            # 4. Quality consistency
            quality_sync = self._check_quality_consistency(video_result, audio_result)
            sync_scores.append(("quality_sync", quality_sync, 0.1))

            # Weighted combination
            overall_sync = sum(score * weight for _, score, weight in sync_scores)

            if os.environ.get('PYTHON_ENV') == 'development':
                logger.debug(
                    f"Audio-visual sync scores: {[(name, score) for name, score, _ in sync_scores]}"
                )

            return max(0.0, min(1.0, overall_sync))

        except Exception as e:
            logger.error(f"Audio-visual sync check error: {e}")
            return 0.5

    def _check_temporal_alignment(
        self, video_result: Dict[str, Any], audio_result: Dict[str, Any]
    ) -> float:
        """Check temporal alignment between video and audio."""
        try:
            # Get temporal analysis data
            video_analysis = video_result.get("video_analysis", {})
            audio_analysis = audio_result.get("audio_analysis", {})

            video_duration = video_analysis.get("video_duration_seconds", 0)
            audio_duration = audio_analysis.get("audio_duration_seconds", 0)

            if video_duration > 0 and audio_duration > 0:
                # Duration alignment
                duration_diff = abs(video_duration - audio_duration)
                max_duration = max(video_duration, audio_duration)

                if max_duration > 0:
                    duration_alignment = max(0, 1 - (duration_diff / max_duration))
                    return duration_alignment

            return 0.8  # Default if no temporal data

        except Exception as e:
            logger.warning(f"Temporal alignment check error: {e}")
            return 0.8

    def _check_quality_consistency(
        self, video_result: Dict[str, Any], audio_result: Dict[str, Any]
    ) -> float:
        """Check consistency in quality metrics."""
        try:
            # Get quality indicators
            video_analysis = video_result.get("video_analysis", {})
            audio_analysis = audio_result.get("audio_analysis", {})

            # Video quality indicators
            video_temporal_consistency = video_analysis.get("temporal_consistency", 0.8)
            video_motion_smoothness = video_analysis.get("motion_smoothness", 0.8)

            # Audio quality indicators
            audio_spectrogram_quality = audio_analysis.get("spectrogram_quality", 0.8)
            audio_voice_pattern_score = audio_analysis.get("voice_pattern_score", 0.8)

            # Calculate average quality for each modality
            video_quality = (video_temporal_consistency + video_motion_smoothness) / 2
            audio_quality = (audio_spectrogram_quality + audio_voice_pattern_score) / 2

            # Quality consistency
            quality_diff = abs(video_quality - audio_quality)
            quality_consistency = max(0, 1 - (quality_diff / 0.5))

            return quality_consistency

        except Exception as e:
            logger.warning(f"Quality consistency check error: {e}")
            return 0.8

    def _check_temporal_consistency_cross_modal(
        self, results: Dict[str, Dict[str, Any]]
    ) -> float:
        """Check temporal consistency across all modalities."""
        try:
            temporal_scores = []

            for modality, result in results.items():
                if modality == "video":
                    video_analysis = result.get("video_analysis", {})
                    temporal_score = video_analysis.get("temporal_consistency", 0.8)
                    temporal_scores.append(temporal_score)
                elif modality == "audio":
                    # Audio doesn't have direct temporal consistency, use quality as proxy
                    audio_analysis = result.get("audio_analysis", {})
                    quality_score = audio_analysis.get("spectrogram_quality", 0.8)
                    temporal_scores.append(quality_score)
                elif modality == "image":
                    # Images don't have temporal aspect, use confidence as proxy
                    confidence = result.get("confidence", 50.0) / 100.0
                    temporal_scores.append(confidence)

            if temporal_scores:
                # Low variance in temporal scores indicates consistency
                temporal_std = np.std(temporal_scores)
                consistency = max(0, 1 - (temporal_std / 0.3))
                return consistency

            return 0.8

        except Exception as e:
            logger.warning(f"Cross-modal temporal consistency error: {e}")
            return 0.8

    def _check_feature_consistency(self, results: Dict[str, Dict[str, Any]]) -> float:
        """Check consistency at feature level across modalities."""
        try:
            # This is a simplified approach - in practice, you'd compare
            # extracted features from different modalities

            consistency_indicators = []

            # Check if detection patterns are consistent
            for modality, result in results.items():
                metrics = result.get("metrics", {})

                if modality == "image":
                    faces_detected = metrics.get("faces_detected", 0)
                    if faces_detected > 0:
                        consistency_indicators.append(0.8)
                    else:
                        consistency_indicators.append(0.4)

                elif modality == "video":
                    frames_analyzed = metrics.get("frames_analyzed", 0)
                    if frames_analyzed > 10:
                        consistency_indicators.append(0.8)
                    else:
                        consistency_indicators.append(0.6)

                elif modality == "audio":
                    audio_duration = metrics.get("audio_duration", 0)
                    if audio_duration > 1.0:
                        consistency_indicators.append(0.8)
                    else:
                        consistency_indicators.append(0.6)

            if consistency_indicators:
                return np.mean(consistency_indicators)

            return 0.7

        except Exception as e:
            logger.warning(f"Feature consistency check error: {e}")
            return 0.7

    def _check_emotion_consistency(self, results: Dict[str, Dict[str, Any]]) -> float:
        """Check emotional consistency between face and voice."""
        try:
            # Extract emotional features from different modalities
            emotion_features = {}
            
            # Extract video emotion features
            if "video" in results:
                video_result = results["video"]
                video_features = video_result.get("features", {})
                
                # Look for emotion-related features
                emotion_features["video"] = {
                    "facial_expressions": video_features.get("facial_expressions", {}),
                    "micro_expressions": video_features.get("micro_expressions", {}),
                    "confidence": video_result.get("confidence", 0.5),
                    "arousal": video_features.get("arousal", 0.5),
                    "valence": video_features.get("valence", 0.5)
                }
            
            # Extract audio emotion features
            if "audio" in results:
                audio_result = results["audio"]
                audio_features = audio_result.get("features", {})
                
                # Look for emotion-related features
                emotion_features["audio"] = {
                    "prosody": audio_features.get("prosody", {}),
                    "pitch_variation": audio_features.get("pitch_variation", 0.5),
                    "speaking_rate": audio_features.get("speaking_rate", 0.5),
                    "voice_quality": audio_features.get("voice_quality", {}),
                    "confidence": audio_result.get("confidence", 0.5),
                    "arousal": audio_features.get("arousal", 0.5),
                    "valence": audio_features.get("valence", 0.5)
                }
            
            # Extract image emotion features (if available)
            if "image" in results:
                image_result = results["image"]
                image_features = image_result.get("features", {})
                
                emotion_features["image"] = {
                    "facial_expressions": image_features.get("facial_expressions", {}),
                    "confidence": image_result.get("confidence", 0.5),
                    "arousal": image_features.get("arousal", 0.5),
                    "valence": image_features.get("valence", 0.5)
                }
            
            # Calculate emotion consistency scores
            consistency_scores = []
            
            # Compare arousal levels between modalities
            arousal_values = []
            for modality, features in emotion_features.items():
                if "arousal" in features:
                    arousal_values.append(features["arousal"])
            
            if len(arousal_values) >= 2:
                arousal_std = np.std(arousal_values)
                arousal_consistency = max(0, 1 - arousal_std)  # Lower std = more consistent
                consistency_scores.append(arousal_consistency)
            
            # Compare valence levels between modalities
            valence_values = []
            for modality, features in emotion_features.items():
                if "valence" in features:
                    valence_values.append(features["valence"])
            
            if len(valence_values) >= 2:
                valence_std = np.std(valence_values)
                valence_consistency = max(0, 1 - valence_std)  # Lower std = more consistent
                consistency_scores.append(valence_consistency)
            
            # Compare confidence patterns
            confidences = []
            for modality, features in emotion_features.items():
                if "confidence" in features:
                    confidences.append(features["confidence"])
            
            if len(confidences) >= 2:
                conf_std = np.std(confidences)
                confidence_consistency = max(0, 1 - (conf_std / 0.5))  # Normalize by expected std
                consistency_scores.append(confidence_consistency)
            
            # Analyze facial expression vs voice prosody consistency
            if "video" in emotion_features and "audio" in emotion_features:
                video_emotions = emotion_features["video"].get("facial_expressions", {})
                audio_prosody = emotion_features["audio"].get("prosody", {})
                
                # Check if high facial expression intensity matches prosodic intensity
                video_intensity = np.mean(list(video_emotions.values())) if video_emotions else 0.5
                audio_intensity = np.mean(list(audio_prosody.values())) if audio_prosody else 0.5
                
                intensity_diff = abs(video_intensity - audio_intensity)
                intensity_consistency = max(0, 1 - intensity_diff)
                consistency_scores.append(intensity_consistency)
            
            # Calculate overall emotion consistency
            if consistency_scores:
                overall_consistency = np.mean(consistency_scores)
                
                # Apply weighting based on number of modalities
                modality_count = len(emotion_features)
                if modality_count >= 3:
                    # More modalities = higher confidence in consistency score
                    weighted_consistency = overall_consistency * 1.1
                elif modality_count == 2:
                    weighted_consistency = overall_consistency
                else:
                    weighted_consistency = overall_consistency * 0.9
                
                return min(1.0, weighted_consistency)
            
            # Fallback to confidence-based consistency if no emotion features
            if len(confidences) >= 2:
                conf_std = np.std(confidences)
                return max(0, 1 - (conf_std / 25))
            
            return 0.8  # Default if insufficient data

        except Exception as e:
            logger.warning(f"Emotion consistency check error: {e}")
            return 0.8

    def detect_conflicts(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Detect conflicts between modalities.

        Args:
            results: Dictionary of modality results

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        try:
            # Check for authenticity conflicts
            authenticities = {
                mod: res.get("authenticity", "UNCERTAIN")
                for mod, res in results.items()
            }

            # Check if we have both authentic and manipulated verdicts
            has_authentic = any(a == "AUTHENTIC MEDIA" for a in authenticities.values())
            has_manipulated = any(
                a == "MANIPULATED MEDIA" for a in authenticities.values()
            )

            if has_authentic and has_manipulated:
                conflicts.append("Conflicting authenticity verdicts across modalities")

                # Identify which modalities conflict
                authentic_mods = [
                    m for m, a in authenticities.items() if a == "AUTHENTIC MEDIA"
                ]
                manipulated_mods = [
                    m for m, a in authenticities.items() if a == "MANIPULATED MEDIA"
                ]

                conflicts.append(f"Authentic: {', '.join(authentic_mods)}")
                conflicts.append(f"Manipulated: {', '.join(manipulated_mods)}")

            # Check for audio-visual sync issues (if both video and audio present)
            if "video" in results and "audio" in results:
                video_conf = results["video"].get("confidence", 50)
                audio_conf = results["audio"].get("confidence", 50)

                if abs(video_conf - audio_conf) > 30:
                    conflicts.append(
                        "Significant confidence mismatch between video and audio"
                    )

            # Check for high uncertainty
            uncertain_mods = [m for m, a in authenticities.items() if a == "UNCERTAIN"]
            if len(uncertain_mods) > 0:
                conflicts.append(f"Uncertain results from: {', '.join(uncertain_mods)}")

        except Exception as e:
            logger.error(f"Conflict detection error: {e}")

        return conflicts

    def calculate_unified_score(
        self, aggregated_score: float, consistency_score: float
    ) -> Tuple[float, str]:
        """
        Calculate final unified score with consistency adjustment.

        Args:
            aggregated_score: Aggregated authenticity score (0-1)
            consistency_score: Cross-modal consistency score (0-1)

        Returns:
            Tuple of (confidence, authenticity_label)
        """
        # Apply consistency factor
        # High consistency boosts confidence, low consistency reduces it
        consistency_factor = 0.8 + (consistency_score * 0.4)  # Range: 0.8 to 1.2

        adjusted_score = aggregated_score * consistency_factor
        adjusted_score = max(0.0, min(1.0, adjusted_score))

        # Determine authenticity label
        if adjusted_score >= 0.5:
            authenticity = "AUTHENTIC MEDIA"
            confidence = adjusted_score * 100
        else:
            authenticity = "MANIPULATED MEDIA"
            confidence = (1 - adjusted_score) * 100

        logger.info(f"Unified score: {confidence:.1f}% ({authenticity})")

        return confidence, authenticity

    def fuse(self, results: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Fuse results from multiple modalities into a unified analysis.

        Args:
            results: Dictionary of modality results (keys: 'image', 'video', 'audio')
            **kwargs: Additional parameters

        Returns:
            Unified analysis result dictionary
        """
        try:
            logger.info(
                f"Fusing results from {len(results)} modalities: {list(results.keys())}"
            )

            # Aggregate scores
            aggregated_score = self.aggregate_scores(results)

            # Check cross-modal consistency
            consistency_score = self.check_cross_modal_consistency(results)

            # Detect conflicts
            conflicts = self.detect_conflicts(results)

            # Calculate unified score
            final_confidence, final_authenticity = self.calculate_unified_score(
                aggregated_score, consistency_score
            )

            # Collect key findings from all modalities
            all_findings = []
            for modality, result in results.items():
                findings = result.get("key_findings", [])
                all_findings.extend([f"[{modality.upper()}] {f}" for f in findings[:2]])

            # Add fusion-specific findings
            if consistency_score > 0.8:
                all_findings.append("High cross-modal consistency detected")
            elif consistency_score < 0.5:
                all_findings.append("Low cross-modal consistency - conflicting signals")

            if conflicts:
                all_findings.extend(conflicts[:2])

            # Build unified result
            unified_result = {
                "success": True,
                "authenticity": final_authenticity,
                "confidence": final_confidence,
                "analysis_date": datetime.now().isoformat(),
                "case_id": f"multimodal-{int(datetime.now().timestamp())}",
                "key_findings": all_findings,
                "multimodal_analysis": {
                    "modalities_analyzed": list(results.keys()),
                    "cross_modal_consistency": consistency_score,
                    "aggregated_score": aggregated_score,
                    "conflicts_detected": len(conflicts) > 0,
                    "conflicts": conflicts,
                },
                "individual_results": {
                    modality: {
                        "authenticity": res.get("authenticity"),
                        "confidence": res.get("confidence"),
                        "key_findings": res.get("key_findings", [])[:2],
                    }
                    for modality, res in results.items()
                },
                "technical_details": {
                    "fusion_method": "weighted_aggregation",
                    "weights_used": self.weights,
                    "consistency_factor": consistency_score,
                },
            }

            # Add modality-specific metrics
            for modality, result in results.items():
                if "metrics" in result:
                    unified_result[f"{modality}_metrics"] = result["metrics"]

                if "video_analysis" in result:
                    unified_result["video_analysis"] = result["video_analysis"]

                if "audio_analysis" in result:
                    unified_result["audio_analysis"] = result["audio_analysis"]

            return unified_result

        except Exception as e:
            logger.error(f"Fusion error: {e}", exc_info=True)
            return {
                "success": False,
                "authenticity": "ANALYSIS FAILED",
                "confidence": 0.0,
                "analysis_date": datetime.now().isoformat(),
                "key_findings": [f"Fusion failed: {str(e)}"],
                "error": str(e),
            }
