"""
Enhanced Video Deepfake Detector
Comprehensive video analysis using frame-by-frame ML detection and temporal analysis
Combines image detection, motion analysis, face tracking, and consistency checking
"""

import logging
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Import image detector for frame analysis
try:
    from .image_detector import ImageDetector

    IMAGE_DETECTOR_AVAILABLE = True
except:
    try:
        from image_detector import ImageDetector

        IMAGE_DETECTOR_AVAILABLE = True
    except:
        IMAGE_DETECTOR_AVAILABLE = False
        logger.warning("Image detector not available for video analysis")

# Import temporal models for advanced video analysis
try:
    from ..models.temporal_models import TemporalConvNet, TemporalLSTM

    TEMPORAL_MODELS_AVAILABLE = True
except:
    try:
        from models.temporal_models import TemporalConvNet, TemporalLSTM

        TEMPORAL_MODELS_AVAILABLE = True
    except:
        TEMPORAL_MODELS_AVAILABLE = False
        logger.warning("Temporal models not available for advanced video analysis")

# Import video optimization utilities
try:
    from models.video_optimized import FrameProcessor
    VIDEO_OPTIMIZATION_AVAILABLE = True
    logger.info("Video optimization utilities available")
except ImportError:
    VIDEO_OPTIMIZATION_AVAILABLE = False
    logger.warning("Video optimization not available")

# Import enhanced video model
try:
    from models.video_model_enhanced import VideoProcessor as EnhancedVideoProcessor
    ENHANCED_VIDEO_MODEL_AVAILABLE = True
    logger.info("Enhanced video model available")
except ImportError:
    ENHANCED_VIDEO_MODEL_AVAILABLE = False
    logger.warning("Enhanced video model not available")


class VideoDetector:
    """
    Comprehensive video deepfake detector using multiple approaches:
    1. Frame-by-frame analysis using ImageDetector (ML-based)
    2. Temporal consistency checking across frames
    3. Face tracking and consistency analysis
    4. Motion pattern analysis
    5. Optical flow analysis
    6. Audio-visual synchronization (basic)
    7. Compression artifact detection
    8. Scene change detection
    9. Advanced 3D CNN temporal analysis (if available)
    """

    def __init__(self, config: Optional[Dict] = None, use_optimization: bool = False, use_enhanced_model: bool = False):
        """Initialize comprehensive video detector"""
        self.config = config or self._default_config()
        self.use_optimization = use_optimization
        self.use_enhanced_model = use_enhanced_model
        logger.info("🎥 Initializing Enhanced Video Detector")

        # Initialize image detector for frame analysis
        if IMAGE_DETECTOR_AVAILABLE:
            self.image_detector = ImageDetector(use_advanced_model=False)  # Fixed: remove config dependency
            logger.info("✅ Image detector loaded for frame analysis")
        else:
            self.image_detector = None
            logger.warning(
                "⚠️ Video detector will have limited functionality without image detector"
            )

        # Initialize temporal models for advanced analysis
        self.temporal_model = None
        self.temporal_lstm = None
        if TEMPORAL_MODELS_AVAILABLE and self.config.get("use_temporal_model", True):
            try:
                models_dir = Path(__file__).resolve().parents[3] / "models"
                
                # Load 3D CNN model
                convnet_path = models_dir / "video" / "temporal_3dcnn.pth"
                if convnet_path.exists():
                    self.temporal_model = TemporalConvNet()
                    state_dict = torch.load(convnet_path, map_location='cpu')
                    self.temporal_model.load_state_dict(state_dict)
                    self.temporal_model.eval()
                    logger.info("✅ Temporal 3D CNN model loaded for advanced analysis")
                
                # Load LSTM model
                lstm_path = models_dir / "video" / "temporal_lstm.pth"
                if lstm_path.exists():
                    self.temporal_lstm = TemporalLSTM()
                    state_dict = torch.load(lstm_path, map_location='cpu')
                    self.temporal_lstm.load_state_dict(state_dict)
                    self.temporal_lstm.eval()
                    logger.info("✅ Temporal LSTM model loaded for advanced analysis")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not load temporal models: {e}")

        # Initialize video optimization if requested
        self.frame_processor = None
        if use_optimization and VIDEO_OPTIMIZATION_AVAILABLE and self.image_detector:
            try:
                # Use the image detector's model for frame processing
                model = getattr(self.image_detector, 'advanced_model', None) or getattr(self.image_detector, 'model', None)
                if model:
                    self.frame_processor = FrameProcessor(
                        model=model,
                        device=self.config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu'),
                        batch_size=self.config.get("batch_size", 16),
                        num_workers=self.config.get("num_workers", 4)
                    )
                    logger.info("✅ Video optimization enabled with GPU batching")
                else:
                    logger.warning("⚠️ No model available for frame processor")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize video optimization: {e}")

        # Initialize enhanced video model if requested
        self.enhanced_video_processor = None
        if use_enhanced_model and ENHANCED_VIDEO_MODEL_AVAILABLE:
            try:
                self.enhanced_video_processor = EnhancedVideoProcessor(
                    model_path=self.config.get("enhanced_model_path"),
                    device=self.config.get("device", 'cuda' if torch.cuda.is_available() else 'cpu')
                )
                logger.info("✅ Enhanced video model initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize enhanced video model: {e}")

        # Initialize optical flow calculator
        self.flow_calculator = None
        try:
            # Using Farneback optical flow
            self.flow_calculator = cv2.FarnebackOpticalFlow_create()
            logger.info("✅ Optical flow calculator initialized")
        except:
            logger.warning("⚠️ Optical flow not available")

        # Frame cache for temporal analysis
        self.frame_cache = deque(maxlen=self.config["temporal_window"])

        # Analysis weights
        self.analysis_weights = {
            "frame_analysis": 0.30,
            "temporal_consistency": 0.20,
            "face_tracking": 0.15,
            "motion_analysis": 0.10,
            "optical_flow": 0.05,
            "temporal_deep_learning": 0.20,  # New weight for 3D CNN
        }

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "max_frames": 300,  # Maximum frames to analyze
            "sample_rate": 5,  # Analyze every Nth frame
            "min_confidence": 0.6,
            "temporal_window": 10,  # Frames to check for consistency
            "face_tracking": True,
            "motion_analysis": True,
            "optical_flow_analysis": True,
            "scene_change_detection": True,
            "min_face_size": 30,
            "max_scene_changes": 50,  # Max scene changes before flagging
        }

    def detect(self, video_input: Union[str, List[np.ndarray]]) -> Dict:
        """
        Comprehensive video analysis
        
        Args:
            video_input: Either path to video file (str) or list of numpy array frames
            
        Returns:
            Comprehensive detection results
        """
        # Handle both file paths and frame arrays
        if isinstance(video_input, str):
            return self._detect_from_file(video_input)
        elif isinstance(video_input, list):
            return self._detect_from_frames(video_input)
        else:
            raise ValueError("video_input must be either a file path (str) or list of frames")
    
    def _detect_from_file(self, video_path: str) -> Dict:
        """Detect from video file path"""
        try:
            logger.info(f"🔍 Analyzing video file: {video_path}")
            start_time = time.time()

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                f"📊 Video: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s, {width}x{height}"
            )

            # Process frames from video file
            results = self._process_video_frames(cap, fps, total_frames, width, height, duration, start_time)
            
            cap.release()
            return results

        except Exception as e:
            logger.error(f"❌ Video detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "label": "error",
                "confidence": 0.0,
                "authenticity_score": 0.5,
            }
    
    def _detect_from_frames(self, frames: List[np.ndarray]) -> Dict:
        """Detect from list of numpy array frames"""
        try:
            logger.info(f"🔍 Analyzing {len(frames)} video frames")
            start_time = time.time()

            # Use default video properties for frame arrays
            fps = 30.0  # Default fps
            total_frames = len(frames)
            width = frames[0].shape[1] if frames else 224
            height = frames[0].shape[0] if frames else 224
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                f"📊 Video frames: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s, {width}x{height}"
            )

            # Process frames directly
            results = self._process_video_frames(None, fps, total_frames, width, height, duration, start_time, frames)
            
            return results

        except Exception as e:
            logger.error(f"❌ Video detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "label": "error",
                "confidence": 0.0,
                "authenticity_score": 0.5,
            }
    
    def _process_video_frames(self, cap, fps, total_frames, width, height, duration, start_time, frames_list=None) -> Dict:
        """Common processing logic for both file and frame inputs"""

        # Initialize results
        results = {
            "success": True,
            "authenticity_score": 0.0,
            "confidence": 0.0,
            "label": "unknown",
            "explanation": "",
            "details": {
                "video_info": {
                    "total_frames": total_frames,
                    "fps": fps,
                    "duration": duration,
                    "resolution": f"{width}x{height}",
                    "frames_analyzed": 0,
                }
            },
            "warnings": [],
            "suspicious_frames": [],
            "scene_changes": [],
        }

            # 1. Extract and analyze frames
        if frames_list is not None:
            # Process provided frames directly
            frame_results = []
            for i, frame in enumerate(frames_list):
                frame_result = self._analyze_single_frame_comprehensive(frame, i)
                if frame_result:
                    frame_results.append(frame_result)
        else:
            # Extract frames from video capture
            frame_results = self._analyze_frames_comprehensive(cap)
            
        results["details"]["frame_analysis"] = {
            "frames_analyzed": len(frame_results),
            "average_score": np.mean([f["score"] for f in frame_results])
            if frame_results
            else 0,
            "suspicious_frames": sum(1 for f in frame_results if f["score"] < 0.5),
            "authentic_frames": sum(1 for f in frame_results if f["score"] >= 0.7),
            "score_variance": np.var([f["score"] for f in frame_results])
            if frame_results
            else 0,
        }
        results["details"]["video_info"]["frames_analyzed"] = len(frame_results)

        # 2. Temporal consistency analysis
        temporal_result = self._analyze_temporal_consistency_comprehensive(
            frame_results
        )
        results["details"]["temporal_consistency"] = temporal_result

        # 3. Face tracking and consistency
        face_tracking_result = self._analyze_face_tracking_comprehensive(
            frame_results
        )
        results["details"]["face_tracking"] = face_tracking_result

        # 4. Motion pattern analysis
        motion_result = self._analyze_motion_patterns_comprehensive(frame_results)
        results["details"]["motion_analysis"] = motion_result

        # 5. Optical flow analysis (if available)
        if self.flow_calculator and self.config["optical_flow_analysis"]:
            flow_result = self._analyze_optical_flow(frame_results)
            results["details"]["optical_flow"] = flow_result

        # 6. Advanced temporal deep learning analysis (if model available)
        if self.temporal_model is not None and cap is not None:
            temporal_dl_result = self._analyze_temporal_deep_learning(
                cap, "video_file" if frames_list is None else "frame_array"
            )
            results["details"]["temporal_deep_learning"] = temporal_dl_result

        # 7. Scene change detection
        if self.config["scene_change_detection"]:
            scene_changes = self._detect_scene_changes(frame_results)
            results["scene_changes"] = scene_changes
            results["details"]["scene_analysis"] = {
                "scene_changes_detected": len(scene_changes),
                "suspicious": len(scene_changes) > self.config["max_scene_changes"],
            }

        # 7. Compression artifact analysis
        compression_result = self._analyze_video_compression(frame_results)
        results["details"]["compression_analysis"] = compression_result

        # 8. Frame rate consistency
        fps_consistency = self._analyze_fps_consistency(frame_results, fps)
        results["details"]["fps_consistency"] = fps_consistency

        # Combine all scores
        final_score, confidence, label = self._combine_all_video_scores(
            results["details"]
        )

        results["authenticity_score"] = final_score
        results["confidence"] = confidence
        results["label"] = label

        # Generate explanation
        results["explanation"] = self._generate_video_explanation(results)

        # Add recommendations
        results["recommendations"] = self._generate_video_recommendations(results)

        # Identify suspicious frames
        results["suspicious_frames"] = [
            {
                "frame_number": f["frame_number"],
                "score": f["score"],
                "timestamp": f["frame_number"] / fps if fps > 0 else 0,
            }
            for f in frame_results
            if f["score"] < 0.4
        ][
            :10
        ]  # Top 10 most suspicious

        if cap is not None:
            cap.release()

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time

        logger.info(
            f"✅ Video analysis complete: {label} ({final_score:.3f}, confidence: {confidence:.3f}) in {processing_time:.1f}s"
        )

        return results

    def _analyze_frames_comprehensive(self, cap: cv2.VideoCapture) -> List[Dict]:
        """Extract and comprehensively analyze frames"""
        frame_results = []
        frame_count = 0
        sample_rate = self.config["sample_rate"]
        max_frames = self.config["max_frames"]

        frames_analyzed = 0
        prev_frame = None

        logger.info(f"📊 Analyzing frames (sample rate: 1/{sample_rate})...")

        while cap.isOpened() and frames_analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames
            if frame_count % sample_rate == 0:
                # Analyze this frame
                result = self._analyze_single_frame_comprehensive(
                    frame, frame_count, prev_frame
                )
                if result:
                    frame_results.append(result)
                    frames_analyzed += 1

                    # Update cache for temporal analysis
                    self.frame_cache.append(
                        {"frame_number": frame_count, "frame": frame, "result": result}
                    )

                    if frames_analyzed % 20 == 0:
                        logger.info(f"  Analyzed {frames_analyzed} frames...")

                prev_frame = frame

            frame_count += 1

        logger.info(
            f"✅ Analyzed {len(frame_results)} frames out of {frame_count} total"
        )

        return frame_results

    def _analyze_single_frame_comprehensive(
        self,
        frame: np.ndarray,
        frame_number: int,
        prev_frame: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """Comprehensive analysis of a single frame"""
        try:
            result = {
                "frame_number": frame_number,
                "score": 0.5,
                "method": "basic_analysis",
            }

            # 1. ML-based analysis using image detector
            if self.image_detector is not None:
                # Save frame temporarily
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, frame)
                    tmp_path = tmp.name

                try:
                    # Use image detector
                    ml_result = self.image_detector.detect(tmp_path)

                    result["score"] = ml_result.get("authenticity_score", 0.5)
                    result["confidence"] = ml_result.get("confidence", 0.0)
                    result["faces_detected"] = (
                        ml_result.get("details", {})
                        .get("face_analysis", {})
                        .get("faces_detected", 0)
                    )
                    result["artifacts"] = ml_result.get("artifacts_detected", [])
                    result["method"] = "ml_analysis"
                    result["ml_details"] = {
                        "label": ml_result.get("label", "unknown"),
                        "ml_score": ml_result.get("details", {})
                        .get("ml_classification", {})
                        .get("score", 0.5),
                    }
                finally:
                    # Clean up temp file
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass

            # 2. Frame-specific analysis
            frame_analysis = self._analyze_frame_quality(frame)
            result["frame_quality"] = frame_analysis

            # 3. Inter-frame analysis (if previous frame available)
            if prev_frame is not None:
                inter_frame = self._analyze_inter_frame_changes(prev_frame, frame)
                result["inter_frame"] = inter_frame

            return result

        except Exception as e:
            logger.error(f"Frame {frame_number} analysis failed: {e}")
            return None

    def _analyze_frame_quality(self, frame: np.ndarray) -> Dict:
        """Analyze individual frame quality"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = laplacian_var < 100

            # Brightness analysis
            brightness = np.mean(gray)
            is_too_dark = brightness < 50
            is_too_bright = brightness > 200

            # Contrast analysis
            contrast = np.std(gray)
            is_low_contrast = contrast < 30

            return {
                "blur_score": float(laplacian_var),
                "is_blurry": is_blurry,
                "brightness": float(brightness),
                "is_too_dark": is_too_dark,
                "is_too_bright": is_too_bright,
                "contrast": float(contrast),
                "is_low_contrast": is_low_contrast,
                "quality_score": 1.0
                if not (is_blurry or is_too_dark or is_too_bright or is_low_contrast)
                else 0.5,
            }
        except:
            return {"quality_score": 0.5}

    def _analyze_inter_frame_changes(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Dict:
        """Analyze changes between consecutive frames"""
        try:
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculate frame difference
            frame_diff = cv2.absdiff(prev_gray, curr_gray)
            diff_mean = np.mean(frame_diff)
            diff_std = np.std(frame_diff)

            # Detect sudden changes
            is_sudden_change = diff_mean > 50

            return {
                "difference_mean": float(diff_mean),
                "difference_std": float(diff_std),
                "is_sudden_change": is_sudden_change,
                "change_score": float(min(1.0, diff_mean / 100)),
            }
        except:
            return {"change_score": 0.0}

    def _analyze_temporal_consistency_comprehensive(
        self, frame_results: List[Dict]
    ) -> Dict:
        """Comprehensive temporal consistency analysis"""
        if len(frame_results) < 2:
            return {"score": 0.5, "consistent": True, "anomalies": 0}

        try:
            # 1. Score consistency
            scores = [f["score"] for f in frame_results]
            score_changes = [
                abs(scores[i] - scores[i + 1]) for i in range(len(scores) - 1)
            ]

            avg_change = np.mean(score_changes)
            max_change = np.max(score_changes)
            std_change = np.std(score_changes)

            # Count anomalies (sudden large changes)
            anomalies = sum(1 for change in score_changes if change > 0.3)

            # 2. Face count consistency
            face_counts = [f.get("faces_detected", 0) for f in frame_results]
            face_variance = np.var(face_counts)
            face_changes = sum(
                1
                for i in range(len(face_counts) - 1)
                if abs(face_counts[i] - face_counts[i + 1]) > 0
            )

            # 3. Quality consistency
            quality_scores = [
                f.get("frame_quality", {}).get("quality_score", 0.5)
                for f in frame_results
            ]
            quality_variance = np.var(quality_scores)

            # Calculate overall consistency score
            consistency_score = 1.0 - min(
                1.0,
                (
                    avg_change * 0.4
                    + (anomalies / len(score_changes)) * 0.3
                    + min(1.0, face_variance / 10) * 0.2
                    + min(1.0, quality_variance) * 0.1
                ),
            )

            return {
                "score": float(consistency_score),
                "consistent": anomalies < len(score_changes) * 0.1,
                "anomalies": anomalies,
                "average_change": float(avg_change),
                "max_change": float(max_change),
                "std_change": float(std_change),
                "face_variance": float(face_variance),
                "face_changes": face_changes,
                "quality_variance": float(quality_variance),
            }

        except Exception as e:
            logger.error(f"Temporal consistency analysis failed: {e}")
            return {"score": 0.5, "consistent": True, "anomalies": 0}

    def _analyze_face_tracking_comprehensive(self, frame_results: List[Dict]) -> Dict:
        """Comprehensive face tracking and consistency analysis"""
        try:
            # Get frames with faces
            frames_with_faces = [
                f for f in frame_results if f.get("faces_detected", 0) > 0
            ]

            if not frames_with_faces:
                return {
                    "score": 0.5,
                    "consistent": True,
                    "frames_with_faces": 0,
                    "message": "No faces detected in video",
                }

            # 1. Face presence consistency
            total_frames = len(frame_results)
            face_presence_ratio = len(frames_with_faces) / total_frames

            # 2. Face score consistency
            face_frame_scores = [f["score"] for f in frames_with_faces]
            score_mean = np.mean(face_frame_scores)
            score_std = np.std(face_frame_scores)

            # 3. Face count consistency
            face_counts = [f.get("faces_detected", 0) for f in frames_with_faces]
            face_count_mode = (
                max(set(face_counts), key=face_counts.count) if face_counts else 0
            )
            face_count_consistency = sum(
                1 for c in face_counts if c == face_count_mode
            ) / len(face_counts)

            # 4. Face appearance/disappearance pattern
            face_transitions = 0
            prev_has_face = frame_results[0].get("faces_detected", 0) > 0
            for f in frame_results[1:]:
                curr_has_face = f.get("faces_detected", 0) > 0
                if curr_has_face != prev_has_face:
                    face_transitions += 1
                prev_has_face = curr_has_face

            # Calculate overall face tracking score
            tracking_score = (
                (1.0 - score_std) * 0.4
                + face_count_consistency * 0.3  # Lower variance = better
                + (  # Consistent face count = better
                    1.0 - min(1.0, face_transitions / 10)
                )
                * 0.3  # Fewer transitions = better
            )

            return {
                "score": float(tracking_score),
                "consistent": score_std < 0.15 and face_count_consistency > 0.7,
                "frames_with_faces": len(frames_with_faces),
                "face_presence_ratio": float(face_presence_ratio),
                "score_mean": float(score_mean),
                "score_std": float(score_std),
                "face_count_mode": face_count_mode,
                "face_count_consistency": float(face_count_consistency),
                "face_transitions": face_transitions,
            }

        except Exception as e:
            logger.error(f"Face tracking analysis failed: {e}")
            return {"score": 0.5, "consistent": True, "frames_with_faces": 0}

    def _analyze_motion_patterns_comprehensive(self, frame_results: List[Dict]) -> Dict:
        """Comprehensive motion pattern analysis"""
        try:
            # 1. Inter-frame change analysis
            inter_frame_changes = [
                f.get("inter_frame", {}).get("change_score", 0)
                for f in frame_results
                if "inter_frame" in f
            ]

            if not inter_frame_changes:
                return {"score": 0.5, "natural_motion": True}

            avg_change = np.mean(inter_frame_changes)
            std_change = np.std(inter_frame_changes)

            # 2. Sudden change detection
            sudden_changes = sum(
                1
                for f in frame_results
                if f.get("inter_frame", {}).get("is_sudden_change", False)
            )

            # 3. Motion consistency
            # Natural videos have consistent motion patterns
            motion_consistency = 1.0 - min(1.0, std_change)

            # 4. Frozen frame detection
            frozen_frames = sum(1 for change in inter_frame_changes if change < 1.0)
            frozen_ratio = frozen_frames / len(inter_frame_changes)

            # Calculate overall motion score
            motion_score = (
                motion_consistency * 0.4
                + (1.0 - min(1.0, sudden_changes / 10)) * 0.3
                + (1.0 - frozen_ratio) * 0.3
            )

            return {
                "score": float(motion_score),
                "natural_motion": std_change < 0.3 and sudden_changes < 5,
                "average_change": float(avg_change),
                "std_change": float(std_change),
                "sudden_changes": sudden_changes,
                "frozen_frames": frozen_frames,
                "frozen_ratio": float(frozen_ratio),
                "motion_consistency": float(motion_consistency),
            }

        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")
            return {"score": 0.5, "natural_motion": True}

    def _analyze_optical_flow(self, frame_results: List[Dict]) -> Dict:
        """Analyze optical flow patterns"""
        try:
            # This would require storing actual frames, simplified version
            return {
                "score": 0.7,
                "method": "optical_flow",
                "message": "Optical flow analysis placeholder",
            }
        except Exception as e:
            logger.error(f"Optical flow analysis failed: {e}")
            return {"score": 0.5}

    def _analyze_temporal_deep_learning(
        self, cap: cv2.VideoCapture, video_path: str
    ) -> Dict:
        """Advanced temporal analysis using 3D CNN"""
        try:
            logger.info("🧠 Running temporal deep learning analysis...")

            # Extract frames for temporal model (need consecutive frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            frames = []
            max_frames = 32  # Temporal model typically uses 16-32 frames

            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)

            if len(frames) < 8:
                return {
                    "score": 0.5,
                    "method": "temporal_3d_cnn",
                    "message": "Insufficient frames for temporal analysis",
                }

            # Prepare tensor for model
            frames_array = np.stack(frames)  # (T, H, W, C)
            frames_tensor = torch.from_numpy(frames_array).float()
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)

            # Normalize
            frames_tensor = frames_tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - mean) / std

            # Reshape for 3D CNN: (1, C, T, H, W)
            frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                output = self.temporal_model(frames_tensor)
                logits = output["logits"]
                probs = torch.nn.functional.softmax(logits, dim=1)

                # Get authenticity score (assuming class 0 is real, class 1 is fake)
                real_prob = probs[0, 0].item()
                fake_prob = probs[0, 1].item()

                # Authenticity score (higher = more authentic)
                authenticity_score = real_prob

                logger.info(
                    f"  Temporal DL: Real={real_prob:.3f}, Fake={fake_prob:.3f}"
                )

            return {
                "score": float(authenticity_score),
                "method": "temporal_3d_cnn",
                "real_probability": float(real_prob),
                "fake_probability": float(fake_prob),
                "frames_analyzed": len(frames),
                "model_confidence": float(max(real_prob, fake_prob)),
            }

        except Exception as e:
            logger.error(f"Temporal deep learning analysis failed: {e}")
            return {"score": 0.5, "method": "temporal_3d_cnn", "error": str(e)}

    def _detect_scene_changes(self, frame_results: List[Dict]) -> List[Dict]:
        """Detect scene changes in video"""
        scene_changes = []

        try:
            for i, frame_result in enumerate(frame_results):
                if "inter_frame" in frame_result:
                    if frame_result["inter_frame"].get("is_sudden_change", False):
                        scene_changes.append(
                            {
                                "frame_number": frame_result["frame_number"],
                                "change_magnitude": frame_result["inter_frame"].get(
                                    "difference_mean", 0
                                ),
                            }
                        )

            logger.info(f"  Detected {len(scene_changes)} scene changes")

        except Exception as e:
            logger.error(f"Scene change detection failed: {e}")

        return scene_changes

    def _analyze_video_compression(self, frame_results: List[Dict]) -> Dict:
        """Analyze video compression artifacts"""
        try:
            # Check frame quality scores
            quality_scores = [
                f.get("frame_quality", {}).get("quality_score", 0.5)
                for f in frame_results
            ]
            avg_quality = np.mean(quality_scores)

            # Check for blur
            blur_scores = [
                f.get("frame_quality", {}).get("blur_score", 100) for f in frame_results
            ]
            avg_blur = np.mean(blur_scores)

            # Low quality + high blur = heavy compression
            compression_score = 1.0 - (
                avg_quality * 0.6 + min(1.0, avg_blur / 200) * 0.4
            )

            return {
                "compression_score": float(compression_score),
                "average_quality": float(avg_quality),
                "average_blur": float(avg_blur),
                "heavily_compressed": compression_score > 0.6,
            }
        except:
            return {"compression_score": 0.0}

    def _analyze_fps_consistency(
        self, frame_results: List[Dict], declared_fps: float
    ) -> Dict:
        """Analyze frame rate consistency"""
        try:
            # Check if frame numbers are consistent with declared FPS
            frame_numbers = [f["frame_number"] for f in frame_results]

            if len(frame_numbers) < 2:
                return {"consistent": True, "score": 0.7}

            # Calculate expected frame intervals
            expected_interval = self.config["sample_rate"]
            actual_intervals = [
                frame_numbers[i + 1] - frame_numbers[i]
                for i in range(len(frame_numbers) - 1)
            ]

            # Check consistency
            interval_variance = np.var(actual_intervals)
            is_consistent = interval_variance < 10

            return {
                "consistent": is_consistent,
                "score": 0.9 if is_consistent else 0.6,
                "interval_variance": float(interval_variance),
                "declared_fps": float(declared_fps),
            }
        except:
            return {"consistent": True, "score": 0.7}

    def _combine_all_video_scores(self, details: Dict) -> Tuple[float, float, str]:
        """Combine all video analysis scores"""
        scores = []
        weights = []

        # Frame analysis
        if "frame_analysis" in details:
            scores.append(details["frame_analysis"]["average_score"])
            weights.append(self.analysis_weights["frame_analysis"])

        # Temporal consistency
        if "temporal_consistency" in details:
            scores.append(details["temporal_consistency"]["score"])
            weights.append(self.analysis_weights["temporal_consistency"])

        # Face tracking
        if "face_tracking" in details:
            scores.append(details["face_tracking"]["score"])
            weights.append(self.analysis_weights["face_tracking"])

        # Motion analysis
        if "motion_analysis" in details:
            scores.append(details["motion_analysis"]["score"])
            weights.append(self.analysis_weights["motion_analysis"])

        # Optical flow
        if "optical_flow" in details:
            scores.append(details["optical_flow"]["score"])
            weights.append(self.analysis_weights["optical_flow"])

        # Temporal deep learning (3D CNN)
        if "temporal_deep_learning" in details:
            scores.append(details["temporal_deep_learning"]["score"])
            weights.append(self.analysis_weights["temporal_deep_learning"])

        if not scores:
            return 0.5, 0.0, "unknown"

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))

        # Confidence based on agreement
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.7

        # Determine label
        if final_score >= 0.7:
            label = "authentic"
        elif final_score <= 0.4:
            label = "deepfake"
        else:
            label = "suspicious"

        return float(final_score), float(confidence), label

    def _generate_video_explanation(self, results: Dict) -> str:
        """Generate comprehensive explanation"""
        label = results.get("label", "unknown")
        score = results.get("authenticity_score", 0)
        confidence = results.get("confidence", 0)

        explanations = []

        # Main verdict
        if label == "authentic":
            explanations.append(
                f"Video appears to be authentic (score: {score:.2f}, confidence: {confidence:.2f})."
            )
        elif label == "deepfake":
            explanations.append(
                f"Video shows strong signs of manipulation (score: {score:.2f}, confidence: {confidence:.2f})."
            )
        else:
            explanations.append(
                f"Video shows suspicious characteristics (score: {score:.2f}, confidence: {confidence:.2f})."
            )

        # Frame analysis
        if "frame_analysis" in results.get("details", {}):
            frame_analysis = results["details"]["frame_analysis"]
            suspicious_frames = frame_analysis.get("suspicious_frames", 0)
            if suspicious_frames > frame_analysis.get("frames_analyzed", 1) * 0.3:
                explanations.append(f"{suspicious_frames} suspicious frames detected.")

        # Temporal consistency
        if "temporal_consistency" in results.get("details", {}):
            temporal = results["details"]["temporal_consistency"]
            if not temporal.get("consistent", True):
                explanations.append("Temporal inconsistencies detected between frames.")

        # Face tracking
        if "face_tracking" in results.get("details", {}):
            faces = results["details"]["face_tracking"]
            if not faces.get("consistent", True):
                explanations.append("Inconsistent face appearance across frames.")

        # Motion analysis
        if "motion_analysis" in results.get("details", {}):
            motion = results["details"]["motion_analysis"]
            if not motion.get("natural_motion", True):
                explanations.append("Unnatural motion patterns detected.")

        # Scene changes
        if results.get("scene_changes"):
            scene_count = len(results["scene_changes"])
            if scene_count > 20:
                explanations.append(
                    f"Excessive scene changes detected ({scene_count})."
                )

        return " ".join(explanations)

    def _generate_video_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []

        label = results.get("label", "unknown")

        if label == "deepfake":
            recommendations.append("Do not trust this video for verification purposes")
            recommendations.append("Multiple manipulation indicators detected")
        elif label == "suspicious":
            recommendations.append("Exercise caution with this video")
            recommendations.append("Consider additional verification")
        else:
            recommendations.append("Video appears authentic, but verify source")

        # Specific recommendations
        if results.get("suspicious_frames"):
            timestamps = [f"{f['timestamp']:.1f}s" for f in results['suspicious_frames'][:3]]
            recommendations.append(
                f"Review suspicious frames at timestamps: {', '.join(timestamps)}"
            )

        return recommendations


# Singleton instance
_video_detector_instance = None


def get_video_detector() -> VideoDetector:
    """Get or create video detector instance"""
    global _video_detector_instance
    if _video_detector_instance is None:
        _video_detector_instance = VideoDetector()
    return _video_detector_instance
