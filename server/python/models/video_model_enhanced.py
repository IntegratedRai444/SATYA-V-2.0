"""
Enhanced Video Model for Deepfake Detection
Implements temporal analysis and frame processing for video deepfake detection.
"""
import numpy as np
import torch
import cv2
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging

# Import temporal models
from .temporal_models import VideoDeepfakeDetector, load_video_model
from .image_model import predict_deepfake  # For frame-level analysis

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing and analysis for deepfake detection."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """Initialize video processor with optional model path."""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.frame_model = None  # For frame-level analysis
        self._initialize_models(model_path)
    
    def _initialize_models(self, model_path: Optional[str]):
        """Initialize video and frame models."""
        try:
            # Initialize video model if path is provided
            if model_path and os.path.exists(model_path):
                self.model = load_video_model(model_path, self.device)
                logger.info(f"Loaded video model from {model_path}")
            else:
                logger.warning("No video model loaded, using frame-based analysis only")
                
            # Initialize frame model for fallback
            self.frame_model = self._load_frame_model()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def _load_frame_model(self):
        """Load frame-level model for fallback analysis."""
        # This would be replaced with actual frame model loading
        return None
    
    def extract_frames(
        self, 
        video_path: str, 
        target_frames: int = 32,
        target_size: Tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """Extract frames from video with temporal sampling."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame indices to sample
            if total_frames <= target_frames:
                # Not enough frames, use all available
                frame_indices = range(total_frames)
            else:
                # Sample frames evenly
                frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, target_size)
                    frames.append(frame)
            
            if not frames:
                raise ValueError("No frames could be read from the video")
                
            return np.stack(frames)
            
        finally:
            cap.release()
    
    def preprocess_frames(
        self, 
        frames: np.ndarray, 
        normalize: bool = True
    ) -> torch.Tensor:
        """Preprocess frames for model input."""
        # Convert to tensor and normalize
        frames_tensor = torch.from_numpy(frames).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        if normalize:
            # Normalize to [0, 1] then apply ImageNet stats
            frames_tensor = frames_tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_tensor = (frames_tensor - mean) / std
        
        return frames_tensor.unsqueeze(0).to(self.device)  # Add batch dim
    
    def analyze_video(
        self, 
        video_path: str,
        use_temporal: bool = True,
        frame_analysis: bool = True,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze video for deepfake detection.
        
        Args:
            video_path: Path to video file
            use_temporal: Whether to use temporal analysis (if model available)
            frame_analysis: Whether to perform frame-level analysis
            confidence_threshold: Minimum confidence score to consider a detection
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Extract and preprocess frames
            frames = self.extract_frames(video_path)
            frames_tensor = self.preprocess_frames(frames)
            
            results = {
                'video_path': video_path,
                'num_frames': len(frames),
                'frame_predictions': [],
                'temporal_analysis': None,
                'final_prediction': None,
                'confidence': 0.0,
                'is_deepfake': False
            }
            
            # Perform temporal analysis if model is available and requested
            if use_temporal and self.model is not None:
                try:
                    with torch.no_grad():
                        output = self.model(frames_tensor)
                        probs = F.softmax(output['logits'], dim=1)
                        confidence, pred = torch.max(probs, dim=1)
                        
                        results.update({
                            'temporal_analysis': {
                                'prediction': 'Fake' if pred.item() == 1 else 'Real',
                                'confidence': confidence.item(),
                                'attention_weights': output['attention_weights'].cpu().numpy().tolist()
                            },
                            'final_prediction': 'Fake' if pred.item() == 1 else 'Real',
                            'confidence': confidence.item(),
                            'is_deepfake': pred.item() == 1 and confidence.item() >= confidence_threshold
                        })
                except Exception as e:
                    logger.error(f"Error in temporal analysis: {e}")
                    if not frame_analysis:
                        raise
            
            # Fall back to frame-level analysis if temporal analysis failed or not requested
            if frame_analysis and (not use_temporal or self.model is None or 'temporal_analysis' not in results):
                frame_results = []
                for i, frame in enumerate(frames):
                    # Use the frame-based model for prediction
                    label, confidence, _ = predict_deepfake(frame)
                    frame_results.append({
                        'frame': i,
                        'prediction': label,
                        'confidence': confidence
                    })
                
                # Aggregate frame results
                fake_confidence = sum(
                    r['confidence'] for r in frame_results 
                    if r['prediction'].lower() == 'fake'
                ) / len(frame_results) if frame_results else 0.0
                
                results.update({
                    'frame_predictions': frame_results,
                    'final_prediction': 'Fake' if fake_confidence >= 0.5 else 'Real',
                    'confidence': max(fake_confidence, 1 - fake_confidence),
                    'is_deepfake': fake_confidence >= confidence_threshold
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing video {video_path}: {e}")
            raise

def predict_video_deepfake(
    video_path: str,
    model_path: Optional[str] = None,
    use_temporal: bool = True,
    frame_analysis: bool = True,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    High-level function to predict if a video is a deepfake.
    
    Args:
        video_path: Path to video file
        model_path: Optional path to temporal model weights
        use_temporal: Whether to use temporal analysis
        frame_analysis: Whether to perform frame-level analysis
        confidence_threshold: Minimum confidence score
        
    Returns:
        Dictionary with prediction results
    """
    processor = VideoProcessor(model_path)
    return processor.analyze_video(
        video_path, 
        use_temporal=use_temporal,
        frame_analysis=frame_analysis,
        confidence_threshold=confidence_threshold
    )
