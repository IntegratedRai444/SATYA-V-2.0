"""Temporal analysis utilities for detecting inconsistencies in videos."""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from scipy import signal

logger = logging.getLogger(__name__)

def check_temporal_consistency(
    frames: List[np.ndarray], 
    frame_results: List[Dict],
    window_size: int = 5,
    threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Analyze temporal consistency between video frames to detect manipulations.
    
    Args:
        frames: List of video frames as numpy arrays
        frame_results: List of frame analysis results
        window_size: Size of the sliding window for analysis
        threshold: Threshold for detecting inconsistencies
        
    Returns:
        Dictionary containing temporal analysis results
    """
    if not frames or len(frames) < 2:
        return {"error": "Not enough frames for temporal analysis"}
    
    try:
        # Convert frames to grayscale and resize for faster processing
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        
        # Calculate optical flow between consecutive frames
        flow_metrics = _calculate_optical_flow_metrics(gray_frames)
        
        # Calculate frame differences
        diff_metrics = _calculate_frame_differences(gray_frames)
        
        # Analyze motion patterns
        motion_analysis = _analyze_motion_patterns(flow_metrics, window_size, threshold)
        
        # Combine results
        results = {
            "flow_metrics": flow_metrics,
            "diff_metrics": diff_metrics,
            "motion_analysis": motion_analysis,
            "inconsistencies": _detect_temporal_inconsistencies(
                flow_metrics, 
                diff_metrics,
                motion_analysis,
                threshold
            )
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        return {"error": f"Temporal analysis failed: {str(e)}"}

def _calculate_optical_flow_metrics(frames: List[np.ndarray]) -> List[Dict]:
    """Calculate optical flow metrics between consecutive frames."""
    flow_metrics = []
    prev_gray = frames[0]
    
    # Initialize with first frame (no flow)
    flow_metrics.append({
        "mean_flow": 0.0,
        "std_flow": 0.0,
        "max_flow": 0.0
    })
    
    for i in range(1, len(frames)):
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, frames[i], 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Calculate flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Calculate statistics
        mean_flow = np.mean(magnitude)
        std_flow = np.std(magnitude)
        max_flow = np.max(magnitude)
        
        flow_metrics.append({
            "mean_flow": float(mean_flow),
            "std_flow": float(std_flow),
            "max_flow": float(max_flow)
        })
        
        prev_gray = frames[i]
    
    return flow_metrics

def _calculate_frame_differences(frames: List[np.ndarray]) -> List[Dict]:
    """Calculate frame differences and related metrics."""
    diff_metrics = []
    
    # Initialize with first frame (no difference)
    diff_metrics.append({
        "mse": 0.0,
        "ssim": 1.0,
        "diff_ratio": 0.0
    })
    
    for i in range(1, len(frames)):
        # Calculate absolute difference
        diff = cv2.absdiff(frames[i-1], frames[i])
        
        # Calculate mean squared error
        mse = np.mean(diff**2)
        
        # Calculate SSIM (structural similarity)
        ssim = _calculate_ssim(frames[i-1], frames[i])
        
        # Calculate difference ratio
        diff_ratio = np.count_nonzero(diff > 10) / diff.size
        
        diff_metrics.append({
            "mse": float(mse),
            "ssim": float(ssim),
            "diff_ratio": float(diff_ratio)
        })
    
    return diff_metrics

def _calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))

def _analyze_motion_patterns(
    flow_metrics: List[Dict], 
    window_size: int,
    threshold: float
) -> Dict[str, Any]:
    """Analyze motion patterns for inconsistencies."""
    if len(flow_metrics) < window_size:
        return {"error": f"Not enough frames for window size {window_size}"}
    
    # Extract flow magnitudes
    mean_flows = [m["mean_flow"] for m in flow_metrics]
    
    # Calculate moving average and standard deviation
    moving_avg = np.convolve(mean_flows, np.ones(window_size)/window_size, mode='valid')
    
    # Find peaks in the flow (sudden changes)
    peaks, _ = signal.find_peaks(mean_flows, height=threshold)
    
    # Calculate flow derivatives
    flow_deriv = np.diff(mean_flows)
    
    return {
        "moving_average": moving_avg.tolist(),
        "peaks": [int(p) for p in peaks],
        "flow_derivative": flow_deriv.tolist(),
        "anomaly_indices": [i for i, d in enumerate(flow_deriv) if abs(d) > threshold]
    }

def _detect_temporal_inconsistencies(
    flow_metrics: List[Dict],
    diff_metrics: List[Dict],
    motion_analysis: Dict[str, Any],
    threshold: float
) -> List[Dict]:
    """Detect temporal inconsistencies based on multiple metrics."""
    inconsistencies = []
    
    # Check for sudden changes in optical flow
    for i in range(1, len(flow_metrics)):
        # Check for large changes in flow magnitude
        flow_change = abs(flow_metrics[i]["mean_flow"] - flow_metrics[i-1]["mean_flow"])
        
        # Check for large changes in frame differences
        diff_change = abs(diff_metrics[i]["mse"] - diff_metrics[i-1]["mse"])
        
        # Check SSIM drop (lower SSIM means more different)
        ssim_drop = diff_metrics[i-1]["ssim"] - diff_metrics[i]["ssim"]
        
        if (flow_change > threshold or 
            diff_change > threshold * 100 or 
            ssim_drop > threshold):
            
            inconsistencies.append({
                "frame": i,
                "type": "temporal_inconsistency",
                "confidence": min(0.99, max(flow_change, diff_change, ssim_drop)),
                "metrics": {
                    "flow_change": float(flow_change),
                    "diff_change": float(diff_change),
                    "ssim_drop": float(ssim_drop)
                }
            })
    
    return inconsistencies
