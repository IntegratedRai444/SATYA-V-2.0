"""
Multimodal Analysis Route
Dedicated route for combined image, video, and audio processing
"""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger(__name__)

# Import unified detector for consistent interface
try:
    from detectors.unified_detector import get_unified_detector, ModalityType
    UNIFIED_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unified detector not available: {e}")
    UNIFIED_DETECTOR_AVAILABLE = False

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads/multimodal")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Fusion weights
FUSION_WEIGHTS = {
    "image": 0.4,
    "video": 0.35,
    "audio": 0.25
}

def fuse_multimodal_results(image_res=None, video_res=None, audio_res=None):
    """
    Fuse multimodal analysis results using weighted ensemble strategy.
    
    Args:
        image_res: Image analysis result
        video_res: Video analysis result  
        audio_res: Audio analysis result
        
    Returns:
        Dict containing fused analysis results
    """
    results = {}
    weights_used = []
    confidences = []
    individual_results = {}
    
    # Process image result
    if image_res and image_res.get("success"):
        confidence = image_res.get("confidence", 0.5)
        results["image"] = {
            "is_fake": confidence < 0.5,
            "confidence": confidence,
            "model": image_res.get("model", "Unknown"),
            "details": image_res.get("details", {})
        }
        weights_used.append(FUSION_WEIGHTS["image"])
        confidences.append(confidence)
        individual_results["image"] = results["image"]
        
    # Process video result
    if video_res and video_res.get("success"):
        confidence = video_res.get("confidence", 0.5)
        results["video"] = {
            "is_fake": confidence < 0.5,
            "confidence": confidence,
            "model": video_res.get("model", "Unknown"),
            "details": video_res.get("details", {})
        }
        weights_used.append(FUSION_WEIGHTS["video"])
        confidences.append(confidence)
        individual_results["video"] = results["video"]
        
    # Process audio result
    if audio_res and audio_res.get("success"):
        confidence = audio_res.get("confidence", 0.5)
        results["audio"] = {
            "is_fake": confidence < 0.5,
            "confidence": confidence,
            "model": audio_res.get("model", "Unknown"),
            "details": audio_res.get("details", {})
        }
        weights_used.append(FUSION_WEIGHTS["audio"])
        confidences.append(confidence)
        individual_results["audio"] = results["audio"]
    
    if not results:
        return {
            "fused_score": 0.5,
            "is_fake": False,
            "consistency_score": 0.0,
            "conflicts": [],
            "modalities_used": [],
            "individual_results": {},
            "key_findings": ["No valid analysis results"]
        }
    
    # Normalize weights
    total_weight = sum(weights_used)
    if total_weight == 0:
        normalized_weights = [1/len(weights_used)] * len(weights_used) if weights_used else []
    else:
        normalized_weights = [w/total_weight for w in weights_used]
    
    # Calculate fused score (weighted average)
    fused_score = sum(conf * weight for conf, weight in zip(confidences, normalized_weights))
    fused_score = max(0.0, min(1.0, fused_score))  # Ensure 0-1 range
    
    # Determine authenticity
    is_fake_final = fused_score < 0.5
    
    # Calculate consistency score
    if len(confidences) > 1:
        consistency_score = 1.0 - (max(confidences) - min(confidences))
        consistency_score = max(0.0, min(1.0, consistency_score))
    else:
        consistency_score = 1.0
    
    # Detect conflicts
    conflicts = []
    for modality, result in results.items():
        diff = abs(result["confidence"] - fused_score)
        if diff > 0.35:
            conflicts.append(f"{modality.capitalize()} result strongly disagrees with fused analysis")
    
    # Generate key findings
    key_findings = []
    if is_fake_final:
        key_findings.append("Cross-modal analysis indicates manipulated media")
    else:
        key_findings.append("Cross-modal analysis indicates authentic media")
        
    if consistency_score > 0.8:
        key_findings.append("High cross-modal confidence agreement")
    elif consistency_score < 0.5:
        key_findings.append("Low cross-modal consistency detected")
        
    if conflicts:
        key_findings.append("Modality conflicts detected - requires further investigation")
    
    # Add modality-specific findings
    if "image" in results and results["image"]["is_fake"]:
        key_findings.append("Visual artifacts detected in image analysis")
    if "video" in results and results["video"]["is_fake"]:
        key_findings.append("Temporal inconsistencies detected in video")
    if "audio" in results and results["audio"]["is_fake"]:
        key_findings.append("Audio anomalies detected in voiceprint analysis")
    
    return {
        "fused_score": fused_score,
        "is_fake": is_fake_final,
        "consistency_score": consistency_score,
        "conflicts": conflicts,
        "modalities_used": list(results.keys()),
        "individual_results": individual_results,
        "key_findings": key_findings
    }


@router.post("/")
async def analyze_multimodal(
    request: Request,
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    options: Optional[str] = Form(None),
):
    """
    Analyze multiple media types together
    """
    try:
        files_processed = []

        # Process image if provided
        if image:
            content = await image.read()
            filename = f"{uuid.uuid4()}_{image.filename}"
            path = UPLOAD_DIR / filename
            with path.open("wb") as f:
                f.write(content)
            files_processed.append({"type": "image", "path": str(path)})

        # Process video if provided
        if video:
            content = await video.read()
            filename = f"{uuid.uuid4()}_{video.filename}"
            path = UPLOAD_DIR / filename
            with path.open("wb") as f:
                f.write(content)
            files_processed.append({"type": "video", "path": str(path)})

        # Process audio if provided
        if audio:
            content = await audio.read()
            filename = f"{uuid.uuid4()}_{audio.filename}"
            path = UPLOAD_DIR / filename
            with path.open("wb") as f:
                f.write(content)
            files_processed.append({"type": "audio", "path": str(path)})

        if not files_processed:
            raise HTTPException(status_code=400, detail="No files provided")

        # Run individual detectors for each modality
        image_result = None
        video_result = None
        audio_result = None
        
        # Initialize unified detector if available
        detector = None
        if UNIFIED_DETECTOR_AVAILABLE:
            try:
                config = {
                    "MODEL_PATH": "models",
                    "ENABLE_GPU": torch.cuda.is_available(),  # Auto-enable GPU if available
                    "ENABLE_FORENSICS": True,
                    "ENABLE_MULTIMODAL": True
                }
                detector = get_unified_detector(config)
                logger.info("✅ Unified detector initialized for multimodal analysis")
            except Exception as e:
                logger.error(f"❌ Failed to initialize unified detector: {e}")
                detector = None
        
        # Analyze image if provided
        if image and detector:
            try:
                image_path = next(f["path"] for f in files_processed if f["type"] == "image")
                with open(image_path, "rb") as f:
                    image_content = f.read()
                
                logger.info(f"🔍 Analyzing image: {image.filename}")
                result = detector.detect_image(image_content, analyze_forensics=True)
                image_result = result.to_dict()
                logger.info(f"✅ Image analysis complete: {result.authenticity} ({result.confidence:.2f})")
            except Exception as e:
                logger.error(f"❌ Image analysis failed: {e}")
                image_result = {"success": False, "error": str(e)}
        
        # Analyze video if provided
        if video and detector:
            try:
                video_path = next(f["path"] for f in files_processed if f["type"] == "video")
                with open(video_path, "rb") as f:
                    video_content = f.read()
                
                logger.info(f"🔍 Analyzing video: {video.filename}")
                result = detector.detect_video(video_content)
                video_result = result.to_dict()
                logger.info(f"✅ Video analysis complete: {result.authenticity} ({result.confidence:.2f})")
            except Exception as e:
                logger.error(f"❌ Video analysis failed: {e}")
                video_result = {"success": False, "error": str(e)}
        
        # Analyze audio if provided
        if audio and detector:
            try:
                audio_path = next(f["path"] for f in files_processed if f["type"] == "audio")
                with open(audio_path, "rb") as f:
                    audio_content = f.read()
                
                logger.info(f"🔍 Analyzing audio: {audio.filename}")
                result = detector.detect_audio(audio_content)
                audio_result = result.to_dict()
                logger.info(f"✅ Audio analysis complete: {result.authenticity} ({result.confidence:.2f})")
            except Exception as e:
                logger.error(f"❌ Audio analysis failed: {e}")
                audio_result = {"success": False, "error": str(e)}
        
        # Perform multimodal fusion
        logger.info("🔀 Performing multimodal fusion analysis")
        fusion = fuse_multimodal_results(image_result, video_result, audio_result)
        
        logger.info(f"✅ Multimodal fusion complete: {'FAKE' if fusion['is_fake'] else 'AUTHENTICIC'} ({fusion['fused_score']:.2f})")
        
        return {
            "success": True,
            "jobId": str(uuid.uuid4()),
            "result": {
                "authenticity": "FAKE MEDIA" if fusion["is_fake"] else "AUTHENTIC MEDIA",
                "confidence": fusion["fused_score"],
                "analysisDate": datetime.utcnow().isoformat(),
                "caseId": f"CASE-{uuid.uuid4().hex[:8].upper()}",
                "keyFindings": fusion["key_findings"],
                "fusionAnalysis": {
                    "aggregatedScore": fusion["fused_score"],
                    "consistencyScore": fusion["consistency_score"],
                    "confidenceLevel": "high" if fusion["fused_score"] > 0.75 else "medium" if fusion["fused_score"] > 0.5 else "low",
                    "conflictsDetected": fusion["conflicts"],
                    "modalitiesUsed": fusion["modalities_used"]
                },
                "individualResults": fusion["individual_results"]
            }
        }

    except Exception as e:
        logger.error(f"Multimodal analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
