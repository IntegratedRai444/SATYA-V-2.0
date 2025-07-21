from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
from deepface import DeepFace
import torch
import torchvision
from torchvision.transforms import functional as F

router = APIRouter()

# Load torchvision face detection model once
face_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
face_model.eval()

@router.post("/analyze/face")
async def analyze_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    try:
        result = DeepFace.analyze(img, actions=['age', 'gender', 'emotion', 'race'])
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepFace error: {str(e)}")

@router.post("/detect/face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb)
    with torch.no_grad():
        prediction = face_model([img_tensor])[0]
    faces = []
    for box, score in zip(prediction['boxes'], prediction['scores']):
        if score > 0.8:
            faces.append({"box": [float(x) for x in box.numpy()], "score": float(score)})
    return {"success": True, "faces": faces} 