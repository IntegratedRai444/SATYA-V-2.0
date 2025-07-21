# SatyaAI Deepfake Detection Platform

## Tech Stack Overview

### Frontend
- **Framework:** React (with TypeScript)
- **Styling:** TailwindCSS, Shadcn/ui, custom CSS
- **State/Data:** React Query, custom hooks, Context API
- **Routing:** Wouter or React Router
- **UI/UX:** Modern dashboard, animated hero, drag-and-drop upload, responsive/mobile-first, accessibility (a11y)
- **Visualization:** Chart.js, custom analytics cards
- **Notifications:** Toasts, alerts, skeleton loaders

### Backend
- **Framework:** Python (Flask)
- **AI/ML:**
  - Custom and open-source deepfake detection models (FaceForensics++, DFDC, Wav2Lip, Resemblyzer, SatyaAI custom models)
  - Ensemble analysis (combining multiple models/APIs)
- **Database:** MongoDB (for detections, analytics, user data)
- **Authentication:** JWT/session-based, SQLite for sessions (optionally PostgreSQL)
- **Task Queue (optional):** Celery + Redis (for background/batch processing)
- **External APIs:** Hive, Sensity, Reality Defender, Azure, etc.
- **File Storage:** Local, S3, or MinIO (for large media files)
- **DevOps:** Docker, GitHub Actions, Vercel/Netlify (frontend), Render/Heroku/AWS (backend)

### AI/ML Model Files & Services
- **Open-Source & Custom Models:**
  - FaceForensics++: CNN for image deepfake detection
  - DFDC Model: EfficientNet for video deepfake detection
  - Wav2Lip/Resemblyzer: Audio-visual and voice deepfake detection
  - SatyaAI Vision/Temporal/Audio/Fusion: Custom models for image, video, audio, and multimodal analysis
  - Ensemble Analyzer: Aggregates results from all models and APIs for a final, highly accurate score
- **External AI APIs:** Hive Moderation, Sensity AI, Reality Defender, Azure Video Indexer
- **Model Management:** Hot-reload, health checks, versioning, weighted voting, confidence calibration, robust aggregation

### Backend API Endpoints
- `/analyze/image`, `/analyze/video`, `/analyze/audio`, `/analyze/webcam`: Accepts media, runs detection, returns results
- `/analyze/ensemble`: Runs all models/APIs and aggregates results
- `/api/analytics`: Returns real-time global stats (total scans, avg. confidence, trends)
- `/api/detections/history`: Returns user’s recent detection history
- `/models/info`: Returns model names, versions, and features
- `/status`: Returns server health, uptime, and active sessions
- `/check/darkweb`: Checks media hash against dark web sources

### Infrastructure & DevOps
- Dockerized for easy deployment
- CI/CD with GitHub Actions
- Environment management with .env files
- Cloud storage for large media (S3/MinIO)
- Monitoring/logging (Prometheus, ELK, etc.)
- Scalable architecture (microservices-ready, load balancing)

### AI/ML File Structure (Typical)
- `server/python/opensource_models.py` — Integrates open-source models (FaceForensics++, DFDC, Wav2Lip, Resemblyzer)
- `server/python/deepfake_analyzer.py` — SatyaAI custom models for image, video, audio, multimodal
- `server/python/ensemble_analyzer.py` — Ensemble logic for combining all models/APIs
- `server/python/api_integrations.py` — External API integrations
- `server/python/models.py` — Model classes, fusion logic, and postprocessing
- `server/python/animations.py` — (Optional) For animated feedback/results
- `server/python/db_mongo.py` — MongoDB connection and analytics aggregation

---

**For a visual architecture diagram or more detailed breakdown, see the docs or request a diagram.**

## Overview
SatyaAI is an advanced, modular deepfake detection platform supporting image, video, audio, and webcam analysis. It features a modern React frontend, robust Python backend (FastAPI), and extensible architecture for research and production use.

## Project Structure
```
SATYA-V-2.0-main (1)/
├── backend/         # Python FastAPI backend, models, utils, reports
├── frontend/        # React + Vite frontend (UI)
├── node-server/     # (Optional) Node.js server/bridge
├── python-bridge/   # (Optional) Python bridge for Node
├── shared/          # Shared code/schema
├── assets/          # Static assets
├── docs/            # Documentation
├── scripts/         # Startup and utility scripts
├── misc/            # Miscellaneous files
```

## Setup & Run
### 1. Backend (Python)
```sh
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
- API docs: http://127.0.0.1:8000/docs

### 2. Frontend (React)
```sh
cd frontend
npm install
npm run dev
```
- App: http://localhost:5173

## Features
- Deepfake detection for images, videos, audio, webcam
- Modern, user-friendly dashboard UI
- Report generation (HTML, PDF, JSON, QR, blockchain hash)
- Extensible: add new models, endpoints, or UI features easily

## Contributing
- Fork, branch, and submit PRs for new features or bugfixes.
- See `/docs/` for advanced usage and architecture.

## License
MIT

# Running Backend Tests

To run all backend tests with coverage:

```sh
cd backend
pytest --cov=backend --cov-report=term-missing
```

Or use the provided script:

```sh
bash tests/pytest_runner.sh
```

**Note:** Always run from the project root, or set `PYTHONPATH=.` to ensure imports work correctly.

---

# Advanced Usage & Architecture

- **Multi-Model Ensemble:** Combines results from multiple specialized models (XceptionNet, ResNet, EfficientNet, Audio CNN, etc.)
- **Facial, Texture, Audio, and Behavioral Analysis:** Advanced landmark, texture, and audio feature extraction.
- **Real Neural Network Inference:** Uses PyTorch, dlib, and face-recognition for real-time, high-accuracy detection.
- **GPU Acceleration:** CUDA support for faster processing (see requirements for optional GPU packages).
- **Comprehensive Reporting:** Generates detailed analysis reports with confidence scores, blockchain hash, and QR code.
- **API Endpoints:** Modular endpoints for image, video, audio, webcam, and batch analysis.

# Troubleshooting & Optimization

- **Common Issues:**
  - Port already in use: Kill processes on ports 3000/5002/5173.
  - Python/Node dependencies missing: Reinstall requirements, clear caches.
  - Webcam not working: Check browser permissions, use HTTPS or localhost.
  - AI models not loading: Check PyTorch/dlib installation, verify model weights.
- **Performance:**
  - Enable GPU for PyTorch if available.
  - Use high-quality images for better accuracy.
  - Use comprehensive analysis for best results.
- **System Health:**
  - Check `/health` endpoints for all services.
  - Monitor CPU, RAM, and GPU usage during analysis.

# Security & Production Deployment

- **CORS:** Restrict origins in production (`allow_origins` in backend/main.py).
- **Authentication:** Add user authentication for all sensitive endpoints.
- **HTTPS:** Use HTTPS for all production deployments.
- **Database:** Use a production database for scan history and user management.
- **Logging & Monitoring:** Integrate with Sentry or similar for error monitoring.
- **Environment Variables:** Use `.env` files for secrets and config (see `.env.example`).
- **File Uploads:** 100MB limit per file, validate file types.

# AI Model Details

- **XceptionNet:** Main image deepfake classifier (PyTorch, custom weights).
- **ResNet50/EfficientNet:** Used for advanced/pure AI modes (feature extraction, ensemble learning).
- **Audio CNN:** For audio spectrogram analysis and voice synthesis detection.
- **Face Recognition:** 128D face encodings for identity and consistency checks.
- **Ensemble Classifier:** Combines multiple model outputs for robust predictions.
- **Model Security:** All models are real, pre-trained, and verified.

# Planned & Supported New Technologies

- **ONNX Runtime:** For cross-platform, hardware-accelerated inference.
- **TensorRT:** For high-speed GPU inference (NVIDIA).
- **Cloud AI APIs:** Optional integration for cloud-based deepfake detection.
- **Advanced Explainability:** SHAP, LIME, and custom visualizations for model transparency.
- **Real-time Notifications:** WebSocket support for live analysis updates.

# Quick Start & Deployment

- **One-Click Start:** Use `start_servers.bat` (Windows) or `start_servers.sh` (Linux/Mac) for all-in-one launch.
- **Manual Start:** Start backend, frontend, and (optionally) Node server separately.
- **Production:** Add authentication, HTTPS, and production database before deploying.

# Troubleshooting

- See the Troubleshooting & Optimization section above for common issues and solutions.
- For advanced debugging, check logs in backend and frontend terminals.

---

**For more details, see the `/docs/` directory or contact the maintainers.** 

# Production Security & Best Practices

- **CORS:** Set the `CORS_ALLOW_ORIGINS` environment variable (comma-separated) for allowed origins in production. Example:
  ```env
  CORS_ALLOW_ORIGINS=https://yourdomain.com,https://admin.yourdomain.com
  ```
- **Authentication:** Use the `get_current_user` dependency in `backend/main.py` to protect sensitive endpoints. Replace the demo logic with JWT, OAuth, or your preferred method.
- **Logging & Monitoring:** The backend uses Python’s `logging` module. For production, integrate with Sentry or another error monitoring service for real-time alerts.
- **HTTPS:** Always use HTTPS in production deployments.
- **Environment Variables:** Store secrets and config in a `.env` file (see `.env.example`). 

---

**Uvicorn import error:**

You are seeing a **Uvicorn import error** when running:

```
uvicorn backend.main:app --reload
```

and the logs show:
```
INFO:     Will watch for changes in these directories: ['C:\\Users\\OMEN\\Downloads\\SATYA-V-2.0-main (1)\\frontend']
...
File "C:\Users\OMEN\AppData\Local\Programs\Python\Python313\Lib\site-packages\uvicorn\config.py", line 435, in load
    self.loaded_app = import_from_string(self.app)
...
raise exc from None
...
```

### **Diagnosis**

- You ran the command from the `frontend` directory, not the project root or `backend`.
- Uvicorn is looking for `backend.main:app` **relative to your current directory**.
- Since you are in `frontend/`, Python cannot find the `backend` package, causing the import error.

---

## **How to Fix**

**Always run the backend server from the project root or the backend directory.**

### **Correct Steps:**

1. **Go to the project root:**
   ```sh
   cd "C:\Users\OMEN\Downloads\SATYA-V-2.0-main (1)"
   ```

2. **Run Uvicorn from the project root:**
   ```sh
   uvicorn backend.main:app --reload
   ```

   - This ensures Python can find the `backend` package.

**OR**

1. **Go to the backend directory:**
   ```sh
   cd backend
   ```

2. **Run Uvicorn with a dot import:**
   ```sh
   uvicorn main:app --reload
   ```

---

## **Summary**

- **Do NOT run Uvicorn from inside the `frontend` directory.**
- Always run from the project root (or `backend/` with the correct import path).

---

**Try this and your backend should start up correctly!**  
If you see any further errors, copy the full error message here and I’ll help you debug. 

## AI/ML Models Integration

SatyaAI leverages a combination of open-source and custom AI/ML models for robust deepfake detection across image, video, and audio modalities:

### Open-Source Models
- **FaceForensics++**: CNN for image deepfake detection (image forensics)
- **DFDC Model**: EfficientNet for video deepfake detection (video forensics)
- **Wav2Lip / Resemblyzer**: Audio-visual and voice deepfake detection (lip sync, speaker verification)

### SatyaAI Custom Models
- **Vision/Temporal/Audio/Fusion**: Proprietary models for image, video, audio, and multimodal deepfake analysis
- **Ensemble Analyzer**: Aggregates results from all models and APIs for a final, highly accurate score
- **Model Management**: Hot-reload, health checks, versioning, weighted voting, confidence calibration, robust aggregation

### External AI APIs (Optional, for Cross-Validation)
- **Hive Moderation**
- **Sensity AI**
- **Reality Defender**
- **Azure Video Indexer**

### Model Integration Files
- `server/python/opensource_models.py`: Integrates open-source models (FaceForensics++, DFDC, Wav2Lip, Resemblyzer)
- `server/python/deepfake_analyzer.py`: SatyaAI custom models for image, video, audio, multimodal
- `server/python/ensemble_analyzer.py`: Ensemble logic for combining all models/APIs
- `server/python/api_integrations.py`: External API integrations
- `server/python/models.py`: Model classes, fusion logic, and postprocessing

---

**All models are managed for hot-reload, health checks, and versioning. Ensemble analyzer ensures robust, accurate results by combining multiple sources.** 