# SATYA AI - Deepfake Detection Platform

## 🚀 Overview
SATYA AI is an advanced deepfake detection platform that leverages cutting-edge AI/ML models to identify and analyze manipulated media across multiple modalities (image, audio, video, text). This full-stack application provides a seamless user experience for uploading, analyzing, and managing media content with robust security and performance features.

## 🌟 Features

### 🔒 Authentication & Security
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- CSRF protection
- Rate limiting and request validation
- Secure password hashing with bcrypt
- Supabase integration for secure auth

### 📱 Frontend (React + TypeScript)
- Modern, responsive UI with Tailwind CSS
- Real-time media analysis dashboard
- Interactive data visualization with Chart.js
- Progressive Web App (PWA) support
- Radix UI components for accessibility
- File upload with drag-and-drop

### ⚙️ Backend Architecture (Dual Backend)
#### Node.js + Express (API Gateway)
- RESTful API with versioning
- Real-time WebSocket support
- Background job processing
- File upload and processing pipeline
- Comprehensive logging and monitoring
- Drizzle ORM with PostgreSQL

#### Python + FastAPI (ML Processing)
- Advanced ML/DL model integration
- Wav2Vec2 for audio deepfake detection
- Computer vision models for image/video analysis
- NLP models for text analysis
- Multimodal fusion detection
- Lazy loading for optimal performance

### 🤖 AI/ML Integration
- **Image Detection**: Deepfake detection using EfficientNet/ResNet
- **Audio Detection**: Wav2Vec2-based deepfake detection
- **Video Detection**: Temporal 3D CNN + optical flow analysis
- **Text Analysis**: RoBERTa-based AI text detection
- **Multimodal Fusion**: Cross-modal analysis for enhanced accuracy
- Batch processing support
- Model versioning and A/B testing

### 📊 Database (PostgreSQL + Supabase)
- Relational data modeling
- Row-level security
- Full-text search
- Real-time subscriptions
- Database migrations with Drizzle ORM

## 🛠 Tech Stack

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS + CSS Modules
- **State Management**: Redux Toolkit + React Query
- **Routing**: React Router v6
- **Form Handling**: React Hook Form
- **UI Components**: Radix UI
- **Charts**: Chart.js
- **Testing**: Jest + React Testing Library
- **E2E**: Playwright

### Backend (Dual Architecture)
#### Node.js + Express (API Gateway)
- **Runtime**: Node.js 18+
- **Framework**: Express.js
- **Language**: TypeScript
- **API**: REST + WebSocket
- **Database**: PostgreSQL + Supabase
- **ORM**: Drizzle ORM
- **Authentication**: JWT + Supabase Auth
- **Testing**: Jest + Supertest

#### Python + FastAPI (ML Processing)
- **Runtime**: Python 3.10+
- **Framework**: FastAPI
- **ML Libraries**: PyTorch, TensorFlow, Transformers
- **Audio Models**: Wav2Vec2, SpeechBrain
- **Vision Models**: EfficientNet, ResNet, OpenCV
- **Text Models**: RoBERTa, HuggingFace Transformers
- **Database**: Supabase Client
- **Testing**: Pytest

### DevOps
- **CI/CD**: GitHub Actions
- **Containerization**: Docker + Docker Compose
- **Process Management**: PM2
- **Monitoring**: Prometheus + Grafana
- **Logging**: Winston + Sentry
- **Load Testing**: K6, Artillery

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ (LTS recommended)
- Python 3.10+ (with pip)
- PostgreSQL 14+ (with PostGIS extension)
- Redis 6+ (for caching and real-time features)
- Docker & Docker Compose (for containerized deployment)
- Supabase account (for authentication and database)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IntegratedRai444/SATYA-V-2.0.git
   cd SATYA-V-2.0
   ```

2. **Install dependencies**
   ```bash
   # Install root dependencies
   npm install
   
   # Install client dependencies
   cd client
   npm install
   
   # Install Node.js server dependencies
   cd ../server
   npm install
   
   # Install Python dependencies
   cd python
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy and configure environment files
   cp .env.example .env
   cp client/.env.example client/.env
   cp server/.env.example server/.env
   cp server/python/.env.example server/python/.env

   # Update the following environment variables in each file:
   # - SUPABASE_URL
   # - SUPABASE_ANON_KEY
   # - SUPABASE_SERVICE_ROLE_KEY
   # - DATABASE_URL
   # - API_BASE_URL (in client/.env)
   # - ENABLE_ML_MODELS=true (in server/python/.env)
   ```

4. **Start the development servers**
   ```bash
   # From root directory - starts both Node.js and Python servers
   npm run dev
   
   # Or start individually:
   # Node.js server only
   npm run dev:server
   
   # Python ML server only  
   npm run dev:python
   
   # Frontend only
   npm run dev:client
   ```

## 🧪 Testing

### Frontend Tests
```bash
# Run React frontend tests
cd client
npm test

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Backend Tests
```bash
# Run Node.js server tests
cd server
npm test

# Run Python ML tests
cd python
python test_ml_models_proper.py

# Run integration tests
npm run test:integration
```

### Load Testing
```bash
# Run baseline load test
npm run test:load:baseline

# Run stress test
npm run test:load:stress

# Run spike test
npm run test:load:spike

# Run all load tests
npm run test:load:all
```

### Health Checks
```bash
# Test all services health
npm run test:health

# Test individual services
curl -f http://localhost:3000/api/health  # Frontend
curl -f http://localhost:5001/health      # Node.js API
curl -f http://localhost:8000/health      # Python ML API
```

## 🚀 Deployment

### Environment Variables
Before deployment, ensure all required environment variables are set:

#### Root .env
- `NODE_ENV=production`
- `API_BASE_URL` - Your production API URL

#### Client/.env
- `VITE_API_URL` - Your production API URL
- `VITE_SUPABASE_URL` - Your Supabase project URL
- `VITE_SUPABASE_ANON_KEY` - Your Supabase anon key

#### Server/.env
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anon/public key
- `SUPABASE_SERVICE_ROLE_KEY` - Your Supabase service role key
- `JWT_SECRET` - Strong secret for JWT signing
- `DATABASE_URL` - Full PostgreSQL connection string

#### Server/python/.env
- `ENABLE_ML_MODELS=true` - Enable ML model processing
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anon key
- `DATABASE_URL` - PostgreSQL connection string
- `USE_GPU=True` - Enable GPU acceleration (if available)

### Production Build
```bash
# Install dependencies with production flag
npm ci --only=production

# Build all applications
npm run build
```

### Start Production Servers
```bash
# Start with PM2 (recommended for production)
npm run start:prod

# Or start individual services
npm start                    # Node.js server
cd server/python && python main_api.py  # Python ML server
```

### Docker Deployment
1. Create production `.env` files in appropriate directories
2. Build and start all containers:

```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Scale services (example)
docker-compose up -d --scale api=4 --scale worker=2
```

### Docker Compose Services
- **Frontend**: React app on port 3000
- **API Gateway**: Node.js server on port 5001
- **ML Processing**: Python FastAPI on port 8000
- **Database**: PostgreSQL with Supabase
- **Redis**: Caching and session storage
- **Monitoring**: Prometheus + Grafana

### Kubernetes Deployment (Optional)
For production deployments, we recommend using Kubernetes. See the `k8s/` directory for example manifests.

## 📚 Documentation

### API Documentation
- **Node.js API**: Available at `/api-docs` path on port 5001
- **Python ML API**: Available at `/api/docs` path on port 8000
- **OpenAPI Specs**: 
  - Node.js: `/api/v1/openapi.json`
  - Python: `/openapi.json`
- **Production URLs**: 
  - `https://your-api-domain.com/api-docs`
  - `https://your-ml-domain.com/api/docs`

### Database Schema
See [SATYAAI_DATABASE_SCHEMA.sql](./SATYAAI_DATABASE_SCHEMA.sql) for detailed schema documentation.

### Architecture
This project uses a **dual-backend architecture**:
- **Node.js API Gateway** (Port 5001): Handles authentication, file uploads, routing
- **Python ML Processing** (Port 8000): Handles AI/ML model inference
- **React Frontend** (Port 3000): User interface and dashboard

### ML Model Documentation
- **Audio Detection**: Wav2Vec2-based models for voice deepfake detection
- **Image Detection**: EfficientNet/ResNet for image manipulation detection  
- **Video Detection**: Temporal analysis with 3D CNNs
- **Text Analysis**: RoBERTa for AI-generated text detection
- **Multimodal Fusion**: Cross-modal analysis for enhanced accuracy

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👥 Team
- **[IntegratedRai444](https://github.com/IntegratedRai444)** - Project Lead & Core Developer
- **Contributors** - Open source community contributors

## 🙏 Acknowledgments
- **HuggingFace** - For pre-trained models and datasets
- **PyTorch & TensorFlow** - For deep learning frameworks
- **FastAPI & Express.js** - For robust web frameworks
- **Supabase** - For backend-as-a-service platform
- **Open Source Community** - For amazing tools and libraries
- **React & TypeScript Communities** - For excellent frontend tooling
