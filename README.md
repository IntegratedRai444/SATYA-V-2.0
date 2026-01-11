# SATYA AI - Deepfake Detection Platform

## 🚀 Overview
SATYA AI is an advanced deepfake detection platform that leverages cutting-edge AI/ML models to identify and analyze manipulated media. This full-stack application provides a seamless user experience for uploading, analyzing, and managing media content with robust security and performance features.

## 🌟 Features

### 🔒 Authentication & Security
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- CSRF protection
- Rate limiting and request validation
- Secure password hashing with bcrypt

### 📱 Frontend (React + TypeScript)
- Modern, responsive UI with Tailwind CSS
- Real-time media analysis dashboard
- Interactive data visualization
- Progressive Web App (PWA) support
- Offline capabilities with service workers

### ⚙️ Backend (Node.js + Express)
- RESTful API with versioning
- Real-time WebSocket support
- Background job processing
- File upload and processing pipeline
- Comprehensive logging and monitoring

### 🤖 AI/ML Integration
- Deepfake detection models
- Media analysis pipeline
- Batch processing support
- Model versioning and A/B testing

### 📊 Database (PostgreSQL + Supabase)
- Relational data modeling
- Row-level security
- Full-text search
- Real-time subscriptions

## 🛠 Tech Stack

### Frontend
- **Framework**: React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS + CSS Modules
- **State Management**: Redux Toolkit + React Query
- **Routing**: React Router v6
- **Form Handling**: React Hook Form
- **Testing**: Jest + React Testing Library
- **E2E**: Playwright

### Backend
- **Runtime**: Node.js 18+
- **Framework**: Express.js
- **Language**: TypeScript
- **API**: REST + WebSocket
- **Database**: PostgreSQL + Supabase
- **ORM**: Drizzle ORM
- **Authentication**: JWT + OAuth 2.0
- **Testing**: Jest + Supertest

### DevOps
- **CI/CD**: GitHub Actions
- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (optional)
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack
- **Infrastructure as Code**: Terraform

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
   git clone https://github.com/your-username/satya-ai.git
   cd satya-ai
   ```

2. **Install dependencies**
   ```bash
   # Install root dependencies
   npm install
   
   # Install client dependencies
   cd client
   npm install
   
   # Install server dependencies
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
   # - DATABASE_URL
   # - API_BASE_URL (in client/.env)
   # - Other service-specific variables
   ```

4. **Start the development servers**
   ```bash
   # From root directory
   npm run dev
   ```

## 🧪 Testing

### Unit Tests
```bash
npm test
```

### Integration Tests
```bash
npm run test:integration
```

### E2E Tests
```bash
npm run test:e2e
```

## 🚀 Deployment

### Environment Variables
Before deployment, ensure all required environment variables are set:
- `NODE_ENV=production`
- `API_BASE_URL` - Your production API URL
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_ANON_KEY` - Your Supabase anon/public key
- `SUPABASE_SERVICE_ROLE_KEY` - Your Supabase service role key (server-side only)
- `JWT_SECRET` - Strong secret for JWT signing
- `DATABASE_URL` - Full PostgreSQL connection string
- `REDIS_URL` - Redis connection URL (if using Redis)

### Production Build
```bash
# Install dependencies with production flag
npm ci --only=production

# Build the application
npm run build
```

### Start Production Server
```bash
npm start
```

### Docker Deployment
1. Create a `.env` file in the project root with your production variables
2. Build and start the containers:

```bash
# Build and start all services
docker-compose -f docker-compose.prod.yml up --build -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale services (example)
docker-compose -f docker-compose.prod.yml up -d --scale api=4 --scale worker=2
```

### Kubernetes Deployment (Optional)
For production deployments, we recommend using Kubernetes. See the `k8s/` directory for example manifests.

## 📚 Documentation

### API Documentation
- Swagger UI: Available at `/api-docs` path on your domain
- OpenAPI Spec: `/api/v1/openapi.json`
- Production URL: `https://your-api-domain.com/api-docs`

### Database Schema
See [DATABASE_SCHEMA.md](./docs/DATABASE_SCHEMA.md) for detailed schema documentation.

### Architecture
See [ARCHITECTURE.md](./docs/ARCHITECTURE.md) for system architecture details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team
- [Your Name](https://github.com/your-username) - Project Lead
- [Team Member](https://github.com/username) - Role

## 🙏 Acknowledgments
- [Awesome Library](https://github.com/awesome/library) - For inspiration
- [Open Source Community](https://opensource.org/) - For amazing tools and libraries
