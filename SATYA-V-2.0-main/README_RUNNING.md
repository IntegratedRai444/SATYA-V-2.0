# 🚀 SatyaAI Deepfake Detection System - Running Guide

## 📋 Prerequisites

### System Requirements
- **Node.js** (v18 or higher)
- **Python** (3.8 or higher)
- **Git** (for cloning)
- **Windows 10/11** (for .bat files) or **Linux/Mac** (for shell scripts)

### Hardware Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: Optional but recommended for faster analysis
- **Webcam**: For webcam analysis feature

## 🛠️ Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SATYA-V-2.0-main
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Install Python Dependencies
```bash
# Install Python requirements
pip install -r requirements_ai_system.txt

# Or if you prefer using uv (faster)
uv pip install -r requirements_ai_system.txt
```

### 4. Set Up Database (Optional)
```bash
# Push database schema
npm run db:push
```

## 🚀 Running the System

### Method 1: Quick Start (Recommended)

#### Windows Users:
```bash
# Double-click the batch file
start_servers.bat
```

#### Linux/Mac Users:
```bash
# Make the script executable
chmod +x start_servers.sh
./start_servers.sh
```

### Method 2: Manual Start

#### Step 1: Start the Python AI Backend
```bash
# Terminal 1 - Start the AI system
python complete_ai_system.py
```
The backend will start on `http://localhost:5002`

#### Step 2: Start the Node.js Server
```bash
# Terminal 2 - Start the main server
npm run dev
```
The server will start on `http://localhost:3000`

#### Step 3: Start the React Frontend
```bash
# Terminal 3 - Start the frontend
cd client
npm run dev
```
The frontend will start on `http://localhost:5173`

### Method 3: All-in-One Command
```bash
# Start everything at once
npm run dev:all
```

## 🌐 Accessing the Application

### Main Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3000
- **AI System**: http://localhost:5002

### API Endpoints
- **Image Analysis**: `POST /api/ai/analyze/image`
- **Video Analysis**: `POST /api/ai/analyze/video`
- **Audio Analysis**: `POST /api/ai/analyze/audio`
- **Batch Analysis**: `POST /api/ai/analyze/batch`
- **Webcam Analysis**: `POST /api/ai/analyze/image` (with imageData)
- **History**: `GET /api/scans`

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```env
# Database
DATABASE_URL=your_database_url_here

# AI Models
ENABLE_GPU=true
MODEL_PATH=./models

# Server
PORT=3000
AI_PORT=5002
```

### AI Model Configuration
The system uses multiple neural network models:
- **ResNet50**: Face analysis
- **EfficientNet**: Texture analysis
- **Vision Transformer**: Advanced analysis
- **Audio CNN**: Audio analysis

## 📱 Using the Application

### 1. Upload Analysis
- Go to the **Upload** tab
- Drag and drop files or click to browse
- Select analysis type (Quick/Comprehensive/Detailed)
- Adjust confidence threshold
- Enable/disable advanced models

### 2. Webcam Analysis
- Go to the **Webcam** tab
- Click "Start Webcam" to activate camera
- Position your face in the frame
- Click "Capture Image" to take a photo
- Click "Analyze Image" to run detection

### 3. Batch Analysis
- Go to the **Batch** tab
- Upload multiple files
- Configure analysis options
- Click "Start Batch Analysis"
- Monitor progress for each file

### 4. View History
- Go to the **History** tab
- Search by filename
- Filter by type or result
- View detailed analysis reports

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Kill processes on specific ports
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:3000 | xargs kill -9
```

#### 2. Python Dependencies Missing
```bash
# Reinstall requirements
pip install --upgrade -r requirements_ai_system.txt

# Or use conda
conda install pytorch torchvision torchaudio -c pytorch
```

#### 3. Node.js Dependencies Issues
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 4. Webcam Not Working
- Check browser permissions
- Ensure HTTPS or localhost
- Try different browsers (Chrome recommended)

#### 5. AI Models Not Loading
```bash
# Check if PyTorch is installed correctly
python -c "import torch; print(torch.__version__)"

# Install CUDA version if you have GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Performance Optimization

#### For Better Speed:
1. **Enable GPU**: Install CUDA version of PyTorch
2. **Increase RAM**: Allocate more memory to Node.js
3. **Use SSD**: Store models on fast storage
4. **Close other apps**: Free up system resources

#### For Better Accuracy:
1. **Use Comprehensive Analysis**: More thorough but slower
2. **Enable Advanced Models**: Uses multiple neural networks
3. **Adjust Confidence Threshold**: Higher threshold = more strict
4. **Use High-Quality Images**: Better input = better results

## 📊 System Status

### Health Check
```bash
# Check if all services are running
curl http://localhost:3000/health
curl http://localhost:5002/health
```

### Performance Monitoring
- **CPU Usage**: Monitor during analysis
- **Memory Usage**: Check for memory leaks
- **GPU Usage**: If using CUDA
- **Network**: Monitor API calls

## 🔒 Security Notes

1. **Local Development**: System runs on localhost by default
2. **File Uploads**: 100MB limit per file
3. **Webcam Access**: Requires user permission
4. **Data Storage**: Analysis results stored locally
5. **API Security**: Add authentication for production

## 📈 Production Deployment

### For Production Use:
1. **Add Authentication**: Implement user login
2. **Use HTTPS**: Secure all communications
3. **Database**: Use production database (PostgreSQL/MySQL)
4. **Load Balancing**: For high traffic
5. **Monitoring**: Add logging and monitoring
6. **Backup**: Regular data backups

### Docker Deployment:
```bash
# Build and run with Docker
docker build -t satyaai .
docker run -p 3000:3000 -p 5002:5002 satyaai
```

## 🆘 Support

### Getting Help:
1. **Check Logs**: Look at terminal output for errors
2. **Test Endpoints**: Use the test scripts provided
3. **Documentation**: Read the README files
4. **Issues**: Report bugs with system information

### Test the System:
```bash
# Run the test script
python test_all_endpoints.py

# Or test individual components
python test_server.py
```

## 🎯 Quick Test

After starting the system, test it with:

1. **Upload a test image** to http://localhost:5173
2. **Use webcam analysis** to test real-time detection
3. **Check history** to see stored results
4. **Try batch upload** with multiple files

The system should provide detailed analysis results with confidence scores and recommendations!

---

**🎉 Congratulations!** Your SatyaAI Deepfake Detection System is now running with enhanced accuracy and all features working! 