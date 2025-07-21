#!/bin/bash

echo "🚀 Starting SatyaAI Deepfake Detection System..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18 or higher."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $PYTHON_PID $NODE_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

echo "🐍 Starting Python AI Backend..."
python3 complete_ai_system.py &
PYTHON_PID=$!

echo "⏳ Waiting for Python backend to start..."
sleep 5

echo "🖥️ Starting Node.js Server..."
npm run dev &
NODE_PID=$!

echo "⏳ Waiting for Node.js server to start..."
sleep 3

echo "🌐 Starting React Frontend..."
cd client && npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 All servers are starting..."
echo ""
echo "📱 Frontend: http://localhost:5173"
echo "🔧 Backend: http://localhost:3000"
echo "🤖 AI System: http://localhost:5002"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for all processes
wait 