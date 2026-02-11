import * as dotenv from 'dotenv';
import express, { type Request, type Response } from 'express';
import { createServer } from 'http';
import path from 'path';

// Load environment variables
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const app = express();
const port = process.env.PORT || 5001;

// Basic CORS middleware
app.use((req, res, next) => {
  const origin = req.headers.origin;
  const allowedOrigins = [
    'http://localhost:3000',
    'http://localhost:5173',
    'http://127.0.0.1:3000',
    'http://127.0.0.1:5173'
  ];
  
  if (origin && allowedOrigins.includes(origin)) {
    res.header('Access-Control-Allow-Origin', origin);
  } else if (!origin) {
    res.header('Access-Control-Allow-Origin', '*');
  }
  
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS, HEAD');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With');
  res.header('Access-Control-Allow-Credentials', 'true');
  
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
    return;
  }
  
  next();
});

// Basic health endpoint
app.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Dashboard stats endpoint
app.get('/api/v2/dashboard/stats', (req: Request, res: Response) => {
  res.json({
    totalAnalyses: 0,
    completedAnalyses: 0,
    pendingAnalyses: 0,
    failedAnalyses: 0,
    averageProcessingTime: 0,
    serverUptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

// Recent activity endpoint
app.get('/api/v2/dashboard/recent-activity', (req: Request, res: Response) => {
  res.json({
    activities: [],
    timestamp: new Date().toISOString()
  });
});

// WebSocket endpoint placeholder
app.get('/api/v2/dashboard/ws', (req: Request, res: Response) => {
  res.status(501).json({
    error: 'WebSocket endpoint not yet implemented',
    message: 'This endpoint will be implemented for real-time updates'
  });
});

// Basic image analysis endpoint
app.post('/api/v2/analysis/image', (req: Request, res: Response) => {
  try {
    // Simple response for now
    res.json({
      success: true,
      message: 'Image analysis endpoint is working',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      timestamp: new Date().toISOString()
    });
  }
});

const server = createServer(app);

server.listen(Number(port), '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});
