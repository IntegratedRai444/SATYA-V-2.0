import * as dotenv from 'dotenv';
import express, { type Request, type Response } from 'express';
import { createServer } from 'http';
import path from 'path';
import webSocketManager from './services/websocket-manager';
import { routes } from './routes';
import { createRequestLogger } from './config/logger';

// Load environment variables
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const app = express();
const port = process.env.PORT || 5001;

// Set development mode for better Supabase auth handling
if (process.env.NODE_ENV !== 'production') {
  process.env.NODE_ENV = 'development';
  console.log('🛠️ Development mode enabled - Optimizing Supabase authentication');
}

// Body parsing middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging middleware
app.use(createRequestLogger());

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
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, x-request-id, x-csrf-token, x-api-version');
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

// API routes are handled by the router
app.use('/api/v2', routes);

// Error handling middleware (must be last)
app.use((err: Error, req: Request, res: Response) => {
  console.error('Unhandled error:', err);
  console.error('Error details:', {
    message: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
    // Don't log entire request body for security
    headers: req.headers
  });
  
  res.status(500).json({
    success: false,
    error: 'INTERNAL_SERVER_ERROR',
    message: 'An unexpected error occurred',
    ...(process.env.NODE_ENV === 'development' && {
      details: err.message,
      stack: err.stack
    })
  });
});

// WebSocket endpoint implementation - handled by webSocketManager
const httpServer = createServer(app);

httpServer.listen(Number(port), '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  
  // Initialize WebSocket manager
  webSocketManager.initialize(httpServer);
  console.log('WebSocket server initialized');
});
