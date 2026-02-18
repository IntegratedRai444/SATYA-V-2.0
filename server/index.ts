import * as dotenv from 'dotenv';
import express, { type Request, type Response } from 'express';
import { createServer } from 'http';
import { IncomingMessage } from 'http';
import { URL } from 'url';
import { WebSocketServer, WebSocket } from 'ws';
import { Duplex } from 'stream';
import path from 'path';
import { routes } from './routes';

// Load environment variables
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const app = express();
const port = process.env.PORT || 5001;

// Set development mode for better Supabase auth handling
if (process.env.NODE_ENV !== 'production') {
  process.env.NODE_ENV = 'development';
  console.log('🛠️ Development mode enabled - Optimizing Supabase authentication');
}

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

// Dashboard stats endpoint
app.get('/api/v2/dashboard/stats', (req: Request, res: Response) => {
  res.header('Access-Control-Allow-Origin', req.headers.origin || '*');
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
  res.header('Access-Control-Allow-Origin', req.headers.origin || '*');
  res.json({
    activities: [],
    timestamp: new Date().toISOString()
  });
});

// WebSocket endpoint implementation
const wss = new WebSocketServer({ 
  noServer: true,
  path: '/api/v2/dashboard/ws' // Explicit path for WebSocket
});

// Track connected clients
const clients = new Map<string, WebSocket>();

// Handle WebSocket upgrade
const httpServer = createServer(app);

httpServer.on('upgrade', (request: IncomingMessage, socket: Duplex, head: Buffer) => {
  const url = new URL(request.url || '', `http://localhost`);
  
  // Only handle WebSocket upgrades for the dashboard WebSocket path
  if (url.pathname === '/api/v2/dashboard/ws') {
    wss.handleUpgrade(request, socket as unknown as Duplex, head, (ws: WebSocket) => {
      wss.emit('connection', ws, request);
    });
  } else {
    // Close connection for other paths
    socket.destroy();
  }
});

wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
  const url = new URL(req.url || '', `http://localhost`);
  const token = url.searchParams.get('token');
  
  if (!token) {
    console.log('WebSocket connection rejected: No token provided');
    ws.close(1008, 'No token provided');
    return;
  }

  console.log('WebSocket client connected');
  
  const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  clients.set(clientId, ws);
  
  ws.on('message', (message: string) => {
    try {
      const data = JSON.parse(message);
      console.log('WebSocket message received:', data);
      
      // Echo back the message for now
      ws.send(JSON.stringify({
        type: 'echo',
        data: data,
        timestamp: new Date().toISOString()
      }));
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  });
  
  ws.on('close', () => {
    console.log('WebSocket client disconnected');
    clients.delete(clientId);
  });
  
  ws.on('error', (error: Error) => {
    console.error('WebSocket error:', error);
  });
});

// API routes are handled by the router
app.use('/api/v2', routes);

httpServer.listen(Number(port), '0.0.0.0', () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  
  // Setup WebSocket server
  wss.on('listening', () => {
    console.log('WebSocket server listening on port', port);
  });
});
