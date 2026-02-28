import * as dotenv from 'dotenv';
import express, { type Request, type Response } from 'express';
import { createServer } from 'http';
import path from 'path';
import cookieParser from 'cookie-parser';
import { setInterval } from 'timers';
import webSocketManager from './services/websocket-manager';
import { routes } from './routes';
import { createRequestLogger } from './config/logger';
import { systemRouter } from './routes/system-health';
import { reconcileStaleJobs, cleanupRunningJobs } from './startup/reconcileJobs';
import { migrateJsonbResults } from './scripts/fixJsonbResults';

// Load environment variables
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const app = express();
const port = process.env.PORT || 5001;

// 🔥 FIX 5 — Memory Hard Floor: Add memory monitoring
const HARD_LIMIT_MB = 4096; // 4GB hard cap

// Memory monitoring function
const checkMemoryUsage = () => {
  const memUsage = process.memoryUsage();
  const rssMB = memUsage.rss / 1024 / 1024;
  
  if (rssMB > HARD_LIMIT_MB) {
    console.error('🚨 HARD MEMORY CAP REACHED', {
      rss: `${rssMB.toFixed(2)}MB`,
      heapTotal: `${(memUsage.heapTotal / 1024 / 1024).toFixed(2)}MB`,
      heapUsed: `${(memUsage.heapUsed / 1024 / 1024).toFixed(2)}MB`,
      external: `${(memUsage.external / 1024 / 1024).toFixed(2)}MB`,
      limit: `${HARD_LIMIT_MB}MB`
    });
    
    // Trigger emergency cleanup
    console.error('🧹 Triggering emergency cleanup due to memory pressure');
    
    // Force garbage collection
    if (global.gc) {
      global.gc();
    }
    
    // In production, you might want to restart the process
    if (process.env.NODE_ENV === 'production') {
      console.error('💀 Memory limit exceeded in production - initiating graceful shutdown');
      process.exit(1);
    }
  }
};

// Check memory every 30 seconds
setInterval(checkMemoryUsage, 30000);

// Set development mode for better Supabase auth handling
if (process.env.NODE_ENV !== 'production') {
  process.env.NODE_ENV = 'development';
  console.log('🛠️ Development mode enabled - Optimizing Supabase authentication');
}

// Body parsing middleware
app.use(cookieParser());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging middleware
app.use((req, res, next) => {
  console.log('🔍 REQUEST DEBUG', {
    path: req.path,
    method: req.method,
    headers: req.headers,
    cookies: req.cookies,
    rawCookies: req.headers.cookie
  });
  next();
});

app.use(createRequestLogger());

// CORS middleware with credentials support
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
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, x-request-id, x-csrf-token, x-api-version, Cookie');
  res.header('Access-Control-Allow-Credentials', 'true');
  res.header('Access-Control-Expose-Headers', 'Set-Cookie');
  
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
app.use('/system', systemRouter);

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

httpServer.listen(Number(port), '0.0.0.0', async () => {
  console.log(`Server running on port ${port}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  
  // 🔥 JSONB CONSISTENCY FIX - One-time migration
  if (process.env.RUN_JSONB_MIGRATION === 'true') {
    console.log('🔧 Running JSONB consistency migration...');
    try {
      await migrateJsonbResults();
      console.log('✅ JSONB migration completed');
    } catch (error) {
      console.error('❌ JSONB migration failed:', error);
      console.log('⚠️  Continuing startup despite migration failure...');
    }
  }
  
  // 🔥 FIX 3 — Job Reconciliation on Server Boot
  console.log('🔄 Starting job reconciliation...');
  await reconcileStaleJobs();
  console.log('✅ Job reconciliation completed');
  
  // Initialize WebSocket manager
  webSocketManager.initialize(httpServer);
  console.log('WebSocket server initialized');
});

// 🔥 FIX 3 — Graceful shutdown handlers
const gracefulShutdown = async (signal: string) => {
  console.log(`\n${signal} received - starting graceful shutdown...`);
  
  try {
    // Cleanup running jobs
    console.log('🧹 Cleaning up running jobs...');
    await cleanupRunningJobs();
    
    // Shutdown WebSocket manager
    console.log('🔌 Shutting down WebSocket manager...');
    webSocketManager.shutdown();
    
    console.log('✅ Graceful shutdown completed');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error during graceful shutdown:', error);
    process.exit(1);
  }
};

// Register shutdown handlers
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));
process.on('SIGUSR2', () => gracefulShutdown('SIGUSR2')); // For nodemon restarts
