import 'dotenv/config';
import express from 'express';
import { createServer, Server } from 'http';
import type { 
  Express as ExpressApp
} from 'express';
import { URL } from 'url';

// Type-safe Python bridge import
// import { pythonBridge } from './services/python-http-bridge';

import promClient from 'prom-client';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import compression from 'compression';
import path from 'path';
import os from 'os';
import fs from 'fs';

// This must be the first import to ensure environment variables are loaded
import './setup-env';

// Import configuration
import { config, ConfigurationError } from './config';
import { validateEnvironment } from './config/validate-env';

// Import middleware
import { 
  notFoundHandler, 
  requestIdMiddleware,
  errorHandler
} from './middleware/error-handler';

// Import routes
import { router as apiRouter } from './routes/index';
import { versionMiddleware } from './middleware/api-version';
import { features } from './config/features';

// Augment the Express Request type to include custom properties
declare global {
  interface Request {
    id?: string;
    startTime?: number;
  }
}


// Import services with proper types
import alertingSystem from './config/alerting-system';


// Mock implementations for development
// const metrics = process.env.NODE_ENV === 'production' 
//   ? { register: new promClient.Registry() } 
//   : { register: new promClient.Registry() };

// Initialize services if needed
if (process.env.NODE_ENV !== 'production') {
  // eslint-disable-next-line no-console
  console.log('Running in development mode');
}

let webSocketManager: unknown = null;

// Initialize WebSocket manager with proper error handling
const initializeWebSocketManager = async () => {
  if (process.env.ENABLE_WEBSOCKETS === 'true') {
    try {
      // Import the default WebSocket manager instance
      const webSocketModule = await import('./services/websocket-manager');
      webSocketManager = webSocketModule.default;
      logger.info('WebSocket manager loaded successfully');
    } catch (error) {
      logger.warn('WebSocket manager not available:', error);
      webSocketManager = null;
    }
  } else {
    logger.info('WebSocket service is disabled');
  }
};

// Initialize WebSocket manager
initializeWebSocketManager();

// Security Configuration
import { configureSecurity, getSecurityConfig } from './config/security-config';

// Configuration and logging
import { logger, logError } from './config/logger';
import { createRequestLogger } from './config/logger';

declare global {
  // Add global type declarations if needed
}

// Initialize environment variables
try {
  validateEnvironment();
  // eslint-disable-next-line no-console
  console.log('Environment variables loaded successfully');
} catch (error) {
  if (error instanceof ConfigurationError) {
    // eslint-disable-next-line no-console
    console.error('❌ Configuration Error:', error.message);
    // eslint-disable-next-line no-console
    console.error('Please check your .env file and ensure all required variables are set.');
    // eslint-disable-next-line no-console
    console.error('Required variables: SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY, JWT_SECRET, DATABASE_URL');
  } else {
    // eslint-disable-next-line no-console
    console.error('❌ Unexpected error during environment validation:', error);
  }
  process.exit(1);
}

const app: ExpressApp = express();
const httpServer: Server = createServer(app);

// Rate Limiter Configurations
// Authentication endpoints - RELAXED FOR DEVELOPMENT
const authRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: process.env.NODE_ENV === 'development' ? 10000 : 100, // Very high limit in dev
  message: 'Too many authentication attempts from this IP, please try again after 15 minutes',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true, // Don't count successful requests
  skipFailedRequests: false,
  skip: () => process.env.NODE_ENV === 'development', // Skip rate limiting in development
});

// Analysis endpoints - moderate limits for resource-intensive operations
const analysisRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 10, // 10 analysis requests per minute per IP
  message: 'Too many analysis requests, please try again in a minute',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: false,
});

// Upload endpoints - strict limits due to file processing overhead
const uploadRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 5, // 5 uploads per minute per IP
  message: 'Too many upload requests, please try again in a minute',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: false,
});

// General API endpoints - generous limits for normal operations
const apiRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute per IP
  message: 'Too many API requests, please try again in a minute',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true, // Don't count successful requests
});

// Compression middleware - must be early in the chain
app.use(compression({
  filter: (req, res) => {
    // Don't compress responses with this request header
    if (req.headers['x-no-compression']) {
      return false;
    }
    
    // Compress all responses larger than 1KB
    return compression.filter(req, res);
  },
  level: 6, // Compression level (0-9, 6 is default balanced)
  threshold: 1024 // Only compress responses larger than 1KB
}));

// Security middleware - must be first
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:", "blob:"],
      connectSrc: ["'self'", "ws:", "wss:", "https:"],
      fontSrc: ["'self'", "https:", "data:"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"]
    }
  },
  crossOriginEmbedderPolicy: false,
  crossOriginOpenerPolicy: false,
  crossOriginResourcePolicy: { policy: 'same-site' },
  dnsPrefetchControl: false,
  frameguard: { action: 'deny' },
  hidePoweredBy: true,
  hsts: { maxAge: 63072000, includeSubDomains: true, preload: true },
  ieNoOpen: true,
  noSniff: true,
  referrerPolicy: { policy: 'same-origin' },
  xssFilter: true
}));

// Trust proxy for rate limiting and security
app.set('trust proxy', 1);

// CORS Configuration
const corsOptions = {
  origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
    // Allow requests with no origin (like mobile apps or curl requests)
    if (!origin) return callback(null, true);
    
    // Get allowed origins from environment or use defaults
    const allowedOrigins = process.env.CORS_ORIGIN 
      ? process.env.CORS_ORIGIN.split(',').map(o => o.trim())
      : [
          'http://localhost:3000',
          'http://localhost:5173',
          'http://127.0.0.1:3000',
          'http://127.0.0.1:5173',
          'https://satyaai.app',
          'https://www.satyaai.app'
        ];
    
    // Allow all subdomains of satyaai.app in production
    const originPattern = /^https?:\/\/([a-z0-9-]+\.)?satyaai\.app(\/.*)?$/i;
    
    if (
      allowedOrigins.includes(origin) || 
      originPattern.test(origin) ||
      process.env.NODE_ENV === 'development'
    ) {
      callback(null, true);
    } else {
      // eslint-disable-next-line no-console
      console.warn(`Blocked CORS request from origin: ${origin}`);
      callback(new Error('Not allowed by CORS'));
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD'],
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Requested-With',
    'Accept',
    'X-API-Key',
    'X-Request-Id',
    'X-CSRF-Token',
    'Access-Control-Allow-Headers',
    'Origin',
    'Accept',
    'X-Requested-With',
    'Access-Control-Request-Method',
    'Access-Control-Request-Headers'
  ],
  exposedHeaders: [
    'Content-Length',
    'Content-Type',
    'X-Request-Id',
    'X-RateLimit-Limit',
    'X-RateLimit-Remaining',
    'X-RateLimit-Reset',
    'X-Total-Count',
    'Link'
  ],
  maxAge: 86400, // 24 hours
  preflightContinue: false,
  optionsSuccessStatus: 204
};

// Apply CORS middleware with options for all routes
app.use((req, res, _next) => {
  // Apply CORS headers for all routes
  const corsHandler = cors(corsOptions);
  corsHandler(req, res, _next);
});

// Error handling middleware
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use((err: Error, req: any, res: any, next: any) => {
  logger.error('Request error', {
    error: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
    path: req.path,
    method: req.method,
    ip: req.ip
  });
  
  if (!res.headersSent) {
    res.status(500).json({ 
      error: 'Internal Server Error',
      message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
    });
  }
  
  // Call next to continue to the next error handler
  next(err);
});

// Configure security middleware
configureSecurity(app);

// JSON and URL-encoded body parsing with security limits
const securityConfig = getSecurityConfig();
app.use(express.json({ limit: securityConfig.api.jsonLimit }));
app.use(express.urlencoded({ 
  extended: true, 
  limit: securityConfig.api.requestSizeLimit 
}));

// Cookie parsing middleware for httpOnly cookies
import cookieParser from 'cookie-parser';
app.use(cookieParser());

// Create custom gauges for health checks
const healthCheckGauge = new promClient.Gauge({
  name: 'health_check_status',
  help: 'Health check status (1 = healthy, 0 = unhealthy)'
});

const databaseStatusGauge = new promClient.Gauge({
  name: 'database_status',
  help: 'Database connection status (1 = healthy, 0 = unhealthy)'
});

const pythonServerStatusGauge = new promClient.Gauge({
  name: 'python_server_status',
  help: 'Python server connection status (1 = healthy, 0 = unhealthy)'
});

// Health endpoints (before authentication middleware)
app.get('/health', async (_req, res) => {
  const startTime = Date.now();
  
  try {
    // Check database connection
    const dbStatus = await checkDatabaseConnection();
    
    // Check Python server status
    const pythonStatus = await checkPythonServer();
    
    // Check file system status
    const fsStatus = await checkFileSystem();
    
    // Check memory usage
    const memoryStatus = checkMemoryUsage();
    
    // Determine overall status
    const allStatuses = [dbStatus, pythonStatus, fsStatus, memoryStatus];
    const isHealthy = allStatuses.every(s => s.status === 'healthy');
    const isDegraded = allStatuses.some(s => s.status === 'degraded');
    
    const status = isHealthy ? 'healthy' : (isDegraded ? 'degraded' : 'unhealthy');
    
    const response = {
      status,
      timestamp: new Date().toISOString(),
      version: '2.0.0',
      uptime: process.uptime(),
      responseTime: Date.now() - startTime,
      components: {
        database: dbStatus,
        pythonServer: pythonStatus,
        fileSystem: fsStatus,
        memory: memoryStatus
      },
      metrics: {
        memoryUsage: process.memoryUsage(),
        cpuUsage: process.cpuUsage(),
        uptime: process.uptime()
      }
    };
    
    // Update health metrics
    if (features.enableMetrics) {
      healthCheckGauge.set(status === 'healthy' ? 1 : 0);
      databaseStatusGauge.set(dbStatus.status === 'healthy' ? 1 : 0);
      pythonServerStatusGauge.set(pythonStatus.status === 'healthy' ? 1 : 0);
    }
    
    res.status(200).json(response);
  } catch (error) {
    logger.error('Health check failed', { error });
    res.status(500).json({
      status: 'unhealthy',
      message: 'Health check failed',
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
});

// Authentication middleware - REMOVED GLOBAL APPLICATION
// authenticate is now imported and used in routes/index.ts for protected routes only

// Trust first proxy (if behind a reverse proxy like nginx)
app.set('trust proxy', 1);

// Enhanced monitoring setup
// import { metricsRegistry, httpMetricsMiddleware } from './config/metrics';
import auditLogger from './config/audit-logger';

// Mock metrics middleware for now
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const httpMetricsMiddleware = () => (req: any, res: any, next: any) => next();

// Audit logging middleware
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const auditLoggerMiddleware: any = (req: any, res: any, next: any) => {
  // Log request details
  const start = Date.now();
  const { method, originalUrl, ip, headers } = req;
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    auditLogger.log({
      timestamp: new Date().toISOString(),
      method: method || 'UNKNOWN',
      url: originalUrl || 'unknown',
      ip: ip || 'unknown',
      status: res.statusCode,
      duration: `${duration}ms`,
      userAgent: headers['user-agent'] || 'unknown'
    });
  });
  
  next();
};

app.use(httpMetricsMiddleware());
app.use(auditLoggerMiddleware);

// Apply rate limiting to specific routes
app.use('/api/auth', authRateLimit);
app.use('/api/analyze', analysisRateLimit);
app.use('/api/upload', uploadRateLimit);
app.use('/api/', apiRateLimit);

// Apply middleware
app.use(requestIdMiddleware);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use(versionMiddleware as any);

// API routes
app.use('/api', apiRouter);

// 404 Handler for undefined routes
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use(notFoundHandler as any);

// Global Error Handler
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use(errorHandler as any);

// Python API routes are now handled by the python-http-bridge service
// All Python requests go through the authenticated Node backend

// Connect alerting system to metrics
alertingSystem.on('error', (error: Error) => {
  logger.error('Alerting system error:', error);
});

alertingSystem.on('alert', (alert: unknown) => {
  logger.warn('ALERT:', alert);
  // Additional alert handling logic can be added here
});

// Prometheus metrics setup (if enabled)
if (features.enableMetrics) {
  const collectDefaultMetrics = promClient.collectDefaultMetrics;
  collectDefaultMetrics();
  
  /**
   * @swagger
   * /metrics:
   *   get:
   *     summary: Prometheus metrics endpoint
   *     description: Exposes application metrics in Prometheus format
   *     tags: [Metrics]
   *     responses:
   *       200:
   *         description: Prometheus metrics in text format
   *         content:
   *           text/plain:
   *             schema:
   *               type: string
   *               example: |
   *                 # HELP process_cpu_user_seconds_total Total user CPU time spent in seconds.
   *                 # TYPE process_cpu_user_seconds_total counter
   *                 process_cpu_user_seconds_total 123.456
   */
  app.get('/metrics', async (_req, res) => {
    try {
      res.set('Content-Type', promClient.register.contentType);
      const metrics = await promClient.register.metrics();
      res.end(metrics);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error('Failed to collect metrics', { error });
      res.status(500).end('Failed to collect metrics');
    }
  });
}

// Import health monitor
import { healthMonitor } from './services/health-monitor';

/**
 * @swagger
 * /health:
 *   get:
 *     summary: Health check endpoint
 *     description: Check the health status of all system components including Python server, database, WebSocket, file system, memory, and processing queue
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: System is healthy or degraded
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   enum: [healthy, degraded, unhealthy]
 *                   example: healthy
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 version:
 *                   type: string
 *                   example: 2.0.0
 *                 components:
 *                   type: object
 *                   properties:
 *                     pythonServer:
 *                       type: object
 *                       properties:
 *                         status:
 *                           type: string
 *                         responseTime:
 *                           type: number
 *                     database:
 *                       type: object
 *                     websocket:
 *                       type: object
 *                     fileSystem:
 *                       type: object
 *                     memory:
 *                       type: object
 *                     processing:
 *                       type: object
 *       503:
 *         description: System is unhealthy
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   example: unhealthy
 *                 message:
 *                   type: string
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 */
// Health endpoints will be moved before authentication middleware

// Helper functions for health checks
async function checkDatabaseConnection() {
  try {
    // Replace with actual database check
    const result = { status: 'healthy' as const, message: 'Database connection successful' };
    return result;
  } catch (error) {
    logger.error('Database check failed', { error });
    return { status: 'unhealthy' as const, message: 'Database connection failed' };
  }
}

async function checkPythonServer() {
  try {
    // Import the Python bridge service to check actual health
    const { checkPythonService } = await import('./services/python-http-bridge');
    const isHealthy = await checkPythonService();
    return { 
      status: isHealthy ? 'healthy' : 'unhealthy', 
      message: isHealthy ? 'Python server is responding' : 'Python server is not responding' 
    };
  } catch (error) {
    logger.error('Python server check failed', { error });
    return { status: 'unhealthy', message: 'Python server is not responding' };
  }
}

async function checkFileSystem() {
  try {
    // Check if we can write to the filesystem
    const testPath = path.join(os.tmpdir(), 'healthcheck');
    await fs.promises.writeFile(testPath, 'test');
    await fs.promises.unlink(testPath);
    return { status: 'healthy', message: 'File system is writable' };
  } catch (error) {
    logger.error('Filesystem check failed', { error });
    return { status: 'unhealthy', message: 'Filesystem is not writable' };
  }
}

function checkMemoryUsage() {
  const memoryUsage = process.memoryUsage();
  const maxMemory = os.totalmem();
  const usedMemory = memoryUsage.heapUsed;
  const memoryPercentage = (usedMemory / maxMemory) * 100;
  
  if (memoryPercentage > 90) {
    return { status: 'unhealthy', message: 'Memory usage critical', usage: memoryPercentage };
  } else if (memoryPercentage > 70) {
    return { status: 'degraded', message: 'Memory usage high', usage: memoryPercentage };
  }
  
  return { status: 'healthy', message: 'Memory usage normal', usage: memoryPercentage };
}

/**
 * @swagger
 * /health/detailed:
 *   get:
 *     summary: Detailed health check with history
 *     description: Get comprehensive health information including current status, historical data, and detailed component metrics
 *     tags: [Health]
 *     responses:
 *       200:
 *         description: Detailed health information
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 status:
 *                   type: string
 *                   enum: [healthy, degraded, unhealthy]
 *                 timestamp:
 *                   type: string
 *                   format: date-time
 *                 current:
 *                   type: object
 *                   description: Current health status of all components
 *                 history:
 *                   type: array
 *                   description: Last 10 health checks
 *                   items:
 *                     type: object
 *                 version:
 *                   type: string
 *       500:
 *         description: Health check failed
 */
app.get('/health/detailed', async (_req, res) => {
  try {
    const currentHealth = healthMonitor.getCurrentHealth();
    const healthHistory = healthMonitor.getHealthHistory(10); // Last 10 checks
    const overallStatus = healthMonitor.getOverallStatus();

    if (!currentHealth) {
      return res.json({
        status: 'initializing',
        message: 'Health monitoring is initializing',
        timestamp: new Date().toISOString(),
        version: '2.0.0'
      });
    }

    res.json({
      status: overallStatus,
      timestamp: new Date().toISOString(),
      current: currentHealth,
      history: healthHistory,
      version: '2.0.0'
    });
  } catch (error) {
    logger.error('Detailed health check error:', error);
    res.status(500).json({
      status: 'error',
      error: (error as Error).message,
      timestamp: new Date().toISOString()
    });
  }
});

// Python AI server health check
app.get('/api/health/python', async (_req, res) => {
  try {
    const pythonStatus = await checkPythonServer();
    const isHealthy = pythonStatus.status === 'healthy';
    res.json({
      success: isHealthy,
      status: pythonStatus.status,
      message: pythonStatus.message,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      status: 'error',
      message: 'Python AI server health check failed',
      error: (error as Error).message
    });
  }
});

// Request logging middleware
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use(createRequestLogger() as any);

// Request ID middleware
app.use(requestIdMiddleware);

// Swagger API Documentation
import { setupSwagger, swaggerSpec } from './config/swagger';

// Setup Swagger documentation
setupSwagger(app as unknown as ExpressApp);

// Serve Swagger spec as JSON
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.get('/api-docs.json', (req: any, res: any) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});

// Main server startup
async function startServer() {
  try {
    // Validate environment variables first
    try {
      validateEnvironment();
    } catch (error) {
      logger.error('Environment validation failed', { error });
      process.exit(1);
    }

    // Validate configuration
    try {
      // Validate critical configuration
      const requiredVars = [
        'PORT',
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY',
        'JWT_SECRET'
      ];
      
      const missingVars = requiredVars.filter(varName => !process.env[varName]);
      
      if (missingVars.length > 0) {
        logger.error('❌ Missing required environment variables:', missingVars);
        logger.error('Please check your .env file and add these variables');
        process.exit(1);
      }
      
      // Validate Supabase URL format
      try {
        new URL(process.env.SUPABASE_URL!);
      } catch (error) {
        logger.error('❌ Invalid SUPABASE_URL format:', process.env.SUPABASE_URL);
        process.exit(1);
      }
      
      logger.info('✅ Configuration validation passed');
    } catch (error) {
      logger.error('Configuration validation failed', { error });
      process.exit(1);
    }

    logConfigurationSummary();
    logger.info('✅ Server initialized with Supabase - no database initialization needed');

    // Start the server
    const serverInstance = httpServer.listen(config.PORT, '0.0.0.0', () => {
      // eslint-disable-next-line no-console
      console.log(`🚀 Server running on http://0.0.0.0:${config.PORT}`);
      // eslint-disable-next-line no-console
      console.log(`📊 Health check: http://0.0.0.0:${config.PORT}/health`);
      // eslint-disable-next-line no-console
      console.log(`📚 API docs: http://0.0.0.0:${config.PORT}/api-docs`);
    });

    // Handle graceful shutdown
    serverInstance.on('close', () => {
      logger.info('Server closed');
    });

    return serverInstance;
  } catch (error) {
    logError(error as Error, { context: 'server_startup' });
    process.exit(1);
  }
}

// ... (rest of the code remains the same)

/**
 * Logs the application configuration summary
 */
function logConfigurationSummary() {
  // eslint-disable-next-line no-console
  console.log('\n=== Application Configuration ===');
  // eslint-disable-next-line no-console
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  // eslint-disable-next-line no-console
  console.log(`Port: ${process.env.PORT || 5001}`);
  // eslint-disable-next-line no-console
  console.log(`Database: ${process.env.DATABASE_URL ? 'Configured' : 'Not configured'}`);
  // eslint-disable-next-line no-console
  console.log(`Redis: ${process.env.REDIS_URL ? 'Configured' : 'Not configured'}`);
  // eslint-disable-next-line no-console
  console.log(`CORS: ${process.env.ENABLE_CORS === 'true' ? 'Enabled' : 'Disabled'}`);
  // eslint-disable-next-line no-console
  console.log('===============================\n');
}

// Start the server
startServer()
  .then((serverInstance) => {
    logConfigurationSummary();
    
    // Initialize WebSocket manager if enabled
    if (webSocketManager && process.env.ENABLE_WEBSOCKETS === 'true') {
      try {
        // webSocketManager is already an initialized instance
        (webSocketManager as { initialize: (server: import('http').Server) => void }).initialize(serverInstance);
        logger.info('WebSocket server initialized');
      } catch (error) {
        logger.error('Failed to initialize WebSocket manager:', error);
      }
    }
    
    // Handle graceful shutdown
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully');
      serverInstance.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });

    process.on('SIGINT', () => {
      logger.info('SIGINT received, shutting down gracefully');
      serverInstance.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });
  })
  .catch((error) => {
    // eslint-disable-next-line no-console
    console.error('Failed to start server:', error);
    process.exit(1);
  });
