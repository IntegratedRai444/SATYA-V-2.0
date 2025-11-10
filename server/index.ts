import 'dotenv/config';
import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic } from "./vite";
import { initializeDatabase } from "./database-init";
import { pythonBridge } from "./services/python-bridge";
import promClient from 'prom-client';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

// Services
import metrics from './services/prometheus-metrics';
import alertingSystem from './services/alerting-system';
import auditLogger from './services/audit-logger';

// Security Configuration
import { configureSecurity, getSecurityConfig } from './config/security-config';

// Import new configuration system
import { 
  config,
  serverConfig,
  pythonConfig,
  features,
  validateConfiguration,
  logConfigurationSummary,
  logger,
  logError
} from './config';
import { createRequestLogger } from './config/logger';

// Import middleware
import {
  securityHeaders,
  corsPreflight
} from './middleware/auth-middleware';
import { 
  errorHandler, 
  notFoundHandler, 
  requestIdMiddleware,
  setupGracefulShutdown 
} from './middleware/error-handler';
import { webSocketManager } from './services/websocket-manager';
import { db } from './db';

const app = express();

// Rate Limiter Configurations
// Authentication endpoints - strict limits to prevent brute force
const authRateLimit = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 requests per window per IP
  message: 'Too many authentication attempts from this IP, please try again after 15 minutes',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: false,
  skipFailedRequests: false,
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

// Security middleware - must be first
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
    },
  },
  crossOriginEmbedderPolicy: false,
}));

// CORS Configuration
const corsOptions: cors.CorsOptions = {
  origin: (origin: string | undefined, callback: (err: Error | null, allow?: boolean) => void) => {
    // Allow requests with no origin (like mobile apps, curl, etc.)
    if (!origin) return callback(null, true);
    
    const allowedOrigins = [
      'http://localhost:5173',
      'http://127.0.0.1:5173',
      'http://localhost:3000',
      'http://127.0.0.1:3000'
    ];
    
    if (allowedOrigins.indexOf(origin) === -1) {
      const msg = `The CORS policy for this site does not allow access from the specified origin: ${origin}`;
      return callback(new Error(msg), false);
    }
    
    return callback(null, true);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Requested-With',
    'Accept',
    'X-Access-Token',
    'X-Refresh-Token'
  ],
  exposedHeaders: [
    'Content-Range',
    'X-Content-Range',
    'X-Access-Token',
    'X-Refresh-Token'
  ],
  maxAge: 600, // 10 minutes
  preflightContinue: false,
  optionsSuccessStatus: 204
};

// Apply CORS middleware
app.use(cors(corsOptions));
app.options('*', cors(corsOptions)); // Enable preflight for all routes

// Handle preflight requests
app.use((req: Request, res: Response, next: NextFunction) => {
  if (req.method === 'OPTIONS') {
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, PATCH, DELETE, OPTIONS');
    const headers = Array.isArray(corsOptions.allowedHeaders) 
      ? corsOptions.allowedHeaders.join(',') 
      : corsOptions.allowedHeaders || '';
    res.header('Access-Control-Allow-Headers', headers);
    return res.status(204).end();
  }
  next();
});
app.use(securityHeaders);
app.use(corsPreflight);

// Configure security middleware
configureSecurity(app);

// JSON and URL-encoded body parsing with security limits
const securityConfig = getSecurityConfig();
app.use(express.json({ limit: securityConfig.api.jsonLimit }));
app.use(express.urlencoded({ 
  extended: true, 
  limit: securityConfig.api.requestSizeLimit 
}));

// Trust first proxy (if behind a reverse proxy like nginx)
app.set('trust proxy', 1);

// Enhanced monitoring setup
app.use(metrics.httpMetricsMiddleware());
app.use(auditLogger.auditMiddleware());

// Advanced rate limiting for different endpoints
app.use('/api/auth/', authRateLimit);
app.use('/api/analyze/', analysisRateLimit);
app.use('/api/upload/', uploadRateLimit);
app.use('/api/', apiRateLimit);

// Connect alerting system to metrics
alertingSystem.on('checkRules', () => {
  // Get current metrics for rule checking
  const currentMetrics = {
    python_server_status: 1, // This would be updated by health checks
    database_status: 1,
    memory_usage_bytes: process.memoryUsage(),
    analysis_errors_total: 0, // This would be tracked by analysis endpoints
    analysis_requests_total: 0,
    analysis_queue_length: 0,
    http_request_duration_seconds: 0,
    system_uptime_seconds: process.uptime(),
    active_sessions: 0
  };
  
  alertingSystem.checkRules(currentMetrics);
});

// Prometheus metrics setup (if enabled)
if (features.enableMetrics) {
  const collectDefaultMetrics = promClient.collectDefaultMetrics;
  collectDefaultMetrics();
}

// Import health monitor
import { healthMonitor } from './services/health-monitor';

// Robust health check endpoint
app.get('/health', async (_req, res) => {
  try {
    const currentHealth = healthMonitor.getCurrentHealth();
    const overallStatus = healthMonitor.getOverallStatus();
    
    if (!currentHealth) {
      return res.status(503).json({
        status: 'unhealthy',
        message: 'Health monitoring not yet initialized',
        timestamp: new Date().toISOString()
      });
    }

    const statusCode = overallStatus === 'healthy' ? 200 : 
                      overallStatus === 'degraded' ? 200 : 503;

    res.status(statusCode).json({
      status: overallStatus,
      timestamp: currentHealth.timestamp.toISOString(),
      version: '2.0.0',
      components: {
        pythonServer: {
          status: currentHealth.pythonServer.status,
          responseTime: currentHealth.pythonServer.responseTime,
          circuitBreakerState: currentHealth.pythonServer.circuitBreakerState
        },
        database: {
          status: currentHealth.database.status,
          responseTime: currentHealth.database.responseTime
        },
        websocket: {
          status: currentHealth.websocket.status,
          connections: currentHealth.websocket.connectionCount,
          users: currentHealth.websocket.connectedUsers
        },
        fileSystem: {
          status: currentHealth.fileSystem.status,
          diskUsage: currentHealth.fileSystem.diskUsage
        },
        memory: {
          status: currentHealth.memory.status,
          usage: `${currentHealth.memory.percentage.toFixed(1)}%`
        },
        processing: {
          status: currentHealth.processing.status,
          queuedJobs: currentHealth.processing.queuedJobs,
          processingJobs: currentHealth.processing.processingJobs,
          errorRate: `${currentHealth.processing.errorRate.toFixed(1)}%`
        }
      }
    });
  } catch (error) {
    res.status(500).json({ 
      status: 'unhealthy', 
      error: (error as Error).message,
      timestamp: new Date().toISOString()
    });
  }
});

// Detailed health endpoint for monitoring systems
app.get('/health/detailed', async (_req, res) => {
  try {
    const currentHealth = healthMonitor.getCurrentHealth();
    const healthHistory = healthMonitor.getHealthHistory(10); // Last 10 checks
    const overallStatus = healthMonitor.getOverallStatus();

    res.json({
      status: overallStatus,
      timestamp: new Date().toISOString(),
      current: currentHealth,
      history: healthHistory,
      version: '2.0.0'
    });
  } catch (error) {
    res.status(500).json({
      status: 'unhealthy',
      error: (error as Error).message,
      timestamp: new Date().toISOString()
    });
  }
});

// Request logging middleware
app.use(createRequestLogger());

// Request ID middleware
app.use(requestIdMiddleware);

// 404 handler (must be after all routes)
app.use(notFoundHandler);

// Global error handler (must be last)
app.use(errorHandler);

// Database initialization is now handled by database-init.ts

// Main server startup
async function startServer() {
  try {
    // Setup graceful shutdown handling
    setupGracefulShutdown();
    
    // Log configuration summary
    logConfigurationSummary();
    
    // Validate configuration
    const configValid = await validateConfiguration();
    if (!configValid) {
      logger.error('Configuration validation failed');
      process.exit(1);
    }
    
    // Initialize database
    const dbInitialized = await initializeDatabase();
    if (!dbInitialized) {
      logger.error('Failed to initialize database');
      process.exit(1);
    }
    
    // Register API routes
    const server = await registerRoutes(app);
    
    // Setup Vite for development
    if (config.NODE_ENV !== 'production') {
      await setupVite(app, server);
    } else {
      // Serve static files in production
      serveStatic(app);
    }
    
    // Start Python bridge
    await pythonBridge.startPythonServer();
    
    server.listen(config.PORT, () => {
      logger.info(`🚀 SatyaAI Server started successfully`, {
        port: config.PORT,
        environment: config.NODE_ENV,
        metricsEnabled: features.enableMetrics,
        pythonServer: pythonConfig.url
      });
      
      // Initialize WebSocket server
      webSocketManager.initialize(server);
      logger.info(`🔌 WebSocket server initialized at ws://localhost:${config.PORT}/ws`);
      
      if (features.enableMetrics) {
        logger.info(`📊 Metrics available at http://localhost:${config.PORT}/metrics`);
      }
      logger.info(`🏥 Health check at http://localhost:${config.PORT}/health`);
    });
    
    return server;
  } catch (error) {
    logError(error as Error, { context: 'server_startup' });
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully...');
  pythonBridge.stopPythonServer();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully...');
  pythonBridge.stopPythonServer();
  process.exit(0);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  logError(error, { context: 'uncaught_exception' });
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection', {
    reason: reason,
    promise: promise,
    context: 'unhandled_rejection'
  });
});

// Start the server
startServer().catch(console.error);
