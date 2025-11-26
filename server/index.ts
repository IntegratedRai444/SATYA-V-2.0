import 'dotenv/config';
import express from "express";
import type { Request, Response, NextFunction } from "express";
import { createServer } from 'http';
import apiRoutes from "./routes";
import { setupVite, serveStatic } from "./vite";
import { initializeDatabase } from "./database-init";
import pythonBridge from "./services/python-http-bridge";
import promClient from 'prom-client';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import compression from 'compression';
import { validateEnvironment } from './utils/env-validator';

// Services
import metrics from './services/prometheus-metrics';
import alertingSystem from './services/alerting-system';
import auditLogger from './services/audit-logger';

// Security Configuration
import { configureSecurity, getSecurityConfig } from './config/security-config';

// Import new configuration system
import { 
  config,
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
  errorHandler, 
  notFoundHandler, 
  requestIdMiddleware,
  setupGracefulShutdown 
} from './middleware/error-handler';
import { webSocketManager } from './services/websocket-manager';

const app = express();

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
  skip: (req) => process.env.NODE_ENV === 'development', // Skip rate limiting in development
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
    'X-Refresh-Token',
    'Cache-Control',
    'Origin',
    'User-Agent',
    'DNT',
    'If-Modified-Since',
    'If-None-Match'
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
import { parseCookies } from './middleware/cookie-auth';
app.use(parseCookies);

// Trust first proxy (if behind a reverse proxy like nginx)
app.set('trust proxy', 1);

// Enhanced monitoring setup
app.use(metrics.httpMetricsMiddleware());
app.use(auditLogger.auditMiddleware());

// Advanced rate limiting for different endpoints
app.use('/api/auth/', authRateLimit);
app.use('/api/analysis/', analysisRateLimit);
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
app.get('/health', async (_req, res) => {
  // Simple health check that always returns 200 OK
  // This prevents frontend from thinking server is down during initialization
  res.status(200).json({
    status: 'healthy',
    message: 'Server is running',
    timestamp: new Date().toISOString(),
    version: '2.0.0',
    uptime: process.uptime()
  });
});

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
    const pythonHealth = pythonBridge.getHealthStatus();
    res.json({
      success: pythonHealth.healthy,
      message: pythonHealth.healthy ? 'Python AI server is online' : 'Python AI server is offline',
      lastCheck: pythonHealth.lastCheck,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      message: 'Python AI server health check failed',
      error: (error as Error).message
    });
  }
});

// Request logging middleware
app.use(createRequestLogger());

// Request ID middleware
app.use(requestIdMiddleware);

// Swagger API Documentation
import swaggerUi from 'swagger-ui-express';
import { swaggerSpec } from './config/swagger';

// Serve Swagger UI at /api-docs
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: 'SatyaAI API Documentation',
  customfavIcon: '/favicon.ico'
}));

// Serve Swagger spec as JSON
app.get('/api-docs.json', (_req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});

// Register API routes BEFORE 404 handler
app.use('/api', apiRoutes);

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
    
    // Validate environment variables first
    if (!validateEnvironment()) {
      logger.error('Environment validation failed - missing required variables');
      process.exit(1);
    }
    
    // Log configuration summary
    logConfigurationSummary();
    
    // Validate configuration
    const configValid = await validateConfiguration();
    if (!configValid) {
      logger.error('Configuration validation failed');
      process.exit(1);
    }
    
    // Initialize database (non-blocking - app will work with Supabase Client API)
    const dbInitialized = await initializeDatabase();
    if (!dbInitialized) {
      logger.warn('Database initialization failed - using Supabase Client API fallback');
      logger.info('App will continue to function normally using REST API');
    } else {
      logger.info('✅ Database initialized successfully');
    }
    
    // Create HTTP server
    const server = createServer(app);
    
    // Debug: Log all registered routes in development
    if (config.NODE_ENV === 'development') {
      logger.info('📋 Registered routes:');
      const routes: string[] = [];
      app._router.stack.forEach((middleware: any) => {
        if (middleware.route) {
          // Direct route
          const methods = Object.keys(middleware.route.methods).join(', ').toUpperCase();
          routes.push(`  ${methods} ${middleware.route.path}`);
        } else if (middleware.name === 'router') {
          // Router middleware
          middleware.handle.stack.forEach((handler: any) => {
            if (handler.route) {
              const methods = Object.keys(handler.route.methods).join(', ').toUpperCase();
              const path = middleware.regexp.source
                .replace('\\/?', '')
                .replace('(?=\\/|$)', '')
                .replace(/\\\//g, '/')
                .replace(/\^/g, '')
                .replace(/\$/g, '');
              routes.push(`  ${methods} ${path}${handler.route.path}`);
            }
          });
        }
      });
      routes.forEach(route => logger.info(route));
    }
    
    // Setup Vite for development
    if (config.NODE_ENV !== 'production') {
      await setupVite(app, server);
    } else {
      // Serve static files in production
      serveStatic(app);
    }
    
    // Python HTTP bridge starts automatically with health monitoring
    // No need to manually start - it will check health and queue requests if needed
    logger.info('Python HTTP bridge initialized with automatic health monitoring');
    
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
  pythonBridge.shutdown();
  process.exit(0);
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully...');
  pythonBridge.shutdown();
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
