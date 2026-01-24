import express, { type Request } from 'express';
import { validateConfig } from './config/validate';
import { securityHeaders } from './middleware/security';
import { apiRateLimit as apiLimiter } from './middleware/advanced-rate-limiting';
import { register as metricsRegister } from './monitoring/metrics';
import { logger } from './config/logger';
import { webSocketService } from './services/websocket/WebSocketManager';
import { tracingMiddleware } from './middleware/tracing';
import { router } from './routes';
import { auditAuth } from './middleware/audit-logger';
import { auditLogger } from './middleware/audit-logger';
import cors from 'cors';
import cookieParser from 'cookie-parser';
import helmet from 'helmet';

// Define custom request interface
interface CustomRequest extends Request {
  requestId: string;
  traceId: string;
}

// Validate environment variables
try {
  validateConfig(process.env);
  logger.info('Configuration validation passed');
} catch (error) {
  logger.error('Configuration validation failed:', { error });
  process.exit(1);
}

// Create Express app
const app = express();
const port = parseInt(process.env.PORT || '5001');

// Trust first proxy (if behind a reverse proxy like nginx)
app.set('trust proxy', 1);

// Apply core middleware
app.use(helmet());
app.use(cors({
  origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:5173'],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-CSRF-Token']
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cookieParser());

// Apply custom middleware
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use(tracingMiddleware() as any);
app.use(securityHeaders);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use(apiLimiter as any);

// Apply global audit logging to all API routes
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use('/api/v2', auditLogger('sensitive_data_access', 'api_endpoint') as any);

// Apply audit logging to authentication routes
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use('/api/v2/auth', auditAuth.login as any);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use('/api/v2/auth', auditAuth.register as any);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use('/api/v2/auth', auditAuth.logout as any);

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', metricsRegister.contentType);
    res.end(await metricsRegister.metrics());
  } catch (error) {
    res.status(500).end('Error generating metrics');
  }
});

// Health check endpoint with service status
app.get('/health', async (req, res) => {
  const healthStatus = {
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    services: {
      database: 'checking...',
      python: 'checking...',
      websockets: process.env.ENABLE_WEBSOCKETS !== 'false' ? 'enabled' : 'disabled'
    }
  };

  // Check Python service
  try {
    const pythonResponse = await fetch(
      `${process.env.PYTHON_HEALTH_CHECK_URL || 'http://localhost:8000/health'}`,
      {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      }
    );
    healthStatus.services.python = pythonResponse.ok ? 'ok' : 'error';
  } catch (error) {
    healthStatus.services.python = 'error';
  }

  // Check database (simplified check)
  try {
    // Add actual DB check here when database service is available
    healthStatus.services.database = 'ok'; // Placeholder
  } catch (error) {
    healthStatus.services.database = 'error';
  }

  const overallStatus = Object.values(healthStatus.services).every(status => 
    status === 'ok' || status === 'enabled' || status === 'disabled'
  ) ? 'ok' : 'degraded';

  res.status(overallStatus === 'ok' ? 200 : 503).json({
    ...healthStatus,
    status: overallStatus
  });
});

// Mount API routes
app.use('/', router);

// Create HTTP server
const server = app.listen(port, async () => {
  logger.info(`Server running on port ${port}`);
  logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
  logger.info(`API routes mounted at /api/v2`);
  
  // Initialize WebSocket service when server starts
  if (process.env.ENABLE_WEBSOCKETS !== 'false') {
    try {
      webSocketService.initialize(server);
      logger.info('WebSocket service initialized');
    } catch (error) {
      logger.error('Failed to initialize WebSocket service', { 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  } else {
    logger.info('WebSocket service is disabled');
  }

  // Startup health check
  try {
    const healthCheck = await fetch(`http://localhost:${port}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000)
    });
    
    if (healthCheck.ok) {
      logger.info('Startup health check passed');
    } else {
      logger.warn('Startup health check failed');
    }
  } catch (error) {
    logger.error('Startup health check error', { error });
  }
});

// Error handling middleware
// eslint-disable-next-line @typescript-eslint/no-explicit-any
app.use((err: Error, req: any, res: any, _next: any): void => {
  // Don't use _next - this is the final error handler
  void _next; // Explicitly mark as unused
  const requestId = (req as CustomRequest).requestId || 'unknown';
  const traceId = (req as CustomRequest).traceId || 'unknown';
  
  logger.error('Unhandled error', {
    error: {
      message: err.message,
      stack: err.stack,
      name: err.name
    },
    request: {
      method: req.method,
      url: req.originalUrl,
      ip: req.ip,
      userAgent: req.get('user-agent')
    },
    requestId,
    traceId,
    timestamp: new Date().toISOString()
  });

  res.status(500).json({
    error: 'Internal server error',
    requestId,
    traceId
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received. Shutting down gracefully...');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received. Shutting down gracefully...');
  server.close(() => {
    logger.info('Server closed');
    process.exit(0);
  });
});

process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception', { error: error.message, stack: error.stack });
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection', { reason, promise });
  process.exit(1);
});

export { app };
