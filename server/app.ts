import express from 'express';
import type { Request, Response, NextFunction } from 'express';
import { validateConfig } from './config/validate';
import { securityHeaders } from './middleware/security';
import { apiRateLimit as apiLimiter } from './middleware/advanced-rate-limiting';
import { register as metricsRegister } from './monitoring/metrics';
import { logger } from './config/logger';
import { webSocketService } from './services/websocket/WebSocketManager';
import { tracingMiddleware } from './middleware/tracing';
import type { AuthUser } from './middleware/auth';

declare global {
  namespace Express {
    interface Request {
      user?: AuthUser;
    }
  }
}

// Validate environment variables
validateConfig(process.env);

const app = express();
const port = process.env.PORT || 3000;

// Apply middleware
app.use(tracingMiddleware());
app.use(securityHeaders);
app.use(apiLimiter);

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', metricsRegister.contentType);
    res.end(await metricsRegister.metrics());
  } catch (error) {
    res.status(500).end('Error generating metrics');
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Create HTTP server
const server = app.listen(port, () => {
  logger.info(`Server running on port ${port}`);
  
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
});

// Error handling middleware
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  const requestId = req.requestId || 'unknown';
  const traceId = req.traceId || 'unknown';
  
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

export { app };
