import client from 'prom-client';
import { logger } from '../config/logger';

// Create a Registry to register the metrics
const register = new client.Registry();

// Enable collection of default metrics
client.collectDefaultMetrics({ 
  register,
  prefix: 'node_',
  gcDurationBuckets: [0.1, 0.5, 1, 2, 5]
});

// HTTP Metrics
export const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.5, 1, 1.5, 2, 3, 5, 10],
  registers: [register]
});

export const httpRequestsTotal = new client.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

export const httpRequestErrors = new client.Counter({
  name: 'http_request_errors_total',
  help: 'Total number of HTTP request errors',
  labelNames: ['method', 'route', 'status_code', 'error_type'],
  registers: [register]
});

// Database Metrics
export const dbQueryDuration = new client.Histogram({
  name: 'db_query_duration_seconds',
  help: 'Duration of database queries in seconds',
  labelNames: ['operation', 'collection', 'success'],
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
  registers: [register]
});

export const dbConnections = new client.Gauge({
  name: 'db_connections',
  help: 'Number of database connections',
  labelNames: ['state'],
  registers: [register]
});

// WebSocket Metrics
export const wsConnections = new client.Gauge({
  name: 'websocket_connections',
  help: 'Number of active WebSocket connections',
  labelNames: ['status'],
  registers: [register]
});

export const wsMessages = new client.Counter({
  name: 'websocket_messages_total',
  help: 'Total number of WebSocket messages processed',
  labelNames: ['type', 'status'],
  registers: [register]
});

export const wsMessageProcessingTime = new client.Histogram({
  name: 'websocket_message_processing_seconds',
  help: 'Time taken to process WebSocket messages',
  labelNames: ['type'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 5],
  registers: [register]
});

// Error Metrics
export const errorCounter = new client.Counter({
  name: 'application_errors_total',
  help: 'Total number of application errors',
  labelNames: ['type', 'severity', 'component'],
  registers: [register]
});

// Memory Metrics
const memoryUsage = new client.Gauge({
  name: 'node_memory_usage_bytes',
  help: 'Memory usage in bytes',
  labelNames: ['type'],
  registers: [register]
});

// Update memory metrics every 5 seconds
setInterval(() => {
  const mem = process.memoryUsage();
  memoryUsage.set({ type: 'rss' }, mem.rss);
  memoryUsage.set({ type: 'heapTotal' }, mem.heapTotal);
  memoryUsage.set({ type: 'heapUsed' }, mem.heapUsed);
  memoryUsage.set({ type: 'external' }, mem.external || 0);
}, 5000).unref();

// Export all metrics
export const metrics = {
  http: {
    requestDuration: httpRequestDuration,
    requestsTotal: httpRequestsTotal,
    requestErrors: httpRequestErrors
  },
  db: {
    queryDuration: dbQueryDuration,
    connections: dbConnections
  },
  websocket: {
    connections: wsConnections,
    messages: wsMessages,
    messageProcessingTime: wsMessageProcessingTime
  },
  errors: errorCounter,
  memory: memoryUsage
};

// Log metrics initialization
logger.info('Metrics collection initialized', {
  defaultMetrics: client.collectDefaultMetrics.length,
  customMetrics: Object.keys(metrics).length,
  timestamp: new Date().toISOString()
});

// Error handling for metrics collection
process.on('uncaughtException', (error) => {
  errorCounter.inc({ 
    type: 'uncaught_exception', 
    severity: 'critical', 
    component: 'metrics'
  });
  
  logger.error('Uncaught exception in metrics collection', {
    error: error.message,
    stack: error.stack,
    timestamp: new Date().toISOString()
  });
});

export { register };
