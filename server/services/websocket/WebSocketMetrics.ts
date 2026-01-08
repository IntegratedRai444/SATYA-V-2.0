import { metrics } from '../../monitoring/metrics';
import { logger } from '../../config/logger';

export class WebSocketMetrics {
  static connectionOpened() {
    metrics.websocket.connections.inc({ status: 'connected' });
    logger.info('WebSocket connection opened', {
      timestamp: new Date().toISOString(),
      activeConnections: metrics.websocket.connections
    });
  }

  static connectionClosed() {
    metrics.websocket.connections.dec({ status: 'connected' });
    logger.info('WebSocket connection closed', {
      timestamp: new Date().toISOString(),
      activeConnections: metrics.websocket.connections
    });
  }

  static messageReceived(messageType: string) {
    metrics.websocket.messages.inc({ type: messageType, status: 'received' });
  }

  static messageProcessed(messageType: string, duration: number) {
    metrics.websocket.messages.inc({ type: messageType, status: 'processed' });
    metrics.websocket.messageProcessingTime
      .labels(messageType)
      .observe(duration);
    
    if (duration > 1) {
      logger.warn('Slow WebSocket message processing', {
        messageType,
        duration,
        timestamp: new Date().toISOString()
      });
    }
  }

  static errorOccurred(errorType: string, error: Error) {
    metrics.errors.inc({
      type: `websocket_${errorType}`,
      severity: 'high',
      component: 'websocket'
    });
    
    logger.error('WebSocket error', {
      errorType,
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    });
  }
}
