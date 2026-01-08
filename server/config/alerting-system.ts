import { EventEmitter } from 'events';
import { logger } from './logger';

class AlertingSystem extends EventEmitter {
  constructor() {
    super();
    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    this.on('error', (error: Error) => {
      logger.error('Alerting system error:', error);
    });

    this.on('alert', (alert: any) => {
      logger.warn('ALERT:', alert);
      // Add additional alert handling logic here (e.g., send email, SMS, etc.)
    });
  }

  public init() {
    logger.info('Alerting system initialized');
  }

  public alert(level: 'info' | 'warn' | 'error', message: string, metadata: Record<string, any> = {}) {
    this.emit('alert', {
      timestamp: new Date().toISOString(),
      level,
      message,
      ...metadata
    });
  }
}

export const alertingSystem = new AlertingSystem();

export default alertingSystem;
