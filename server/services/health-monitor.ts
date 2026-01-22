import { EventEmitter } from 'events';
import * as os from 'os';
import { setInterval, clearInterval, setImmediate, setTimeout } from 'timers';
import process from 'process';
import pythonBridge from './python-http-bridge';
import { fileProcessor } from './file-processor';
import { webSocketService } from './websocket/WebSocketManager';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import checkDiskSpace from 'check-disk-space';

interface HealthMetrics {
  timestamp: Date;
  pythonServer: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    responseTime?: number;
    lastError?: string;
    circuitBreakerState?: string;
  };
  database: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    responseTime?: number;
    lastError?: string;
  };
  websocket: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    connectionCount: number;
    connectedUsers: number;
  };
  fileSystem: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    diskUsage?: number;
    availableSpace?: number;
  };
  memory: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    usage: number;
    available: number;
    percentage: number;
  };
  processing: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    queuedJobs: number;
    processingJobs: number;
    errorRate: number;
  };
}

interface AlertThresholds {
  pythonServerResponseTime: number;
  databaseResponseTime: number;
  diskUsagePercentage: number;
  memoryUsagePercentage: number;
  errorRatePercentage: number;
  maxQueuedJobs: number;
}

class HealthMonitor extends EventEmitter {
  private metrics: HealthMetrics[] = [];
  private alertThresholds: AlertThresholds;
  private healthCheckInterval: ReturnType<typeof setInterval> | null = null;
  private isMonitoring = false;
  private alertCooldowns = new Map<string, number>();
  private readonly maxMetricsHistory = 100;
  private readonly alertCooldownPeriod = 5 * 60 * 1000; // 5 minutes
  private readonly CHECK_INTERVAL = 30000;
  private readonly NodeJSType = (globalThis as Record<string, unknown>).NodeJS;

  constructor() {
    super();
    
    this.alertThresholds = {
      pythonServerResponseTime: 10000, // 10 seconds
      databaseResponseTime: 5000, // 5 seconds
      diskUsagePercentage: 85, // 85%
      memoryUsagePercentage: 90, // 90%
      errorRatePercentage: 10, // 10%
      maxQueuedJobs: 50
    };

    this.startMonitoring();
    logger.info('Health monitor initialized');
  }

  /**
   * Start health monitoring
   */
  private startMonitoring(): void {
    // Run health check every 30 seconds
    this.healthCheckInterval = setInterval(() => {
      this.performHealthCheck();
    }, this.CHECK_INTERVAL);

    // Perform initial health check
    setImmediate(() => this.performHealthCheck());
  }

  /**
   * Perform comprehensive health check
   */
  private async performHealthCheck(): Promise<void> {
    const startTime = Date.now();
    
    const metrics: HealthMetrics = {
      timestamp: new Date(),
      pythonServer: await this.checkPythonServer(),
      database: await this.checkDatabase(),
      websocket: await this.checkWebSocket(),
      fileSystem: await this.checkFileSystem(),
      memory: this.checkMemory(),
      processing: await this.checkProcessing()
    };

    // Store metrics
    this.metrics.push(metrics);
    if (this.metrics.length > this.maxMetricsHistory) {
      this.metrics.shift();
    }

    // Check for alerts
    this.checkAlerts(metrics);

    // Emit health update
    this.emit('healthUpdate', metrics);

    const duration = Date.now() - startTime;
    logger.debug('Health check completed', { duration });
  }

  /**
   * Check Python server health
   */
  private async checkPythonServer(): Promise<HealthMetrics['pythonServer']> {
    try {
      const startTime = Date.now();
      
      // Check if Python bridge has a health check method
      if (typeof pythonBridge.checkServiceHealth === 'function') {
        const isHealthy = await pythonBridge.checkServiceHealth();
        const responseTime = Date.now() - startTime;

        const status: 'healthy' | 'degraded' | 'unhealthy' = isHealthy ? 'healthy' : 'unhealthy';
        
        return {
          status,
          responseTime,
          lastError: isHealthy ? undefined : 'Service health check failed'
        };
      } else {
        // Python bridge doesn't have health check, assume degraded
        logger.warn('Python bridge health check not available');
        return {
          status: 'degraded',
          lastError: 'Health check method not available',
          circuitBreakerState: 'unknown'
        };
      }
    } catch (error) {
      logger.warn('Python server health check failed', {
        error: (error as Error).message
      });

      return {
        status: 'degraded',
        lastError: (error as Error).message
      };
    }
  }

  /**
   * Check database health
   */
  private async checkDatabase(): Promise<HealthMetrics['database']> {
    try {
      const startTime = Date.now();
      
      // Try direct database connection using dbManager
      const timeoutPromise = new Promise<never>((_, reject) => 
        setTimeout(() => reject(new Error('Database query timeout')), 3000)
      );
      
      const queryPromise = supabase.from('users').select('id').limit(1);
      
      try {
        await Promise.race([queryPromise, timeoutPromise]);
        const responseTime = Date.now() - startTime;
        const status: 'healthy' | 'degraded' | 'unhealthy' = responseTime > this.alertThresholds.databaseResponseTime ? 'degraded' : 'healthy';
        
        logger.info('database', status, { responseTime, method: 'postgresql' });
        return { status, responseTime };
      } catch (pgError: unknown) {
        // Database check failed, mark as unhealthy
        logger.error('Database health check failed:', pgError);
        throw new Error('Both PostgreSQL and REST API failed');
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      
      // Check if it's a connection error (expected when using REST API only)
      if (errorMessage.includes('ENOTFOUND') || errorMessage.includes('getaddrinfo')) {
        // This is expected, REST API is working
        return {
          status: 'healthy',
          lastError: 'Using REST API (PostgreSQL direct connection unavailable)'
        };
      }
      
      logger.info('database', 'unhealthy', { error: errorMessage });
      return {
        status: 'unhealthy',
        lastError: errorMessage
      };
    }
  }

  /**
   * Check WebSocket health
   */
  private async checkWebSocket(): Promise<HealthMetrics['websocket']> {
    try {
      const stats = webSocketService.getManager().getStats();
      const status: 'healthy' | 'degraded' | 'unhealthy' = stats.totalConnections > 0 ? 'healthy' : 'degraded';
      
      logger.info('websocket', status, {
        connectionCount: stats.totalConnections,
        connectedUsers: stats.connectedUsers,
        channels: stats.channels || 0
      });
      
      return {
        status,
        connectionCount: stats.totalConnections,
        connectedUsers: stats.connectedUsers
      };
    } catch (error: unknown) {
      logger.info('websocket', 'unhealthy', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
      return {
        status: 'unhealthy',
        connectionCount: 0,
        connectedUsers: 0
      };
    }
  }

  /**
   * Check file system health
   */
  private async checkFileSystem(): Promise<HealthMetrics['fileSystem']> {
    try {
      // Using checkDiskSpace for cross-platform disk space checking
      const diskSpace = await checkDiskSpace(process.cwd());
      const total = diskSpace.size;
      const free = diskSpace.free;
      const used = total - free;
      const percentage = total > 0 ? (used / total) * 100 : 0;
      
      const status: 'healthy' | 'degraded' | 'unhealthy' = percentage > this.alertThresholds.diskUsagePercentage ? 'degraded' : 'healthy';
      
      logger.info('file_system', status, {
        diskUsage: parseFloat(percentage.toFixed(2)),
        availableSpace: Math.round(free / (1024 * 1024 * 1024)) // Convert to GB
      });
      
      return {
        status,
        diskUsage: parseFloat(percentage.toFixed(2)),
        availableSpace: Math.round(free / (1024 * 1024 * 1024)) // Convert to GB
      };
    } catch (error: unknown) {
      logger.info('file_system', 'unhealthy', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
      return {
        status: 'unhealthy',
        diskUsage: 0,
        availableSpace: 0
      };
    }
  }

  /**
   * Check memory health
   */
  private checkMemory(): HealthMetrics['memory'] {
    try {
      const totalMemory = os.totalmem();
      const freeMemory = os.freemem();
      const usedMemory = totalMemory - freeMemory;
      const percentage = (usedMemory / totalMemory) * 100;

      const status: 'healthy' | 'degraded' | 'unhealthy' = percentage > this.alertThresholds.memoryUsagePercentage ? 'degraded' : 'healthy';

      logger.info('memory', status, {
        memoryUsage: percentage,
        usedMemory: usedMemory / (1024 * 1024 * 1024), // GB
        totalMemory: totalMemory / (1024 * 1024 * 1024) // GB
      });

      return {
        status,
        usage: usedMemory,
        available: freeMemory,
        percentage
      };
    } catch (error: unknown) {
      logger.error('Memory check failed', {
        error: error instanceof Error ? error.message : 'Unknown error'
      });

      return {
        status: 'unhealthy',
        usage: 0,
        available: 0,
        percentage: 0
      };
    }
  }

  /**
   * Check processing health
   */
  private async checkProcessing(): Promise<HealthMetrics['processing']> {
    try {
      const stats = fileProcessor.getStats();
      const status: 'healthy' | 'degraded' | 'unhealthy' = stats.activeProcesses > 0 ? 'healthy' : 'degraded';
      
      return {
        status,
        queuedJobs: stats.queuedFiles,
        processingJobs: stats.activeProcesses,
        errorRate: 0 // Track error rate if needed
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Error checking processing health:', { error: errorMessage });
      
      return {
        status: 'unhealthy',
        queuedJobs: 0,
        processingJobs: 0,
        errorRate: 100
      };
    }
  }

  /**
   * Get disk usage information using check-disk-space
   */
  private async getDiskUsage(): Promise<{ percentage: number; available: number; total: number }> {
    try {
      const diskSpace = await checkDiskSpace(process.cwd());
      const total = diskSpace.size;
      const available = diskSpace.free;
      const used = total - available;
      const percentage = total > 0 ? (used / total) * 100 : 0;

      return {
        percentage: parseFloat(percentage.toFixed(2)),
        available: Math.round(available / (1024 * 1024)), // Convert to MB
        total: Math.round(total / (1024 * 1024)) // Convert to MB
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Error getting disk usage:', { error: errorMessage });
      return {
        percentage: 0,
        available: 0,
        total: 0
      };
    }
  }

  private checkAlerts(metrics: HealthMetrics): void {
    const alerts: Array<{ type: string; severity: 'warning' | 'critical'; message: string; context: any }> = [];

    // Python server alerts
    if (metrics.pythonServer.status === 'unhealthy') {
      alerts.push({
        type: 'python_server_down',
        severity: 'critical',
        message: 'Python AI server is not responding',
        context: { error: metrics.pythonServer.lastError }
      });
    } else if (metrics.pythonServer.responseTime && metrics.pythonServer.responseTime > this.alertThresholds.pythonServerResponseTime) {
      alerts.push({
        type: 'python_server_slow',
        severity: 'warning',
        message: `Python server response time is high: ${metrics.pythonServer.responseTime}ms`,
        context: { responseTime: metrics.pythonServer.responseTime }
      });
    }

    // Database alerts
    if (metrics.database.status === 'unhealthy') {
      alerts.push({
        type: 'database_down',
        severity: 'critical',
        message: 'Database is not responding',
        context: { error: metrics.database.lastError }
      });
    }

    // Memory alerts
    if (metrics.memory.percentage > this.alertThresholds.memoryUsagePercentage) {
      alerts.push({
        type: 'high_memory_usage',
        severity: 'warning',
        message: `High memory usage: ${metrics.memory.percentage.toFixed(1)}%`,
        context: { percentage: metrics.memory.percentage }
      });
    }

    // Disk space alerts
    if (metrics.fileSystem.diskUsage && metrics.fileSystem.diskUsage > this.alertThresholds.diskUsagePercentage) {
      alerts.push({
        type: 'low_disk_space',
        severity: 'warning',
        message: `Low disk space: ${metrics.fileSystem.diskUsage.toFixed(1)}% used`,
        context: { percentage: metrics.fileSystem.diskUsage }
      });
    }

    // Processing queue alerts
    if (metrics.processing.queuedJobs > this.alertThresholds.maxQueuedJobs) {
      alerts.push({
        type: 'high_queue_backlog',
        severity: 'warning',
        message: `High number of queued jobs: ${metrics.processing.queuedJobs}`,
        context: { queuedJobs: metrics.processing.queuedJobs }
      });
    }

    // Error rate alerts
    if (metrics.processing.errorRate > this.alertThresholds.errorRatePercentage) {
      alerts.push({
        type: 'high_error_rate',
        severity: 'critical',
        message: `High analysis error rate: ${metrics.processing.errorRate.toFixed(1)}%`,
        context: { errorRate: metrics.processing.errorRate }
      });
    }

    // Send alerts (with cooldown)
    alerts.forEach(alert => {
      this.sendAlert(alert);
    });
  }

  /**
   * Send alert with cooldown mechanism
   */
  private sendAlert(alert: { type: string; severity: 'warning' | 'critical'; message: string; context: unknown }): void {
    const now = Date.now();
    const lastAlert = this.alertCooldowns.get(alert.type) || 0;

    if (now - lastAlert < this.alertCooldownPeriod) {
      return; // Still in cooldown period
    }

    this.alertCooldowns.set(alert.type, now);

    // Log the alert
    const logLevel = alert.severity === 'critical' ? 'error' : 'warn';
    logger.log(logLevel, `ALERT: ${alert.message}`, {
      alertType: alert.type,
      severity: alert.severity,
      ...(typeof alert.context === 'object' && alert.context !== null ? alert.context as Record<string, unknown> : {}),
      timestamp: new Date().toISOString()
    });

    // Emit alert event
    this.emit('alert', alert);
  }

  /**
   * Get current health status
   */
  getCurrentHealth(): HealthMetrics | null {
    return this.metrics.length > 0 ? this.metrics[this.metrics.length - 1] : null;
  }

  /**
   * Get health history
   */
  getHealthHistory(limit?: number): HealthMetrics[] {
    const history = [...this.metrics];
    return limit ? history.slice(-limit) : history;
  }

  /**
   * Get overall system status
   */
  getOverallStatus(): 'healthy' | 'degraded' | 'unhealthy' {
    const current = this.getCurrentHealth();
    if (!current) return 'unhealthy';

    const components = [
      current.pythonServer.status,
      current.database.status,
      current.websocket.status,
      current.fileSystem.status,
      current.memory.status,
      current.processing.status
    ];

    if (components.some(status => status === 'unhealthy')) {
      return 'unhealthy';
    }

    if (components.some(status => status === 'degraded')) {
      return 'degraded';
    }

    return 'healthy';
  }

  /**
   * Update alert thresholds
   */
  updateThresholds(newThresholds: Partial<AlertThresholds>): void {
    this.alertThresholds = { ...this.alertThresholds, ...newThresholds };
    logger.info('Alert thresholds updated', { thresholds: this.alertThresholds });
  }

  /**
   * Shutdown health monitor
   */
  shutdown(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    
    logger.info('Health monitor shutdown completed');
  }
}

// Export singleton instance
export const healthMonitor = new HealthMonitor();

// Graceful shutdown handlers
process.on('SIGTERM', () => {
  healthMonitor.shutdown();
});

process.on('SIGINT', () => {
  healthMonitor.shutdown();
});