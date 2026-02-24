import { Request, Response } from 'express';
import { logger } from '../config/logger';
import { setInterval } from 'timers';
import jobManager from '../services/job-manager';
import webSocketManager from '../services/websocket-manager';
import rateLimiter from '../middleware/rate-limiter';
import errorMonitor from '../services/error-monitor';

interface SystemMetrics {
  timestamp: string;
  uptime: number;
  memory: {
    used: number;
    total: number;
    percentage: number;
  };
  jobs: {
    total: number;
    running: number;
    completed: number;
    failed: number;
    averageProcessingTime: number;
  };
  websocket: {
    connectedClients: number;
    totalConnections: number;
  };
  rateLimit: {
    totalUsers: number;
    activeUsers: number;
  };
  errors: {
    total: number;
    bySeverity: Record<string, number>;
    byCategory: Record<string, number>;
    recent: Array<{
      timestamp: string;
      message: string;
      severity: string;
    }>;
  };
}

class MetricsCollector {
  private startTime = Date.now();
  private jobMetrics = {
    total: 0,
    completed: 0,
    failed: 0,
    totalProcessingTime: 0
  };

  // Record job start
  recordJobStart(): void {
    this.jobMetrics.total++;
  }

  // Record job completion
  recordJobCompleted(processingTime: number): void {
    this.jobMetrics.completed++;
    this.jobMetrics.totalProcessingTime += processingTime;
  }

  // Record job failure
  recordJobFailed(): void {
    this.jobMetrics.failed++;
  }

  // Get current metrics
  getMetrics(): SystemMetrics {
    const now = Date.now();
    const uptime = now - this.startTime;
    
    // Memory usage
    const memUsage = process.memoryUsage();
    const memory = {
      used: memUsage.heapUsed,
      total: memUsage.heapTotal,
      percentage: Math.round((memUsage.heapUsed / memUsage.heapTotal) * 100)
    };

    // Job metrics
    const runningJobs = jobManager.getRunningJobsCount();
    const averageProcessingTime = this.jobMetrics.completed > 0 
      ? this.jobMetrics.totalProcessingTime / this.jobMetrics.completed 
      : 0;

    const jobs = {
      total: this.jobMetrics.total,
      running: runningJobs,
      completed: this.jobMetrics.completed,
      failed: this.jobMetrics.failed,
      averageProcessingTime: Math.round(averageProcessingTime)
    };

    // WebSocket metrics
    const websocketStats = webSocketManager.getConnectionStats();
    const websocket = {
      connectedClients: websocketStats.totalConnections,
      totalConnections: websocketStats.totalConnections
    };

    // Rate limit metrics
    const rateLimitStats = rateLimiter.getStats();
    const rateLimit = {
      totalUsers: rateLimitStats.totalUsers,
      activeUsers: rateLimitStats.activeUsers
    };

    // Error metrics
    const errorStats = errorMonitor.getErrorStats();
    const errors = {
      total: errorStats.total,
      bySeverity: errorStats.bySeverity,
      byCategory: errorStats.byCategory,
      recent: errorStats.recent.slice(-5) // Last 5 errors
    };

    return {
      timestamp: new Date().toISOString(),
      uptime,
      memory,
      jobs,
      websocket,
      rateLimit,
      errors
    };
  }

  // Reset metrics (call periodically)
  reset(): void {
    this.jobMetrics = {
      total: 0,
      completed: 0,
      failed: 0,
      totalProcessingTime: 0
    };
    logger.info('[METRICS] Metrics reset');
  }
}

// Singleton instance
const metricsCollector = new MetricsCollector();

// Reset metrics every hour
setInterval(() => {
  metricsCollector.reset();
}, 60 * 60 * 1000);

export default metricsCollector;

// Metrics endpoint handler
export const getSystemMetrics = async (req: Request, res: Response) => {
  try {
    const metrics = metricsCollector.getMetrics();
    
    logger.info('[METRICS] System metrics requested', {
      uptime: metrics.uptime,
      memoryUsage: metrics.memory.percentage,
      activeJobs: metrics.jobs.running
    });

    res.json({
      success: true,
      data: metrics
    });

  } catch (error) {
    logger.error('[METRICS] Failed to get system metrics', error);
    
    res.status(500).json({
      success: false,
      error: {
        code: 'METRICS_ERROR',
        message: 'Failed to retrieve system metrics'
      }
    });
  }
};

// Export functions for recording metrics
export const recordJobStart = () => metricsCollector.recordJobStart();
export const recordJobCompleted = (processingTime: number) => metricsCollector.recordJobCompleted(processingTime);
export const recordJobFailed = () => metricsCollector.recordJobFailed();
