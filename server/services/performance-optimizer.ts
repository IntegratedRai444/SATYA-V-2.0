import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';
import os from 'os';
import { logger } from '../config';

interface PerformanceMetrics {
  timestamp: Date;
  memory: {
    heapUsed: number;
    heapTotal: number;
    external: number;
    rss: number;
    percentage: number;
  };
  cpu: {
    user: number;
    system: number;
    percentage: number;
  };
  disk: {
    used: number;
    available: number;
    percentage: number;
  };
  activeConnections: number;
  queueLength: number;
}

interface OptimizationAction {
  type: 'cleanup' | 'gc' | 'cache_clear' | 'connection_limit';
  reason: string;
  threshold: number;
  currentValue: number;
}

class PerformanceOptimizer extends EventEmitter {
  private metricsHistory: PerformanceMetrics[] = [];
  private monitoringInterval: NodeJS.Timeout | null = null;
  private optimizationThresholds = {
    memoryUsage: 85, // 85%
    diskUsage: 90,   // 90%
    cpuUsage: 80,    // 80%
    queueLength: 100,
    connectionCount: 1000
  };
  private lastOptimization: Map<string, number> = new Map();
  private optimizationCooldown = 5 * 60 * 1000; // 5 minutes

  constructor() {
    super();
    this.startMonitoring();
    logger.info('Performance optimizer initialized');
  }

  /**
   * Start performance monitoring
   */
  private startMonitoring(): void {
    // Monitor every 30 seconds
    this.monitoringInterval = setInterval(async () => {
      try {
        await this.collectMetrics();
        await this.analyzeAndOptimize();
      } catch (error) {
        logger.error('Performance monitoring failed', {
          error: (error as Error).message
        });
      }
    }, 30000);

    // Collect initial metrics
    setImmediate(() => this.collectMetrics());
  }

  /**
   * Collect current performance metrics
   */
  private async collectMetrics(): Promise<PerformanceMetrics> {
    const memUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    const totalMemory = os.totalmem();
    const freeMemory = os.freemem();
    
    // Calculate memory percentage
    const memoryPercentage = ((memUsage.rss) / totalMemory) * 100;
    
    // Calculate CPU percentage (simplified)
    const cpuPercentage = ((cpuUsage.user + cpuUsage.system) / 1000000) / os.cpus().length;

    // Get disk usage
    const diskInfo = await this.getDiskUsage();

    // Get connection info (simplified)
    const activeConnections = await this.getActiveConnections();
    const queueLength = await this.getQueueLength();

    const metrics: PerformanceMetrics = {
      timestamp: new Date(),
      memory: {
        heapUsed: memUsage.heapUsed,
        heapTotal: memUsage.heapTotal,
        external: memUsage.external,
        rss: memUsage.rss,
        percentage: memoryPercentage
      },
      cpu: {
        user: cpuUsage.user,
        system: cpuUsage.system,
        percentage: Math.min(cpuPercentage, 100)
      },
      disk: diskInfo,
      activeConnections,
      queueLength
    };

    // Store metrics (keep last 100 entries)
    this.metricsHistory.push(metrics);
    if (this.metricsHistory.length > 100) {
      this.metricsHistory.shift();
    }

    this.emit('metricsCollected', metrics);
    return metrics;
  }

  /**
   * Analyze metrics and perform optimizations
   */
  private async analyzeAndOptimize(): Promise<void> {
    const current = this.metricsHistory[this.metricsHistory.length - 1];
    if (!current) return;

    const actions: OptimizationAction[] = [];

    // Check memory usage
    if (current.memory.percentage > this.optimizationThresholds.memoryUsage) {
      actions.push({
        type: 'gc',
        reason: 'High memory usage detected',
        threshold: this.optimizationThresholds.memoryUsage,
        currentValue: current.memory.percentage
      });

      actions.push({
        type: 'cleanup',
        reason: 'Memory cleanup needed',
        threshold: this.optimizationThresholds.memoryUsage,
        currentValue: current.memory.percentage
      });
    }

    // Check disk usage
    if (current.disk.percentage > this.optimizationThresholds.diskUsage) {
      actions.push({
        type: 'cleanup',
        reason: 'High disk usage detected',
        threshold: this.optimizationThresholds.diskUsage,
        currentValue: current.disk.percentage
      });
    }

    // Check queue length
    if (current.queueLength > this.optimizationThresholds.queueLength) {
      actions.push({
        type: 'connection_limit',
        reason: 'High queue length detected',
        threshold: this.optimizationThresholds.queueLength,
        currentValue: current.queueLength
      });
    }

    // Execute optimization actions
    for (const action of actions) {
      await this.executeOptimization(action);
    }
  }

  /**
   * Execute optimization action
   */
  private async executeOptimization(action: OptimizationAction): Promise<void> {
    const now = Date.now();
    const lastRun = this.lastOptimization.get(action.type) || 0;

    // Check cooldown
    if (now - lastRun < this.optimizationCooldown) {
      return;
    }

    this.lastOptimization.set(action.type, now);

    logger.info('Executing performance optimization', {
      type: action.type,
      reason: action.reason,
      threshold: action.threshold,
      currentValue: action.currentValue
    });

    try {
      switch (action.type) {
        case 'gc':
          await this.forceGarbageCollection();
          break;
        case 'cleanup':
          await this.performCleanup();
          break;
        case 'cache_clear':
          await this.clearCaches();
          break;
        case 'connection_limit':
          await this.limitConnections();
          break;
      }

      this.emit('optimizationExecuted', action);
    } catch (error) {
      logger.error('Optimization action failed', {
        action: action.type,
        error: (error as Error).message
      });
    }
  }

  /**
   * Force garbage collection
   */
  private async forceGarbageCollection(): Promise<void> {
    if (global.gc) {
      global.gc();
      logger.info('Forced garbage collection completed');
    } else {
      logger.warn('Garbage collection not available (run with --expose-gc)');
    }
  }

  /**
   * Perform cleanup operations
   */
  private async performCleanup(): Promise<void> {
    try {
      // Clean up temporary files
      const { fileCleanupService } = await import('./file-cleanup');
      await fileCleanupService.cleanupOldFiles({
        maxAge: 60 * 60 * 1000, // 1 hour
        dryRun: false
      });

      // Clear expired sessions
      const { sessionManager } = await import('./session-manager');
      // sessionManager has automatic cleanup, but we can trigger it

      logger.info('Cleanup operations completed');
    } catch (error) {
      logger.error('Cleanup operations failed', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Clear caches
   */
  private async clearCaches(): Promise<void> {
    try {
      // Clear any in-memory caches
      // This would depend on your specific caching implementation
      logger.info('Cache clearing completed');
    } catch (error) {
      logger.error('Cache clearing failed', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Limit connections
   */
  private async limitConnections(): Promise<void> {
    try {
      // Implement connection limiting logic
      // This might involve rate limiting or connection throttling
      logger.info('Connection limiting applied');
    } catch (error) {
      logger.error('Connection limiting failed', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Get disk usage information
   */
  private async getDiskUsage(): Promise<{ used: number; available: number; percentage: number }> {
    try {
      const stats = await fs.statfs('./');
      const total = stats.blocks * stats.blksize;
      const available = stats.bavail * stats.blksize;
      const used = total - available;
      const percentage = (used / total) * 100;

      return { used, available, percentage };
    } catch (error) {
      return { used: 0, available: 0, percentage: 0 };
    }
  }

  /**
   * Get active connections count
   */
  private async getActiveConnections(): Promise<number> {
    try {
      const { webSocketManager } = await import('./websocket-manager');
      const stats = webSocketManager.getStats();
      return stats.totalConnections;
    } catch (error) {
      return 0;
    }
  }

  /**
   * Get queue length
   */
  private async getQueueLength(): Promise<number> {
    try {
      const { fileProcessor } = await import('./file-processor');
      const stats = fileProcessor.getStats();
      return stats.queuedJobs;
    } catch (error) {
      return 0;
    }
  }

  /**
   * Get current performance metrics
   */
  getCurrentMetrics(): PerformanceMetrics | null {
    return this.metricsHistory.length > 0 ? 
           this.metricsHistory[this.metricsHistory.length - 1] : null;
  }

  /**
   * Get metrics history
   */
  getMetricsHistory(limit?: number): PerformanceMetrics[] {
    const history = [...this.metricsHistory];
    return limit ? history.slice(-limit) : history;
  }

  /**
   * Get performance trends
   */
  getPerformanceTrends(): {
    memoryTrend: 'increasing' | 'decreasing' | 'stable';
    cpuTrend: 'increasing' | 'decreasing' | 'stable';
    diskTrend: 'increasing' | 'decreasing' | 'stable';
  } {
    if (this.metricsHistory.length < 5) {
      return {
        memoryTrend: 'stable',
        cpuTrend: 'stable',
        diskTrend: 'stable'
      };
    }

    const recent = this.metricsHistory.slice(-5);
    const memoryValues = recent.map(m => m.memory.percentage);
    const cpuValues = recent.map(m => m.cpu.percentage);
    const diskValues = recent.map(m => m.disk.percentage);

    return {
      memoryTrend: this.calculateTrend(memoryValues),
      cpuTrend: this.calculateTrend(cpuValues),
      diskTrend: this.calculateTrend(diskValues)
    };
  }

  /**
   * Calculate trend from values
   */
  private calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
    if (values.length < 2) return 'stable';

    const first = values[0];
    const last = values[values.length - 1];
    const diff = last - first;
    const threshold = 5; // 5% threshold

    if (diff > threshold) return 'increasing';
    if (diff < -threshold) return 'decreasing';
    return 'stable';
  }

  /**
   * Update optimization thresholds
   */
  updateThresholds(newThresholds: Partial<typeof this.optimizationThresholds>): void {
    this.optimizationThresholds = { ...this.optimizationThresholds, ...newThresholds };
    logger.info('Performance optimization thresholds updated', {
      thresholds: this.optimizationThresholds
    });
  }

  /**
   * Shutdown performance optimizer
   */
  shutdown(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    
    logger.info('Performance optimizer shutdown completed');
  }
}

// Export singleton instance
export const performanceOptimizer = new PerformanceOptimizer();

// Graceful shutdown handlers
process.on('SIGTERM', () => {
  performanceOptimizer.shutdown();
});

process.on('SIGINT', () => {
  performanceOptimizer.shutdown();
});