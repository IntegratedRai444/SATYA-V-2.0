import { logger } from '../config/logger';
import { cacheManager } from './cache-manager';

interface MemoryStats {
  heapUsed: number;
  heapTotal: number;
  external: number;
  rss: number;
  heapUsedMB: number;
  heapTotalMB: number;
  externalMB: number;
  rssMB: number;
  heapUsagePercent: number;
}

class MemoryMonitor {
  private monitorInterval: NodeJS.Timeout | null = null;
  private highMemoryThreshold: number = 1024; // 1GB in MB
  private criticalMemoryThreshold: number = 1536; // 1.5GB in MB
  private lastGCTime: number = Date.now();
  private gcCooldown: number = 300000; // 5 minutes

  /**
   * Start memory monitoring
   */
  start(intervalMs: number = 60000): void {
    if (this.monitorInterval) {
      logger.warn('Memory monitor already running');
      return;
    }

    this.monitorInterval = setInterval(() => {
      this.checkMemory();
    }, intervalMs);

    logger.info('Memory monitor started', { intervalMs });
  }

  /**
   * Stop memory monitoring
   */
  stop(): void {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval);
      this.monitorInterval = null;
      logger.info('Memory monitor stopped');
    }
  }

  /**
   * Get current memory statistics
   */
  getStats(): MemoryStats {
    const usage = process.memoryUsage();
    
    return {
      heapUsed: usage.heapUsed,
      heapTotal: usage.heapTotal,
      external: usage.external,
      rss: usage.rss,
      heapUsedMB: Math.round(usage.heapUsed / 1024 / 1024),
      heapTotalMB: Math.round(usage.heapTotal / 1024 / 1024),
      externalMB: Math.round(usage.external / 1024 / 1024),
      rssMB: Math.round(usage.rss / 1024 / 1024),
      heapUsagePercent: Math.round((usage.heapUsed / usage.heapTotal) * 100)
    };
  }

  /**
   * Check memory and take action if needed
   */
  private checkMemory(): void {
    const stats = this.getStats();
    
    // Log memory stats periodically
    logger.debug('Memory stats', {
      heapUsedMB: stats.heapUsedMB,
      heapTotalMB: stats.heapTotalMB,
      heapUsagePercent: stats.heapUsagePercent,
      rssMB: stats.rssMB
    });

    // Check for high memory usage
    if (stats.heapUsedMB > this.criticalMemoryThreshold) {
      logger.error('CRITICAL: Memory usage exceeds threshold', {
        heapUsedMB: stats.heapUsedMB,
        threshold: this.criticalMemoryThreshold,
        heapUsagePercent: stats.heapUsagePercent
      });
      
      // Aggressive cleanup
      this.performCleanup(true);
    } else if (stats.heapUsedMB > this.highMemoryThreshold) {
      logger.warn('High memory usage detected', {
        heapUsedMB: stats.heapUsedMB,
        threshold: this.highMemoryThreshold,
        heapUsagePercent: stats.heapUsagePercent
      });
      
      // Normal cleanup
      this.performCleanup(false);
    }
  }

  /**
   * Perform memory cleanup
   */
  private performCleanup(aggressive: boolean = false): void {
    const now = Date.now();
    
    // Check GC cooldown
    if (now - this.lastGCTime < this.gcCooldown) {
      logger.debug('Skipping cleanup - in cooldown period');
      return;
    }

    logger.info('Performing memory cleanup', { aggressive });

    try {
      // Clear old cache entries
      if (aggressive) {
        // Clear more aggressively - reduce cache size by 50%
        const stats = cacheManager.getStats();
        const targetSize = Math.floor(stats.totalKeys / 2);
        
        logger.info('Aggressive cache cleanup', {
          currentSize: stats.totalKeys,
          targetSize
        });
        
        // Clear half the cache
        const keys = cacheManager.keys();
        for (let i = 0; i < targetSize && i < keys.length; i++) {
          cacheManager.delete(keys[i]);
        }
      } else {
        // Normal cleanup - just remove expired entries
        cacheManager['cleanupExpired']();
      }

      // Trigger garbage collection if available
      if (global.gc) {
        logger.info('Triggering garbage collection');
        global.gc();
        this.lastGCTime = now;
        
        // Log memory after GC
        setTimeout(() => {
          const afterStats = this.getStats();
          logger.info('Memory after GC', {
            heapUsedMB: afterStats.heapUsedMB,
            heapUsagePercent: afterStats.heapUsagePercent
          });
        }, 1000);
      } else {
        logger.warn('Garbage collection not available - run with --expose-gc flag');
      }
    } catch (error) {
      logger.error('Error during memory cleanup', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Force garbage collection (if available)
   */
  forceGC(): boolean {
    if (global.gc) {
      const before = this.getStats();
      global.gc();
      const after = this.getStats();
      
      logger.info('Forced garbage collection', {
        beforeMB: before.heapUsedMB,
        afterMB: after.heapUsedMB,
        freedMB: before.heapUsedMB - after.heapUsedMB
      });
      
      return true;
    }
    
    logger.warn('Garbage collection not available');
    return false;
  }
}

export const memoryMonitor = new MemoryMonitor();

// Start monitoring in production
if (process.env.NODE_ENV === 'production') {
  memoryMonitor.start(60000); // Check every minute
} else if (process.env.NODE_ENV === 'development') {
  memoryMonitor.start(300000); // Check every 5 minutes in dev
}
