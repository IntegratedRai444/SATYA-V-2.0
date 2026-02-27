import { logger } from '../config/logger';
import { cacheManager } from './cache-manager';
import { setInterval, clearInterval } from 'timers';
import * as os from 'os';

type Timer = ReturnType<typeof setInterval>;

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
  systemLoadAvg: number[];
  cpuUsage: number;
  uptime: number;
}

interface LongRunningJobStats {
  jobId: string;
  startTime: number;
  peakMemoryMB: number;
  currentMemoryMB: number;
  memoryGrowthRate: number;
  isAtRisk: boolean;
}

class MemoryMonitor {
  private monitorInterval: Timer | null = null;
  private highMemoryThreshold: number = 1024; // 1GB in MB
  private criticalMemoryThreshold: number = 1536; // 1.5GB in MB
  private lastGCTime: number = Date.now();
  private gcCooldown: number = 300000; // 5 minutes
  private longRunningJobs = new Map<string, LongRunningJobStats>();
  private memoryHistory: Array<{timestamp: number; memoryMB: number}> = [];
  private maxHistorySize: number = 100; // Keep last 100 measurements
  
  // LONG-RUNNING STABILITY: Track job memory patterns
  trackJobMemory(jobId: string): void {
    const stats = this.getStats();
    const jobStats: LongRunningJobStats = {
      jobId,
      startTime: Date.now(),
      peakMemoryMB: stats.heapUsedMB,
      currentMemoryMB: stats.heapUsedMB,
      memoryGrowthRate: 0,
      isAtRisk: false
    };
    
    this.longRunningJobs.set(jobId, jobStats);
    logger.debug('🔒 Started tracking job memory', { jobId, initialMemory: stats.heapUsedMB });
  }
  
  untrackJobMemory(jobId: string): void {
    const jobStats = this.longRunningJobs.get(jobId);
    if (jobStats) {
      const duration = Date.now() - jobStats.startTime;
      const finalMemory = this.getStats().heapUsedMB;
      const memoryGrowth = finalMemory - jobStats.peakMemoryMB;
      
      logger.info('🔒 Job memory tracking completed', {
        jobId,
        durationMs: duration,
        peakMemoryMB: jobStats.peakMemoryMB,
        finalMemoryMB: finalMemory,
        memoryGrowthMB: memoryGrowth,
        avgGrowthRate: jobStats.memoryGrowthRate
      });
      
      this.longRunningJobs.delete(jobId);
    }
  }
  
  // LONG-RUNNING STABILITY: Analyze memory patterns for jobs
  private analyzeLongRunningJobs(): void {
    const currentStats = this.getStats();
    
    for (const [jobId, jobStats] of this.longRunningJobs.entries()) {
      const duration = Date.now() - jobStats.startTime;
      
      // Update current memory and peak
      jobStats.currentMemoryMB = currentStats.heapUsedMB;
      jobStats.peakMemoryMB = Math.max(jobStats.peakMemoryMB, currentStats.heapUsedMB);
      
      // Calculate growth rate (MB per minute)
      if (duration > 60000) { // After 1 minute
        jobStats.memoryGrowthRate = (currentStats.heapUsedMB - jobStats.peakMemoryMB) / (duration / 60000);
      }
      
      // Check if job is at risk
      const riskFactors = [
        currentStats.heapUsedMB > this.criticalMemoryThreshold,
        jobStats.memoryGrowthRate > 50, // Growing faster than 50MB/min
        duration > 10 * 60 * 1000, // Running longer than 10 minutes
        currentStats.heapUsagePercent > 90
      ];
      
      jobStats.isAtRisk = riskFactors.some(factor => factor);
      
      if (jobStats.isAtRisk) {
        logger.warn('🔒 Long-running job at risk', {
          jobId,
          durationMs: duration,
          currentMemoryMB: currentStats.heapUsedMB,
          growthRate: jobStats.memoryGrowthRate,
          riskFactors: riskFactors.map((factor, i) => ({
            factor: ['CriticalMemory', 'HighGrowthRate', 'LongDuration', 'HighHeapUsage'][i],
            triggered: factor
          }))
        });
      }
    }
  }
  
  // LONG-RUNNING STABILITY: Get jobs at risk for proactive cleanup
  getJobsAtRisk(): Array<{jobId: string; riskFactors: string[]}> {
    const atRiskJobs: Array<{jobId: string; riskFactors: string[]}> = [];
    
    for (const [jobId, jobStats] of this.longRunningJobs.entries()) {
      if (jobStats.isAtRisk) {
        const riskFactors: string[] = [];
        const currentStats = this.getStats();
        
        if (currentStats.heapUsedMB > this.criticalMemoryThreshold) {
          riskFactors.push('CriticalMemory');
        }
        if (jobStats.memoryGrowthRate > 50) {
          riskFactors.push('HighGrowthRate');
        }
        if (Date.now() - jobStats.startTime > 10 * 60 * 1000) {
          riskFactors.push('LongDuration');
        }
        if (currentStats.heapUsagePercent > 90) {
          riskFactors.push('HighHeapUsage');
        }
        
        atRiskJobs.push({ jobId, riskFactors });
      }
    }
    
    return atRiskJobs;
  }

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
   * Get current memory statistics with system info
   */
  getStats(): MemoryStats {
    const usage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    const loadAvg = os.loadavg();
    
    return {
      heapUsed: usage.heapUsed,
      heapTotal: usage.heapTotal,
      external: usage.external,
      rss: usage.rss,
      heapUsedMB: Math.round(usage.heapUsed / 1024 / 1024),
      heapTotalMB: Math.round(usage.heapTotal / 1024 / 1024),
      externalMB: Math.round(usage.external / 1024 / 1024),
      rssMB: Math.round(usage.rss / 1024 / 1024),
      heapUsagePercent: Math.round((usage.heapUsed / usage.heapTotal) * 100),
      systemLoadAvg: loadAvg,
      cpuUsage: cpuUsage.user + cpuUsage.system,
      uptime: process.uptime()
    };
  }

  /**
   * Check memory and take action if needed
   */
  private checkMemory(): void {
    const stats = this.getStats();
    
    // Store memory history for trend analysis
    this.memoryHistory.push({
      timestamp: Date.now(),
      memoryMB: stats.heapUsedMB
    });
    
    // Keep history size bounded
    if (this.memoryHistory.length > this.maxHistorySize) {
      this.memoryHistory.shift();
    }
    
    // Analyze long-running jobs
    this.analyzeLongRunningJobs();
    
    // Log memory stats periodically in development
    if (process.env.NODE_ENV === 'development') {
      logger.debug('Memory stats', {
        heapUsedMB: stats.heapUsedMB,
        heapTotalMB: stats.heapTotalMB,
        heapUsagePercent: stats.heapUsagePercent,
        rssMB: stats.rssMB,
        longRunningJobs: this.longRunningJobs.size,
        jobsAtRisk: this.getJobsAtRisk().length
      });
    }

    // Check for high memory usage
    if (stats.heapUsedMB > this.criticalMemoryThreshold) {
      logger.error('CRITICAL: Memory usage exceeds threshold', {
        heapUsedMB: stats.heapUsedMB,
        threshold: this.criticalMemoryThreshold,
        heapUsagePercent: stats.heapUsagePercent,
        longRunningJobs: this.longRunningJobs.size,
        jobsAtRisk: this.getJobsAtRisk().length
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
      if (process.env.NODE_ENV === 'development') {
        logger.debug('Skipping cleanup - in cooldown period');
      }
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
   * Get memory trend analysis
   */
  getMemoryTrend(): {
    trend: 'increasing' | 'decreasing' | 'stable';
    rateMBPerMinute: number;
    predictedMaxMB: number;
    timeToThreshold: number | null;
  } {
    if (this.memoryHistory.length < 10) {
      return {
        trend: 'stable',
        rateMBPerMinute: 0,
        predictedMaxMB: this.getStats().heapUsedMB,
        timeToThreshold: null
      };
    }
    
    const recent = this.memoryHistory.slice(-10);
    const old = this.memoryHistory.slice(-20, -10);
    
    if (old.length === 0) {
      return {
        trend: 'stable',
        rateMBPerMinute: 0,
        predictedMaxMB: this.getStats().heapUsedMB,
        timeToThreshold: null
      };
    }
    
    const recentAvg = recent.reduce((sum, p) => sum + p.memoryMB, 0) / recent.length;
    const oldAvg = old.reduce((sum, p) => sum + p.memoryMB, 0) / old.length;
    
    const timeDiff = (recent[recent.length - 1].timestamp - old[0].timestamp) / 60000; // minutes
    const memoryDiff = recentAvg - oldAvg;
    const rate = memoryDiff / timeDiff; // MB per minute
    
    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (Math.abs(rate) > 1) {
      trend = rate > 0 ? 'increasing' : 'decreasing';
    }
    
    const currentMemory = this.getStats().heapUsedMB;
    const predictedMax = trend === 'increasing' ? currentMemory + (rate * 60) : currentMemory; // Predict 1 hour ahead
    
    let timeToThreshold: number | null = null;
    if (trend === 'increasing' && rate > 0) {
      timeToThreshold = (this.criticalMemoryThreshold - currentMemory) / rate * 60000; // milliseconds
    }
    
    return {
      trend,
      rateMBPerMinute: rate,
      predictedMaxMB: predictedMax,
      timeToThreshold
    };
  }
  
  /**
   * Get comprehensive health report
   */
  getHealthReport(): {
    stats: MemoryStats;
    trend: ReturnType<typeof MemoryMonitor.prototype.getMemoryTrend>;
    longRunningJobs: number;
    jobsAtRisk: Array<{jobId: string; riskFactors: string[]}>;
    recommendations: string[];
  } {
    const stats = this.getStats();
    const trend = this.getMemoryTrend();
    const jobsAtRisk = this.getJobsAtRisk();
    const recommendations: string[] = [];
    
    // Generate recommendations
    if (stats.heapUsagePercent > 85) {
      recommendations.push('Memory usage is high - consider reducing concurrent jobs');
    }
    
    if (trend.trend === 'increasing' && trend.rateMBPerMinute > 10) {
      recommendations.push('Memory is growing rapidly - monitor for leaks');
    }
    
    if (jobsAtRisk.length > 0) {
      recommendations.push(`${jobsAtRisk.length} jobs at risk - consider proactive cleanup`);
    }
    
    if (trend.timeToThreshold && trend.timeToThreshold < 30 * 60 * 1000) {
      recommendations.push('Approaching memory threshold - expect GC soon');
    }
    
    return {
      stats,
      trend,
      longRunningJobs: this.longRunningJobs.size,
      jobsAtRisk,
      recommendations
    };
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
