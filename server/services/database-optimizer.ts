import { dbManager } from '../db';
import { logger } from '../config/logger';

interface User {
  id: number;
  username: string;
  email: string | null;
  password: string;
  full_name: string | null;
  api_key: string | null;
  role: string;
  failed_login_attempts: number;
  last_failed_login: string | null;
  is_locked: boolean;
  lockout_until: string | null;
  created_at: string;
  updated_at: string;
}

interface Scan {
  id: number;
  user_id: number | null;
  filename: string;
  type: string;
  result: string;
  confidence_score: number;
  detection_details: string | null;
  metadata: string | null;
  created_at: string;
  updated_at: string;
}

interface QueryCache {
  key: string;
  data: any;
  timestamp: number;
  ttl: number;
}

interface DatabaseStats {
  totalUsers: number;
  totalScans: number;
  activeUsers: number;
  recentScans: number;
  databaseSize: number;
  queryPerformance: {
    averageQueryTime: number;
    slowQueries: number;
  };
}

class DatabaseOptimizer {
  private queryCache: Map<string, QueryCache> = new Map();
  private queryTimes: number[] = [];
  private cleanupInterval: NodeJS.Timeout | null = null;
  private readonly maxCacheSize = 1000;
  private readonly defaultTTL = 5 * 60 * 1000; // 5 minutes
  private readonly slowQueryThreshold = 1000; // 1 second

  constructor() {
    this.startCacheCleanup();
    logger.info('Database optimizer initialized');
  }

  /**
   * Start cache cleanup process
   */
  private startCacheCleanup(): void {
    // Clean up expired cache entries every 2 minutes
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpiredCache();
    }, 2 * 60 * 1000);
  }

  /**
   * Execute query with caching and performance monitoring
   */
  async executeQuery<T>(
    queryKey: string,
    queryFn: () => Promise<T>,
    ttl: number = this.defaultTTL,
    useCache: boolean = true
  ): Promise<T> {
    const startTime = Date.now();

    try {
      // Check cache first
      if (useCache) {
        const cached = this.getFromCache<T>(queryKey);
        if (cached) {
          const queryTime = Date.now() - startTime;
          this.recordQueryTime(queryTime);
          return cached;
        }
      }

      // Execute query
      const result = await queryFn();
      const queryTime = Date.now() - startTime;

      // Record performance
      this.recordQueryTime(queryTime);

      // Log slow queries
      if (queryTime > this.slowQueryThreshold) {
        logger.warn('Slow query detected', {
          queryKey,
          queryTime,
          threshold: this.slowQueryThreshold
        });
      }

      // Cache result
      if (useCache && result) {
        this.setCache(queryKey, result, ttl);
      }

      return result;
    } catch (error) {
      const queryTime = Date.now() - startTime;
      this.recordQueryTime(queryTime);

      logger.error('Database query failed', {
        queryKey,
        queryTime,
        error: (error as Error).message
      });

      throw error;
    }
  }

  /**
   * Set value in cache
   */
  private setCache(key: string, data: any, ttl: number = this.defaultTTL): void {
    if (this.queryCache.size >= this.maxCacheSize) {
      // Remove the first (oldest) entry to make space
      const firstKey = this.queryCache.keys().next().value;
      if (firstKey) {
        this.queryCache.delete(firstKey);
      }
    }
    this.queryCache.set(key, {
      key,
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  /**
   * Get from cache
   */
  private getFromCache<T>(key: string): T | null {
    const cached = this.queryCache.get(key);
    if (!cached) return null;
    
    // Check if cache entry has expired
    if (Date.now() > cached.timestamp + cached.ttl) {
      this.queryCache.delete(key);
      return null;
    }
    
    return cached.data as T;
  }

  /**
   * Clean up expired cache entries
   */
  private cleanupExpiredCache(): void {
    const now = Date.now();
    let cleanedCount = 0;

    for (const [key, cached] of this.queryCache.entries()) {
      if (now > cached.timestamp + cached.ttl) {
        this.queryCache.delete(key);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      logger.info(`Cleaned up ${cleanedCount} expired cache entries`);
    }
  }

  /**
   * Get user by ID with caching
   */
  async getUserById(userId: number): Promise<User> {
    return this.executeQuery<User>(
      `user:${userId}`,
      async () => {
        const users = await dbManager.find('users', { id: userId }, { limit: 1 });
        if (!users || users.length === 0) {
          throw new Error(`User with ID ${userId} not found`);
        }
        return users[0] as User;
      },
      10 * 60 * 1000 // 10 minutes TTL for user data
    );
  }

  /**
   * Get user scans with caching
   */
  async getUserScans(
    userId: number, 
    limit: number = 20, 
    offset: number = 0
  ): Promise<Scan[]> {
    return this.executeQuery<Scan[]>(
      `user_scans:${userId}:${limit}:${offset}`,
      async () => {
        const scans = await dbManager.find('scans', 
          { user_id: userId },
          { 
            orderBy: { column: 'created_at', ascending: false },
            limit,
            offset 
          }
        );
        return scans as unknown as Scan[];
      },
      2 * 60 * 1000 // 2 minutes TTL for scan data
    );
  }

  /**
   * Get dashboard statistics with caching
   */
  async getDashboardStats(userId: string): Promise<{
    totalScans: number;
    recentScans: number;
    lastScanDate: Date | null;
  }> {
    return this.executeQuery(
      `dashboard_stats:${userId}`,
      async () => {
        // Get total scans count
        const allScans = await dbManager.find('scans', { user_id: userId });
        const totalScans = allScans.length;
        
        // Get recent scans (last 7 days)
        const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
        const recentScans = await dbManager.find('scans', 
          { 
            user_id: userId,
            created_at: { $gt: oneWeekAgo },
          },
          { 
            orderBy: { column: 'created_at', ascending: false },
            limit: 5 
          }
        ) as Array<{ created_at: string }>;

        return {
          totalScans,
          recentScans: recentScans.length,
          lastScanDate: recentScans[0]?.created_at ? new Date(recentScans[0].created_at) : null
        };
      },
      5 * 60 * 1000 // 5 minutes TTL
    );
  }

  /**
   * Get system statistics
   */
  async getSystemStats(): Promise<DatabaseStats> {
    return this.executeQuery(
      'system_stats',
      async () => {
        // Get total users count
        const allUsers = await dbManager.find('users', {});
        const totalUsers = allUsers.length;
        
        // Get total scans count
        const allScans = await dbManager.find('scans', {});
        const totalScans = allScans.length;
        
        // Get active users (logged in within last 7 days)
        const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
        const activeUsers = (await dbManager.find('users', {
          last_login: { $gt: oneWeekAgo },
        })).length;
        
        // Get recent scans (last 24 hours)
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
        const recentScans = (await dbManager.find('scans', {
          created_at: { $gt: oneDayAgo },
        })).length;

        // Calculate average query time and slow queries
        const averageQueryTime = this.queryTimes.length > 0 
          ? this.queryTimes.reduce((a, b) => a + b, 0) / this.queryTimes.length 
          : 0;

        const slowQueries = this.queryTimes.filter(time => time > this.slowQueryThreshold).length;

        return {
          totalUsers,
          totalScans,
          activeUsers,
          recentScans,
          databaseSize: 0, // Would need database-specific implementation
          queryPerformance: {
            averageQueryTime,
            slowQueries
          }
        };
      },
      5 * 60 * 1000 // 5 minutes TTL
    );
  }

  /**
   * Record query execution time
   */
  private recordQueryTime(time: number): void {
    this.queryTimes.push(time);
    
    // Keep only last 1000 query times
    if (this.queryTimes.length > 1000) {
      this.queryTimes.shift();
    }
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): {
    size: number;
    maxSize: number;
    hitRate: number;
    averageQueryTime: number;
    slowQueries: number;
  } {
    const averageQueryTime = this.queryTimes.length > 0 
      ? this.queryTimes.reduce((a, b) => a + b, 0) / this.queryTimes.length 
      : 0;

    const slowQueries = this.queryTimes.filter(time => time > this.slowQueryThreshold).length;

    return {
      size: this.queryCache.size,
      maxSize: this.maxCacheSize,
      hitRate: 0, // Would need to track hits vs misses
      averageQueryTime,
      slowQueries
    };
  }

  /**
   * Shutdown database optimizer
   */
  shutdown(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }

    this.queryCache.clear();
    logger.info('Database optimizer shutdown completed');
  }
}

// Export singleton instance
export const databaseOptimizer = new DatabaseOptimizer();

// Graceful shutdown handlers
process.on('SIGTERM', () => {
  databaseOptimizer.shutdown();
});

process.on('SIGINT', () => {
  databaseOptimizer.shutdown();
});