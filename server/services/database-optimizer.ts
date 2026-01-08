import { db } from '../db';
import { users, scans } from '@shared/schema';
import { eq, lt, and, desc, sql } from 'drizzle-orm';
import { logger } from '../config';

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
   * Get user by ID with caching
   */
  async getUserById(userId: number): Promise<any> {
    return this.executeQuery(
      `user:${userId}`,
      () => db.select().from(users).where(eq(users.id, userId)).limit(1),
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
  ): Promise<any[]> {
    return this.executeQuery(
      `user_scans:${userId}:${limit}:${offset}`,
      () => db.select()
        .from(scans)
        .where(eq(scans.userId, userId))
        .orderBy(desc(scans.createdAt))
        .limit(limit)
        .offset(offset),
      2 * 60 * 1000 // 2 minutes TTL for scan data
    );
  }

  /**
   * Get dashboard statistics with caching
   */
  async getDashboardStats(userId: number): Promise<any> {
    return this.executeQuery(
      `dashboard_stats:${userId}`,
      async () => {
        const [userScansCount, recentScans] = await Promise.all([
          db.select({ count: sql<number>`count(*)` })
            .from(scans)
            .where(eq(scans.userId, userId)),
          db.select()
            .from(scans)
            .where(and(
              eq(scans.userId, userId),
              lt(scans.createdAt, new Date(Date.now() - 7 * 24 * 60 * 60 * 1000))
            ))
            .limit(5)
        ]);

        return {
          totalScans: userScansCount[0]?.count || 0,
          recentScans: recentScans.length,
          lastScanDate: recentScans[0]?.createdAt || null
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
        const [totalUsersResult, totalScansResult, activeUsersResult, recentScansResult] = await Promise.all([
          db.select({ count: sql<number>`count(*)` }).from(users),
          db.select({ count: sql<number>`count(*)` }).from(scans),
          // Using updatedAt as a proxy for last login time since lastLoginAt doesn't exist
          db.select({ count: sql<number>`count(distinct ${users.id})` })
            .from(users)
            .where(lt(users.updatedAt, new Date(Date.now() - 7 * 24 * 60 * 60 * 1000))),
          db.select({ count: sql<number>`count(*)` })
            .from(scans)
            .where(lt(scans.createdAt, new Date(Date.now() - 24 * 60 * 60 * 1000)))
        ]);

        const averageQueryTime = this.queryTimes.length > 0 
          ? this.queryTimes.reduce((a, b) => a + b, 0) / this.queryTimes.length 
          : 0;

        const slowQueries = this.queryTimes.filter(time => time > this.slowQueryThreshold).length;

        return {
          totalUsers: totalUsersResult[0]?.count || 0,
          totalScans: totalScansResult[0]?.count || 0,
          activeUsers: activeUsersResult[0]?.count || 0,
          recentScans: recentScansResult[0]?.count || 0,
          databaseSize: 0, // Would need database-specific implementation
          queryPerformance: {
            averageQueryTime,
            slowQueries
          }
        };
      },
      60 * 1000 // 1 minute TTL
    );
  }

  /**
   * Optimize database by cleaning up old data
   */
  async optimizeDatabase(): Promise<{
    deletedScans: number;
    deletedUsers: number;
    optimizedQueries: number;
  }> {
    logger.info('Starting database optimization');

    const results = {
      deletedScans: 0,
      deletedUsers: 0,
      optimizedQueries: 0
    };

    try {
      // Clean up old scans (older than 90 days)
      const oldScansThreshold = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);
      const deletedScans = await db.delete(scans)
        .where(lt(scans.createdAt, oldScansThreshold))
        .returning({ count: sql<number>`count(*)` });
      
      results.deletedScans = deletedScans[0]?.count || 0;

      // Clean up inactive users (no login for 1 year and no scans)
      const inactiveUsersThreshold = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000);
      
      // First, find users with no recent activity
      const inactiveUsers = await db.select({ id: users.id })
        .from(users)
        .where(and(
          lt(users.updatedAt, inactiveUsersThreshold),
          sql`${users.id} NOT IN (SELECT DISTINCT user_id FROM scans WHERE created_at > ${inactiveUsersThreshold})`
        ));

      // Delete inactive users
      if (inactiveUsers.length > 0) {
        const userIds = inactiveUsers.map(u => u.id);
        for (const userId of userIds) {
          await db.delete(users).where(eq(users.id, userId));
          results.deletedUsers++;
        }
      }

      // Clear query cache to force fresh data
      this.clearCache();
      results.optimizedQueries = this.queryCache.size;

      logger.info('Database optimization completed', results);

      return results;
    } catch (error) {
      logger.error('Database optimization failed', {
        error: (error as Error).message
      });
      throw error;
    }
  }

  /**
   * Create database indexes for better performance
   */
  async createOptimalIndexes(): Promise<void> {
    try {
      // Note: Index creation would be database-specific
      // For SQLite, you might use raw SQL queries
      
      const indexQueries = [
        'CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans(user_id)',
        'CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at)',
        'CREATE INDEX IF NOT EXISTS idx_users_updated_at ON users(updated_at)',
        'CREATE INDEX IF NOT EXISTS idx_scans_user_created ON scans(user_id, created_at)'
      ];

      for (const query of indexQueries) {
        await db.execute(sql.raw(query));
      }

      logger.info('Database indexes created successfully');
    } catch (error) {
      logger.error('Failed to create database indexes', {
        error: (error as Error).message
      });
    }
  }

  /**
   * Get from cache
   */
  private getFromCache<T>(key: string): T | null {
    const cached = this.queryCache.get(key);
    if (!cached) return null;

    if (Date.now() > cached.timestamp + cached.ttl) {
      this.queryCache.delete(key);
      return null;
    }

    return cached.data as T;
  }

  /**
   * Set cache
   */
  private setCache(key: string, data: any, ttl: number): void {
    // Implement LRU eviction if cache is full
    if (this.queryCache.size >= this.maxCacheSize) {
      const oldestKey = this.queryCache.keys().next().value;
      if (oldestKey) {
        this.queryCache.delete(oldestKey);
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
   * Clear cache
   */
  clearCache(pattern?: string): void {
    if (pattern) {
      const regex = new RegExp(pattern);
      for (const [key] of this.queryCache) {
        if (regex.test(key)) {
          this.queryCache.delete(key);
        }
      }
    } else {
      this.queryCache.clear();
    }

    logger.info('Database cache cleared', { pattern });
  }

  /**
   * Clean up expired cache entries
   */
  private cleanupExpiredCache(): void {
    const now = Date.now();
    let cleanedCount = 0;

    for (const [key, cached] of this.queryCache) {
      if (now > cached.timestamp + cached.ttl) {
        this.queryCache.delete(key);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      logger.debug('Cleaned up expired cache entries', { cleanedCount });
    }
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