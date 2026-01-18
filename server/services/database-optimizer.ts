import { supabase } from '../config/supabase';
import { logger } from '../config/logger';

interface User {
  id: string;
  username: string;
  email: string | null;
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
  id: string;
  user_id: string | null;
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
  async getUserById(userId: string): Promise<User> {
    return this.executeQuery<User>(
      `user:${userId}`,
      async () => {
        const { data: users, error } = await supabase
          .from('users')
          .select('*')
          .eq('id', userId)
          .limit(1);
        
        if (error || !users || users.length === 0) {
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
    userId: string, 
    limit: number = 20, 
    offset: number = 0
  ): Promise<Scan[]> {
    return this.executeQuery<Scan[]>(
      `user_scans:${userId}:${limit}:${offset}`,
      async () => {
        const { data: scans, error } = await supabase
          .from('scans')
          .select('*')
          .eq('user_id', userId)
          .order('created_at', { ascending: false })
          .range(offset, offset + limit - 1);
        
        if (error) {
          throw new Error(`Failed to get scans for user ${userId}: ${error.message}`);
        }
        return scans as Scan[];
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
        const { data: allScans, error: totalError } = await supabase
          .from('scans')
          .select('id')
          .eq('user_id', userId);
        
        if (totalError) {
          throw new Error(`Failed to get total scans: ${totalError.message}`);
        }
        const totalScans = allScans?.length || 0;
        
        // Get recent scans (last 7 days)
        const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
        const { data: recentScans, error: recentError } = await supabase
          .from('scans')
          .select('created_at')
          .eq('user_id', userId)
          .gte('created_at', oneWeekAgo)
          .order('created_at', { ascending: false })
          .limit(5);

        if (recentError) {
          throw new Error(`Failed to get recent scans: ${recentError.message}`);
        }

        return {
          totalScans,
          recentScans: recentScans?.length || 0,
          lastScanDate: recentScans && recentScans.length > 0 ? new Date(recentScans[0].created_at) : null
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
        const { data: allUsers, error: usersError } = await supabase
          .from('users')
          .select('id');
        
        if (usersError) {
          throw new Error(`Failed to get users count: ${usersError.message}`);
        }
        const totalUsers = allUsers?.length || 0;
        
        // Get total scans count
        const { data: allScans, error: scansError } = await supabase
          .from('scans')
          .select('id');
        
        if (scansError) {
          throw new Error(`Failed to get scans count: ${scansError.message}`);
        }
        const totalScans = allScans?.length || 0;
        
        // Get active users (logged in within last 7 days)
        const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
        const { data: activeUsers, error: activeError } = await supabase
          .from('users')
          .select('id')
          .gte('last_login', oneWeekAgo);
        
        if (activeError) {
          throw new Error(`Failed to get active users: ${activeError.message}`);
        }
        const activeUsersCount = activeUsers?.length || 0;
        
        // Get recent scans (last 24 hours)
        const oneDayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
        const { data: recentScans, error: recentScansError } = await supabase
          .from('scans')
          .select('id')
          .gte('created_at', oneDayAgo);
        
        if (recentScansError) {
          throw new Error(`Failed to get recent scans: ${recentScansError.message}`);
        }
        const recentScansCount = recentScans?.length || 0;

        // Calculate average query time and slow queries
        const averageQueryTime = this.queryTimes.length > 0 
          ? this.queryTimes.reduce((a, b) => a + b, 0) / this.queryTimes.length 
          : 0;

        const slowQueries = this.queryTimes.filter(time => time > this.slowQueryThreshold).length;

        return {
          totalUsers,
          totalScans,
          activeUsers: activeUsersCount,
          recentScans: recentScansCount,
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