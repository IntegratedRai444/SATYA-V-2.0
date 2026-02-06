import { EventEmitter } from 'events';
import { logger } from '../config/logger';

/**
 * Cache Entry Interface
 */
interface CacheEntry<T = any> {
  key: string;
  value: T;
  ttl: number;
  createdAt: Date;
  expiresAt: Date;
  hitCount: number;
  lastAccessed: Date;
}

/**
 * Cache Statistics Interface
 */
interface CacheStats {
  totalKeys: number;
  hits: number;
  misses: number;
  hitRate: number;
  memoryUsage: number;
  oldestEntry: Date | null;
  newestEntry: Date | null;
}

/**
 * Cache Manager Service
 * Provides in-memory caching with TTL support, LRU eviction, and statistics
 */
export class CacheManager extends EventEmitter {
  private cache: Map<string, CacheEntry>;
  private hits: number = 0;
  private misses: number = 0;
  private maxSize: number;
  private defaultTTL: number;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(maxSize: number = 1000, defaultTTL: number = 300000) { // 5 minutes default
    super();
    this.cache = new Map();
    this.maxSize = maxSize;
    this.defaultTTL = defaultTTL;
    this.startCleanup();
    logger.info('CacheManager initialized', { maxSize, defaultTTL });
  }

  /**
   * Get value from cache
   */
  get<T = any>(key: string): T | null {
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      this.emit('miss', key);
      return null;
    }

    // Check if expired
    if (new Date() > entry.expiresAt) {
      this.cache.delete(key);
      this.misses++;
      this.emit('expired', key);
      return null;
    }

    // Update access stats
    entry.hitCount++;
    entry.lastAccessed = new Date();
    this.hits++;
    this.emit('hit', key);

    return entry.value as T;
  }

  /**
   * Set value in cache
   */
  set<T = any>(key: string, value: T, ttl?: number): void {
    // Check size limit and evict if necessary
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      this.evictLRU();
    }

    const actualTTL = ttl || this.defaultTTL;
    const now = new Date();
    const expiresAt = new Date(now.getTime() + actualTTL);

    const entry: CacheEntry<T> = {
      key,
      value,
      ttl: actualTTL,
      createdAt: now,
      expiresAt,
      hitCount: 0,
      lastAccessed: now
    };

    this.cache.set(key, entry);
    this.emit('set', key, value);

    if (process.env.NODE_ENV === 'development') {
      logger.debug('Cache set', { key, ttl: actualTTL });
    }
  }

  /**
   * Delete value from cache
   */
  delete(key: string): boolean {
    const deleted = this.cache.delete(key);
    if (deleted) {
      this.emit('delete', key);
      if (process.env.NODE_ENV === 'development') {
      logger.debug('Cache delete', { key });
    }
    }
    return deleted;
  }

  /**
   * Check if key exists in cache
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    // Check if expired
    if (new Date() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    const size = this.cache.size;
    this.cache.clear();
    this.emit('clear');
    logger.info('Cache cleared', { entriesRemoved: size });
  }

  /**
   * Get or set pattern - fetch from cache or compute and cache
   */
  async getOrSet<T = any>(
    key: string,
    factory: () => Promise<T> | T,
    ttl?: number
  ): Promise<T> {
    // Try to get from cache
    const cached = this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Compute value
    const value = await factory();

    // Store in cache
    this.set(key, value, ttl);

    return value;
  }

  /**
   * Invalidate cache entries by pattern
   */
  invalidatePattern(pattern: string | RegExp): number {
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    let count = 0;

    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
        count++;
      }
    }

    if (count > 0) {
      this.emit('invalidate', pattern, count);
      logger.info('Cache invalidated by pattern', { pattern: pattern.toString(), count });
    }

    return count;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const entries = Array.from(this.cache.values());
    const totalRequests = this.hits + this.misses;
    const hitRate = totalRequests > 0 ? (this.hits / totalRequests) * 100 : 0;

    // Calculate memory usage (rough estimate)
    const memoryUsage = this.estimateMemoryUsage();

    return {
      totalKeys: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: Math.round(hitRate * 100) / 100,
      memoryUsage,
      oldestEntry: entries.length > 0 
        ? entries.reduce((oldest, entry) => 
            entry.createdAt < oldest ? entry.createdAt : oldest, 
            entries[0].createdAt
          )
        : null,
      newestEntry: entries.length > 0
        ? entries.reduce((newest, entry) => 
            entry.createdAt > newest ? entry.createdAt : newest,
            entries[0].createdAt
          )
        : null
    };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.hits = 0;
    this.misses = 0;
    logger.info('Cache statistics reset');
  }

  /**
   * Evict least recently used entry
   */
  private evictLRU(): void {
    let lruKey: string | null = null;
    let lruTime: Date | null = null;

    for (const [key, entry] of this.cache.entries()) {
      if (!lruTime || entry.lastAccessed < lruTime) {
        lruTime = entry.lastAccessed;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
      this.emit('evict', lruKey);
      if (process.env.NODE_ENV === 'development') {
      logger.debug('Cache LRU eviction', { key: lruKey });
    }
    }
  }

  /**
   * Start automatic cleanup of expired entries
   */
  private startCleanup(): void {
    // Run cleanup every minute
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpired();
    }, 60000);
  }

  /**
   * Clean up expired entries
   */
  private cleanupExpired(): void {
    const now = new Date();
    let count = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
        count++;
      }
    }

    if (count > 0) {
      this.emit('cleanup', count);
      if (process.env.NODE_ENV === 'development') {
      logger.debug('Cache cleanup completed', { entriesRemoved: count });
    }
    }
  }

  /**
   * Estimate memory usage (rough calculation)
   */
  private estimateMemoryUsage(): number {
    let size = 0;

    for (const entry of this.cache.values()) {
      // Rough estimate: JSON stringify size
      try {
        size += JSON.stringify(entry.value).length;
      } catch {
        size += 1000; // Default estimate for non-serializable objects
      }
    }

    return size;
  }

  /**
   * Stop cleanup interval
   */
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }
    this.clear();
    logger.info('CacheManager destroyed');
  }

  /**
   * Get all keys in cache
   */
  keys(): string[] {
    return Array.from(this.cache.keys());
  }

  /**
   * Get cache size
   */
  size(): number {
    return this.cache.size;
  }

  /**
   * Get entry details (for monitoring)
   */
  getEntry(key: string): CacheEntry | null {
    return this.cache.get(key) || null;
  }
}

// Export singleton instance
export const cacheManager = new CacheManager(
  parseInt(process.env.CACHE_MAX_SIZE || '1000'),
  parseInt(process.env.CACHE_DEFAULT_TTL || '300000')
);

// Export class for testing
export default CacheManager;
