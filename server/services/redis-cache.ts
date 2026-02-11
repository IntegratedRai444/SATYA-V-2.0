/**
 * Redis Cache Service
 * Provides distributed caching using Redis
 */

import Redis from 'ioredis';
import { logger } from '../config/logger';

interface RedisCacheConfig {
  host: string;
  port: number;
  password?: string;
  db?: number;
  keyPrefix?: string;
  enabled: boolean;
}

class RedisCacheService {
  private client: any;
  private config: RedisCacheConfig;
  private connected = false;

  constructor() {
    this.config = {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD || undefined,
      db: parseInt(process.env.REDIS_DB || '0'),
      keyPrefix: process.env.REDIS_KEY_PREFIX || 'satyaai:',
      enabled: process.env.ENABLE_REDIS === 'true',
    };

    if (this.config.enabled) {
      this.connect();
    } else {
      logger.info('Redis cache is disabled');
    }
  }

  /**
   * Connect to Redis
   */
  private connect(): void {
    try {
      this.client = (Redis as any).constructor({
        host: this.config.host,
        port: this.config.port,
        password: this.config.password,
        db: this.config.db,
        keyPrefix: this.config.keyPrefix,
        retryStrategy: (times: number) => {
          const delay = Math.min(times * 50, 2000);
          return delay;
        },
        maxRetriesPerRequest: 3,
      });

      this.client.on('connect', () => {
        this.connected = true;
        logger.info('✓ Redis connected', {
          host: this.config.host,
          port: this.config.port,
        });
      });

      this.client.on('error', (error: any) => {
        logger.error('Redis error', error);
        this.connected = false;
      });

      this.client.on('close', () => {
        this.connected = false;
        logger.warn('Redis connection closed');
      });
    } catch (error) {
      logger.error('Failed to initialize Redis', error as Error);
    }
  }

  /**
   * Set a value in cache
   */
  async set(key: string, value: any, ttl?: number): Promise<boolean> {
    if (!this.isAvailable()) return false;

    try {
      const serialized = JSON.stringify(value);
      if (ttl) {
        await this.client!.setex(key, ttl, serialized);
      } else {
        await this.client!.set(key, serialized);
      }
      return true;
    } catch (error) {
      logger.error('Redis set error', error as Error, { key });
      return false;
    }
  }

  /**
   * Get a value from cache
   */
  async get<T = any>(key: string): Promise<T | null> {
    if (!this.isAvailable()) return null;

    try {
      const value = await this.client!.get(key);
      if (!value) return null;
      return JSON.parse(value) as T;
    } catch (error) {
      logger.error('Redis get error', error as Error, { key });
      return null;
    }
  }

  /**
   * Delete a key from cache
   */
  async del(key: string): Promise<boolean> {
    if (!this.isAvailable()) return false;

    try {
      await this.client!.del(key);
      return true;
    } catch (error) {
      logger.error('Redis del error', error as Error, { key });
      return false;
    }
  }

  /**
   * Check if key exists
   */
  async exists(key: string): Promise<boolean> {
    if (!this.isAvailable()) return false;

    try {
      const result = await this.client!.exists(key);
      return result === 1;
    } catch (error) {
      logger.error('Redis exists error', error as Error, { key });
      return false;
    }
  }

  /**
   * Set expiration on a key
   */
  async expire(key: string, seconds: number): Promise<boolean> {
    if (!this.isAvailable()) return false;

    try {
      await this.client!.expire(key, seconds);
      return true;
    } catch (error) {
      logger.error('Redis expire error', error as Error, { key });
      return false;
    }
  }

  /**
   * Increment a value
   */
  async incr(key: string): Promise<number | null> {
    if (!this.isAvailable()) return null;

    try {
      return await this.client!.incr(key);
    } catch (error) {
      logger.error('Redis incr error', error as Error, { key });
      return null;
    }
  }

  /**
   * Get multiple keys
   */
  async mget<T = any>(keys: string[]): Promise<(T | null)[]> {
    if (!this.isAvailable()) return keys.map(() => null);

    try {
      const values = await this.client!.mget(keys);
      return values.map((v: string | null) => v ? JSON.parse(v) as T : null);
    } catch (error) {
      logger.error('Redis mget error', error as Error, { keys });
      return keys.map(() => null);
    }
  }

  /**
   * Set multiple keys
   */
  async mset(entries: Record<string, any>): Promise<boolean> {
    if (!this.isAvailable()) return false;

    try {
      const pipeline = this.client!.pipeline();
      for (const [key, value] of Object.entries(entries)) {
        pipeline.set(key, JSON.stringify(value));
      }
      await pipeline.exec();
      return true;
    } catch (error) {
      logger.error('Redis mset error', error as Error);
      return false;
    }
  }

  /**
   * Get all keys matching pattern
   */
  async keys(pattern: string): Promise<string[]> {
    if (!this.isAvailable()) return [];

    try {
      return await this.client!.keys(pattern);
    } catch (error) {
      logger.error('Redis keys error', error as Error, { pattern });
      return [];
    }
  }

  /**
   * Clear all keys matching pattern
   */
  async clear(pattern: string = '*'): Promise<number> {
    if (!this.isAvailable()) return 0;

    try {
      const keys = await this.keys(pattern);
      if (keys.length === 0) return 0;

      const pipeline = this.client!.pipeline();
      keys.forEach(key => pipeline.del(key));
      await pipeline.exec();
      return keys.length;
    } catch (error) {
      logger.error('Redis clear error', error as Error, { pattern });
      return 0;
    }
  }

  /**
   * Get cache statistics
   */
  async getStats(): Promise<Record<string, any>> {
    if (!this.isAvailable()) {
      return { connected: false };
    }

    try {
      const info = await this.client!.info('stats');
      const dbSize = await this.client!.dbsize();

      return {
        connected: this.connected,
        dbSize,
        info: this.parseRedisInfo(info),
      };
    } catch (error) {
      logger.error('Redis stats error', error as Error);
      return { connected: false, error: (error as Error).message };
    }
  }

  /**
   * Parse Redis INFO command output
   */
  private parseRedisInfo(info: string): Record<string, string> {
    const result: Record<string, string> = {};
    info.split('\r\n').forEach(line => {
      if (line && !line.startsWith('#')) {
        const [key, value] = line.split(':');
        if (key && value) {
          result[key] = value;
        }
      }
    });
    return result;
  }

  /**
   * Check if Redis is available
   */
  isAvailable(): boolean {
    return this.connected && this.client !== null;
  }

  /**
   * Close Redis connection
   */
  async close(): Promise<void> {
    if (this.client) {
      await this.client.quit();
      this.connected = false;
      logger.info('Redis connection closed');
    }
  }
}

// Export singleton instance
export const redisCache = new RedisCacheService();
export default redisCache;
