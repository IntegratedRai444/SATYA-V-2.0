import { createClient, RedisClientType } from 'redis';
import { logger } from '../config/logger';

let client: RedisClientType;

/**
 * Initialize Redis client
 */
export const initRedis = async (): Promise<void> => {
  if (client?.isOpen) {
    return;
  }

  client = createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379',
    socket: {
      reconnectStrategy: (retries) => {
        if (retries > 5) {
          logger.error('Max Redis reconnection attempts reached');
          return new Error('Max reconnection attempts reached');
        }
        // Exponential backoff
        return Math.min(retries * 100, 5000);
      },
    },
  });

  client.on('error', (err) => {
    logger.error('Redis error:', { error: err.message });
  });

  client.on('connect', () => {
    logger.info('Connected to Redis');
  });

  client.on('reconnecting', () => {
    logger.info('Reconnecting to Redis...');
  });

  try {
    await client.connect();
  } catch (error) {
    logger.error('Failed to connect to Redis:', { 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
    throw error;
  }
};

/**
 * Check Redis connection
 * @returns Promise<boolean> True if connection is successful
 */
export const checkRedisConnection = async (): Promise<boolean> => {
  if (!client) {
    try {
      await initRedis();
    } catch (error) {
      return false;
    }
  }

  try {
    await client.ping();
    return true;
  } catch (error) {
    logger.error('Redis ping failed:', { 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
    return false;
  }
};

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received. Closing Redis connection...');
  if (client?.isOpen) {
    await client.quit();
  }
  logger.info('Redis connection closed');
});

export const cache = {
  /**
   * Get value from cache
   */
  get: async <T>(key: string): Promise<T | null> => {
    try {
      if (!client?.isOpen) {
        await initRedis();
      }
      const value = await client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      logger.error('Cache get error:', { 
        key, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      return null;
    }
  },

  /**
   * Set value in cache with TTL (in seconds)
   */
  set: async <T>(
    key: string, 
    value: T, 
    ttlSeconds?: number
  ): Promise<boolean> => {
    try {
      if (!client?.isOpen) {
        await initRedis();
      }
      const stringValue = JSON.stringify(value);
      if (ttlSeconds) {
        await client.setEx(key, ttlSeconds, stringValue);
      } else {
        await client.set(key, stringValue);
      }
      return true;
    } catch (error) {
      logger.error('Cache set error:', { 
        key, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      return false;
    }
  },

  /**
   * Delete key from cache
   */
  del: async (key: string): Promise<boolean> => {
    try {
      if (!client?.isOpen) {
        await initRedis();
      }
      const result = await client.del(key);
      return result > 0;
    } catch (error) {
      logger.error('Cache delete error:', { 
        key, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      return false;
    }
  },

  /**
   * Clear all keys with a specific prefix
   */
  clearByPrefix: async (prefix: string): Promise<void> => {
    try {
      if (!client?.isOpen) {
        await initRedis();
      }
      const keys = await client.keys(`${prefix}:*`);
      if (keys.length > 0) {
        await client.del(keys);
      }
    } catch (error) {
      logger.error('Cache clear by prefix error:', { 
        prefix, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
    }
  },

  /**
   * Close the Redis connection
   */
  close: async (): Promise<void> => {
    if (client?.isOpen) {
      await client.quit();
    }
  },
};
