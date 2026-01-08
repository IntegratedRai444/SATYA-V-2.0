import { createClient, RedisClientType } from 'redis';
import { logger } from './logger';

type RedisClient = ReturnType<typeof createClient>;

class RedisManager {
  private static instance: RedisManager;
  private client: RedisClient | null = null;
  private isConnecting = false;
  private connectionPromise: Promise<void> | null = null;
  private isEnabled: boolean;

  private constructor() {
    const redisEnabled = process.env.REDIS_ENABLED;
    this.isEnabled = redisEnabled !== 'false' && !!process.env.REDIS_URL?.trim();
    
    if (!this.isEnabled) {
      logger.info('Redis is disabled. Set REDIS_ENABLED=true and provide REDIS_URL to enable.');
    } else {
      logger.info('Redis is enabled. Initializing connection...');
    }
  }

  public static getInstance(): RedisManager {
    if (!RedisManager.instance) {
      RedisManager.instance = new RedisManager();
    }
    return RedisManager.instance;
  }

  private async createClient(): Promise<RedisClient | null> {
    if (!this.isEnabled) {
      logger.info('Redis is disabled. No client created.');
      return null;
    }

    try {
      const client = createClient({
        url: process.env.REDIS_URL || 'redis://localhost:6379',
        socket: {
          reconnectStrategy: this.reconnectStrategy.bind(this),
        },
      });

      await client.connect();
      return client as unknown as RedisClient;
    } catch (error) {
      logger.error('Failed to create Redis client:', error);
      return null;
    }
  }

  private reconnectStrategy(retries: number): number | Error {
    if (retries > 5) {
      return new Error('Max Redis reconnection attempts reached');
    }
    return Math.min(retries * 100, 5000);
  }

  public async getClient(): Promise<RedisClient | null> {
    if (!this.isEnabled) {
      return null;
    }

    if (this.client) {
      return this.client;
    }

    if (this.isConnecting && this.connectionPromise) {
      await this.connectionPromise;
      return this.client;
    }

    this.isConnecting = true;
    this.connectionPromise = this.initializeConnection();
    
    try {
      await this.connectionPromise;
      return this.client;
    } catch (error) {
      this.isConnecting = false;
      this.connectionPromise = null;
      logger.warn('Redis connection failed, running in degraded mode', { 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      return null;
    }
  }

  private async initializeConnection(): Promise<void> {
    if (!this.isEnabled) {
      logger.info('Redis is disabled. Skipping connection.');
      return;
    }

    try {
      const client = await this.createClient();
      if (!client) {
        throw new Error('Failed to create Redis client');
      }
      
      this.client = client;
      
      client.on('error', (err) => {
        logger.warn('Redis client error:', err);
      });

      client.on('connect', () => {
        logger.info('Connected to Redis');
      });

      // Set a connection timeout
      const connectionTimeout = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Redis connection timeout')), 5000)
      );

      await Promise.race([
        client.connect(),
        connectionTimeout
      ]);
      
      logger.info('Redis client connected successfully');
    } catch (error) {
      logger.error('Failed to connect to Redis:', error);
      this.client = null;
      throw error; // Re-throw to be caught by getClient()
    } finally {
      this.isConnecting = false;
      this.connectionPromise = null;
    }
  }

  public async close(): Promise<void> {
    try {
      if (this.client && this.client.isOpen) {
        await this.client.quit();
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.warn('Error while closing Redis connection:', { error: errorMessage });
    } finally {
      this.client = null;
    }
  }

  public isConnected(): boolean {
    return this.isEnabled && (this.client?.isOpen || false);
  }

  public isAvailable(): boolean {
    return this.isEnabled && this.isConnected();
  }

  public isRedisEnabled(): boolean {
    return this.isEnabled && this.isConnected();
  }

  /**
   * Set a key with expiration
   */
  public async setEx(key: string, seconds: number, value: string): Promise<void> {
    if (!this.isEnabled) {
      logger.debug('Redis is disabled - setEx() operation skipped');
      return;
    }

    try {
      const client = await this.getClient();
      if (client) {
        await client.setEx(key, seconds, value);
      } else {
        logger.warn('Redis client not available - setEx() operation skipped');
      }
    } catch (error) {
      logger.error('Redis setEx() failed:', error instanceof Error ? error.message : 'Unknown error');
      // Don't throw - allow the application to continue without Redis
    }
  }

  /**
   * Get a value by key
   */
  public async get(key: string): Promise<string | null> {
    if (!this.isEnabled) {
      logger.debug('Redis is disabled - get() returning null');
      return null;
    }

    try {
      const client = await this.getClient();
      if (!client) {
        logger.debug('Redis client not available - get() returning null');
        return null;
      }

      return await client.get(key);
    } catch (error) {
      logger.error('Redis get() failed:', error instanceof Error ? error.message : 'Unknown error');
      return null; // Return null on error to allow graceful degradation
    }
  }
}

// Export a singleton instance
export const redisManager = RedisManager.getInstance();

// For backward compatibility
async function getRedisClient() {
  if (process.env.REDIS_ENABLED === 'false' || !process.env.REDIS_URL) {
    logger.warn('getRedisClient called but Redis is disabled. Check your configuration.');
    return null;
  }
  return redisManager.getClient();
}

export default redisManager;
