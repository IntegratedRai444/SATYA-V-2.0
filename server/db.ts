import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from "@shared/schema";
import Redis from 'ioredis';
import { logger } from './config';
import { supabase } from './services/supabase-client';

// Validate required environment variables
const requiredEnvVars = [
  'DATABASE_URL',
  'JWT_SECRET',
  'SUPABASE_URL',
  'SUPABASE_ANON_KEY'
];

const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);
if (missingVars.length > 0) {
  const errorMsg = `Missing required environment variables: ${missingVars.join(', ')}`;
  logger.error(errorMsg);
  throw new Error(errorMsg);
}

// Re-export supabase client from single source
export { supabase };

// PostgreSQL connection string for Supabase
const connectionString = process.env.DATABASE_URL!;

// Connection configuration with enhanced settings
const dbConfig = {
  max: process.env.DB_POOL_SIZE ? parseInt(process.env.DB_POOL_SIZE, 10) : 10,
  idle_timeout: 20,
  connect_timeout: 5,
  ssl: process.env.NODE_ENV === 'production' ? 'require' : false,
  prepare: false,
  connection: {
    application_name: `satyaai-${process.env.NODE_ENV || 'development'}`,
    statement_timeout: 10000, // 10 seconds
    query_timeout: 30000, // 30 seconds
  },
  onnotice: (notice: any) => {
    if (notice.severity === 'WARNING') {
      logger.warn('Database warning', { notice });
    }
  },
  onparameter: (key: string) => {
    if (key === 'application_name') return false; // Don't log application name
    return true;
  },
  onclose: () => {
    const error = new Error('Database connection closed');
    logger.error('Database connection closed unexpectedly', { error: error.message });
    process.exit(1); // Exit process to allow process manager to restart
  },
  ontimeout: () => {
    const error = new Error('Database connection timeout');
    logger.error('Database connection timeout', { error: error.message });
    throw error;
  },
  transform: {
    column: (col: string) => col.toLowerCase(),
  },
} as const;

// Create PostgreSQL connection with enhanced error handling
let client: ReturnType<typeof postgres> | null = null;
let dbInitialized = false;

async function initializeDatabase() {
  try {
    if (client) {
      await client.end();
    }

    client = postgres(connectionString, dbConfig);

    // Test connection with timeout
    await Promise.race([
      client`SELECT 1`,
      new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Connection timeout')), 5000)
      )
    ]);
    
    logger.info('✅ PostgreSQL connection successful');
    dbInitialized = true;
    
    // Set up periodic connection check
    setInterval(async () => {
      try {
        await client`SELECT 1`;
      } catch (error) {
        logger.error('Database connection check failed', { error });
        dbInitialized = false;
      }
    }, 60000); // Check every minute
    
  } catch (error) {
    const errorMsg = `❌ Failed to connect to database: ${(error as Error).message}`;
    logger.error(errorMsg, { error });
    dbInitialized = false;
    
    // Attempt to reconnect after delay
    setTimeout(initializeDatabase, 10000);
  }
}

// Initialize database connection
initializeDatabase();

// Safe database export with error handling
export const db = new Proxy({} as ReturnType<typeof drizzle>, {
  get(_, prop) {
    if (!dbInitialized) {
      throw new Error('Database not connected. Please check your connection and try again.');
    }
    return (client as any)[prop];
  },
  apply(target, thisArg, args) {
    if (!dbInitialized) {
      throw new Error('Database not connected. Please check your connection and try again.');
    }
    return (drizzle as any).apply(thisArg, [client, { schema, ...args }]);
  }
});

export const isDbConnected = () => dbInitialized;

// Redis caching with enhanced configuration
let redis: Redis | null = null;
const redisConfig = {
  host: process.env.REDIS_HOST || 'localhost',
  port: Number(process.env.REDIS_PORT) || 6379,
  password: process.env.REDIS_PASSWORD,
  db: 0,
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  connectTimeout: 10000,
  retryStrategy: (times: number) => {
    if (times > 3) {
      logger.error('Max Redis retry attempts reached');
      return null; // Stop retrying after 3 attempts
    }
    return Math.min(times * 1000, 5000); // Backoff up to 5 seconds
  },
  reconnectOnError: (err: Error) => {
    const targetError = 'READONLY';
    if (err.message.includes(targetError)) {
      return true; // Only reconnect for READONLY errors
    }
    return false;
  }
};

// Initialize Redis if in production or explicitly enabled
if (process.env.NODE_ENV === 'production' || process.env.ENABLE_REDIS === 'true') {
  try {
    redis = new Redis(redisConfig);
    
    redis.on('connect', () => {
      logger.info('✅ Redis connected successfully');
    });
    
    redis.on('error', (error) => {
      logger.error('Redis error', { error: error.message });
    });
    
    redis.on('reconnecting', () => {
      logger.info('Attempting to reconnect to Redis...');
    });
    
  } catch (error) {
    logger.error('Redis initialization failed', { 
      error: (error as Error).message 
    });
    redis = null;
  }
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received. Closing database connections...');
  
  try {
    if (client) {
      await client.end();
      logger.info('Database connection closed');
    }
    
    if (redis) {
      await redis.quit();
      logger.info('Redis connection closed');
    }
    
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown', { error });
    process.exit(1);
  }
});

export { redis };
