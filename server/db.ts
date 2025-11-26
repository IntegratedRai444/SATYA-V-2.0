import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from "@shared/schema";
import Redis from 'ioredis';
import { logger } from './config';
import { supabase } from './services/supabase-client';

// Validate required environment variables
if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is required');
}

// Re-export supabase client from single source
export { supabase };

// PostgreSQL connection string for Supabase
const connectionString = process.env.DATABASE_URL;

logger.info('🔗 Initializing database connection...');

// Create PostgreSQL connection with postgres.js with error handling
let client: ReturnType<typeof postgres> | null = null;
let dbInitialized = false;

try {
  client = postgres(connectionString, {
    max: 10,
    idle_timeout: 20,
    connect_timeout: 5,
    ssl: 'require',
    prepare: false,
    onnotice: () => {},
    connection: {
      application_name: 'satyaai'
    },
    // Add error handler to prevent crashes
    onclose: () => {
      logger.warn('⚠️  Database connection closed, using REST API fallback');
      dbInitialized = false;
    }
  });
  
  // Test connection with timeout
  Promise.race([
    client`SELECT 1`,
    new Promise((_, reject) => setTimeout(() => reject(new Error('Connection timeout')), 3000))
  ]).then(() => {
    logger.info('✅ PostgreSQL connection successful');
    dbInitialized = true;
  }).catch((err) => {
    logger.warn('⚠️  PostgreSQL connection failed, using REST API fallback', {
      error: err.message
    });
    dbInitialized = false;
  });
} catch (error) {
  logger.warn('⚠️  PostgreSQL initialization failed, using REST API fallback', {
    error: (error as Error).message
  });
  dbInitialized = false;
}

// Safe database export - only create drizzle instance if client exists
// Use a proxy to provide better error messages when db is not connected
export const db = client ? drizzle(client, { schema }) : new Proxy({} as any, {
  get: () => {
    throw new Error('Database not connected. Using Supabase REST API fallback.');
  }
});
export const isDbConnected = () => dbInitialized;

// Redis caching for production (disabled in development)
let redis: Redis | null = null;

// Only initialize Redis in production
if (process.env.NODE_ENV === 'production') {
  try {
    redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: Number(process.env.REDIS_PORT) || 6379,
      password: process.env.REDIS_PASSWORD || undefined,
      db: 0,
      maxRetriesPerRequest: 3,
      enableReadyCheck: false,
      lazyConnect: true
    });
    
    redis.connect().catch(err => {
      logger.warn('Redis connection failed, continuing without Redis', {
        error: err.message
      });
      redis = null;
    });
  } catch (error) {
    logger.warn('Redis initialization failed, continuing without Redis', {
      error: (error as Error).message
    });
    redis = null;
  }
}

export { redis };
