import { drizzle } from 'drizzle-orm/better-sqlite3';
import Database from 'better-sqlite3';
import * as schema from "@shared/schema";
import Redis from 'ioredis';

// Use SQLite for local development and WAL mode for better concurrency
const sqlite = new Database('./db.sqlite');
sqlite.pragma('journal_mode = WAL');
sqlite.pragma('synchronous = NORMAL');
sqlite.pragma('cache_size = -64000'); // 64MB cache

export const db = drizzle(sqlite, { schema });

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
      console.warn('Redis connection failed, continuing without Redis:', err.message);
      redis = null;
    });
  } catch (error) {
    console.warn('Redis initialization failed, continuing without Redis:', error);
    redis = null;
  }
}

export { redis };
