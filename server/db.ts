import { drizzle, type PostgresJsDatabase } from 'drizzle-orm/postgres-js';
import postgres, { type Sql } from 'postgres';
import * as schema from "../shared/schema";
import { logger } from './config/logger';
import { config } from './config/environment';
import { redisManager } from './config/redis';
import { supabase } from './services/supabase-client';

// Re-export supabase client from single source
export { supabase };

type DatabaseType = ReturnType<typeof drizzle<typeof schema>> & { $client: Sql };

class DatabaseManager {
  private static instance: DatabaseManager;
  private sql: Sql | null = null;
  private db: DatabaseType | null = null;
  private isInitialized = false;
  private isInitializing = false;
  private initPromise: Promise<void> | null = null;

  private constructor() {}

  public static getInstance(): DatabaseManager {
    if (!DatabaseManager.instance) {
      DatabaseManager.instance = new DatabaseManager();
    }
    return DatabaseManager.instance;
  }

  private validateConfig() {
    if (!config.DATABASE_URL) {
      throw new Error('DATABASE_URL is required in environment variables');
    }
  }

  private createConnection() {
    this.validateConfig();

    const dbConfig = {
      max: process.env.DB_POOL_SIZE ? parseInt(process.env.DB_POOL_SIZE, 10) : 10,
      idle_timeout: 20,
      connect_timeout: 5,
      ssl: process.env.NODE_ENV === 'production' ? 'require' : false,
      prepare: false,
      connection: {
        application_name: `satyaai-${process.env.NODE_ENV || 'development'}`,
        statement_timeout: 10000, // 10 seconds
      },
      onnotice: (notice: any) => {
        if (notice.severity === 'WARNING') {
          logger.warn('Database warning', { notice });
        }
      },
      onparameter: (key: string) => {
        if (key === 'application_name') return false;
        return true;
      },
      onclose: () => {
        logger.warn('Database connection closed');
      },
      transform: {
        column: (col: string) => col.toLowerCase(),
      },
    } as const;

    this.sql = postgres(config.DATABASE_URL!, dbConfig);
    const db = drizzle(this.sql, { schema });
    this.db = Object.assign(db, { $client: this.sql });
  }

  public async getDb(): Promise<DatabaseType> {
    if (this.db) return this.db;
    
    if (this.isInitializing && this.initPromise) {
      await this.initPromise;
      return this.db!;
    }

    this.isInitializing = true;
    this.initPromise = this.initialize();
    
    try {
      await this.initPromise;
      return this.db!;
    } catch (error) {
      this.isInitializing = false;
      this.initPromise = null;
      logger.error('Failed to initialize database connection', { 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      throw error;
    }
  }

  private async initialize(): Promise<void> {
    try {
      if (!this.sql || !this.db) {
        this.createConnection();
      }
      
      // Test the connection
      if (this.sql) {
        await this.sql`SELECT 1`;
        this.isInitialized = true;
        logger.info('Database connection established');
      }
    } catch (error) {
      this.isInitialized = false;
      logger.error('Database connection failed', { 
        error: error instanceof Error ? error.message : 'Unknown error' 
      });
      throw error;
    }
  }

  public isConnected(): boolean {
    return this.isInitialized;
  }

  public async close(): Promise<void> {
    if (this.sql) {
      await this.sql.end();
      this.sql = null;
      this.db = null;
      this.isInitialized = false;
      this.isInitializing = false;
      this.initPromise = null;
      logger.info('Database connection closed');
    }
  }
}

// Create database manager instance
const dbManager = DatabaseManager.getInstance();

// Export the database instance
export const db = new Proxy({} as DatabaseType, {
  get: (_, prop) => {
    if (!dbManager.isConnected()) {
      throw new Error('Database not connected. Please check your connection and try again.');
    }
    return (dbManager as any)[prop];
  },
  apply: (target, thisArg, args) => {
    if (!dbManager.isConnected()) {
      throw new Error('Database not connected. Please check your connection and try again.');
    }
    return (drizzle as any).apply(thisArg, args);
  }
});

export const isDbConnected = (): boolean => dbManager.isConnected();

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received. Closing database connections...');
  
  try {
    await dbManager.close();
    
    if (redisManager) {
      try {
        await redisManager.close();
        logger.info('Redis connection closed');
      } catch (error) {
        if (error && typeof error === 'object' && 'code' in error && error.code === 'ECONNREFUSED') {
          logger.error('Redis connection refused. Is the Redis server running?');
          process.exit(1);
        } else {
          logger.error('Error closing Redis connection', { 
            error: error instanceof Error ? error.message : 'Unknown error' 
          });
          process.exit(1);
        }
      }
    }
    
    process.exit(0);
  } catch (error) {
    logger.error('Error during shutdown', { 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
    process.exit(1);
  }
});
