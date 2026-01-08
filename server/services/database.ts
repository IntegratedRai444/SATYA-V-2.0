import { Pool } from 'pg';
import { logger } from '../config/logger';

// Database connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

/**
 * Check database connection
 * @returns Promise<boolean> True if connection is successful
 */
export const checkDatabaseConnection = async (): Promise<boolean> => {
  const client = await pool.connect().catch((err) => {
    logger.error('Database connection error', { error: err.message });
    return null;
  });

  if (!client) {
    return false;
  }

  try {
    // Simple query to check connection
    await client.query('SELECT 1');
    return true;
  } catch (error) {
    logger.error('Database query error', { 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
    return false;
  } finally {
    client.release();
  }
};

// Handle connection errors
pool.on('error', (err) => {
  logger.error('Unexpected error on idle client', { error: err.message });
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received. Closing database connections...');
  await pool.end();
  logger.info('Database connections closed');
});

export default pool;
