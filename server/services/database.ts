import { supabase } from '../config/supabase';
import { logger } from '../config/logger';

// Use Supabase client instead of PostgreSQL pool
// const pool = new Pool({
//   connectionString: process.env.DATABASE_URL,
//   ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
//   max: 20,
//   idleTimeoutMillis: 30000,
//   connectionTimeoutMillis: 2000,
// });

/**
 * Check database connection
 * @returns Promise<boolean> True if connection is successful
 */
export const checkDatabaseConnection = async (): Promise<boolean> => {
  try {
    // Test Supabase connection with a simple health check
    const { data, error } = await supabase
      .from('tasks')
      .select('id')
      .limit(1);
    
    if (error) {
      logger.error('Supabase connection error', { error: error.message });
      return false;
    }
    
    logger.info('Database connection successful');
    return true;
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    logger.error('Database check failed', { error: errorMessage });
    return false;
  }
};

// Handle connection errors
// pool.on('error', (err) => {
//   logger.error('Unexpected error on idle client', { error: err.message });
// });

// Graceful shutdown (Supabase handles connection pooling)
process.on('SIGTERM', async () => {
  logger.info('Database connection shutting down');
});

process.on('SIGINT', async () => {
  logger.info('Database connection shutting down');
});

// export default pool;
