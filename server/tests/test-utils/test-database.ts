import { migrate } from 'drizzle-orm/node-postgres/migrator';
import { Pool } from 'pg';
import { drizzle } from 'drizzle-orm/node-postgres';
import { join } from 'path';

export async function setupTestDatabase() {
  // Create a test database connection
  const testDbName = `test_${Date.now()}`;
  const pool = new Pool({
    connectionString: process.env.TEST_DATABASE_URL || 'postgres://postgres:postgres@localhost:5432/postgres',
  });

  try {
    // Create a new test database
    await pool.query(`CREATE DATABASE ${testDbName}`);
    
    // Update environment variable for the test database
    process.env.DATABASE_URL = `postgres://postgres:postgres@localhost:5432/${testDbName}`;
    
    // Run migrations on the test database
    const db = drizzle(pool);
    await migrate(db, { migrationsFolder: join(__dirname, '../../../migrations') });
    
    return {
      pool,
      db,
      dbName: testDbName,
    };
  } catch (error) {
    console.error('Error setting up test database:', error);
    await pool.end();
    throw error;
  }
}

export async function teardownTestDatabase(pool: Pool, dbName: string) {
  try {
    // Close all connections to the test database
    await pool.query(`
      SELECT pg_terminate_backend(pg_stat_activity.pid)
      FROM pg_stat_activity
      WHERE pg_stat_activity.datname = $1
      AND pid <> pg_backend_pid();
    `, [dbName]);
    
    // Drop the test database
    await pool.query(`DROP DATABASE IF EXISTS ${dbName}`);
  } catch (error) {
    console.error('Error tearing down test database:', error);
  } finally {
    await pool.end();
  }
}
