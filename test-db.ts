import { Pool } from 'pg';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import path from 'path';

// Resolve __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config({ path: path.join(__dirname, 'server', '.env') });

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false } // Required for Supabase
});

async function testConnection() {
  const client = await pool.connect();
  try {
    console.log(' Successfully connected to the database');
    const result = await client.query('SELECT current_database(), current_user, version()');
    console.log('Database Info:', result.rows[0]);
  } catch (err) {
    console.error(' Connection error:', (err as Error).message);
  } finally {
    client.release();
    await pool.end();
  }
}

testConnection();
