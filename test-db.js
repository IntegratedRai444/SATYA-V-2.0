const { Pool } = require('pg');
require('dotenv').config({ path: './server/.env' });

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: { rejectUnauthorized: false } // Required for Supabase
});

async function testConnection() {
  try {
    const client = await pool.connect();
    console.log(' Successfully connected to the database');
    const result = await client.query('SELECT current_database(), current_user, version()');
    console.log('Database Info:', result.rows[0]);
    await client.release();
  } catch (err) {
    console.error(' Connection error:', err.message);
  } finally {
    await pool.end();
  }
}

testConnection();
