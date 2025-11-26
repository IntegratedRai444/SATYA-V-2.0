#!/usr/bin/env tsx

/**
 * Database Migration Runner
 * Runs all pending database migrations
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import postgres from 'postgres';
import { config } from 'dotenv';
import { resolve } from 'path';

// Load environment variables
config({ path: resolve(process.cwd(), '.env') });

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  console.error('❌ DATABASE_URL is not defined in environment variables');
  process.exit(1);
}

async function runMigrations() {
  console.log('🔄 Starting database migrations...\n');

  const migrationClient = postgres(DATABASE_URL!, { max: 1 });
  const db = drizzle(migrationClient);

  try {
    console.log('📂 Looking for migrations in: ./server/db/migrations');
    
    await migrate(db, {
      migrationsFolder: './server/db/migrations',
    });

    console.log('\n✅ All migrations completed successfully!');
    
  } catch (error) {
    console.error('\n❌ Migration failed:', error);
    process.exit(1);
  } finally {
    await migrationClient.end();
  }
}

// Run migrations
runMigrations()
  .then(() => {
    console.log('\n✨ Database is up to date');
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n💥 Fatal error:', error);
    process.exit(1);
  });
