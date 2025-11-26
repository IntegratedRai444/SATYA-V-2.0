/**
 * Database Migration Runner
 * Applies all pending migrations to the database
 */

import { db, isDbConnected } from '../db';
import { sql } from 'drizzle-orm';
import { logger } from '../config';
import fs from 'fs/promises';
import path from 'path';

interface Migration {
  id: string;
  filename: string;
  sql: string;
  appliedAt?: Date;
}

async function getMigrationFiles(): Promise<Migration[]> {
  const migrationsDir = path.join(__dirname, 'migrations');
  
  try {
    const files = await fs.readdir(migrationsDir);
    const sqlFiles = files.filter(f => f.endsWith('.sql')).sort();
    
    const migrations: Migration[] = [];
    
    for (const filename of sqlFiles) {
      const filePath = path.join(migrationsDir, filename);
      const content = await fs.readFile(filePath, 'utf-8');
      
      migrations.push({
        id: filename.replace('.sql', ''),
        filename,
        sql: content
      });
    }
    
    return migrations;
  } catch (error) {
    logger.error('Error reading migration files', { error });
    return [];
  }
}

async function createMigrationsTable(): Promise<void> {
  try {
    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS migrations (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `);
    
    logger.info('Migrations table created/verified');
  } catch (error) {
    logger.error('Error creating migrations table', { error });
    throw error;
  }
}

async function getAppliedMigrations(): Promise<Set<string>> {
  try {
    const result = await db.execute(sql`SELECT id FROM migrations`);
    return new Set(result.rows.map((row: any) => row.id));
  } catch (error) {
    logger.error('Error getting applied migrations', { error });
    return new Set();
  }
}

async function applyMigration(migration: Migration): Promise<boolean> {
  try {
    logger.info(`Applying migration: ${migration.filename}`);
    
    // Split SQL into individual statements
    const statements = migration.sql
      .split(';')
      .map(s => s.trim())
      .filter(s => s.length > 0 && !s.startsWith('--'));
    
    // Execute each statement
    for (const statement of statements) {
      try {
        await db.execute(sql.raw(statement));
      } catch (error: any) {
        // Ignore "already exists" errors
        if (!error.message?.includes('already exists')) {
          throw error;
        }
      }
    }
    
    // Record migration as applied
    await db.execute(sql`
      INSERT INTO migrations (id, filename, applied_at)
      VALUES (${migration.id}, ${migration.filename}, CURRENT_TIMESTAMP)
      ON CONFLICT (id) DO NOTHING
    `);
    
    logger.info(`✓ Migration applied: ${migration.filename}`);
    return true;
  } catch (error) {
    logger.error(`✗ Migration failed: ${migration.filename}`, { error });
    return false;
  }
}

export async function runMigrations(): Promise<{
  success: boolean;
  applied: number;
  failed: number;
  skipped: number;
}> {
  const results = {
    success: true,
    applied: 0,
    failed: 0,
    skipped: 0
  };
  
  try {
    // Check database connection
    if (!isDbConnected()) {
      logger.warn('Database not connected, skipping migrations');
      return { ...results, success: false };
    }
    
    logger.info('Starting database migrations...');
    
    // Create migrations table
    await createMigrationsTable();
    
    // Get all migrations
    const migrations = await getMigrationFiles();
    const appliedMigrations = await getAppliedMigrations();
    
    logger.info(`Found ${migrations.length} migration files`);
    logger.info(`${appliedMigrations.size} migrations already applied`);
    
    // Apply pending migrations
    for (const migration of migrations) {
      if (appliedMigrations.has(migration.id)) {
        logger.debug(`Skipping already applied migration: ${migration.filename}`);
        results.skipped++;
        continue;
      }
      
      const success = await applyMigration(migration);
      
      if (success) {
        results.applied++;
      } else {
        results.failed++;
        results.success = false;
      }
    }
    
    logger.info('Migration summary:', results);
    
    return results;
  } catch (error) {
    logger.error('Migration process failed', { error });
    return { ...results, success: false };
  }
}

// CLI execution
if (require.main === module) {
  runMigrations()
    .then(results => {
      console.log('\n=== Migration Results ===');
      console.log(`Applied: ${results.applied}`);
      console.log(`Skipped: ${results.skipped}`);
      console.log(`Failed: ${results.failed}`);
      console.log(`Status: ${results.success ? '✓ SUCCESS' : '✗ FAILED'}`);
      
      process.exit(results.success ? 0 : 1);
    })
    .catch(error => {
      console.error('Migration error:', error);
      process.exit(1);
    });
}
