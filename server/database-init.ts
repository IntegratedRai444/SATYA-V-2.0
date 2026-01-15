import { dbManager, supabase } from './db';
import { users, scans, userPreferences } from '../shared/schema';
import * as bcrypt from 'bcrypt';
import { sql, eq, and } from 'drizzle-orm';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import { drizzle, PostgresJsDatabase } from 'drizzle-orm/postgres-js';
import postgres, { type Sql } from 'postgres';
import path from 'path';
import * as fs from 'fs';
import { logger } from './config/logger';
import { config } from './config/environment';

// Initialize database connection
const connectionString = process.env.DATABASE_URL;
if (!connectionString) {
  throw new Error('DATABASE_URL environment variable is required');
}

const client: Sql = postgres(connectionString);
const db: PostgresJsDatabase = drizzle(client);

// Ensure database schema exists
async function ensureSchema() {
  try {
    logger.info('🔍 Verifying database schema...');
    
    // Drop users table if it exists to ensure clean state
    await client`
      DROP TABLE IF EXISTS users CASCADE;
    `;
    
    // Create users table with all required columns
    await client`
      CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(255) NOT NULL UNIQUE,
        password TEXT NOT NULL,
        email VARCHAR(255),
        full_name VARCHAR(255),
        api_key TEXT UNIQUE,
        role VARCHAR(50) NOT NULL DEFAULT 'user',
        failed_login_attempts INTEGER NOT NULL DEFAULT 0,
        last_failed_login TIMESTAMP,
        is_locked BOOLEAN NOT NULL DEFAULT false,
        lockout_until TIMESTAMP,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
    `;
    
    logger.info('✅ Users table created');
    
    // Drop scans table if it exists
    await client`
      DROP TABLE IF EXISTS scans CASCADE;
    `;
    
    // Create scans table
    await client`
      CREATE TABLE scans (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        filename VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL,
        result VARCHAR(50) NOT NULL,
        confidence_score NUMERIC(5,2) NOT NULL,
        detection_details JSONB,
        metadata JSONB,
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
    `;
    
    logger.info('✅ Scans table created');
    
    // Drop user_preferences table if it exists
    await client`
      DROP TABLE IF EXISTS user_preferences CASCADE;
    `;
    
    // Create user_preferences table
    await client`
      CREATE TABLE user_preferences (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        theme VARCHAR(50) DEFAULT 'dark',
        language VARCHAR(50) DEFAULT 'english',
        confidence_threshold INTEGER DEFAULT 80,
        enable_notifications BOOLEAN DEFAULT true,
        auto_analyze BOOLEAN DEFAULT true,
        sensitivity_level VARCHAR(50) DEFAULT 'medium',
        created_at TIMESTAMP NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
        UNIQUE(user_id)
      );
    `;
    
    logger.info('✅ User preferences table created');
    return true;
  } catch (error) {
    logger.error('❌ Failed to ensure database schema:', error);
    throw error;
  }
}

// Initialize database connection and run migrations
export async function initializeDatabase(): Promise<boolean> {
  try {
    logger.info('🗄️  Initializing SatyaAI database...');
    
    // Ensure schema exists first
    await ensureSchema();
    
    // Initialize the database connection
    await dbManager.initialize();
    
    // Set timezone to UTC using the direct SQL command
    await client`SET TIME ZONE 'UTC'`;
    logger.info('✓ Database timezone set to UTC');

    // Seed initial data if needed
    await seedInitialData();
    
    logger.info('✅ Database initialization completed successfully');
    return true;
    
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error during database initialization';
    logger.error('❌ Database initialization failed:', { error: errorMessage });
    
    // Check if the error is due to missing migration files
    if (error instanceof Error && error.message.includes('no such file or directory')) {
      logger.warn('⚠️  Migration files not found. Creating initial migration...');
      try {
        // Create initial migration if it doesn't exist
        const initialMigrationPath = path.join(__dirname, 'drizzle', '0000_initial_migration.sql');
        if (!fs.existsSync(initialMigrationPath)) {
          logger.info('Creating initial migration file...');
          fs.writeFileSync(initialMigrationPath, '-- Initial database schema');
          logger.info('✅ Created initial migration file');
          
          // Retry migration after creating the initial file
          logger.info('Retrying migration...');
          await migrate(db, { migrationsFolder: path.join(__dirname, 'drizzle') });
          logger.info('✅ Initial migration completed successfully');
          return true;
        }
      } catch (retryError) {
        const retryErrorMsg = retryError instanceof Error ? retryError.message : 'Unknown error';
        logger.error('❌ Failed to create initial migration:', { error: retryErrorMsg });
      }
    }
    
    throw error;
  }
}

// Seed initial data if the database is empty
async function seedInitialData(): Promise<void> {
  try {
    // Check if we already have users
    const { data: existingUsers, error } = await supabase
      .from('users')
      .select('*')
      .limit(1);
    
    if (error) throw error;
    
    if (!existingUsers || existingUsers.length === 0) {
      logger.info('📊 Seeding initial data...');
      
      // Create demo user using direct SQL to avoid any ORM issues
      const hashedPassword = await bcrypt.hash('DemoPassword123!', 12);
      const newUser = await client<Array<{ id: number }>>`
        INSERT INTO users (username, email, password, full_name, role, failed_login_attempts, is_locked, created_at, updated_at)
        VALUES ('demo', 'demo@satyaai.com', ${hashedPassword}, 'Demo User', 'user', 0, false, NOW(), NOW())
        RETURNING id;
      `;
      
      if (!newUser || newUser.length === 0) throw new Error('Failed to create demo user');
      const demoUserId = newUser[0].id;
      logger.info('✓ Demo user created (username: demo, password: DemoPassword123!)');
      
      // Create admin user using direct SQL
      const adminPassword = await bcrypt.hash('AdminPassword123!', 12);
      const adminUser = await client<Array<{ id: number }>>`
        INSERT INTO users (username, email, password, full_name, role, failed_login_attempts, is_locked, created_at, updated_at)
        VALUES ('admin', 'admin@satyaai.com', ${adminPassword}, 'System Administrator', 'admin', 0, false, NOW(), NOW())
        RETURNING id;
      `;
      
      if (!adminUser || adminUser.length === 0) throw new Error('Failed to create admin user');
      const adminUserId = adminUser[0].id;
      
      console.log('✓ Admin user created (username: admin, password: AdminPassword123!)');
      
      // Add sample scans for demo user
      const sampleScans = [
        {
          userId: demoUserId,
          filename: 'sample_authentic_image.jpg',
          type: 'image',
          result: 'authentic',
          confidenceScore: 98,
          detectionDetails: [
            {
              name: 'Facial Landmark Analysis',
              category: 'face',
              confidence: 98,
              description: 'No inconsistencies detected in facial features and landmarks.'
            },
            {
              name: 'Pixel-level Analysis',
              category: 'technical',
              confidence: 96,
              description: 'Natural compression artifacts consistent with camera capture.'
            }
          ],
          metadata: { 
            size: '1.2 MB',
            resolution: '1920x1080',
            format: 'JPEG',
            camera: 'iPhone 13 Pro'
          }
        },
        {
          userId: demoUserId,
          filename: 'suspicious_video_clip.mp4',
          type: 'video',
          result: 'deepfake',
          confidenceScore: 87,
          detectionDetails: [
            {
              name: 'Temporal Consistency Analysis',
              category: 'video',
              confidence: 89,
              description: 'Inconsistent temporal patterns detected in facial movements.'
            },
            {
              name: 'Lip-sync Analysis',
              category: 'audio-visual',
              confidence: 85,
              description: 'Slight misalignment between lip movements and audio detected.'
            }
          ],
          metadata: { 
            resolution: '720p', 
            duration: '2:34 min', 
            size: '24.7 MB',
            fps: 30,
            codec: 'H.264'
          }
        },
        {
          userId: demoUserId,
          filename: 'voice_sample.wav',
          type: 'audio',
          result: 'authentic',
          confidenceScore: 94,
          detectionDetails: [
            {
              name: 'Voice Pattern Analysis',
              category: 'audio',
              confidence: 94,
              description: 'Natural voice patterns and breathing detected.'
            },
            {
              name: 'Spectral Analysis',
              category: 'technical',
              confidence: 92,
              description: 'Frequency spectrum consistent with human vocal tract.'
            }
          ],
          metadata: { 
            duration: '45 seconds',
            sample_rate: '44.1 kHz',
            bit_depth: '16-bit',
            size: '3.8 MB'
          }
        }
      ];
      
      // Insert sample scans in a transaction
      await db.transaction(async (tx) => {
        for (const scan of sampleScans) {
          await tx.insert(scans).values({
            ...scan,
            detectionDetails: JSON.stringify(scan.detectionDetails),
            metadata: JSON.stringify(scan.metadata)
          });
        }
      });
      
      console.log('✓ Sample scans created');
      
      // Add user preferences in a transaction
      await db.transaction(async (tx) => {
        // Demo user preferences
        await tx.insert(userPreferences).values({
          userId: demoUserId,
          theme: 'dark',
          language: 'english',
          confidenceThreshold: 80,
          enableNotifications: true,
          autoAnalyze: true,
          sensitivityLevel: 'medium'
        });
        
        // Admin user preferences
        await tx.insert(userPreferences).values({
          userId: adminUserId,
          theme: 'dark',
          language: 'english',
          confidenceThreshold: 85,
          enableNotifications: true,
          autoAnalyze: false,
          sensitivityLevel: 'high'
        });
      });
      
      console.log('✓ User preferences created');
      console.log('📊 Initial data seeding completed');
    } else {
      console.log('✓ Database already contains data, skipping seeding');
    }
  } catch (error) {
    console.error('Error seeding initial data:', error);
    throw error;
  }
}

// Reset the database by dropping and recreating all tables
export async function resetDatabase(): Promise<boolean> {
  try {
    console.warn('⚠️  Resetting database! This will drop all data!');
    
    // Get all table names
    const result = await client`
      SELECT tablename 
      FROM pg_tables 
      WHERE schemaname = 'public'
    `;
    
    const tables = (result as unknown as Array<{ tablename: string }>).map(row => row.tablename);
    
    // Disable foreign key checks temporarily
    await db.execute(sql`SET session_replication_role = 'replica'`);
    
    // Drop all tables
    for (const table of tables) {
      await db.execute(sql.raw(`DROP TABLE IF EXISTS "${table}" CASCADE`));
      console.log(`Dropped table: ${table}`);
    }
    
    // Re-enable foreign key checks
    await db.execute(sql`SET session_replication_role = 'origin'`);
    
    console.log('✓ All tables dropped');
    
    // Re-run migrations to recreate schema
    await initializeDatabase();
    
    return true;
  } catch (error) {
    console.error('❌ Database reset failed:', error);
    return false;
  }
}

export async function checkDatabaseHealth(): Promise<{
  healthy: boolean;
  tables: string[];
  userCount: number;
  scanCount: number;
  error?: string;
}> {
  try {
    // Check if tables exist
    const tablesResult = await client`
      SELECT tablename as name 
      FROM pg_tables 
      WHERE schemaname = 'public'
    `;
    
    // Count users and scans
    const userCount = await db.select().from(users).then(rows => rows.length);
    const scanCount = await db.select().from(scans).then(rows => rows.length);
    
    return {
      healthy: true,
      tables: (tablesResult as any[]).map((t: any) => t.name),
      userCount,
      scanCount
    };
  } catch (error) {
    return {
      healthy: false,
      tables: [],
      userCount: 0,
      scanCount: 0,
      error: (error as Error).message
    };
  }
}