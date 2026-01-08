import { db } from './db';
import { users, scans, userPreferences } from '../shared/schema';
import * as bcrypt from 'bcrypt';
import { sql, eq, and } from 'drizzle-orm';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import path from 'path';
import * as fs from 'fs';
import { logger } from './config/logger';

// Initialize database connection and run migrations
export async function initializeDatabase(): Promise<boolean> {
  try {
    logger.info('🗄️  Initializing SatyaAI database...');
    
    // Create migrations folder if it doesn't exist
    const migrationsFolder = path.join(process.cwd(), 'drizzle');
    
    // Run migrations

    // Set timezone to UTC
    await db.execute(sql`SET TIME ZONE 'UTC'`);
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
    const existingUsers = await db.select().from(users).limit(1);
    
    if (existingUsers.length === 0) {
      console.log('📊 Seeding initial data...');
      
      // Create demo user
      const hashedPassword = await bcrypt.hash('DemoPassword123!', 12);
      const [newUser] = await db.insert(users).values({
        username: 'demo',
        email: 'demo@satyaai.com',
        password: hashedPassword,
        fullName: 'Demo User',
        role: 'user'
      }).returning();
      
      console.log('✓ Demo user created (username: demo, password: DemoPassword123!)');
      
      // Create admin user
      const adminPassword = await bcrypt.hash('AdminPassword123!', 12);
      const [adminUser] = await db.insert(users).values({
        username: 'admin',
        email: 'admin@satyaai.com',
        password: adminPassword,
        fullName: 'System Administrator',
        role: 'admin'
      }).returning();
      
      console.log('✓ Admin user created (username: admin, password: AdminPassword123!)');
      
      // Add sample scans for demo user
      const sampleScans = [
        {
          userId: newUser.id,
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
          userId: newUser.id,
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
          userId: newUser.id,
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
          userId: newUser.id,
          theme: 'dark',
          language: 'english',
          confidenceThreshold: 80,
          enableNotifications: true,
          autoAnalyze: true,
          sensitivityLevel: 'medium'
        });
        
        // Admin user preferences
        await tx.insert(userPreferences).values({
          userId: adminUser.id,
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
    const result = await db.execute(sql`
      SELECT tablename 
      FROM pg_tables 
      WHERE schemaname = 'public'
    `);
    
    const tables = (result as any[]).map((r: any) => r.tablename);
    
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
    const tablesResult = await db.execute(sql`
      SELECT tablename as name 
      FROM pg_tables 
      WHERE schemaname = 'public'
    `);
    
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