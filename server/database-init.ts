import { db } from './db';
import { users, scans, userPreferences } from '@shared/schema';
import * as bcrypt from 'bcrypt';
import { sql } from 'drizzle-orm';

export async function initializeDatabase(): Promise<boolean> {
  try {
    console.log('🗄️  Initializing SatyaAI database...');

    // Create tables if they don't exist (SQLite with WAL mode)
    await db.run(sql`PRAGMA journal_mode = WAL`);
    await db.run(sql`PRAGMA synchronous = NORMAL`);
    await db.run(sql`PRAGMA cache_size = -64000`); // 64MB cache
    await db.run(sql`PRAGMA foreign_keys = ON`);

    console.log('✓ Database configuration applied');

    // Check if tables exist and create them if needed
    await createTablesIfNotExist();

    // Seed initial data if database is empty
    await seedInitialData();

    console.log('✅ Database initialization completed successfully');
    return true;
  } catch (error) {
    console.error('❌ Database initialization failed:', error);
    return false;
  }
}

async function createTablesIfNotExist(): Promise<void> {
  try {
    // Create users table
    await db.run(sql`
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL,
        email TEXT,
        full_name TEXT,
        role TEXT NOT NULL DEFAULT 'user',
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )
    `);

    // Create scans table
    await db.run(sql`
      CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT NOT NULL,
        type TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence_score INTEGER NOT NULL,
        detection_details TEXT,
        metadata TEXT,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    `);

    // Create user_preferences table
    await db.run(sql`
      CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        theme TEXT DEFAULT 'dark',
        language TEXT DEFAULT 'english',
        confidence_threshold INTEGER DEFAULT 75,
        enable_notifications INTEGER DEFAULT 1,
        auto_analyze INTEGER DEFAULT 1,
        sensitivity_level TEXT DEFAULT 'medium',
        FOREIGN KEY (user_id) REFERENCES users(id)
      )
    `);

    // Create indexes for better performance
    await db.run(sql`CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)`);
    await db.run(sql`CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)`);
    await db.run(sql`CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans(user_id)`);
    await db.run(sql`CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at)`);
    await db.run(sql`CREATE INDEX IF NOT EXISTS idx_scans_type ON scans(type)`);
    await db.run(sql`CREATE INDEX IF NOT EXISTS idx_scans_result ON scans(result)`);

    console.log('✓ Database tables and indexes created/verified');
  } catch (error) {
    console.error('Error creating tables:', error);
    throw error;
  }
}

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
          detectionDetails: JSON.stringify([
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
          ]),
          metadata: JSON.stringify({ 
            size: '1.2 MB',
            resolution: '1920x1080',
            format: 'JPEG',
            camera: 'iPhone 13 Pro'
          })
        },
        {
          userId: newUser.id,
          filename: 'suspicious_video_clip.mp4',
          type: 'video',
          result: 'deepfake',
          confidenceScore: 87,
          detectionDetails: JSON.stringify([
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
          ]),
          metadata: JSON.stringify({ 
            resolution: '720p', 
            duration: '2:34 min', 
            size: '24.7 MB',
            fps: 30,
            codec: 'H.264'
          })
        },
        {
          userId: newUser.id,
          filename: 'voice_sample.wav',
          type: 'audio',
          result: 'authentic',
          confidenceScore: 94,
          detectionDetails: JSON.stringify([
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
          ]),
          metadata: JSON.stringify({ 
            duration: '45 seconds',
            sample_rate: '44.1 kHz',
            bit_depth: '16-bit',
            size: '3.8 MB'
          })
        }
      ];
      
      for (const scan of sampleScans) {
        await db.insert(scans).values(scan);
      }
      
      console.log('✓ Sample scans created');
      
      // Add user preferences for demo user
      await db.insert(userPreferences).values({
        userId: newUser.id,
        theme: 'dark',
        language: 'english',
        confidenceThreshold: 80,
        enableNotifications: true,
        autoAnalyze: true,
        sensitivityLevel: 'medium'
      });
      
      // Add user preferences for admin user
      await db.insert(userPreferences).values({
        userId: adminUser.id,
        theme: 'dark',
        language: 'english',
        confidenceThreshold: 85,
        enableNotifications: true,
        autoAnalyze: false,
        sensitivityLevel: 'high'
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

export async function resetDatabase(): Promise<boolean> {
  try {
    console.log('🗑️  Resetting database...');
    
    // Drop all tables
    await db.run(sql`DROP TABLE IF EXISTS user_preferences`);
    await db.run(sql`DROP TABLE IF EXISTS scans`);
    await db.run(sql`DROP TABLE IF EXISTS users`);
    
    console.log('✓ Tables dropped');
    
    // Recreate database
    const success = await initializeDatabase();
    
    if (success) {
      console.log('✅ Database reset completed successfully');
    }
    
    return success;
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
    const tables = await db.run(sql`
      SELECT name FROM sqlite_master 
      WHERE type='table' AND name NOT LIKE 'sqlite_%'
    `);
    
    // Count users and scans
    const userCount = await db.select().from(users).then(rows => rows.length);
    const scanCount = await db.select().from(scans).then(rows => rows.length);
    
    return {
      healthy: true,
      tables: tables.map((t: any) => t.name),
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