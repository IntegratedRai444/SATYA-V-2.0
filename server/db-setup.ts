import { db } from "./db";
import { users, scans, userPreferences } from "@shared/schema";
import { eq } from "drizzle-orm";
import * as bcrypt from "bcrypt";

// Function to seed initial data
async function seedInitialData() {
  console.log("Seeding initial data...");
  
  // Create default user
  const demoUser = {
    username: 'demo',
    password: await bcrypt.hash('password', 10),
    email: 'demo@satyaai.com',
    fullName: 'Demo User'
  };
  
  // Check if user already exists
  const existingUser = await db.select().from(users).where(eq(users.username, demoUser.username));
  
  let userId = 1;
  if (existingUser.length === 0) {
    const [createdUser] = await db.insert(users).values(demoUser).returning({ id: users.id });
    userId = createdUser.id;
    console.log(`Created default user with ID: ${userId}`);
  } else {
    userId = existingUser[0].id;
    console.log(`Default user already exists with ID: ${userId}`);
  }
  
  // Sample scan data
  const sampleScans = [
    {
      userId,
      filename: 'Profile_Image.jpg',
      type: 'image',
      result: 'authentic',
      confidenceScore: 98,
      detectionDetails: [
        {
          name: 'Facial Landmark Analysis',
          category: 'face',
          confidence: 98,
          description: 'No inconsistencies detected in facial features.'
        },
        {
          name: 'Image Metadata Integrity',
          category: 'general',
          confidence: 97,
          description: 'Metadata is consistent with original camera source.'
        }
      ],
      metadata: {
        size: '1.2 MB'
      }
    },
    {
      userId,
      filename: 'Interview_Clip.mp4',
      type: 'video',
      result: 'deepfake',
      confidenceScore: 95,
      detectionDetails: [
        {
          name: 'Facial Landmark Analysis',
          category: 'face',
          confidence: 97,
          description: 'Inconsistent eye blinking pattern and unnatural lip movements detected.'
        },
        {
          name: 'Audio-Visual Sync',
          category: 'audio',
          confidence: 96,
          description: 'Mouth movements do not consistently match audio phonemes.'
        },
        {
          name: 'Temporal Coherence',
          category: 'frame',
          confidence: 92,
          description: 'Unnatural transitions between frames at timestamps 0:42, 1:15, and 1:47.'
        }
      ],
      metadata: {
        resolution: '720p',
        duration: '2:34 min',
        size: '24.7 MB'
      }
    },
    {
      userId,
      filename: 'Voice_Message.mp3',
      type: 'audio',
      result: 'authentic',
      confidenceScore: 89,
      detectionDetails: [
        {
          name: 'Voice Pattern Analysis',
          category: 'audio',
          confidence: 89,
          description: 'Voice patterns are consistent throughout the recording.'
        },
        {
          name: 'Audio Spectrum Analysis',
          category: 'audio',
          confidence: 92,
          description: 'Spectral features match natural human voice characteristics.'
        }
      ],
      metadata: {
        duration: '0:47 min',
        size: '1.8 MB'
      }
    }
  ];
  
  // Check if we have scans already
  const existingScans = await db.select().from(scans);
  
  if (existingScans.length === 0) {
    for (const scan of sampleScans) {
      await db.insert(scans).values(scan);
    }
    console.log(`Added ${sampleScans.length} sample scans`);
  } else {
    console.log(`Sample scans already exist, skipping seed data`);
  }
  
  // Create default user preferences
  const userPref = {
    userId,
    theme: 'dark',
    language: 'english',
    confidenceThreshold: 75,
    enableNotifications: true,
    autoAnalyze: true,
    sensitivityLevel: 'medium'
  };
  
  const existingPrefs = await db.select().from(userPreferences).where(eq(userPreferences.userId, userId));
  
  if (existingPrefs.length === 0) {
    await db.insert(userPreferences).values(userPref);
    console.log(`Created default user preferences for user ID: ${userId}`);
  } else {
    console.log(`User preferences already exist for user ID: ${userId}`);
  }
  
  console.log("Seeding complete!");
}

// Main migration and setup function
async function setupDatabase() {
  try {
    console.log("Setting up database...");
    
    // Push schema to database
    // In a production application, you would use proper migrations
    console.log("Pushing schema to database...");
    
    // Seed initial data
    await seedInitialData();
    
    console.log("Database setup complete!");
  } catch (error) {
    console.error("Error setting up database:", error);
  }
}

// Only run setup when this file is executed directly (not imported)
if (require.main === module) {
  setupDatabase().then(() => {
    process.exit(0);
  }).catch(err => {
    console.error("Database setup failed:", err);
    process.exit(1);
  });
}

export default setupDatabase;