import { db } from "./db";
import { storage } from "./storage";
import { users, scans, userPreferences } from "@shared/schema";
import { eq } from "drizzle-orm";
import * as bcrypt from "bcrypt";

async function seedDatabase() {
  try {
    console.log("Starting database seeding...");
    
    // Create default user
    const password = await bcrypt.hash('password', 10);
    const demoUser = {
      username: 'demo',
      password,
      email: 'demo@satyaai.com',
      fullName: 'Demo User'
    };
    
    // Check if user already exists
    const existingUser = await db.select().from(users).where(eq(users.username, demoUser.username));
    
    let userId = 1;
    if (existingUser.length === 0) {
      const createdUser = await storage.createUser(demoUser);
      userId = createdUser.id;
      console.log(`Created demo user with ID: ${userId}`);
    } else {
      userId = existingUser[0].id;
      console.log(`Demo user already exists with ID: ${userId}`);
    }
    
    // Check if we have scans already
    const existingScans = await db.select().from(scans);
    
    if (existingScans.length === 0) {
      console.log("Adding sample scans...");
      
      // Add sample scans
      await storage.createScan({
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
      });
      
      await storage.createScan({
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
      });
      
      await storage.createScan({
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
      });
      
      console.log("Sample scans added successfully");
    } else {
      console.log("Scans already exist, skipping sample data");
    }
    
    // Check and create user preferences if needed
    const existingPrefs = await db.select().from(userPreferences).where(eq(userPreferences.userId, userId));
    
    if (existingPrefs.length === 0) {
      await db.insert(userPreferences).values({
        userId,
        theme: 'dark',
        language: 'english',
        confidenceThreshold: 75,
        enableNotifications: true,
        autoAnalyze: true,
        sensitivityLevel: 'medium'
      });
      console.log(`Created default user preferences for user ID: ${userId}`);
    } else {
      console.log(`User preferences already exist for user ID: ${userId}`);
    }
    
    console.log("Database seeding completed successfully!");
    
  } catch (error) {
    console.error("Error seeding database:", error);
  }
}

// Only execute when run directly
if (require.main === module) {
  seedDatabase()
    .then(() => {
      console.log("Database initialization complete");
      process.exit(0);
    })
    .catch(error => {
      console.error("Database initialization failed:", error);
      process.exit(1);
    });
}

export default seedDatabase;