import 'dotenv/config';
import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { db } from "./db";
import { users, scans, userPreferences } from "@shared/schema";
import { eq } from "drizzle-orm";
import * as bcrypt from "bcrypt";
import * as pythonBridge from "./python-bridge-adapter";

const app = express();
app.use(express.json({ limit: '50mb' })); // Increase limit for base64 images
app.use(express.urlencoded({ extended: false, limit: '50mb' }));

// Logging middleware
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        // Truncate large response bodies
        const responseStr = JSON.stringify(capturedJsonResponse);
        logLine += ` :: ${responseStr.length > 100 ? responseStr.slice(0, 100) + "…" : responseStr}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      log(logLine);
    }
  });

  next();
});

// Setup CORS for development
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept, Authorization");
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }
  next();
});

// Initialize database and seed data
async function seedInitialData() {
  try {
    log("Initializing database...");
    
    // Check if we already have users
    const existingUsers = await db.select().from(users);
    
    if (existingUsers.length === 0) {
      log("Creating demo user...");
      // Create default user
      const hashedPassword = await bcrypt.hash('password', 10);
      const [user] = await db.insert(users).values({
        username: 'demo',
        password: hashedPassword,
        email: 'demo@satyaai.com',
        fullName: 'Demo User'
      }).returning();
      
      log(`Created demo user with ID: ${user.id}`);
      
      // Create sample scans
      const sampleScans = [
        {
          userId: user.id,
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
          userId: user.id,
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
          userId: user.id,
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
      
      for (const scan of sampleScans) {
        await db.insert(scans).values(scan);
      }
      
      log("Added sample scans");
      
      // Create user preferences
      await db.insert(userPreferences).values({
        userId: user.id,
        theme: 'dark',
        language: 'english',
        confidenceThreshold: 75,
        enableNotifications: true,
        autoAnalyze: true,
        sensitivityLevel: 'medium'
      });
      
      log("Created default user preferences");
    } else {
      log("Database already contains data, skipping initialization");
    }
    
    log("Database initialization complete");
  } catch (error) {
    log(`Database initialization error: ${error}`);
  }
}

// Main application bootstrap
(async () => {
  try {
    // Initialize and seed database
    await seedInitialData();
    
    // Initialize advanced detection capabilities
    console.log("Initializing advanced detection capabilities");
    // Start the Python server for deepfake detection
    try {
      await pythonBridge.startPythonServer();
    } catch (error) {
      console.log(`Python server initialization failed: ${error.message}`);
      console.log("Continuing with mock detection services...");
    }
    
    // Register API routes
    const server = await registerRoutes(app);

    // Global error handler
    app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
      const status = err.status || err.statusCode || 500;
      const message = err.message || "Internal Server Error";
      
      log(`Error: ${err.message}`);
      log(err.stack);

      res.status(status).json({ message });
    });

    // Set up Vite or static file serving
    if (app.get("env") === "development") {
      await setupVite(app, server);
    } else {
      serveStatic(app);
    }

    // Start server
    const port = process.env.PORT || 3000;
    server.listen({
      port,
      host: "0.0.0.0",
      reusePort: true,
    }, () => {
      log(`SatyaAI server running on port ${port}`);
    });
  } catch (error) {
    log(`Failed to start server: ${error}`);
    process.exit(1);
  }
})();
