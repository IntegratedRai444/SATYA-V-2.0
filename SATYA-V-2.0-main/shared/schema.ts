import { pgTable, text, serial, integer, timestamp, boolean, json } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema
export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  email: text("email"),
  fullName: text("full_name"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// Scan schema
export const scans = pgTable("scans", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id),
  filename: text("filename").notNull(),
  type: text("type").notNull(), // 'image', 'video', 'audio'
  result: text("result").notNull(), // 'authentic', 'deepfake'
  confidenceScore: integer("confidence_score").notNull(),
  detectionDetails: json("detection_details"),
  metadata: json("metadata"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

// User preferences schema
export const userPreferences = pgTable("user_preferences", {
  id: serial("id").primaryKey(),
  userId: integer("user_id").references(() => users.id).notNull().unique(),
  theme: text("theme").default("dark"),
  language: text("language").default("english"),
  confidenceThreshold: integer("confidence_threshold").default(75),
  enableNotifications: boolean("enable_notifications").default(true),
  autoAnalyze: boolean("auto_analyze").default(true),
  sensitivityLevel: text("sensitivity_level").default("medium"),
});

// Insert schemas
export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
  email: true,
  fullName: true,
});

export const insertScanSchema = createInsertSchema(scans).pick({
  userId: true,
  filename: true,
  type: true,
  result: true,
  confidenceScore: true,
  detectionDetails: true,
  metadata: true,
});

export const insertUserPreferencesSchema = createInsertSchema(userPreferences).pick({
  userId: true,
  theme: true,
  language: true,
  confidenceThreshold: true,
  enableNotifications: true,
  autoAnalyze: true,
  sensitivityLevel: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertScan = z.infer<typeof insertScanSchema>;
export type Scan = typeof scans.$inferSelect;

export type InsertUserPreferences = z.infer<typeof insertUserPreferencesSchema>;
export type UserPreferences = typeof userPreferences.$inferSelect;
