import { sqliteTable, text, integer, real } from "drizzle-orm/sqlite-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema
export const users = sqliteTable("users", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
  email: text("email"),
  fullName: text("full_name"),
  role: text("role").notNull().default("user"),
  createdAt: integer("created_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
  updatedAt: integer("updated_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
});

// Scan schema
export const scans = sqliteTable("scans", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  userId: integer("user_id").references(() => users.id),
  filename: text("filename").notNull(),
  type: text("type").notNull(), // 'image', 'video', 'audio'
  result: text("result").notNull(), // 'authentic', 'deepfake'
  confidenceScore: integer("confidence_score").notNull(),
  detectionDetails: text("detection_details"), // JSON as text for SQLite
  metadata: text("metadata"), // JSON as text for SQLite
  createdAt: integer("created_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
});

// User preferences schema
export const userPreferences = sqliteTable("user_preferences", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  userId: integer("user_id").references(() => users.id).notNull().unique(),
  theme: text("theme").default("dark"),
  language: text("language").default("english"),
  confidenceThreshold: integer("confidence_threshold").default(75),
  enableNotifications: integer("enable_notifications", { mode: 'boolean' }).default(true),
  autoAnalyze: integer("auto_analyze", { mode: 'boolean' }).default(true),
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
