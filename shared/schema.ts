import { pgTable, text, integer, timestamp, boolean, varchar, uuid } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema (PostgreSQL)
export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  username: varchar("username", { length: 255 }).notNull().unique(),
  password: text("password").notNull(),
  email: varchar("email", { length: 255 }),
  fullName: varchar("full_name", { length: 255 }),
  apiKey: text("api_key").unique(),
  role: varchar("role", { length: 50 }).notNull().default("user"),
  failedLoginAttempts: integer("failed_login_attempts").notNull().default(0),
  lastFailedLogin: timestamp("last_failed_login"),
  isLocked: boolean("is_locked").notNull().default(false),
  lockoutUntil: timestamp("lockout_until"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

// Scan schema (PostgreSQL)
export const scans = pgTable("scans", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id, { onDelete: 'cascade' }),
  filename: text("filename").notNull(),
  type: varchar("type", { length: 50 }).notNull(), // 'image', 'video', 'audio'
  result: varchar("result", { length: 50 }).notNull(), // 'authentic', 'deepfake'
  confidenceScore: integer("confidence_score").notNull(),
  detectionDetails: text("detection_details"), // JSON as text
  metadata: text("metadata"), // JSON as text
  createdAt: timestamp("created_at").notNull().defaultNow(),
});

// User preferences schema (PostgreSQL)
export const userPreferences = pgTable("user_preferences", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id, { onDelete: 'cascade' }).notNull().unique(),
  theme: varchar("theme", { length: 50 }).default("dark"),
  language: varchar("language", { length: 50 }).default("english"),
  confidenceThreshold: integer("confidence_threshold").default(75),
  enableNotifications: boolean("enable_notifications").default(true),
  autoAnalyze: boolean("auto_analyze").default(true),
  sensitivityLevel: varchar("sensitivity_level", { length: 50 }).default("medium"),
});

// Tasks schema (PostgreSQL) - for task management system
export const tasks = pgTable("tasks", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").references(() => users.id, { onDelete: 'cascade' }).notNull(),
  type: varchar("type", { length: 50 }).notNull(), // 'image', 'video', 'audio', 'webcam', 'multimodal'
  status: varchar("status", { length: 50 }).notNull().default("queued"), // 'queued', 'processing', 'completed', 'failed'
  progress: integer("progress").notNull().default(0), // 0-100
  fileName: text("file_name").notNull(),
  fileSize: integer("file_size").notNull(), // in bytes
  fileType: varchar("file_type", { length: 100 }).notNull(), // MIME type
  filePath: text("file_path").notNull(),
  reportCode: varchar("report_code", { length: 100 }), // SATYA-IMG-20250118-0001
  result: text("result"), // JSON as text - analysis result
  error: text("error"), // JSON as text - error information
  metadata: text("metadata"), // JSON as text - additional metadata
  createdAt: timestamp("created_at").notNull().defaultNow(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
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

export const insertTaskSchema = createInsertSchema(tasks).pick({
  userId: true,
  type: true,
  status: true,
  progress: true,
  fileName: true,
  fileSize: true,
  fileType: true,
  filePath: true,
  reportCode: true,
  result: true,
  error: true,
  metadata: true,
});

// Types
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

export type InsertScan = z.infer<typeof insertScanSchema>;
export type Scan = typeof scans.$inferSelect;

export type InsertUserPreferences = z.infer<typeof insertUserPreferencesSchema>;
export type UserPreferences = typeof userPreferences.$inferSelect;

export type InsertTask = z.infer<typeof insertTaskSchema>;
export type Task = typeof tasks.$inferSelect;
