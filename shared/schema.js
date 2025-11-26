import { pgTable, text, integer, serial, timestamp, boolean, varchar } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
// User schema (PostgreSQL)
export const users = pgTable("users", {
    id: serial("id").primaryKey(),
    username: varchar("username", { length: 255 }).notNull().unique(),
    password: text("password").notNull(),
    email: varchar("email", { length: 255 }),
    fullName: varchar("full_name", { length: 255 }),
    role: varchar("role", { length: 50 }).notNull().default("user"),
    createdAt: timestamp("created_at").notNull().defaultNow(),
    updatedAt: timestamp("updated_at").notNull().defaultNow(),
});
// Scan schema (PostgreSQL)
export const scans = pgTable("scans", {
    id: serial("id").primaryKey(),
    userId: integer("user_id").references(() => users.id),
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
    id: serial("id").primaryKey(),
    userId: integer("user_id").references(() => users.id).notNull().unique(),
    theme: varchar("theme", { length: 50 }).default("dark"),
    language: varchar("language", { length: 50 }).default("english"),
    confidenceThreshold: integer("confidence_threshold").default(75),
    enableNotifications: boolean("enable_notifications").default(true),
    autoAnalyze: boolean("auto_analyze").default(true),
    sensitivityLevel: varchar("sensitivity_level", { length: 50 }).default("medium"),
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
