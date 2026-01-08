import { PgTableWithColumns } from 'drizzle-orm/pg-core';
import { AnyPgColumn } from 'drizzle-orm/pg-core';

// User schema
export const users: PgTableWithColumns<{
  id: typeof import('drizzle-orm/pg-core').serial;
  username: string;
  password: string;
  email: string | null;
  fullName: string | null;
  apiKey: string | null;
  role: string;
  createdAt: Date;
  updatedAt: Date;
}>;

// Scan schema
export const scans: PgTableWithColumns<{
  id: typeof import('drizzle-orm/pg-core').serial;
  userId: number | null;
  status: string;
  result: string | null;
  createdAt: Date;
  updatedAt: Date;
}>;

// User preferences schema
export const userPreferences: PgTableWithColumns<{
  id: typeof import('drizzle-orm/pg-core').serial;
  userId: number;
  theme: string;
  notifications: boolean;
  language: string;
  createdAt: Date;
  updatedAt: Date;
}>;

// Tasks schema
export const tasks: PgTableWithColumns<{
  id: typeof import('drizzle-orm/pg-core').serial;
  name: string;
  status: string;
  progress: number | null;
  result: string | null;
  error: string | null;
  startedAt: Date | null;
  completedAt: Date | null;
  createdAt: Date;
  updatedAt: Date;
}>;

// Insert types
export type InsertUser = typeof users.$inferInsert;
export type User = typeof users.$inferSelect;
export type InsertScan = typeof scans.$inferInsert;
export type Scan = typeof scans.$inferSelect;
export type InsertUserPreferences = typeof userPreferences.$inferInsert;
export type UserPreferences = typeof userPreferences.$inferSelect;
export type InsertTask = typeof tasks.$inferInsert;
export type Task = typeof tasks.$inferSelect;
