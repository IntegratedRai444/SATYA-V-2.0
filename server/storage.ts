/**
 * Storage interface and implementation for SatyaAI
 * Provides abstraction layer for data persistence
 */

import { db } from './db';
import { users, scans, userPreferences } from '@shared/schema';
import { eq, desc, and } from 'drizzle-orm';
import type { InsertUser, User, InsertScan, Scan } from '@shared/schema';

export interface IStorage {
  // User operations
  createUser(user: InsertUser): Promise<User>;
  getUserById(id: number): Promise<User | null>;
  getUserByUsername(username: string): Promise<User | null>;
  updateUser(id: number, updates: Partial<User>): Promise<User | null>;
  
  // Scan operations
  createScan(scan: InsertScan): Promise<Scan>;
  getScanById(id: number): Promise<Scan | null>;
  getScansByUserId(userId: number, limit?: number): Promise<Scan[]>;
  updateScan(id: number, updates: Partial<Scan>): Promise<Scan | null>;
  deleteScan(id: number): Promise<boolean>;
  
  // Analytics
  getUserStats(userId: number): Promise<{
    totalScans: number;
    authenticScans: number;
    deepfakeScans: number;
    avgConfidence: number;
  }>;
}

class DatabaseStorage implements IStorage {
  async createUser(user: InsertUser): Promise<User> {
    const [newUser] = await db.insert(users).values(user).returning();
    return newUser;
  }

  async getUserById(id: number): Promise<User | null> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || null;
  }

  async getUserByUsername(username: string): Promise<User | null> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || null;
  }

  async updateUser(id: number, updates: Partial<User>): Promise<User | null> {
    const [updatedUser] = await db
      .update(users)
      .set(updates)
      .where(eq(users.id, id))
      .returning();
    return updatedUser || null;
  }

  async createScan(scan: InsertScan): Promise<Scan> {
    const [newScan] = await db.insert(scans).values(scan).returning();
    return newScan;
  }

  async getScanById(id: number): Promise<Scan | null> {
    const [scan] = await db.select().from(scans).where(eq(scans.id, id));
    return scan || null;
  }

  async getScansByUserId(userId: number, limit: number = 50): Promise<Scan[]> {
    return await db
      .select()
      .from(scans)
      .where(eq(scans.userId, userId))
      .orderBy(desc(scans.createdAt))
      .limit(limit);
  }

  async updateScan(id: number, updates: Partial<Scan>): Promise<Scan | null> {
    const [updatedScan] = await db
      .update(scans)
      .set(updates)
      .where(eq(scans.id, id))
      .returning();
    return updatedScan || null;
  }

  async deleteScan(id: number): Promise<boolean> {
    const result = await db.delete(scans).where(eq(scans.id, id));
    return result.changes > 0;
  }

  async getUserStats(userId: number): Promise<{
    totalScans: number;
    authenticScans: number;
    deepfakeScans: number;
    avgConfidence: number;
  }> {
    const userScans = await db
      .select()
      .from(scans)
      .where(eq(scans.userId, userId));

    const totalScans = userScans.length;
    const authenticScans = userScans.filter(scan => scan.result === 'authentic').length;
    const deepfakeScans = userScans.filter(scan => scan.result === 'deepfake' || scan.result === 'manipulated').length;
    
    const avgConfidence = totalScans > 0 
      ? Math.round(userScans.reduce((sum, scan) => sum + (scan.confidenceScore || 0), 0) / totalScans)
      : 0;

    return {
      totalScans,
      authenticScans,
      deepfakeScans,
      avgConfidence
    };
  }
}

// Export singleton instance
export const storage = new DatabaseStorage();
export default storage;