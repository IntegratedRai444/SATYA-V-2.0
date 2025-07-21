import { InsertScan, Scan, User, InsertUser, insertScanSchema } from "@shared/schema";
import { IStorage } from "./storage";
import { db } from "./db";
import { eq } from "drizzle-orm";
import { users, scans } from "@shared/schema";

// Database implementation of the storage interface
export class DatabaseStorage implements IStorage {
  async getUser(id: number): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user;
  }

  async createUser(userData: InsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .returning();
    return user;
  }

  async createScan(scanData: InsertScan): Promise<Scan> {
    const [scan] = await db
      .insert(scans)
      .values(scanData)
      .returning();
    return scan;
  }

  async getScanById(id: number): Promise<Scan | undefined> {
    const [scan] = await db.select().from(scans).where(eq(scans.id, id));
    return scan;
  }

  async getAllScans(): Promise<Scan[]> {
    return await db.select().from(scans).orderBy(scans.createdAt);
  }

  async getRecentScans(limit: number = 3): Promise<Scan[]> {
    return await db
      .select()
      .from(scans)
      .orderBy(scans.createdAt)
      .limit(limit);
  }

  async deleteScan(id: number): Promise<boolean> {
    await db.delete(scans).where(eq(scans.id, id));
    return true;
  }

  // Additional helper methods
  
  async getScansByUserId(userId: number): Promise<Scan[]> {
    return await db
      .select()
      .from(scans)
      .where(eq(scans.userId, userId))
      .orderBy(scans.createdAt);
  }
  
  async getScansByType(type: string): Promise<Scan[]> {
    return await db
      .select()
      .from(scans)
      .where(eq(scans.type, type))
      .orderBy(scans.createdAt);
  }
  
  async getScansByResult(result: string): Promise<Scan[]> {
    return await db
      .select()
      .from(scans)
      .where(eq(scans.result, result))
      .orderBy(scans.createdAt);
  }
  
  async searchScansByFilename(query: string): Promise<Scan[]> {
    // Note: In a more advanced implementation, we'd use a LIKE query here
    return await db
      .select()
      .from(scans)
      .where(eq(scans.filename, query))
      .orderBy(scans.createdAt);
  }
}