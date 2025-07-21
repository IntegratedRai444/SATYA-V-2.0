import { InsertScan, Scan, User, InsertUser } from "@shared/schema";
import { z } from "zod";
import { DatabaseStorage } from "./storage.db";

// Interface for storage operations
export interface IStorage {
  getUser(id: number): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  createScan(scan: InsertScan): Promise<Scan>;
  getScanById(id: number): Promise<Scan | undefined>;
  getAllScans(): Promise<Scan[]>;
  getRecentScans(limit?: number): Promise<Scan[]>;
  deleteScan(id: number): Promise<boolean>;
  getScansByUserId(userId: number): Promise<Scan[]>;
  getScansByType(type: string): Promise<Scan[]>;
  getScansByResult(result: string): Promise<Scan[]>;
  searchScansByFilename(query: string): Promise<Scan[]>;
}

// Export an instance of the database storage implementation
export const storage = new DatabaseStorage();
