import { db } from '../db';
import { users } from '@shared/schema';
import { eq, and } from 'drizzle-orm';
import crypto from 'crypto';
import { InferSelectModel } from 'drizzle-orm';

export class ApiKeyService {
  private static readonly KEY_LENGTH = 32;
  private static readonly HASH_ALGORITHM = 'sha256';
  private static readonly ENCODING = 'hex' as const;

  /**
   * Generate a new API key
   */
  static generateKey(): { key: string; hashedKey: string } {
    const key = crypto.randomBytes(this.KEY_LENGTH).toString('hex');
    const hashedKey = this.hashKey(key);
    return { key, hashedKey };
  }

  /**
   * Hash an API key
   */
  private static hashKey(key: string): string {
    return crypto
      .createHash(this.HASH_ALGORITHM)
      .update(key)
      .digest(this.ENCODING);
  }

  /**
   * Verify an API key
   */
  static verifyKey(key: string, hashedKey: string): boolean {
    return this.hashKey(key) === hashedKey;
  }

  /**
   * Get user by API key
   */
  static async getUserByApiKey(apiKey: string): Promise<{ id: number; email: string | null; role: string; apiKey: string | null } | null> {
    try {
      const hashedKey = this.hashKey(apiKey);
      
      const result = await db
        .select({
          id: users.id,
          email: users.email,
          role: users.role,
          apiKey: users.apiKey
        })
        .from(users)
        .where(eq(users.apiKey as any, hashedKey))
        .limit(1);

      return result[0] || null;
    } catch (error) {
      console.error('Error getting user by API key:', error);
      return null;
    }
  }
}

export const apiKeyService = new ApiKeyService();
