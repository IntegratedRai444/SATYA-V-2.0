import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import crypto from 'crypto';

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
  static async getUserByApiKey(apiKey: string): Promise<{ id: string; email: string | null; role: string; apiKey: string | null } | null> {
    try {
      const hashedKey = this.hashKey(apiKey);
      
      const { data: result, error } = await supabase
        .from('users')
        .select('id, email, role, api_key')
        .eq('api_key', hashedKey)
        .limit(1);

      if (error || !result || result.length === 0) return null;
      
      return {
        id: result[0].id,
        email: result[0].email,
        role: result[0].role,
        apiKey: result[0].api_key
      };
    } catch (error) {
      console.error('Error getting user by API key:', error);
      return null;
    }
  }
}

export const apiKeyService = new ApiKeyService();
