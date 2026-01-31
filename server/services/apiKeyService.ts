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
        .from('api_keys')
        .select(`
          user_id,
          users!inner (
            id,
            email,
            role
          )
        `)
        .eq('key_hash', hashedKey)
        .eq('is_active', true)
        .is('deleted_at', null)
        .single() as { data: { user_id: string; users: { id: string; email: string | null; role: string } } | null; error: { message: string } | null };

      if (error || !result) {
        logger.warn('API key lookup failed', { error: error?.message });
        return null;
      }
      
      return {
        id: result.users.id,
        email: result.users.email,
        role: result.users.role,
        apiKey: hashedKey
      };
    } catch (error) {
      logger.error('Error getting user by API key:', error);
      return null;
    }
  }
}

export const apiKeyService = new ApiKeyService();
