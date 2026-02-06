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
   * Get user by API key with security enforcement
   */
  static async getUserByApiKey(apiKey: string, ipAddress?: string): Promise<{ id: string; email: string | null; role: string; apiKey: string | null } | null> {
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
          ),
          expires_at,
          rate_limit_per_minute,
          last_used_at,
          last_used_ip
        `)
        .eq('key_hash', hashedKey)
        .eq('is_active', true)
        .is('deleted_at', null)
        .single() as { data: { 
          user_id: string; 
          users: { id: string; email: string | null; role: string };
          expires_at: string | null;
          rate_limit_per_minute: number;
          last_used_at: string | null;
          last_used_ip: string | null;
        } | null; 
        error: { message: string } | null };

      if (error || !result) {
        logger.warn('API key lookup failed', { error: error?.message });
        return null;
      }

      // Check if API key has expired
      if (result.expires_at && new Date() > new Date(result.expires_at)) {
        logger.warn('API key expired', { userId: result.users.id, expiresAt: result.expires_at });
        return null;
      }

      // Check rate limiting (simple implementation - in production, use Redis)
      const now = new Date();
      const oneMinuteAgo = new Date(now.getTime() - 60 * 1000);
      
      if (result.last_used_at && new Date(result.last_used_at) > oneMinuteAgo) {
        // This is a simplified rate limit check
        // In production, you'd want to track actual request count per minute
        if (result.rate_limit_per_minute <= 1) {
          logger.warn('API key rate limit exceeded', { 
            userId: result.users.id, 
            rateLimit: result.rate_limit_per_minute 
          });
          return null;
        }
      }

      // Update usage tracking
      await supabase
        .from('api_keys')
        .update({
          last_used_at: now.toISOString(),
          last_used_ip: ipAddress || 'unknown'
        })
        .eq('user_id', result.users.id)
        .eq('key_hash', hashedKey);
      
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
