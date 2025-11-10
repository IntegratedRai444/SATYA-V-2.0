import { v4 as uuidv4 } from 'uuid';
import { createHash, randomBytes } from 'crypto';
import { Request } from 'express';
import { logger } from '../config/logger';

interface ApiKey {
  id: string;
  key?: string; // Made optional since we don't store the raw key
  hashedKey: string;
  name: string;
  userId: string;
  expiresAt?: Date;
  lastUsedAt?: Date;
  createdAt: Date;
  isActive: boolean;
  permissions: string[];
  metadata?: Record<string, any>;
}

// In-memory store (replace with database in production)
const apiKeyStore = new Map<string, ApiKey>();

class ApiKeyService {
  private static instance: ApiKeyService;
  
  private constructor() {}
  
  public static getInstance(): ApiKeyService {
    if (!ApiKeyService.instance) {
      ApiKeyService.instance = new ApiKeyService();
    }
    return ApiKeyService.instance;
  }

  // Generate a new API key
  public async generateKey(userId: string, name: string, permissions: string[] = [], expiresInDays?: number): Promise<{ key: string; apiKey: Omit<ApiKey, 'key' | 'hashedKey'> }> {
    const rawKey = `sk_${uuidv4().replace(/-/g, '')}`;
    const hashedKey = this.hashKey(rawKey);
    const expiresAt = expiresInDays ? new Date(Date.now() + expiresInDays * 24 * 60 * 60 * 1000) : undefined;
    
    const apiKey: ApiKey = {
      id: uuidv4(),
      key: rawKey,
      hashedKey,
      name,
      userId,
      expiresAt,
      createdAt: new Date(),
      isActive: true,
      permissions,
    };
    
    // Store the hashed key (not the raw key)
    const { key: _, ...storedKey } = apiKey;
    apiKeyStore.set(apiKey.id, { ...storedKey, hashedKey });
    
    return { key: rawKey, apiKey: storedKey };
  }

  // Verify an API key
  public async verifyKey(apiKey: string): Promise<{ isValid: boolean; keyData?: Omit<ApiKey, 'key' | 'hashedKey'> }> {
    const hashedKey = this.hashKey(apiKey);
    
    // Find the key in the store
    const keyEntry = Array.from(apiKeyStore.values()).find(
      (key) => key.hashedKey === hashedKey
    );
    
    if (!keyEntry || !keyEntry.isActive) {
      return { isValid: false };
    }
    
    // Check if key is expired
    if (keyEntry.expiresAt && new Date() > keyEntry.expiresAt) {
      logger.warn(`Expired API key used: ${keyEntry.id}`);
      return { isValid: false };
    }
    
    // Update last used timestamp
    keyEntry.lastUsedAt = new Date();
    apiKeyStore.set(keyEntry.id, keyEntry);
    
    // Return key data without sensitive information
    const { hashedKey: _, ...keyData } = keyEntry;
    return { isValid: true, keyData };
  }
  
  // Rotate an API key (invalidate old, create new)
  public async rotateKey(keyId: string, userId: string): Promise<{ newKey: string; oldKey: Omit<ApiKey, 'key' | 'hashedKey'> }> {
    const oldKey = apiKeyStore.get(keyId);
    
    if (!oldKey || oldKey.userId !== userId) {
      throw new Error('API key not found or access denied');
    }
    
    // Generate a new key with the same permissions and metadata
    const { key: newKey, apiKey } = await this.generateKey(
      userId,
      `${oldKey.name} (rotated)`,
      oldKey.permissions,
      oldKey.expiresAt ? Math.ceil((oldKey.expiresAt.getTime() - Date.now()) / (1000 * 60 * 60 * 24)) : undefined
    );
    
    // Mark old key as inactive
    oldKey.isActive = false;
    apiKeyStore.set(keyId, oldKey);
    
    return {
      newKey,
      oldKey: {
        id: oldKey.id,
        name: oldKey.name,
        userId: oldKey.userId,
        expiresAt: oldKey.expiresAt,
        lastUsedAt: oldKey.lastUsedAt,
        createdAt: oldKey.createdAt,
        isActive: oldKey.isActive,
        permissions: oldKey.permissions,
        metadata: oldKey.metadata
      }
    };
  }
  
  // Revoke an API key
  public async revokeKey(keyId: string, userId: string): Promise<boolean> {
    const key = apiKeyStore.get(keyId);
    
    if (key && key.userId === userId) {
      key.isActive = false;
      apiKeyStore.set(keyId, key);
      return true;
    }
    
    return false;
  }
  
  // List all API keys for a user
  public async listUserKeys(userId: string): Promise<Omit<ApiKey, 'key' | 'hashedKey'>[]> {
    return Array.from(apiKeyStore.values())
      .filter(key => key.userId === userId)
      .map(({ key: _, hashedKey: __, ...rest }) => rest);
  }

  // Track API key usage
  public async trackApiKeyUsage(apiKey: string, req: Request): Promise<boolean> {
    try {
      const hashedKey = this.hashKey(apiKey);
      const keyEntry = Array.from(apiKeyStore.values()).find(
        (key) => key.hashedKey === hashedKey
      );

      if (keyEntry) {
        keyEntry.lastUsedAt = new Date();
        // Here you would typically update the database
        // await db.updateApiKeyUsage(keyEntry.id, { lastUsedAt: keyEntry.lastUsedAt });
        
        // Log the usage
        logger.info('API key used', {
          keyId: keyEntry.id,
          userId: keyEntry.userId,
          path: req.path,
          method: req.method,
          ip: req.ip,
          userAgent: req.get('user-agent')
        });
        
        return true;
      }
      return false;
    } catch (error) {
      logger.error('Failed to track API key usage', { error });
      return false;
    }
  }
  
  // Hash the API key for secure storage
  private hashKey(key: string): string {
    return createHash('sha256').update(key).digest('hex');
  }
  
  // Generate a secure random string
  public generateRandomString(length = 32): string {
    return randomBytes(Math.ceil(length / 2))
      .toString('hex')
      .slice(0, length);
  }
}

export const apiKeyService = ApiKeyService.getInstance();
