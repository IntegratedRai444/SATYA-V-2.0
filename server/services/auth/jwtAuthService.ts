import { verify, sign, JwtPayload, SignOptions, decode, Jwt } from 'jsonwebtoken';
import { logger } from '../../config/logger';

// Throw error if JWT_SECRET is not set or is using default value
const JWT_SECRET = process.env.JWT_SECRET || '';
if (!JWT_SECRET || JWT_SECRET === 'your-secret-key') {
  throw new Error('FATAL: JWT_SECRET is not properly configured in environment variables');
}

const JWT_ALGORITHM = 'HS256' as const;
const DEFAULT_EXPIRES_IN = '15m';
const REFRESH_EXPIRES_IN = '7d';

// In-memory token blacklist (in production, use Redis or database)
const tokenBlacklist = new Set<string>();

interface JwtPayloadExtended extends JwtPayload {
  userId: string;
  username?: string;
  iat: number;
  exp: number;
  nbf?: number;
  jti?: string;
}

export class JwtAuthService {
  /**
   * Verify a JWT token
   */
  static verifyToken(token: string): JwtPayloadExtended | null {
    try {
      // Check if token is blacklisted
      if (tokenBlacklist.has(token)) {
        logger.warn('Attempted to use blacklisted token');
        return null;
      }

      const decoded = verify(token, JWT_SECRET, { 
        algorithms: [JWT_ALGORITHM],
        ignoreExpiration: false,
        ignoreNotBefore: false,
        clockTolerance: 30, // 30 seconds clock skew
      });
      
      if (!decoded || typeof decoded === 'string') {
        logger.warn('Invalid token format');
        return null;
      }
      
      // Type assertion with proper type checking
      const payload = decoded as JwtPayloadExtended;
      
      // Use sub claim as userId if userId is not present
      if (!payload.userId && payload.sub) {
        payload.userId = payload.sub;
      }
      
      if (!payload.userId) {
        logger.warn('Invalid token: missing userId or sub claim');
        return null;
      }

      // Additional token validation
      const now = Math.floor(Date.now() / 1000);
      
      // Create a new object that matches JwtPayloadExtended
      const result: JwtPayloadExtended = {
        userId: payload.userId,
        iat: payload.iat || now,
        exp: payload.exp || now + 3600, // Default 1 hour if not provided
        ...(payload.username && { username: payload.username }),
        ...(payload.nbf && { nbf: payload.nbf }),
        ...(payload.jti && { jti: payload.jti })
      };

      // Validate timestamps
      if (!result.iat || !result.exp || result.exp <= now) {
        logger.warn('Invalid token: missing or expired timestamps');
        return null;
      }

      // Validate not before
      if (result.nbf && result.nbf > now) {
        logger.warn('Token not yet valid');
        return null;
      }

      return result;
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const errorStack = error instanceof Error ? error.stack : undefined;
      
      logger.error('JWT verification failed', { 
        error: errorMessage,
        stack: errorStack 
      });
      return null;
    }
  }

  /**
   * Sign a new JWT token
   */
  static signToken(payload: Omit<JwtPayloadExtended, 'iat' | 'exp' | 'nbf'>, expiresIn: string | number = DEFAULT_EXPIRES_IN): string {
    const now = Math.floor(Date.now() / 1000);
    
    // Calculate expiration time
    const expiresInSeconds = typeof expiresIn === 'number' 
      ? expiresIn 
      : expiresIn === '7d' 
        ? 7 * 24 * 60 * 60 // 7 days in seconds
        : 15 * 60; // Default to 15 minutes for any other string
    
    const options: SignOptions = {
      algorithm: JWT_ALGORITHM,
      expiresIn: expiresInSeconds,
      issuer: 'satya-ai',
      audience: ['web', 'mobile'],
      jwtid: Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15),
      notBefore: 0,
    };

    // Add security-related claims
    const enhancedPayload: JwtPayloadExtended = {
      ...payload,
      userId: payload.userId, // Ensure userId is included
      username: payload.username,
      iat: now,
      nbf: now - 30, // Add 30s clock tolerance
      exp: now + expiresInSeconds,
    };

    return sign(enhancedPayload, JWT_SECRET, options);
  }

  static invalidateToken(token: string): void {
    try {
      const decoded = decode(token, { complete: true, json: true });
      if (!decoded?.payload || typeof decoded.payload === 'string') {
        logger.warn('Invalid token format for invalidation');
        return;
      }
      const payload = decoded.payload as JwtPayloadExtended;
      
      if (payload?.exp) {
        const now = Math.floor(Date.now() / 1000);
        const expiresIn = payload.exp - now;
        
        if (expiresIn > 0) {
          tokenBlacklist.add(token);
          // Schedule removal from blacklist after token expires
          setTimeout(() => tokenBlacklist.delete(token), expiresIn * 1000);
        }
      }
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      logger.error('Failed to invalidate token', { error: errorMessage });
    }
  }
}

export const jwtAuthService = JwtAuthService;
