import { sign, verify, VerifyErrors, JwtPayload } from 'jsonwebtoken';
import { v4 as uuidv4 } from 'uuid';
import redis from '../../config/redis';
import { logger } from '../../config/logger';

// In-memory token blacklist for when Redis is disabled
const inMemoryBlacklist = new Map<string, number>();

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
const ACCESS_TOKEN_EXPIRY = '15m';
const REFRESH_TOKEN_EXPIRY = '7d';
const BLACKLIST_PREFIX = 'token:blacklist:';

interface TokenPayload extends JwtPayload {
  sub: string;
  email: string;
  role: string;
  jti: string;
  sessionId: string;
  type: 'access' | 'refresh';
}

/**
 * Generate a new JWT token
 */
export const generateToken = (
  payload: Omit<TokenPayload, 'iat' | 'exp'>,
  expiresIn: string = ACCESS_TOKEN_EXPIRY
): string => {
  // Explicitly type the JWT payload
  const jwtPayload = {
    ...payload,
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + (expiresIn === ACCESS_TOKEN_EXPIRY ? 15 * 60 : 7 * 24 * 60 * 60)
  };

  // Sign with explicit type assertion for options
  return sign(
    jwtPayload,
    JWT_SECRET,
    {
      jwtid: payload.jti,
      algorithm: 'HS256'
    } as const
  );
};

/**
 * Verify a JWT token
 */
export const verifyToken = (token: string): Promise<TokenPayload> => {
  return new Promise((resolve, reject) => {
    verify(token, JWT_SECRET, (err: VerifyErrors | null, decoded) => {
      if (err) return reject(err);
      resolve(decoded as TokenPayload);
    });
  });
};

/**
 * Add a token to the blacklist
 */
async function blacklistToken(jti: string, expiresInSeconds: number): Promise<void> {
  try {
    if (redis.isRedisEnabled()) {
      await redis.setEx(
        `${BLACKLIST_PREFIX}${jti}`,
        expiresInSeconds,
        '1'
      );
    } else {
      // Use in-memory blacklist when Redis is disabled
      const expiresAt = Date.now() + (expiresInSeconds * 1000);
      inMemoryBlacklist.set(jti, expiresAt);
      
      // Clean up expired entries periodically
      setTimeout(() => {
        const now = Date.now();
        for (const [key, expiry] of inMemoryBlacklist.entries()) {
          if (expiry <= now) {
            inMemoryBlacklist.delete(key);
          }
        }
      }, 60 * 1000); // Clean up every minute
    }
  } catch (error) {
    logger.error('Failed to blacklist token:', error);
    throw new Error('Failed to blacklist token');
  }
};

/**
 * Check if a token is blacklisted
 */
async function isTokenBlacklisted(jti: string): Promise<boolean> {
  try {
    if (redis.isRedisEnabled()) {
      const result = await redis.get(`${BLACKLIST_PREFIX}${jti}`);
      return result !== null;
    } else {
      // Check in-memory blacklist
      const expiry = inMemoryBlacklist.get(jti);
      if (!expiry) return false;
      
      // Remove if expired
      if (expiry <= Date.now()) {
        inMemoryBlacklist.delete(jti);
        return false;
      }
      
      return true;
    }
  } catch (error) {
    logger.error('Failed to check token blacklist:', error);
    return false;
  }
}

/**
 * Generate a new token pair (access + refresh)
 */
export const generateTokenPair = (user: { id: string; email: string; role: string }) => {
  const sessionId = uuidv4();
  const accessTokenJti = uuidv4();
  const refreshTokenJti = uuidv4();

  const accessToken = generateToken({
    sub: user.id,
    email: user.email,
    role: user.role,
    type: 'access',
    jti: accessTokenJti,
    sessionId
  }, ACCESS_TOKEN_EXPIRY);

  const refreshToken = generateToken({
    sub: user.id,
    email: user.email,
    role: user.role,
    type: 'refresh',
    jti: refreshTokenJti,
    sessionId
  }, REFRESH_TOKEN_EXPIRY);

  return {
    accessToken,
    refreshToken,
    accessTokenJti,
    refreshTokenJti,
    sessionId,
    expiresIn: 15 * 60 // 15 minutes in seconds
  };
};

/**
 * Rotate refresh token (invalidate old one, generate new one)
 */
export const rotateRefreshToken = async (oldRefreshToken: string) => {
  try {
    // Verify the old refresh token
    const decoded = await verifyToken(oldRefreshToken);
    
    // Check if token is already blacklisted
    if (await isTokenBlacklisted(decoded.jti)) {
      throw new Error('Token has been revoked');
    }

    // Blacklist the old refresh token
    await blacklistToken(decoded.jti, 7 * 24 * 60 * 60); // 7 days

    // Generate new token pair
    const user = {
      id: decoded.sub,
      email: decoded.email,
      role: decoded.role
    };

    return generateTokenPair(user);
  } catch (error) {
    logger.error('Failed to rotate refresh token:', error);
    throw new Error('Invalid or expired refresh token');
  }
};
