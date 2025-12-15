import { Request, Response, NextFunction } from 'express';
import jwt, { JwtPayload, VerifyErrors } from 'jsonwebtoken';
import { logger } from '../config/logger';
import { rateLimit } from 'express-rate-limit';
import { redis } from '../db';

// Rate limiting configuration
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  standardHeaders: true,
  legacyHeaders: false,
  message: 'Too many requests from this IP, please try again after 15 minutes',
  skip: (req) => {
    // Skip rate limiting for health checks and static assets
    return req.path === '/health' || req.path.startsWith('/static');
  }
});

// Token blacklist check
async function isTokenBlacklisted(token: string): Promise<boolean> {
  if (!redis) return false;
  try {
    const result = await redis.get(`blacklist:${token}`);
    return result !== null;
  } catch (error) {
    logger.error('Redis blacklist check failed', { error });
    return false;
  }
}

// Add token to blacklist
export async function blacklistToken(token: string, expiresIn: number): Promise<void> {
  if (!redis) return;
  try {
    await redis.set(`blacklist:${token}`, '1', 'EX', expiresIn);
  } catch (error) {
    logger.error('Failed to blacklist token', { error });
  }
}

declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
        exp: number;
        iat: number;
      };
    }
  }
}

export const authenticateJWT = async (req: Request, res: Response, next: NextFunction) => {
  // Apply rate limiting
  authLimiter(req, res, async () => {
    try {
      const authHeader = req.headers.authorization;
      
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        logger.warn('Missing or invalid Authorization header');
        return res.status(401).json({ 
          error: 'Unauthorized',
          message: 'Authentication token is required' 
        });
      }

      const token = authHeader.split(' ')[1];
      
      // Check token blacklist
      if (await isTokenBlacklisted(token)) {
        logger.warn('Attempt to use blacklisted token');
        return res.status(401).json({ 
          error: 'Unauthorized',
          message: 'Token has been revoked' 
        });
      }

      // Verify JWT
      jwt.verify(token, process.env.JWT_SECRET!, 
        async (err: VerifyErrors | null, decoded: any) => {
          if (err) {
            const errorMessage = err.name === 'TokenExpiredError' 
              ? 'Token has expired' 
              : 'Invalid token';
            
            logger.warn('JWT verification failed', { 
              error: err.message,
              path: req.path,
              ip: req.ip 
            });
            
            return res.status(401).json({ 
              error: 'Unauthorized',
              message: errorMessage 
            });
          }

          // Check token expiration
          const now = Math.floor(Date.now() / 1000);
          if (decoded.exp && decoded.exp < now) {
            return res.status(401).json({ 
              error: 'Unauthorized',
              message: 'Token has expired' 
            });
          }

          // Set user in request
          req.user = {
            id: decoded.id,
            email: decoded.email,
            role: decoded.role,
            exp: decoded.exp,
            iat: decoded.iat
          };

          // Add token to request for potential rate limiting or logging
          (req as any).token = token;
          
          next();
        }
      );
    } catch (error) {
      logger.error('Authentication error', { 
        error: (error as Error).message,
        path: req.path,
        ip: req.ip 
      });
      
      return res.status(500).json({ 
        error: 'Internal Server Error',
        message: 'An error occurred during authentication' 
      });
    }
  });
};

export const requireRole = (role: string) => {
  return (req: Request, res: Response, next: NextFunction) => {
    if (req.user?.role === role) {
      next();
    } else {
      res.status(403).json({ error: 'Insufficient permissions' });
    }
  };
};

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  logger.error('API Error', {
    error: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
  });

  res.status(500).json({
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined,
  });
};
