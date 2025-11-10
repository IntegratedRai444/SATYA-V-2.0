import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { jwtAuthService } from '../services/jwt-auth-service';
import { sessionManager } from '../services/session-manager';
import { logger, logSecurity } from '../config/logger';
import { config } from '../config';

const router = Router();

// Validation schemas
const registerSchema = z.object({
  username: z.string()
    .min(3, 'Username must be at least 3 characters')
    .max(50, 'Username must be less than 50 characters')
    .regex(/^[a-zA-Z0-9_-]+$/, 'Username can only contain letters, numbers, underscores, and hyphens'),
  password: z.string()
    .min(8, 'Password must be at least 8 characters')
    .max(128, 'Password must be less than 128 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, 'Password must contain at least one lowercase letter, one uppercase letter, and one number'),
  email: z.string()
    .email('Invalid email format')
    .optional(),
  fullName: z.string()
    .min(1, 'Full name is required')
    .max(100, 'Full name must be less than 100 characters')
    .optional()
});

const loginSchema = z.object({
  username: z.string()
    .min(1, 'Username is required'),
  password: z.string()
    .min(1, 'Password is required')
});

const changePasswordSchema = z.object({
  currentPassword: z.string()
    .min(1, 'Current password is required'),
  newPassword: z.string()
    .min(8, 'New password must be at least 8 characters')
    .max(128, 'New password must be less than 128 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, 'New password must contain at least one lowercase letter, one uppercase letter, and one number')
});

// Helper function to extract token from request
function extractToken(req: Request): string | null {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.startsWith('Bearer ')) {
    return authHeader.substring(7);
  }
  return null;
}

// Helper function to get client IP
function getClientIP(req: Request): string {
  return req.ip || req.connection.remoteAddress || 'unknown';
}

/**
 * POST /api/auth/register
 * Register a new user account
 */
router.post('/register', async (req: Request, res: Response) => {
  try {
    // Validate request body
    const validatedData = registerSchema.parse(req.body);
    
    // Log registration attempt
    logger.info('User registration attempt', {
      username: validatedData.username,
      email: validatedData.email,
      ip: getClientIP(req),
      userAgent: req.get('User-Agent')
    });

    // Attempt registration
    const result = await jwtAuthService.register(validatedData);

    if (result.success && result.token && result.user) {
      // Create session
      const payload = await jwtAuthService.verifyToken(result.token);
      if (payload) {
        await sessionManager.createSession(
          result.token,
          payload,
          getClientIP(req),
          req.get('User-Agent') || 'Unknown'
        );
      }

      // Log successful registration
      logSecurity('User registered successfully', {
        username: validatedData.username,
        userId: result.user.id,
        ip: getClientIP(req)
      });

      // Return success response (exclude sensitive data)
      res.status(201).json({
        success: true,
        message: result.message,
        token: result.token,
        user: {
          id: result.user.id,
          username: result.user.username,
          email: result.user.email,
          role: result.user.role
        }
      });
    } else {
      // Log failed registration
      logSecurity('User registration failed', {
        username: validatedData.username,
        reason: result.message,
        ip: getClientIP(req)
      });

      res.status(400).json({
        success: false,
        message: result.message
      });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      // Validation error
      const errorMessages = error.errors.map(err => `${err.path.join('.')}: ${err.message}`);
      
      logger.warn('Registration validation failed', {
        errors: errorMessages,
        ip: getClientIP(req)
      });

      res.status(400).json({
        success: false,
        message: 'Validation failed',
        errors: errorMessages
      });
    } else {
      // Internal server error
      logger.error('Registration internal error', {
        error: (error as Error).message,
        stack: (error as Error).stack,
        ip: getClientIP(req)
      });

      res.status(500).json({
        success: false,
        message: config.NODE_ENV === 'production' ? 'Registration failed' : (error as Error).message
      });
    }
  }
});

/**
 * POST /api/auth/login
 * Authenticate user and return JWT token
 */
router.post('/login', async (req: Request, res: Response) => {
  try {
    // Validate request body
    const validatedData = loginSchema.parse(req.body);
    
    // Log login attempt
    logger.info('User login attempt', {
      username: validatedData.username,
      ip: getClientIP(req),
      userAgent: req.get('User-Agent')
    });

    // Attempt login
    const result = await jwtAuthService.login(validatedData);

    if (result.success && result.token && result.user) {
      // Create session
      const payload = await jwtAuthService.verifyToken(result.token);
      if (payload) {
        await sessionManager.createSession(
          result.token,
          payload,
          getClientIP(req),
          req.get('User-Agent') || 'Unknown'
        );
      }

      // Log successful login
      logSecurity('User logged in successfully', {
        username: validatedData.username,
        userId: result.user.id,
        ip: getClientIP(req)
      });

      res.json({
        success: true,
        message: result.message,
        token: result.token,
        user: {
          id: result.user.id,
          username: result.user.username,
          email: result.user.email,
          role: result.user.role
        }
      });
    } else {
      // Log failed login
      logSecurity('User login failed', {
        username: validatedData.username,
        reason: result.message,
        ip: getClientIP(req)
      });

      res.status(401).json({
        success: false,
        message: result.message
      });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      // Validation error
      const errorMessages = error.errors.map(err => `${err.path.join('.')}: ${err.message}`);
      
      logger.warn('Login validation failed', {
        errors: errorMessages,
        ip: getClientIP(req)
      });

      res.status(400).json({
        success: false,
        message: 'Validation failed',
        errors: errorMessages
      });
    } else {
      // Internal server error
      logger.error('Login internal error', {
        error: (error as Error).message,
        stack: (error as Error).stack,
        ip: getClientIP(req)
      });

      res.status(500).json({
        success: false,
        message: config.NODE_ENV === 'production' ? 'Login failed' : (error as Error).message
      });
    }
  }
});

/**
 * POST /api/auth/logout
 * Logout user and invalidate token
 */
router.post('/logout', async (req: Request, res: Response) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token first to get user info for logging
    const payload = await jwtAuthService.verifyToken(token);
    
    // Destroy session
    const sessionDestroyed = await sessionManager.destroySession(token);
    
    if (sessionDestroyed) {
      // Log successful logout
      logSecurity('User logged out successfully', {
        userId: payload?.userId,
        username: payload?.username,
        ip: getClientIP(req)
      });

      res.json({
        success: true,
        message: 'Logout successful'
      });
    } else {
      res.status(500).json({
        success: false,
        message: 'Logout failed'
      });
    }
  } catch (error) {
    logger.error('Logout internal error', {
      error: (error as Error).message,
      stack: (error as Error).stack,
      ip: getClientIP(req)
    });

    res.status(500).json({
      success: false,
      message: config.NODE_ENV === 'production' ? 'Logout failed' : (error as Error).message
    });
  }
});

/**
 * GET /api/auth/session
 * Validate current session and return user info
 */
router.get('/session', async (req: Request, res: Response) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token
    const payload = await jwtAuthService.verifyToken(token);
    
    if (!payload) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }

    // Get fresh user data
    const user = await jwtAuthService.getUserById(payload.userId);
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        fullName: user.fullName,
        role: user.role,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    logger.error('Session validation error', {
      error: (error as Error).message,
      stack: (error as Error).stack,
      ip: getClientIP(req)
    });

    res.status(500).json({
      success: false,
      message: config.NODE_ENV === 'production' ? 'Session validation failed' : (error as Error).message
    });
  }
});

/**
 * POST /api/auth/change-password
 * Change user password (requires authentication)
 */
router.post('/change-password', async (req: Request, res: Response) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token
    const payload = await jwtAuthService.verifyToken(token);
    
    if (!payload) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }

    // Validate request body
    const validatedData = changePasswordSchema.parse(req.body);
    
    // Log password change attempt
    logSecurity('Password change attempt', {
      userId: payload.userId,
      username: payload.username,
      ip: getClientIP(req)
    });

    // Attempt password change
    const result = await jwtAuthService.updatePassword(
      payload.userId,
      validatedData.currentPassword,
      validatedData.newPassword
    );

    if (result.success) {
      // Log successful password change
      logSecurity('Password changed successfully', {
        userId: payload.userId,
        username: payload.username,
        ip: getClientIP(req)
      });

      res.json({
        success: true,
        message: result.message
      });
    } else {
      // Log failed password change
      logSecurity('Password change failed', {
        userId: payload.userId,
        username: payload.username,
        reason: result.message,
        ip: getClientIP(req)
      });

      res.status(400).json({
        success: false,
        message: result.message
      });
    }
  } catch (error) {
    if (error instanceof z.ZodError) {
      // Validation error
      const errorMessages = error.errors.map(err => `${err.path.join('.')}: ${err.message}`);
      
      res.status(400).json({
        success: false,
        message: 'Validation failed',
        errors: errorMessages
      });
    } else {
      // Internal server error
      logger.error('Change password internal error', {
        error: (error as Error).message,
        stack: (error as Error).stack,
        ip: getClientIP(req)
      });

      res.status(500).json({
        success: false,
        message: config.NODE_ENV === 'production' ? 'Password change failed' : (error as Error).message
      });
    }
  }
});

/**
 * POST /api/auth/refresh
 * Refresh authentication token
 */
router.post('/refresh', async (req: Request, res: Response) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify current token
    const payload = await jwtAuthService.verifyToken(token);
    
    if (!payload) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }

    // Get fresh user data
    const user = await jwtAuthService.getUserById(payload.userId);
    
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'User not found'
      });
    }

    // Generate new token
    const newToken = await jwtAuthService.generateToken({
      userId: user.id,
      username: user.username,
      email: user.email,
      role: user.role
    });

    // Update session with new token
    const sessionRefreshed = await sessionManager.refreshSession(token, newToken);
    
    if (!sessionRefreshed) {
      return res.status(401).json({
        success: false,
        message: 'Session refresh failed'
      });
    }

    // Log successful token refresh
    logSecurity('Token refreshed successfully', {
      userId: user.id,
      username: user.username,
      ip: getClientIP(req)
    });

    res.json({
      success: true,
      message: 'Token refreshed successfully',
      token: newToken,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role
      }
    });
  } catch (error) {
    logger.error('Token refresh error', {
      error: (error as Error).message,
      stack: (error as Error).stack,
      ip: getClientIP(req)
    });

    res.status(500).json({
      success: false,
      message: config.NODE_ENV === 'production' ? 'Token refresh failed' : (error as Error).message
    });
  }
});

/**
 * GET /api/auth/status
 * Check authentication status
 */
router.get('/status', async (req: Request, res: Response) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.json({
        success: true,
        authenticated: false,
        message: 'No token provided'
      });
    }

    // Verify token
    const payload = await jwtAuthService.verifyToken(token);
    
    if (!payload) {
      return res.json({
        success: true,
        authenticated: false,
        message: 'Invalid or expired token'
      });
    }

    // Get user data
    const user = await jwtAuthService.getUserById(payload.userId);
    
    if (!user) {
      return res.json({
        success: true,
        authenticated: false,
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      authenticated: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role
      },
      tokenValid: true
    });
  } catch (error) {
    logger.error('Auth status check error', {
      error: (error as Error).message,
      stack: (error as Error).stack,
      ip: getClientIP(req)
    });

    res.json({
      success: true,
      authenticated: false,
      message: 'Authentication check failed'
    });
  }
});

/**
 * GET /api/auth/profile
 * Get current user profile
 */
router.get('/profile', async (req: Request, res: Response) => {
  try {
    const token = extractToken(req);
    
    if (!token) {
      return res.status(401).json({
        success: false,
        message: 'No token provided'
      });
    }

    // Verify token
    const payload = await jwtAuthService.verifyToken(token);
    
    if (!payload) {
      return res.status(401).json({
        success: false,
        message: 'Invalid or expired token'
      });
    }

    // Get user data
    const user = await jwtAuthService.getUserById(payload.userId);
    
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    res.json({
      success: true,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        fullName: user.fullName,
        role: user.role,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    logger.error('Profile fetch error', {
      error: (error as Error).message,
      stack: (error as Error).stack,
      ip: getClientIP(req)
    });

    res.status(500).json({
      success: false,
      message: config.NODE_ENV === 'production' ? 'Profile fetch failed' : (error as Error).message
    });
  }
});

export default router;