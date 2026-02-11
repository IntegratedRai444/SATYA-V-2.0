import { Router, Request, Response } from 'express';
import { validationResult } from 'express-validator';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';
import * as bcrypt from 'bcryptjs';
import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import crypto from 'crypto';

// Define interfaces for better type safety
interface SupabaseUser {
  id: string;
  email?: string;
  user_metadata?: Record<string, unknown>;
  email_confirmed_at?: string;
  created_at?: string;
}

// Supabase Auth Response Types
interface SupabaseAuthResponse {
  user: SupabaseUser;
  session?: {
    access_token: string;
    refresh_token: string;
    expires_in?: number;
  };
}

interface SupabaseSessionResponse {
  session?: {
    access_token: string;
    refresh_token: string;
    expires_in?: number;
  };
}

const router = Router();

// Helper functions for account security
async function checkAccountLockout(email: string): Promise<boolean> {
  try {
    const { data: user } = await supabase
      .from('users')
      .select('failed_login_attempts, locked_until')
      .eq('email', email)
      .single();

    if (!user) return false;

    // Check if account is currently locked
    if (user.locked_until && new Date() < new Date(user.locked_until)) {
      return true;
    }

    // Lock account after 5 failed attempts
    if (user.failed_login_attempts >= 5) {
      const lockoutUntil = new Date(Date.now() + 30 * 60 * 1000); // 30 minutes
      await supabase
        .from('users')
        .update({ locked_until: lockoutUntil.toISOString() })
        .eq('email', email);
      return true;
    }

    return false;
  } catch (error) {
    logger.error('Error checking account lockout:', error);
    return false;
  }
}

async function trackFailedLogin(email: string, ipAddress: string): Promise<void> {
  try {
    // Get current failed attempts count
    const { data: user } = await supabase
      .from('users')
      .select('failed_login_attempts')
      .eq('email', email)
      .single();

    const currentAttempts = user?.failed_login_attempts || 0;

    // Increment failed login attempts
    await supabase
      .from('users')
      .update({ 
        failed_login_attempts: currentAttempts + 1,
        last_failed_login: new Date().toISOString()
      })
      .eq('email', email);

    // Log the failed attempt
    await supabase
      .from('login_attempts')
      .insert({
        email,
        ip_address: ipAddress,
        user_agent: 'Unknown', // Would be passed from request in real implementation
        success: false,
        created_at: new Date().toISOString()
      });
  } catch (error) {
    logger.error('Error tracking failed login:', error);
  }
}

async function resetFailedLoginAttempts(email: string): Promise<void> {
  try {
    await supabase
      .from('users')
      .update({ 
        failed_login_attempts: 0,
        locked_until: null
      })
      .eq('email', email);
  } catch (error) {
    logger.error('Error resetting failed login attempts:', error);
  }
}

async function logLoginAttempt(email: string, ipAddress: string, status: 'success' | 'failed'): Promise<void> {
  try {
    await supabase
      .from('login_attempts')
      .insert({
        email,
        ip_address: ipAddress,
        user_agent: 'Unknown', // Would be passed from request in real implementation
        success: status === 'success',
        created_at: new Date().toISOString()
      });
  } catch (error) {
    logger.error('Error logging login attempt:', error);
  }
}

// Helper functions for auth sessions
async function createAuthSession(userId: string, refreshToken: string, ipAddress: string, userAgent: string): Promise<void> {
  try {
    // Hash the refresh token
    const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex');
    
    // Set expiration to 7 days from now
    const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString();
    
    await supabase
      .from('auth_sessions')
      .insert({
        user_id: userId,
        refresh_token_hash: refreshTokenHash,
        ip_address: ipAddress,
        user_agent: userAgent,
        expires_at: expiresAt,
        created_at: new Date().toISOString()
      });
  } catch (error) {
    logger.error('Error creating auth session:', error);
  }
}

async function revokeAuthSession(refreshToken: string): Promise<void> {
  try {
    // Hash the refresh token to find the session
    const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex');
    
    await supabase
      .from('auth_sessions')
      .update({ revoked_at: new Date().toISOString() })
      .eq('refresh_token_hash', refreshTokenHash)
      .is('revoked_at', null);
  } catch (error) {
    logger.error('Error revoking auth session:', error);
  }
}

async function validateAuthSession(refreshToken: string): Promise<{ userId: string; sessionId: string } | null> {
  try {
    // Hash the refresh token to find the session
    const refreshTokenHash = crypto.createHash('sha256').update(refreshToken).digest('hex');
    
    const { data: session } = await supabase
      .from('auth_sessions')
      .select('id, user_id, expires_at, revoked_at')
      .eq('refresh_token_hash', refreshTokenHash)
      .is('revoked_at', null)
      .single();
    
    if (!session) {
      return null;
    }
    
    // Check if session has expired
    if (new Date() > new Date(session.expires_at)) {
      // Mark as revoked
      await supabase
        .from('auth_sessions')
        .update({ revoked_at: new Date().toISOString() })
        .eq('id', session.id);
      return null;
    }
    
    return {
      userId: session.user_id,
      sessionId: session.id
    };
  } catch (error) {
    logger.error('Error validating auth session:', error);
    return null;
  }
}

// Rate limiting for auth endpoints
const authRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts per window
  message: 'Too many authentication attempts. Please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: false,
});

// Password validation schema
const passwordSchema = z.string()
  .min(8, 'Password must be at least 8 characters long')
  .max(128, 'Password must be less than 128 characters')
  .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character');

// Registration schema
const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: passwordSchema,
  full_name: z.string().min(2, 'Full name must be at least 2 characters').max(100, 'Full name must be less than 100 characters'),
  username: z.string().min(3, 'Username must be at least 3 characters').max(30, 'Username must be less than 30 characters').regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, and underscores'),
});

// Login schema
const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
});

// Forgot password schema
const forgotPasswordSchema = z.object({
  email: z.string().email('Invalid email address'),
});

// Reset password schema
const resetPasswordSchema = z.object({
  token: z.string().uuid('Invalid reset token'),
  password: passwordSchema,
});

// POST /auth/register - User Registration
router.post('/register', authRateLimiter, async (req: Request, res: Response) => {
  try {
    // Validate request body
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: errors.array()
      });
    }

    const { email, password, full_name, username } = req.body;

    // Validate with schema
    const schemaValidation = registerSchema.safeParse({ email, password, full_name, username });
    if (!schemaValidation.success) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: schemaValidation.error.issues[0].message,
        details: schemaValidation.error.issues
      });
    }

    // Check if user already exists
    const { data: existingUser } = await supabase
      .from('users')
      .select('email')
      .eq('email', email)
      .single();

    if (existingUser) {
      return res.status(409).json({
        success: false,
        error: 'USER_EXISTS',
        message: 'User with this email already exists'
      });
    }

    // Hash password
    const saltRounds = 12;
    const hashedPassword = await new Promise<string>((resolve, reject) => {
      (bcrypt as any).hash(password, saltRounds, (err: any, hash: any) => {
        if (err) reject(err);
        else resolve(hash as string);
      });
    });

    // Create user in Supabase
    const { data: user, error } = await supabase.auth.admin.createUser({
      email,
      password: hashedPassword,
      email_confirm: true,
      user_metadata: {
        username,
        full_name,
        role: 'user',
        created_at: new Date().toISOString()
      }
    });

    if (error) {
      logger.error('User creation failed:', { error, email });
      return res.status(500).json({
        success: false,
        error: 'REGISTRATION_FAILED',
        message: 'Failed to create user account'
      });
    }

    // Create user profile record
    const { error: profileError } = await supabase
      .from('users')
      .insert({
        id: user.user.id,
        email,
        username,
        full_name,
        role: 'user',
        is_active: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });

    if (profileError) {
      logger.error('User profile creation failed:', { profileError, userId: user.user.id });
      // Don't fail the request, user was created successfully
    }

    logger.info('User registered successfully:', { userId: user.user.id, email });

    res.status(201).json({
      success: true,
      message: 'User registered successfully. Please check your email for verification.',
      data: {
        user: {
          id: user.user.id,
          email,
          username,
          full_name,
          role: 'user'
        }
      }
    });
  } catch (error) {
    logger.error('Registration error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during registration'
    });
  }
});

// POST /auth/login - User Login
router.post('/login', authRateLimiter, async (req: Request, res: Response) => {
  try {
    // Validate request body
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: errors.array()
      });
    }

    const { email, password } = req.body;

    // Validate with schema
    const schemaValidation = loginSchema.safeParse({ email, password });
    if (!schemaValidation.success) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: schemaValidation.error.issues[0].message,
        details: schemaValidation.error.issues
      });
    }

    // Check account lockout status before authentication
    const isLocked = await checkAccountLockout(email);
    if (isLocked) {
      return res.status(423).json({
        success: false,
        error: 'ACCOUNT_LOCKED',
        message: 'Account has been locked due to too many failed login attempts. Please try again later.'
      });
    }

    // Authenticate with Supabase
    const { data: user, error } = await supabase.auth.signInWithPassword({
      email,
      password
    });

    if (error) {
      // Track failed login attempt
      await trackFailedLogin(email, req.ip || 'unknown');
      
      // Check if account should be locked
      const shouldLock = await checkAccountLockout(email);
      if (shouldLock) {
        return res.status(423).json({
          success: false,
          error: 'ACCOUNT_LOCKED',
          message: 'Account has been locked due to too many failed login attempts. Please try again later.'
        });
      }
      
      logger.warn('Login failed:', { error, email });
      return res.status(401).json({
        success: false,
        error: 'INVALID_CREDENTIALS',
        message: 'Invalid email or password'
      });
    }

    // Check if email is verified
    if (!(user as SupabaseAuthResponse).user.email_confirmed_at) {
      return res.status(403).json({
        success: false,
        error: 'EMAIL_NOT_VERIFIED',
        message: 'Please verify your email address before logging in'
      });
    }

    // Reset failed login attempts on successful login
    await resetFailedLoginAttempts(email);

    // Update last login
    await supabase
      .from('users')
      .update({ last_login: new Date().toISOString() })
      .eq('id', (user as SupabaseAuthResponse).user.id);

    // Log successful login attempt
    await logLoginAttempt(email, req.ip || 'unknown', 'success');

    logger.info('User logged in successfully:', { userId: (user as SupabaseAuthResponse).user.id, email });

    // Create auth session with refresh token
    const refreshToken = (user as SupabaseAuthResponse).session?.refresh_token || '';
    if (refreshToken) {
      await createAuthSession(
        (user as SupabaseAuthResponse).user.id,
        refreshToken,
        req.ip || 'unknown',
        req.get('User-Agent') || 'unknown'
      );
    }

    // Create session and set HTTP-only cookie
    const { data: session } = await supabase.auth.setSession({
      access_token: user.session?.access_token,
      refresh_token: user.session?.refresh_token
    });

    // Set HTTP-only secure cookie for access token
    res.cookie('sb-access-token', (session as SupabaseSessionResponse).session?.access_token || '', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 15 * 60 * 1000, // 15 minutes
      path: '/'
    });

    // Set HTTP-only secure cookie for refresh token
    res.cookie('sb-refresh-token', (session as SupabaseSessionResponse).session?.refresh_token || '', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
      path: '/'
    });

    res.json({
      success: true,
      message: 'Login successful',
      data: {
        user: {
          id: (user as SupabaseAuthResponse).user.id,
          email: (user as SupabaseAuthResponse).user.email,
          role: (user as SupabaseAuthResponse).user.user_metadata?.role || 'user',
          email_verified: !!(user as SupabaseAuthResponse).user.email_confirmed_at,
          created_at: (user as SupabaseAuthResponse).user.created_at
        }
      }
    });
  } catch (error) {
    logger.error('Login error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during login'
    });
  }
});

// POST /auth/logout - User Logout
router.post('/logout', async (req: Request, res: Response) => {
  try {
    // Get refresh token from cookie
    const refreshToken = req.cookies?.['sb-refresh-token'];
    
    // Revoke auth session if refresh token exists
    if (refreshToken) {
      await revokeAuthSession(refreshToken);
    }

    // Clear cookies
    res.clearCookie('sb-access-token');
    res.clearCookie('sb-refresh-token');

    // Sign out from Supabase
    await supabase.auth.signOut();

    logger.info('User logged out successfully');

    res.json({
      success: true,
      message: 'Logged out successfully'
    });
  } catch (error) {
    logger.error('Logout error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during logout'
    });
  }
});

// POST /auth/refresh - Refresh Access Token
router.post('/refresh', 
  rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 refresh attempts per window
    message: 'Too many refresh attempts. Please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
  }),
  async (req: Request, res: Response) => {
  try {
    // Get refresh token from cookie
    const refreshToken = req.cookies?.['sb-refresh-token'];
    if (!refreshToken) {
      return res.status(401).json({
        success: false,
        error: 'REFRESH_TOKEN_REQUIRED',
        message: 'Refresh token is required'
      });
    }

    // Validate auth session before refresh
    const sessionValidation = await validateAuthSession(refreshToken);
    if (!sessionValidation) {
      return res.status(401).json({
        success: false,
        error: 'INVALID_SESSION',
        message: 'Invalid or expired session'
      });
    }

    // Refresh session
    const { data, error } = await supabase.auth.refreshSession(refreshToken);
    if (error) {
      logger.warn('Token refresh failed:', { error });
      return res.status(401).json({
        success: false,
        error: 'REFRESH_FAILED',
        message: 'Failed to refresh session'
      });
    }

    // Update last login
    if (data.user?.id) {
      await supabase
        .from('users')
        .update({ last_login: new Date().toISOString() })
        .eq('id', data.user.id);
    }

    logger.info('Token refreshed successfully', { userId: data.user?.id });
    res.json({
      success: true,
      data: {
        accessToken: data.session?.access_token,
        refreshToken: data.session?.refresh_token,
        user: {
          id: data.user?.id,
          email: data.user?.email,
          user_metadata: data.user?.user_metadata
        }
      }
    });
  } catch (error) {
    logger.error('Refresh error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during token refresh'
    });
  }
});

// POST /auth/forgot-password - Forgot Password
router.post('/forgot-password', authRateLimiter, async (req: Request, res: Response) => {
  try {
    // Validate request body
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: errors.array()
      });
    }

    const { email } = req.body;

    // Validate with schema
    const schemaValidation = forgotPasswordSchema.safeParse({ email });
    if (!schemaValidation.success) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: schemaValidation.error.issues[0].message,
        details: schemaValidation.error.issues
      });
    }

    // Check if user exists
    const { data: user } = await supabase
      .from('users')
      .select('email')
      .eq('email', email)
      .single();

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'USER_NOT_FOUND',
        message: 'No account found with this email address'
      });
    }

    // Generate reset token
    const resetToken = uuidv4();
    const resetTokenExpiry = new Date(Date.now() + 60 * 60 * 1000); // 1 hour

    // Store reset token in database (you would typically email this)
    const tokenHash = crypto.createHash('sha256').update(resetToken).digest('hex');
    const expiresAt = new Date(Date.now() + 60 * 60 * 1000); // 1 hour from now
    
    await supabase
      .from('password_reset_tokens')
      .insert({
        token_hash: tokenHash,
        email,
        expires_at: expiresAt.toISOString(),
        created_at: new Date().toISOString()
      });

    logger.info('Password reset requested:', { email });

    res.json({
      success: true,
      message: 'Password reset instructions have been sent to your email',
      // In production, you would send an email here
      data: {
        reset_token: resetToken, // Only for development/testing
        expires_at: resetTokenExpiry.toISOString()
      }
    });
  } catch (error) {
    logger.error('Forgot password error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during password reset'
    });
  }
});

// POST /auth/reset-password - Reset Password
router.post('/reset-password', authRateLimiter, async (req: Request, res: Response) => {
  try {
    // Validate request body
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: errors.array()
      });
    }

    const { token, password } = req.body;

    // Validate with schema
    const schemaValidation = resetPasswordSchema.safeParse({ token, password });
    if (!schemaValidation.success) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: schemaValidation.error.issues[0].message,
        details: schemaValidation.error.issues
      });
    }

    // Verify reset token
    const tokenHash = crypto.createHash('sha256').update(token).digest('hex');
    
    const { data: resetRecord, error: tokenError } = await supabase
      .from('password_reset_tokens')
      .select('*')
      .eq('token_hash', tokenHash)
      .is('used_at', null)
      .single();

    if (tokenError || !resetRecord) {
      return res.status(400).json({
        success: false,
        error: 'INVALID_TOKEN',
        message: 'Invalid or expired reset token'
      });
    }

    // Check if token has expired
    if (new Date() > new Date(resetRecord.expires_at)) {
      return res.status(400).json({
        success: false,
        error: 'TOKEN_EXPIRED',
        message: 'Reset token has expired'
      });
    }

    // Get user from reset record
    const { data: user } = await supabase
      .from('users')
      .select('id')
      .eq('email', resetRecord.email)
      .single();

    if (!user) {
      return res.status(404).json({
        success: false,
        error: 'USER_NOT_FOUND',
        message: 'User not found'
      });
    }

    // Hash new password
    const saltRounds = 12;
    const hashedPassword = await new Promise<string>((resolve, reject) => {
      (bcrypt as any).hash(password, saltRounds, (err: any, hash: any) => {
        if (err) reject(err);
        else resolve(hash as string);
      });
    });

    // Update user password in Supabase auth
    const { error: updateError } = await supabase.auth.admin.updateUserById(user.id, {
      password: hashedPassword
    });

    if (updateError) {
      logger.error('Password update failed:', { updateError, userId: user.id });
      return res.status(500).json({
        success: false,
        error: 'PASSWORD_UPDATE_FAILED',
        message: 'Failed to update password'
      });
    }

    // Mark reset token as used
    await supabase
      .from('password_reset_tokens')
      .update({ used_at: new Date().toISOString() })
      .eq('token_hash', tokenHash);

    logger.info('Password reset successful:', { userId: user.id, email: resetRecord.email });

    res.json({
      success: true,
      message: 'Password reset successfully'
    });
  } catch (error) {
    logger.error('Reset password error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during password reset'
    });
  }
});

// GET /auth/me - Get Current User
router.get('/me', async (req: Request, res: Response) => {
  try {
    // Get token from cookie
    const accessToken = req.cookies?.['sb-access-token'];
    if (!accessToken) {
      return res.status(401).json({
        success: false,
        error: 'AUTH_REQUIRED',
        message: 'Authentication required'
      });
    }

    // Verify token with Supabase
    const { data: user, error } = await supabase.auth.getUser(accessToken);

    if (error) {
      return res.status(401).json({
        success: false,
        error: 'INVALID_TOKEN',
        message: 'Invalid or expired token'
      });
    }

    res.json({
      success: true,
      data: {
        user: {
          id: user.user?.id || '',
          email: user.user?.email || '',
          role: (user.user?.user_metadata?.role as string) || 'user',
          email_verified: !!user.user?.email_confirmed_at,
          created_at: user.user?.created_at
        }
      }
    });
  } catch (error) {
    logger.error('Get user error:', error);
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error'
    });
  }
});

export { router as authRouter };
