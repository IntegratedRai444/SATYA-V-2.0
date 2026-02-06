import { Router, Request, Response } from 'express';
import { validationResult } from 'express-validator';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import rateLimit from 'express-rate-limit';
import * as bcrypt from 'bcryptjs';
import { z } from 'zod';

// Define interfaces for better type safety
interface SupabaseSession {
  access_token: string;
  refresh_token: string;
  user: {
    id: string;
    email?: string;
    user_metadata?: Record<string, unknown>;
  };
}

const router = Router();

// Rate limiting for auth endpoints
const authRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts per window
  message: 'Too many authentication attempts. Please try again later.',
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: false,
});

// Enhanced password validation with security requirements
const passwordSchema = z.string()
  .min(12, 'Password must be at least 12 characters long')
  .max(128, 'Password must be less than 128 characters')
  .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character')
  .refine((password) => {
    // Check for common weak patterns
    const weakPatterns = [
      /password/i,
      /123456/i,
      /qwerty/i,
      /admin/i,
      /letmein/i
    ];
    return !weakPatterns.some(pattern => pattern.test(password));
  }, 'Password contains common weak patterns');

// Registration schema with enhanced validation
const registerSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: passwordSchema,
  full_name: z.string().min(2, 'Full name must be at least 2 characters').max(100, 'Full name must be less than 100 characters'),
  username: z.string()
    .min(3, 'Username must be at least 3 characters')
    .max(30, 'Username must be less than 30 characters')
    .regex(/^[a-zA-Z0-9_]+$/, 'Username can only contain letters, numbers, and underscores')
    .refine((username) => {
      // Check for common weak patterns
      const weakPatterns = [
        /password/i,
        /123456/i,
        /qwerty/i,
        /admin/i,
        /letmein/i
      ];
      return !weakPatterns.some(pattern => pattern.test(username));
    }, 'Username contains common weak patterns')
});

// Login schema with enhanced security
const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required')
});

// Track failed login attempts for account lockout
const failedAttempts = new Map<string, { count: number; lastAttempt: number; lockedUntil: number }>();

// Enhanced rate limiting with account lockout
const createAccountLockoutLimiter = () => {
  return rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 3, // Reduced max attempts for security
    message: 'Account temporarily locked due to too many failed attempts. Please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req: Request) => {
      const email = req.body?.email || 'unknown';
      const clientIP = req.ip || 'unknown';
      return `auth-lockout:${email}:${clientIP}`;
    },
    handler: (req: Request, res: Response) => {
      const email = req.body?.email;
      const clientIP = req.ip || 'unknown';
      const key = `auth-lockout:${email}:${clientIP}`;
      const attempts = failedAttempts.get(key);
      
      if (attempts && attempts.count >= 3) {
        const lockoutDuration = Math.min(30 * 60 * 1000, attempts.count * 5 * 60 * 1000); // Progressive lockout
        attempts.lockedUntil = Date.now() + lockoutDuration;
        
        logger.warn('Account locked due to failed attempts', {
          email,
          ip: clientIP,
          attempts: attempts.count,
          lockoutDuration: lockoutDuration / 1000 / 60
        });
        
        return res.status(429).json({
          success: false,
          error: 'ACCOUNT_LOCKED',
          message: `Account locked for ${Math.ceil(lockoutDuration / 60000)} minutes due to too many failed login attempts.`,
          retryAfter: Math.ceil(lockoutDuration / 1000)
        });
      }
    }
  });
};

// POST /auth/register - Enhanced User Registration
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

    // Enhanced validation with security checks
    const schemaValidation = registerSchema.safeParse({ email, password, full_name, username });
    if (!schemaValidation.success) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: schemaValidation.error.issues[0].message,
        details: schemaValidation.error.issues
      });
    }

    // Check for disposable email domains
    const disposableDomains = ['tempmail.org', '10minutemail.com', 'guerrillamail.com', 'mailinator.org'];
    const domain = email.split('@')[1]?.toLowerCase();
    if (domain && disposableDomains.includes(domain)) {
      logger.warn('Registration attempt with disposable email', { email, domain });
      return res.status(400).json({
        success: false,
        error: 'DISPOSABLE_EMAIL',
        message: 'Disposable email addresses are not allowed'
      });
    }

    // Check if user already exists
    const { data: existingUser } = await supabase
      .from('users')
      .select('email')
      .eq('email', email)
      .single();

    if (existingUser) {
      // Use generic error message to prevent email enumeration
      return res.status(409).json({
        success: false,
        error: 'REGISTRATION_FAILED',
        message: 'Registration failed'
      });
    }

    // Hash password with enhanced security
    const saltRounds = 12;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Create user in Supabase with enhanced metadata
    const { data: user, error } = await supabase.auth.admin.createUser({
      email,
      password: hashedPassword,
      email_confirm: true,
      user_metadata: {
        username,
        full_name,
        role: 'user',
        created_at: new Date().toISOString(),
        registration_ip: req.ip || 'unknown',
        user_agent: req.headers['user-agent'] || 'unknown'
      }
    });

    if (error) {
      logger.error('User creation failed:', { error: error.message, email });
      return res.status(500).json({
        success: false,
        error: 'REGISTRATION_FAILED',
        message: 'Registration failed'
      });
    }

    // Create user profile record with security tracking
    const { error: profileError } = await supabase
      .from('users')
      .insert({
        id: user.user?.id || '',
        email,
        username,
        full_name,
        role: 'user',
        is_active: true,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        last_login: null,
        login_count: 0,
        security_flags: {
          email_verified: false,
          mfa_enabled: false,
          password_strength: 'strong'
        }
      });

    if (profileError) {
      logger.error('User profile creation failed:', { profileError, userId: user.user?.id || '' });
    }

    logger.info('User registered successfully:', { userId: user.user?.id || '', email });

    // Return success without exposing sensitive data
    res.status(201).json({
      success: true,
      message: 'Registration successful. Please check your email for verification.',
      data: {
        user: {
          id: user.user?.id || '',
          email,
          username,
          full_name,
          role: 'user'
        }
      }
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Registration error:', { error: errorMessage });
    res.status(500).json({
      success: false,
      error: 'INTERNAL_ERROR',
      message: 'Internal server error during registration'
    });
  }
});

// POST /auth/login - Enhanced User Login with Account Lockout
router.post('/login', createAccountLockoutLimiter(), async (req: Request, res: Response) => {
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

    // Enhanced validation
    const schemaValidation = loginSchema.safeParse({ email, password });
    if (!schemaValidation.success) {
      return res.status(400).json({
        success: false,
        error: 'VALIDATION_ERROR',
        message: schemaValidation.error.issues[0].message,
        details: schemaValidation.error.issues
      });
    }

    // Check account lockout status
    const ip = req.ip || 'unknown';
    const key = `auth-lockout:${email}:${ip}`;
    const attempts = failedAttempts.get(key) || { count: 0, lastAttempt: 0, lockedUntil: 0 };

    if (attempts.lockedUntil && Date.now() < attempts.lockedUntil) {
      logger.warn('Login attempt on locked account', { email, ip, lockedUntil: attempts.lockedUntil });
      return res.status(423).json({
        success: false,
        error: 'ACCOUNT_LOCKED',
        message: `Account is locked. Please try again in ${Math.ceil((attempts.lockedUntil - Date.now()) / 60000)} minutes.`,
        retryAfter: Math.ceil((attempts.lockedUntil - Date.now()) / 60000)
      });
    }

    // Authenticate with Supabase
    const { data: user, error } = await supabase.auth.signInWithPassword({
      email,
      password
    });

    if (error) {
      // Track failed attempt
      attempts.count += 1;
      attempts.lastAttempt = Date.now();
      failedAttempts.set(key, attempts);

      logger.warn('Login failed:', { error: error.message, email, ip, attempts: attempts.count });
      
      // Generic error message to prevent user enumeration
      return res.status(401).json({
        success: false,
        error: 'INVALID_CREDENTIALS',
        message: 'Invalid email or password'
      });
    }

    // Check if email is verified
    if (!user.user?.email_confirmed_at) {
      return res.status(403).json({
        success: false,
        error: 'EMAIL_NOT_VERIFIED',
        message: 'Please verify your email address before logging in'
      });
    }

    // Successful login - reset failed attempts
    if (attempts.count > 0) {
      logger.info('Successful login after failed attempts', { email, ip, previousAttempts: attempts.count });
      failedAttempts.delete(key);
    }

    // Update last login and security tracking
    const currentLoginCount = await supabase
      .from('users')
      .select('login_count')
      .eq('id', user.user?.id || '')
      .single();
      
    await supabase
      .from('users')
      .update({ 
        last_login: new Date().toISOString(),
        login_count: (currentLoginCount.data?.login_count || 0) + 1,
        updated_at: new Date().toISOString(),
        security_flags: {
          email_verified: !!user.user?.email_confirmed_at,
          last_login_ip: req.ip || 'unknown',
          last_login_user_agent: req.headers['user-agent'] || 'unknown'
        }
      })
      .eq('id', user.user?.id || '');

    logger.info('User logged in successfully:', { userId: user.user?.id || '', email });

    // Create session with enhanced security  
    const { data: session } = await supabase.auth.setSession({
      access_token: user.session.access_token,
      refresh_token: user.session.refresh_token
    });

    // Set HTTP-only secure cookies with enhanced security
    const cookieOptions: {
      httpOnly: boolean;
      secure: boolean;
      sameSite: 'strict' | 'lax' | 'none';
      maxAge: number;
      path: string;
      domain?: string;
    } = {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 15 * 60 * 1000, // 15 minutes
      path: '/',
      // Add additional security attributes
      domain: process.env.NODE_ENV === 'production' ? '.satyaai.com' : undefined
    };

    res.cookie('sb-access-token', (session as unknown as SupabaseSession)?.access_token || '', cookieOptions);
    res.cookie('sb-refresh-token', (session as unknown as SupabaseSession)?.refresh_token || '', {
      ...cookieOptions,
      maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
    });

    res.json({
      success: true,
      message: 'Login successful',
      data: {
        user: {
          id: user.user?.id || '',
          email,
          username: user.user?.user_metadata?.username || '',
          full_name: user.user?.user_metadata?.full_name || '',
          role: user.user?.user_metadata?.role || 'user'
        }
      }
    });
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Login error:', { error: errorMessage });
    res.status(500).json({
      success: false,
      error: 'LOGIN_ERROR',
      message: 'Internal server error during login'
    });
  }
});

// POST /auth/logout - Enhanced Logout with Session Invalidation
router.post('/logout', async (req: Request, res: Response) => {
  try {
    // ...
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Logout error:', { error: errorMessage });
    res.status(500).json({
      success: false,
      error: 'LOGOUT_ERROR',
      message: 'Internal server error during logout'
    });
  }
});

// GET /auth/me - Enhanced Current User with Security Headers
router.get('/me', async (req: Request, res: Response) => {
  try {
    // ...
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error('Get user error:', { error: errorMessage });
    res.status(500).json({
      success: false,
      error: 'GET_USER_ERROR',
      message: 'Internal server error'
    });
  }
});

export { router as authRouter };
