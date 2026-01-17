import { Router, type Request as ExpressRequest, type Response, type NextFunction, type RequestHandler } from 'express';
import { body, type ValidationChain } from 'express-validator';
import { validateRequest } from '../middleware/validate-request';
import { logger } from '../config/logger';
import { 
  authenticate, 
  loginLimiter, 
  csrfProtection,
  generateCsrfToken,
  apiLimiter as authLimiter,
  refreshTokenLimiter,
  type Request as AuthRequest
} from '../middleware/auth.middleware';
import { supabase } from '../config/supabase';

import { User as SupabaseUser } from '@supabase/supabase-js';

// Use the same user type as defined in auth.middleware.ts
type AuthUser = {
  id: string;
  email: string;
  role: string;
  email_verified: boolean;
  user_metadata?: Record<string, any>;
};

// Extend the Express Request type to include our auth user
declare module 'express-serve-static-core' {
  interface Request {
    user?: AuthUser;
  }
}

type Request = Express.Request;

export const authRouter = Router();

// CSRF Token Endpoint
authRouter.get('/csrf-token', (req: ExpressRequest, res: Response) => {
  try {
    const { token, cookie } = generateCsrfToken();
    res.setHeader('Set-Cookie', cookie);
    res.json({ token });
  } catch (error) {
    logger.error('CSRF token generation failed:', error);
    res.status(500).json({
      success: false,
      code: 'CSRF_TOKEN_ERROR',
      message: 'Failed to generate CSRF token'
    });
  }
});

// Helper function to handle errors
const handleError = (error: unknown, res: Response, defaultMessage: string, code = 'AUTH_ERROR') => {
  const errorMessage = error instanceof Error ? error.message : defaultMessage;
  logger.error('Auth error:', error);
  return res.status(400).json({
    success: false,
    code,
    message: errorMessage
  });
};

// Sign up with email/password
authRouter.post(
  '/signup',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 8 })
      .withMessage('Password must be at least 8 characters long'),
    body('username').notEmpty().withMessage('Username is required'),
    body('fullName').optional(),
    validateRequest,
    authLimiter as any, // Type assertion for express-rate-limit
    csrfProtection,
    async (req: ExpressRequest, res: Response) => {
      try {
        const { email, password, username, fullName } = req.body;
        
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${process.env.CLIENT_URL}/auth/callback`
          },
        });

        if (error) {
          throw error;
        }
        
        res.status(201).json({
          success: true,
          message: 'Signup successful. Please check your email for verification.',
          user: data.user,
        });
      } catch (error: unknown) {
        logger.error('Signup error:', error);
        res.status(400).json({
          success: false,
          error: error instanceof Error ? error.message : 'Signup failed',
        });
      }
    }
  ]
);

// Alias for /signup to maintain consistency
authRouter.post(
  '/register',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 8 })
      .withMessage('Password must be at least 8 characters long'),
    body('username').notEmpty().withMessage('Username is required'),
    body('fullName').optional(),
    validateRequest,
    authLimiter as any, // Type assertion for express-rate-limit
    csrfProtection,
    async (req: ExpressRequest, res: Response) => {
      try {
        const { email, password, username, fullName } = req.body;
        
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${process.env.CLIENT_URL}/auth/callback`
          },
        });

        if (error) {
          throw error;
        }
        
        res.status(201).json({
          success: true,
          message: 'Signup successful. Please check your email for verification.',
          user: data.user,
        });
      } catch (error: unknown) {
        logger.error('Signup error:', error);
        res.status(400).json({
          success: false,
          error: error instanceof Error ? error.message : 'Signup failed',
        });
      }
    }
  ]
);


// Login with email/password
authRouter.post(
  '/login',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').notEmpty().withMessage('Password is required'),
    validateRequest,
    loginLimiter as any, // Type assertion for express-rate-limit
    csrfProtection as any // CSRF protection enabled for login
  ],
  async (req: ExpressRequest, res: Response, next: NextFunction) => {
    try {
      const { email, password } = req.body;
      
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        throw error;
      }
      
      // Check if email is verified
      if (!data.user?.email_confirmed_at) {
        // Optionally, you can resend verification email here
        // await supabase.auth.resendOtp({ email, type: 'signup' });
        
        return res.status(403).json({
          success: false,
          code: 'EMAIL_NOT_VERIFIED',
          message: 'Please verify your email address before logging in. Check your inbox for the verification link.'
        });
      }
      
      res.json({
        success: true,
        message: 'Login successful',
        user: data.user,
        session: data.session
      });
    } catch (error: unknown) {
      logger.error('Login error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Invalid email or password';
      
      // Handle specific error cases
      if (errorMessage.includes('Invalid login credentials')) {
        return res.status(401).json({
          success: false,
          code: 'INVALID_CREDENTIALS',
          message: 'Invalid email or password'
        });
      }
      
      res.status(401).json({
        success: false,
        code: 'LOGIN_FAILED',
        message: errorMessage
      });
    }
  }
);

// Refresh session
authRouter.post<{}, any, { refresh_token?: string }>(
  '/refresh',
  [
    csrfProtection as any,
    refreshTokenLimiter as any,
    async (req: ExpressRequest, res: Response) => {
      try {
        const { refresh_token } = req.body;
        
        if (!refresh_token) {
          return res.status(400).json({
            success: false,
            code: 'REFRESH_TOKEN_REQUIRED',
            message: 'Refresh token is required'
          });
        }

        const { data, error } = await supabase.auth.refreshSession({
          refresh_token
        });
        
        if (error || !data.session) {
          logger.warn('Failed to refresh session', { 
            error: error?.message || 'No session data returned',
            hasSession: !!data?.session,
            ip: req.ip
          });
          
          return res.status(401).json({
            success: false,
            code: 'INVALID_REFRESH_TOKEN',
            message: 'Invalid or expired refresh token. Please log in again.'
          });
        }
        
        // Generate new CSRF token for the session
        const { token: csrfToken, cookie: csrfCookie } = generateCsrfToken();
        
        // Set the CSRF token as HTTP-only cookie
        res.setHeader('Set-Cookie', csrfCookie);
        
        // Return the new tokens and user data
        return res.json({
          success: true,
          access_token: data.session.access_token,
          refresh_token: data.session.refresh_token,
          expires_in: data.session.expires_in,
          token_type: data.session.token_type,
          csrf_token: csrfToken,
          user: {
            id: data.user?.id,
            email: data.user?.email,
            role: data.user?.user_metadata?.role || 'user',
            email_confirmed_at: data.user?.email_confirmed_at,
            last_sign_in_at: data.user?.last_sign_in_at
          }
        });
      } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error during refresh';
        logger.error('Session refresh error:', { 
          error: errorMessage,
          stack: error instanceof Error ? error.stack : undefined,
          ip: req.ip
        });
        
        return res.status(500).json({
          success: false,
          code: 'REFRESH_ERROR',
          message: 'An error occurred while refreshing the session',
          error: process.env.NODE_ENV === 'development' ? errorMessage : undefined
        });
      }
    }
  ]
);

// Get current user
authRouter.get<{}, any, {}, {}>(
  '/me', 
  [
    authenticate as any,
    authLimiter as any,
    csrfProtection as any,
    async (req: Request, res: Response) => {
    try {
      // Get fresh user data from Supabase
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      
      if (userError || !user) {
        throw new Error('User not found');
      }
      
      // Get additional user data from your custom table if needed
      const { data: userData } = await supabase
        .from('users')
        .select('*')
        .eq('id', user.id)
        .single();
      
      return res.json({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          role: user.user_metadata?.role || 'user',
          ...(userData || {})
        }
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch user profile';
      logger.error('Profile fetch error:', { error: errorMessage });
      return res.status(500).json({
        success: false,
        code: 'PROFILE_FETCH_ERROR',
        message: 'Failed to fetch user profile'
      });
    }
  }
] as RequestHandler[]);

// Logout
authRouter.post<{}, any, {}, {}>(
  '/logout',
  [
    authenticate as any,
    csrfProtection as any,
    async (req: Request, res: Response) => {
    try {
      // Clear Supabase session
      await supabase.auth.signOut();
      
      // Clear cookies
      res.clearCookie('sb-access-token');
      
      res.json({
        success: true,
        message: 'Logged out successfully'
      });
    } catch (error: unknown) {
      logger.error('Logout error:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to log out'
      });
    }
    }
  ]
);

// ...

// Change password route
authRouter.post(
  '/change-password',
  [
    authenticate as any,
    body('currentPassword').notEmpty(),
    body('newPassword').isLength({ min: 8 }),
    validateRequest as any,
  async (req: Request, res: Response) => {
    try {
      const { currentPassword, newPassword } = req.body;
      
      // First verify current password
      const { error: verifyError } = await supabase.auth.signInWithPassword({
        email: req.user?.email || '',
        password: currentPassword
      });
      
      if (verifyError) {
        throw new Error('Current password is incorrect');
      }
      
      // Update to new password
      const { error } = await supabase.auth.updateUser({
        password: newPassword
      });

      if (error) throw error;
      
      res.json({ 
        success: true, 
        message: 'Password changed successfully' 
      });
    } catch (error: unknown) {
      logger.error('Change password error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to change password'
      });
    }
    }
  ]
);

// ...

// Get active sessions (requires admin role)
authRouter.get('/sessions', 
  authenticate as any, 
  async (req: Request, res: Response) => {
  try {
    // Note: This requires admin privileges
    const { data: { users }, error } = await supabase.auth.admin.listUsers();

    if (error) throw error;

    res.json({ 
      success: true, 
      sessions: users.map(user => ({
        id: user.id,
        email: user.email,
        lastSignIn: user.last_sign_in_at,
        createdAt: user.created_at,
        isActive: user.last_sign_in_at !== null,
      })),
    });
  } catch (error: unknown) {
    logger.error('Get active sessions error:', error);
    res.status(400).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to get active sessions'
    });
  }
});

// Revoke a specific session (admin only)
authRouter.post(
  '/sessions/revoke',
  [
    authenticate as any,
    body('userId').notEmpty(),
    validateRequest as any,
    async (req: Request, res: Response) => {
    try {
      const { userId } = req.body;
      
      // Note: This requires admin privileges
      const { error } = await supabase.auth.admin.deleteUser(userId);
      
      if (error) throw error;
      
      res.json({ success: true, message: 'Session revoked successfully' });
    } catch (error: unknown) {
      logger.error('Revoke session error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to revoke session'
      });
    }
    }
  ]
);
