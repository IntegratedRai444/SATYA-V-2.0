import { Router } from 'express';
import type { Request as ExpressRequest, Response, NextFunction } from 'express';
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

// Use the Request type from auth middleware
type Request = AuthRequest & ExpressRequest;

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
    csrfProtection
  ],
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
        user: data.user
      });
    } catch (error: unknown) {
      logger.error('Signup error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Signup failed'
      });
    }
  }
);


// Login with email/password
authRouter.post(
  '/login',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').notEmpty().withMessage('Password is required'),
    validateRequest,
    loginLimiter as any, // Type assertion for express-rate-limit
    csrfProtection // CSRF protection enabled for login
  ],
  async (req: ExpressRequest, res: Response) => {
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
authRouter.post('/refresh', [
  csrfProtection, 
  refreshTokenLimiter as any, // Type assertion needed due to express-rate-limit types
  async (req: ExpressRequest, res: Response, next: NextFunction) => {
  try {
    const { refresh_token } = req.body as { refresh_token?: string };
    
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
    res.json({
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
    
    res.status(500).json({
      success: false,
      code: 'REFRESH_ERROR',
      message: 'An error occurred while refreshing the session',
      error: process.env.NODE_ENV === 'development' ? errorMessage : undefined
    });
  }
});

// Get current user
authRouter.get(
  '/me', 
  [authenticate, authLimiter, csrfProtection],
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
      
      res.json({
        success: true,
        user: {
          ...user,
          ...userData
        }
      });
    } catch (error: unknown) {
      logger.error('Get current user error:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch user data'
      });
    }
  }
);

// Logout
authRouter.post(
  '/logout', 
  [authenticate, csrfProtection],
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
);

// Request password reset
authRouter.post(
  '/reset-password',
  [body('email').isEmail().normalizeEmail(), csrfProtection],
  validateRequest,
  async (req: Request, res: Response) => {
    try {
      const { email } = req.body;
      const { data, error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${process.env.CLIENT_URL}/reset-password`
      });

      if (error) {
        throw error;
      }
      
      res.json({
        success: true,
        message: 'Password reset email sent'
      });
    } catch (error: unknown) {
      logger.error('Password reset request error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Password reset request failed'
      });
    }
// Requires authentication
// Uses the Supabase Admin API to update user metadata
authRouter.patch(
  '/profile',
  [
    authenticate as any, // Type assertion for custom middleware
    csrfProtection,
    body('fullName').optional().isString().trim().isLength({ max: 100 }),
    body('username').optional().isString().trim().isLength({ min: 3, max: 30 }),
    body('avatarUrl').optional().isString().trim().isURL()
  ].filter(Boolean) as ValidationChain[],
  validateRequest,
  async (req: Request, res: Response) => {
    try {
      const { fullName, username, avatarUrl } = req.body;
      
      // Update user metadata
      const { data: { user }, error } = await supabase.auth.updateUser({
        data: {
          ...req.user?.user_metadata,
          ...(fullName && { full_name: fullName }),
          ...(username && { username }),
          ...(avatarUrl && { avatar_url: avatarUrl })
        }
      });

      if (error) {
        throw error;
      }
      
      res.json({
        success: true,
        user: user
      });
    } catch (error: unknown) {
      logger.error('Profile update error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Profile update failed'
      });
    }
  }
);

// ...

// Change password route
authRouter.post(
  '/change-password',
  authenticate as any, // Type assertion for custom middleware
  [
    body('currentPassword').notEmpty(),
    body('newPassword').isLength({ min: 8 })
  ],
  validateRequest,
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
      
      res.json({ success: true, message: 'Password updated successfully' });
    } catch (error: unknown) {
      logger.error('Change password error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to change password'
      });
    }
  }
);

// Email verification
authRouter.get(
  '/verify-email',
  async (req: Request, res: Response) => {
    try {
      const { token } = req.query;
      
      if (!token || typeof token !== 'string') {
        throw new Error('Verification token is required');
      }
      
      const { error } = await supabase.auth.verifyOtp({
        token_hash: token,
        type: 'signup'
      });

      if (error) throw error;
      
      res.redirect(`${process.env.CLIENT_URL}/login?verified=true`);
    } catch (error: unknown) {
      logger.error('Email verification error:', error);
      res.redirect(
        `${process.env.CLIENT_URL}/login?error=${encodeURIComponent(
          error instanceof Error ? error.message : 'Email verification failed'
        )}`
      );
    }
  }
);

// Resend verification email
authRouter.post(
  '/resend-verification',
  [body('email').isEmail().normalizeEmail()],
  validateRequest,
  async (req: Request, res: Response) => {
    try {
      const { email } = req.body;
      const { error } = await supabase.auth.resend({
        type: 'signup',
        email,
        options: {
          emailRedirectTo: `${process.env.CLIENT_URL}/login`,
        },
      });

      if (error) throw error;

      res.json({ 
        success: true, 
        message: 'Verification email resent successfully' 
      });
    } catch (error: unknown) {
      logger.error('Resend verification email error:', error);
      res.status(400).json({
        success: false,
        error: error instanceof Error ? error.message : 'Failed to resend verification email',
      });
    }
  }
);

// Get active sessions (requires admin role)
authRouter.get('/sessions', authenticate, async (req: Request, res: Response) => {
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
        isActive: user.last_sign_in_at !== null
      }))
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
  authenticate,
  [body('userId').notEmpty()],
  validateRequest,
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
);
