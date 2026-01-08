import { Router, type Request as ExpressRequest, type Response, type NextFunction } from 'express';
import { body, type ValidationChain } from 'express-validator';
import { validateRequest } from '../middleware/validate-request';
import { logger } from '../config/logger';
import { 
  authenticate, 
  loginLimiter, 
  csrfProtection,
  apiLimiter as authLimiter,
  type Request as AuthRequest
} from '../middleware/auth.middleware';
import { supabase } from '../config/supabase';

// Use the Request type from auth middleware
type Request = AuthRequest;

export const authRouter = Router();

// Helper function to handle errors
const handleError = (error: unknown, res: Response, defaultMessage: string) => {
  const errorMessage = error instanceof Error ? error.message : defaultMessage;
  logger.error('Auth error:', error);
  return res.status(400).json({
    success: false,
    error: errorMessage
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
    authLimiter,
    csrfProtection
  ],
  async (req: Request, res: Response) => {
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
    loginLimiter,
    csrfProtection
  ],
  async (req: Request, res: Response) => {
    try {
      const { email, password } = req.body;
      
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        throw error;
      }
      
      res.json({
        success: true,
        message: 'Login successful',
        user: data.user,
        session: data.session
      });
    } catch (error: unknown) {
      logger.error('Login error:', error);
      res.status(401).json({
        success: false,
        error: (error as Error).message || 'Invalid email or password'
      });
    }
  }
);

// Refresh session
authRouter.post('/refresh', async (req, res, next) => {
  try {
    const { refresh_token } = req.body;
    if (!refresh_token) {
      return res.status(400).json({
        success: false,
        error: 'Refresh token is required'
      });
    }

    const { data, error } = await supabase.auth.refreshSession({
      refresh_token,
      // @ts-ignore
      session: req.session,
    });
    
    if (error) {
      throw new Error(error.message);
    }
    
    res.json({
      success: true,
      user: data.user,
      session: data.session
    });
  } catch (error: unknown) {
    logger.error('Session refresh error:', error);
    res.status(401).json({
      success: false,
      error: 'Failed to refresh session'
    });
  }
});

// Get current user
authRouter.get(
  '/me', 
  [authenticate, authLimiter],
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
  [body('email').isEmail().normalizeEmail()],
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
  }
);

// Update user profile
// Requires authentication
// Uses the Supabase Admin API to update user metadata
authRouter.patch(
  '/profile',
  [
    authenticate,
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

// Get user profile
authRouter.get('/profile', authenticate, async (req: Request, res: Response) => {
  try {
    res.json({
      success: true,
      user: req.user
    });
  } catch (error: unknown) {
    logger.error('Get profile error:', error);
    res.status(400).json({
      success: false,
      error: 'Failed to get user profile'
    });
  }
});

// OAuth providers configuration
authRouter.get('/providers', (req: Request, res: Response) => {
  const providers = [
    {
      id: 'google',
      name: 'Google',
      enabled: process.env.OAUTH_GOOGLE_ENABLED === 'true',
      url: '/auth/google/authorize'
    },
    {
      id: 'github',
      name: 'GitHub',
      enabled: process.env.OAUTH_GITHUB_ENABLED === 'true',
      url: '/auth/github/authorize'
    },
    // Add more providers as needed
  ].filter(provider => provider.enabled);
  
  res.json({
    success: true,
    providers
  });
});

// Initialize OAuth flow
authRouter.get('/:provider/authorize', async (req: Request, res: Response) => {
  const { provider } = req.params;
  const redirectTo = req.query.redirectTo || '/';
  
  // Store the redirect URL in the session or state
  const state = JSON.stringify({ redirectTo });
  
  // Generate the OAuth URL
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: provider as any,
    options: {
      redirectTo: `${process.env.API_URL}/auth/${provider}/callback`,
      queryParams: { state }
    }
  });
  
  if (error) {
    logger.error(`OAuth ${provider} error:`, error);
    return res.redirect(`/login?error=${encodeURIComponent(error.message)}`);
  }
  
  res.redirect(data.url);
});

// OAuth callback
authRouter.get('/:provider/callback', async (req: Request, res: Response) => {
  const { provider } = req.params;
  const { code, state } = req.query;
  
  try {
    if (!code) {
      throw new Error('Authorization code is required');
    }
    
    // Exchange the code for a session
    const { data, error } = await supabase.auth.exchangeCodeForSession(code as string);
    
    if (error) {
      throw new Error(error.message);
    }
    
    // Redirect to the client with the session
    const stateObj = state ? JSON.parse(state as string) : {};
    const redirectUrl = new URL(stateObj.redirectTo || '/', process.env.CLIENT_URL);
    
    // Set the session in a cookie or local storage via query params
    redirectUrl.searchParams.set('access_token', data.session.access_token);
    redirectUrl.searchParams.set('refresh_token', data.session.refresh_token);
    
    res.redirect(redirectUrl.toString());
  } catch (error: unknown) {
    logger.error(`OAuth ${provider} callback error:`, error);
    const redirectUrl = new URL('/login', process.env.CLIENT_URL);
    redirectUrl.searchParams.set(
      'error', 
      error instanceof Error ? error.message : 'Authentication failed'
    );
    res.redirect(redirectUrl.toString());
  }
});

// Change password route
authRouter.post(
  '/change-password',
  authenticate,
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
