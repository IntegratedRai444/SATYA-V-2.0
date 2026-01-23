import { Router, type Request as ExpressRequest, type Response } from 'express';
import { body } from 'express-validator';
import { validateRequest } from '../middleware/validate-request';
import { logger } from '../config/logger';
import { 
  authenticate, 
  loginLimiter, 
  refreshTokenLimiter,
  csrfProtection,
  generateCsrfToken,
  apiLimiter as authLimiter
} from '../middleware/auth.middleware';
import { supabase } from '../config/supabase';


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

// Sign up with email/password
authRouter.post(
  '/signup',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 8 })
      .withMessage('Password must be at least 8 characters long'),
    validateRequest,
    authLimiter,
    csrfProtection,
    async (req: ExpressRequest, res: Response) => {
      try {
        const { email, password, user_metadata } = req.body;
        
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: {
            data: user_metadata || {},
          }
        });

        if (error) {
          logger.warn('Signup failed', { error: error.message, email });
          return res.status(400).json({
            success: false,
            code: 'SIGNUP_FAILED',
            message: error.message
          });
        }

        // User profile is created automatically by database trigger
        // No need for manual insertion here

        res.status(201).json({
          success: true,
          message: 'User registered successfully. Please check your email to verify your account.',
          user: data.user
        });
      } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : 'Signup failed';
        logger.error('Signup error:', error);
        res.status(500).json({
          success: false,
          code: 'SIGNUP_ERROR',
          message: errorMessage
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
    loginLimiter,
    csrfProtection
  ],
  async (req: ExpressRequest, res: Response) => {
    try {
      const { email, password } = req.body;
      
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password
      });

      if (error) {
        logger.warn('Login failed', { error: error.message, email });
        return res.status(401).json({
          success: false,
          code: 'INVALID_CREDENTIALS',
          message: 'Invalid email or password'
        });
      }
      
      // Check if email is verified
      if (!data.user?.email_confirmed_at) {
        return res.status(403).json({
          success: false,
          code: 'EMAIL_NOT_VERIFIED',
          message: 'Please verify your email address before logging in.'
        });
      }
      
      // Set secure HTTP-only cookies
      const isProduction = process.env.NODE_ENV === 'production';
      const baseCookieOptions = {
        httpOnly: true,
        secure: isProduction,
        sameSite: isProduction ? 'none' : 'lax' as const,
        path: '/',
        domain: isProduction ? process.env.COOKIE_DOMAIN : undefined,
      } as const;

      // Set access token cookie (short-lived)
      res.cookie('sb-access-token', data.session?.access_token, {
        ...baseCookieOptions,
        maxAge: 15 * 60 * 1000 // 15 minutes
      });

      // Set refresh token cookie (longer-lived)
      // Path MUST match v2 refresh endpoint
      res.cookie('sb-refresh-token', data.session?.refresh_token, {
        ...baseCookieOptions,
        path: '/api/v2/auth/refresh',
        maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
      });
      
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
        code: 'LOGIN_FAILED',
        message: 'Login failed'
      });
    }
  }
);

// Get current user
authRouter.get(
  '/me', 
  [
    authenticate,
    authLimiter,
    csrfProtection,
    async (req: ExpressRequest, res: Response) => {
      try {
        // Get fresh user data from Supabase
        const { data: { user }, error: userError } = await supabase.auth.getUser();
        
        if (userError || !user) {
          throw new Error('User not found');
        }
        
        // Get additional user data from custom table
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
  ]
);

// Refresh access token
authRouter.post(
  '/refresh',
  [
    body('refresh_token').notEmpty().withMessage('Refresh token is required'),
    validateRequest,
    refreshTokenLimiter,
    csrfProtection,
    async (req: ExpressRequest, res: Response) => {
      try {
        const { refresh_token } = req.body;
        
        // Get refresh token from cookie if not in body
        const tokenToUse = refresh_token || req.cookies?.['sb-refresh-token'];
        
        if (!tokenToUse) {
          return res.status(401).json({
            success: false,
            code: 'REFRESH_TOKEN_REQUIRED',
            message: 'Refresh token is required'
          });
        }
        
        const { data, error } = await supabase.auth.refreshSession(tokenToUse);
        
        if (error) {
          logger.warn('Token refresh failed', { error: error.message });
          return res.status(401).json({
            success: false,
            code: 'REFRESH_FAILED',
            message: 'Failed to refresh session'
          });
        }
        
        // Set new access token cookie
        const isProduction = process.env.NODE_ENV === 'production';
        const baseCookieOptions = {
          httpOnly: true,
          secure: isProduction,
          sameSite: isProduction ? 'none' : 'lax' as const,
          path: '/',
          domain: isProduction ? process.env.COOKIE_DOMAIN : undefined,
        } as const;

        res.cookie('sb-access-token', data.session?.access_token, {
          ...baseCookieOptions,
          maxAge: 15 * 60 * 1000 // 15 minutes
        });
        
        res.json({
          success: true,
          message: 'Token refreshed successfully',
          session: data.session
        });
      } catch (error: unknown) {
        const errorMessage = error instanceof Error ? error.message : 'Token refresh failed';
        logger.error('Refresh token error:', error);
        res.status(401).json({
          success: false,
          code: 'REFRESH_ERROR',
          message: errorMessage
        });
      }
    }
  ]
);

// Logout
authRouter.post(
  '/logout',
  [
    authenticate,
    csrfProtection,
    async (req: ExpressRequest, res: Response) => {
      try {
        // Clear Supabase session
        await supabase.auth.signOut();
        
        // Clear cookies
        const cookieOptions = {
          httpOnly: true,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'strict' as const,
          path: '/',
          domain: process.env.COOKIE_DOMAIN || undefined
        };

        res.clearCookie('sb-access-token', cookieOptions);
        res.clearCookie('sb-refresh-token', {
          ...cookieOptions,
          path: '/api/v2/auth/refresh'
        });
        
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
