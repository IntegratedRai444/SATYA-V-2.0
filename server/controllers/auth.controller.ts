import type { Request as ExpressRequest, Response as ExpressResponse } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import * as authService from '../services/authService';
import { generateTokenPair, rotateRefreshToken } from '../services/auth/tokenService';
import { rateLimiter } from '../middleware/rateLimiter';
import jwt from 'jsonwebtoken';
import { addMinutes } from 'date-fns';
import { validatePassword } from '../../shared/utils/password';

interface AuthRequest extends ExpressRequest {
  user?: {
    id: string;
    email: string;
    role: string;
    email_verified: boolean;
    user_metadata?: Record<string, any>;
  };
}

type Response = ExpressResponse;

export const authController = {
  async register(req: AuthRequest, res: Response) {
    try {
      const { email, password, user_metadata } = req.body;
      
      // Validate password complexity
      const passwordValidation = validatePassword(password);
      if (!passwordValidation.isValid) {
        return res.status(400).json({
          success: false,
          code: 'INVALID_PASSWORD',
          message: 'Password does not meet requirements',
          errors: passwordValidation.errors
        });
      }
      
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { 
          data: user_metadata,
          emailRedirectTo: `${process.env.CLIENT_URL}/auth/callback`
        }
      });

      if (error) {
        logger.error('Supabase signup error:', error);
        throw error;
      }

      if (!data.user) {
        throw new Error('No user data returned from authentication service');
      }

      res.status(201).json({
        success: true,
        message: 'Signup successful. Please check your email for verification.',
        user: data.user
      });
    } catch (error: any) {
      logger.error('Registration error:', error);
      
      // Handle specific Supabase errors
      if (error.status === 400) {
        return res.status(400).json({
          success: false,
          code: error.code || 'REGISTRATION_FAILED',
          message: error.message || 'Invalid registration data',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
      
      res.status(500).json({ 
        success: false, 
        code: 'REGISTRATION_ERROR',
        message: 'Failed to register user',
        error: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  },

  async login(req: AuthRequest, res: Response) {
    try {
      const { email, password: userPassword } = req.body;
      const clientIp = req.ip || req.socket?.remoteAddress || 'unknown-ip';
      
      // Input validation
      if (!email || !userPassword) {
        return res.status(400).json({
          success: false,
          code: 'MISSING_CREDENTIALS',
          message: 'Email and password are required'
        });
      }

      // Check if user exists in our users table
      const { data: dbUser, error: userError } = await supabase
        .from('users')
        .select('*')
        .eq('email', email.trim())
        .limit(1);

      if (userError) {
        logger.error('Database error checking user:', userError);
      }

      const user = dbUser?.[0];

      if (user) {
        // Check if account is locked
        if (user.failed_login_attempts >= 5) {
          const lastFailedLogin = user.last_failed_login ? new Date(user.last_failed_login) : null;
          const lockoutExpires = lastFailedLogin ? new Date(lastFailedLogin.getTime() + 15 * 60 * 1000) : null;
          
          if (lockoutExpires && lockoutExpires > new Date()) {
            const remainingMinutes = Math.ceil((lockoutExpires.getTime() - Date.now()) / (60 * 1000));
            return res.status(429).json({
              success: false,
              code: 'ACCOUNT_LOCKED',
              message: `Account locked. Please try again in ${remainingMinutes} minutes.`,
              retryAfter: remainingMinutes * 60
            });
          } else {
            // Reset failed attempts if lockout period has passed
            await supabase
              .from('users')
              .update({ 
                failed_login_attempts: 0,
                last_failed_login: null,
                is_active: true 
              })
              .eq('id', user.id);
          }
        }
      }

      // Attempt login with Supabase Auth
      const { data, error } = await supabase.auth.signInWithPassword({
        email: email.trim(),
        password: userPassword
      });

      if (error) {
        // Handle failed login attempt
        if (user) {
          const failedAttempts = (user.failed_login_attempts || 0) + 1;
          const isLocked = failedAttempts >= 5;
          
          await supabase
            .from('users')
            .update({ 
              failed_login_attempts: failedAttempts,
              last_failed_login: new Date().toISOString(),
              is_active: !isLocked
            })
            .eq('id', user.id);

          if (isLocked) {
            logger.warn(`Account locked for email: ${email} after multiple failed attempts`, { 
              userId: user.id,
              ip: clientIp,
              timestamp: new Date().toISOString()
            });
            
            return res.status(429).json({
              success: false,
              code: 'ACCOUNT_LOCKED',
              message: 'Account locked due to too many failed login attempts. Please try again in 15 minutes.',
              retryAfter: 900 // 15 minutes in seconds
            });
          }
        }

        logger.warn(`Failed login attempt for email: ${email}`, { 
          error: error.message,
          ip: clientIp,
          remainingAttempts: user ? 5 - (user.failed_login_attempts || 0) - 1 : 4
        });
        
        return res.status(401).json({
          success: false,
          code: 'INVALID_CREDENTIALS',
          message: `Invalid email or password. ${user ? `${5 - (user.failed_login_attempts || 0) - 1} attempts remaining.` : ''}`
        });
      }

      // Get the user data from the response
      const authUser = data.user;
      
      // Reset failed login attempts on successful login
      if (user) {
        await supabase
          .from('users')
          .update({ 
            failed_login_attempts: 0,
            last_failed_login: null,
            last_login: new Date().toISOString(),
            last_ip: clientIp,
            is_active: true
          })
          .eq('id', user.id);
      }
      
      // Generate token pair using our token service
      const tokenPair = generateTokenPair({
        id: authUser.id,
        email: authUser.email || '',
        role: authUser.user_metadata?.role || 'user'
      });
      
      logger.info(`Successful login for user: ${authUser.email}`, {
        userId: authUser.id,
        ip: clientIp,
        timestamp: new Date().toISOString()
      });

      // Set secure HTTP-only cookies with proper configuration
      const isProduction = process.env.NODE_ENV === 'production';
      const baseCookieOptions = {
        httpOnly: true,
        secure: isProduction,
        sameSite: isProduction ? 'none' : 'lax' as const,
        path: '/',
        domain: isProduction ? process.env.COOKIE_DOMAIN : undefined,
      } as const;

      // Set access token cookie (short-lived)
      res.cookie('satya_access_token', tokenPair.accessToken, {
        ...baseCookieOptions,
        maxAge: 15 * 60 * 1000 // 15 minutes
      });

      // Set refresh token cookie (longer-lived, httpOnly, secure)
      res.cookie('satya_refresh_token', tokenPair.refreshToken, {
        ...baseCookieOptions,
        path: '/api/auth/refresh',
        maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
      });

      // Return user data (excluding sensitive information)
      const userData = {
        id: authUser.id,
        email: authUser.email,
        role: authUser.user_metadata?.role || 'user',
        email_verified: authUser.email_confirmed_at !== null,
        full_name: authUser.user_metadata?.full_name,
      };
      
      res.json({
        success: true,
        user: userData,
        expiresIn: 15 * 60 // 15 minutes in seconds
      });
    } catch (error) {
      logger.error('Login error:', error);
      res.status(500).json({ 
        success: false, 
        code: 'LOGIN_ERROR',
        message: 'An error occurred during login' 
      });
    }
  },

  async refreshToken(req: AuthRequest, res: Response) {
    try {
      const refreshToken = req.cookies?.satya_refresh_token;
      if (!refreshToken) {
        return res.status(401).json({
          success: false,
          code: 'MISSING_REFRESH_TOKEN',
          message: 'Refresh token is required'
        });
      }
      
      // Check rate limiting for refresh token endpoint
      const clientIp = req.ip || req.socket?.remoteAddress || 'unknown-ip';
      const userId = req.user?.id?.toString() || 'anonymous';
      
      try {
        // This will throw if rate limit is exceeded
        await rateLimiter('auth')(req, res, () => {});
      } catch (error) {
        logger.warn('Rate limit exceeded for refresh token', { userId, ip: clientIp });
        return res.status(429).json({
          success: false,
          code: 'RATE_LIMIT_EXCEEDED',
          message: 'Too many requests, please try again later'
        });
      }

      // Rotate the refresh token (invalidates the old one, generates new ones)
      const tokenPair = await rotateRefreshToken(refreshToken);
      
      // Set secure HTTP-only cookies with proper configuration
      const isProduction = process.env.NODE_ENV === 'production';
      const baseCookieOptions = {
        httpOnly: true,
        secure: isProduction,
        sameSite: isProduction ? 'none' : 'lax' as const,
        path: '/',
        domain: isProduction ? process.env.COOKIE_DOMAIN : undefined,
      } as const;

      // Set new access token cookie
      res.cookie('satya_access_token', tokenPair.accessToken, {
        ...baseCookieOptions,
        maxAge: 15 * 60 * 1000 // 15 minutes
      });

      // Set new refresh token cookie
      res.cookie('satya_refresh_token', tokenPair.refreshToken, {
        ...baseCookieOptions,
        path: '/api/auth/refresh',
        maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
      });

      return res.json({
        success: true,
        expiresIn: 15 * 60 // 15 minutes in seconds
      });
    } catch (error) {
      logger.error('Token refresh error:', error);
      
      if (error instanceof jwt.TokenExpiredError) {
        return res.status(401).json({
          success: false,
          code: 'TOKEN_EXPIRED',
          message: 'Refresh token has expired'
        });
      }

      if (error instanceof jwt.JsonWebTokenError) {
        return res.status(401).json({
          success: false,
          code: 'INVALID_TOKEN',
          message: 'Invalid refresh token'
        });
      }

      res.status(500).json({ 
        success: false, 
        code: 'REFRESH_ERROR',
        message: 'Failed to refresh token' 
      });
    }
  },

  async logout(req: AuthRequest, res: Response) {
    try {
      // Log logout event
      if (req.user) {
        logger.info(`User logged out`, { 
          userId: req.user.id,
          ip: req.ip || req.socket?.remoteAddress || 'unknown-ip',
          timestamp: new Date().toISOString()
        });
      }
      
      const accessToken = req.cookies?.satya_access_token;
      const refreshToken = req.cookies?.satya_refresh_token;

      // Clear cookies with same options as when they were set
      const cookieOptions = {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict' as const,
        path: '/',
        domain: process.env.COOKIE_DOMAIN || undefined
      };

      res.clearCookie('satya_access_token', cookieOptions);
      res.clearCookie('satya_refresh_token', {
        ...cookieOptions,
        path: '/api/auth/refresh'
      });

      // Blacklist tokens if they exist (simplified version without database)
      if (accessToken) {
        try {
          const decoded = jwt.verify(accessToken, process.env.JWT_SECRET || 'your-secret-key') as any;
          logger.info('Token invalidated during logout', { jti: decoded?.jti });
        } catch (error) {
          logger.error('Error during token invalidation:', error);
        }
      }

      res.json({
        success: true,
        message: 'Logged out successfully'
      });
    } catch (error) {
      logger.error('Logout error:', error);
      res.status(500).json({
        success: false,
        code: 'LOGOUT_FAILED',
        message: 'Failed to complete logout process'
      });
    }
  },

  async getProfile(req: AuthRequest, res: Response) {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
      }
      
      const { data: user, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', userId)
        .limit(1);
      
      if (error || !user || user.length === 0) {
        return res.status(404).json({ success: false, error: 'User not found' });
      }
      
      // Remove sensitive data
      const { password: _, ...userData } = user[0];
      res.json({ success: true, user: userData });
    } catch (error) {
      logger.error('Error getting user profile:', error);
      res.status(500).json({ success: false, error: 'Failed to get user profile' });
    }
  },

  async updateProfile(req: AuthRequest, res: Response) {
    try {
      const { fullName, email } = req.body;
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ success: false, error: 'Unauthorized' });
      }
      
      const { data: updatedUser, error } = await supabase
        .from('users')
        .update({ 
          full_name: fullName as string, 
          email: email as string, 
          updated_at: new Date().toISOString() 
        })
        .eq('id', userId)
        .select('*')
        .single();
      
      if (error || !updatedUser) {
        return res.status(404).json({ success: false, error: 'User not found' });
      }
      
      // Remove sensitive data
      const { password: _, ...userData } = updatedUser;
      res.json({ success: true, user: userData });
    } catch (error) {
      logger.error('Error updating profile:', error);
      res.status(500).json({ success: false, error: 'Failed to update profile' });
    }
  },

  async changePassword(req: AuthRequest, res: Response) {
    try {
      const userId = req.user?.id;
      if (!userId) {
        return res.status(401).json({ success: false, message: 'Unauthorized' });
      }
      
      const { currentPassword, newPassword } = req.body;
      
      // Get user from database
      const { data: user, error } = await supabase
        .from('users')
        .select('*')
        .eq('id', userId)
        .limit(1);
      
      if (error || !user || user.length === 0) {
        return res.status(404).json({ success: false, message: 'User not found' });
      }
      
      // In a real application, verify current password here
      // For now, we'll just update the password in Supabase Auth
      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword
      });
      
      if (updateError) {
        logger.error('Password update error:', updateError);
        return res.status(400).json({ success: false, message: 'Failed to update password' });
      }
      
      res.json({ success: true, message: 'Password updated successfully' });
    } catch (error) {
      logger.error('Change password error:', error);
      res.status(500).json({ success: false, message: 'Internal server error' });
    }
  },

  // OAuth endpoints
  async oauthRedirect(req: AuthRequest, res: Response) {
    try {
      const { provider } = req.params;
      const { redirectUri } = req.query;
      
      if (!['google', 'github', 'microsoft'].includes(provider)) {
        return res.status(400).json({ success: false, message: 'Invalid OAuth provider' });
      }

      // For now, redirect to frontend with provider info
      const redirectUrl = new URL(process.env.FRONTEND_URL || 'http://localhost:3000');
      redirectUrl.pathname = `/oauth/${provider}`;
      if (redirectUri) {
        redirectUrl.searchParams.set('redirect_uri', redirectUri as string);
      }
      
      res.redirect(redirectUrl.toString());
    } catch (error) {
      logger.error('OAuth redirect error:', error);
      res.status(500).json({ success: false, message: 'OAuth initialization failed' });
    }
  },

  async oauthCallback(req: AuthRequest, res: Response) {
    try {
      const { provider } = req.params;
      const { code, state } = req.query;

      if (!code || !state) {
        return res.status(400).json({ success: false, message: 'Invalid OAuth callback parameters' });
      }

      // For now, just redirect to the frontend with the code and state
      const redirectUrl = new URL(process.env.FRONTEND_URL || 'http://localhost:3000');
      redirectUrl.pathname = '/oauth/callback';
      redirectUrl.searchParams.set('provider', provider);
      redirectUrl.searchParams.set('code', code as string);
      redirectUrl.searchParams.set('state', state as string);
      
      res.redirect(redirectUrl.toString());
    } catch (error) {
      logger.error('OAuth callback error:', error);
      res.redirect(`${process.env.FRONTEND_URL || 'http://localhost:3000'}/login?error=oauth_failed`);
    }
  }
};
