import type { Request as ExpressRequest, Response as ExpressResponse } from 'express';
import { supabase } from '../config/supabase';
import { logger } from '../config/logger';
import { validatePassword } from '../../shared/utils/password';

// Node.js globals
declare const process: typeof globalThis.process;

interface AuthRequest extends ExpressRequest {
  user?: {
    id: string;
    email: string;
    role: string;
    email_verified: boolean;
    user_metadata?: Record<string, unknown>;
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
      
      // Sign up with Supabase Auth
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: { 
          data: user_metadata,
          emailRedirectTo: `${process.env.CLIENT_URL || 'http://localhost:5173'}/auth/callback`
        }
      });

      if (error) {
        logger.error('Supabase signup error:', error);
        throw error;
      }

      if (!data.user) {
        throw new Error('No user data returned from authentication service');
      }

      // Create user profile in public.users table
      if (data.user) {
        const { error: profileError } = await supabase
          .from('users')
          .insert([
            { 
              id: data.user.id,
              email: data.user.email,
              role: 'user',
              created_at: new Date().toISOString()
            }
          ]);

        if (profileError) {
          logger.error('Error creating user profile:', profileError);
          // Don't throw error, user is created in Supabase Auth
          // Profile can be created later
        }
      }

      res.status(201).json({
        success: true,
        message: 'Signup successful. Please check your email for verification.',
        user: data.user
      });
    } catch (error: unknown) {
      logger.error('Registration error:', error);
      
      // Handle specific Supabase errors
      if (error && typeof error === 'object' && 'status' in error && (error as { status: number }).status === 400) {
        const supabaseError = error as { code?: string; message?: string };
        return res.status(400).json({
          success: false,
          code: supabaseError.code || 'REGISTRATION_FAILED',
          message: supabaseError.message || 'Invalid registration data',
          error: process.env.NODE_ENV === 'development' ? error : undefined
        });
      }
      
      res.status(500).json({ 
        success: false, 
        code: 'REGISTRATION_ERROR',
        message: 'Failed to register user',
        error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
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
      
      // Get the session from Supabase
      const session = data.session;
      
      if (!session) {
        throw new Error('Failed to create session');
      }

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
      res.cookie('sb-access-token', session.access_token, {
        ...baseCookieOptions,
        maxAge: 15 * 60 * 1000 // 15 minutes
      });

      // Set refresh token cookie (longer-lived, httpOnly, secure)
      res.cookie('sb-refresh-token', session.refresh_token, {
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
        access_token: session.access_token,
        refresh_token: session.refresh_token,
        expires_in: session.expires_in,
        token_type: 'bearer'
      };
      
      res.json({
        success: true,
        user: userData,
        expiresIn: session.expires_in
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
      const refreshToken = req.cookies?.['sb-refresh-token'];
      
      if (!refreshToken) {
        return res.status(401).json({
          success: false,
          code: 'NO_REFRESH_TOKEN',
          message: 'No refresh token provided'
        });
      }

      const { data, error } = await supabase.auth.refreshSession({
        refresh_token: refreshToken
      });

      if (error) {
        logger.error('Token refresh error:', error);
        return res.status(401).json({
          success: false,
          code: 'INVALID_REFRESH_TOKEN',
          message: 'Invalid refresh token'
        });
      }

      if (!data.session) {
        return res.status(401).json({
          success: false,
          code: 'NO_SESSION',
          message: 'No session created'
        });
      }

      // Set new access token cookie
      const isProduction = process.env.NODE_ENV === 'production';
      const cookieOptions = {
        httpOnly: true,
        secure: isProduction,
        sameSite: isProduction ? 'none' : 'lax' as const,
        path: '/',
        domain: isProduction ? process.env.COOKIE_DOMAIN : undefined,
      } as const;

      res.cookie('sb-access-token', data.session.access_token, {
        ...cookieOptions,
        maxAge: 15 * 60 * 1000 // 15 minutes
      });

      res.json({
        success: true,
        access_token: data.session.access_token,
        expires_in: data.session.expires_in,
        token_type: 'bearer'
      });
    } catch (error) {
      logger.error('Refresh token error:', error);
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
      
      // Sign out from Supabase
      await supabase.auth.signOut();

      // Clear cookies with same options as when they were set
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
        path: '/api/auth/refresh'
      });

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
      if (!req.user) {
        return res.status(401).json({ 
          success: false, 
          code: 'UNAUTHORIZED',
          message: 'Not authenticated' 
        });
      }
      
      // Get fresh user data from Supabase
      const { data: { user }, error } = await supabase.auth.getUser();
      
      if (error || !user) {
        return res.status(404).json({ 
          success: false, 
          code: 'USER_NOT_FOUND',
          message: 'User not found' 
        });
      }
      
      // Get additional user data from custom table
      const { data: userData } = await supabase
        .from('users')
        .select('*')
        .eq('id', user.id)
        .single();
      
      res.json({ 
        success: true, 
        user: {
          id: user.id,
          email: user.email,
          email_verified: user.email_confirmed_at !== null,
          user_metadata: user.user_metadata,
          created_at: user.created_at,
          updated_at: user.updated_at,
          ...(userData || {})
        }
      });
    } catch (error) {
      logger.error('Error getting user profile:', error);
      res.status(500).json({ 
        success: false, 
        code: 'PROFILE_ERROR',
        message: 'Failed to get user profile' 
      });
    }
  },

  async updateProfile(req: AuthRequest, res: Response) {
    try {
      if (!req.user) {
        return res.status(401).json({ 
          success: false, 
          code: 'UNAUTHORIZED',
          message: 'Not authenticated' 
        });
      }
      
      const { fullName, email } = req.body;
      const userId = req.user.id;
      
      // Update in Supabase Auth
      const { data: authUser, error: authError } = await supabase.auth.updateUser({
        email: email,
        data: {
          full_name: fullName
        }
      });
      
      if (authError) {
        logger.error('Error updating auth profile:', authError);
        return res.status(400).json({ 
          success: false, 
          code: 'UPDATE_AUTH_ERROR',
          message: 'Failed to update auth profile' 
        });
      }
      
      // Update in custom users table
      const { data: updatedUser, error: dbError } = await supabase
        .from('users')
        .update({ 
          full_name: fullName as string, 
          email: email as string, 
          updated_at: new Date().toISOString() 
        })
        .eq('id', userId)
        .select('*')
        .single();
      
      if (dbError) {
        logger.error('Error updating user profile:', dbError);
        // Don't fail if DB update fails, auth update succeeded
      }
      
      res.json({ 
        success: true, 
        user: {
          id: authUser.user?.id,
          email: authUser.user?.email,
          email_verified: authUser.user?.email_confirmed_at !== null,
          user_metadata: authUser.user?.user_metadata,
          created_at: authUser.user?.created_at,
          updated_at: authUser.user?.updated_at,
          ...(updatedUser || {})
        }
      });
    } catch (error) {
      logger.error('Error updating profile:', error);
      res.status(500).json({ 
        success: false, 
        code: 'UPDATE_PROFILE_ERROR',
        message: 'Failed to update profile' 
      });
    }
  },

  async changePassword(req: AuthRequest, res: Response) {
    try {
      if (!req.user) {
        return res.status(401).json({ 
          success: false, 
          code: 'UNAUTHORIZED',
          message: 'Not authenticated' 
        });
      }
      
      const { currentPassword, newPassword } = req.body;
      
      // Validate new password
      const passwordValidation = validatePassword(newPassword);
      if (!passwordValidation.isValid) {
        return res.status(400).json({
          success: false,
          code: 'INVALID_PASSWORD',
          message: 'Password does not meet requirements',
          errors: passwordValidation.errors
        });
      }
      
      // Verify current password by attempting to sign in
      const { error: verifyError } = await supabase.auth.signInWithPassword({
        email: req.user.email,
        password: currentPassword
      });
      
      if (verifyError) {
        return res.status(400).json({ 
          success: false, 
          code: 'INVALID_CURRENT_PASSWORD',
          message: 'Current password is incorrect' 
        });
      }
      
      // Update password in Supabase Auth
      const { error: updateError } = await supabase.auth.updateUser({
        password: newPassword
      });
      
      if (updateError) {
        logger.error('Password update error:', updateError);
        return res.status(400).json({ 
          success: false, 
          code: 'PASSWORD_UPDATE_FAILED',
          message: 'Failed to update password' 
        });
      }
      
      res.json({ 
        success: true, 
        message: 'Password updated successfully' 
      });
    } catch (error) {
      logger.error('Change password error:', error);
      res.status(500).json({ 
        success: false, 
        code: 'CHANGE_PASSWORD_ERROR',
        message: 'Internal server error' 
      });
    }
  },

  // OAuth endpoints - simplified for Supabase
  async oauthRedirect(req: AuthRequest, res: Response) {
    try {
      const { provider } = req.params;
      
      if (!['google', 'github', 'microsoft'].includes(provider)) {
        return res.status(400).json({ 
          success: false, 
          code: 'INVALID_PROVIDER',
          message: 'Invalid OAuth provider' 
        });
      }

      // For Supabase OAuth, redirect to frontend to handle OAuth flow
      const baseUrl = process.env.CLIENT_URL || process.env.FRONTEND_URL || 'http://localhost:5173';
      const redirectUrl = `${baseUrl}/auth/oauth/${provider}`;
      
      res.redirect(redirectUrl);
    } catch (error) {
      logger.error('OAuth redirect error:', error);
      res.status(500).json({ 
        success: false, 
        code: 'OAUTH_ERROR',
        message: 'OAuth initialization failed' 
      });
    }
  },

  async oauthCallback(req: AuthRequest, res: Response) {
    try {
      const { provider } = req.params;
      const { code, state } = req.query;

      if (!code || !state) {
        return res.status(400).json({ 
          success: false, 
          code: 'INVALID_OAUTH_PARAMS',
          message: 'Invalid OAuth callback parameters' 
        });
      }

      // For Supabase OAuth, redirect to frontend with OAuth data
      const baseUrl = process.env.CLIENT_URL || process.env.FRONTEND_URL || 'http://localhost:5173';
      const redirectUrl = `${baseUrl}/auth/oauth/callback?provider=${provider}&code=${code}&state=${state}`;
      
      res.redirect(redirectUrl);
    } catch (error) {
      logger.error('OAuth callback error:', error);
      const fallbackUrl = process.env.CLIENT_URL || process.env.FRONTEND_URL || 'http://localhost:5173';
      res.redirect(`${fallbackUrl}/login?error=oauth_failed`);
    }
  }
};
