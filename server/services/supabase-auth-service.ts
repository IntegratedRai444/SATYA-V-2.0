import { supabase } from './supabase-client';
import bcrypt from 'bcrypt';
import jwt from 'jsonwebtoken';
import { securityConfig, logger } from '../config';

export interface JWTPayload {
  userId: number;
  username: string;
  email?: string;
  role: string;
  iat?: number;
  exp?: number;
}

interface AuthResult {
  success: boolean;
  message: string;
  token?: string;
  user?: {
    id: number;
    username: string;
    email?: string;
    fullName?: string;
    role: string;
    createdAt?: Date;
  };
}

class SupabaseAuthService {
  private jwtSecret: string;
  private jwtExpiresIn: string;

  constructor() {
    this.jwtSecret = securityConfig.jwtSecret;
    this.jwtExpiresIn = securityConfig.jwtExpiresIn;
  }

  generateToken(payload: Omit<JWTPayload, 'iat' | 'exp'>): string {
    return jwt.sign(payload, this.jwtSecret, {
      expiresIn: this.jwtExpiresIn,
      issuer: 'satyaai',
      audience: 'satyaai-users'
    } as jwt.SignOptions);
  }

  async verifyToken(token: string): Promise<JWTPayload | null> {
    try {
      const decoded = jwt.verify(token, this.jwtSecret, {
        issuer: 'satyaai',
        audience: 'satyaai-users'
      }) as JWTPayload;

      // Verify user still exists
      const { data, error } = await supabase
        .from('users')
        .select('id')
        .eq('id', decoded.userId)
        .single();

      if (error || !data) {
        return null;
      }

      return decoded;
    } catch (error) {
      logger.debug('Token verification error', { error: (error as Error).message });
      return null;
    }
  }

  async register(userData: {
    username: string;
    password: string;
    email?: string;
    fullName?: string;
  }): Promise<AuthResult> {
    try {
      // Check if user exists
      const { data: existingUser } = await supabase
        .from('users')
        .select('id')
        .eq('username', userData.username)
        .single();

      if (existingUser) {
        return {
          success: false,
          message: 'Username already exists'
        };
      }

      // Check email if provided
      if (userData.email) {
        const { data: existingEmail } = await supabase
          .from('users')
          .select('id')
          .eq('email', userData.email)
          .single();

        if (existingEmail) {
          return {
            success: false,
            message: 'Email already registered'
          };
        }
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(userData.password, 12);

      // Create user
      const { data: newUser, error } = await supabase
        .from('users')
        .insert({
          username: userData.username,
          password: hashedPassword,
          email: userData.email,
          full_name: userData.fullName,
          role: 'user',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .select()
        .single();

      if (error || !newUser) {
        logger.error('Registration error', { error: error?.message });
        return {
          success: false,
          message: 'Registration failed'
        };
      }

      // Generate token
      const token = this.generateToken({
        userId: newUser.id,
        username: newUser.username,
        email: newUser.email || undefined,
        role: newUser.role
      });

      return {
        success: true,
        message: 'User registered successfully',
        token,
        user: {
          id: newUser.id,
          username: newUser.username,
          email: newUser.email,
          fullName: newUser.full_name,
          role: newUser.role
        }
      };
    } catch (error) {
      logger.error('Registration error', { error: (error as Error).message });
      return {
        success: false,
        message: 'Registration failed'
      };
    }
  }

  async login(credentials: {
    username: string;
    password: string;
  }): Promise<AuthResult> {
    try {
      // Find user
      const { data: user, error } = await supabase
        .from('users')
        .select('*')
        .eq('username', credentials.username)
        .single();

      if (error || !user) {
        return {
          success: false,
          message: 'Invalid username or password'
        };
      }

      // Verify password
      const passwordMatch = await bcrypt.compare(credentials.password, user.password);

      if (!passwordMatch) {
        return {
          success: false,
          message: 'Invalid username or password'
        };
      }

      // Generate token
      const token = this.generateToken({
        userId: user.id,
        username: user.username,
        email: user.email || undefined,
        role: user.role
      });

      return {
        success: true,
        message: 'Login successful',
        token,
        user: {
          id: user.id,
          username: user.username,
          email: user.email,
          fullName: user.full_name,
          role: user.role,
          createdAt: new Date(user.created_at)
        }
      };
    } catch (error) {
      logger.error('Login error', { error: (error as Error).message });
      return {
        success: false,
        message: 'Login failed'
      };
    }
  }

  async getUserById(userId: number): Promise<any> {
    try {
      const { data, error } = await supabase
        .from('users')
        .select('id, username, email, full_name, role, created_at')
        .eq('id', userId)
        .single();

      if (error || !data) {
        return null;
      }

      return {
        id: data.id,
        username: data.username,
        email: data.email,
        fullName: data.full_name,
        role: data.role,
        createdAt: new Date(data.created_at)
      };
    } catch (error) {
      logger.error('Get user error', { error: (error as Error).message });
      return null;
    }
  }

  async updatePassword(userId: number, currentPassword: string, newPassword: string): Promise<AuthResult> {
    try {
      const { data: user, error } = await supabase
        .from('users')
        .select('password')
        .eq('id', userId)
        .single();

      if (error || !user) {
        return {
          success: false,
          message: 'User not found'
        };
      }

      const passwordMatch = await bcrypt.compare(currentPassword, user.password);

      if (!passwordMatch) {
        return {
          success: false,
          message: 'Current password is incorrect'
        };
      }

      const hashedPassword = await bcrypt.hash(newPassword, 12);

      const { error: updateError } = await supabase
        .from('users')
        .update({ 
          password: hashedPassword,
          updated_at: new Date().toISOString()
        })
        .eq('id', userId);

      if (updateError) {
        return {
          success: false,
          message: 'Password update failed'
        };
      }

      return {
        success: true,
        message: 'Password updated successfully'
      };
    } catch (error) {
      logger.error('Update password error', { error: (error as Error).message });
      return {
        success: false,
        message: 'Password update failed'
      };
    }
  }

  async updateUserProfile(userId: number, updates: { fullName?: string; email?: string }): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('users')
        .update({
          full_name: updates.fullName,
          email: updates.email,
          updated_at: new Date().toISOString()
        })
        .eq('id', userId);

      return !error;
    } catch (error) {
      logger.error('Update profile error', { error: (error as Error).message });
      return false;
    }
  }

  async deleteUser(userId: number): Promise<boolean> {
    try {
      const { error } = await supabase
        .from('users')
        .delete()
        .eq('id', userId);

      return !error;
    } catch (error) {
      logger.error('Delete user error', { error: (error as Error).message });
      return false;
    }
  }
}

export const supabaseAuthService = new SupabaseAuthService();
