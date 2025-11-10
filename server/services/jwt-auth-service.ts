import jwt from 'jsonwebtoken';
import bcrypt from 'bcrypt';
import { db } from '../db';
import { users, insertUserSchema } from '@shared/schema';
import { eq } from 'drizzle-orm';
import { z } from 'zod';
import { securityConfig, logger, logSecurity } from '../config';

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
    role: string;
  };
}

class JWTAuthService {
  private jwtSecret: string;
  private jwtExpiresIn: string;
  private blacklistedTokens: Set<string> = new Set();

  constructor() {
    this.jwtSecret = securityConfig.jwtSecret;
    this.jwtExpiresIn = securityConfig.jwtExpiresIn;
    
    if (this.jwtSecret.length < 32) {
      logger.warn('JWT_SECRET should be at least 32 characters long for security');
    }
  }

  // Generate JWT token
  generateToken(payload: Omit<JWTPayload, 'iat' | 'exp'>): string {
    return jwt.sign(payload, this.jwtSecret, {
      expiresIn: this.jwtExpiresIn,
      issuer: 'satyaai',
      audience: 'satyaai-users'
    });
  }

  // Verify JWT token
  async verifyToken(token: string): Promise<JWTPayload | null> {
    try {
      if (this.blacklistedTokens.has(token)) {
        return null;
      }

      const decoded = jwt.verify(token, this.jwtSecret, {
        issuer: 'satyaai',
        audience: 'satyaai-users'
      }) as JWTPayload;

      // Verify user still exists in database
      const user = await db.select().from(users).where(eq(users.id, decoded.userId)).limit(1);
      if (user.length === 0) {
        return null;
      }

      return decoded;
    } catch (error) {
      logger.debug('Token verification error', { error: (error as Error).message });
      return null;
    }
  }

  // Register new user
  async register(userData: {
    username: string;
    password: string;
    email?: string;
    fullName?: string;
  }): Promise<AuthResult> {
    try {
      // Validate input
      const validatedData = insertUserSchema.parse(userData);

      // Check if user already exists
      const existingUser = await db.select()
        .from(users)
        .where(eq(users.username, validatedData.username))
        .limit(1);

      if (existingUser.length > 0) {
        return {
          success: false,
          message: 'Username already exists'
        };
      }

      // Check email if provided
      if (validatedData.email) {
        const existingEmail = await db.select()
          .from(users)
          .where(eq(users.email, validatedData.email))
          .limit(1);

        if (existingEmail.length > 0) {
          return {
            success: false,
            message: 'Email already registered'
          };
        }
      }

      // Hash password
      const saltRounds = 12;
      const hashedPassword = await bcrypt.hash(validatedData.password, saltRounds);

      // Create user
      const [newUser] = await db.insert(users).values({
        username: validatedData.username,
        password: hashedPassword,
        email: validatedData.email,
        fullName: validatedData.fullName,
        role: 'user'
      }).returning();

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
          email: newUser.email || undefined,
          role: newUser.role
        }
      };
    } catch (error) {
      logger.error('Registration error', { 
        error: (error as Error).message,
        username: userData.username 
      });
      
      if (error instanceof z.ZodError) {
        return {
          success: false,
          message: 'Invalid input data: ' + error.errors.map(e => e.message).join(', ')
        };
      }

      return {
        success: false,
        message: 'Registration failed'
      };
    }
  }

  // Login user
  async login(credentials: {
    username: string;
    password: string;
  }): Promise<AuthResult> {
    try {
      // Find user
      const [user] = await db.select()
        .from(users)
        .where(eq(users.username, credentials.username))
        .limit(1);

      if (!user) {
        return {
          success: false,
          message: 'Invalid username or password'
        };
      }

      // Verify password
      const isValidPassword = await bcrypt.compare(credentials.password, user.password);
      if (!isValidPassword) {
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
          email: user.email || undefined,
          role: user.role
        }
      };
    } catch (error) {
      logger.error('Login error', { 
        error: (error as Error).message,
        username: credentials.username 
      });
      return {
        success: false,
        message: 'Login failed'
      };
    }
  }

  // Logout user (blacklist token)
  async logout(token: string): Promise<AuthResult> {
    try {
      // Add token to blacklist
      this.blacklistedTokens.add(token);
      
      return {
        success: true,
        message: 'Logout successful'
      };
    } catch (error) {
      logger.error('Logout error', { error: (error as Error).message });
      return {
        success: false,
        message: 'Logout failed'
      };
    }
  }

  // Check if token is blacklisted
  isTokenBlacklisted(token: string): boolean {
    return this.blacklistedTokens.has(token);
  }

  // Clean expired tokens from blacklist (call periodically)
  cleanExpiredTokens(): void {
    const expiredTokens: string[] = [];
    
    this.blacklistedTokens.forEach((token) => {
      try {
        jwt.verify(token, this.jwtSecret);
      } catch (error) {
        // Token is expired or invalid, remove from blacklist
        expiredTokens.push(token);
      }
    });
    
    expiredTokens.forEach(token => this.blacklistedTokens.delete(token));
    
    if (expiredTokens.length > 0) {
      logger.info(`Cleaned ${expiredTokens.length} expired tokens from blacklist`);
    }
  }

  // Get user by ID
  async getUserById(userId: number) {
    try {
      const [user] = await db.select({
        id: users.id,
        username: users.username,
        email: users.email,
        fullName: users.fullName,
        role: users.role,
        createdAt: users.createdAt
      }).from(users).where(eq(users.id, userId)).limit(1);

      return user || null;
    } catch (error) {
      logger.error('Get user error', { error: (error as Error).message, userId });
      return null;
    }
  }

  // Update user password
  async updatePassword(userId: number, currentPassword: string, newPassword: string): Promise<AuthResult> {
    try {
      // Get user
      const [user] = await db.select().from(users).where(eq(users.id, userId)).limit(1);
      if (!user) {
        return {
          success: false,
          message: 'User not found'
        };
      }

      // Verify current password
      const isValidPassword = await bcrypt.compare(currentPassword, user.password);
      if (!isValidPassword) {
        return {
          success: false,
          message: 'Current password is incorrect'
        };
      }

      // Hash new password
      const saltRounds = 12;
      const hashedPassword = await bcrypt.hash(newPassword, saltRounds);

      // Update password
      await db.update(users)
        .set({ password: hashedPassword, updatedAt: new Date() })
        .where(eq(users.id, userId));

      return {
        success: true,
        message: 'Password updated successfully'
      };
    } catch (error) {
      logger.error('Update password error', { error: (error as Error).message, userId });
      return {
        success: false,
        message: 'Password update failed'
      };
    }
  }
}

// Export singleton instance
export const jwtAuthService = new JWTAuthService();

// Clean expired tokens every hour
setInterval(() => {
  jwtAuthService.cleanExpiredTokens();
}, 60 * 60 * 1000);