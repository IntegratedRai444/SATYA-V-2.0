/**
 * Mock Authentication Service
 * Provides simulated authentication without requiring the Python server
 */
export interface LoginResponse {
  success: boolean;
  message: string;
  token?: string;
  user?: {
    id: number;
    username: string;
    email?: string;
    role?: string;
  };
}

export interface SessionResponse {
  valid: boolean;
  message?: string;
  user?: {
    id: number;
    username: string;
    email?: string;
    role?: string;
  };
}

/**
 * Simple mock authentication service
 */
class MockAuthService {
  private initialized = false;
  private activeTokens: Record<string, any> = {};

  /**
   * Initialize the authentication system
   */
  async initialize(): Promise<boolean> {
    this.initialized = true;
    
    // Create demo user token
    const demoToken = `demo-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
    this.activeTokens[demoToken] = {
      user: {
        id: 1,
        username: 'demo',
        email: 'demo@example.com',
        role: 'user'
      },
      created: Date.now()
    };
    
    console.log('Mock auth service initialized successfully');
    return true;
  }

  /**
   * Authenticate a user with username and password
   */
  async login(username: string, password: string): Promise<LoginResponse> {
    // For demo purposes, accept any login
    if (username && password) {
      const token = `user-${Date.now()}-${Math.random().toString(36).substring(2, 15)}`;
      this.activeTokens[token] = {
        user: {
          id: 1,
          username,
          email: `${username}@example.com`,
          role: 'user'
        },
        created: Date.now()
      };
      
      return {
        success: true,
        message: 'Authentication successful',
        token,
        user: {
          id: 1,
          username,
          email: `${username}@example.com`,
          role: 'user'
        }
      };
    }
    
    return {
      success: false,
      message: 'Invalid username or password'
    };
  }

  /**
   * Validate a session token
   */
  async validateSession(token: string): Promise<SessionResponse> {
    if (this.activeTokens[token]) {
      return {
        valid: true,
        message: 'Session is valid',
        user: this.activeTokens[token].user
      };
    }
    
    return {
      valid: false,
      message: 'Invalid or expired session'
    };
  }

  /**
   * Logout and invalidate session
   */
  async logout(token: string): Promise<{ success: boolean; message: string }> {
    if (this.activeTokens[token]) {
      delete this.activeTokens[token];
      return {
        success: true,
        message: 'Logout successful'
      };
    }
    
    return {
      success: true,
      message: 'Session already invalidated'
    };
  }
}

// Export a singleton instance
export const mockAuthService = new MockAuthService();