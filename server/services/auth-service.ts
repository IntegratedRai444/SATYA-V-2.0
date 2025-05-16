// Import our Python bridge adapter
import { 
  startPythonServer, 
  waitForServerReady, 
  login as pythonLogin,
  logout as pythonLogout,
  validateSession as pythonValidateSession 
} from "./python-adapter.js";

// Interface for login response
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

// Interface for session validation response
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
 * Authentication service that interfaces with the Python server
 */
class AuthService {
  private initialized = false;

  /**
   * Initialize the authentication system
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) {
      return true;
    }

    try {
      // Start the Python server if not already running
      await startPythonServer();
      // Wait for it to be ready
      const ready = await waitForServerReady();
      
      this.initialized = ready;
      return ready;
    } catch (error) {
      console.error('Failed to initialize auth service:', error);
      return false;
    }
  }

  /**
   * Authenticate a user with username and password
   */
  async login(username: string, password: string): Promise<LoginResponse> {
    try {
      // Ensure the Python server is running
      await this.initialize();
      
      // Use the Python bridge login function
      const result = await pythonLogin(username, password);
      
      return {
        success: result.success,
        message: result.message || 'Authentication failed',
        token: result.token,
        user: result.user,
      };
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        message: 'Authentication system error',
      };
    }
  }

  /**
   * Validate a session token
   */
  async validateSession(token: string): Promise<SessionResponse> {
    try {
      // Ensure Python server is running
      await this.initialize();
      
      // Use the Python bridge validateSession function
      const result = await pythonValidateSession(token);
      
      return {
        valid: result.valid,
        message: result.message,
        user: result.user,
      };
    } catch (error) {
      console.error('Session validation error:', error);
      return {
        valid: false,
        message: 'Session validation error',
      };
    }
  }

  /**
   * Logout and invalidate session
   */
  async logout(token: string): Promise<{ success: boolean; message: string }> {
    try {
      // Ensure Python server is running
      await this.initialize();
      
      // Use the Python bridge logout function
      const result = await pythonLogout();
      
      return {
        success: result.success,
        message: result.message || 'Logout successful',
      };
    } catch (error) {
      console.error('Logout error:', error);
      return {
        success: false,
        message: 'Logout system error',
      };
    }
  }
}

// Export a singleton instance
export const authService = new AuthService();