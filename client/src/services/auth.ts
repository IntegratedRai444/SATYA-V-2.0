import { apiClient } from '@/lib/api/apiClient';
import logger from '../lib/logger';

// Constants for storage keys
const ACCESS_TOKEN_KEY = 'satya_access_token';
const REFRESH_TOKEN_KEY = 'satya_refresh_token';
const USER_KEY = 'satya_user';
const TOKEN_TIMESTAMP_KEY = 'satya_token_timestamp';

// Note: TOKEN_EXPIRY_TIME is not currently used, but keeping it for future reference
// const TOKEN_EXPIRY_TIME = 30 * 60 * 1000; // 30 minutes

export interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  full_name?: string;
}

export interface AuthResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}

// Token Management
export const getAuthToken = (): string | null => {
  return localStorage.getItem(ACCESS_TOKEN_KEY);
};

export const getRefreshToken = (): string | null => {
  return localStorage.getItem(REFRESH_TOKEN_KEY);
};

export const setAuthToken = (token: string): void => {
  localStorage.setItem(ACCESS_TOKEN_KEY, token);};

export const setRefreshToken = (token: string): void => {
  localStorage.setItem(REFRESH_TOKEN_KEY, token);
};

export const removeAuthToken = (): void => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

// User Management
export const getCurrentUser = (): User | null => {
  const user = localStorage.getItem(USER_KEY);
  return user ? JSON.parse(user) : null;
};

const setCurrentUser = (user: User): void => {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
};

// Auth Operations
export const login = async (username: string, password: string, role?: 'user' | 'admin'): Promise<AuthResponse> => {
  try {
    // Input validation
    if (!username || !password) {
      throw new Error('Username and password are required');
    }

    // Sanitize input
    const sanitizedUsername = username.trim();
    
    // Additional validation for email format if needed
    if (sanitizedUsername.includes('@') && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(sanitizedUsername)) {
      throw new Error('Please enter a valid email address');
    }

    // Password strength validation
    if (password.length < 8) {
      throw new Error('Password must be at least 8 characters long');
    }

    const response = await apiClient.post<AuthResponse>(
      '/api/auth/login', 
      {
        username: sanitizedUsername,
        password,
        role
      },
      {
        skipAuth: true, // Skip auth header for login request
        timeout: 15000 // 15 second timeout
      }
    );

    if (!response.access_token || !response.refresh_token) {
      throw new Error('Invalid response from server');
    }

    // Save tokens and user data
    setAuthToken(response.access_token);
    setRefreshToken(response.refresh_token);
    setCurrentUser(response.user);
    
    // Set token timestamp for auto-logout
    localStorage.setItem(TOKEN_TIMESTAMP_KEY, Date.now().toString());
    
    logger.info('Login successful', { 
      userId: response.user.id, 
      role: response.user.role 
    });
    
    return response;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Login failed';
    logger.error(`Login failed: ${errorMessage}`);
    logger.debug(`Login attempt with username: ${username ? 'provided' : 'not provided'}`);
    throw new Error(errorMessage);
  }
};

export const logout = async (): Promise<void> => {
  try {
    // Get refresh token before clearing
    const refreshToken = getRefreshToken();
    
    // Clear all auth-related data first to prevent race conditions
    removeAuthToken();
    localStorage.removeItem(TOKEN_TIMESTAMP_KEY);
    
    // Call server to invalidate session
    if (refreshToken) {
      try {
        await apiClient.post(
          '/api/auth/logout', 
          { refresh_token: refreshToken },
          { 
            skipAuth: true, // Don't use auth header for logout
            withCredentials: true // Include cookies
          }
        );
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        logger.warn(`Logout API call failed, but proceeding with client cleanup: ${errorMessage}`);
      }
    }
    
    // Clear any cached data that might be user-specific
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('satya_') || key.startsWith('sb-')) {
        localStorage.removeItem(key);
      }
    });
    
    // Clear session storage
    sessionStorage.clear();
    
    // Dispatch events for other parts of the app to handle
    const authEvent = new Event('auth-logout');
    window.dispatchEvent(authEvent);
    
    // Redirect to login page
    window.location.href = '/login';
    
    // Force stop any pending requests
    if (typeof window !== 'undefined') {
      // This will prevent any pending requests from completing
      window.stop();
    }
    
    logger.info('User logged out successfully');
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Logout error: ${errorMessage}`);
    
    // Ensure we still redirect to login even if there was an error
    window.location.href = '/login';
  }
};

export const refreshToken = async (): Promise<{ access_token: string; refresh_token?: string }> => {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    logger.warn('Refresh token not available');
    throw new Error('No refresh token available');
  }

  try {
    const response = await apiClient.post<{ 
      access_token: string; 
      refresh_token?: string;
      expires_in?: number;
    }>(
      '/api/auth/refresh',
      { refresh_token: refreshToken },
      { skipAuth: true, timeout: 10000 }
    );
    
    if (!response.access_token) {
      throw new Error('Invalid token response');
    }
    
    // Update the access token
    setAuthToken(response.access_token);
    
    // If a new refresh token is provided, update it
    if (response.refresh_token) {
      setRefreshToken(response.refresh_token);
    }
    
    // Update token timestamp
    localStorage.setItem(TOKEN_TIMESTAMP_KEY, Date.now().toString());
    
    logger.info('Token refreshed successfully');
    
    return {
      access_token: response.access_token,
      refresh_token: response.refresh_token
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Token refresh failed';
    logger.error(`Token refresh failed: ${errorMessage}`);
    logger.debug(`Refresh token available: ${!!refreshToken}`);
    
    // Clear auth data on refresh failure
    removeAuthToken();
    localStorage.removeItem(TOKEN_TIMESTAMP_KEY);
    
    throw new Error('Session expired. Please log in again.');
  }
};

export const getCurrentUserProfile = async (): Promise<User> => {
  const response = await apiClient.get<User>('/api/auth/me');
  setCurrentUser(response);
  return response;
};

export const updateProfile = async (updates: Partial<User>): Promise<User> => {
  const response = await apiClient.put<User>('/api/auth/me', updates);
  setCurrentUser(response);
  return response;
};

export const isAuthenticated = (): boolean => {
  const token = getAuthToken();
  if (!token) return false;

  try {
    // Check token expiration
    const tokenData = JSON.parse(atob(token.split('.')[1]));
    const isExpired = tokenData.exp * 1000 < Date.now();
    
    if (isExpired) {
      logger.info('Token expired, logging out');
      logout();
      return false;
    }
    
    // Check if we have a valid user
    const user = getCurrentUser();
    if (!user || !user.id) {
      logger.warn('No valid user data found');
      return false;
    }
    
    return true;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    logger.error(`Error validating authentication: ${errorMessage}`);
    return false;
  }
};

export const hasRole = (requiredRole: string | string[]): boolean => {
  if (!isAuthenticated()) return false;
  
  const user = getCurrentUser();
  if (!user || !user.role) return false;
  
  // Handle both single role and array of roles
  if (Array.isArray(requiredRole)) {
    return requiredRole.includes(user.role);
  }
  
  return user.role === requiredRole;
};

export const authService = {
  login,
  logout,
  refreshToken,
  getCurrentUser,
  getCurrentUserProfile,
  updateProfile,
  isAuthenticated,
  hasRole,
  getAuthToken,
  removeAuthToken
};
