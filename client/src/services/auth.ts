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

// Cookie management utilities
const setCookie = (name: string, value: string, days: number = 7): void => {
  const date = new Date();
  date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
  const expires = `expires=${date.toUTCString()}`;
  document.cookie = `${name}=${value};${expires};path=/;SameSite=Strict${window.location.protocol === 'https:' ? ';Secure' : ''}`;
};

const getCookie = (name: string): string | null => {
  const nameEQ = name + '=';
  const ca = document.cookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') c = c.substring(1, c.length);
    if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
  }
  return null;
};

const deleteCookie = (name: string): void => {
  document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/`;
};

// Token Management
export const getAuthToken = (): string | null => {
  return getCookie(ACCESS_TOKEN_KEY);
};

export const getRefreshToken = (): string | null => {
  return getCookie(REFRESH_TOKEN_KEY);
};

export const setAuthToken = (token: string, rememberMe: boolean = false): void => {
  setCookie(ACCESS_TOKEN_KEY, token, rememberMe ? 30 : 1); // 30 days if remember me, else 1 day
};

export const setRefreshToken = (token: string, rememberMe: boolean = false): void => {
  setCookie(REFRESH_TOKEN_KEY, token, rememberMe ? 30 : 1);
};

export const removeAuthToken = (): void => {
  deleteCookie(ACCESS_TOKEN_KEY);
  deleteCookie(REFRESH_TOKEN_KEY);
  deleteCookie(USER_KEY);
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
export interface LoginCredentials {
  username: string;
  password: string;
  rememberMe?: boolean;
  role?: 'user' | 'admin';
}

export const login = async ({
  username,
  password,
  rememberMe = false,
  role
}: LoginCredentials): Promise<AuthResponse> => {
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

    // Get CSRF token first
    const csrfResponse = await fetch(`${import.meta.env.VITE_API_URL}/auth/csrf-token`, {
      credentials: 'include'
    });
    
    if (!csrfResponse.ok) {
      throw new Error('Failed to get CSRF token');
    }
    
    const { token: csrfToken } = await csrfResponse.json();
    
    // Make login request with CSRF token
    const response = await fetch(`${import.meta.env.VITE_API_URL}/auth/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken
      },
      credentials: 'include',
      body: JSON.stringify({ 
        username: sanitizedUsername, 
        password,
        ...(role && { role })
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || 'Login failed');
    }

    const data = await response.json();

    if (!data.access_token || !data.refresh_token) {
      throw new Error('Invalid response from server');
    }

    // Save tokens and user data
    setAuthToken(data.access_token, rememberMe);
    setRefreshToken(data.refresh_token, rememberMe);
    
    // Store user data in localStorage (excluding sensitive info)
    const { password: _, ...safeUserData } = data.user;
    setCurrentUser(safeUserData);

    // Set token timestamp
    localStorage.setItem(TOKEN_TIMESTAMP_KEY, Date.now().toString());
    
    logger.info('User logged in successfully');
    return data;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Login failed';
    logger.error(`Login failed: ${errorMessage}`);
    logger.debug(`Login attempt with username: ${username ? 'provided' : 'not provided'}`);
    
    // Clear any partial auth state on failure
    removeAuthToken();
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
    throw new Error('No refresh token available');
  }

  try {
    // Get CSRF token first
    const csrfResponse = await fetch(`${import.meta.env.VITE_API_URL}/auth/csrf-token`, {
      credentials: 'include'
    });
    
    if (!csrfResponse.ok) {
      throw new Error('Failed to get CSRF token');
    }
    
    const { token: csrfToken } = await csrfResponse.json();
    
    // Make refresh token request
    const response = await fetch(`${import.meta.env.VITE_API_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken
      },
      credentials: 'include',
      body: JSON.stringify({ refresh_token: refreshToken })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || 'Token refresh failed');
    }

    const data = await response.json();

    if (!data.access_token) {
      throw new Error('Invalid response from server');
    }

    // Update tokens
    const rememberMe = localStorage.getItem(TOKEN_TIMESTAMP_KEY) ? 
      Date.now() - parseInt(localStorage.getItem(TOKEN_TIMESTAMP_KEY) || '0', 10) > 7 * 24 * 60 * 60 * 1000 : false;
    
    setAuthToken(data.access_token, rememberMe);
    if (data.refresh_token) {
      setRefreshToken(data.refresh_token, rememberMe);
    }

    // Update token timestamp
    localStorage.setItem(TOKEN_TIMESTAMP_KEY, Date.now().toString());
    
    logger.debug('Token refreshed successfully');
    return data;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Token refresh failed';
    logger.error(`Token refresh failed: ${errorMessage}`);
    
    // Clear auth state on refresh failure
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
