/**
 * Real Authentication Service
 * Handles secure token management and authentication operations
 */

import logger from '../lib/logger';

const TOKEN_KEY = 'satyaai_auth_token';
const TOKEN_EXPIRY_KEY = 'satyaai_token_expiry';
const REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before expiry

export type AuthToken = string | null;

/**
 * Securely get authentication token from storage
 */
export const getAuthToken = (): Promise<string> => {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
    
    if (!token) {
      return Promise.reject(new Error('No authentication token found'));
    }
    
    // Check if token is expired
    if (expiry && Date.now() > parseInt(expiry)) {
      removeAuthToken();
      return Promise.reject(new Error('Authentication token expired'));
    }
    
    return Promise.resolve(token);
  } catch (error) {
    logger.error('Error retrieving auth token', error as Error);
    return Promise.reject(error);
  }
};

/**
 * Securely set authentication token with expiry
 */
export const setAuthToken = (token: string, expiresIn: number = 24 * 60 * 60 * 1000): void => {
  try {
    localStorage.setItem(TOKEN_KEY, token);
    const expiryTime = Date.now() + expiresIn;
    localStorage.setItem(TOKEN_EXPIRY_KEY, expiryTime.toString());
    logger.info('Auth token set successfully');
  } catch (error) {
    logger.error('Error setting auth token', error as Error);
    throw error;
  }
};

/**
 * Remove authentication token and cleanup
 */
export const removeAuthToken = (): void => {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(TOKEN_EXPIRY_KEY);
    logger.info('Auth token removed');
  } catch (error) {
    logger.error('Error removing auth token', error as Error);
  }
};

/**
 * Check if user is authenticated with valid token
 */
export const isAuthenticated = async (): Promise<boolean> => {
  try {
    await getAuthToken();
    return true;
  } catch (error) {
    return false;
  }
};

/**
 * Check if token needs refresh
 */
export const shouldRefreshToken = (): boolean => {
  try {
    const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
    if (!expiry) return false;
    
    const timeUntilExpiry = parseInt(expiry) - Date.now();
    return timeUntilExpiry < REFRESH_THRESHOLD && timeUntilExpiry > 0;
  } catch (error) {
    return false;
  }
};

/**
 * Get token expiry time
 */
export const getTokenExpiry = (): Date | null => {
  try {
    const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
    return expiry ? new Date(parseInt(expiry)) : null;
  } catch (error) {
    return null;
  }
};

/**
 * Clear all authentication data
 */
export const clearAuthData = (): void => {
  removeAuthToken();
  // Clear any other auth-related data
  try {
    localStorage.removeItem('user_data');
    localStorage.removeItem('user_preferences');
  } catch (error) {
    logger.error('Error clearing auth data', error as Error);
  }
};
