import { apiClient } from '@/lib/api/apiClient';
import logger from '../lib/logger';

const ACCESS_TOKEN_KEY = 'satya_access_token';
const REFRESH_TOKEN_KEY = 'satya_refresh_token';
const USER_KEY = 'satya_user';

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
    const response = await apiClient.post<AuthResponse>('/api/auth/login', {
      username,
      password,
      role
    });

    // Save tokens and user data
    setAuthToken(response.access_token);
    setRefreshToken(response.refresh_token);
    setCurrentUser(response.user);
    
    logger.info('Login successful', { username, role });
    return response;
  } catch (error) {
    logger.error('Login failed', { error, username });
    throw error;
  }
};

export const logout = async (): Promise<void> => {
  try {
    await apiClient.post('/api/auth/logout', {});
  } catch (error) {
    logger.warn('Logout request failed', { error });
  } finally {
    // Always clear local auth data
    removeAuthToken();
  }
};

export const refreshToken = async (): Promise<{ access_token: string }> => {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }

  try {
    const response = await apiClient.post<{ access_token: string }>(
      '/api/auth/refresh',
      { refresh_token: refreshToken }
    );
    
    setAuthToken(response.access_token);
    return response;
  } catch (error) {
    logger.error('Token refresh failed', { error });
    removeAuthToken();
    throw error;
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
  return !!getAuthToken();
};

export const hasRole = (requiredRole: string): boolean => {
  const user = getCurrentUser();
  return user?.role === requiredRole;
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
