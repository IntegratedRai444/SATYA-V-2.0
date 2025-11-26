import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import apiClient from '../lib/api';
import { removeAuthToken, shouldRefreshToken } from '../services/auth';
import logger from '../lib/logger';

export interface User {
  id: string;
  username: string;
  email?: string;
  fullName?: string;
  role: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  connectionStatus: 'connected' | 'disconnected' | 'checking';
  login: (username: string, password: string, overrideRole?: 'user' | 'admin') => Promise<boolean>;
  register: (username: string, email: string, password: string) => Promise<boolean>;
  logout: () => Promise<void>;
  error: string | null;
  clearError: () => void;
  retryConnection: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');

  // Check if user is authenticated on app start
  useEffect(() => {
    // Run diagnostics in development
    if (import.meta.env.DEV) {
      logger.info('SatyaAI Frontend Starting...');
      apiClient.runDiagnostics().then(() => {
        checkAuthStatus();
      });
    } else {
      checkAuthStatus();
    }

    // Listen for auth cleared event (when token is removed)
    const handleAuthCleared = () => {
      logger.info('Auth cleared event received');
      setUser(null);
      setError('Session expired. Please login again.');
    };

    // Listen for storage changes from other tabs
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'satyaai_auth_token' && !e.newValue) {
        logger.info('Token removed from storage (other tab)');
        setUser(null);
        setError('Session expired. Please login again.');
      }
    };

    window.addEventListener('auth-cleared', handleAuthCleared);
    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('auth-cleared', handleAuthCleared);
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  // Separate effect for token refresh interval that depends on user
  useEffect(() => {
    if (!user) return;

    // Setup token refresh interval only when user is logged in
    const refreshInterval = setInterval(() => {
      if (shouldRefreshToken()) {
        logger.info('Token refresh needed, refreshing token');
        // Refresh token
        apiClient.refreshToken()
          .then((result) => {
            if (result.success) {
              logger.info('Token refreshed successfully in background');
            }
          })
          .catch((error) => {
            logger.error('Background token refresh failed', error);
            // Don't clear auth here, let the interceptor handle it on next request
          });
      }
    }, 60000); // Check every minute

    return () => {
      clearInterval(refreshInterval);
    };
  }, [user]); // ✅ Now depends on user

  const checkAuthStatus = async () => {
    try {
      setIsLoading(true);
      setConnectionStatus('checking');
      setError(null);

      logger.info('Starting authentication check...');

      // Check if backend is available
      const healthCheck = await apiClient.checkBackendHealth();

      if (!healthCheck.connected) {
        logger.error('Backend server unavailable');
        setConnectionStatus('disconnected');
        setError(`Backend server unavailable: ${healthCheck.error || 'Cannot connect to server'}`);
        setIsLoading(false);
        // Don't clear auth on network errors - keep token for when backend comes back
        return;
      }

      setConnectionStatus('connected');
      logger.info('Backend is available');

      // Check existing authentication
      if (!apiClient.isAuthenticated()) {
        logger.info('No existing authentication found');
        setIsLoading(false);
        return;
      }

      logger.info('Validating existing session...');

      // Try to refresh token if it's close to expiry
      if (shouldRefreshToken()) {
        logger.info('Token needs refresh, attempting refresh...');
        const refreshResult = await apiClient.refreshToken();
        if (!refreshResult.success) {
          logger.warn('Token refresh failed, clearing auth');
          apiClient.clearAuth();
          setUser(null);
          setIsLoading(false);
          return;
        }
        logger.info('Token refreshed successfully');
      }

      const sessionResult = await apiClient.validateSession();

      if (sessionResult.valid && sessionResult.user) {
        logger.info('Session valid, user authenticated');
        setUser({
          id: sessionResult.user.id?.toString() || '',
          username: sessionResult.user.username || '',
          email: sessionResult.user.email,
          fullName: sessionResult.user.fullName,
          role: sessionResult.user.role || 'user'
        });
      } else {
        logger.warn('Session invalid, clearing auth');
        apiClient.clearAuth();
        setUser(null);
      }
    } catch (error: any) {
      logger.error('Auth check failed', error);

      // Only clear auth on actual auth errors, not network errors
      if (error.response?.status === 401) {
        logger.warn('401 error, clearing auth');
        apiClient.clearAuth();
        setUser(null);
        setError('Session expired. Please login again.');
      } else {
        // Network error - keep auth and show connection error
        setConnectionStatus('disconnected');
        if (error.code === 'ECONNREFUSED') {
          setError('Cannot connect to SatyaAI server. Please ensure the backend is running.');
        } else if (error.name === 'TimeoutError') {
          setError('Connection timeout. The server may be overloaded.');
        } else {
          setError(`Connection error: ${error.message}`);
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string, overrideRole?: 'user' | 'admin'): Promise<boolean> => {
    try {
      setError(null);
      setIsLoading(true);

      logger.info('Attempting login', { username });
      const response = await apiClient.login(username, password);

      if (response.success && response.user && response.token) {
        // API client already sets the token in its login method with expiry
        // No need to call setAuthToken again as it's redundant

        // Use overrideRole if provided (for Rishabh Kapoor's role selection)
        const userRole = overrideRole || response.user.role || 'user';

        const userData = {
          id: response.user.id?.toString() || '',
          username: response.user.username || username,
          email: response.user.email,
          fullName: response.user.fullName,
          role: userRole
        };

        // Set user state
        setUser(userData);

        logger.info('User logged in successfully', { username, role: userRole });

        // Trigger WebSocket connection after successful login
        window.dispatchEvent(new Event('auth-success'));

        // Return true to indicate success
        return true;
      } else {
        const errorMessage = response.message || 'Login failed';
        setError(errorMessage);
        logger.warn('Login failed', { username, reason: errorMessage });
        return false;
      }
    } catch (error: any) {
      const errorMessage = error.message || 'Login failed. Please check your connection.';
      setError(errorMessage);
      logger.error('Login error', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (username: string, email: string, password: string): Promise<boolean> => {
    try {
      setError(null);
      setIsLoading(true);

      const response = await apiClient.register(username, email, password);

      if (response.success && response.user && response.token) {
        // API client already sets the token in its register method with expiry

        setUser({
          id: response.user.id?.toString() || '',
          username: response.user.username || username,
          email: response.user.email || email,
          fullName: response.user.fullName,
          role: response.user.role || 'user'
        });

        logger.info('User registered successfully', { username });
        return true;
      } else {
        const errorMessage = response.errors ? response.errors.join(', ') : response.message;
        setError(errorMessage || 'Registration failed');
        logger.warn('Registration failed', { username, reason: errorMessage });
        return false;
      }
    } catch (error: any) {
      setError(error.message || 'Registration failed');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async (): Promise<void> => {
    try {
      await apiClient.logout();
      logger.info('User logged out successfully');
    } catch (error) {
      logger.warn('Logout request failed', error as Error);
    } finally {
      setUser(null);
      removeAuthToken();
      apiClient.clearAuth();
    }
  };

  const clearError = () => {
    setError(null);
  };

  const retryConnection = async () => {
    logger.info('Retrying connection...');
    setError(null);
    await checkAuthStatus();
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    connectionStatus,
    login,
    register,
    logout,
    error,
    clearError,
    retryConnection,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}