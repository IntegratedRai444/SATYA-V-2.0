import { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode, FC } from 'react';
import { apiClient } from '../lib/api';
import logger from '../lib/logger';

// Types
type ConnectionStatus = 'checking' | 'connected' | 'disconnected';

interface User {
  id: string;
  username: string;
  email: string;
  role: string;
  fullName?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  connectionStatus: ConnectionStatus;
  login: (username: string, password: string, overrideRole?: 'user' | 'admin') => Promise<boolean>;
  register: (username: string, email: string, password: string) => Promise<boolean>;
  logout: (message?: string) => Promise<void>;
  refreshToken: () => Promise<boolean>;
  checkAuthStatus: () => Promise<boolean>;
  retryConnection: () => Promise<void>;
  clearError: () => void;
  getCsrfToken: () => string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: FC<AuthProviderProps> = ({ children }) => {
  // State management
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('checking');
  const [csrfToken, setCsrfToken] = useState<string | null>(null);

  // Get CSRF token from cookie
  const getCsrfToken = useCallback((): string | null => {
    if (typeof document === 'undefined') return null;
    const match = document.cookie.match(/XSRF-TOKEN=([^;]+)/);
    return match ? decodeURIComponent(match[1]) : null;
  }, []);

  // Logout function
  const handleLogout = useCallback(async (message?: string): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Clear auth state
      setUser(null);
      setConnectionStatus('disconnected');
      
      // Call server to clear httpOnly cookies
      try {
        await apiClient.post('/auth/logout', {}, { withCredentials: true });
      } catch (err) {
        logger.warn('Logout API call failed, but proceeding with client-side cleanup', err);
      }
      
      // Notify other tabs
      window.dispatchEvent(new Event('auth-cleared'));
      
      // Redirect to login page
      window.location.href = '/login' + (message ? `?message=${encodeURIComponent(message)}` : '');
      
      logger.info('User logged out successfully');
    } catch (error) {
      const errorMessage = error instanceof Error ? error : new Error('Failed to logout');
      logger.error('Logout error:', errorMessage);
      setError(errorMessage.message);
      throw errorMessage;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Refresh token function
  const refreshToken = useCallback(async (): Promise<boolean> => {
    try {
      const response = await apiClient.post<{ user: User }>(
        '/auth/refresh',
        {},
        { 
          skipAuth: true,
          withCredentials: true
        }
      );
      
      if (response.data?.user) {
        setUser(response.data.user);
        setConnectionStatus('connected');
        return true;
      }
      return false;
    } catch (error) {
      const errorMessage = error instanceof Error ? error : new Error('Failed to refresh token');
      logger.error('Failed to refresh token:', errorMessage);
      // If refresh fails, log the user out
      await handleLogout('Your session has expired. Please log in again.');
      return false;
    }
  }, [handleLogout]);


  // Check authentication status
  const checkAuthStatus = useCallback(async (): Promise<boolean> => {
    try {
      setIsLoading(true);
      
      // Check if we have a valid CSRF token
      const currentCsrfToken = getCsrfToken();
      if (!currentCsrfToken) {
        setConnectionStatus('disconnected');
        return false;
      }
      setCsrfToken(currentCsrfToken);

      try {
        const response = await apiClient.get<{ user: User }>('/auth/me', {
          withCredentials: true
        });
        
        if (response.data?.user) {
          setUser(response.data.user);
          setConnectionStatus('connected');
          return true;
        }
        
        // If no user data, try to refresh the session
        return await refreshToken();
      } catch (err) {
        // Type-safe error handling
        const error = err as any;
        
        // If /auth/me fails with 401, try to refresh token
        if (error?.response?.status === 401) {
          return await refreshToken();
        }
        
        // For other errors, log and return false
        const errorMessage = error instanceof Error ? error : new Error('Authentication check failed');
        logger.error('Auth check error:', errorMessage);
        return false;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error : new Error('Failed to check auth status');
      logger.error('Auth check failed:', errorMessage);
      setConnectionStatus('disconnected');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [getCsrfToken, refreshToken]);


  // Login function
  const handleLogin = useCallback(async (username: string, password: string, overrideRole?: 'user' | 'admin'): Promise<boolean> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Input validation
      if (!username?.trim() || !password) {
        throw new Error('Please enter both username and password');
      }

      // Get CSRF token first if we don't have it
      let currentCsrfToken = csrfToken || getCsrfToken();
      if (!currentCsrfToken) {
        try {
          // Try to get a new CSRF token
          await apiClient.get('/auth/csrf-token', { withCredentials: true });
          currentCsrfToken = getCsrfToken();
          if (currentCsrfToken) {
            setCsrfToken(currentCsrfToken);
          }
        } catch (csrfError) {
          logger.warn('Failed to get CSRF token:', csrfError);
        }
      }

      const response = await apiClient.post<{ user: User }>(
        '/auth/login',
        { 
          username: username.trim(), 
          password,
          ...(overrideRole && { role: overrideRole })
        },
        { 
          skipAuth: true,
          withCredentials: true,
          headers: {
            'X-Requested-With': 'XMLHttpRequest',
            ...(currentCsrfToken && { 'X-XSRF-TOKEN': currentCsrfToken })
          }
        }
      );

      const userData = response?.data?.user;
      if (userData) {
        const userInfo: User = {
          id: userData.id,
          username: userData.username,
          email: userData.email,
          role: userData.role || 'user',
          ...(userData.fullName && { fullName: userData.fullName })
        };

        // Update state
        setUser(userInfo);
        setConnectionStatus('connected');
        
        // Log successful login
        logger.info(`User ${userInfo.username} logged in successfully`);
        
        return true;
      }
      
      return false;
    } catch (err) {
      let errorMessage = 'Login failed. Please try again.';
      
      // Handle different types of errors
      if (err && typeof err === 'object') {
        const error = err as any;
        if (error?.response?.data?.message) {
          errorMessage = error.response.data.message;
        } else if (error.message) {
          errorMessage = error.message;
        }
      } else if (typeof err === 'string') {
        errorMessage = err;
      }
      
      logger.error(`Login error: ${errorMessage}`);
      setError(errorMessage);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [csrfToken, getCsrfToken]);

  const handleRegister = useCallback(async (_username: string, _email: string, _password: string): Promise<boolean> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Your registration logic here
      // Example:
      // const response = await apiClient.post('/auth/register', { username, email, password });
      // return response.status === 201;
      return true;
    } catch (error) {
      const errorMessage = error instanceof Error ? error : new Error('Registration failed');
      setError(errorMessage.message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRetryConnection = useCallback(async (): Promise<void> => {
    try {
      await checkAuthStatus();
    } catch (error) {
      const errorMessage = error instanceof Error ? error : new Error('Connection error');
      setError(errorMessage.message);
      throw errorMessage;
    }
  }, [checkAuthStatus]);

  const handleClearError = useCallback((): void => {
    setError(null);
  }, []);

  const contextValue = useMemo<AuthContextType>(() => ({
    user,
    isAuthenticated: !!user,
    isLoading,
    error,
    connectionStatus,
    login: handleLogin,
    register: handleRegister,
    logout: handleLogout,
    refreshToken,
    checkAuthStatus,
    retryConnection: handleRetryConnection,
    clearError: handleClearError,
    getCsrfToken
  }), [
    user, 
    isLoading, 
    error, 
    connectionStatus, 
    handleLogin, 
    handleRegister, 
    handleLogout, 
    refreshToken, 
    checkAuthStatus, 
    handleRetryConnection, 
    handleClearError,
    getCsrfToken
  ]);

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );

  // Initialize auth state on mount
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        await checkAuthStatus();
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Failed to initialize auth');
        logger.error('Auth initialization error:', error);
        setError(error.message);
      }
    };
    
    initializeAuth();
  }, [checkAuthStatus]);
  

  // Event handlers
  const handleAuthCleared = useCallback((): void => {
    logger.info('Auth cleared event received');
    setUser(null);
    setConnectionStatus('disconnected');
  }, []);

  const handleStorageChange = useCallback((e: StorageEvent): void => {
    if (e.key === 'satyaai_auth_token' && !e.newValue) {
      logger.info('Token removed from storage (other tab)');
      setUser(null);
      setConnectionStatus('disconnected');
      setError('Session expired. Please login again.');
    }
  }, []);


  // Handle CSRF token changes
  useEffect(() => {
    // Check for CSRF token on mount
    const currentCsrfToken = getCsrfToken();
    if (currentCsrfToken && currentCsrfToken !== csrfToken) {
      setCsrfToken(currentCsrfToken);
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        checkAuthStatus().catch(err => {
          const error = err instanceof Error ? err : new Error('Failed to check auth status');
          logger.error('Visibility change auth check failed:', error);
        });
      }
    };

    // Set up event listeners
    window.addEventListener('auth-cleared', handleAuthCleared as EventListener);
    window.addEventListener('storage', handleStorageChange);
    document.addEventListener('visibilitychange', handleVisibilityChange);

    // Initial auth check
    checkAuthStatus().catch(err => {
      logger.error('Initial auth check failed:', err);
    });

    // Clean up event listeners
    return () => {
      window.removeEventListener('auth-cleared', handleAuthCleared as EventListener);
      window.removeEventListener('storage', handleStorageChange);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [checkAuthStatus, handleAuthCleared, getCsrfToken, csrfToken, handleStorageChange]);
}; // End of AuthProvider component
