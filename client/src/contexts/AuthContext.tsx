import { createContext, useContext, useState, useEffect, useCallback, useMemo, ReactNode, FC } from 'react';
import { apiClient } from '../lib/api/apiClient';
import { logout } from '../services/auth';
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

interface AuthResponse {
  success: boolean;
  user: User;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  connectionStatus: ConnectionStatus;
  initialAuthCheckComplete: boolean;
  login: (username: string, password: string, overrideRole?: 'user' | 'admin') => Promise<boolean>;
  register: (name: string, email: string, password: string) => Promise<boolean>;
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
  const [initialAuthCheckComplete, setInitialAuthCheckComplete] = useState(false);
  // Removed unused initialized state

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
      
      // Clear auth state immediately for better UX
      setUser(null);
      setConnectionStatus('disconnected');
      
      // Call the auth service logout
      try {
        await logout();
      } catch (error) {
        // Even if there's an error, we want to proceed with client-side cleanup
        const errorMessage = error instanceof Error ? error.message : 'Unknown error during logout';
        logger.warn('Logout process had issues, but continuing with cleanup', { errorMessage });
      }
      
      // Clear any remaining auth data
      localStorage.removeItem('satya_auth_state');
      sessionStorage.removeItem('authState');
      
      // Notify other tabs
      window.dispatchEvent(new Event('auth-cleared'));
      
      // Build redirect URL with optional message
      const redirectUrl = new URL('/login', window.location.origin);
      if (message) {
        redirectUrl.searchParams.set('message', encodeURIComponent(message));
      }
      
      // Force a hard redirect to ensure all state is cleared
      window.location.href = redirectUrl.toString();
      
      // Prevent any further execution after redirect
      throw new Error('Logout redirecting');
      
    } catch (error) {
      // If we get here, the redirect might have failed
      if (error instanceof Error && error.message !== 'Logout redirecting') {
        const errorMessage = error.message || 'Unknown error during logout';
        logger.error(`Logout error: ${errorMessage}`);
        setError('Failed to log out properly. Please refresh the page.');
        throw new Error(errorMessage);
      }
    } finally {
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
      setIsLoading(false);
    }
  }, [logout]);

  // Refresh token function with retry logic
  const refreshToken = useCallback(async (retryCount = 0): Promise<boolean> => {
    const MAX_RETRIES = 2;
    
    try {
      const response = await apiClient.post<{ 
        user: User;
        accessToken?: string;
        expiresIn?: number;
      }>(
        '/auth/refresh',
        {},
        { 
          skipAuth: true,
          withCredentials: true,
          headers: {
            'X-Refresh-Token': 'true'
          }
        }
      );
      
      if (response?.user) {
        // Update auth state
        setUser(response.user);
        setConnectionStatus('connected');
        
        // Store auth state in session storage for persistence
        try {
          sessionStorage.setItem('authState', JSON.stringify({
            user: response.user,
            timestamp: Date.now()
          }));
        } catch (e) {
          logger.warn('Failed to persist auth state:', e);
        }
        
        return true;
      }
      
      return false;
    } catch (error: unknown) {
      const axiosError = error as { response?: { status: number } };
      
      // If we get a 401 and have retries left, try again after a delay
      if (axiosError?.response?.status === 401 && retryCount < MAX_RETRIES) {
        await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
        return refreshToken(retryCount + 1);
      }
      
      // Clear any existing auth state on failure
      try {
        sessionStorage.removeItem('authState');
      } catch (e) {
        logger.warn('Failed to clear auth state:', e);
      }
      
      // Only show logout message if we're actually logged in
      if (user) {
        await handleLogout('Your session has expired. Please log in again.');
      }
      
      return false;
    }
  }, [handleLogout, user]);


  // Check authentication status
  const checkAuthStatus = useCallback(async (): Promise<boolean> => {
    // Skip if we already have a user and valid token
    if (user && connectionStatus === 'connected') {
      return true;
    }

    // Set initial loading state only if not already loading
    if (!isLoading) {
      setIsLoading(true);
    }
    
    try {
      // First try to get a fresh CSRF token
      try {
        await apiClient.get('/auth/csrf-token', { withCredentials: true });
        const freshCsrfToken = getCsrfToken();
        if (freshCsrfToken) {
          setCsrfToken(freshCsrfToken);
        }
      } catch (csrfError) {
        logger.warn('Failed to get CSRF token:', csrfError);
      }

      // Check existing session
      try {
        const response = await apiClient.get<{ data: AuthResponse }>('/auth/me', {
          withCredentials: true,
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
          }
        });
        
        if (response.data?.user) {
          setUser(response.data.user);
          setConnectionStatus('connected');
          return true;
        }
      } catch (meError: unknown) {
        // Silently handle 401 from /me as we'll try refresh next
        const axiosError = meError as { response?: { status: number } };
        if (axiosError?.response?.status !== 401) {
          logger.warn('Error checking auth status:', meError);
        }
      }
      
      // If /me fails, try to refresh the session
      try {
        return await refreshToken();
      } catch (refreshError: unknown) {
        // Only log if it's not a 401 (which is expected when not logged in)
        const axiosError = refreshError as { response?: { status: number } };
        if (axiosError?.response?.status !== 401) {
          const error = refreshError instanceof Error ? refreshError : new Error(String(refreshError));
          logger.error('Token refresh failed:', error);
        }
        return false;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error during auth check';
      logger.error(`Auth check failed: ${errorMessage}`);
      return false;
    } finally {
      // Only set loading to false if we're the ones who set it to true
      if (isLoading) {
        setIsLoading(false);
      }
    }
  }, [user, connectionStatus, isLoading, getCsrfToken, refreshToken]);


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

      if (response?.user) {
        const userInfo: User = {
          id: response.user.id,
          username: response.user.username,
          email: response.user.email,
          role: response.user.role || 'user',
          ...(response.user.fullName && { fullName: response.user.fullName })
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

  const handleRegister = useCallback(async (name: string, email: string, password: string): Promise<boolean> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Input validation
      if (!name?.trim() || !email?.trim() || !password) {
        throw new Error('Please fill in all required fields');
      }

      // Get CSRF token first if we don't have it
      let currentCsrfToken = csrfToken || getCsrfToken();
      if (!currentCsrfToken) {
        try {
          await apiClient.get('/auth/csrf-token', { withCredentials: true });
          currentCsrfToken = getCsrfToken();
          if (currentCsrfToken) {
            setCsrfToken(currentCsrfToken);
          }
        } catch (csrfError) {
          logger.warn('Failed to get CSRF token:', csrfError);
        }
      }

      // Register user
      const registerResponse = await apiClient.post<{ success: boolean; user: User; message: string }>(
        '/auth/signup',
        { username: name, email, password },
        { 
          skipAuth: true,
          withCredentials: true,
          headers: {
            'X-Requested-With': 'XMLHttpRequest',
            ...(currentCsrfToken && { 'X-XSRF-TOKEN': currentCsrfToken })
          }
        }
      );

      if (registerResponse?.success && registerResponse?.user) {
        // Auto-login after successful registration
        const loginResponse = await apiClient.post<{ user: User }>(
          '/auth/login',
          { username: name, password },
          { 
            skipAuth: true,
            withCredentials: true,
            headers: {
              'X-Requested-With': 'XMLHttpRequest',
              ...(currentCsrfToken && { 'X-XSRF-TOKEN': currentCsrfToken })
            }
          }
        );

        if (loginResponse?.user) {
          setUser(loginResponse.user);
          setConnectionStatus('connected');
          logger.info(`User ${name} registered and logged in successfully`);
          return true;
        }
      }
      
      throw new Error('Registration completed but auto-login failed');
    } catch (err: unknown) {
      let errorMessage = 'Registration failed';
      const error = err as Error & {
        response?: {
          status?: number;
          data?: {
            message?: string;
          };
        };
      };

      if (error?.response?.data?.message) {
        errorMessage = error.response.data.message;

        // Handle specific error cases
        if (error.response.status === 400) {
          if (errorMessage.includes('already registered')) {
            errorMessage = 'An account with this email already exists';
          } else if (errorMessage.includes('validation')) {
            errorMessage = 'Invalid registration data. Please check your input.';
          }
        } else if (error.response.status === 409) {
          errorMessage = 'An account with this email or username already exists';
        }
      } else if (error?.message) {
        errorMessage = error.message;
      }
      
      const registrationError = new Error(errorMessage);
      logger.error(`Registration error: ${errorMessage}`);
      setError(registrationError.message);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [csrfToken, getCsrfToken]);

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
    initialAuthCheckComplete,
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
    initialAuthCheckComplete,
    handleLogin, 
    handleRegister, 
    handleLogout, 
    refreshToken, 
    checkAuthStatus, 
    handleRetryConnection, 
    handleClearError,
    getCsrfToken
  ]);

  // Check auth status on mount and handle session persistence
  useEffect(() => {
    let isMounted = true;
    let retryTimeout: NodeJS.Timeout;

    const initAuth = async () => {
      try {
        // Try to get auth state from session storage first
        try {
          const savedAuth = sessionStorage.getItem('authState');
          if (savedAuth) {
            const { user: savedUser, timestamp } = JSON.parse(savedAuth);
            // Only use saved auth if it's less than 1 hour old
            if (Date.now() - timestamp < 60 * 60 * 1000) {
              setUser(savedUser);
              setConnectionStatus('connected');
              // Still validate with server in background
              checkAuthStatus().catch(() => {
                if (isMounted) {
                  setUser(null);
                  setConnectionStatus('disconnected');
                }
              });
              return;
            }
          }
        } catch (e: unknown) {
          const error = e instanceof Error ? e : new Error(String(e));
          logger.warn('Failed to load auth state from session storage:', error);
        }

        // If no saved auth or it's expired, check with server
        const isAuthenticated = await checkAuthStatus();
        
        if (isMounted) {
          setInitialAuthCheckComplete(true);
          
          // If not authenticated but we have a session cookie, try one more time after a delay
          if (!isAuthenticated && document.cookie.includes('satya_session')) {
            retryTimeout = setTimeout(async () => {
              if (isMounted) {
                await checkAuthStatus();
              }
            }, 1000); // Wait 1 second before retry
          }
        }
      } catch (error) {
        const err = error instanceof Error ? error : new Error(String(error));
        logger.error('Initial auth check failed:', err);
        if (isMounted) {
          setInitialAuthCheckComplete(true);
        }
      }
    };

    // Handle storage events (for cross-tab auth state)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'authState' && e.newValue) {
        try {
          // Parse the auth state but we don't need to use it directly
          // as we'll call initAuth() to handle the state update
          void initAuth();
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error';
          logger.error(`Error handling storage change: ${errorMessage}`);
        }
      }
    };

    // Set up event listeners
    window.addEventListener('storage', handleStorageChange);
    
    // Initialize auth
    void initAuth();
    
    return () => {
      isMounted = false;
      clearTimeout(retryTimeout);
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [checkAuthStatus]);

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
  

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
