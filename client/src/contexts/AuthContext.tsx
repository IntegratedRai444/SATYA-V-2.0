import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import apiClient from '../lib/api';

export interface User {
  id: string;
  username: string;
  email?: string;
  role: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  connectionStatus: 'connected' | 'disconnected' | 'checking';
  login: (username: string, password: string) => Promise<boolean>;
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
      console.log('🚀 SatyaAI Frontend Starting...');
      apiClient.runDiagnostics().then(() => {
        checkAuthStatus();
      });
    } else {
      checkAuthStatus();
    }
  }, []);

  const checkAuthStatus = async () => {
    try {
      setIsLoading(true);
      setConnectionStatus('checking');
      setError(null);
      
      console.log('🔍 Starting authentication check...');
      
      // First, check if backend is available
      const healthCheck = await apiClient.checkBackendHealth();
      
      if (!healthCheck.connected) {
        console.warn('⚠️ Backend not available, checking for demo mode...');
        setConnectionStatus('disconnected');
        
        // In development, enable demo mode even if backend is down
        if (import.meta.env.DEV || import.meta.env.VITE_BYPASS_AUTH === 'true') {
          console.log('✅ Development mode: Enabling demo user (backend offline)');
          const demoUser = {
            id: '1',
            username: 'demo',
            email: 'demo@satyaai.com',
            role: 'user'
          };
          
          apiClient.setAuthToken('demo-token-12345');
          setUser(demoUser);
          setError('Running in demo mode - backend server unavailable');
          setIsLoading(false);
          return;
        } else {
          setError(`Backend server unavailable: ${healthCheck.error}`);
          setIsLoading(false);
          return;
        }
      }
      
      setConnectionStatus('connected');
      console.log('✅ Backend is available');
      
      // Auto-login demo user in development or when bypass is enabled
      if (import.meta.env.DEV || import.meta.env.VITE_BYPASS_AUTH === 'true') {
        console.log('✅ Development mode: Auto-logging in demo user');
        const demoUser = {
          id: '1',
          username: 'demo',
          email: 'demo@satyaai.com',
          role: 'user'
        };
        
        apiClient.setAuthToken('demo-token-12345');
        setUser(demoUser);
        setIsLoading(false);
        return;
      }
      
      // Check existing authentication
      if (!apiClient.isAuthenticated()) {
        console.log('ℹ️ No existing authentication found');
        setIsLoading(false);
        return;
      }

      console.log('🔍 Validating existing session...');
      const sessionResult = await apiClient.validateSession();
      
      if (sessionResult.valid && sessionResult.user) {
        console.log('✅ Session valid, user authenticated');
        setUser({
          id: sessionResult.user.id?.toString() || '',
          username: sessionResult.user.username || '',
          email: sessionResult.user.email,
          role: sessionResult.user.role || 'user'
        });
      } else {
        console.log('❌ Session invalid, clearing auth');
        apiClient.clearAuth();
        setUser(null);
      }
    } catch (error: any) {
      console.error('❌ Auth check failed:', error);
      setConnectionStatus('disconnected');
      
      // Provide specific error messages
      if (error.code === 'ECONNREFUSED') {
        setError('Cannot connect to SatyaAI server. Please ensure the backend is running.');
      } else if (error.name === 'TimeoutError') {
        setError('Connection timeout. The server may be overloaded.');
      } else {
        setError(`Authentication failed: ${error.message}`);
      }
      
      apiClient.clearAuth();
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  };

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      setError(null);
      setIsLoading(true);

      const response = await apiClient.login(username, password);
      
      if (response.success && response.user) {
        setUser({
          id: response.user.id?.toString() || '',
          username: response.user.username || username,
          email: response.user.email,
          role: response.user.role || 'user'
        });
        return true;
      } else {
        setError(response.message || 'Login failed');
        return false;
      }
    } catch (error: any) {
      setError(error.message || 'Login failed');
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
      
      if (response.success && response.user) {
        setUser({
          id: response.user.id?.toString() || '',
          username: response.user.username || username,
          email: response.user.email || email,
          role: response.user.role || 'user'
        });
        return true;
      } else {
        const errorMessage = response.errors ? response.errors.join(', ') : response.message;
        setError(errorMessage || 'Registration failed');
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
    } catch (error) {
      console.warn('Logout request failed:', error);
    } finally {
      setUser(null);
      apiClient.clearAuth();
    }
  };

  const clearError = () => {
    setError(null);
  };

  const retryConnection = async () => {
    console.log('🔄 Retrying connection...');
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