/**
 * React Hook for Supabase Authentication
 * Replaces localStorage-based auth with proper Supabase Auth
 */

import { useState, useEffect, useCallback } from 'react';
import { getCurrentUser, login, register, logout, isAuthenticated, onAuthStateChange } from '../services/supabaseAuth';
import type { User, LoginCredentials, RegisterData, AuthResponse } from '../services/supabaseAuth';

export const useSupabaseAuth = () => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize auth state
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        setIsLoading(true);
        const currentUser = await getCurrentUser();
        setUser(currentUser);
      } catch (err) {
        console.error('Auth initialization error:', err);
        setError('Failed to initialize authentication');
      } finally {
        setIsLoading(false);
      }
    };

    initializeAuth();

    // Listen to auth state changes
    const { data: { subscription } } = onAuthStateChange((user) => {
      setUser(user);
      setIsLoading(false);
      setError(null);
    });

    return () => {
      subscription?.unsubscribe();
    };
  }, []);

  const handleLogin = useCallback(async (credentials: LoginCredentials): Promise<AuthResponse> => {
    try {
      setError(null);
      setIsLoading(true);
      const response = await login(credentials);
      setUser(response.user);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Login failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRegister = useCallback(async (userData: RegisterData): Promise<AuthResponse> => {
    try {
      setError(null);
      setIsLoading(true);
      const response = await register(userData);
      setUser(response.user);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Registration failed';
      setError(errorMessage);
      throw new Error(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleLogout = useCallback(async (): Promise<void> => {
    try {
      setError(null);
      setIsLoading(true);
      await logout();
      setUser(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Logout failed';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const checkAuth = useCallback(async (): Promise<boolean> => {
    try {
      const isAuth = await isAuthenticated();
      if (!isAuth) {
        setUser(null);
      }
      return isAuth;
    } catch (err) {
      console.error('Auth check error:', err);
      setUser(null);
      return false;
    }
  }, []);

  return {
    user,
    isLoading,
    error,
    isAuthenticated: !!user,
    login: handleLogin,
    register: handleRegister,
    logout: handleLogout,
    checkAuth,
    clearError: () => setError(null),
  };
};
