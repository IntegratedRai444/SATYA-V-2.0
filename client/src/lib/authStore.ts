/**
 * Centralized Auth Store
 * Single source of truth for authentication state
 * Prevents auth storms and multiple listeners
 */

import { create } from 'zustand';
import { Session, User } from '@supabase/supabase-js';

export interface AuthState {
  session: Session | null;
  user: User | null;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
}

export interface AuthStore extends AuthState {
  setSession: (session: Session | null) => void;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  signOut: () => void;
}

// Create auth store with singleton pattern
export const useAuthStore = create<AuthStore>((set) => ({
  // Initial state
  session: null,
  user: null,
  loading: false,
  error: null,
  isAuthenticated: false,

  // Actions
  setSession: (session) => {
    set((state) => ({
      ...state,
      session,
      user: session?.user ?? null,
      isAuthenticated: !!session?.user,
      error: null,
    }));
  },

  setUser: (user) => {
    set((state) => ({
      ...state,
      user,
      isAuthenticated: !!user,
      error: null,
    }));
  },

  setLoading: (loading) => {
    set((state) => ({
      ...state,
      loading,
    }));
  },

  setError: (error) => {
    set((state) => ({
      ...state,
      error,
      loading: false,
    }));
  },

  clearError: () => {
    set((state) => ({
      ...state,
      error: null,
    }));
  },

  signOut: () => {
    set({
      session: null,
      user: null,
      loading: false,
      error: null,
      isAuthenticated: false,
    });
  },
}));

// Selectors for common use cases
export const useAuth = () => {
  const store = useAuthStore();
  
  return {
    // State
    session: store.session,
    user: store.user,
    loading: store.loading,
    error: store.error,
    isAuthenticated: store.isAuthenticated,
    
    // Actions
    setSession: store.setSession,
    setUser: store.setUser,
    setLoading: store.setLoading,
    setError: store.setError,
    clearError: store.clearError,
    signOut: store.signOut,
  };
};

// Global selector for session access (used by interceptors)
export const getSession = () => useAuthStore.getState().session;
export const getUser = () => useAuthStore.getState().user;
export const isAuthenticated = () => useAuthStore.getState().isAuthenticated;

// Subscribe to auth state changes (for components that need real-time updates)
export const subscribeToAuth = (callback: (state: AuthState) => void) => {
  return useAuthStore.subscribe(callback);
};
