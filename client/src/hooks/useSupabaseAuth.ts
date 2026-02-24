import { useEffect } from 'react';
import { User, Session } from '@supabase/supabase-js';
import { useAuthStore } from '../store/authStore';

export interface UseSupabaseAuthReturn {
  user: User | null;
  session: Session | null;
  loading: boolean;
  error: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, metadata?: Record<string, unknown>) => Promise<void>;
  signOut: () => Promise<void>;
}

export const useSupabaseAuth = (): UseSupabaseAuthReturn => {
  // Use global auth store instead of creating multiple listeners
  const authState = useAuthStore();

  // Ensure auth is initialized (only runs once globally)
  useEffect(() => {
    // The store handles initialization automatically on import
    // No additional listeners needed
  }, []);

  return {
    user: authState.user,
    session: authState.session,
    loading: authState.loading,
    error: authState.error,
    signIn: authState.signIn,
    signUp: authState.signUp,
    signOut: authState.signOut,
  };
};
