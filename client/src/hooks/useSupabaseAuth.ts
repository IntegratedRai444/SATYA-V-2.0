import { useState, useEffect } from 'react';
import { User, Session } from '@supabase/supabase-js';
import { supabase } from '../lib/supabaseSingleton';
import { clearTokenCache } from '../lib/auth/getAccessToken';

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
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Listen for auth state changes and clear cache when needed
  useEffect(() => {
    const subscription = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_OUT' || event === 'TOKEN_REFRESHED') {
        clearTokenCache();
      }
      setUser(session?.user ?? null);
      setSession(session);
      setLoading(false);
    });

    return () => {
      const sub = subscription as any;
      if (sub?.subscription) {
        sub.subscription.unsubscribe();
      }
    };
  }, []); // Empty dependency array - run only once

  const signIn = async (email: string, password: string) => {
    try {
      setError(null);
      setLoading(true);
      console.log('Attempting login with:', { email, passwordLength: password.length });
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      // Don't navigate here - let the auth state change handle it
      console.log('Login successful, waiting for auth state update');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to sign in');
    } finally {
      setLoading(false);
    }
  };

  const signUp = async (email: string, password: string, metadata?: Record<string, unknown>) => {
    try {
      setError(null);
      setLoading(true);
      const { data, error } = await supabase.auth.signUp({ 
        email, 
        password,
        options: {
          data: metadata
        }
      });
      
      if (error) throw error;
      
      // Create user profile in database after successful signup
      if (data.user && metadata) {
        try {
          // Profile creation would be implemented here
          // const profileData = {
          //   id: data.user.id,
          //   email: data.user.email || '',
          //   username: (metadata.name as string) || (metadata.full_name as string) || (data.user.email || '').split('@')[0],
          //   full_name: (metadata.full_name as string) || (metadata.name as string),
          //   role: 'user' as const,
          //   is_active: true,
          //   created_at: new Date().toISOString(),
          //   updated_at: new Date().toISOString()
          // };
          // const { error: profileError } = await supabase.from('users').insert(profileData);
        } catch {
          // Error during profile creation setup
        }
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to sign up');
    } finally {
      setLoading(false);
    }
  };

  const signOut = async () => {
    try {
      setError(null);
      setLoading(true);
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to sign out');
    } finally {
      setLoading(false);
    }
  };

  return {
    user,
    session,
    loading,
    error,
    signIn,
    signUp,
    signOut,
  };
};
