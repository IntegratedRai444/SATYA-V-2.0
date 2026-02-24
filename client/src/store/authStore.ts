import { create } from 'zustand';
import { User, Session } from '@supabase/supabase-js';
import { supabase } from '@/lib/supabaseSingleton';

interface AuthState {
  user: User | null;
  session: Session | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  initialize: () => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, metadata?: Record<string, unknown>) => Promise<void>;
  signOut: () => Promise<void>;
  clearError: () => void;
}

let authListenerSetup = false;

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  session: null,
  loading: true,
  error: null,

  initialize: async () => {
    // Prevent multiple listeners
    if (authListenerSetup) return;
    authListenerSetup = true;

    try {
      // Get initial session
      const { data: { session } } = await supabase.auth.getSession();
      set({ user: session?.user ?? null, session, loading: false });

      // Set up single auth state listener
      supabase.auth.onAuthStateChange(
        (event, session) => {
          console.log(`[AUTH] State changed: ${event}`);
          set({ 
            user: session?.user ?? null, 
            session, 
            loading: false,
            error: null 
          });
        }
      );
    } catch (error) {
      set({ 
        error: error instanceof Error ? error.message : 'Failed to initialize auth',
        loading: false 
      });
    }
  },

  signIn: async (email: string, password: string) => {
    set({ loading: true, error: null });
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      // Auth state change will update the store
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to sign in' });
    } finally {
      set({ loading: false });
    }
  },

  signUp: async (email: string, password: string, metadata?: Record<string, unknown>) => {
    set({ loading: true, error: null });
    try {
      const { error } = await supabase.auth.signUp({ 
        email, 
        password,
        options: { data: metadata }
      });
      if (error) throw error;
      
      // Profile creation would be implemented here if needed
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to sign up' });
    } finally {
      set({ loading: false });
    }
  },

  signOut: async () => {
    set({ loading: true, error: null });
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      // Auth state change will update the store
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to sign out' });
    } finally {
      set({ loading: false });
    }
  },

  clearError: () => set({ error: null }),
}));

// Initialize auth store on import
const authStore = useAuthStore.getState();
authStore.initialize();
