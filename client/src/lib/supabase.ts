import { createClient } from '@supabase/supabase-js';
import type { Database } from '@/types/supabase.types';

// Environment configuration
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
const isDev = import.meta.env.DEV;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

// Token refresh settings
const TOKEN_REFRESH_MARGIN = 5 * 60 * 1000; // 5 minutes before expiry
let refreshTimeout: NodeJS.Timeout | null = null;

// Create and export the Supabase client
export const supabase = createClient<Database>(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    debug: isDev,
  },
  global: {
    headers: {
      'x-application-name': 'satyaai-client',
      'x-client-version': '1.0.0',
    },
  },
});

// Session management

// Schedule token refresh before it expires
function scheduleTokenRefresh(expiresAt: number | undefined) {
  if (!expiresAt) return;
  clearRefreshTimeout();
  
  const expiresInMs = expiresAt * 1000 - Date.now();
  const refreshInMs = Math.max(expiresInMs - TOKEN_REFRESH_MARGIN, 1000);
  
  if (refreshInMs > 0) {
    refreshTimeout = setTimeout(async () => {
      try {
        const { data, error } = await supabase.auth.refreshSession();
        if (error) throw error;
        if (data.session?.expires_at) {
          scheduleTokenRefresh(data.session.expires_at);
        }
      } catch (error) {
        console.error('Error refreshing session:', error);
        // Retry with exponential backoff
        setTimeout(() => scheduleTokenRefresh(expiresAt), 30000);
      }
    }, refreshInMs);
  }
}

// Clear any pending refresh
function clearRefreshTimeout() {
  if (refreshTimeout) {
    clearTimeout(refreshTimeout);
    refreshTimeout = null;
  }
}

// Set up auth state change listener
supabase.auth.onAuthStateChange((event, session) => {
  if ((event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') && session?.expires_at) {
    scheduleTokenRefresh(session.expires_at);
  } else if (event === 'SIGNED_OUT') {
    clearRefreshTimeout();
  }
});

// Initialize session refresh on first load
supabase.auth.getSession().then(({ data: { session } }) => {
  if (session?.expires_at) {
    scheduleTokenRefresh(session.expires_at);
  }
});

// Development logging
if (isDev && typeof window !== 'undefined') {
  supabase.auth.getSession().then(({ data: { session } }) => {
    console.log('Supabase connection test', session ? 'successful' : 'no active session');
  }).catch(console.error);
}

export default supabase;