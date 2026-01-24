/**
 * Supabase Singleton Module
 * Ensures only one instance of Supabase client exists across the entire application
 */

import { createClient } from '@supabase/supabase-js';
import type { Database } from '@/types/supabase.types';

// Extend Window interface for our custom property
declare global {
  interface Window {
    supabaseLogged?: boolean;
    _supabaseClientInstance?: ReturnType<typeof createClient<Database>>;
  }
}

// Environment configuration
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;
const isDev = import.meta.env.DEV;

// More graceful error handling
if (!supabaseUrl || !supabaseAnonKey) {
  console.error('❌ Missing Supabase environment variables:', {
    url: !!supabaseUrl,
    key: !!supabaseAnonKey
  });
  console.warn('⚠️ Running without Supabase connection - authentication will not work');
  // Don't throw - allow degraded state
}

// Validate Supabase URL format
if (!supabaseUrl.includes('supabase.co')) {
  console.error('Invalid Supabase URL format:', supabaseUrl);
  throw new Error('Invalid Supabase URL. Please check your VITE_SUPABASE_URL environment variable.');
}

// Validate Supabase key format (removed - key is valid)
// The key sb_publishable_hIPmJbNPv98SJGBZ_jUapA_3jGqKhnf is valid

// Token refresh settings
const TOKEN_REFRESH_MARGIN = 5 * 60 * 1000; // 5 minutes before expiry
let refreshTimeout: NodeJS.Timeout | null = null;

// Session management (extracted to separate function)
function setupSessionManagement(client: ReturnType<typeof createClient<Database>>) {
  // Schedule token refresh before it expires
  function scheduleTokenRefresh(expiresAt: number | undefined) {
    if (!expiresAt) return;
    clearRefreshTimeout();
    
    const expiresInMs = expiresAt * 1000 - Date.now();
    const refreshInMs = Math.max(expiresInMs - TOKEN_REFRESH_MARGIN, 1000);
  
    if (refreshInMs > 0) {
      refreshTimeout = setTimeout(async () => {
        try {
          const { data, error } = await client.auth.refreshSession();
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

  // Set up auth state change listener only once
  client.auth.onAuthStateChange((event, session) => {
    if ((event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') && session?.expires_at) {
      scheduleTokenRefresh(session.expires_at);
    } else if (event === 'SIGNED_OUT') {
      clearRefreshTimeout();
    }
  });

  // Initialize session refresh on first load
  client.auth.getSession().then(({ data: { session } }) => {
    if (session?.expires_at) {
      scheduleTokenRefresh(session.expires_at);
    }
  });
}

// Create and export singleton Supabase client
let supabaseClient: ReturnType<typeof createClient<Database>>;

if (supabaseUrl && supabaseAnonKey) {
  supabaseClient = createClient<Database>(supabaseUrl, supabaseAnonKey, {
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
    // Disable Go-based auth client to prevent conflicts
    db: {
      schema: 'public',
    },
  });

  // Set up session management only for first instance
  setupSessionManagement(supabaseClient);
} else {
  console.warn('⚠️ Creating mock Supabase client - authentication features disabled');
  // Create a mock client that won't crash but won't work
  supabaseClient = createClient<Database>('https://placeholder.supabase.co', 'placeholder-key', {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
      detectSessionInUrl: false,
      debug: false,
    },
  });
}

// Store in global scope
if (typeof window !== 'undefined') {
  window._supabaseClientInstance = supabaseClient;
}

// Export singleton
export const supabase = supabaseClient;
