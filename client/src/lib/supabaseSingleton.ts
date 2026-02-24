/**
 * Supabase Singleton Module
 * Ensures only one instance of Supabase client exists across the entire application
 */

// Console filter to suppress Supabase debug noise
if (typeof window !== 'undefined') {
  const originalLog = console.log;
  console.log = (...args: any[]) => {
    const msg = args?.[0]?.toString() ?? '';
    // Filter out common Supabase noise
    if (msg.includes('GoTrueClient')) return;
    if (msg.includes('_acquireLock')) return;
    if (msg.includes('auto refresh token')) return;
    if (msg.includes('refreshSession')) return;
    if (msg.includes('No session')) return;
    originalLog(...args);
  };
}

import { createClient } from '@supabase/supabase-js';
import type { Database } from '@/types/supabase.types';

// Extend Window interface for our custom property
declare global {
  interface Window {
    supabaseLogged?: boolean;
    _supabaseClientInstance?: ReturnType<typeof createClient<Database>>;
    _supabaseInitialized?: boolean;
  }
}

// Environment configuration
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

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

// Validate Supabase key format
// Key validation is handled by Supabase client

// Token refresh settings
const TOKEN_REFRESH_MARGIN = 5 * 60 * 1000; // 5 minutes before expiry
let refreshTimeout: ReturnType<typeof setTimeout> | null = null;

// Debouncing to prevent rapid auth calls
let lastAuthCall = 0;
const AUTH_CALL_DEBOUNCE = 1000; // 1 second debounce

// Set up auth state change listener only once
let authListenerSetup = false;
function setupSessionManagement(client: ReturnType<typeof createClient<Database>>) {
  // Prevent multiple listeners
  if (authListenerSetup) {
    console.warn('Auth listener already setup, skipping duplicate initialization');
    return;
  }
  authListenerSetup = true;
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

  // Initialize session refresh on first load
  client.auth.getSession().then(({ data: { session } }) => {
    const now = Date.now();
    if (now - lastAuthCall < AUTH_CALL_DEBOUNCE) {
      console.log('Auth call debounced, skipping duplicate call');
      return session;
    }
    lastAuthCall = now;
    
    if (session?.expires_at) {
      scheduleTokenRefresh(session.expires_at);
    }
  });
}

// Create and export singleton Supabase client
let supabaseClient: ReturnType<typeof createClient<Database>>;

// Prevent multiple initialization in browser environment
if (typeof window !== 'undefined' && window._supabaseInitialized) {
  console.warn('⚠️ Supabase client already initialized, returning existing instance');
}

if (supabaseUrl && supabaseAnonKey) {
  if (typeof window !== 'undefined' && window._supabaseInitialized) {
    console.warn('⚠️ Supabase client already initialized, returning existing instance');
    // Return existing instance if already initialized
    supabaseClient = window._supabaseClientInstance!;
  } else {
    supabaseClient = createClient<Database>(supabaseUrl, supabaseAnonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
        debug: false, // Disable Supabase debug noise
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
    
    // Mark as initialized
    if (typeof window !== 'undefined') {
      window._supabaseInitialized = true;
      window._supabaseClientInstance = supabaseClient;
    }
  }
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
