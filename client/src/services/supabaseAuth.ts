/**
 * Supabase Auth Service - Migration from localStorage
 * This replaces localStorage-based authentication with proper Supabase Auth
 */

import { supabase } from '../lib/supabaseSingleton';

export interface User {
  id: string;
  email: string;
  username: string;
  role: string;
  email_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  user: User;
  message: string;
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
  token_type?: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData extends LoginCredentials {
  username: string;
  fullName?: string;
}

/**
 * Get current authenticated user from Supabase
 */
export const getCurrentUser = async (): Promise<User | null> => {
  try {
    const { data: { user }, error } = await supabase.auth.getUser();
    
    if (error || !user) {
      return null;
    }

    // Transform Supabase user to our User format
    return {
      id: user.id,
      email: user.email || '',
      username: user.user_metadata?.username || user.email?.split('@')[0] || '',
      role: user.user_metadata?.role || 'user',
      email_verified: user.email_confirmed_at != null,
      created_at: user.created_at,
      updated_at: user.updated_at || user.created_at,
    };
  } catch (error) {
    console.error('Error getting current user:', error);
    return null;
  }
};

/**
 * Sign up with email and password
 */
export const register = async (userData: RegisterData): Promise<AuthResponse> => {
  try {
    const { data, error } = await supabase.auth.signUp({
      email: userData.email,
      password: userData.password,
      options: {
        data: {
          username: userData.username,
          full_name: userData.fullName || userData.username,
          role: 'user',
        }
      }
    });

    if (error) {
      throw new Error(error.message);
    }

    if (!data.user || !data.session) {
      throw new Error('Registration failed');
    }

    // Transform to our format
    const user: User = {
      id: data.user.id,
      email: data.user.email || '',
      username: userData.username,
      role: 'user',
      email_verified: false,
      created_at: data.user.created_at,
      updated_at: data.user.created_at,
    };

    return {
      user,
      message: 'Registration successful',
      access_token: data.session.access_token,
      refresh_token: data.session.refresh_token,
      expires_in: data.session.expires_in,
      token_type: data.session.token_type,
    };
  } catch (error) {
    console.error('Registration error:', error);
    throw error;
  }
};

/**
 * Sign in with email and password
 */
export const login = async (credentials: LoginCredentials): Promise<AuthResponse> => {
  try {
    const { data, error } = await supabase.auth.signInWithPassword({
      email: credentials.email,
      password: credentials.password,
    });

    if (error) {
      throw new Error(error.message);
    }

    if (!data.user || !data.session) {
      throw new Error('Login failed');
    }

    // Transform to our format
    const user: User = {
      id: data.user.id,
      email: data.user.email || '',
      username: data.user.user_metadata?.username || data.user.email?.split('@')[0] || '',
      role: data.user.user_metadata?.role || 'user',
      email_verified: data.user.email_confirmed_at != null,
      created_at: data.user.created_at,
      updated_at: data.user.updated_at || data.user.created_at,
    };

    return {
      user,
      message: 'Login successful',
      access_token: data.session.access_token,
      refresh_token: data.session.refresh_token,
      expires_in: data.session.expires_in,
      token_type: data.session.token_type,
    };
  } catch (error) {
    console.error('Login error:', error);
    throw error;
  }
};

/**
 * Sign out
 */
export const logout = async (): Promise<void> => {
  try {
    const { error } = await supabase.auth.signOut();
    
    if (error) {
      console.error('Logout error:', error);
      throw new Error('Logout failed');
    }
  } catch (error) {
    console.error('Logout error:', error);
    throw error;
  }
};

/**
 * Check if user is authenticated
 */
export const isAuthenticated = async (): Promise<boolean> => {
  const user = await getCurrentUser();
  return user !== null;
};

/**
 * Refresh session
 */
export const refreshSession = async (): Promise<AuthResponse> => {
  try {
    const { data, error } = await supabase.auth.refreshSession();

    if (error) {
      throw new Error(error.message);
    }

    if (!data.session) {
      throw new Error('Session refresh failed');
    }

    const user = await getCurrentUser();
    if (!user) {
      throw new Error('User not found after refresh');
    }

    return {
      user,
      message: 'Session refreshed',
      access_token: data.session.access_token,
      refresh_token: data.session.refresh_token,
      expires_in: data.session.expires_in,
      token_type: data.session.token_type,
    };
  } catch (error) {
    console.error('Session refresh error:', error);
    throw error;
  }
};

/**
 * Update user metadata
 */
export const updateUserMetadata = async (metadata: Record<string, any>): Promise<User> => {
  try {
    const { data, error } = await supabase.auth.updateUser({
      data: metadata
    });

    if (error) {
      throw new Error(error.message);
    }

    if (!data.user) {
      throw new Error('User update failed');
    }

    const user: User = {
      id: data.user.id,
      email: data.user.email || '',
      username: data.user.user_metadata?.username || data.user.email?.split('@')[0] || '',
      role: data.user.user_metadata?.role || 'user',
      email_verified: data.user.email_confirmed_at != null,
      created_at: data.user.created_at,
      updated_at: data.user.updated_at || data.user.created_at,
    };

    return user;
  } catch (error) {
    console.error('User update error:', error);
    throw error;
  }
};

/**
 * Reset password
 */
export const resetPassword = async (email: string): Promise<void> => {
  try {
    const { error } = await supabase.auth.resetPasswordForEmail(email);

    if (error) {
      throw new Error(error.message);
    }
  } catch (error) {
    console.error('Password reset error:', error);
    throw error;
  }
};

/**
 * Get current authentication token for WebSocket connections
 */
export const getAuthToken = async (): Promise<string | null> => {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token || null;
  } catch (error) {
    console.error('Error getting auth token:', error);
    return null;
  }
};

/**
 * Listen to auth state changes
 */
export const onAuthStateChange = (callback: (user: User | null) => void) => {
  return supabase.auth.onAuthStateChange(async (event, session) => {
    if (event === 'SIGNED_IN' && session?.user) {
      const user = await getCurrentUser();
      callback(user);
    } else {
      callback(null);
    }
  });
};
