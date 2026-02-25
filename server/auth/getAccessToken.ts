import { createClient } from '@supabase/supabase-js';
import { logger } from '../config/logger';

// Initialize Supabase client for server-side
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Token caching to prevent repeated Supabase calls
let cachedToken: string | null = null;
let tokenExpiry: number = 0;
const TOKEN_BUFFER = 5 * 60 * 1000; // 5 minutes buffer before expiry

// Validate JWT token format
const isValidJWT = (token: string): boolean => {
  if (!token || typeof token !== 'string') return false;
  
  const parts = token.split('.');
  return parts.length === 3; // JWT should have 3 parts: header.payload.signature
}

export async function getAccessToken(): Promise<string | null> {
  const now = Date.now();
  
  // Return cached token if still valid and properly formatted
  if (cachedToken && now < tokenExpiry && isValidJWT(cachedToken)) {
    return cachedToken;
  }
  
  try {
    // Fetch fresh token from Supabase
    const { data, error } = await supabase.auth.getSession();
    
    if (error) {
      logger.error('Supabase session error:', error);
      clearTokenCache();
      return null;
    }
    
    if (data.session?.access_token) {
      const token = data.session.access_token;
      
      // Validate token format before caching
      if (!isValidJWT(token)) {
        logger.error('Invalid JWT token format received from Supabase');
        clearTokenCache();
        return null;
      }
      
      const expiresAt = data.session.expires_at ? new Date(data.session.expires_at).getTime() : 0;
      const expiryTime = expiresAt - TOKEN_BUFFER; // Refresh 5 minutes before expiry
      
      cachedToken = token;
      tokenExpiry = expiryTime;
      
      return cachedToken;
    }
    
    // No valid session
    clearTokenCache();
    return null;
  } catch (error) {
    logger.error('Error getting access token:', error);
    clearTokenCache();
    return null;
  }
}

// Clear token cache on logout or session change
export function clearTokenCache(): void {
  cachedToken = null;
  tokenExpiry = 0;
}

export async function getCurrentUser(): Promise<{ id: string; email: string } | null> {
  try {
    const { data, error } = await supabase.auth.getUser();
    
    if (error) {
      logger.error('Error getting current user:', error);
      return null;
    }
    
    return data.user ? { id: data.user.id, email: data.user.email || '' } : null;
  } catch (error) {
    logger.error('Unexpected error getting current user:', error);
    return null;
  }
}
