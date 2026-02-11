import { supabase } from "@/lib/supabaseSingleton"

// Token caching to prevent repeated Supabase calls
let cachedToken: string | null = null;
let tokenExpiry: number = 0;
const TOKEN_BUFFER = 5 * 60 * 1000; // 5 minutes buffer before expiry

export async function getAccessToken(): Promise<string | null> {
  const now = Date.now();
  
  // Return cached token if still valid
  if (cachedToken && now < tokenExpiry) {
    return cachedToken;
  }
  
  try {
    // Fetch fresh token from Supabase
    const { data } = await supabase.auth.getSession();
    
    if (data.session?.access_token) {
      const expiresAt = data.session.expires_at ? new Date(data.session.expires_at).getTime() : 0;
      const expiryTime = expiresAt - TOKEN_BUFFER; // Refresh 5 minutes before expiry
      
      cachedToken = data.session.access_token;
      tokenExpiry = expiryTime;
      
      return cachedToken;
    }
    
    // No valid session
    cachedToken = null;
    tokenExpiry = 0;
    return null;
  } catch (error) {
    console.error('Error getting access token:', error);
    cachedToken = null;
    tokenExpiry = 0;
    return null;
  }
}

// Clear token cache on logout or session change
export function clearTokenCache(): void {
  cachedToken = null;
  tokenExpiry = 0;
}

export async function getCurrentUser(): Promise<{ id: string; email: string } | null> {
  const { data } = await supabase.auth.getUser()
  return data.user ? { id: data.user.id, email: data.user.email || '' } : null
}
