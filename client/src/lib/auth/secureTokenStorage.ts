/**
 * Secure Token Storage Utility
 * Provides encrypted storage for sensitive tokens
 */

// Encryption key (in production, this should come from environment variables)
const ENCRYPTION_KEY = 'satya-ai-token-encryption-key-2024';

/**
 * Simple XOR encryption for token obfuscation
 * Note: This is basic obfuscation, not true encryption
 * For production, use proper encryption libraries
 */
const simpleEncrypt = (text: string): string => {
  const encrypted = [];
  for (let i = 0; i < text.length; i++) {
    const charCode = text.charCodeAt(i) ^ ENCRYPTION_KEY.charCodeAt(i % ENCRYPTION_KEY.length);
    encrypted.push(charCode);
  }
  return btoa(String.fromCharCode(...encrypted));
};

const simpleDecrypt = (encryptedText: string): string => {
  try {
    const decoded = atob(encryptedText);
    const decrypted = [];
    for (let i = 0; i < decoded.length; i++) {
      const charCode = decoded.charCodeAt(i) ^ ENCRYPTION_KEY.charCodeAt(i % ENCRYPTION_KEY.length);
      decrypted.push(charCode);
    }
    return String.fromCharCode(...decrypted);
  } catch {
    return '';
  }
};

/**
 * Securely store access token
 */
export const setSecureAccessToken = (token: string): void => {
  try {
    const encrypted = simpleEncrypt(token);
    sessionStorage.setItem('satya_access_token', encrypted);
  } catch (error) {
    console.error('Failed to store access token securely:', error);
    // Fallback to localStorage if sessionStorage fails
    localStorage.setItem('satya_access_token', token);
  }
};

/**
 * Securely retrieve access token
 */
export const getSecureAccessToken = (): string | null => {
  try {
    const encrypted = sessionStorage.getItem('satya_access_token');
    if (encrypted) {
      return simpleDecrypt(encrypted);
    }
    
    // Fallback to localStorage if sessionStorage doesn't have it
    const fallback = localStorage.getItem('satya_access_token');
    if (fallback) {
      // Migrate to secure storage
      setSecureAccessToken(fallback);
      localStorage.removeItem('satya_access_token');
      return fallback;
    }
    
    return null;
  } catch (error) {
    console.error('Failed to retrieve access token securely:', error);
    // Fallback to unencrypted storage
    return localStorage.getItem('satya_access_token');
  }
};

/**
 * Securely store refresh token
 */
export const setSecureRefreshToken = (token: string): void => {
  try {
    const encrypted = simpleEncrypt(token);
    sessionStorage.setItem('satya_refresh_token', encrypted);
  } catch (error) {
    console.error('Failed to store refresh token securely:', error);
    // Fallback to localStorage if sessionStorage fails
    localStorage.setItem('satya_refresh_token', token);
  }
};

/**
 * Securely retrieve refresh token
 */
export const getSecureRefreshToken = (): string | null => {
  try {
    const encrypted = sessionStorage.getItem('satya_refresh_token');
    if (encrypted) {
      return simpleDecrypt(encrypted);
    }
    
    // Fallback to localStorage if sessionStorage doesn't have it
    const fallback = localStorage.getItem('satya_refresh_token');
    if (fallback) {
      // Migrate to secure storage
      setSecureRefreshToken(fallback);
      localStorage.removeItem('satya_refresh_token');
      return fallback;
    }
    
    return null;
  } catch (error) {
    console.error('Failed to retrieve refresh token securely:', error);
    // Fallback to unencrypted storage
    return localStorage.getItem('satya_refresh_token');
  }
};

/**
 * Clear all stored tokens securely
 */
export const clearSecureTokens = (): void => {
  try {
    sessionStorage.removeItem('satya_access_token');
    sessionStorage.removeItem('satya_refresh_token');
  } catch (error) {
    console.error('Failed to clear tokens from sessionStorage:', error);
  }
  
  // Always clear from fallback storage
  localStorage.removeItem('satya_access_token');
  localStorage.removeItem('satya_refresh_token');
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
};

/**
 * Check if tokens are stored securely
 */
export const hasSecureTokens = (): boolean => {
  return !!getSecureAccessToken() && !!getSecureRefreshToken();
};
