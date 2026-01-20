import axios from 'axios';

// Create a dedicated axios instance for CSRF requests
const csrfClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5001/api/v2',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
});

let csrfToken: string | null = null;
let tokenPromise: Promise<string> | null = null;

interface CsrfTokenResponse {
  token: string;
  expiresIn?: number;
}

/**
 * Fetches a new CSRF token from the server
 */
export const fetchCsrfToken = async (): Promise<string> => {
  try {
    const response = await csrfClient.get<{ data: CsrfTokenResponse }>('/auth/csrf-token');
    
    if (!response.data?.data?.token) {
      throw new Error('Invalid CSRF token response from server');
    }
    
    const { token, expiresIn } = response.data.data;
    csrfToken = token;
    
    // If the token has an expiration, set a timeout to refresh it before it expires
    if (expiresIn) {
      const refreshTime = Math.max(1000, expiresIn * 1000 - 60000); // Refresh 1 minute before expiry
      setTimeout(() => {
        if (csrfToken) {  // Only refresh if we still have a token
          fetchCsrfToken().catch(console.error);
        }
      }, refreshTime);
    }
    
    return csrfToken || ''; // Ensure we always return a string
  } catch (error) {
    console.error('Failed to fetch CSRF token:', error);
    // Clear the token on error to force a new fetch on next attempt
    csrfToken = null;
    throw error;
  }
};

/**
 * Gets the current CSRF token, fetching a new one if needed
 * Uses a single promise to prevent multiple simultaneous requests
 */
export const getCsrfToken = async (): Promise<string> => {
  // Return the token if we already have it
  if (csrfToken) {
    return csrfToken || ''; // Ensure we always return a string
  }
  
  // If we're already fetching a token, return the existing promise
  if (tokenPromise) {
    return tokenPromise;
  }
  
  // Otherwise, fetch a new token
  tokenPromise = fetchCsrfToken()
    .finally(() => {
      // Clear the promise when done
      tokenPromise = null;
    });
  
  return tokenPromise;
};

/**
 * Clears the stored CSRF token (e.g., on logout)
 */
export const clearCsrfToken = (): void => {
  csrfToken = null;
  // Don't clear tokenPromise here as it might be in use by other requests
};

// Export as default for backward compatibility
const csrfService = {
  fetchCsrfToken,
  getCsrfToken,
  clearCsrfToken
};

export default csrfService;
