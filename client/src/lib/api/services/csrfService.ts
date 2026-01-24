import axios from 'axios';

// Create a dedicated axios instance for CSRF requests
const csrfClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
});

let csrfToken: string | null = null;
let tokenPromise: Promise<string> | null = null;

/**
 * Fetches a new CSRF token from the server
 * TEMPORARILY DISABLED - CSRF endpoint not implemented
 */
export const fetchCsrfToken = async (): Promise<string> => {
  try {
    // TODO: Re-enable CSRF when endpoint is implemented
    // const response = await csrfClient.get<{ token: string }>('/api/v2/auth/csrf-token');
    // return response.data?.token || '';
    
    // Return placeholder token for now
    return 'csrf-disabled-placeholder';
  } catch (error) {
    console.error('Failed to fetch CSRF token:', error);
    // Clear token on error to force a new fetch on next attempt
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
