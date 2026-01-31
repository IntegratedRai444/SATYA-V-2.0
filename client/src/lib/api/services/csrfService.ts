let csrfToken: string | null = null;
let tokenPromise: Promise<string> | null = null;

/**
 * Fetches a new CSRF token from the server
 * TEMPORARILY DISABLED - CSRF endpoint not implemented
 */
export const fetchCsrfToken = async (): Promise<string> => {
  // TODO: Re-enable CSRF when endpoint is implemented
  return 'csrf-placeholder';
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
