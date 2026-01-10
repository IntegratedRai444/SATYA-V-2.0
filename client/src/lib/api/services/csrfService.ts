import { apiClient } from '../client';

let csrfToken: string | null = null;

/**
 * Fetches a new CSRF token from the server
 */
export const fetchCsrfToken = async (): Promise<string> => {
  try {
    const response = await apiClient.get('/auth/csrf-token', { withCredentials: true });
    if (response.data?.token) {
      csrfToken = response.data.token;
      return csrfToken;
    }
    throw new Error('No CSRF token received');
  } catch (error) {
    console.error('Failed to fetch CSRF token:', error);
    throw error;
  }
};

/**
 * Gets the current CSRF token, fetching a new one if needed
 */
export const getCsrfToken = async (): Promise<string> => {
  if (csrfToken) return csrfToken;
  return await fetchCsrfToken();
};

/**
 * Clears the stored CSRF token (e.g., on logout)
 */
export const clearCsrfToken = (): void => {
  csrfToken = null;
};

export default {
  fetchCsrfToken,
  getCsrfToken,
  clearCsrfToken
};
