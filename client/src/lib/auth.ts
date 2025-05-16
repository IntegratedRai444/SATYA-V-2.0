import axios from 'axios';
import { startPythonServer, login as pythonLogin, logout as pythonLogout, validateSession } from '../../server/python-bridge';

// Authentication functions
export async function login(username: string, password: string) {
  try {
    // Check if server is started
    await startPythonServer();
    
    // Call login function from python-bridge
    const result = await pythonLogin(username, password);
    
    return result;
  } catch (error) {
    console.error('Login error:', error);
    return {
      success: false,
      message: 'Authentication system error'
    };
  }
}

export async function logout() {
  try {
    // Call logout function from python-bridge
    const result = await pythonLogout();
    
    if (result.success) {
      // Clear local storage
      localStorage.removeItem('satyaai_token');
      localStorage.removeItem('satyaai_user');
    }
    
    return result;
  } catch (error) {
    console.error('Logout error:', error);
    return {
      success: false,
      message: 'Logout system error'
    };
  }
}

export async function checkAuth() {
  try {
    // Get token from local storage
    const token = localStorage.getItem('satyaai_token');
    
    if (!token) {
      return {
        isAuthenticated: false,
        user: null
      };
    }
    
    // Validate token with server
    const result = await validateSession(token);
    
    if (!result.valid) {
      // Clear invalid token
      localStorage.removeItem('satyaai_token');
      localStorage.removeItem('satyaai_user');
      
      return {
        isAuthenticated: false,
        user: null
      };
    }
    
    return {
      isAuthenticated: true,
      user: result.user
    };
  } catch (error) {
    console.error('Auth check error:', error);
    return {
      isAuthenticated: false,
      user: null,
      error: 'Authentication check failed'
    };
  }
}

// Protected API client (ensures auth token is included)
export const authFetch = async (url: string, options: RequestInit = {}) => {
  const token = localStorage.getItem('satyaai_token');
  
  if (!token) {
    throw new Error('Authentication required');
  }
  
  const headers = {
    ...(options.headers || {}),
    'Authorization': `Bearer ${token}`
  };
  
  const response = await fetch(url, {
    ...options,
    headers
  });
  
  if (response.status === 401) {
    // Clear invalid token
    localStorage.removeItem('satyaai_token');
    localStorage.removeItem('satyaai_user');
    
    throw new Error('Authentication expired');
  }
  
  return response;
};

// Auth-protected Axios instance
export const authAxios = axios.create();

// Add auth token to all requests
authAxios.interceptors.request.use(config => {
  const token = localStorage.getItem('satyaai_token');
  
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  return config;
});

// Handle 401 responses globally
authAxios.interceptors.response.use(
  response => response,
  error => {
    if (error.response && error.response.status === 401) {
      // Clear invalid token
      localStorage.removeItem('satyaai_token');
      localStorage.removeItem('satyaai_user');
      
      // Redirect to login page
      window.location.href = '/login';
    }
    
    return Promise.reject(error);
  }
);