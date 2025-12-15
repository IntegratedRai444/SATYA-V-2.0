import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

// Create axios instance with base URL
const apiClient: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001/api',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for handling errors
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as any;
    
    // If error is 401 and we haven't tried to refresh yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh the token
        const refreshToken = localStorage.getItem('refresh_token');
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }
        
        const response = await axios.post(
          `${import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001/api'}/auth/refresh-token`,
          { refreshToken }
        );
        
        const { accessToken } = response.data;
        localStorage.setItem('access_token', accessToken);
        
        // Update the authorization header
        originalRequest.headers.Authorization = `Bearer ${accessToken}`;
        
        // Retry the original request
        return apiClient(originalRequest);
      } catch (refreshError) {
        // If refresh fails, log out the user
        console.error('Session expired. Please log in again.');
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

export { apiClient };
