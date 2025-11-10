import axios from 'axios';

// Note: Using port 3000 to match the backend server
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3000/api';

// Debug log
console.log('🔧 API Base URL:', API_BASE_URL);

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication token
api.interceptors.request.use(
  (config) => {
    console.log(`🌐 API Request: ${config.method?.toUpperCase()} ${config.url}`, config.params ? { params: config.params } : '');
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`, {
      status: response.status,
      data: response.data
    });
    return response;
  },
  (error) => {
    if (error.response) {
      // Handle specific status codes
      if (error.response.status === 401) {
        // Handle unauthorized access
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
      return Promise.reject({
        message: error.response.data?.message || 'An error occurred',
        status: error.response.status,
        data: error.response.data,
      });
    } else if (error.request) {
      // The request was made but no response was received
      return Promise.reject({
        message: 'No response from server. Please check your connection.',
        status: 0,
      });
    } else {
      // Something happened in setting up the request
      return Promise.reject({
        message: error.message || 'An error occurred',
        status: -1,
      });
    }
  }
);

// API functions
export const dashboardAPI = {
  // Dashboard Stats
  getDashboardStats: async () => {
    const response = await api.get('/dashboard/stats');
    return response.data;
  },

  // File Upload
  uploadFile: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/scan/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Get Scan Results
  getScanResults: async (scanId: string) => {
    const response = await api.get(`/scan/results/${scanId}`);
    return response.data;
  },

  // Get Recent Scans
  getRecentScans: async (limit = 5) => {
    const response = await api.get(`/scans/recent?limit=${limit}`);
    return response.data;
  },

  // Get Scan History
  getScanHistory: async (page = 1, limit = 10) => {
    const response = await api.get(`/scans?page=${page}&limit=${limit}`);
    return response.data;
  },

  // Get Scan Analytics
  getAnalytics: async (period = '30d') => {
    const response = await api.get(`/analytics?period=${period}`);
    return response.data;
  },
};

export default api;
