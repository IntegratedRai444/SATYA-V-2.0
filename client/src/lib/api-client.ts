import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError, AxiosProgressEvent } from 'axios';

// Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  statusCode?: number;
  [key: string]: any;
}

interface RequestOptions extends AxiosRequestConfig {
  onUploadProgress?: (progressEvent: AxiosProgressEvent) => void;
  onDownloadProgress?: (progressEvent: AxiosProgressEvent) => void;
}

class ApiClient {
  private client: AxiosInstance;
  private authToken: string | null = null;

  constructor(baseURL: string = '/api') {
    this.client = axios.create({
      baseURL,
      timeout: 30000, // 30 seconds
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      withCredentials: true,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response) {
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          console.error('API Error:', {
            status: error.response.status,
            statusText: error.response.statusText,
            data: error.response.data,
            config: error.config,
          });
        } else if (error.request) {
          // The request was made but no response was received
          console.error('No response received:', error.request);
        } else {
          // Something happened in setting up the request that triggered an Error
          console.error('Request setup error:', error.message);
        }
        return Promise.reject(error);
      }
    );
  }

  // Set auth token
  public setAuthToken(token: string | null): void {
    this.authToken = token;
    if (token) {
      localStorage.setItem('auth_token', token);
    } else {
      localStorage.removeItem('auth_token');
    }
  }

  // Get auth token
  public getAuthToken(): string | null {
    return this.authToken || localStorage.getItem('auth_token');
  }

  // Check if user is authenticated
  public isAuthenticated(): boolean {
    return !!this.getAuthToken();
  }

  // Generic request method
  public async request<T = any>(
    method: 'get' | 'post' | 'put' | 'delete' | 'patch',
    url: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.request({
        method,
        url,
        data,
        ...options,
      });

      return {
        success: true,
        data: response.data,
        statusCode: response.status,
      };
    } catch (error) {
      if (axios.isAxiosError(error)) {
        return {
          success: false,
          error: error.response?.data?.message || error.message,
          statusCode: error.response?.status,
          ...(error.response?.data || {}),
        };
      }
      
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        success: false,
        error: errorMessage,
      };
    }
  }

  // Convenience methods
  public get<T = any>(url: string, params?: any, options: RequestOptions = {}) {
    return this.request<T>('get', url, undefined, { params, ...options });
  }

  public post<T = any>(url: string, data?: any, options: RequestOptions = {}) {
    return this.request<T>('post', url, data, options);
  }

  public put<T = any>(url: string, data?: any, options: RequestOptions = {}) {
    return this.request<T>('put', url, data, options);
  }

  public delete<T = any>(url: string, data?: any, options: RequestOptions = {}) {
    return this.request<T>('delete', url, data, options);
  }

  public patch<T = any>(url: string, data?: any, options: RequestOptions = {}) {
    return this.request<T>('patch', url, data, options);
  }

  // File upload with progress
  public async uploadFile<T = any>(
    url: string,
    file: File,
    fieldName: string = 'file',
    onProgress?: (progress: number) => void,
    additionalData: Record<string, any> = {}
  ): Promise<ApiResponse<T>> {
    const formData = new FormData();
    formData.append(fieldName, file);

    // Append additional data to form data
    Object.entries(additionalData).forEach(([key, value]) => {
      if (value !== undefined) {
        formData.append(key, value);
      }
    });

    return this.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
  }
}

// Create a singleton instance
const apiClient = new ApiClient(import.meta.env.VITE_API_URL || '/api');

export default apiClient;
