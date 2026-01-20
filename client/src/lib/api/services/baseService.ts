import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';
import { getCsrfToken } from './csrfService';

// Create axios instances
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5001/api/v2',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
});

const authApiClient = axios.create({
  baseURL: import.meta.env.VITE_AUTH_API_URL || '/auth',
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
  },
});

export type RequestHeaders = {
  'X-Request-Id'?: string;
  'Content-Type'?: string;
  'X-CSRF-Token'?: string;
  [key: string]: string | undefined;
};

export type RequestOptions = {
  id?: string;
  retryCount?: number;
  timeout?: number;
  skipAuth?: boolean;
  signal?: AbortSignal;
  headers?: RequestHeaders;
  withCredentials?: boolean;
  skipCSRF?: boolean;
  [key: string]: any; // For any additional properties
};

export class BaseService {
  protected basePath: string;
  protected isAuthService: boolean;
  private client: typeof apiClient;

  constructor(basePath: string, isAuthService: boolean = false) {
    this.basePath = basePath;
    this.isAuthService = isAuthService;
    this.client = isAuthService ? authApiClient : apiClient;
  }

  protected async get<T = any>(
    endpoint: string = '',
    params: any = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const headers: RequestHeaders = {
      ...options.headers,
      'X-Request-Id': options.id || uuidv4(),
    };

    // Add CSRF token for non-GET requests or when explicitly requested
    if (!options.skipCSRF && !options.skipAuth) {
      const csrfToken = await getCsrfToken();
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
      }
    }

    const response = await this.client.get<T>(
      `${this.basePath}${endpoint}`,
      {
        ...options,
        params,
        headers,
        withCredentials: options.withCredentials ?? true,
      }
    );
    return response.data;
  }

  protected async post<T = any>(
    endpoint: string = '',
    data: any = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const headers = {
      ...options.headers,
      'X-Request-Id': options.id || uuidv4(),
      'Content-Type': 'application/json',
    };

    // Add CSRF token for non-GET requests or when explicitly requested
    if (!options.skipCSRF && !options.skipAuth) {
      const csrfToken = await getCsrfToken();
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
      }
    }

    const response = await this.client.post<T>(
      `${this.basePath}${endpoint}`,
      data,
      {
        ...options,
        headers,
        withCredentials: options.withCredentials ?? true,
      }
    );
    return response.data;
  }

  protected async put<T = any>(
    endpoint: string = '',
    data: any = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const headers = {
      ...options.headers,
      'X-Request-Id': options.id || uuidv4(),
      'Content-Type': 'application/json',
    };

    // Add CSRF token for non-GET requests or when explicitly requested
    if (!options.skipCSRF && !options.skipAuth) {
      const csrfToken = await getCsrfToken();
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
      }
    }

    const response = await this.client.put<T>(
      `${this.basePath}${endpoint}`,
      data,
      {
        ...options,
        headers,
        withCredentials: options.withCredentials ?? true,
      }
    );
    return response.data;
  }

  protected async delete<T = any>(
    endpoint: string = '',
    options: RequestOptions = {}
  ): Promise<T> {
    const headers: RequestHeaders = {
      ...options.headers,
      'X-Request-Id': options.id || uuidv4(),
    };

    // Add CSRF token for non-GET requests or when explicitly requested
    if (!options.skipCSRF && !options.skipAuth) {
      const csrfToken = await getCsrfToken();
      if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
      }
    }

    const response = await this.client.delete<T>(
      `${this.basePath}${endpoint}`,
      {
        ...options,
        headers,
        withCredentials: options.withCredentials ?? true,
      }
    );
    return response.data;
  }

  protected async patch<T = any>(
    endpoint: string = '',
    data: any = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const response = await this.client.patch<T>(
      `${this.basePath}${endpoint}`,
      data,
      {
        ...options,
        headers: {
          ...options.headers,
          'X-Request-Id': options.id || uuidv4(),
        },
      }
    );
    return response.data;
  }

  protected createAbortController(): AbortController {
    return new AbortController();
  }

  protected generateRequestId(): string {
    return uuidv4();
  }
}
