import { v4 as uuidv4 } from 'uuid';
import { apiClient, authApiClient } from '../client';

export type RequestOptions = {
  id?: string;
  retryCount?: number;
  timeout?: number;
  skipAuth?: boolean;
  signal?: AbortSignal;
  headers?: Record<string, string>;
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
    const response = await this.client.get<T>(
      `${this.basePath}${endpoint}`,
      {
        ...options,
        params,
        headers: {
          ...options.headers,
          'X-Request-Id': options.id || uuidv4(),
        },
      }
    );
    return response.data;
  }

  protected async post<T = any>(
    endpoint: string = '',
    data: any = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const response = await this.client.post<T>(
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

  protected async put<T = any>(
    endpoint: string = '',
    data: any = {},
    options: RequestOptions = {}
  ): Promise<T> {
    const response = await this.client.put<T>(
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

  protected async delete<T = any>(
    endpoint: string = '',
    options: RequestOptions = {}
  ): Promise<T> {
    const response = await this.client.delete<T>(
      `${this.basePath}${endpoint}`,
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
