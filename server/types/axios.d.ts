import 'axios';

declare module 'axios' {
  export interface AxiosRequestConfig {
    /**
     * Number of times the request has been retried
     */
    retryCount?: number;
    
    /**
     * Whether the request is a retry
     */
    _retry?: boolean;
    
    /**
     * Request timeout in milliseconds
     */
    timeout?: number;
  }
}

export {};
