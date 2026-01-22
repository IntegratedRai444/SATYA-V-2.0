import { BaseService } from './baseService';

export interface User {
  id: string;
  email: string;
  username: string;
  role: string;
  email_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  user: User;
  message: string;
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
  token_type?: string;
}

const TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before token expires

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData extends LoginCredentials {
  username: string;
  fullName?: string;
}

export class AuthService extends BaseService {
  private static instance: AuthService;
  private lastTokenRefresh: number = 0;
  private tokenExpiry: number = 0;
  private refreshSubscribers: Array<{
    resolve: () => void;
    reject: (error: Error) => void;
  }> = [];
  private isRefreshing = false;
  private refreshLock: Promise<AuthResponse> | null = null;

  public constructor() {
    super('/auth');
  }

  public static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await this.post<AuthResponse>('/login', credentials, {
      withCredentials: true,
      skipAuth: true
    });
    return response;
  }

  async register(userData: RegisterData): Promise<AuthResponse> {
    const response = await this.post<AuthResponse>('/register', userData, {
      withCredentials: true,
      skipAuth: true
    });
    return response;
  }

  async refreshToken(): Promise<AuthResponse> {
    // If refresh is already in progress, return the existing promise
    if (this.isRefreshing && this.refreshLock) {
      return this.refreshLock;
    }

    // Set the refresh lock
    this.isRefreshing = true;
    this.refreshLock = new Promise<AuthResponse>((resolve, reject) => {
      (async () => {
        try {
          const response = await this.post<AuthResponse>(
            '/refresh',
            {},
            {
              skipAuth: true,
              withCredentials: true,
              headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-Refresh-Token': 'true',
                'Content-Type': 'application/json'
              }
            }
          );

          if (!response.access_token) {
            throw new Error('No access token in refresh response');
          }

          // Update tokens and expiry
          this.lastTokenRefresh = Date.now();
          if (response.expires_in) {
            this.tokenExpiry = this.lastTokenRefresh + (response.expires_in * 1000);
          }

          // Notify all subscribers
          this.refreshSubscribers.forEach(({ resolve }) => resolve());
          this.refreshSubscribers = [];

          resolve(response);
        } catch (error) {
          // Notify all subscribers of the error
          this.refreshSubscribers.forEach(({ reject: rej }) => rej(error as Error));
          this.refreshSubscribers = [];
          reject(error);
        } finally {
          // Clear the refresh lock
          this.isRefreshing = false;
          this.refreshLock = null;
        }
      })();
    });

    return this.refreshLock;
  }

  async ensureValidToken(): Promise<boolean> {
    const now = Date.now();
    const timeUntilExpiry = this.tokenExpiry - now;
    
    // If token is expired or about to expire, refresh it
    if (timeUntilExpiry < TOKEN_REFRESH_THRESHOLD) {
      try {
        await this.refreshToken();
        return true;
      } catch (error) {
        return false;
      }
    }
    return true;
  }

  async logout(): Promise<void> {
    try {
      await this.post('/logout', {});
    } finally {
      this.clearAuth();
    }
  }

  async getCurrentUser(): Promise<User> {
    return this.get<User>('/me');
  }

  async requestPasswordReset(email: string): Promise<void> {
    await this.post('/forgot-password', { email });
  }

  async resetPassword(token: string, password: string): Promise<void> {
    await this.post('/reset-password', { token, password });
  }

  isAuthenticated(): boolean {
    // We'll check authentication status via an API call
    // This is a placeholder - the actual implementation will make a request to /me
    return false;
  }

  public clearAuth(): void {
    this.lastTokenRefresh = 0;
    this.tokenExpiry = 0;
    this.isRefreshing = false;
    this.refreshLock = null;
    localStorage.removeItem('access_token');
    this.refreshSubscribers = [];
  }

  // Add this method to check authentication state
  async checkAuth(): Promise<boolean> {
    try {
      await this.getCurrentUser();
      return true;
    } catch (error) {
      return false;
    }
  }
}

export const authService = new AuthService();

export default authService;
