import { BaseService } from './baseService';

export interface User {
  id: string;
  email: string;
  username: string;
  role: string;
  createdAt: string;
  updatedAt: string;
}

export interface AuthResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
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
    resolve: (token: string) => void;
    reject: (error: Error) => void;
  }> = [];
  private isRefreshing = false;
  private readonly TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before expiry
  private refreshLock: Promise<AuthResponse> | null = null;

  public constructor() {
    super('/auth');
  }

  public static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new (AuthService as any)();
    }
    return AuthService.instance;
  }

  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await this.post<AuthResponse>('/login', credentials);
    this.setAuthTokens(response);
    return response;
  }

  async register(userData: RegisterData): Promise<AuthResponse> {
    const response = await this.post<AuthResponse>('/register', userData);
    this.setAuthTokens(response);
    return response;
  }

  async refreshToken(): Promise<AuthResponse> {
    // If we have a valid token that's not expired, return it
    if (this.tokenExpiry > Date.now() + this.TOKEN_REFRESH_THRESHOLD) {
      const token = localStorage.getItem('access_token');
      if (token) {
        return { 
          user: { 
            id: '', 
            email: '', 
            username: '', 
            role: 'user',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          },
          accessToken: token, 
          refreshToken: '',
          expiresIn: Math.floor((this.tokenExpiry - Date.now()) / 1000) 
        };
      }
    }

    // If refresh is already in progress, return the existing promise
    if (this.isRefreshing && this.refreshLock) {
      return this.refreshLock;
    }

    // Set the refresh lock
    this.isRefreshing = true;
    this.refreshLock = new Promise<AuthResponse>(async (resolve, reject) => {
      try {
        const response = await this.post<AuthResponse>(
          '/refresh',
          {},
          {
            skipAuth: true,
            headers: {
              'X-Refresh-Token': 'true',
              'Content-Type': 'application/json'
            }
          }
        );

        if (!response.accessToken) {
          throw new Error('No access token in refresh response');
        }

        // Update tokens and expiry
        this.lastTokenRefresh = Date.now();
        this.tokenExpiry = this.lastTokenRefresh + (response.expiresIn * 1000);
        localStorage.setItem('access_token', response.accessToken);

        // Notify all subscribers
        this.refreshSubscribers.forEach(({ resolve }) => resolve(response.accessToken));
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

  private setAuthTokens(response: AuthResponse): void {
    if (response.accessToken) {
      this.lastTokenRefresh = Date.now();
      this.tokenExpiry = this.lastTokenRefresh + (response.expiresIn * 1000);
      localStorage.setItem('access_token', response.accessToken);
      
      // Notify all waiting requests
      this.refreshSubscribers.forEach(({ resolve }) => resolve(response.accessToken));
      this.refreshSubscribers = [];
    }
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
