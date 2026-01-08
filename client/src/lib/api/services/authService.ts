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
  constructor() {
    super('/auth');
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
    try {
      // The refresh token is automatically sent via httpOnly cookie
      const response = await this.post<AuthResponse>('/refresh-token', {});
      return response;
    } catch (error) {
      this.clearAuth();
      throw error;
    }
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

  private setAuthTokens(_authData: AuthResponse): void {
    // Tokens are now handled by httpOnly cookies from the server
    // No client-side storage needed
  }

  private clearAuth(): void {
    // Clear auth state on the client
    // Cookies will be cleared by the server's response
  }
}

export const authService = new AuthService();

export default authService;
