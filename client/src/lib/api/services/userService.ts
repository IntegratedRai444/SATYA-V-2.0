import { BaseService } from './baseService';

export interface UserProfile {
  id: string;
  email: string;
  username: string;
  fullName?: string;
  avatar?: string;
  role: string;
  preferences: {
    theme?: 'light' | 'dark' | 'system';
    notifications: {
      email: boolean;
      push: boolean;
    };
    language: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface UpdateProfileData {
  fullName?: string;
  avatar?: File | string;
  preferences?: {
    theme?: 'light' | 'dark' | 'system';
    notifications?: {
      email?: boolean;
      push?: boolean;
    };
    language?: string;
  };
}

export interface ChangePasswordData {
  currentPassword: string;
  newPassword: string;
}

export class UserService extends BaseService {
  constructor() {
    super('/api/v2/user');
  }

  async getProfile(): Promise<UserProfile> {
    return this.get<UserProfile>('/me');
  }

  async updateProfile(data: UpdateProfileData): Promise<UserProfile> {
    const formData = new FormData();

    // Handle file upload if avatar is a File
    if (data.avatar && data.avatar instanceof File) {
      formData.append('avatar', data.avatar);
      delete data.avatar;
    }

    // Add other fields to form data
    Object.entries(data).forEach(([key, value]) => {
      if (value !== undefined) {
        if (typeof value === 'object') {
          formData.append(key, JSON.stringify(value));
        } else {
          formData.append(key, value as string);
        }
      }
    });

    return this.patch<UserProfile>(
      '/me',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
  }

  async changePassword(data: ChangePasswordData): Promise<void> {
    await this.post('/change-password', data);
  }

  async deleteAccount(): Promise<void> {
    await this.delete('/me');
  }

  async getUsageStatistics(): Promise<{
    totalAnalyses: number;
    storageUsed: number;
    lastActive: string;
  }> {
    return this.get('/me/usage');
  }

  async getActivityLog(params: {
    limit?: number;
    offset?: number;
    type?: string;
  } = {}): Promise<{ items: any[]; total: number }> {
    return this.get('/me/activity', params);
  }

  async uploadAvatar(file: File): Promise<{ url: string }> {
    const formData = new FormData();
    formData.append('avatar', file);

    return this.post<{ url: string }>(
      '/me/avatar',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
  }
}

export const userService = new UserService();

export default userService;
