import api from '@/lib/api';
import logger from '@/lib/logger';

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'editor' | 'viewer';
  lastActive?: string;
  avatar?: string;
  permissions: string[];
}

export interface TeamMember extends User {
  joinDate: string;
  status: 'active' | 'pending' | 'suspended';
}

class UserService {
  private currentUser: User | null = null;
  private teamMembers: TeamMember[] = [];

  public async getCurrentUser(): Promise<User> {
    if (this.currentUser) return this.currentUser;

    try {
      const response = await api.get('/profile');
      if (response.success && response.data) {
        this.currentUser = {
          id: response.data.id.toString(),
          email: response.data.email,
          name: response.data.username,
          role: response.data.role || 'viewer',
          permissions: response.data.permissions || [],
          lastActive: response.data.lastActive
        };
        return this.currentUser;
      }
      throw new Error(response.error || 'Failed to fetch user profile');
    } catch (error) {
      logger.error('Failed to fetch current user', error as Error);
      throw error;
    }
  }

  public async getTeamMembers(): Promise<TeamMember[]> {
    try {
      const response = await api.get('/api/team/members');
      this.teamMembers = response.data;
      return this.teamMembers;
    } catch (error) {
      logger.error('Failed to fetch team members', error as Error);
      return [];
    }
  }

  public async inviteUser(email: string, role: User['role']): Promise<void> {
    try {
      await api.post('/api/team/invite', { email, role });
      logger.info('User invited successfully', { email, role });
    } catch (error) {
      logger.error('Failed to invite user', error as Error);
      throw error;
    }
  }

  public async updateUserRole(userId: string, role: User['role']): Promise<void> {
    try {
      await api.put(`/api/team/members/${userId}/role`, { role });
      logger.info('User role updated successfully', { userId, role });

      // Update local cache
      const user = this.teamMembers.find(m => m.id === userId);
      if (user) user.role = role;
    } catch (error) {
      logger.error('Failed to update user role', error as Error);
      throw error;
    }
  }

  public async removeTeamMember(userId: string): Promise<void> {
    try {
      await api.delete(`/api/team/members/${userId}`);
      logger.info('Team member removed successfully', { userId });

      // Update local cache
      this.teamMembers = this.teamMembers.filter(m => m.id !== userId);
    } catch (error) {
      logger.error('Failed to remove team member', error as Error);
      throw error;
    }
  }

  public hasPermission(permission: string): boolean {
    if (!this.currentUser) return false;
    return this.currentUser.permissions.includes(permission) ||
      this.currentUser.role === 'admin';
  }
}

export const userService = new UserService();
