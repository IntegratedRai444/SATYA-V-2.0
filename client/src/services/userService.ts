import apiClient from '@/lib/api';

// Extend the Window interface to include the apiClient
declare global {
  interface Window {
    apiClient: typeof apiClient;
  }
}

// Make apiClient available globally for debugging
if (process.env.NODE_ENV === 'development') {
  window.apiClient = apiClient;
}

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
      const response = await apiClient.getProfile();
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
      console.error('Failed to fetch current user:', error);
      throw error;
    }
  }

  public async getTeamMembers(): Promise<TeamMember[]> {
    try {
      // In a real app, you would have a dedicated endpoint for team members
      // For now, we'll return the current user as a placeholder
      const currentUser = await this.getCurrentUser();
      return [{
        ...currentUser,
        joinDate: new Date().toISOString(),
        status: 'active' as const
      }];
    } catch (error) {
      console.error('Failed to fetch team members:', error);
      return [];
    }
  }

  public async inviteUser(email: string, role: User['role']): Promise<void> {
    try {
      // In a real app, you would call the invite endpoint
      console.log(`Inviting ${email} with role ${role}`);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
    } catch (error) {
      console.error('Failed to invite user:', error);
      throw error;
    }
  }

  public async updateUserRole(userId: string, role: User['role']): Promise<void> {
    try {
      // In a real app, you would call the update role endpoint
      console.log(`Updating user ${userId} role to ${role}`);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Update local cache
      const user = this.teamMembers.find(m => m.id === userId);
      if (user) user.role = role;
    } catch (error) {
      console.error('Failed to update user role:', error);
      throw error;
    }
  }

  public async removeTeamMember(userId: string): Promise<void> {
    try {
      // In a real app, you would call the delete endpoint
      console.log(`Removing team member ${userId}`);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Update local cache
      this.teamMembers = this.teamMembers.filter(m => m.id !== userId);
    } catch (error) {
      console.error('Failed to remove team member:', error);
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
