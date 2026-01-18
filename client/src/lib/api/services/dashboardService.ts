import { BaseService } from './baseService';

export interface DashboardStats {
  totalAnalyses: number;
  authenticMedia: number;
  manipulatedMedia: number;
  uncertainScans: number;
  recentActivity: Array<{
    id: string;
    type: string;
    date: string;
    status: string;
  }>;
}

export interface ApiDashboardStats {
  totalAnalyses: number;
  authenticMedia: number;
  manipulatedMedia: number;
  uncertainScans: number;
  recentActivity: Array<{
    id: string;
    type: string;
    date: string;
    status: string;
  }>;
}

class DashboardService extends BaseService {
  constructor() {
    super(''); // Use empty base path since we're using direct routes
  }

  async getDashboardStats(): Promise<ApiDashboardStats> {
    return this.get<ApiDashboardStats>('/dashboard/stats');
  }

  async getUserAnalytics(): Promise<any> {
    return this.get<any>('/analytics/user');
  }

  async getAnalysisHistory(params?: { limit?: number }): Promise<any> {
    return this.get<any>('/analytics/history', params);
  }
}

export default new DashboardService();
