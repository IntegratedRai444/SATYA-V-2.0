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
    super(''); // Use empty base path since Vite proxy handles /api routing
  }

  async getDashboardStats(): Promise<ApiDashboardStats> {
    return this.get<ApiDashboardStats>('/api/v2/dashboard/stats');
  }

  async getUserAnalytics(): Promise<unknown> {
    return this.get<unknown>('/api/v2/dashboard/analytics');
  }

  async getAnalysisHistory(params?: { limit?: number }): Promise<unknown> {
    return this.get<unknown>('/api/v2/history', params);
  }
}

export default new DashboardService();
