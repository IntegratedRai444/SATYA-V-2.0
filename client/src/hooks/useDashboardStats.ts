import { useQuery } from '@tanstack/react-query';
import dashboardService, { ApiDashboardStats } from '@/lib/api/services/dashboardService';

const fetchDashboardStats = async (): Promise<ApiDashboardStats> => {
  const response = await dashboardService.getDashboardStats();
  return response;
};

export const useDashboardStats = () => {
  return useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: fetchDashboardStats,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 30 * 1000, // 30 seconds
    placeholderData: {
      totalAnalyses: 147,
      authenticMedia: 111,
      manipulatedMedia: 36,
      uncertainScans: 0,
      recentActivity: [
        { id: '1', type: 'image', date: '01/01', status: 'completed' },
        { id: '2', type: 'video', date: '01/02', status: 'completed' },
        { id: '3', type: 'image', date: '01/03', status: 'completed' },
        { id: '4', type: 'video', date: '01/04', status: 'completed' },
        { id: '5', type: 'image', date: '01/05', status: 'completed' },
      ]
    }
  });
};