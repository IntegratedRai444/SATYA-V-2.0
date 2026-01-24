import { useQuery } from '@tanstack/react-query';
import dashboardService, { ApiDashboardStats } from '@/lib/api/services/dashboardService';

const fetchDashboardStats = async (): Promise<ApiDashboardStats> => {
  try {
    const response = await dashboardService.getDashboardStats();
    return response;
  } catch (error) {
    console.error('Failed to fetch dashboard stats:', error);
    
    // Check if it's a network error (service not running)
    if (error instanceof Error && (
      error.message.includes('Network Error') ||
      error.message.includes('ERR_CONNECTION_REFUSED') ||
      error.message.includes('ERR_CONNECTION_RESET')
    )) {
      console.warn('Backend services not available - using mock data');
    }
    
    // Return enhanced default data to prevent crashes
    return {
      totalAnalyses: 0,
      authenticMedia: 0,
      manipulatedMedia: 0,
      uncertainScans: 0,
      recentActivity: []
    };
  }
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