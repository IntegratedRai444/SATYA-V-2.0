import { useQuery } from '@tanstack/react-query';

interface DashboardStats {
  analyzedMedia: { count: number; growth: string };
  detectedDeepfakes: { count: number; growth: string };
  avgDetectionTime: { time: string; improvement: string };
  detectionAccuracy: { percentage: number; improvement: string };
  dailyActivity: Array<{
    date: string;
    analyses: number;
    deepfakes: number;
  }>;
}

const fetchDashboardStats = async (): Promise<DashboardStats> => {
  const response = await fetch('/api/dashboard/stats');
  if (!response.ok) {
    throw new Error('Failed to fetch dashboard stats');
  }
  return response.json();
};

export const useDashboardStats = () => {
  return useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: fetchDashboardStats,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 30 * 1000, // 30 seconds
    placeholderData: {
      analyzedMedia: { count: 147, growth: '+23%' },
      detectedDeepfakes: { count: 36, growth: '+12%' },
      avgDetectionTime: { time: '4.2s', improvement: '-18%' },
      detectionAccuracy: { percentage: 96, improvement: '+3%' },
      dailyActivity: [
        { date: '01/01', analyses: 12, deepfakes: 2 },
        { date: '01/02', analyses: 8, deepfakes: 1 },
        { date: '01/03', analyses: 15, deepfakes: 3 },
        { date: '01/04', analyses: 22, deepfakes: 4 },
        { date: '01/05', analyses: 18, deepfakes: 2 },
        { date: '01/06', analyses: 25, deepfakes: 5 },
        { date: '01/07', analyses: 30, deepfakes: 6 },
        { date: '01/08', analyses: 28, deepfakes: 4 },
        { date: '01/09', analyses: 35, deepfakes: 7 },
        { date: '01/10', analyses: 32, deepfakes: 5 },
        { date: '01/11', analyses: 40, deepfakes: 8 },
        { date: '01/12', analyses: 38, deepfakes: 6 },
        { date: '01/13', analyses: 45, deepfakes: 9 },
        { date: '01/14', analyses: 42, deepfakes: 7 },
        { date: '01/15', analyses: 48, deepfakes: 10 },
        { date: '01/16', analyses: 52, deepfakes: 11 },
        { date: '01/17', analyses: 55, deepfakes: 12 },
        { date: '01/18', analyses: 58, deepfakes: 13 },
        { date: '01/19', analyses: 62, deepfakes: 14 },
        { date: '01/20', analyses: 65, deepfakes: 15 },
      ]
    }
  });
};