import { useState } from 'react';
import { useSupabaseAuth } from './useSupabaseAuth';

type TimeRange = '7d' | '30d' | '90d';
type AnalysisType = 'all' | 'images' | 'videos' | 'audio';

export const useDashboard = () => {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [analysisType, setAnalysisType] = useState<AnalysisType>('all');
  
  // Get auth state to prevent 401 race
  const { session } = useSupabaseAuth();
  
  return {
    timeRange,
    setTimeRange,
    analysisType,
    setAnalysisType,
    // Add guard to prevent early API calls without auth
    isAuthenticated: !!session?.access_token,
    session
  };
};
