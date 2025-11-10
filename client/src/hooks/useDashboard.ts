import { useState } from 'react';

type TimeRange = '7d' | '30d' | '90d';
type AnalysisType = 'all' | 'images' | 'videos' | 'audio';

export const useDashboard = () => {
  const [timeRange, setTimeRange] = useState<TimeRange>('30d');
  const [analysisType, setAnalysisType] = useState<AnalysisType>('all');

  return {
    timeRange,
    setTimeRange,
    analysisType,
    setAnalysisType,
  };
};
