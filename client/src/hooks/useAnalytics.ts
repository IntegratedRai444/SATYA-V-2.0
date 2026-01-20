import { useCallback } from 'react';
import dashboardService from '@/lib/api/services/dashboardService';
import { useFetch } from './useFetch';

export interface AnalyticsData {
  totalScans: number;
  scansByType: {
    image: number;
    video: number;
    audio: number;
  };
  scansByDate: Array<{
    date: string;
    count: number;
  }>;
  detectionRate: number;
  falsePositiveRate: number;
  avgProcessingTime: number;
  // Add more analytics metrics as needed
}

export function useAnalytics(_timeRange: '7d' | '30d' | '90d' | 'all' = '30d') {
  // Use the new useFetch hook to eliminate duplication
  const { data, isLoading, error, refetch } = useFetch<AnalyticsData>(
    async () => {
      console.log('Fetching analytics data...');
      const response = await dashboardService.getUserAnalytics();
      
      if (!response) {
        throw new Error('No data received from API');
      }

      // Transform API response to AnalyticsData format
      const analyticsData: AnalyticsData = {
        totalScans: response.totalAnalyses || 0,
        scansByType: { image: 0, video: 0, audio: 0 }, // TODO: Calculate real distribution from API
        scansByDate: response.recentActivity ? [
          { date: 'Last 7 Days', count: response.recentActivity.filter((_: any, i: number) => i < 7).length },
          { date: 'Last 30 Days', count: response.recentActivity.length }
        ] : [],
        detectionRate: response.manipulatedMedia && response.totalAnalyses
          ? (response.manipulatedMedia / response.totalAnalyses)
          : 0,
        falsePositiveRate: 0.05, // TODO: Calculate real false positive rate from API
        avgProcessingTime: 2.5, // TODO: Calculate real processing time from API
      };
      
      return analyticsData;
    },
    {
      enabled: true,
      staleTime: 5 * 60 * 1000 // 5 minutes
    }
  );

  const exportData = useCallback(async (format: 'csv' | 'json' = 'json') => {
    try {
      if (!data) return;

      let content: string;
      let mimeType: string;
      let fileExtension: string;

      if (format === 'csv') {
        // Convert data to CSV format
        const headers = Object.keys(data.scansByType).join(',') + '\n';
        const values = Object.values(data.scansByType).join(',');
        content = headers + values;
        mimeType = 'text/csv';
        fileExtension = 'csv';
      } else {
        // Default to JSON
        content = JSON.stringify(data, null, 2);
        mimeType = 'application/json';
        fileExtension = 'json';
      }

      // Create a download link
      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analytics-${new Date().toISOString().split('T')[0]}.${fileExtension}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      return true;
    } catch (err) {
      console.error('Error exporting analytics:', err);
      return false;
    }
  }, [data]);

  return {
    data,
    isLoading,
    error,
    refetch,
    exportData,
  };
}

