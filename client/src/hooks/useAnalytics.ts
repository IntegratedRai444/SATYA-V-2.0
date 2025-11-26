import { useCallback } from 'react';
import api from '@/lib/api';
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

export function useAnalytics(timeRange: '7d' | '30d' | '90d' | 'all' = '30d') {
  // Use the new useFetch hook to eliminate duplication
  const { data, isLoading, error, refetch } = useFetch<AnalyticsData>(
    async () => {
      const response = await api.getDashboardStats();

      if (!response.success) {
        throw new Error(response.error || 'Failed to fetch analytics');
      }

      const apiData = response.data;

      if (!apiData) {
        throw new Error('No data received from API');
      }

      // Transform API response to AnalyticsData format
      const analyticsData: AnalyticsData = {
        totalScans: apiData.totalScans || 0,
        scansByType: apiData.scansByType || { image: 0, video: 0, audio: 0 },
        scansByDate: apiData.recentActivity ? [
          { date: 'Last 7 Days', count: apiData.recentActivity.last7Days || 0 },
          { date: 'Last 30 Days', count: apiData.recentActivity.last30Days || 0 },
          { date: 'This Month', count: apiData.recentActivity.thisMonth || 0 }
        ] : [],
        detectionRate: apiData.manipulatedScans && apiData.totalScans
          ? (apiData.manipulatedScans / apiData.totalScans)
          : 0,
        falsePositiveRate: 0.05, // Default value
        avgProcessingTime: 2.5, // Default value
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

