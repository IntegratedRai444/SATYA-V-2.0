import { useState, useCallback, useEffect } from 'react';

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
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchAnalytics = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Replace with your actual API call to fetch analytics data
      // const response = await fetch(`/api/analytics?range=${timeRange}`);
      // if (!response.ok) throw new Error('Failed to fetch analytics');
      // const result = await response.json();
      // setData(result.data);
      
      // Mock data for now
      const mockData: AnalyticsData = {
        totalScans: 1245,
        scansByType: {
          image: 850,
          video: 320,
          audio: 75,
        },
        scansByDate: Array.from({ length: 30 }, (_, i) => ({
          date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
          count: Math.floor(Math.random() * 50) + 10,
        })),
        detectionRate: 0.87,
        falsePositiveRate: 0.05,
        avgProcessingTime: 2.5,
      };
      
      setData(mockData);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch analytics'));
      console.error('Error fetching analytics:', err);
    } finally {
      setIsLoading(false);
    }
  }, [timeRange]);

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

  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchAnalytics,
    exportData,
  };
}
