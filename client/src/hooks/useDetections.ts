import { useState, useCallback, useEffect } from 'react';
import { Detection } from '@/types/dashboard';

export interface DetectionFilters {
  status?: 'completed' | 'processing' | 'failed';
  type?: 'image' | 'video' | 'audio';
  startDate?: string;
  endDate?: string;
  query?: string;
  sortBy?: 'newest' | 'oldest' | 'confidence';
  page?: number;
  limit?: number;
}

export interface DetectionResult {
  items: Detection[];
  total: number;
  page: number;
  limit: number;
  totalPages: number;
}

export function useDetections(initialFilters: DetectionFilters = {}) {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [filters, setFilters] = useState<DetectionFilters>({
    page: 1,
    limit: 10,
    sortBy: 'newest',
    ...initialFilters,
  });
  const [pagination, setPagination] = useState({
    total: 0,
    totalPages: 1,
    currentPage: 1,
  });

  const fetchDetections = useCallback(async (customFilters?: Partial<DetectionFilters>) => {
    try {
      setLoading(true);
      setError(null);
      
      const appliedFilters = { ...filters, ...customFilters };
      const queryParams = new URLSearchParams();
      
      // Add filters to query params
      Object.entries(appliedFilters).forEach(([key, value]) => {
        if (value !== undefined && value !== '') {
          queryParams.append(key, String(value));
        }
      });
      
      // Replace with your actual API call
      // const response = await fetch(`/api/detections?${queryParams.toString()}`);
      // if (!response.ok) throw new Error('Failed to fetch detections');
      // const data = await response.json();
      
      // Mock data for now
      const mockData: DetectionResult = {
        items: Array.from({ length: 10 }, (_, i) => ({
          id: `detection-${i}`,
          type: ['image', 'video', 'audio'][Math.floor(Math.random() * 3)] as 'image' | 'video' | 'audio',
          status: ['completed', 'processing', 'failed'][Math.floor(Math.random() * 3)] as 'completed' | 'processing' | 'failed',
          timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
          confidence: Math.round(Math.random() * 1000) / 10,
        })),
        total: 42,
        page: appliedFilters.page || 1,
        limit: appliedFilters.limit || 10,
        totalPages: 5,
      };
      
      setDetections(mockData.items);
      setPagination({
        total: mockData.total,
        totalPages: mockData.totalPages,
        currentPage: mockData.page,
      });
      
      return mockData;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to fetch detections');
      setError(error);
      console.error('Error fetching detections:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [filters]);

  const updateFilters = useCallback((newFilters: Partial<DetectionFilters>) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters,
      ...(newFilters.page === undefined ? { page: 1 } : {}), // Reset to first page when filters change
    }));
  }, []);

  const refresh = useCallback(() => fetchDetections(), [fetchDetections]);

  const deleteDetection = useCallback(async (id: string) => {
    try {
      // Replace with your actual API call to delete a detection
      // const response = await fetch(`/api/detections/${id}`, { method: 'DELETE' });
      // if (!response.ok) throw new Error('Failed to delete detection');
      
      // Refresh the list after deletion
      await refresh();
      return true;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to delete detection');
      console.error('Error deleting detection:', error);
      throw error;
    }
  }, [refresh]);

  // Initial fetch
  useEffect(() => {
    fetchDetections();
  }, [filters, fetchDetections]);

  return {
    detections,
    loading,
    error,
    filters,
    pagination,
    updateFilters,
    refresh,
    deleteDetection,
  };
}
