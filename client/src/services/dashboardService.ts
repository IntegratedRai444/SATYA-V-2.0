import { z } from 'zod';
import apiClient from '../lib/api';
import type { DashboardStats, Detection } from '../types/dashboard';
import { toast } from '../components/ui/use-toast';
import type { ApiResponse } from '../lib/api';

// Zod schemas for runtime validation
const StatItemSchema = z.object({
  title: z.string(),
  value: z.union([z.string(), z.number()]),
  change: z.string(),
  trend: z.enum(['up', 'down', 'neutral']),
  icon: z.any().optional(),
});

const AnalysisDataSchema = z.object({
  name: z.string(),
  value: z.number(),
});

const ActivityItemSchema = z.object({
  id: z.union([z.string(), z.number()]),
  user: z.string(),
  action: z.string(),
  time: z.string(),
});

export interface StatItem extends z.infer<typeof StatItemSchema> {}
export interface AnalysisData extends z.infer<typeof AnalysisDataSchema> {}
export interface ActivityItem extends z.infer<typeof ActivityItemSchema> {}

export interface DashboardData {
  stats: StatItem[];
  monthlyAnalysis: AnalysisData[];
  performanceMetrics: AnalysisData[];
  recentActivity: ActivityItem[];
}

// Type definitions for dashboard data
interface DashboardStatsResponse {
  stats: DashboardStats;
}

interface AnalyticsResponse {
  monthly: AnalysisData[];
  metrics: AnalysisData[];
}

interface ActivityResponse {
  items: Detection[];
  total: number;
}

// Error class for API errors
class DashboardServiceError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public code?: string
  ) {
    super(message);
    this.name = 'DashboardServiceError';
  }
}

// Transform API stats to dashboard stats
const transformStats = (apiStats: DashboardStats): StatItem[] => {
  return [
    {
      title: 'Total Analyses',
      value: apiStats.analyzedMedia.count,
      change: apiStats.analyzedMedia.growth,
      trend: apiStats.analyzedMedia.growthType === 'positive' ? 'up' : 'down',
    },
    {
      title: 'Detected Deepfakes',
      value: apiStats.detectedDeepfakes.count,
      change: apiStats.detectedDeepfakes.growth,
      trend: apiStats.detectedDeepfakes.growthType === 'positive' ? 'up' : 'down',
    },
    {
      title: 'Avg. Detection Time',
      value: apiStats.avgDetectionTime.time,
      change: apiStats.avgDetectionTime.improvement,
      trend: apiStats.avgDetectionTime.improvementType === 'positive' ? 'down' : 'up',
    },
    {
      title: 'Detection Accuracy',
      value: `${apiStats.detectionAccuracy.percentage}%`,
      change: apiStats.detectionAccuracy.improvement,
      trend: apiStats.detectionAccuracy.improvementType === 'positive' ? 'up' : 'down',
    },
  ];
};

// Transform API activity to dashboard activity
const transformActivity = (detections: Detection[]): ActivityItem[] => {
  return detections.slice(0, 5).map((detection) => ({
    id: detection.id,
    user: 'System', // This would come from user data in a real app
    action: `Detection ${detection.status}`,
    time: new Date(detection.timestamp).toLocaleTimeString(),
  }));
};

/**
 * Fetches dashboard data from the API
 * @param params - Query parameters
 * @returns Processed dashboard data
 * @throws {DashboardServiceError} When there's an error fetching or processing data
 */
export const fetchDashboardData = async (params: {
  timeRange: string;
  analysisType: string;
}): Promise<DashboardData> => {
  try {
    type StatsResponse = ApiResponse<{ stats: DashboardStats }>;
    type AnalyticsResponse = ApiResponse<{ 
      monthly: AnalysisData[]; 
      metrics: AnalysisData[] 
    }>;
    type ActivityResponse = ApiResponse<Detection[]>;

    // Fetch all dashboard data in parallel using the apiClient methods
    const [statsRes, analyticsRes, activityRes] = await Promise.all([
      apiClient.getDashboardStats(),
      apiClient.getUserAnalytics(),
      apiClient.getRecentActivity()
    ]);

    // Validate and transform responses
    const stats = statsRes.success ? transformStats(statsRes.data) : [];
    const monthlyAnalysis = analyticsRes.success ? (analyticsRes.data as any)?.monthly || [] : [];
    const performanceMetrics = analyticsRes.success ? (analyticsRes.data as any)?.metrics || [] : [];
    const recentActivity = activityRes.success ? transformActivity((activityRes.data as any)?.items || []) : [];

    // Validate data with Zod schemas
    const validatedData = {
      stats: StatItemSchema.array().parse(stats),
      monthlyAnalysis: AnalysisDataSchema.array().parse(monthlyAnalysis),
      performanceMetrics: AnalysisDataSchema.array().parse(performanceMetrics),
      recentActivity: ActivityItemSchema.array().parse(recentActivity),
    };

    return validatedData;
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('Validation error in dashboard service:', error.errors);
      throw new DashboardServiceError(
        'Data validation failed',
        422,
        'VALIDATION_ERROR'
      );
    } else if (error instanceof DashboardServiceError) {
      throw error;
    } else if (error instanceof Error) {
      // Show error toast to user
      toast({
        title: 'Error',
        description: error.message || 'Failed to load dashboard data',
        variant: 'destructive',
      });

      // Return default empty data structure
      return {
        stats: [],
        monthlyAnalysis: [],
        performanceMetrics: [],
        recentActivity: [],
      };
    } else {
      // Handle non-Error case
      console.error('Unexpected error type:', error);
      throw new DashboardServiceError(
        'An unknown error occurred',
        500,
        'UNKNOWN_ERROR'
      );
    }
  }
};

export default {
  fetchDashboardData,
};
