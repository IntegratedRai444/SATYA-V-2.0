import { z } from 'zod';
import apiClient, { DashboardStats as ApiDashboardStats } from '../lib/api';
import logger from '../lib/logger';
import type { Detection } from '../types/dashboard';
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

export interface StatItem extends z.infer<typeof StatItemSchema> { }
export interface AnalysisData extends z.infer<typeof AnalysisDataSchema> { }
export interface ActivityItem extends z.infer<typeof ActivityItemSchema> { }

export interface DashboardData {
  stats: StatItem[];
  monthlyAnalysis: AnalysisData[];
  performanceMetrics: AnalysisData[];
  recentActivity: ActivityItem[];
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
const transformStats = (apiStats: ApiDashboardStats): StatItem[] => {
  return [
    {
      title: 'Total Analyses',
      value: apiStats.totalScans,
      change: '+0%', // Placeholder as API doesn't provide growth yet
      trend: 'neutral',
    },
    {
      title: 'Detected Deepfakes',
      value: apiStats.manipulatedScans,
      change: '+0%',
      trend: 'neutral',
    },
    {
      title: 'Avg. Confidence',
      value: `${Math.round(apiStats.averageConfidence * 100)}%`,
      change: '+0%',
      trend: 'neutral',
    },
    {
      title: 'Uncertain Scans',
      value: apiStats.uncertainScans,
      change: '+0%',
      trend: 'neutral',
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
export const fetchDashboardData = async (_params: {
  timeRange: string;
  analysisType: string;
}): Promise<DashboardData> => {

  try {
    // Define expected API response types for clarity and type safety
    type StatsApiResponse = ApiResponse<ApiDashboardStats>;
    type AnalyticsApiResponse = ApiResponse<{
      monthly: AnalysisData[];
      metrics: AnalysisData[];
    }>;
    type ActivityApiResponse = ApiResponse<{ history: Detection[] }>;

    // Fetch all dashboard data in parallel using the apiClient methods
    const [statsRes, analyticsRes, activityRes] = await Promise.all([
      apiClient.getDashboardStats() as Promise<StatsApiResponse>,
      apiClient.getUserAnalytics() as Promise<AnalyticsApiResponse>,
      apiClient.getAnalysisHistory({ limit: 5 }) as Promise<ActivityApiResponse>
    ]);

    // Validate and transform responses
    const stats = statsRes.success && statsRes.data ? transformStats(statsRes.data) : [];
    const monthlyAnalysis = analyticsRes.success ? (analyticsRes.data as any)?.monthly || [] : [];
    const performanceMetrics = analyticsRes.success ? (analyticsRes.data as any)?.metrics || [] : [];
    const recentActivity = activityRes.success ? transformActivity((activityRes.data as any)?.history || []) : [];

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
      logger.error('Validation error in dashboard service', new Error(JSON.stringify(error.errors)));
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
      logger.error('Unexpected error type', new Error(String(error)));
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
