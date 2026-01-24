import { z } from 'zod';
import api, { DashboardStats as ApiDashboardStats } from '../lib/api';
import logger from '../lib/logger';
import type { Detection } from '../types/dashboard';
import type { ApiResponse } from '../lib/api';
import { toast } from '../components/ui/use-toast';

// Define the interface for API DashboardStats to include uncertainScans
interface ExtendedApiDashboardStats extends ApiDashboardStats {
  uncertainScans: number;
}

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
const transformStats = (apiStats: ExtendedApiDashboardStats): StatItem[] => {
  return [
    {
      title: 'Total Analyses',
      value: apiStats.totalScans,
      change: '+0%', // TODO: Calculate real growth from historical data
      trend: 'neutral',
    },
    {
      title: 'Detected Deepfakes',
      value: apiStats.manipulatedScans,
      change: '+0%', // TODO: Calculate real growth from historical data
      trend: 'neutral',
    },
    {
      title: 'Avg. Confidence',
      value: `${Math.round(apiStats.averageConfidence * 100)}%`,
      change: '+0%', // TODO: Calculate real growth from historical data
      trend: 'neutral',
    },
    {
      title: 'Uncertain Scans',
      value: apiStats.uncertainScans,
      change: '+0%', // TODO: Calculate real growth from historical data
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
    // Fetch all dashboard data in parallel using apiClient methods
    const [statsRes, analyticsRes, activityRes] = await Promise.all([
      api.get('/api/v2/dashboard/stats') as Promise<ApiResponse<ApiDashboardStats>>,
      api.get('/api/v2/dashboard/analytics') as Promise<ApiResponse<any>>,
      api.get('/api/v2/history', { params: { limit: 5 } }) as Promise<ApiResponse<any>>
    ]);

    // Validate and transform responses
    const stats = statsRes.success && statsRes.data ? transformStats(statsRes.data as ExtendedApiDashboardStats) : [];
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
      logger.error('Validation error in dashboard service', new Error(JSON.stringify(error.issues)));
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
