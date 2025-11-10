import { db } from '../db';
import { users, scans, userPreferences } from '@shared/schema';
import { eq, and, gte, lte, count, avg, desc, asc, sql } from 'drizzle-orm';
import { logger } from '../config';
import { CustomError } from '../middleware/error-handler';

export interface DashboardStats {
  totalScans: number;
  authenticScans: number;
  manipulatedScans: number;
  uncertainScans: number;
  averageConfidence: number;
  scansByType: {
    image: number;
    video: number;
    audio: number;
  };
  recentActivity: {
    last7Days: number;
    last30Days: number;
    thisMonth: number;
  };
  confidenceDistribution: {
    high: number; // 80-100%
    medium: number; // 50-79%
    low: number; // 0-49%
  };
  topFindings: string[];
}

export interface ScanHistoryOptions {
  limit?: number;
  offset?: number;
  type?: 'image' | 'video' | 'audio';
  result?: 'authentic' | 'deepfake' | 'uncertain';
  dateFrom?: Date;
  dateTo?: Date;
  sortBy?: 'createdAt' | 'confidenceScore' | 'filename';
  sortOrder?: 'asc' | 'desc';
}

export interface UserAnalytics {
  user: {
    id: number;
    username: string;
    email?: string;
    memberSince: Date;
    totalScans: number;
  };
  usage: {
    scansThisWeek: number;
    scansThisMonth: number;
    averageScansPerWeek: number;
    mostActiveDay: string;
    preferredFileType: string;
  };
  accuracy: {
    averageConfidence: number;
    highConfidenceScans: number;
    uncertainResults: number;
  };
  trends: {
    weeklyActivity: Array<{ date: string; count: number }>;
    monthlyBreakdown: Array<{ month: string; authentic: number; manipulated: number }>;
  };
}

class DashboardService {
  /**
   * Get comprehensive dashboard statistics for a user
   * @param userId - The ID of the user
   * @throws {CustomError} If user not found or database error occurs
   */
  async getDashboardStats(userId: number): Promise<DashboardStats> {
    if (!userId) {
      throw new CustomError('User ID is required', 400);
    }

    try {
      // Verify user exists
      const user = await db.query.users.findFirst({
        where: eq(users.id, userId),
        columns: { id: true }
      });

      if (!user) {
        throw new CustomError('User not found', 404);
      }

      const now = new Date();
      const thirtyDaysAgo = new Date();
      thirtyDaysAgo.setDate(now.getDate() - 30);

      // Get basic scan statistics
      const [stats] = await db
        .select({
          totalScans: count(),
          authenticScans: count(
            sql`CASE WHEN ${scans.result} = 'authentic' THEN 1 END`
          ),
          manipulatedScans: count(
            sql`CASE WHEN ${scans.result} = 'deepfake' THEN 1 END`
          ),
          uncertainScans: count(
            sql`CASE WHEN ${scans.result} = 'uncertain' THEN 1 END`
          ),
          averageConfidence: avg(scans.confidenceScore).mapWith(Number)
        })
        .from(scans)
        .where(eq(scans.userId, userId));

      // Get scans by type
      const scansByType = await db
        .select({
          type: scans.type,
          count: count()
        })
        .from(scans)
        .where(eq(scans.userId, userId))
        .groupBy(scans.type);

      // Get recent activity
      const [recentActivity] = await db
        .select({
          last7Days: count(
            sql`CASE WHEN ${scans.createdAt} >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END`
          ),
          last30Days: count(
            sql`CASE WHEN ${scans.createdAt} >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END`
          ),
          thisMonth: count(
            sql`CASE WHEN date_trunc('month', ${scans.createdAt}) = date_trunc('month', CURRENT_DATE) THEN 1 END`
          )
        })
        .from(scans)
        .where(and(
          eq(scans.userId, userId),
          gte(scans.createdAt, thirtyDaysAgo)
        ));

      // Get confidence distribution
      const [confidenceDist] = await db
        .select({
          high: count(
            sql`CASE WHEN ${scans.confidenceScore} >= 0.8 THEN 1 END`
          ),
          medium: count(
            sql`CASE WHEN ${scans.confidenceScore} >= 0.5 AND ${scans.confidenceScore} < 0.8 THEN 1 END`
          ),
          low: count(
            sql`CASE WHEN ${scans.confidenceScore} < 0.5 THEN 1 END`
          )
        })
        .from(scans)
        .where(eq(scans.userId, userId));

      if (!stats) {
        throw new Error('Failed to fetch dashboard statistics');
      }

      // Get top findings with error handling
      let topFindings: { keyFindings: any }[] = [];
      try {
        topFindings = await db
          .select({
            keyFindings: scans.keyFindings
          })
          .from(scans)
          .where(and(
            eq(scans.userId, userId),
            sql`${scans.keyFindings} IS NOT NULL`
          ))
          .orderBy(desc(scans.confidenceScore))
          .limit(5);
      } catch (error) {
        console.error('Error fetching top findings:', error);
        // Continue with empty array if we can't get top findings
        topFindings = [];
      }

      // Process and return the data with type safety
      const result: DashboardStats = {
        totalScans: Number(stats.totalScans) || 0,
        authenticScans: Number(stats.authenticScans) || 0,
        manipulatedScans: Number(stats.manipulatedScans) || 0,
        uncertainScans: Number(stats.uncertainScans) || 0,
        averageConfidence: Number(stats.averageConfidence) || 0,
        scansByType: {
          image: Number((stats as any).scansByType?.image) || 0,
          video: Number((stats as any).scansByType?.video) || 0,
          audio: Number((stats as any).scansByType?.audio) || 0,
        },
        recentActivity: {
          last7Days: Number((stats as any).recentActivity?.last7Days) || 0,
          last30Days: Number((stats as any).recentActivity?.last30Days) || 0,
          thisMonth: Number((stats as any).recentActivity?.thisMonth) || 0,
        },
        confidenceDistribution: {
          high: Number((stats as any).confidenceDistribution?.high) || 0,
          medium: Number((stats as any).confidenceDistribution?.medium) || 0,
          low: Number((stats as any).confidenceDistribution?.low) || 0,
        },
        topFindings: Array.from(
          new Set(
            topFindings
              .flatMap(scan => scan.keyFindings || [])
              .filter(Boolean)
              .slice(0, 5)
          )
        )
      };

      return result;
    } catch (error) {
      logger.error('Failed to fetch dashboard stats', {
        error,
        userId
      });
      
      if (error instanceof CustomError) {
        throw error;
      }
      
      throw new CustomError('Failed to fetch dashboard statistics', 500);
    }
  }

  /**
   * Get paginated scan history with filtering options
   * @param userId - The ID of the user
   * @param options - Filtering and pagination options
   * @throws {CustomError} If invalid parameters or database error occurs
   */
  async getScanHistory(userId: number, options: ScanHistoryOptions = {}) {
    if (!userId) {
      throw new CustomError('User ID is required', 400);
    }

    try {
      const {
        limit = 20,
        offset = 0,
        type,
        result,
        dateFrom,
        dateTo,
        sortBy = 'createdAt',
        sortOrder = 'desc'
      } = options;

      // Build where conditions
      const conditions = [eq(scans.userId, userId)];
      
      if (type) {
        conditions.push(eq(scans.type, type));
      }
      
      if (result) {
        conditions.push(eq(scans.result, result));
      }
      
      if (dateFrom) {
        conditions.push(gte(scans.createdAt, dateFrom));
      }
      
      if (dateTo) {
        conditions.push(lte(scans.createdAt, dateTo));
      }

      // Get total count for pagination
      const [totalResult] = await db
        .select({ count: count() })
        .from(scans)
        .where(and(...conditions));

      const total = Number(totalResult?.count || 0);
      const hasMore = offset + limit < total;

      // Get paginated results
      const orderBy = [];
      if (sortBy === 'createdAt') {
        orderBy.push(sortOrder === 'asc' ? asc(scans.createdAt) : desc(scans.createdAt));
      } else if (sortBy === 'confidenceScore') {
        orderBy.push(sortOrder === 'asc' ? asc(scans.confidenceScore) : desc(scans.confidenceScore));
      } else if (sortBy === 'filename') {
        orderBy.push(sortOrder === 'asc' ? asc(scans.filename) : desc(scans.filename));
      }

      const scanResults = await db
        .select()
        .from(scans)
        .where(and(...conditions))
        .orderBy(...orderBy)
        .limit(limit)
        .offset(offset);

      return {
        scans: scanResults.map(scan => ({
          id: scan.id,
          filename: scan.filename,
          type: scan.type,
          result: scan.result,
          confidenceScore: scan.confidenceScore,
          createdAt: scan.createdAt,
          processingTime: scan.processingTime || 0 // Calculate processing time if needed
        })),
        pagination: {
          total,
          limit,
          offset,
          hasMore
        }
      };
    } catch (error) {
      logger.error('Failed to fetch scan history', {
        error,
        userId,
        options
      });
      
      if (error instanceof CustomError) {
        throw error;
      }
      
      throw new CustomError('Failed to fetch scan history', 500);
    }
  }

  /**
   * Get system-wide statistics (admin only)
   * @throws {CustomError} If database error occurs
   */
  async getSystemStats() {
    try {
      logger.debug('Fetching system stats');

      // Get total users
      const totalUsersResult = await db.select({ count: count() }).from(users);
      const totalUsers = totalUsersResult[0]?.count || 0;

      // Get all scans
      const allScans = await db.select().from(scans);
      const totalScans = allScans.length;

      // Calculate time-based metrics
      const now = new Date();
      const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const weekStart = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

      const scannerId = 'default'; // Default scanner ID
      const scansToday = allScans.filter(scan => scan.createdAt >= todayStart).length;
      const scansThisWeek = allScans.filter(scan => scan.createdAt >= weekStart).length;

      // Calculate average confidence
      const averageConfidence = totalScans > 0
        ? Math.round(allScans.reduce((sum, scan) => sum + scan.confidenceScore, 0) / totalScans)
        : 0;

      // Get top users by scan count
      const userScanCounts = new Map<number, number>();
      allScans.forEach(scan => {
        if (scan.userId) {
          userScanCounts.set(scan.userId, (userScanCounts.get(scan.userId) || 0) + 1);
        }
      });

      const topUserIds = Array.from(userScanCounts.entries())
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([userId]) => userId);

      const topUsersData = await db
        .select({ id: users.id, username: users.username })
        .from(users)
        .where(eq(users.id, topUserIds[0] || 0)); // This would need to be improved for multiple IDs

      const topUsers = topUsersData.map(user => ({
        username: user.username,
        scanCount: userScanCounts.get(user.id) || 0
      }));

      // Scans by type
      const scansByType = {
        image: allScans.filter(scan => scan.type === 'image').length,
        video: allScans.filter(scan => scan.type === 'video').length,
        audio: allScans.filter(scan => scan.type === 'audio').length
      };

      // Accuracy metrics
      const authenticCount = allScans.filter(scan => scan.result === 'authentic').length;
      const manipulatedCount = allScans.filter(scan => scan.result === 'deepfake').length;
      const uncertainCount = totalScans - authenticCount - manipulatedCount;

      const accuracyMetrics = {
        authenticRate: totalScans > 0 ? Math.round((authenticCount / totalScans) * 100) : 0,
        manipulatedRate: totalScans > 0 ? Math.round((manipulatedCount / totalScans) * 100) : 0,
        uncertainRate: totalScans > 0 ? Math.round((uncertainCount / totalScans) * 100) : 0
      };

      return {
        totalUsers,
        totalScans,
        scansToday,
        scansThisWeek,
        averageConfidence,
        topUsers,
        scansByType,
        accuracyMetrics
      };

    } catch (error) {
      logger.error('Failed to get system stats', {
        error: (error as Error).message
      });
      throw error;
    }
  }

  /**
   * Extract top findings from scan results
   */
  /**
   * Calculate number of active days for a user
   */
  private async calculateActiveDays(userId: number, since: Date): Promise<number> {
    const result = await db
      .select({
        days: sql<number>`COUNT(DISTINCT DATE(${scans.createdAt}))`
      })
      .from(scans)
      .where(and(
        eq(scans.userId, userId),
        gte(scans.createdAt, since)
      ));

    return Number(result[0]?.days || 0);
  }

  /**
   * Extract and deduplicate top findings from scan results
   */
  private extractTopFindings(scans: any[]): string[] {
    if (!scans || !Array.isArray(scans)) {
      return [];
    }

    const findings = new Map<string, number>();
    
    scans.forEach(scan => {
      const findingsData = scan.detectionDetails ? JSON.parse(scan.detectionDetails) : {};
      if (findingsData.keyFindings && Array.isArray(findingsData.keyFindings)) {
        findingsData.keyFindings.forEach((finding: string) => {
          if (typeof finding === 'string' && finding.trim()) {
            findings.set(finding, (findings.get(finding) || 0) + 1);
          }
        });
      }
    });

    return Array.from(findings.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([finding]) => finding);
  }

  /**
   * Calculate activity by day of week
   */
  private calculateDayActivity(scans: Array<{ createdAt?: string | Date }>): Record<string, number> {
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const dayCounts = days.reduce<Record<string, number>>((acc, day) => {
      acc[day] = 0;
      return acc;
    }, {});

    scans.forEach(scan => {
      if (scan.createdAt) {
        const day = new Date(scan.createdAt).getDay();
        dayCounts[days[day]]++;
      }
    });

    return dayCounts;
  }

  /**
   * Generate weekly activity data
   */
  private generateWeeklyActivity(scans: Array<{ createdAt?: string | Date }>): Array<{ date: string; count: number }> {
    const now = new Date();
    const weekAgo = new Date(now);
    weekAgo.setDate(now.getDate() - 7);

    const dateMap = new Map<string, number>();
    
    // Initialize with 0 counts for each day
    for (let i = 0; i < 7; i++) {
      const date = new Date(weekAgo);
      date.setDate(weekAgo.getDate() + i);
      const dateStr = date.toISOString().split('T')[0];
      dateMap.set(dateStr, 0);
    }

    // Count scans per day
    scans.forEach(scan => {
      if (scan.createdAt) {
        const scanDate = new Date(scan.createdAt);
        if (scanDate >= weekAgo) {
          const dateStr = scanDate.toISOString().split('T')[0];
          dateMap.set(dateStr, (dateMap.get(dateStr) || 0) + 1);
        }
      }
    });

    // Convert to array of objects
    return Array.from(dateMap.entries())
      .map(([date, count]) => ({
        date,
        count
      }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }

  /**
   * Generate monthly breakdown data
   */
  private generateMonthlyBreakdown(
    scans: Array<{ 
      createdAt?: string | Date;
      result?: 'authentic' | 'deepfake' | 'uncertain';
    }>
  ): Array<{ month: string; authentic: number; manipulated: number }> {
    const monthMap = new Map<string, { authentic: number; manipulated: number }>();
    
    scans.forEach(scan => {
      if (scan.createdAt) {
        const date = new Date(scan.createdAt);
        const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
        
        if (!monthMap.has(monthKey)) {
          monthMap.set(monthKey, { authentic: 0, manipulated: 0 });
        }
        
        const monthData = monthMap.get(monthKey)!;
        
        if (scan.result === 'authentic') {
          monthData.authentic++;
        } else if (scan.result === 'deepfake') {
          monthData.manipulated++;
        }
      }
    });
    
    // Convert to array and sort by date
    return Array.from(monthMap.entries())
      .map(([month, counts]) => ({
        month: new Date(`${month}-01`).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
        ...counts
      }))
      .sort((a, b) => new Date(a.month).getTime() - new Date(b.month).getTime());
  }
}

// Export singleton instance
export const dashboardService = new DashboardService();