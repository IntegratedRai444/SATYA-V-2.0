import { db } from '../db';
import { tasks, type Task, type InsertTask } from '@shared/schema';
import { eq, and, desc } from 'drizzle-orm';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../config';

/**
 * Task Manager Service
 * Handles task creation, status tracking, and lifecycle management
 */

export interface FileMetadata {
  name: string;
  size: number;
  type: string;
  path: string;
}

export interface TaskFilters {
  status?: string;
  type?: string;
  limit?: number;
  offset?: number;
}

export interface ErrorInfo {
  message: string;
  code?: string;
  details?: any;
}

export interface AnalysisResult {
  authenticity: string;
  confidence: number;
  detectionDetails: any[];
  metadata?: any;
}

export class TaskManager {
  /**
   * Create a new task
   * @param type - Type of analysis (image, video, audio, webcam, multimodal)
   * @param fileInfo - File metadata
   * @param userId - User ID
   * @returns Created task
   */
  async createTask(
    type: string,
    fileInfo: FileMetadata,
    userId: number
  ): Promise<Task> {
    try {
      const reportCode = this.generateReportCode(type);
      
      const [task] = await db.insert(tasks).values({
        userId,
        type,
        status: 'queued',
        progress: 0,
        fileName: fileInfo.name,
        fileSize: fileInfo.size,
        fileType: fileInfo.type,
        filePath: fileInfo.path,
        reportCode,
        metadata: JSON.stringify({
          originalName: fileInfo.name,
          uploadedAt: new Date().toISOString()
        })
      }).returning();

      return task;
    } catch (error) {
      logger.error('Error creating task', error as Error);
      throw new Error('Failed to create task');
    }
  }

  /**
   * Get task status by ID
   * @param taskId - Task ID
   * @returns Task object
   */
  async getTaskStatus(taskId: number): Promise<Task | null> {
    try {
      const [task] = await db
        .select()
        .from(tasks)
        .where(eq(tasks.id, taskId))
        .limit(1);

      return task || null;
    } catch (error) {
      logger.error('Error getting task status', error as Error);
      throw new Error('Failed to get task status');
    }
  }

  /**
   * Update task progress
   * @param taskId - Task ID
   * @param progress - Progress percentage (0-100)
   * @param message - Optional progress message
   */
  async updateTaskProgress(
    taskId: number,
    progress: number,
    message?: string
  ): Promise<void> {
    try {
      const updateData: any = {
        progress: Math.min(100, Math.max(0, progress)),
        status: progress === 0 ? 'queued' : progress === 100 ? 'completed' : 'processing'
      };

      // Set startedAt if task is starting
      if (progress > 0 && progress < 100) {
        const task = await this.getTaskStatus(taskId);
        if (task && !task.startedAt) {
          updateData.startedAt = new Date();
        }
      }

      // Add message to metadata if provided
      if (message) {
        const task = await this.getTaskStatus(taskId);
        if (task) {
          const metadata = task.metadata ? JSON.parse(task.metadata) : {};
          metadata.lastMessage = message;
          metadata.lastUpdate = new Date().toISOString();
          updateData.metadata = JSON.stringify(metadata);
        }
      }

      await db
        .update(tasks)
        .set(updateData)
        .where(eq(tasks.id, taskId));
    } catch (error) {
      logger.error('Error updating task progress', error as Error);
      throw new Error('Failed to update task progress');
    }
  }

  /**
   * Complete a task with results
   * @param taskId - Task ID
   * @param result - Analysis result
   */
  async completeTask(taskId: number, result: AnalysisResult): Promise<void> {
    try {
      await db
        .update(tasks)
        .set({
          status: 'completed',
          progress: 100,
          result: JSON.stringify(result),
          completedAt: new Date()
        })
        .where(eq(tasks.id, taskId));
    } catch (error) {
      logger.error('Error completing task', error as Error);
      throw new Error('Failed to complete task');
    }
  }

  /**
   * Fail a task with error information
   * @param taskId - Task ID
   * @param error - Error information
   */
  async failTask(taskId: number, error: ErrorInfo): Promise<void> {
    try {
      await db
        .update(tasks)
        .set({
          status: 'failed',
          error: JSON.stringify(error),
          completedAt: new Date()
        })
        .where(eq(tasks.id, taskId));
    } catch (error) {
      logger.error('Error failing task', error as Error);
      throw new Error('Failed to mark task as failed');
    }
  }

  /**
   * Get user tasks with optional filters
   * @param userId - User ID
   * @param filters - Optional filters
   * @returns Array of tasks
   */
  async getUserTasks(userId: number, filters?: TaskFilters): Promise<Task[]> {
    try {
      // Build where conditions
      const conditions = [eq(tasks.userId, userId)];
      
      if (filters?.status) {
        conditions.push(eq(tasks.status, filters.status));
      }
      
      if (filters?.type) {
        conditions.push(eq(tasks.type, filters.type));
      }

      let query = db
        .select()
        .from(tasks)
        .where(and(...conditions))
        .orderBy(desc(tasks.createdAt));

      // Apply pagination
      if (filters?.limit) {
        query = query.limit(filters.limit) as any;
      }

      if (filters?.offset) {
        query = query.offset(filters.offset) as any;
      }

      const userTasks = await query;
      return userTasks;
    } catch (error) {
      logger.error('Error getting user tasks', error as Error);
      throw new Error('Failed to get user tasks');
    }
  }

  /**
   * Generate a unique report code
   * @param type - Analysis type
   * @returns Report code in format SATYA-{TYPE}-{YYYYMMDD}-{SEQUENCE}
   */
  private generateReportCode(type: string): string {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const dateStr = `${year}${month}${day}`;
    
    // Generate a random 4-digit sequence
    const sequence = Math.floor(1000 + Math.random() * 9000);
    
    // Type prefix
    const typePrefix = type.substring(0, 3).toUpperCase();
    
    return `SATYA-${typePrefix}-${dateStr}-${sequence}`;
  }

  /**
   * Get task statistics for a user
   * @param userId - User ID
   * @returns Task statistics
   */
  async getTaskStatistics(userId: number): Promise<{
    total: number;
    queued: number;
    processing: number;
    completed: number;
    failed: number;
  }> {
    try {
      const userTasks = await this.getUserTasks(userId);
      
      return {
        total: userTasks.length,
        queued: userTasks.filter(t => t.status === 'queued').length,
        processing: userTasks.filter(t => t.status === 'processing').length,
        completed: userTasks.filter(t => t.status === 'completed').length,
        failed: userTasks.filter(t => t.status === 'failed').length
      };
    } catch (error) {
      logger.error('Error getting task statistics', error as Error);
      throw new Error('Failed to get task statistics');
    }
  }

  /**
   * Delete old completed tasks (cleanup)
   * @param daysOld - Delete tasks older than this many days
   * @param userId - Optional user ID to limit cleanup to specific user
   */
  async cleanupOldTasks(daysOld: number = 30, userId?: number): Promise<number> {
    try {
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - daysOld);

      // This would need a proper implementation with date comparison
      // For now, just return 0 as placeholder
      return 0;
    } catch (error) {
      logger.error('Error cleaning up old tasks', error as Error);
      throw new Error('Failed to cleanup old tasks');
    }
  }
}

// Export singleton instance
export const taskManager = new TaskManager();
