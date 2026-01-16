import { dbManager, type TableName } from '../db'; // Database connection manager

type UpdateTaskProgressData = {
  progress: number;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  startedAt?: Date;
  metadata?: string;
};
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
  details?: Record<string, unknown>;
  stack?: string;
}

export interface DetectionDetail {
  type: string;
  confidence: number;
  location?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  metadata?: Record<string, unknown>;
}

export interface AnalysisResult {
  authenticity: 'real' | 'fake' | 'inconclusive';
  confidence: number;
  detectionDetails: DetectionDetail[];
  metadata?: {
    processingTime?: number;
    modelVersion?: string;
    [key: string]: unknown;
  };
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
    userId: string
  ): Promise<Task> {
    try {
      const reportCode = this.generateReportCode(type);
      
      const taskData = {
        user_id: userId,
        type,
        status: 'queued' as const,
        progress: 0,
        file_name: fileInfo.name,
        file_size: fileInfo.size,
        file_type: fileInfo.type,
        file_path: fileInfo.path,
        report_code: reportCode,
        metadata: JSON.stringify({
          originalName: fileInfo.name,
          uploadedAt: new Date().toISOString()
        })
      };
      const task = await dbManager.create('tasks', taskData) as unknown as Task;

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
      const task = await dbManager.findById('tasks', taskId.toString()) as Record<string, any> | null;
      
      if (!task) return null;
      
      // Map the database record to Task type
      return {
        id: task.id,
        createdAt: new Date(task.created_at),
        updatedAt: task.updated_at ? new Date(task.updated_at) : null,
        userId: task.user_id,
        type: task.type,
        status: task.status,
        progress: task.progress,
        fileName: task.file_name,
        fileSize: task.file_size,
        fileType: task.file_type,
        filePath: task.file_path,
        reportCode: task.report_code,
        result: task.result ? JSON.parse(task.result) : null,
        error: task.error ? JSON.parse(task.error) : null,
        startedAt: task.started_at ? new Date(task.started_at) : null,
        completedAt: task.completed_at ? new Date(task.completed_at) : null,
        metadata: task.metadata ? JSON.parse(task.metadata) : {}
      } as Task;
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
      const updateData: UpdateTaskProgressData = {
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

      await dbManager.update('tasks', taskId.toString(), updateData);
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
      await dbManager.update('tasks', taskId.toString(), {
        status: 'completed',
        progress: 100,
        result: JSON.stringify(result),
        completed_at: new Date().toISOString()
      });
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
      await dbManager.update('tasks', taskId.toString(), {
        status: 'failed',
        error: JSON.stringify(error),
        completed_at: new Date().toISOString()
      });
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
  async getUserTasks(userId: string, filters: TaskFilters = {}): Promise<Task[]> {
    try {
      // Build query options
      const queryOptions: any = {
        where: { user_id: userId },
        orderBy: { column: 'created_at', ascending: false }
      };
      
      // Apply status filter if provided
      if (filters.status) {
        queryOptions.where.status = filters.status;
      }

      // Apply pagination
      if (filters.limit) {
        queryOptions.limit = filters.limit;
      }
      if (filters.offset) {
        queryOptions.offset = filters.offset;
      }

      const userTasks = await dbManager.find('tasks', queryOptions) as unknown as Task[];
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
  async getTaskStatistics(userId: string): Promise<{
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
