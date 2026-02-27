import { supabase } from '../config/supabase';
import { logger } from '../config/logger';

export interface Task {
  id: string;
  createdAt: Date;
  updatedAt: Date | null;
  userId: string;
  type: string;
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  fileName: string;
  fileSize: number;
  fileType: string;
  filePath: string;
  reportCode: string;
  result: AnalysisResult | null;
  error: ErrorInfo | null;
  startedAt: Date | null;
  completedAt: Date | null;
  metadata: Record<string, unknown>;
}

type UpdateTaskProgressData = {
  progress: number;
  status: 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  startedAt?: Date;
  metadata?: string;
};

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
      const taskData = {
        user_id: userId,
        type,
        filename: fileInfo.name,
        result: 'processing',
        confidence_score: 0,
        detection_details: null,
        metadata: {
          originalName: fileInfo.name,
          fileSize: fileInfo.size,
          fileType: fileInfo.type,
          filePath: fileInfo.path,
          uploadedAt: new Date().toISOString()
        },
        created_at: new Date().toISOString()
      };
      const { data: task, error } = await supabase
        .from('scans')
        .insert(taskData)
        .select('*')
        .single();

      if (error || !task) {
        throw new Error(`Failed to create task: ${error?.message}`);
      }

      return task as Task;
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
  async getTaskStatus(taskId: string): Promise<Task | null> {
    try {
      const { data: task, error } = await supabase
        .from('scans')
        .select('*')
        .eq('id', taskId)
        .single();
      
      if (error || !task) return null;
      
      // Map the database record to Task type
      return {
        id: task.id,
        createdAt: new Date(task.created_at),
        updatedAt: task.updated_at ? new Date(task.updated_at) : null,
        userId: task.user_id,
        type: task.type,
        status: task.result === 'processing' ? 'processing' : task.result === 'failed' ? 'failed' : 'completed',
        progress: task.confidence_score > 0 ? 100 : 0,
        fileName: task.filename,
        fileSize: 0, // Not stored in scans table
        fileType: task.type,
        filePath: '', // Not stored in scans table
        reportCode: '', // Not stored in scans table
        result: task.detection_details ? task.detection_details : null,
        error: task.confidence_score === 0 ? { message: 'Analysis failed' } : null,
        startedAt: task.created_at ? new Date(task.created_at) : null,
        completedAt: task.updated_at ? new Date(task.updated_at) : null,
        metadata: task.metadata ? task.metadata : {}
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
    taskId: string,
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
          const metadata = task.metadata ? JSON.parse(JSON.stringify(task.metadata)) : {};
          metadata.lastMessage = message;
          metadata.lastUpdate = new Date().toISOString();
          updateData.metadata = JSON.stringify(metadata);
        }
      }

      const { error } = await supabase
        .from('scans')
        .update({
          result: updateData.status === 'completed' ? 'completed' : updateData.status === 'failed' ? 'failed' : 'processing',
          confidence_score: updateData.progress === 100 ? 1 : updateData.progress / 100,
          updated_at: new Date().toISOString()
        })
        .eq('id', taskId);

      if (error) {
        throw new Error(`Failed to update task progress: ${error.message}`);
      }
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
  async completeTask(taskId: string, result: AnalysisResult): Promise<void> {
    try {
      const { error } = await supabase
        .from('scans')
        .update({
          result: 'completed',
          confidence_score: result.confidence || 1,
          detection_details: result,
          updated_at: new Date().toISOString()
        })
        .eq('id', taskId);

      if (error) {
        throw new Error(`Failed to complete task: ${error.message}`);
      }
    } catch (error) {
      logger.error('Error completing task', error as Error);
      throw new Error('Failed to complete task');
    }
  }

  /**
   * Fail a task with error information
   * @param taskId - Task ID
   * @param errorInfo - Error information
   */
  async failTask(taskId: string): Promise<void> {
    try {
      const { error: updateError } = await supabase
        .from('scans')
        .update({
          result: 'failed',
          confidence_score: 0,
          updated_at: new Date().toISOString()
        })
        .eq('id', taskId);

      if (updateError) {
        throw new Error(`Failed to fail task: ${updateError.message}`);
      }
    } catch (error) {
      logger.error('Error failing task', error as Error);
      throw new Error('Failed to fail task');
    }
  }

  /**
   * Get user tasks with optional filters
   * @param userId - User ID
   * @param filters - Optional filters
   * @returns Array of tasks
   */
  async getUserTasks(userId: string): Promise<Task[]> {
    try {
      // Build query
      const { data, error } = await supabase
        .from('scans')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });
      
      if (error) {
        throw new Error(`Failed to get user tasks: ${error.message}`);
      }
      
      // Map scans to Task interface
      return (data || []).map((scan) => ({
        id: scan.id,
        createdAt: new Date(scan.created_at),
        updatedAt: scan.updated_at ? new Date(scan.updated_at) : null,
        userId: scan.user_id,
        type: scan.type,
        status: scan.result === 'processing' ? 'processing' : scan.result === 'failed' ? 'failed' : 'completed',
        progress: scan.confidence_score > 0 ? 100 : 0,
        fileName: scan.filename,
        fileSize: 0,
        fileType: scan.type,
        filePath: '',
        reportCode: '',
        result: scan.detection_details ? scan.detection_details : null,
        error: scan.confidence_score === 0 ? { message: 'Analysis failed' } : null,
        startedAt: scan.created_at ? new Date(scan.created_at) : null,
        completedAt: scan.updated_at ? new Date(scan.updated_at) : null,
        metadata: scan.metadata ? scan.metadata : {}
      })) as Task[];
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
   */
  async cleanupOldTasks(daysOld: number = 30): Promise<number> {
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
