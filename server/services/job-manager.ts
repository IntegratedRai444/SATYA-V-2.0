import { logger } from '../config/logger';
import { supabase } from '../config/supabase';
import webSocketManager from './websocket-manager';
import { AbortController } from 'abort-controller';
import { setInterval } from 'timers';

// Track running jobs in memory
interface RunningJob {
  id: string;
  userId: string;
  modality: string;
  startTime: number;
  status: 'running' | 'cancelled' | 'completed' | 'failed';
  abortController?: AbortController;
}

class JobManager {
  private runningJobs = new Map<string, RunningJob>();

  // Add a new job to tracking
  addJob(jobId: string, userId: string, modality: string, abortController?: AbortController): void {
    const job: RunningJob = {
      id: jobId,
      userId,
      modality,
      startTime: Date.now(),
      status: 'running',
      abortController
    };
    
    this.runningJobs.set(jobId, job);
    logger.info(`Job added to tracking: ${jobId}`, { userId, modality });
  }

  // Cancel a running job
  async cancelJob(jobId: string, userId: string): Promise<boolean> {
    const job = this.runningJobs.get(jobId);
    
    if (!job) {
      logger.warn(`Attempted to cancel non-existent job: ${jobId}`);
      return false;
    }

    if (job.userId !== userId) {
      logger.warn(`Unauthorized cancellation attempt for job: ${jobId}`, { 
        requestedBy: userId, 
        jobOwner: job.userId 
      });
      return false;
    }

    if (job.status !== 'running') {
      logger.warn(`Cannot cancel job in status: ${job.status}`, { jobId });
      return false;
    }

    try {
      // Mark as cancelled
      job.status = 'cancelled';
      
      // Abort the request if we have an abort controller
      if (job.abortController) {
        job.abortController.abort();
      }

      // Update database
      await supabase
        .from('tasks')
        .update({ 
          status: 'cancelled',
          updated_at: new Date().toISOString()
        })
        .eq('id', jobId)
        .eq('user_id', userId);

      // Send WebSocket notification
      webSocketManager.sendJobCancelled(userId, jobId);

      // Remove from tracking
      this.runningJobs.delete(jobId);
      
      logger.info(`Job cancelled successfully: ${jobId}`, { userId });
      return true;
    } catch (error) {
      logger.error(`Failed to cancel job: ${jobId}`, { error, userId });
      return false;
    }
  }

  // Get job status
  getJobStatus(jobId: string): RunningJob | null {
    return this.runningJobs.get(jobId) || null;
  }

  // Get all jobs for a user
  getUserJobs(userId: string): RunningJob[] {
    return Array.from(this.runningJobs.values())
      .filter(job => job.userId === userId);
  }

  // Remove completed/failed jobs (cleanup)
  removeJob(jobId: string): void {
    const job = this.runningJobs.get(jobId);
    if (job && (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled')) {
      this.runningJobs.delete(jobId);
      if (process.env.NODE_ENV === 'development') {
      logger.debug(`Job removed from tracking: ${jobId}`);
    }
    }
  }

  // Get running jobs count
  getRunningJobsCount(): number {
    return Array.from(this.runningJobs.values())
      .filter(job => job.status === 'running').length;
  }

  // Cleanup old jobs (call this periodically)
  async cleanup(): Promise<void> {
    const now = Date.now();
    const maxAge = 30 * 60 * 1000; // 30 minutes
    const maxProcessingTime = 10 * 60 * 1000; // 10 minutes max processing time

    for (const [jobId, job] of this.runningJobs.entries()) {
      // Clean up stale jobs
      if (now - job.startTime > maxAge) {
        logger.warn(`Cleaning up stale job: ${jobId}`, { 
          age: now - job.startTime,
          status: job.status 
        });
        this.runningJobs.delete(jobId);
      }
      
      // Timeout jobs running too long
      if (job.status === 'running' && (now - job.startTime > maxProcessingTime)) {
        logger.warn(`Timing out job: ${jobId}`, { 
          processingTime: now - job.startTime,
          status: job.status 
        });
        
        // Mark as failed in database
        const updateResult = await supabase
          .from('tasks')
          .update({ 
            status: 'failed',
            error: 'Job timed out after maximum processing time',
            updated_at: new Date().toISOString()
          })
          .eq('id', jobId);
          
        if (updateResult.error) {
          logger.error(`Failed to update timed out job ${jobId}:`, updateResult.error);
        } else {
          // Send WebSocket notification for timeout
          webSocketManager.sendJobFailed(job.userId, jobId, 'Job timed out after maximum processing time');
          
          // Remove from tracking
          this.runningJobs.delete(jobId);
          logger.info(`Job ${jobId} marked as failed due to timeout`);
        }
      }
    }
  }
}

// Create singleton instance
const jobManager = new JobManager();

// Cleanup stale jobs every 5 minutes
setInterval(async () => {
  await jobManager.cleanup();
}, 5 * 60 * 1000);

export default jobManager;
