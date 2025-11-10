import { EventEmitter } from 'events';
import { logger } from '../config';

export interface ProcessingJob {
  id: string;
  userId: number;
  type: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  result?: any;
  error?: string;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  metrics?: Record<string, any>;
  [key: string]: any;
}

export class FileProcessor extends EventEmitter {
  private jobQueue: ProcessingJob[] = [];
  private activeJobs: Map<string, ProcessingJob> = new Map();
  private completedJobs: Map<string, ProcessingJob> = new Map();
  private failedJobs: Map<string, ProcessingJob> = new Map();
  private maxConcurrentJobs: number;

  constructor(maxConcurrentJobs = 3) {
    super();
    this.maxConcurrentJobs = maxConcurrentJobs;
    this.processQueue();
  }

  public async addJob(
    userId: number,
    type: string,
    data: any,
    options: any = {}
  ): Promise<string> {
    const job: ProcessingJob = {
      id: `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId,
      type,
      status: 'queued',
      progress: 0,
      data,
      createdAt: Date.now(),
      ...options
    };

    this.jobQueue.push(job);
    this.emit('jobQueued', job);
    logger.info('Job queued', { jobId: job.id, userId, type });
    
    return job.id;
  }

  public getJob(jobId: string): ProcessingJob | undefined {
    return (
      this.activeJobs.get(jobId) ||
      this.completedJobs.get(jobId) ||
      this.failedJobs.get(jobId)
    );
  }

  public getStats() {
    return {
      queued: this.jobQueue.length,
      processing: this.activeJobs.size,
      completed: this.completedJobs.size,
      failed: this.failedJobs.size,
      totalJobs: this.jobQueue.length + this.activeJobs.size + this.completedJobs.size + this.failedJobs.size,
      failedJobs: this.failedJobs.size,
      queuedJobs: this.jobQueue.length,
      processingJobs: this.activeJobs.size,
      completedJobs: this.completedJobs.size
    };
  }

  private async processQueue(): Promise<void> {
    while (true) {
      if (this.activeJobs.size < this.maxConcurrentJobs && this.jobQueue.length > 0) {
        const job = this.jobQueue.shift();
        if (job) {
          await this.processJob(job);
        }
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  private async processJob(job: ProcessingJob): Promise<void> {
    try {
      job.status = 'processing';
      job.startedAt = Date.now();
      this.activeJobs.set(job.id, job);
      
      this.emit('jobStarted', job);
      this.emit('jobProgress', job, 'initializing', 5);

      // Simulate processing
      for (let progress = 10; progress <= 100; progress += 10) {
        if (job.status === 'failed') break;
      
        job.progress = progress;
        this.emit('jobProgress', job, 'processing', progress);
      
        // Simulate work
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      if (job.status !== 'failed') {
        job.status = 'completed';
        job.completedAt = Date.now();
        job.result = { success: true, message: 'Processing completed successfully' };
        this.completedJobs.set(job.id, job);
        this.emit('jobCompleted', job);
      }
    } catch (error) {
      job.status = 'failed';
      job.error = (error as Error).message;
      this.failedJobs.set(job.id, job);
      this.emit('jobFailed', job, error);
      logger.error('Job processing failed', { jobId: job.id, error });
    } finally {
      this.activeJobs.delete(job.id);
    }
  }

  public updateJobProgress(jobId: string, progress: number, stage?: string): void {
    const job = this.activeJobs.get(jobId);
    if (job) {
      job.progress = Math.min(100, Math.max(0, progress));
      this.emit('jobProgress', job, stage || 'processing', progress);
    }
  }

  public failJob(jobId: string, error: Error): void {
    const job = this.activeJobs.get(jobId) || this.jobQueue.find(j => j.id === jobId);
    if (job) {
      job.status = 'failed';
      job.error = error.message;
      this.failedJobs.set(jobId, job);
      this.emit('jobFailed', job, error);
      
      // Remove from active or queue
      this.activeJobs.delete(jobId);
      this.jobQueue = this.jobQueue.filter(j => j.id !== jobId);
    }
  }
}

// Export singleton instance
export const fileProcessor = new FileProcessor();

// Export the interface
export interface ProcessingJob extends ProcessingJobBase {}

// Internal interface for the implementation
interface ProcessingJobBase {
  id: string;
  userId: number;
  type: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  result?: any;
  error?: string;
  createdAt: number;
  startedAt?: number;
  completedAt?: number;
  metrics?: Record<string, any>;
  [key: string]: any;
}
