import api from "@/lib/api"
import { JOB_TIMEOUTS } from "@/config/job-timeouts"
import { normalizeJobResponse } from "./jobIdNormalizer"  // SAFETY SHIM: Add backward compatibility



export interface AnalysisJobStatus {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress?: number; // 0-100
  result?: AnalysisResult;
  error?: string;
  metadata?: {
    processingTime?: number;
    modelInfo?: Record<string, unknown>;
    queuePosition?: number;
    estimatedTimeRemaining?: number;
  };
}

export interface AnalysisResult {
  isAuthentic: boolean;
  confidence: number;
  details: {
    isDeepfake: boolean;
    modelInfo: Record<string, unknown>;
    analysisType: string;
    features?: Record<string, unknown>;
  };
  metrics: {
    processingTime: number;
    modelVersion: string;
    accuracy?: number;
    precision?: number;
    recall?: number;
  };
  proof?: {
    signature: string;
    timestamp: string;
    modelHash: string;
    dataHash: string;
  };
}

export interface PollingOptions {
  interval?: number; // Initial polling interval in ms
  maxInterval?: number; // Maximum polling interval in ms
  timeout?: number; // Maximum time to wait in ms
  maxRetries?: number; // Maximum number of retries
  backoffFactor?: number; // Exponential backoff factor
  jitter?: boolean; // Add random jitter to prevent thundering herd
  onProgress?: (progress: number) => void; // Progress callback
  onStatusChange?: (status: string) => void; // Status change callback
  enableMetrics?: boolean; // Enable performance monitoring
}

export interface PollingMetrics {
  startTime: number;
  endTime?: number;
  totalRequests: number;
  successfulRequests: number;
  failedRequests: number;
  averageResponseTime: number;
  backoffCount: number;
}

class PollingError extends Error {
  constructor(
    message: string,
    public code: string,
    public retryable: boolean = true,
    public metadata?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'PollingError';
  }
}

/**
 * Advanced polling function with exponential backoff and comprehensive error handling
 */
export function pollAnalysisResult(
  jobId: string,
  options: PollingOptions = {}
): {
  promise: Promise<AnalysisJobStatus>;
  cancel: () => void;
  getMetrics: () => PollingMetrics;
} {
  // Default options
  const config: Required<PollingOptions> = {
    interval: options.interval || 2000, // Start with 2 seconds
    maxInterval: options.maxInterval || JOB_TIMEOUTS.POLLING_MAX_INTERVAL,
    timeout: options.timeout || JOB_TIMEOUTS.POLLING_TIMEOUT,
    maxRetries: options.maxRetries || 10,
    backoffFactor: options.backoffFactor || 1.5,
    jitter: options.jitter !== false,
    onProgress: options.onProgress || (() => {}),
    onStatusChange: options.onStatusChange || (() => {}),
    enableMetrics: options.enableMetrics !== false
  };

  // State management
  let isCancelled = false;
  let currentInterval = config.interval;
  let retryCount = 0;
  let consecutiveErrors = 0;
  
  // Metrics tracking
  const metrics: PollingMetrics = {
    startTime: Date.now(),
    totalRequests: 0,
    successfulRequests: 0,
    failedRequests: 0,
    averageResponseTime: 0,
    backoffCount: 0
  };

  let responseTimes: number[] = [];

  // Cleanup function
  const cancel = () => {
    isCancelled = true;
  };

  // Calculate next interval with exponential backoff and jitter
  const calculateNextInterval = (): number => {
    if (consecutiveErrors === 0) {
      return config.interval; // Reset to base interval on success
    }

    // Exponential backoff
    let nextInterval = Math.min(
      currentInterval * config.backoffFactor,
      config.maxInterval
    );

    // Add jitter to prevent thundering herd
    if (config.jitter) {
      const jitterRange = nextInterval * 0.1; // 10% jitter
      nextInterval += (Math.random() - 0.5) * jitterRange;
    }

    currentInterval = Math.max(nextInterval, 1000); // Minimum 1 second
    metrics.backoffCount++;

    return currentInterval;
  };

  // Make API request with timeout and error handling
  const makeRequest = async (): Promise<AnalysisJobStatus> => {
    const startTime = Date.now();
    
    try {
      metrics.totalRequests++;
      
      const response = await api.get(`/results/${jobId}`, {
        timeout: 10000, // 10 second timeout per request
        headers: {
          'X-Request-ID': `poll-${jobId}-${Date.now()}`,
          'X-Polling-Retry': retryCount.toString()
        }
      }) as { data: AnalysisJobStatus };

      const responseTime = Date.now() - startTime;
      responseTimes.push(responseTime);
      metrics.averageResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;

      // SAFETY SHIM: Normalize response for backward compatibility
      const normalizedData = normalizeJobResponse(response.data);
      
      if (!normalizedData) {
        throw new PollingError(
          'No data received from server',
          'NO_DATA',
          true
        );
      }

      const jobStatus: AnalysisJobStatus = normalizedData as AnalysisJobStatus;  // ✅ FIX: Type assertion for compatibility
      
      // Validate response structure
      if (!jobStatus.id || !jobStatus.status) {
        throw new PollingError(
          'Invalid response structure',
          'INVALID_RESPONSE',
          true
        );
      }

      metrics.successfulRequests++;
      consecutiveErrors = 0; // Reset error count on success

      // Call progress callback if available
      if (jobStatus.progress !== undefined) {
        config.onProgress(jobStatus.progress);
      }

      return jobStatus;

    } catch (error) {
      metrics.failedRequests++;
      consecutiveErrors++;
      
      const responseTime = Date.now() - startTime;
      responseTimes.push(responseTime);
      metrics.averageResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;

      // Determine if error is retryable
      const isRetryable = shouldRetry(error);
      
      throw new PollingError(
        `Polling request failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        'REQUEST_FAILED',
        isRetryable,
        { originalError: error, retryCount, consecutiveErrors }
      );
    }
  };

  // Determine if error should be retried
  const shouldRetry = (error: unknown): boolean => {
    if (error instanceof PollingError) {
      return error.retryable;
    }

    if (error instanceof Error) {
      // Network errors are retryable
      if (error.message.includes('Network Error') || 
          error.message.includes('timeout') ||
          error.message.includes('ECONNREFUSED')) {
        return true;
      }

      // 4xx client errors are not retryable (except 429)
      if (error.message.includes('400') || 
          error.message.includes('401') || 
          error.message.includes('403') || 
          error.message.includes('404')) {
        return false;
      }

      // 5xx server errors are retryable
      if (error.message.includes('500') || 
          error.message.includes('502') || 
          error.message.includes('503') || 
          error.message.includes('504')) {
        return true;
      }
    }

    return true; // Default to retryable
  };

  // Main polling logic
  const poll = async (): Promise<AnalysisJobStatus> => {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        if (!isCancelled) {
          metrics.endTime = Date.now();
          reject(new PollingError(
            `Polling timeout after ${config.timeout}ms`,
            'TIMEOUT',
            false,
            { metrics }
          ));
        }
      }, config.timeout);

      const executePoll = async () => {
        if (isCancelled) {
          clearTimeout(timeoutId);
          reject(new PollingError('Polling cancelled', 'CANCELLED', false));
          return;
        }

        try {
          const result = await makeRequest();
          
          // Call status change callback if status changed
          config.onStatusChange(result.status);

          // Check if polling should stop
          if (result.status === 'completed') {
            metrics.endTime = Date.now();
            clearTimeout(timeoutId);
            resolve(result);
            return;
          }

          if (result.status === 'failed' || result.status === 'cancelled') {
            metrics.endTime = Date.now();
            clearTimeout(timeoutId);
            reject(new PollingError(
              result.error || `Analysis ${result.status}`,
              result.status.toUpperCase(),
              false,
              { jobStatus: result, metrics }
            ));
            return;
          }

          // Continue polling
          retryCount++;
          const nextInterval = calculateNextInterval();
          
          setTimeout(executePoll, nextInterval);

        } catch (error) {
          if (isCancelled) {
            clearTimeout(timeoutId);
            reject(new PollingError('Polling cancelled', 'CANCELLED', false));
            return;
          }

          // Check if we should retry
          if (retryCount >= config.maxRetries || 
              (error instanceof PollingError && !error.retryable)) {
            metrics.endTime = Date.now();
            clearTimeout(timeoutId);
            reject(error);
            return;
          }

          // Retry with backoff
          const nextInterval = calculateNextInterval();
          setTimeout(executePoll, nextInterval);
        }
      };

      // Start polling
      executePoll();
    });
  };

  return {
    promise: poll(),
    cancel,
    getMetrics: () => ({ ...metrics })
  };
}

/**
 * Simplified polling function for basic use cases
 */
export function pollAnalysisResultSimple(
  jobId: string,
  onUpdate: (data: AnalysisJobStatus) => void,
  onError: (error: Error) => void,
  interval: number = 3000
): () => void {
  const { promise, cancel } = pollAnalysisResult(jobId, {
    interval,
    timeout: 120000, // 2 minutes
    maxRetries: 5
  });

  promise
    .then(onUpdate)
    .catch(onError);

  return cancel;
}

/**
 * Batch polling for multiple jobs
 */
export function pollMultipleJobs(
  jobIds: string[],
  options: PollingOptions = {}
): {
    promise: Promise<Map<string, AnalysisJobStatus>>;
    cancel: () => void;
    getMetrics: () => Map<string, PollingMetrics>;
  } {
  const results = new Map<string, AnalysisJobStatus>();
  const metricsMap = new Map<string, PollingMetrics>();
  const polls = new Map<string, { cancel: () => void; getMetrics: () => PollingMetrics }>();

  const cancel = () => {
    polls.forEach(poll => poll.cancel());
  };

  const getMetrics = () => new Map(metricsMap);

  const promise = Promise.all(
    jobIds.map(jobId => {
      const poll = pollAnalysisResult(jobId, options);
      polls.set(jobId, poll);
      
      return poll.promise
        .then(result => {
          results.set(jobId, result);
          metricsMap.set(jobId, poll.getMetrics());
          return { jobId, result };
        })
        .catch(error => {
          metricsMap.set(jobId, poll.getMetrics());
          throw error;
        });
    })
  ).then(() => results);

  return { promise, cancel, getMetrics };
}

/**
 * Utility function to estimate polling time based on job type and size
 */
export function estimatePollingTime(
  jobType: 'image' | 'video' | 'audio' | 'multimodal',
  fileSize?: number
): {
  estimatedTime: number;
  recommendedInterval: number;
  recommendedTimeout: number;
} {
  const baseTimes = {
    image: 5000,      // 5 seconds
    video: 30000,     // 30 seconds
    audio: 15000,     // 15 seconds
    multimodal: 45000 // 45 seconds
  };

  const baseTime = baseTimes[jobType] || baseTimes.image;
  
  // Adjust for file size (if provided)
  let multiplier = 1;
  if (fileSize) {
    const sizeMB = fileSize / (1024 * 1024);
    if (sizeMB > 50) multiplier = 2;
    else if (sizeMB > 20) multiplier = 1.5;
    else if (sizeMB > 10) multiplier = 1.2;
  }

  const estimatedTime = Math.ceil(baseTime * multiplier);
  const recommendedInterval = Math.min(estimatedTime / 10, 5000); // Poll 10 times during processing
  const recommendedTimeout = estimatedTime * 3; // 3x safety margin

  return {
    estimatedTime,
    recommendedInterval: Math.max(recommendedInterval, 1000),
    recommendedTimeout
  };
}