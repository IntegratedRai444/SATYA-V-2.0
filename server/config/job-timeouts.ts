/**
 * Job Timeout Configuration - Single Source of Truth
 * 
 * All timeout values must reference these constants to ensure consistency
 * across job manager, routes, and polling.
 */

export const JOB_TIMEOUTS = {
  // Maximum processing time for any job type
  MAX_PROCESSING_TIME: 15 * 60 * 1000, // 15 minutes
  
  // Cleanup intervals
  CLEANUP_INTERVAL: 5 * 60 * 1000, // 5 minutes
  MAX_JOB_AGE: 30 * 60 * 1000, // 30 minutes before cleanup
  
  // Polling timeouts
  POLLING_TIMEOUT: 5 * 60 * 1000, // 5 minutes polling timeout
  POLLING_MAX_INTERVAL: 30 * 1000, // 30 seconds max polling interval
  
  // Request timeouts
  PYTHON_REQUEST_TIMEOUT: 15 * 60 * 1000, // 15 minutes for Python inference
  PYTHON_INFERENCE_TIMEOUT: 15 * 60 * 1000, // 15 minutes for ML inference
  PYTHON_ASYNCIO_TIMEOUT: 15 * 60 * 1000, // 15 minutes for asyncio.wait_for
} as const;

// Export individual constants for easier importing
export const {
  MAX_PROCESSING_TIME,
  CLEANUP_INTERVAL,
  MAX_JOB_AGE,
  POLLING_TIMEOUT,
  POLLING_MAX_INTERVAL,
  PYTHON_REQUEST_TIMEOUT,
} = JOB_TIMEOUTS;
