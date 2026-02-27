/**
 * Job Timeout Configuration - Client Side
 * 
 * Client-side timeout constants that must match server-side values
 * to ensure consistent polling behavior.
 */

export const JOB_TIMEOUTS = {
  // Polling timeouts
  POLLING_TIMEOUT: 5 * 60 * 1000, // 5 minutes polling timeout
  POLLING_MAX_INTERVAL: 30 * 1000, // 30 seconds max polling interval
  
  // Maximum processing time (for estimation)
  MAX_PROCESSING_TIME: 15 * 60 * 1000, // 15 minutes
} as const;
