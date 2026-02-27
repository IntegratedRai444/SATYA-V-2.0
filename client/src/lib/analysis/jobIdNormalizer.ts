/**
 * BACKUP SAFETY SHIM - Layer 6 Stabilization
 * 
 * Temporary normalization layer to prevent runtime crashes during jobId migration
 * Remove after all components are updated to use consistent naming
 */

export interface LegacyJobResponse {
  success: boolean;
  job_id?: string;  // Legacy snake_case
  jobId?: string;   // New camelCase
  status: string;
  id?: string;      // ✅ FIX: Add id field for AnalysisJobStatus compatibility
  result?: any;   // ✅ FIX: Add result field for response structure
}

/**
 * Normalizes job ID field names across response formats
 * Prevents crashes during migration from job_id to jobId
 */
export function normalizeJobId(response: LegacyJobResponse): string | undefined {
  // Try new format first, then fallback to legacy
  return response.jobId || response.job_id;
}

/**
 * Normalizes entire job response structure
 * Ensures consistent shape for polling and UI components
 */
export function normalizeJobResponse(response: any): LegacyJobResponse {
  if (!response) return response;
  
  return {
    ...response,
    jobId: response.jobId || response.job_id,
    id: response.id || response.jobId || response.job_id,
    result: response.result || response.data?.result // Handle nested result structure
  };
}
