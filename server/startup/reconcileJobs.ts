import { logger } from '../config/logger';
import { supabaseAdmin } from '../config/supabase';
import { safeJson } from '../utils/result-parser';

/**
 * 🔥 FIX 3 — Job Reconciliation on Server Boot (BIG UPGRADE)
 * 
 * This function cleans up stale jobs that may be stuck in "processing" state
 * due to server crashes, redeploys, or other interruptions.
 * 
 * Jobs are considered stale if they've been in "processing" state for more
 * than 30 minutes without completion.
 */
export async function reconcileStaleJobs(): Promise<void> {
  const STALE_MINUTES = 30;
  const cutoff = new Date(Date.now() - STALE_MINUTES * 60 * 1000);

  try {
    logger.info('[RECONCILIATION] Starting stale job reconciliation', {
      staleMinutes: STALE_MINUTES,
      cutoffTime: cutoff.toISOString()
    });

    // Try a safer query approach - fetch tasks and filter in JavaScript
    const { data: allTasks, error: fetchError } = await supabaseAdmin
      .from('tasks')
      .select('id, created_at, user_id, type, result')
      .lt('created_at', cutoff.toISOString())
      .limit(1000);

    if (fetchError) {
      logger.error('[RECONCILIATION] Failed to fetch tasks for reconciliation', {
        error: fetchError.message,
        details: fetchError
      });
      return;
    }

    // Filter for processing jobs in JavaScript (safer approach)
    const staleJobs = allTasks?.filter(task => {
      const result = task.result;
      
      // Check if it's a JSON object with status
      if (typeof result === 'object' && result !== null) {
        return result.status === 'processing';
      }
      
      // Check if it's a legacy string
      if (typeof result === 'string') {
        return result === 'processing';
      }
      
      return false;
    }).map(task => ({
      id: task.id,
      created_at: task.created_at,
      user_id: task.user_id,
      type: task.type
    }));

    if (!staleJobs || staleJobs.length === 0) {
      logger.info('[RECONCILIATION] No stale jobs found');
      return;
    }

    logger.warn('[RECONCILIATION] Found stale jobs', {
      count: staleJobs.length,
      jobs: staleJobs.map(job => ({
        id: job.id,
        userId: job.user_id,
        type: job.type,
        createdAt: job.created_at,
        ageMinutes: Math.floor((Date.now() - new Date(job.created_at).getTime()) / 60000)
      }))
    });

    // Update all stale jobs to failed status
    const reconciliationPromises = staleJobs.map(async (job) => {
      try {
        const { error: updateError } = await supabaseAdmin
          .from('tasks')
          .update({
            result: safeJson({ status: 'failed', reason: 'stale_job', recovered: true }),
            confidence_score: 0,
            updated_at: new Date().toISOString()
          })
          .eq('id', job.id);

        if (updateError) {
          logger.error('[RECONCILIATION] Failed to update stale job', {
            jobId: job.id,
            error: updateError.message
          });
          return { success: false, jobId: job.id, error: updateError.message };
        }

        logger.info('[RECONCILIATION] Reconciled stale job', {
          jobId: job.id,
          userId: job.user_id,
          type: job.type,
          ageMinutes: Math.floor((Date.now() - new Date(job.created_at).getTime()) / 60000)
        });

        return { success: true, jobId: job.id };
      } catch (err) {
        logger.error('[RECONCILIATION] Exception updating stale job', {
          jobId: job.id,
          error: err instanceof Error ? err.message : String(err)
        });
        return { success: false, jobId: job.id, error: String(err) };
      }
    });

    const results = await Promise.allSettled(reconciliationPromises);
    
    const successful = results.filter(r => 
      r.status === 'fulfilled' && r.value?.success
    ).length;
    
    const failed = results.length - successful;

    logger.info('[RECONCILIATION] Job reconciliation completed', {
      total: results.length,
      successful,
      failed,
      duration: Date.now() - cutoff.getTime()
    });

    if (failed > 0) {
      logger.warn('[RECONCILIATION] Some jobs failed to reconcile', {
        failedCount: failed,
        successRate: `${((successful / results.length) * 100).toFixed(1)}%`
      });
    }

  } catch (err) {
    logger.error('[RECONCILIATION] Critical error during job reconciliation', {
      error: err instanceof Error ? err.message : String(err),
      stack: err instanceof Error ? err.stack : undefined
    });
  }
}

/**
 * Graceful shutdown cleanup - mark running jobs as failed
 * This is called during server shutdown to prevent orphaned jobs
 */
export async function cleanupRunningJobs(): Promise<void> {
  try {
    logger.info('[CLEANUP] Starting running job cleanup for shutdown');

    // Use safe approach - fetch tasks and filter in JavaScript
    const { data: allTasks, error: fetchError } = await supabaseAdmin
      .from('tasks')
      .select('id, user_id, type, result')
      .limit(1000);

    if (fetchError) {
      logger.error('[CLEANUP] Failed to fetch running jobs', {
        error: fetchError.message
      });
      return;
    }

    // Filter for processing jobs in JavaScript (safer approach)
    const runningJobs = allTasks?.filter(task => {
      const result = task.result;
      
      // Check if it's a JSON object with status
      if (typeof result === 'object' && result !== null) {
        return result.status === 'processing';
      }
      
      // Check if it's a legacy string
      if (typeof result === 'string') {
        return result === 'processing';
      }
      
      return false;
    }).map(task => ({
      id: task.id,
      user_id: task.user_id,
      type: task.type
    }));

    if (!runningJobs || runningJobs.length === 0) {
      logger.info('[CLEANUP] No running jobs to clean up');
      return;
    }

    logger.info('[CLEANUP] Cleaning up running jobs', {
      count: runningJobs.length
    });

    const cleanupPromises = runningJobs.map(async (job) => {
      try {
        await supabaseAdmin
          .from('tasks')
          .update({
            result: safeJson({ status: 'failed', reason: 'shutdown_cleanup' }),
            confidence_score: 0,
            updated_at: new Date().toISOString()
          })
          .eq('id', job.id);

        logger.info('[CLEANUP] Cleaned up running job', {
          jobId: job.id,
          userId: job.user_id,
          type: job.type
        });

        return { success: true, jobId: job.id };
      } catch (err) {
        logger.error('[CLEANUP] Failed to cleanup running job', {
          jobId: job.id,
          error: err instanceof Error ? err.message : String(err)
        });
        return { success: false, jobId: job.id };
      }
    });

    const results = await Promise.allSettled(cleanupPromises);
    const successful = results.filter(r => 
      r.status === 'fulfilled' && r.value?.success
    ).length;

    logger.info('[CLEANUP] Job cleanup completed', {
      total: results.length,
      successful,
      failed: results.length - successful
    });

  } catch (err) {
    logger.error('[CLEANUP] Critical error during job cleanup', {
      error: err instanceof Error ? err.message : String(err)
    });
  }
}
