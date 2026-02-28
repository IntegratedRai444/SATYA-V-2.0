import { supabaseAdmin } from '../config/supabase';
import { logger } from '../config/logger';

/**
 * One-time migration script to fix JSONB consistency in tasks.result column
 * Converts legacy string values to proper JSON objects
 */

export async function migrateJsonbResults(): Promise<void> {
  // Guard against multiple runs
  const migrationKey = 'JSONB_MIGRATION_COMPLETED';
  
  try {
    logger.info('[JSONB MIGRATION] Starting JSONB consistency fix...');
    
    // Check if migration already completed
    const { data: settings } = await supabaseAdmin
      .from('settings')
      .select('value')
      .eq('key', migrationKey)
      .single();
    
    if (settings) {
      logger.info('[JSONB MIGRATION] Migration already completed, skipping');
      return;
    }
    
    // Step 2: Convert string values to JSON objects
    const processMigrationData = async (rows: { id: string; result: string }[]) => {
      const migrationPromises = rows.map(async (row) => {
        const stringValue = row.result as string;
        
        // Convert string to JSON object
        let jsonValue: unknown;
        
        if (stringValue === 'processing') {
          jsonValue = { status: 'processing' };
        } else if (stringValue === 'completed') {
          jsonValue = { status: 'completed' };
        } else if (stringValue === 'failed') {
          jsonValue = { status: 'failed' };
        } else if (stringValue === 'cancelled') {
          jsonValue = { status: 'cancelled' };
        } else if (stringValue === 'pending') {
          jsonValue = { status: 'pending' };
        } else {
          // For unknown strings, wrap generically
          jsonValue = { value: stringValue };
          logger.warn(`[JSONB MIGRATION] Unknown status string: ${stringValue}`, { taskId: row.id });
        }
        
        // Update the row
        const { error: updateError } = await supabaseAdmin
          .from('tasks')
          .update({ result: jsonValue })
          .eq('id', row.id);
        
        if (updateError) {
          logger.error(`[JSONB MIGRATION] Failed to update row ${row.id}`, updateError);
          return { success: false, id: row.id, error: updateError.message };
        }
        
        logger.info(`[JSONB MIGRATION] Migrated row ${row.id}: "${stringValue}" → ${JSON.stringify(jsonValue)}`);
        return { success: true, id: row.id };
      });
      
      const results = await Promise.allSettled(migrationPromises);
      
      const successful = results.filter(r => 
        r.status === 'fulfilled' && r.value?.success
      ).length;
      
      const failed = results.length - successful;
      
      logger.info(`[JSONB MIGRATION] Migration completed`, {
        total: results.length,
        successful,
        failed,
        successRate: `${((successful / results.length) * 100).toFixed(1)}%`
      });
      
      if (failed > 0) {
        logger.error(`[JSONB MIGRATION] ${failed} rows failed to migrate`);
      }
      
      // Step 3: Mark migration as complete
      await markMigrationComplete();
      
      logger.info('[JSONB MIGRATION] ✅ JSONB consistency fix completed successfully');
    };
    
    // Step 1: Find legacy string values using raw SQL
    const { data: stringResults, error: findError } = await supabaseAdmin
      .rpc('find_legacy_string_results');
    
    if (findError) {
      logger.error('[JSONB MIGRATION] Failed to find string results', findError);
      // If RPC doesn't exist, try a different approach
      logger.warn('[JSONB MIGRATION] RPC not available, trying manual approach');
      
      // Use a simple approach - fetch all tasks and filter in code
      const { data: allTasks, error: allTasksError } = await supabaseAdmin
        .from('tasks')
        .select('id, result')
        .limit(1000); // Limit to prevent memory issues
      
      if (allTasksError) {
        logger.error('[JSONB MIGRATION] Failed to fetch all tasks', allTasksError);
        throw allTasksError;
      }
      
      // Debug: Log what data types we actually see
      const typeCounts = {
        string: 0,
        object: 0,
        null: 0,
        other: 0
      };
      
      allTasks?.forEach(task => {
        const result = task.result;
        if (typeof result === 'string') {
          typeCounts.string++;
        } else if (typeof result === 'object' && result !== null) {
          typeCounts.object++;
        } else if (result === null) {
          typeCounts.null++;
        } else {
          typeCounts.other++;
        }
      });
      
      logger.info('[JSONB MIGRATION] Data type analysis', typeCounts);
      
      // Filter for string values in JavaScript (more comprehensive)
      const stringResultsFiltered = allTasks?.filter(task => {
        const result = task.result;
        
        // Check if result is a string (legacy format)
        if (typeof result !== 'string') {
          return false;
        }
        
        // Check for known status strings
        const knownStatuses = ['processing', 'completed', 'failed', 'cancelled', 'pending'];
        if (knownStatuses.includes(result)) {
          return true;
        }
        
        // Check for string that looks like a JSON object (malformed)
        if (result.startsWith('{') || result.startsWith('"')) {
          return false; // This is already JSON or malformed JSON
        }
        
        // If it's a string that doesn't look like JSON, it's probably legacy
        return result.length < 100; // Reasonable length check
      }) || [];
      
      logger.info(`[JSONB MIGRATION] Found ${stringResultsFiltered.length} legacy strings out of ${allTasks?.length || 0} total tasks`);
      
      if (!stringResultsFiltered || stringResultsFiltered.length === 0) {
        logger.info('[JSONB MIGRATION] No legacy string results found, migration complete');
        await markMigrationComplete();
        return;
      }
      
      logger.warn(`[JSONB MIGRATION] Found ${stringResultsFiltered.length} rows with legacy string results (manual)`);
      
      await processMigrationData(stringResultsFiltered);
      return;
    }
    
    if (!stringResults || stringResults.length === 0) {
      logger.info('[JSONB MIGRATION] No legacy string results found, migration complete');
      await markMigrationComplete();
      return;
    }
    
    logger.warn(`[JSONB MIGRATION] Found ${stringResults.length} rows with legacy string results`);
    
    await processMigrationData(stringResults);
    
  } catch (error) {
    logger.error('[JSONB MIGRATION] Critical error during migration', error);
    throw error;
  }
}

async function markMigrationComplete(): Promise<void> {
  try {
    // Create settings table if it doesn't exist (simple key-value store)
    await supabaseAdmin
      .from('settings')
      .upsert({
        key: 'JSONB_MIGRATION_COMPLETED',
        value: new Date().toISOString(),
        created_at: new Date().toISOString()
      }, {
        onConflict: 'key'
      });
    
    logger.info('[JSONB MIGRATION] Migration marked as complete');
  } catch (error) {
    // If settings table doesn't exist, that's okay for now
    logger.warn('[JSONB MIGRATION] Could not mark migration complete (settings table may not exist)', error);
  }
}

// Run migration if called directly
if (require.main === module) {
  migrateJsonbResults()
    .then(() => {
      console.log('✅ JSONB migration completed');
      process.exit(0);
    })
    .catch((error) => {
      console.error('❌ JSONB migration failed:', error);
      process.exit(1);
    });
}
