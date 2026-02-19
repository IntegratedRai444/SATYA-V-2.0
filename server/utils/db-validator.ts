/**
 * Database Safety Validation Utility
 * Validates database tables and columns with safe handling
 */

import { supabase } from '../config/supabase';
import { logger } from '../config/logger';

export interface TableSchema {
  name: string;
  requiredColumns: string[];
  optionalColumns: string[];
}

// Define expected table schemas
const REQUIRED_TABLES: TableSchema[] = [
  {
    name: 'tasks',
    requiredColumns: ['id', 'user_id', 'type', 'status', 'created_at'],
    optionalColumns: ['confidence', 'is_deepfake', 'model_name', 'model_version', 'analysis_data', 'proof_json', 'file_name', 'mime_type', 'size_bytes', 'updated_at', 'completed_at', 'error_message', 'deleted_at']
  },
  {
    name: 'users',
    requiredColumns: ['id', 'email', 'created_at'],
    optionalColumns: ['user_metadata', 'email_confirmed_at', 'last_sign_in_at']
  },
  {
    name: 'chat_messages',
    requiredColumns: ['id', 'user_id', 'message', 'created_at'],
    optionalColumns: ['role', 'metadata', 'updated_at']
  }
];

/**
 * Validates if a table exists in the database
 */
export const validateTableExists = async (tableName: string): Promise<boolean> => {
  try {
    const { data, error } = await supabase
      .from('information_schema.tables')
      .select('table_name')
      .eq('table_schema', 'public')
      .eq('table_name', tableName);
    
    if (error) {
      logger.error(`[DB] Table validation failed for ${tableName}:`, error);
      return false;
    }
    
    return data && data.length > 0;
  } catch (error) {
    logger.error(`[DB] Table validation error for ${tableName}:`, error);
    return false;
  }
};

/**
 * Validates if all required columns exist in a table
 */
export const validateTableColumns = async (tableName: string): Promise<boolean> => {
  try {
    const { data, error } = await supabase
      .from('information_schema.columns')
      .select('column_name')
      .eq('table_schema', 'public')
      .eq('table_name', tableName);
    
    if (error) {
      logger.error(`[DB] Column validation failed for ${tableName}:`, error);
      return false;
    }
    
    const tableSchema = REQUIRED_TABLES.find(t => t.name === tableName);
    if (!tableSchema) {
      logger.warn(`[DB] Table ${tableName} not found in schema definition`);
      return false;
    }
    
    const existingColumns = data?.map(col => col.column_name) || [];
    const missingRequired = tableSchema.requiredColumns.filter(col => !existingColumns.includes(col));
    
    if (missingRequired.length > 0) {
      logger.error(`[DB] Missing required columns in ${tableName}:`, missingRequired);
      return false;
    }
    
    logger.info(`[DB] Table ${tableName} validation passed`);
    return true;
  } catch (error) {
    logger.error(`[DB] Column validation error for ${tableName}:`, error);
    return false;
  }
};

/**
 * Validates all required tables and columns
 */
export const validateDatabaseSchema = async (): Promise<{ valid: boolean; errors: string[] }> => {
  const errors: string[] = [];
  
  for (const table of REQUIRED_TABLES) {
    const tableExists = await validateTableExists(table.name);
    if (!tableExists) {
      errors.push(`Table ${table.name} does not exist`);
      continue;
    }
    
    const columnsValid = await validateTableColumns(table.name);
    if (!columnsValid) {
      errors.push(`Table ${table.name} has invalid column structure`);
    }
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
};

/**
 * Safe query wrapper with null handling
 */
export const safeQuery = async <T>(
  queryFn: () => Promise<{ data?: T; error?: unknown }>,
  defaultValue: T
): Promise<T> => {
  try {
    const result = await queryFn();
    
    if (result.error) {
      logger.error('[DB] Query failed:', result.error);
      throw new Error('Database query failed');
    }
    
    return result.data || defaultValue;
  } catch (error) {
    logger.error('[DB] Query error:', error);
    throw new Error('Database operation failed');
  }
};
