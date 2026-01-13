import { createClient } from '@supabase/supabase-js';
import { logger } from './config/logger';
import { config } from './config/environment';

// Define a basic Database type that can be extended with your schema
type Database = {
  public: {
    Tables: Record<string, {
      Row: Record<string, any>;
      Insert: Record<string, any>;
      Update: Record<string, any>;
    }>;
  };
};

// Extract the Tables type from Database
type Tables = Database['public']['Tables'];

// Table name type (string literal)
type TableName = keyof Tables & string;

// Row type for a table
type TableRow<T extends TableName> = Tables[T] extends { Row: infer R } ? R : never;

// Insert type for a table
type TableInsert<T extends TableName> = Tables[T] extends { Insert: infer I } ? I : never;

// Update type for a table
type TableUpdate<T extends TableName> = Tables[T] extends { Update: infer U } ? U : never;

// Initialize Supabase client
const supabaseUrl = config.SUPABASE_URL;
const supabaseKey = config.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error('Missing required Supabase configuration');
}

// Single Supabase client instance
export const supabase = createClient<Database>(supabaseUrl, supabaseKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false,
  },
  db: {
    schema: 'public',
  },
});

class DatabaseManager {
  private static instance: DatabaseManager;
  private isInitialized = false;
  private isInitializing = false;
  private initPromise: Promise<void> | null = null;

  private constructor() {}

  public static getInstance(): DatabaseManager {
    if (!DatabaseManager.instance) {
      DatabaseManager.instance = new DatabaseManager();
    }
    return DatabaseManager.instance;
  }

  private validateConfig() {
    if (!config.SUPABASE_URL || !config.SUPABASE_SERVICE_ROLE_KEY) {
      throw new Error('Supabase configuration is required');
    }
  }

  async initialize() {
    if (this.isInitialized) return;
    if (this.isInitializing && this.initPromise) {
      return this.initPromise;
    }

    this.isInitializing = true;
    this.initPromise = (async () => {
      try {
        this.validateConfig();
        await this.testConnection();
        this.isInitialized = true;
        logger.info('✅ Database connection established');
      } catch (error) {
        logger.error('❌ Failed to initialize database:', error);
        throw error;
      } finally {
        this.isInitializing = false;
      }
    })();

    return this.initPromise;
  }

  private async testConnection() {
    try {
      const { data, error } = await supabase.rpc('version');
      if (error) throw error;
      logger.debug('Database version:', data);
      return true;
    } catch (error) {
      logger.error('Database connection test failed:', error);
      throw error;
    }
  }

  public get client() {
    return supabase;
  }

  public async close() {
    // Supabase client handles its own connection pooling
    logger.info('Supabase client closed');
  }

  // Common database operations
  async findById<T extends TableName>(
    table: T,
    id: string
  ): Promise<TableRow<T> | null> {
    try {
      const { data, error } = await supabase
        .from(String(table))
        .select('*')
        .eq('id', id)
        .maybeSingle();

      if (error) {
        logger.error(`Error finding ${String(table)} by ID:`, error);
        throw error;
      }

      return data as TableRow<T> | null;
    } catch (error) {
      logger.error(`Error in findById for table ${String(table)}:`, error);
      throw error;
    }
  }

  async find<T extends TableName>(
    table: T,
    filters: Record<string, any> = {},
    options: {
      limit?: number;
      offset?: number;
      orderBy?: { column: string; ascending: boolean };
    } = {}
  ): Promise<TableRow<T>[]> {
    try {
      let query = supabase.from(String(table)).select('*');

      // Apply filters
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined) {
          query = query.eq(key, value);
        }
      });

      // Apply options
      if (options.limit) query = query.limit(options.limit);
      if (options.offset) query = query.range(options.offset, options.offset + (options.limit || 1) - 1);
      if (options.orderBy) {
        query = query.order(options.orderBy.column, { 
          ascending: options.orderBy.ascending 
        });
      }

      const { data, error } = await query;

      if (error) {
        logger.error(`Error finding ${String(table)}:`, error);
        throw error;
      }

      return (data || []) as TableRow<T>[];
    } catch (error) {
      logger.error(`Error in find for table ${String(table)}:`, error);
      throw error;
    }
  }

  async create<T extends TableName>(
    table: T,
    data: TableInsert<T>
  ): Promise<TableRow<T>> {
    try {
      const { data: result, error } = await supabase
        .from(String(table))
        .insert(data as any)
        .select()
        .single();

      if (error) {
        logger.error(`Error creating ${String(table)}:`, error);
        throw error;
      }

      return result as TableRow<T>;
    } catch (error) {
      logger.error(`Error in create for table ${String(table)}:`, error);
      throw error;
    }
  }

  async update<T extends TableName>(
    table: T,
    id: string,
    data: TableUpdate<T>
  ): Promise<TableRow<T>> {
    try {
      // Use type assertion to handle the update operation
      const { data: result, error } = await (supabase
        .from(String(table)) as any)
        .update(data)
        .eq('id', id)
        .select()
        .single();

      if (error) {
        logger.error(`Error updating ${String(table)}:`, error);
        throw error;
      }

      if (!result) {
        throw new Error(`No record found with id ${id} in table ${String(table)}`);
      }

      return result as TableRow<T>;
    } catch (error) {
      logger.error(`Error in update for table ${String(table)}:`, error);
      throw error;
    }
  }

  async delete<T extends TableName>(table: T, id: string): Promise<boolean> {
    try {
      const { error, count } = await supabase
        .from(String(table))
        .delete()
        .eq('id', id);

      if (error) {
        logger.error(`Error deleting from ${String(table)}:`, error);
        throw error;
      }

      if (count === 0) {
        logger.warn(`No record found with id ${id} in table ${String(table)}`);
        return false;
      }

      return true;
    } catch (error) {
      logger.error(`Error in delete for table ${String(table)}:`, error);
      throw error;
    }
  }

  async execute<T = any>(query: string, params: any[] = []): Promise<T[]> {
    try {
      // Use the correct RPC method for executing raw SQL
      const { data, error } = await (supabase as any).rpc('execute_sql', {
        query,
        params: params || []
      });

      if (error) {
        logger.error('Error executing raw query:', error);
        throw error;
      }

      // Handle the response based on the actual structure
      if (data && Array.isArray(data)) {
        return data as T[];
      } else if (data && (data as any).rows) {
        return (data as any).rows as T[];
      }
      
      return [];
    } catch (error) {
      logger.error('Error in execute query:', error);
      throw error;
    }
  }
}

// Create and export database manager instance
export const dbManager = DatabaseManager.getInstance();

// Check if database is connected
export const isDbConnected = async (): Promise<boolean> => {
  try {
    // Use a simple query to test the connection
    const { error } = await supabase.from('pg_tables').select('*').limit(1);
    return !error;
  } catch (error) {
    logger.error('Database connection check failed:', error);
    return false;
  }
};

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received. Closing database connections...');
  
  try {
    await dbManager.close();
    logger.info('Database connections closed successfully');
  } catch (error) {
    logger.error('Error closing database connections:', error);
  } finally {
    process.exit(0);
  }
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received. Closing database connections...');
  
  try {
    await dbManager.close();
    logger.info('Database connections closed successfully');
  } catch (error) {
    logger.error('Error closing database connections:', error);
  } finally {
    process.exit(0);
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', async (error) => {
  logger.error('Uncaught exception:', error);
  
  try {
    await dbManager.close();
  } catch (err) {
    logger.error('Error closing database connections during uncaught exception:', err);
  } finally {
    process.exit(1);
  }
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  // Consider whether to exit the process here
});

// Initialize the database connection when this module is imported
dbManager.initialize().catch((error) => {
  logger.error('Failed to initialize database:', error);
  process.exit(1);
});
