import { supabase } from './config/supabase';

// Database connection using Supabase
export const db = supabase;

// Helper function for database operations
export const executeQuery = async (query: string, params?: any[]) => {
  try {
    const { data, error } = await supabase.rpc('execute_sql', { query, params });
    if (error) throw error;
    return data;
  } catch (error) {
    console.error('Database query error:', error);
    throw error;
  }
};

export default db;