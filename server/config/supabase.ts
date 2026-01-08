import { createClient } from '@supabase/supabase-js';
import { config } from './environment';
import { logger } from './logger';

// Use the centralized configuration
export const supabaseConfig = {
  url: config.SUPABASE_URL,
  anonKey: config.SUPABASE_ANON_KEY,
  serviceRoleKey: config.SUPABASE_SERVICE_ROLE_KEY,
  jwtSecret: config.SUPABASE_JWT_SECRET,
  auth: {
    autoRefreshToken: true,
    persistSession: false,
    detectSessionInUrl: false,
  },
  db: {
    schema: 'public',
  },
} as const;

// Create Supabase clients
export const supabase = createClient(
  supabaseConfig.url,
  supabaseConfig.anonKey,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
      detectSessionInUrl: false,
    },
  }
);

export const supabaseAdmin = createClient(
  supabaseConfig.url,
  supabaseConfig.serviceRoleKey,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
      detectSessionInUrl: false,
    },
  }
);

// Test Supabase connection
export async function testSupabaseConnection() {
  try {
    const { data, error } = await supabase.auth.getSession();
    
    if (error) {
      logger.error('Supabase connection test failed', { error: error.message });
      return false;
    }
    
    logger.info('✅ Supabase client connected successfully');
    return true;
  } catch (error) {
    logger.error('Supabase connection error', { error });
    return false;
  }
}

// Verify JWT from Authorization header
export async function verifyAuthHeader(authHeader: string | undefined) {
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return { valid: false, error: 'Invalid authorization header' };
  }

  const token = authHeader.split(' ')[1];
  
  try {
    const { data: { user }, error } = await supabase.auth.getUser(token);
    
    if (error || !user) {
      return { valid: false, error: error?.message || 'Invalid token' };
    }
    
    return { valid: true, user };
  } catch (error) {
    return { valid: false, error: 'Token verification failed' };
  }
}

// Middleware to protect routes
export const requireAuth = async (req: any, res: any, next: any) => {
  const authHeader = req.headers.authorization || req.headers.Authorization;
  const { valid, user, error } = await verifyAuthHeader(authHeader);
  
  if (!valid) {
    return res.status(401).json({ 
      success: false, 
      error: error || 'Not authenticated' 
    });
  }
  
  req.user = user;
  next();
};

export default {
  supabase,
  supabaseAdmin,
  config: supabaseConfig,
  testConnection: testSupabaseConnection,
  verifyAuthHeader,
  requireAuth,
};
