-- ============================================================================
-- Auth System Enhancements
-- Adds OAuth 2.0 and JWT token management
-- ============================================================================

-- Enable UUID extension for secure token generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. Update Users Table for OAuth
-- ============================================================================

-- Add OAuth provider columns to users table
ALTER TABLE users 
  ADD COLUMN IF NOT EXISTS provider VARCHAR(50) DEFAULT 'email',
  ADD COLUMN IF NOT EXISTS provider_id TEXT,
  ADD COLUMN IF NOT EXISTS avatar_url TEXT,
  ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS verification_token TEXT,
  ADD COLUMN IF NOT EXISTS verification_expires TIMESTAMP WITH TIME ZONE,
  ADD COLUMN IF NOT EXISTS reset_token TEXT,
  ADD COLUMN IF NOT EXISTS reset_expires TIMESTAMP WITH TIME ZONE,
  ADD COLUMN IF NOT EXISTS last_ip INET,
  ADD COLUMN IF NOT EXISTS login_count INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS last_failed_login TIMESTAMP WITH TIME ZONE,
  ADD COLUMN IF NOT EXISTS last_password_change TIMESTAMP WITH TIME ZONE,
  ADD COLUMN IF NOT EXISTS timezone VARCHAR(50) DEFAULT 'UTC',
  ADD COLUMN IF NOT EXISTS locale VARCHAR(10) DEFAULT 'en-US';

-- Make password nullable for OAuth users
ALTER TABLE users ALTER COLUMN password DROP NOT NULL;

-- Add constraints for OAuth
ALTER TABLE users 
  ADD CONSTRAINT users_provider_check 
  CHECK (provider IN ('email', 'google', 'github', 'microsoft', 'apple')),
  ADD CONSTRAINT users_provider_id_check 
  CHECK ((provider = 'email' AND password IS NOT NULL) OR (provider != 'email' AND provider_id IS NOT NULL));

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_provider ON users(provider, provider_id);

-- ============================================================================
-- 2. Create Sessions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS sessions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  refresh_token TEXT NOT NULL,
  user_agent TEXT,
  ip_address INET,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  is_revoked BOOLEAN DEFAULT FALSE,
  revoked_at TIMESTAMP WITH TIME ZONE,
  last_used_at TIMESTAMP WITH TIME ZONE,
  metadata JSONB
);

-- Indexes for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_refresh_token ON sessions(refresh_token) WHERE NOT is_revoked;
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at) WHERE NOT is_revoked;

-- ============================================================================
-- 3. Create OAuth States Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS oauth_states (
  state TEXT PRIMARY KEY,
  provider VARCHAR(50) NOT NULL,
  redirect_uri TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  metadata JSONB
);

-- Cleanup expired states
CREATE OR REPLACE FUNCTION cleanup_expired_oauth_states()
RETURNS TRIGGER AS $$
BEGIN
  DELETE FROM oauth_states WHERE expires_at < NOW();
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for cleanup
CREATE TRIGGER trigger_cleanup_oauth_states
AFTER INSERT ON oauth_states
EXECUTE FUNCTION cleanup_expired_oauth_states();

-- ============================================================================
-- 4. Create Revoked Tokens Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS revoked_tokens (
  jti TEXT PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  revoked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  reason TEXT
);

-- Index for cleanup
CREATE INDEX IF NOT EXISTS idx_revoked_tokens_expires ON revoked_tokens(expires_at);

-- ============================================================================
-- 5. Create Audit Logs Table (if not exists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
  id BIGSERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  action VARCHAR(100) NOT NULL,
  entity_type VARCHAR(50),
  entity_id INTEGER,
  ip_address INET,
  user_agent TEXT,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for audit logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);

-- ============================================================================
-- 6. Create Functions
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to create audit log entry
CREATE OR REPLACE FUNCTION create_audit_log(
  p_user_id INTEGER,
  p_action VARCHAR(100),
  p_entity_type VARCHAR(50) DEFAULT NULL,
  p_entity_id INTEGER DEFAULT NULL,
  p_ip_address INET DEFAULT NULL,
  p_user_agent TEXT DEFAULT NULL,
  p_metadata JSONB DEFAULT NULL
) RETURNS BIGINT AS $$
DECLARE
  log_id BIGINT;
BEGIN
  INSERT INTO audit_logs (
    user_id, 
    action, 
    entity_type, 
    entity_id, 
    ip_address, 
    user_agent, 
    metadata
  ) VALUES (
    p_user_id, 
    p_action, 
    p_entity_type, 
    p_entity_id, 
    p_ip_address, 
    p_user_agent, 
    p_metadata
  ) RETURNING id INTO log_id;
  
  RETURN log_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 7. Create Triggers
-- ============================================================================

-- Update triggers for updated_at
CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at
BEFORE UPDATE ON sessions
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 8. Create Views
-- ============================================================================

-- View for active sessions
CREATE OR REPLACE VIEW active_sessions AS
SELECT 
  s.*,
  u.username,
  u.email
FROM sessions s
JOIN users u ON s.user_id = u.id
WHERE s.is_revoked = FALSE 
  AND s.expires_at > NOW();

-- View for user login stats
CREATE OR REPLACE VIEW user_login_stats AS
SELECT 
  u.id,
  u.username,
  u.email,
  u.role,
  u.last_login,
  u.login_count,
  u.failed_login_attempts,
  u.last_failed_login,
  (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id AND NOT s.is_revoked) AS active_sessions
FROM users u;

-- ============================================================================
-- 9. Add Sample Data (for development)
-- ============================================================================

-- Insert a default admin user if not exists (password: Admin@123)
INSERT INTO users (
  username, 
  password, 
  email, 
  full_name, 
  role, 
  is_active, 
  is_verified,
  provider
) VALUES (
  'admin', 
  '$2b$10$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 
  'admin@satyaai.live', 
  'Admin User', 
  'admin', 
  TRUE, 
  TRUE,
  'email'
) ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- 10. Grant Permissions
-- ============================================================================

-- Grant necessary permissions to the database user
-- Replace 'your_db_user' with your actual database user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO your_db_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_db_user;

-- ============================================================================
-- 11. Create Indexes for Performance
-- ============================================================================

-- Add any additional indexes needed for your queries
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit_logs(entity_type, entity_id);

-- ============================================================================
-- 12. Cleanup Function
-- ============================================================================

-- Function to cleanup expired sessions and revoked tokens
CREATE OR REPLACE FUNCTION cleanup_expired_auth_data()
RETURNS VOID AS $$
BEGIN
  -- Delete expired sessions
  DELETE FROM sessions 
  WHERE expires_at < NOW() 
  RETURNING id, user_id, expires_at 
  INTO TEMP expired_sessions;
  
  -- Log the cleanup
  INSERT INTO audit_logs (action, entity_type, metadata)
  SELECT 
    'session_expired', 
    'session', 
    jsonb_build_object(
      'session_id', id,
      'user_id', user_id,
      'expired_at', expires_at
    )
  FROM expired_sessions;
  
  -- Delete expired revoked tokens
  DELETE FROM revoked_tokens 
  WHERE expires_at < NOW();
  
  -- Delete expired OAuth states
  DELETE FROM oauth_states 
  WHERE expires_at < NOW();
  
  -- Delete old audit logs (keep 90 days)
  DELETE FROM audit_logs 
  WHERE created_at < (NOW() - INTERVAL '90 days');
  
  RETURN;
END;
$$ LANGUAGE plpgsql;

-- Schedule the cleanup function to run daily
-- Uncomment and run this in your database after creating the function
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('0 3 * * *', 'SELECT cleanup_expired_auth_data()');

-- ============================================================================
-- 13. Update User Preferences Table
-- ============================================================================

-- Add OAuth-related preferences if they don't exist
ALTER TABLE user_preferences 
  ADD COLUMN IF NOT EXISTS email_notifications BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS push_notifications BOOLEAN DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS two_factor_enabled BOOLEAN DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS preferred_provider VARCHAR(50);

-- Add constraint for preferred provider
ALTER TABLE user_preferences 
  ADD CONSTRAINT preferred_provider_check 
  CHECK (preferred_provider IS NULL OR preferred_provider IN ('email', 'google', 'github', 'microsoft', 'apple'));

-- ============================================================================
-- 14. Create OAuth Provider Config Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS oauth_providers (
  provider VARCHAR(50) PRIMARY KEY,
  client_id TEXT NOT NULL,
  client_secret TEXT NOT NULL,
  is_enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  metadata JSONB
);

-- Add update trigger for oauth_providers
CREATE TRIGGER update_oauth_providers_updated_at
BEFORE UPDATE ON oauth_providers
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 15. Create User Tokens Table (for email verification, password reset, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_tokens (
  token TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type VARCHAR(50) NOT NULL,
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  used_at TIMESTAMP WITH TIME ZONE,
  metadata JSONB
);

-- Indexes for user_tokens
CREATE INDEX IF NOT EXISTS idx_user_tokens_user_id ON user_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_user_tokens_type ON user_tokens(type);
CREATE INDEX IF NOT EXISTS idx_user_tokens_expires ON user_tokens(expires_at);

-- Add constraint for token type
ALTER TABLE user_tokens 
  ADD CONSTRAINT token_type_check 
  CHECK (type IN ('email_verification', 'password_reset', 'magic_link', 'api_key'));

-- ============================================================================
-- 16. Create API Keys Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS api_keys (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  key_hash TEXT NOT NULL,
  last_used_at TIMESTAMP WITH TIME ZONE,
  expires_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  is_active BOOLEAN DEFAULT TRUE,
  metadata JSONB
);

-- Indexes for api_keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);

-- Add update trigger for api_keys
CREATE TRIGGER update_api_keys_updated_at
BEFORE UPDATE ON api_keys
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 17. Create User Sessions View
-- ============================================================================

CREATE OR REPLACE VIEW user_sessions AS
SELECT 
  s.id,
  s.user_id,
  u.username,
  u.email,
  s.user_agent,
  s.ip_address,
  s.created_at,
  s.updated_at,
  s.last_used_at,
  s.expires_at,
  s.is_revoked,
  s.revoked_at,
  CASE 
    WHEN s.is_revoked THEN 'revoked'
    WHEN s.expires_at < NOW() THEN 'expired'
    WHEN s.last_used_at IS NULL THEN 'new'
    ELSE 'active'
  END AS status
FROM sessions s
JOIN users u ON s.user_id = u.id;

-- ============================================================================
-- 18. Create User Activity View
-- ============================================================================

CREATE OR REPLACE VIEW user_activity AS
SELECT 
  u.id AS user_id,
  u.username,
  u.email,
  u.role,
  u.last_login,
  u.login_count,
  COUNT(DISTINCT s.id) FILTER (WHERE NOT s.is_revoked) AS active_sessions,
  COUNT(DISTINCT a.id) FILTER (WHERE a.action = 'login_success') AS successful_logins,
  COUNT(DISTINCT a.id) FILTER (WHERE a.action = 'login_failed') AS failed_logins,
  MAX(a.created_at) FILTER (WHERE a.action = 'login_success') AS last_successful_login,
  MAX(a.created_at) FILTER (WHERE a.action = 'login_failed') AS last_failed_login,
  COUNT(DISTINCT t.id) AS total_tasks,
  COUNT(DISTINCT sc.id) AS total_scans
FROM users u
LEFT JOIN sessions s ON u.id = s.user_id AND NOT s.is_revoked AND s.expires_at > NOW()
LEFT JOIN audit_logs a ON u.id = a.user_id AND a.action IN ('login_success', 'login_failed')
LEFT JOIN tasks t ON u.id = t.user_id
LEFT JOIN scans sc ON u.id = sc.user_id
GROUP BY u.id, u.username, u.email, u.role, u.last_login, u.login_count;

-- ============================================================================
-- 19. Create Function to Generate Random String
-- ============================================================================

CREATE OR REPLACE FUNCTION generate_random_string(length INTEGER) 
RETURNS TEXT AS $$
DECLARE
  chars TEXT := 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  result TEXT := '';
  i INTEGER := 0;
  rand_int INTEGER;
BEGIN
  FOR i IN 1..length LOOP
    rand_int := floor(random() * length(chars) + 1)::INTEGER;
    result := result || substr(chars, rand_int, 1);
  END LOOP;
  RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 20. Create Function to Generate Secure Random Token
-- ============================================================================

CREATE OR REPLACE FUNCTION generate_secure_token(length INTEGER DEFAULT 64)
RETURNS TEXT AS $$
BEGIN
  RETURN encode(gen_random_bytes((length + 1) / 2), 'hex');
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- END OF MIGRATION
-- ============================================================================

-- Print success message
DO $$
BEGIN
  RAISE NOTICE '✅ Auth system enhancements migration completed successfully';
END $$;
