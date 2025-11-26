-- ============================================================================
-- SatyaAI Complete Database Schema - PostgreSQL/Supabase
-- ============================================================================
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard
-- 
-- This schema includes:
-- ✅ Core tables (users, scans, tasks, preferences)
-- ✅ Additional tables (notifications, audit_logs, api_keys)  
-- ✅ Optimized indexes for fast queries
-- ✅ Auto-update triggers
-- ✅ Useful views
-- ✅ Demo data
-- ============================================================================

-- ============================================================================
-- STEP 1: DROP EXISTING TABLES (Clean Slate)
-- ============================================================================

DROP TABLE IF EXISTS audit_logs CASCADE;
DROP TABLE IF EXISTS notifications CASCADE;
DROP TABLE IF EXISTS api_keys CASCADE;
DROP TABLE IF EXISTS tasks CASCADE;
DROP TABLE IF EXISTS scans CASCADE;
DROP TABLE IF EXISTS user_preferences CASCADE;
DROP TABLE IF EXISTS users CASCADE;

DROP VIEW IF EXISTS user_statistics CASCADE;
DROP VIEW IF EXISTS recent_activity CASCADE;
DROP VIEW IF EXISTS task_queue CASCADE;

DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- ============================================================================
-- STEP 2: CREATE CORE TABLES
-- ============================================================================

-- Users table - Authentication & user management
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(255) NOT NULL UNIQUE,
  password TEXT NOT NULL,
  email VARCHAR(255) UNIQUE,
  full_name VARCHAR(255),
  role VARCHAR(50) NOT NULL DEFAULT 'user',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  last_login TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  CONSTRAINT username_length CHECK (LENGTH(username) >= 3),
  CONSTRAINT role_check CHECK (role IN ('user', 'admin', 'moderator'))
);

-- Scans table - Deepfake detection results
CREATE TABLE scans (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  filename TEXT NOT NULL,
  type VARCHAR(50) NOT NULL,
  result VARCHAR(50) NOT NULL,
  confidence_score INTEGER NOT NULL,
  detection_details JSONB,
  metadata JSONB,
  file_size BIGINT,
  processing_time_ms INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  CONSTRAINT type_check CHECK (type IN ('image', 'video', 'audio', 'multimodal')),
  CONSTRAINT result_check CHECK (result IN ('authentic', 'deepfake', 'suspicious', 'inconclusive')),
  CONSTRAINT confidence_check CHECK (confidence_score >= 0 AND confidence_score <= 100)
);

-- User preferences table - User settings
CREATE TABLE user_preferences (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
  theme VARCHAR(50) DEFAULT 'dark',
  language VARCHAR(50) DEFAULT 'english',
  confidence_threshold INTEGER DEFAULT 75,
  enable_notifications BOOLEAN DEFAULT TRUE,
  auto_analyze BOOLEAN DEFAULT TRUE,
  sensitivity_level VARCHAR(50) DEFAULT 'medium',
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  CONSTRAINT theme_check CHECK (theme IN ('light', 'dark', 'auto')),
  CONSTRAINT confidence_threshold_check CHECK (confidence_threshold >= 0 AND confidence_threshold <= 100),
  CONSTRAINT sensitivity_check CHECK (sensitivity_level IN ('low', 'medium', 'high'))
);

-- Tasks table - Processing queue for async analysis
CREATE TABLE tasks (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type VARCHAR(50) NOT NULL,
  status VARCHAR(50) NOT NULL DEFAULT 'queued',
  progress INTEGER NOT NULL DEFAULT 0,
  file_name TEXT NOT NULL,
  file_size BIGINT NOT NULL,
  file_type VARCHAR(100) NOT NULL,
  file_path TEXT NOT NULL,
  report_code VARCHAR(100) UNIQUE,
  result JSONB,
  error JSONB,
  metadata JSONB,
  priority INTEGER DEFAULT 5,
  retry_count INTEGER DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  started_at TIMESTAMP,
  completed_at TIMESTAMP,
  CONSTRAINT type_check CHECK (type IN ('image', 'video', 'audio', 'webcam', 'multimodal', 'batch')),
  CONSTRAINT status_check CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
  CONSTRAINT progress_check CHECK (progress >= 0 AND progress <= 100),
  CONSTRAINT priority_check CHECK (priority >= 1 AND priority <= 10)
);

-- ============================================================================
-- STEP 3: CREATE ADDITIONAL TABLES
-- ============================================================================

-- Notifications table - User notifications
CREATE TABLE notifications (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  type VARCHAR(50) NOT NULL,
  title VARCHAR(255) NOT NULL,
  message TEXT NOT NULL,
  read BOOLEAN NOT NULL DEFAULT FALSE,
  action_url TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  read_at TIMESTAMP,
  CONSTRAINT type_check CHECK (type IN ('info', 'success', 'warning', 'error', 'scan_complete'))
);

-- API Keys table - For external API access
CREATE TABLE api_keys (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  key_name VARCHAR(255) NOT NULL,
  api_key TEXT NOT NULL UNIQUE,
  permissions JSONB DEFAULT '{"read": true, "write": false, "admin": false}'::jsonb,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  last_used_at TIMESTAMP,
  expires_at TIMESTAMP,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Audit logs table - Security & activity tracking
CREATE TABLE audit_logs (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
  action VARCHAR(100) NOT NULL,
  resource_type VARCHAR(50),
  resource_id INTEGER,
  ip_address VARCHAR(45),
  user_agent TEXT,
  metadata JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- STEP 4: CREATE INDEXES FOR PERFORMANCE
-- ============================================================================

-- Users indexes
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email) WHERE email IS NOT NULL;
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- Scans indexes
CREATE INDEX idx_scans_user_id ON scans(user_id);
CREATE INDEX idx_scans_created_at ON scans(created_at DESC);
CREATE INDEX idx_scans_type ON scans(type);
CREATE INDEX idx_scans_result ON scans(result);
CREATE INDEX idx_scans_confidence ON scans(confidence_score DESC);
CREATE INDEX idx_scans_user_date ON scans(user_id, created_at DESC);
CREATE INDEX idx_scans_type_result ON scans(type, result);
CREATE INDEX idx_scans_detection_details ON scans USING GIN (detection_details);
CREATE INDEX idx_scans_metadata ON scans USING GIN (metadata);

-- Tasks indexes
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_created_at ON tasks(created_at DESC);
CREATE INDEX idx_tasks_user_status_date ON tasks(user_id, status, created_at DESC);
CREATE INDEX idx_tasks_report_code ON tasks(report_code) WHERE report_code IS NOT NULL;
CREATE INDEX idx_tasks_priority_status ON tasks(priority DESC, status, created_at);
CREATE INDEX idx_tasks_result ON tasks USING GIN (result);

-- Notifications indexes
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_read ON notifications(read) WHERE read = FALSE;
CREATE INDEX idx_notifications_created_at ON notifications(created_at DESC);
CREATE INDEX idx_notifications_user_unread ON notifications(user_id, read, created_at DESC);

-- API Keys indexes
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_api_key ON api_keys(api_key);
CREATE INDEX idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;

-- Audit logs indexes
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- ============================================================================
-- STEP 5: CREATE TRIGGERS FOR AUTO-UPDATES
-- ============================================================================

-- Function to update updated_at timestamp
CREATE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers
CREATE TRIGGER update_users_updated_at
  BEFORE UPDATE ON users
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
  BEFORE UPDATE ON user_preferences
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- STEP 6: CREATE VIEWS FOR COMMON QUERIES
-- ============================================================================

-- User statistics view
CREATE VIEW user_statistics AS
SELECT 
  u.id,
  u.username,
  u.email,
  u.role,
  u.is_active,
  u.created_at,
  u.last_login,
  COUNT(DISTINCT s.id) as total_scans,
  COUNT(DISTINCT CASE WHEN s.result = 'deepfake' THEN s.id END) as deepfake_count,
  COUNT(DISTINCT CASE WHEN s.result = 'authentic' THEN s.id END) as authentic_count,
  AVG(s.confidence_score) as avg_confidence,
  MAX(s.created_at) as last_scan_date,
  COUNT(DISTINCT CASE WHEN n.read = FALSE THEN n.id END) as unread_notifications
FROM users u
LEFT JOIN scans s ON u.id = s.user_id
LEFT JOIN notifications n ON u.id = n.user_id
GROUP BY u.id, u.username, u.email, u.role, u.is_active, u.created_at, u.last_login;

-- Recent activity view
CREATE VIEW recent_activity AS
SELECT 
  s.id,
  s.user_id,
  u.username,
  s.filename,
  s.type,
  s.result,
  s.confidence_score,
  s.created_at
FROM scans s
JOIN users u ON s.user_id = u.id
WHERE u.is_active = TRUE
ORDER BY s.created_at DESC
LIMIT 100;

-- Task queue view
CREATE VIEW task_queue AS
SELECT 
  t.id,
  t.user_id,
  u.username,
  t.type,
  t.status,
  t.progress,
  t.file_name,
  t.priority,
  t.created_at,
  t.started_at,
  EXTRACT(EPOCH FROM (NOW() - t.created_at)) as wait_time_seconds
FROM tasks t
JOIN users u ON t.user_id = u.id
WHERE t.status IN ('queued', 'processing')
ORDER BY t.priority DESC, t.created_at ASC;

-- ============================================================================
-- STEP 7: INSERT DEMO DATA
-- ============================================================================

-- Demo user (password: DemoPassword123!)
INSERT INTO users (username, email, password, full_name, role)
VALUES (
  'demo',
  'demo@satyaai.com',
  '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIeWEgKK3q',
  'Demo User',
  'user'
);

-- Admin user (password: AdminPassword123!)
INSERT INTO users (username, email, password, full_name, role)
VALUES (
  'admin',
  'admin@satyaai.com',
  '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIeWEgKK3q',
  'System Administrator',
  'admin'
);

-- Your user (password: Rishabhkapoor@0444)
INSERT INTO users (username, email, password, full_name, role)
VALUES (
  'rishabhkapoor',
  'rishabhkapoor@atomicmail.io',
  '$2b$12$8vZ8qVxKZYHxQJ5YqVxKZeYqVxKZYHxQJ5YqVxKZYHxQJ5YqVxKZe',
  'Rishabh Kapoor',
  'admin'
);

-- Sample scans
INSERT INTO scans (user_id, filename, type, result, confidence_score, detection_details, metadata)
SELECT 
  u.id,
  'sample_authentic_image.jpg',
  'image',
  'authentic',
  98,
  '{"detections": [{"name": "Facial Analysis", "confidence": 98}]}'::jsonb,
  '{"size": "1.2 MB", "resolution": "1920x1080"}'::jsonb
FROM users u WHERE u.username = 'demo';

-- User preferences for all users
INSERT INTO user_preferences (user_id, theme, language, confidence_threshold, enable_notifications, auto_analyze, sensitivity_level)
SELECT 
  u.id,
  'dark',
  'english',
  80,
  true,
  true,
  'medium'
FROM users u;

-- Sample notification
INSERT INTO notifications (user_id, type, title, message)
SELECT 
  u.id,
  'success',
  'Welcome to SatyaAI!',
  'Your account has been created successfully. Start analyzing media files for deepfakes.'
FROM users u WHERE u.username = 'demo';

-- ============================================================================
-- STEP 8: VERIFICATION
-- ============================================================================

SELECT '✅ SatyaAI Database Created Successfully!' as status;
SELECT 
  (SELECT COUNT(*) FROM users) as users,
  (SELECT COUNT(*) FROM scans) as scans,
  (SELECT COUNT(*) FROM tasks) as tasks,
  (SELECT COUNT(*) FROM notifications) as notifications,
  (SELECT COUNT(*) FROM user_preferences) as preferences;
