-- Database Optimization Migration
-- Created: 2025-01-11
-- Purpose: Add indexes for performance optimization

-- Scans table indexes
CREATE INDEX IF NOT EXISTS idx_scans_user_id ON scans(user_id);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);
CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(status);
CREATE INDEX IF NOT EXISTS idx_scans_type ON scans(type);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_scans_user_status ON scans(user_id, status);
CREATE INDEX IF NOT EXISTS idx_scans_user_created ON scans(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_scans_user_type ON scans(user_id, type);

-- Analysis results indexes
CREATE INDEX IF NOT EXISTS idx_analysis_results_scan_id ON analysis_results(scan_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON analysis_results(created_at);

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- Sessions table indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

-- Notifications table indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_read ON notifications(read);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at);

-- Composite index for unread notifications query
CREATE INDEX IF NOT EXISTS idx_notifications_user_unread 
  ON notifications(user_id, read, created_at DESC);

-- Add foreign key constraints if not exists
-- Note: SQLite has limited ALTER TABLE support, these should be in initial schema

-- Performance optimization: Analyze tables
ANALYZE scans;
ANALYZE users;
ANALYZE analysis_results;
ANALYZE sessions;
ANALYZE notifications;
