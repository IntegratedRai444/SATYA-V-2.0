-- ============================================================
-- SatyaAI Schema v2.0 (Supabase PostgreSQL)
-- Author: Rishabh Kapoor (Founder)
-- ============================================================

-- Extensions (uuid + crypto)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- ENUM TYPES
-- ============================================================

-- User role type
DO $$ BEGIN
  CREATE TYPE public.user_role AS ENUM ('user','admin','moderator');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Analysis job status type
DO $$ BEGIN
  CREATE TYPE public.analysis_status AS ENUM ('pending','queued','processing','completed','failed');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Media type type
DO $$ BEGIN
  CREATE TYPE public.media_type AS ENUM ('image','video','audio','multimodal','webcam');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- Notification type
DO $$ BEGIN
  CREATE TYPE public.notification_type AS ENUM ('info','success','warning','error','scan_complete');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- CORE TABLES
-- ============================================================

-- User table (extends auth.users)
CREATE TABLE IF NOT EXISTS public.users (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  username TEXT UNIQUE NOT NULL,
  email TEXT UNIQUE,
  full_name TEXT,
  avatar_url TEXT,
  role public.user_role NOT NULL DEFAULT 'user',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  last_login TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT username_format CHECK (username ~ '^[a-zA-Z0-9_]{3,30}$')
);

-- User preferences
CREATE TABLE IF NOT EXISTS public.user_preferences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
  theme TEXT DEFAULT 'dark',
  language TEXT DEFAULT 'english',
  confidence_threshold INT DEFAULT 75,
  enable_notifications BOOLEAN DEFAULT TRUE,
  auto_analyze BOOLEAN DEFAULT TRUE,
  sensitivity_level TEXT DEFAULT 'medium',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT theme_check CHECK (theme IN ('light','dark','auto')),
  CONSTRAINT confidence_check CHECK (confidence_threshold BETWEEN 0 AND 100),
  CONSTRAINT sensitivity_check CHECK (sensitivity_level IN ('low','medium','high'))
);

-- Analysis jobs (queue/tasks)
CREATE TABLE IF NOT EXISTS public.analysis_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  status public.analysis_status NOT NULL DEFAULT 'pending',
  media_type public.media_type NOT NULL,
  file_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT,
  file_hash TEXT,
  progress INT NOT NULL DEFAULT 0,
  metadata JSONB,
  error_message TEXT,
  priority INT DEFAULT 5,
  retry_count INT DEFAULT 0,
  report_code TEXT UNIQUE,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT progress_check CHECK (progress BETWEEN 0 AND 100),
  CONSTRAINT priority_check CHECK (priority BETWEEN 1 AND 10)
);

-- Analysis results
CREATE TABLE IF NOT EXISTS public.analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES public.analysis_jobs(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL DEFAULT 'SatyaAI',
  confidence FLOAT,
  is_deepfake BOOLEAN,
  analysis_data JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT confidence_float_check CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
);

-- Notifications
CREATE TABLE IF NOT EXISTS public.notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  type public.notification_type NOT NULL DEFAULT 'info',
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  is_read BOOLEAN NOT NULL DEFAULT FALSE,
  action_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  read_at TIMESTAMPTZ
);

-- API keys (optional external API access)
CREATE TABLE IF NOT EXISTS public.api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  permissions JSONB NOT NULL DEFAULT '{"read": true, "write": false, "admin": false}'::jsonb,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  last_used_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at TIMESTAMPTZ
);

-- ============================================================
-- INDEXES
-- ============================================================

-- Users
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON public.users(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_created_at ON public.users(created_at DESC);

-- Analysis jobs
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON public.analysis_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON public.analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_at ON public.analysis_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_media_type ON public.analysis_jobs(media_type);

-- Analysis results
CREATE INDEX IF NOT EXISTS idx_analysis_results_job_id ON public.analysis_results(job_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON public.analysis_results(created_at);

-- Notifications
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON public.notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON public.notifications(created_at);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON public.notifications(is_read) WHERE NOT is_read;
CREATE INDEX IF NOT EXISTS idx_notifications_type ON public.notifications(type);

-- API keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON public.api_keys(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON public.api_keys(is_active) WHERE is_active = TRUE;

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================

-- Enable RLS on all tables
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;

-- Users policies
CREATE POLICY "Users can view their own profile" 
  ON public.users FOR SELECT 
  USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON public.users FOR UPDATE
  USING (auth.uid() = id);

-- User preferences policies
CREATE POLICY "Users can manage their preferences"
  ON public.user_preferences
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Analysis jobs policies
CREATE POLICY "Users can manage their analysis jobs"
  ON public.analysis_jobs
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Analysis results policies
CREATE POLICY "Users can view their analysis results"
  ON public.analysis_results
  USING (EXISTS (
    SELECT 1 FROM public.analysis_jobs 
    WHERE public.analysis_jobs.id = public.analysis_results.job_id 
    AND public.analysis_jobs.user_id = auth.uid()
  ));

-- Notifications policies
CREATE POLICY "Users can manage their notifications"
  ON public.notifications
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- API keys policies
CREATE POLICY "Users can manage their API keys"
  ON public.api_keys
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add updated_at triggers
CREATE TRIGGER set_updated_at
BEFORE UPDATE ON public.users
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER set_updated_at
BEFORE UPDATE ON public.user_preferences
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER set_updated_at
BEFORE UPDATE ON public.analysis_jobs
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- ============================================================
-- COMPLETED SCHEMA
-- ============================================================

COMMENT ON SCHEMA public IS 'SatyaAI v2.0 Database Schema - Deepfake Detection Platform';

-- ============================================================
-- SCHEMA VERSION
-- ============================================================

-- Track schema version for future migrations
CREATE TABLE IF NOT EXISTS public.schema_migrations (
  id SERIAL PRIMARY KEY,
  version VARCHAR(50) NOT NULL UNIQUE,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  description TEXT
);

-- Insert initial schema version if not exists
INSERT INTO public.schema_migrations (version, description)
VALUES ('1.0.0', 'Initial database schema')
ON CONFLICT (version) DO NOTHING;

-- ============================================================
-- DATABASE FUNCTIONS
-- ============================================================

-- Function to generate report codes (e.g., SATYA-IMG-20230101-0001)
CREATE OR REPLACE FUNCTION public.generate_report_code(media_type TEXT)
RETURNS TEXT AS $$
DECLARE
  prefix TEXT;
  date_str TEXT := to_char(CURRENT_DATE, 'YYYYMMDD');
  seq_num INT;
BEGIN
  -- Determine prefix based on media type
  CASE media_type
    WHEN 'image' THEN prefix := 'IMG';
    WHEN 'video' THEN prefix := 'VID';
    WHEN 'audio' THEN prefix := 'AUD';
    WHEN 'webcam' THEN prefix := 'WEB';
    ELSE prefix := 'GEN';
  END CASE;
  
  -- Get next sequence number for this date and type
  SELECT COALESCE(MAX(SUBSTRING(report_code, -4)::INT), 0) + 1 INTO seq_num
  FROM public.analysis_jobs
  WHERE report_code LIKE 'SATYA-' || prefix || '-' || date_str || '-%';
  
  -- Format: SATYA-{TYPE}-{DATE}-{SEQ}
  RETURN 'SATYA-' || prefix || '-' || date_str || '-' || LPAD(seq_num::TEXT, 4, '0');
END;
$$ LANGUAGE plpgsql;

-- Function to update job progress
CREATE OR REPLACE FUNCTION public.update_job_progress(
  job_id UUID,
  new_progress INT,
  status_text TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
  UPDATE public.analysis_jobs
  SET 
    progress = LEAST(GREATEST(new_progress, 0), 100),
    status = COALESCE(status_text::public.analysis_status, 
                     CASE 
                       WHEN new_progress >= 100 THEN 'completed'::public.analysis_status
                       WHEN new_progress > 0 THEN 'processing'::public.analysis_status
                       ELSE status
                     END),
    updated_at = NOW(),
    completed_at = CASE WHEN new_progress >= 100 THEN NOW() ELSE completed_at END
  WHERE id = job_id;
END;
$$ LANGUAGE plpgsql;
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
