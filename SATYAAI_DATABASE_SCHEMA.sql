-- ============================================================
-- SatyaAI Schema v2.1 (Complete Production Schema)
-- For: Supabase PostgreSQL
-- Author: Rishabh Kapoor (Founder)
-- Project: SatyaAI
-- 
-- CHANGES v2.1:
-- - Added chat system tables (Satya Sentinel persistence)
-- - Added file upload tracking
-- - Added batch processing tables
-- - Enhanced user preferences with chat settings
-- - Added notification_type enum with chat support
-- - Complete RLS policies for all tables
-- ============================================================

-- ============================================================
-- EXTENSIONS (uuid + crypto)
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- UUID helpers
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid(), crypto tools

-- ============================================================
-- ENUM TYPES
-- ============================================================

DO $$ BEGIN
  CREATE TYPE public.user_role AS ENUM ('user','admin','moderator');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE public.analysis_status AS ENUM ('pending','queued','processing','completed','failed','cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE public.media_type AS ENUM ('image','video','audio','multimodal','webcam','batch');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE public.notification_type AS ENUM ('info','success','warning','error','scan_complete','chat');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- CORE TABLES (Complete Supabase Auth Integration)
-- ============================================================

-- Users (extends auth.users) - Updated for Supabase Auth
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

-- User preferences (enhanced with chat settings)
CREATE TABLE IF NOT EXISTS public.user_preferences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
  theme TEXT NOT NULL DEFAULT 'dark',
  language TEXT NOT NULL DEFAULT 'english',
  confidence_threshold INT NOT NULL DEFAULT 75,
  enable_notifications BOOLEAN NOT NULL DEFAULT TRUE,
  auto_analyze BOOLEAN NOT NULL DEFAULT TRUE,
  sensitivity_level TEXT NOT NULL DEFAULT 'medium',
  chat_model TEXT DEFAULT 'gpt-4',
  chat_enabled BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT theme_check CHECK (theme IN ('light','dark','auto')),
  CONSTRAINT confidence_check CHECK (confidence_threshold BETWEEN 0 AND 100),
  CONSTRAINT sensitivity_check CHECK (sensitivity_level IN ('low','medium','high'))
);

-- Analysis jobs queue (updated for UUID user_id)
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
  priority INT NOT NULL DEFAULT 5,
  retry_count INT NOT NULL DEFAULT 0,
  report_code TEXT UNIQUE,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT progress_check CHECK (progress BETWEEN 0 AND 100),
  CONSTRAINT priority_check CHECK (priority BETWEEN 1 AND 10)
);

-- Analysis results (enhanced with model versioning)
CREATE TABLE IF NOT EXISTS public.analysis_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES public.analysis_jobs(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL DEFAULT 'SatyaAI',
  model_version TEXT,
  confidence FLOAT,
  is_deepfake BOOLEAN,
  analysis_data JSONB NOT NULL,
  proof_json JSONB,
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

-- API keys (store only hashed key)
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

-- Audit logs (security + activity tracking)
CREATE TABLE IF NOT EXISTS public.audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  action TEXT NOT NULL,
  table_name TEXT NOT NULL,
  row_id UUID,
  changes JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================

-- Users table indexes
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON public.users(role);

-- Analysis jobs indexes
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON public.analysis_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON public.analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_at ON public.analysis_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_report_code ON public.analysis_jobs(report_code);

-- Analysis results indexes
CREATE INDEX IF NOT EXISTS idx_analysis_results_job_id ON public.analysis_results(job_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_model_name ON public.analysis_results(model_name);

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
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.file_uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.batch_jobs ENABLE ROW LEVEL SECURITY;

-- Users RLS policies
DROP POLICY IF EXISTS "Users can view their own profile" ON public.users;
CREATE POLICY "Users can view their own profile"
ON public.users FOR SELECT
USING (id = auth.uid());

DROP POLICY IF EXISTS "Users can update their own profile" ON public.users;
CREATE POLICY "Users can update their own profile"
ON public.users FOR UPDATE
USING (id = auth.uid())
WITH CHECK (id = auth.uid());

-- Analysis jobs RLS policies
DROP POLICY IF EXISTS "Users can manage their analysis jobs" ON public.analysis_jobs;
CREATE POLICY "Users can manage their analysis jobs"
ON public.analysis_jobs FOR ALL
USING (user_id = auth.uid())
WITH CHECK (user_id = auth.uid());

-- Analysis results RLS policies
DROP POLICY IF EXISTS "Users can view their analysis results" ON public.analysis_results;
CREATE POLICY "Users can view their analysis results"
ON public.analysis_results FOR SELECT
USING (
  EXISTS (
    SELECT 1
    FROM public.analysis_jobs j
    WHERE j.id = analysis_results.job_id
      AND j.user_id = auth.uid()
  )
);

-- ============================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- ============================================================

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER handle_users_updated_at
BEFORE UPDATE ON public.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_analysis_jobs_updated_at
BEFORE UPDATE ON public.analysis_jobs
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_analysis_results_updated_at
BEFORE UPDATE ON public.analysis_results
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

-- Triggers for new tables
CREATE TRIGGER handle_chat_conversations_updated_at
BEFORE UPDATE ON public.chat_conversations
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_batch_jobs_updated_at
BEFORE UPDATE ON public.batch_jobs
FOR EACH ROW
EXECUTE FUNCTION public.handle_updated_at();

-- ============================================================
-- MIGRATION NOTES
-- ============================================================

/*
SatyaAI Schema v2.1 - Complete Production Schema

CHANGES v2.1:
- Added chat system tables (Satya Sentinel persistence)
- Added file upload tracking with metadata
- Added batch processing tables
- Enhanced user preferences with chat settings
- Added notification_type enum with chat support
- Complete RLS policies for all tables
- Updated to pure Supabase Auth (UUID-based)

NEW TABLES:
- chat_conversations: Chat session management
- chat_messages: Message persistence
- file_uploads: File metadata tracking
- batch_jobs: Batch processing operations

To apply this schema:
1. Go to your Supabase project
2. Open SQL Editor
3. Copy-paste this entire file
4. Click "Run"

After running this schema:
- Complete chat system persistence
- File upload tracking and cleanup
- Batch processing capabilities
- Full localStorage migration support
- Production-ready security with RLS
- Each user sees only their own data

This schema supports 100% of your SatyaAI features including:
✅ Complete authentication system
✅ Analysis pipeline with history
✅ Chat system (Satya Sentinel)
✅ File management
✅ Batch processing
✅ User preferences
✅ Notifications and audit logs
*/
  last_login TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT username_format CHECK (username ~ '^[a-zA-Z0-9_]{3,30}$')
);

-- User preferences (1 row per user)
CREATE TABLE IF NOT EXISTS public.user_preferences (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
  theme TEXT NOT NULL DEFAULT 'dark',
  language TEXT NOT NULL DEFAULT 'english',
  confidence_threshold INT NOT NULL DEFAULT 75,
  enable_notifications BOOLEAN NOT NULL DEFAULT TRUE,
  auto_analyze BOOLEAN NOT NULL DEFAULT TRUE,
  sensitivity_level TEXT NOT NULL DEFAULT 'medium',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT theme_check CHECK (theme IN ('light','dark','auto')),
  CONSTRAINT confidence_check CHECK (confidence_threshold BETWEEN 0 AND 100),
  CONSTRAINT sensitivity_check CHECK (sensitivity_level IN ('low','medium','high'))
);

-- Analysis jobs queue (this is your "tasks")
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
  priority INT NOT NULL DEFAULT 5,
  retry_count INT NOT NULL DEFAULT 0,
  report_code TEXT UNIQUE,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT progress_check CHECK (progress BETWEEN 0 AND 100),
  CONSTRAINT priority_check CHECK (priority BETWEEN 1 AND 10)
);

-- Analysis results (1 job can have multiple model outputs)
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

-- API keys (store only hashed key)
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

-- Audit logs (security + activity tracking)
CREATE TABLE IF NOT EXISTS public.audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
  action TEXT NOT NULL,
  resource_type TEXT,
  resource_id UUID,
  metadata JSONB,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- CHAT SYSTEM TABLES (NEW v2.1)
-- ============================================================

-- Chat conversations (Satya Sentinel persistence)
CREATE TABLE IF NOT EXISTS public.chat_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT,
  model TEXT NOT NULL DEFAULT 'gpt-4',
  context JSONB DEFAULT '{}',
  is_active BOOLEAN DEFAULT TRUE,
  message_count INT DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS public.chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES public.chat_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  token_count INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- FILE MANAGEMENT TABLES (NEW v2.1)
-- ============================================================

-- File uploads tracking
CREATE TABLE IF NOT EXISTS public.file_uploads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  original_name TEXT NOT NULL,
  stored_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  storage_bucket TEXT DEFAULT 'uploads',
  mime_type TEXT NOT NULL,
  file_size BIGINT NOT NULL,
  file_hash TEXT,
  is_temporary BOOLEAN DEFAULT TRUE,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- BATCH PROCESSING TABLES (NEW v2.1)
-- ============================================================

-- Batch jobs
CREATE TABLE IF NOT EXISTS public.batch_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  total_files INT NOT NULL DEFAULT 0,
  processed_files INT NOT NULL DEFAULT 0,
  failed_files INT NOT NULL DEFAULT 0,
  settings JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

-- ============================================================
-- SCHEMA MIGRATIONS
-- ============================================================

-- Schema migrations tracker
CREATE TABLE IF NOT EXISTS public.schema_migrations (
  id SERIAL PRIMARY KEY,
  version VARCHAR(50) NOT NULL UNIQUE,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  description TEXT
);

INSERT INTO public.schema_migrations (version, description)
VALUES ('2.0.0', 'SatyaAI final all-in-one schema (Supabase Auth + Jobs + Results)')
ON CONFLICT (version) DO NOTHING;

INSERT INTO public.schema_migrations (version, description)
VALUES ('2.1.0', 'Added chat system, file tracking, and batch processing tables')
ON CONFLICT (version) DO NOTHING;

-- ============================================================
-- INDEXES (FAST QUERIES)
-- ============================================================

-- Users
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON public.users(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_users_created_at ON public.users(created_at DESC);

-- Preferences
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON public.user_preferences(user_id);

-- Jobs
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON public.analysis_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_status ON public.analysis_jobs(status);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_created_at ON public.analysis_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_media_type ON public.analysis_jobs(media_type);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_priority ON public.analysis_jobs(priority DESC, status, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_file_hash ON public.analysis_jobs(file_hash);

-- Results
CREATE INDEX IF NOT EXISTS idx_analysis_results_job_id ON public.analysis_results(job_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_created_at ON public.analysis_results(created_at DESC);

-- Notifications
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON public.notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON public.notifications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON public.notifications(user_id, created_at DESC)
  WHERE is_read = FALSE;

-- Chat indexes
CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_id ON public.chat_conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_conversations_active ON public.chat_conversations(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON public.chat_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON public.chat_messages(created_at DESC);

-- File uploads indexes
CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id ON public.file_uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_temporary ON public.file_uploads(is_temporary, expires_at) WHERE is_temporary = TRUE;
CREATE INDEX IF NOT EXISTS idx_file_uploads_hash ON public.file_uploads(file_hash);

-- Batch jobs indexes
CREATE INDEX IF NOT EXISTS idx_batch_jobs_user_id ON public.batch_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON public.batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON public.batch_jobs(created_at DESC);

-- API Keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON public.api_keys(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON public.api_keys(expires_at) WHERE expires_at IS NOT NULL;

-- Audit Logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON public.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON public.audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON public.audit_logs(created_at DESC);

-- ============================================================
-- TRIGGERS + FUNCTIONS
-- ============================================================

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: users.updated_at
DROP TRIGGER IF EXISTS trg_users_updated_at ON public.users;
CREATE TRIGGER trg_users_updated_at
BEFORE UPDATE ON public.users
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Trigger: preferences.updated_at
DROP TRIGGER IF EXISTS trg_preferences_updated_at ON public.user_preferences;
CREATE TRIGGER trg_preferences_updated_at
BEFORE UPDATE ON public.user_preferences
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Trigger: jobs.updated_at
DROP TRIGGER IF EXISTS trg_jobs_updated_at ON public.analysis_jobs;
CREATE TRIGGER trg_jobs_updated_at
BEFORE UPDATE ON public.analysis_jobs
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- ============================================================
-- AUTH SYNC (MOST IMPORTANT PART)
-- When user registers in Supabase Auth -> create public.users row
-- ============================================================

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  base_username TEXT;
  unique_username TEXT;
  suffix INT := 1;
  max_attempts INT := 5;
  attempts INT := 0;
  username_exists BOOLEAN;
BEGIN
  -- Generate base username from email
  base_username := LOWER(COALESCE(
    NULLIF(REGEXP_REPLACE(NEW.raw_user_meta_data->>'username', '[^a-zA-Z0-9_]', '', 'g'), ''),
    REGEXP_REPLACE(SPLIT_PART(NEW.email, '@', 1), '[^a-zA-Z0-9]', '', 'g')
  ));
  
  -- Ensure username is not empty
  IF base_username = '' OR base_username IS NULL THEN
    base_username := 'user';
  END IF;
  
  -- Ensure username meets length requirements
  base_username := SUBSTRING(base_username FROM 1 FOR 25);
  
  -- Check if username exists and append random suffix if needed
  unique_username := base_username;
  LOOP
    SELECT EXISTS (SELECT 1 FROM public.users WHERE username = unique_username) INTO username_exists;
    EXIT WHEN NOT username_exists OR attempts >= max_attempts;
    
    -- Try with a random suffix
    unique_username := base_username || '_' || FLOOR(random() * 1000)::TEXT;
    attempts := attempts + 1;
  END LOOP;
  
  -- If all attempts failed, use a UUID-based username
  IF attempts >= max_attempts AND username_exists THEN
    unique_username := 'user_' || REPLACE(gen_random_uuid()::TEXT, '-', '_');
  END IF;

  -- Insert the new user with the unique username
  INSERT INTO public.users (id, email, username, full_name, avatar_url)
  VALUES (
    NEW.id,
    NEW.email,
    unique_username,
    COALESCE(NEW.raw_user_meta_data->>'full_name', ''),
    COALESCE(NEW.raw_user_meta_data->>'avatar_url', '')
  )
  ON CONFLICT (id) DO NOTHING;

  -- Insert default preferences
  INSERT INTO public.user_preferences (user_id, chat_model, chat_enabled)
  VALUES (NEW.id, 'gpt-4', TRUE)
  ON CONFLICT (user_id) DO NOTHING;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Recreate trigger safely
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- REPORT CODE GENERATOR
-- ============================================================

CREATE OR REPLACE FUNCTION public.generate_report_code(p_media_type TEXT)
RETURNS TEXT AS $$
DECLARE
  prefix TEXT;
  date_str TEXT := to_char(CURRENT_DATE, 'YYYYMMDD');
  seq_num INT;
BEGIN
  CASE p_media_type
    WHEN 'image' THEN prefix := 'IMG';
    WHEN 'video' THEN prefix := 'VID';
    WHEN 'audio' THEN prefix := 'AUD';
    WHEN 'webcam' THEN prefix := 'WEB';
    WHEN 'multimodal' THEN prefix := 'MLT';
    WHEN 'batch' THEN prefix := 'BCH';
    ELSE prefix := 'GEN';
  END CASE;

  SELECT COALESCE(MAX(SUBSTRING(report_code, -4)::INT), 0) + 1 INTO seq_num
  FROM public.analysis_jobs
  WHERE report_code LIKE 'SATYA-' || prefix || '-' || date_str || '-%';

  RETURN 'SATYA-' || prefix || '-' || date_str || '-' || LPAD(seq_num::TEXT, 4, '0');
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- JOB UPDATE HELPER (backend can call this)
-- ============================================================

CREATE OR REPLACE FUNCTION public.update_job_progress(
  p_job_id UUID,
  p_new_progress INT,
  p_status TEXT DEFAULT NULL
) RETURNS VOID AS $$
BEGIN
  UPDATE public.analysis_jobs
  SET
    progress = LEAST(GREATEST(p_new_progress, 0), 100),
    status = COALESCE(
      p_status::public.analysis_status,
      CASE
        WHEN p_new_progress >= 100 THEN 'completed'::public.analysis_status
        WHEN p_new_progress > 0 THEN 'processing'::public.analysis_status
        ELSE status
      END
    ),
    updated_at = NOW(),
    completed_at = CASE WHEN p_new_progress >= 100 THEN NOW() ELSE completed_at END
  WHERE id = p_job_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================

ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- USERS
DROP POLICY IF EXISTS "Users can view their own profile" ON public.users;
CREATE POLICY "Users can view their own profile"
  ON public.users FOR SELECT
  USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update their own profile" ON public.users;
CREATE POLICY "Users can update their own profile"
  ON public.users FOR UPDATE
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

-- PREFERENCES
DROP POLICY IF EXISTS "Users can manage their preferences" ON public.user_preferences;
CREATE POLICY "Users can manage their preferences"
  ON public.user_preferences FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- JOBS
DROP POLICY IF EXISTS "Users can manage their analysis jobs" ON public.analysis_jobs;
CREATE POLICY "Users can manage their analysis jobs"
  ON public.analysis_jobs FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- RESULTS
DROP POLICY IF EXISTS "Users can view their analysis results" ON public.analysis_results;
CREATE POLICY "Users can view their analysis results"
  ON public.analysis_results FOR SELECT
  USING (
    EXISTS (
      SELECT 1
      FROM public.analysis_jobs j
      WHERE j.id = analysis_results.job_id
        AND j.user_id = auth.uid()
    )
  );

-- Allow inserting results only through service/backend (optional but recommended)
DROP POLICY IF EXISTS "Service role can insert results" ON public.analysis_results;
CREATE POLICY "Service role can insert results"
  ON public.analysis_results FOR INSERT
  TO service_role
  WITH CHECK (true);

-- Chat conversations policies
DROP POLICY IF EXISTS "Users can manage their conversations" ON public.chat_conversations;
CREATE POLICY "Users can manage their conversations"
  ON public.chat_conversations FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Chat messages policies
DROP POLICY IF EXISTS "Users can view their conversation messages" ON public.chat_messages;
CREATE POLICY "Users can view their conversation messages"
  ON public.chat_messages FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.chat_conversations c
      WHERE c.id = chat_messages.conversation_id AND c.user_id = auth.uid()
    )
  );

-- File uploads policies
DROP POLICY IF EXISTS "Users can manage their files" ON public.file_uploads;
CREATE POLICY "Users can manage their files"
  ON public.file_uploads FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Batch jobs policies
DROP POLICY IF EXISTS "Users can manage their batch jobs" ON public.batch_jobs;
CREATE POLICY "Users can manage their batch jobs"
  ON public.batch_jobs FOR ALL
  USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- NOTIFICATIONS
DROP POLICY IF EXISTS "Users can manage their notifications" ON public.notifications;
CREATE POLICY "Users can manage their notifications"
  ON public.notifications FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- API KEYS
DROP POLICY IF EXISTS "Users can manage their API keys" ON public.api_keys;
CREATE POLICY "Users can manage their API keys"
  ON public.api_keys FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- AUDIT LOGS (read only for owner; insert by service role)
DROP POLICY IF EXISTS "Users can view their audit logs" ON public.audit_logs;
CREATE POLICY "Users can view their audit logs"
  ON public.audit_logs FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Service role can insert audit logs" ON public.audit_logs;
CREATE POLICY "Service role can insert audit logs"
  ON public.audit_logs FOR INSERT
  TO service_role
  WITH CHECK (true);

-- ============================================================
-- OPTIONAL VIEWS (helpful)
-- ============================================================

CREATE OR REPLACE VIEW public.user_statistics AS
SELECT
  u.id,
  u.username,
  u.email,
  u.role,
  u.is_active,
  u.created_at,
  u.last_login,
  COUNT(DISTINCT j.id) AS total_jobs,
  COUNT(DISTINCT CASE WHEN j.status = 'completed' THEN j.id END) AS completed_jobs,
  COUNT(DISTINCT CASE WHEN j.status = 'failed' THEN j.id END) AS failed_jobs,
  MAX(j.created_at) AS last_job_date,
  COUNT(DISTINCT CASE WHEN n.is_read = FALSE THEN n.id END) AS unread_notifications
FROM public.users u
LEFT JOIN public.analysis_jobs j ON u.id = j.user_id
LEFT JOIN public.notifications n ON u.id = n.user_id
GROUP BY u.id, u.username, u.email, u.role, u.is_active, u.created_at, u.last_login;

-- ============================================================
-- FINAL COMMENT
-- ============================================================
COMMENT ON SCHEMA public IS 'SatyaAI Schema v2.0 - final all-in-one schema';
