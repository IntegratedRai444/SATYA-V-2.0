-- ============================================================
-- SatyaAI Schema v2.6 (Production-Ready | Tasks Single Source)
-- For: Supabase PostgreSQL
-- Author: Rishabh Kapoor (Founder)
-- Project: SatyaAI / HyperSatya X
--
-- KEY PRINCIPLE:
-- ✅ tasks is the ONLY source of truth for analysis jobs + results
-- ❌ analysis_history removed to prevent duplication
-- ============================================================

-- ============================================================
-- FULL RESET / DROP EXISTING OBJECTS (SAFE ORDER)
-- ============================================================

-- Drop views first
DROP VIEW IF EXISTS public.analysis_jobs_view CASCADE;
DROP VIEW IF EXISTS public.scans_view CASCADE;

-- Drop triggers
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP TRIGGER IF EXISTS set_default_task_report_code_trigger ON public.tasks;

-- Drop functions
DROP FUNCTION IF EXISTS public.cleanup_old_notifications(integer) CASCADE;
DROP FUNCTION IF EXISTS public.cleanup_expired_files() CASCADE;
DROP FUNCTION IF EXISTS public.get_user_stats(uuid) CASCADE;
DROP FUNCTION IF EXISTS public.handle_new_user() CASCADE;
DROP FUNCTION IF EXISTS public.handle_updated_at() CASCADE;
DROP FUNCTION IF EXISTS public.set_default_task_report_code() CASCADE;
DROP FUNCTION IF EXISTS public.generate_report_code(text) CASCADE;
DROP FUNCTION IF EXISTS public.hash_api_key(text) CASCADE;
DROP FUNCTION IF EXISTS public.verify_api_key(text, text) CASCADE;

-- Drop legacy / duplicates if exist
DROP TABLE IF EXISTS public.analysis_results CASCADE;
DROP TABLE IF EXISTS public.analysis_jobs CASCADE;
DROP TABLE IF EXISTS public.analysis_history CASCADE;
DROP TABLE IF EXISTS public.analysis_history_legacy CASCADE;
DROP TABLE IF EXISTS public.scans CASCADE;
DROP TABLE IF EXISTS public.schema_migrations CASCADE;

-- Optional legacy sequences
DROP SEQUENCE IF EXISTS public.scans_id_seq CASCADE;
DROP SEQUENCE IF EXISTS public.schema_migrations_id_seq CASCADE;

-- Drop core tables (children first)
DROP TABLE IF EXISTS public.audit_logs CASCADE;
DROP TABLE IF EXISTS public.api_keys CASCADE;
DROP TABLE IF EXISTS public.notifications CASCADE;
DROP TABLE IF EXISTS public.batch_jobs CASCADE;
DROP TABLE IF EXISTS public.file_uploads CASCADE;
DROP TABLE IF EXISTS public.chat_messages CASCADE;
DROP TABLE IF EXISTS public.chat_conversations CASCADE;
DROP TABLE IF EXISTS public.tasks CASCADE;
DROP TABLE IF EXISTS public.user_preferences CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;

-- Drop enum types
DO $$ BEGIN
  DROP TYPE IF EXISTS public.notification_type CASCADE;
  DROP TYPE IF EXISTS public.media_type CASCADE;
  DROP TYPE IF EXISTS public.analysis_status CASCADE;
  DROP TYPE IF EXISTS public.user_role CASCADE;
EXCEPTION WHEN undefined_object THEN NULL;
END $$;

-- ============================================================
-- EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA public;

-- ============================================================
-- ENUM TYPES
-- ============================================================

DO $$ BEGIN
  CREATE TYPE public.user_role AS ENUM ('user', 'admin', 'moderator');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE public.analysis_status AS ENUM ('pending', 'queued', 'processing', 'completed', 'failed', 'cancelled');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE public.media_type AS ENUM ('image', 'video', 'audio', 'multimodal', 'webcam', 'batch');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE TYPE public.notification_type AS ENUM ('info', 'success', 'warning', 'error', 'scan_complete', 'chat');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ============================================================
-- CORE TABLES
-- ============================================================

-- ============================================================
-- USERS TABLE (extends auth.users)
-- ============================================================
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
  deleted_at TIMESTAMPTZ,
  CONSTRAINT username_format CHECK (username ~ '^[a-zA-Z0-9_]{3,30}$'),
  CONSTRAINT email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- ============================================================
-- USER PREFERENCES
-- ============================================================
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
  deleted_at TIMESTAMPTZ,
  CONSTRAINT theme_check CHECK (theme IN ('light', 'dark', 'auto')),
  CONSTRAINT confidence_check CHECK (confidence_threshold BETWEEN 0 AND 100),
  CONSTRAINT sensitivity_check CHECK (sensitivity_level IN ('low', 'medium', 'high'))
);

-- ============================================================
-- TASKS TABLE (Single Source of Truth)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  type VARCHAR(50) NOT NULL CHECK (type IN ('analysis', 'batch', 'cleanup', 'export')),
  status public.analysis_status NOT NULL DEFAULT 'pending',
  progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
  file_name TEXT NOT NULL,
  file_size BIGINT NOT NULL CHECK (file_size > 0 AND file_size < 10737418240),
  file_type VARCHAR(100) NOT NULL,
  file_path TEXT NOT NULL,
  report_code TEXT NOT NULL UNIQUE,
  result JSONB,
  error JSONB,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at TIMESTAMPTZ,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ============================================================
-- CHAT CONVERSATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.chat_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  is_archived BOOLEAN NOT NULL DEFAULT FALSE,
  deleted_at TIMESTAMPTZ
);

-- ============================================================
-- CHAT MESSAGES
-- ============================================================
CREATE TABLE IF NOT EXISTS public.chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES public.chat_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  deleted_at TIMESTAMPTZ
);

-- ============================================================
-- FILE UPLOADS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.file_uploads (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  file_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_size BIGINT NOT NULL CHECK (file_size > 0),
  mime_type TEXT,
  is_processed BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ
);

-- ============================================================
-- BATCH JOBS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.batch_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  status public.analysis_status NOT NULL DEFAULT 'pending',
  total_items INT NOT NULL CHECK (total_items > 0),
  processed_items INT NOT NULL DEFAULT 0 CHECK (processed_items >= 0),
  failed_items INT NOT NULL DEFAULT 0 CHECK (failed_items >= 0),
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ
);

-- ============================================================
-- NOTIFICATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.notifications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  type public.notification_type NOT NULL DEFAULT 'info',
  title TEXT NOT NULL,
  message TEXT NOT NULL,
  is_read BOOLEAN NOT NULL DEFAULT FALSE,
  action_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  read_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ
);

-- ============================================================
-- API KEYS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  key_hash TEXT NOT NULL UNIQUE,
  key_algorithm TEXT NOT NULL DEFAULT 'bcrypt',
  permissions JSONB NOT NULL DEFAULT '{"read": true, "write": false, "admin": false}'::jsonb,
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  last_used_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  revoked_at TIMESTAMPTZ,
  deleted_at TIMESTAMPTZ,
  CONSTRAINT name_length_check CHECK (length(name) >= 3 AND length(name) <= 100)
);

-- ============================================================
-- AUDIT LOGS
-- ============================================================
CREATE TABLE IF NOT EXISTS public.audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE SET NULL,
  action TEXT NOT NULL,
  table_name TEXT NOT NULL,
  row_id UUID,
  changes JSONB,
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES FOR PERFORMANCE (soft delete aware)
-- ============================================================

-- Users
CREATE INDEX IF NOT EXISTS idx_users_email ON public.users(email) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_users_username ON public.users(username) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_users_role ON public.users(role) WHERE is_active = TRUE AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_users_created_at ON public.users(created_at DESC);

-- Preferences
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON public.user_preferences(user_id) WHERE deleted_at IS NULL;

-- Tasks
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON public.tasks(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_status ON public.tasks(status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_report_code ON public.tasks(report_code);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON public.tasks(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_user_status_created ON public.tasks(user_id, status, created_at DESC) WHERE deleted_at IS NULL;

-- Chat conversations/messages
CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_id ON public.chat_conversations(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated_at ON public.chat_conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON public.chat_messages(conversation_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON public.chat_messages(created_at DESC);

-- File uploads
CREATE INDEX IF NOT EXISTS idx_file_uploads_user_id ON public.file_uploads(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_file_uploads_expires_at ON public.file_uploads(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_file_uploads_processed ON public.file_uploads(is_processed) WHERE NOT is_processed AND deleted_at IS NULL;

-- Batch jobs
CREATE INDEX IF NOT EXISTS idx_batch_jobs_user_id ON public.batch_jobs(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON public.batch_jobs(status) WHERE deleted_at IS NULL;

-- Notifications
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON public.notifications(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON public.notifications(user_id, is_read) WHERE NOT is_read AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON public.notifications(created_at DESC);

-- API keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON public.api_keys(is_active) WHERE is_active = TRUE AND deleted_at IS NULL;

-- Audit logs
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON public.audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON public.audit_logs(created_at DESC);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Generate report codes
CREATE OR REPLACE FUNCTION public.generate_report_code(p_media_type TEXT DEFAULT 'multimodal')
RETURNS TEXT AS $$
DECLARE
  chars TEXT := 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  result TEXT := '';
  prefix TEXT := '';
  rand_int INT;
  max_attempts INT := 10;
  attempt INT := 0;
BEGIN
  CASE p_media_type
    WHEN 'image' THEN prefix := 'IMG';
    WHEN 'video' THEN prefix := 'VID';
    WHEN 'audio' THEN prefix := 'AUD';
    WHEN 'webcam' THEN prefix := 'CAM';
    WHEN 'batch' THEN prefix := 'BAT';
    ELSE prefix := 'MLT';
  END CASE;

  LOOP
    result := prefix || '-';
    FOR i IN 1..8 LOOP
      rand_int := floor(random() * length(chars) + 1)::int;
      result := result || substr(chars, rand_int, 1);
    END LOOP;

    IF NOT EXISTS (SELECT 1 FROM public.tasks WHERE report_code = result) THEN
      RETURN result;
    END IF;

    attempt := attempt + 1;
    IF attempt >= max_attempts THEN
      RAISE EXCEPTION 'Could not generate unique report code after % attempts', max_attempts;
    END IF;
  END LOOP;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger: auto set report_code for tasks
CREATE OR REPLACE FUNCTION public.set_default_task_report_code()
RETURNS TRIGGER AS $$
DECLARE
  media_type TEXT;
BEGIN
  IF NEW.report_code IS NULL OR NEW.report_code = '' THEN
    CASE
      WHEN NEW.file_type LIKE 'image%' THEN media_type := 'image';
      WHEN NEW.file_type LIKE 'video%' THEN media_type := 'video';
      WHEN NEW.file_type LIKE 'audio%' THEN media_type := 'audio';
      WHEN NEW.file_type LIKE 'webcam%' THEN media_type := 'webcam';
      WHEN NEW.file_type LIKE 'batch%' THEN media_type := 'batch';
      ELSE media_type := 'multimodal';
    END CASE;
    NEW.report_code := public.generate_report_code(media_type);
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- AUTH TRIGGER: Create user profile + preferences
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
  base_username TEXT;
  new_username TEXT;
  suffix TEXT;
  counter INT := 1;
  user_email TEXT;
BEGIN
  user_email := COALESCE(NEW.email, 'user');

  base_username := LOWER(REGEXP_REPLACE(split_part(user_email, '@', 1), '[^a-z0-9_]', '', 'g'));

  IF length(base_username) < 3 THEN
    base_username := 'user' || substr(md5(random()::text), 1, 8);
  END IF;

  new_username := base_username;
  WHILE EXISTS (SELECT 1 FROM public.users WHERE username = new_username) AND counter < 100 LOOP
    suffix := substr(md5(random()::text), 1, 4);
    new_username := base_username || '_' || suffix;
    counter := counter + 1;
  END LOOP;

  INSERT INTO public.users (id, username, email, role)
  VALUES (NEW.id, new_username, NEW.email, 'user');

  INSERT INTO public.user_preferences (user_id)
  VALUES (NEW.id);

  RETURN NEW;
EXCEPTION WHEN OTHERS THEN
  RAISE WARNING 'Error creating user profile: %', SQLERRM;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- UPDATED_AT TRIGGERS
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
  t record;
BEGIN
  FOR t IN
    SELECT table_name
    FROM information_schema.columns
    WHERE column_name = 'updated_at'
      AND table_schema = 'public'
      AND table_name NOT IN ('audit_logs')
  LOOP
    EXECUTE format('DROP TRIGGER IF EXISTS handle_%s_updated_at ON public.%I', t.table_name, t.table_name);
    EXECUTE format('CREATE TRIGGER handle_%s_updated_at
                    BEFORE UPDATE ON public.%I
                    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at()', t.table_name, t.table_name);
  END LOOP;
END;
$$;

-- ============================================================
-- REPORT CODE TRIGGER ON TASKS
-- ============================================================
DROP TRIGGER IF EXISTS set_default_task_report_code_trigger ON public.tasks;
CREATE TRIGGER set_default_task_report_code_trigger
BEFORE INSERT ON public.tasks
FOR EACH ROW
EXECUTE FUNCTION public.set_default_task_report_code();

-- ============================================================
-- RLS ENABLE
-- ============================================================
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.file_uploads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.batch_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- RLS POLICIES
-- ============================================================

-- USERS
DROP POLICY IF EXISTS "Users can view their own profile" ON public.users;
CREATE POLICY "Users can view their own profile"
ON public.users FOR SELECT
USING (id = auth.uid());

DROP POLICY IF EXISTS "Users can update their own profile" ON public.users;
CREATE POLICY "Users can update their own profile"
ON public.users FOR UPDATE
USING (id = auth.uid())
WITH CHECK (id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all users" ON public.users;
CREATE POLICY "Service role can manage all users"
ON public.users FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- USER PREFERENCES
DROP POLICY IF EXISTS "Users can manage their own preferences" ON public.user_preferences;
CREATE POLICY "Users can manage their own preferences"
ON public.user_preferences FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all preferences" ON public.user_preferences;
CREATE POLICY "Service role can manage all preferences"
ON public.user_preferences FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- TASKS
DROP POLICY IF EXISTS "Users can manage their own tasks" ON public.tasks;
CREATE POLICY "Users can manage their own tasks"
ON public.tasks FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all tasks" ON public.tasks;
CREATE POLICY "Service role can manage all tasks"
ON public.tasks FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- CHAT CONVERSATIONS
DROP POLICY IF EXISTS "Users can manage their chat conversations" ON public.chat_conversations;
CREATE POLICY "Users can manage their chat conversations"
ON public.chat_conversations FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all conversations" ON public.chat_conversations;
CREATE POLICY "Service role can manage all conversations"
ON public.chat_conversations FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- CHAT MESSAGES
DROP POLICY IF EXISTS "Users can view their chat messages" ON public.chat_messages;
CREATE POLICY "Users can view their chat messages"
ON public.chat_messages FOR SELECT
USING (
  deleted_at IS NULL AND
  EXISTS (
    SELECT 1 FROM public.chat_conversations c
    WHERE c.id = chat_messages.conversation_id
      AND c.user_id = auth.uid()
      AND c.deleted_at IS NULL
  )
);

DROP POLICY IF EXISTS "Users can insert user messages" ON public.chat_messages;
CREATE POLICY "Users can insert user messages"
ON public.chat_messages FOR INSERT
WITH CHECK (
  role = 'user'
  AND EXISTS (
    SELECT 1 FROM public.chat_conversations c
    WHERE c.id = chat_messages.conversation_id
      AND c.user_id = auth.uid()
      AND c.deleted_at IS NULL
  )
);

DROP POLICY IF EXISTS "Service role can manage all messages" ON public.chat_messages;
CREATE POLICY "Service role can manage all messages"
ON public.chat_messages FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- FILE UPLOADS
DROP POLICY IF EXISTS "Users can manage their file uploads" ON public.file_uploads;
CREATE POLICY "Users can manage their file uploads"
ON public.file_uploads FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all file uploads" ON public.file_uploads;
CREATE POLICY "Service role can manage all file uploads"
ON public.file_uploads FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- BATCH JOBS
DROP POLICY IF EXISTS "Users can manage their batch jobs" ON public.batch_jobs;
CREATE POLICY "Users can manage their batch jobs"
ON public.batch_jobs FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all batch jobs" ON public.batch_jobs;
CREATE POLICY "Service role can manage all batch jobs"
ON public.batch_jobs FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- NOTIFICATIONS
DROP POLICY IF EXISTS "Users can manage their notifications" ON public.notifications;
CREATE POLICY "Users can manage their notifications"
ON public.notifications FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all notifications" ON public.notifications;
CREATE POLICY "Service role can manage all notifications"
ON public.notifications FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- API KEYS
DROP POLICY IF EXISTS "Users can manage their API keys" ON public.api_keys;
CREATE POLICY "Users can manage their API keys"
ON public.api_keys FOR ALL
USING (user_id = auth.uid() AND deleted_at IS NULL)
WITH CHECK (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all api keys" ON public.api_keys;
CREATE POLICY "Service role can manage all api keys"
ON public.api_keys FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- AUDIT LOGS
DROP POLICY IF EXISTS "Users can view their own audit logs" ON public.audit_logs;
CREATE POLICY "Users can view their own audit logs"
ON public.audit_logs FOR SELECT
USING (user_id = auth.uid());

DROP POLICY IF EXISTS "Service role can manage all audit logs" ON public.audit_logs;
CREATE POLICY "Service role can manage all audit logs"
ON public.audit_logs FOR ALL TO service_role
USING (true)
WITH CHECK (true);

-- ============================================================
-- UTILITY FUNCTIONS
-- ============================================================

CREATE OR REPLACE FUNCTION public.cleanup_expired_files()
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  WITH deleted AS (
    UPDATE public.file_uploads
    SET deleted_at = NOW()
    WHERE expires_at IS NOT NULL
      AND expires_at < NOW()
      AND deleted_at IS NULL
    RETURNING id
  )
  SELECT COUNT(*) INTO deleted_count FROM deleted;

  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.cleanup_old_notifications(days_old INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER;
BEGIN
  WITH deleted AS (
    UPDATE public.notifications
    SET deleted_at = NOW()
    WHERE created_at < NOW() - (days_old || ' days')::INTERVAL
      AND is_read = TRUE
      AND deleted_at IS NULL
    RETURNING id
  )
  SELECT COUNT(*) INTO deleted_count FROM deleted;

  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION public.get_user_stats(target_user_id UUID)
RETURNS TABLE(
  total_analyses BIGINT,
  completed_analyses BIGINT,
  pending_analyses BIGINT,
  total_chats BIGINT,
  unread_notifications BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    (SELECT COUNT(*) FROM public.tasks WHERE user_id = target_user_id AND type = 'analysis' AND deleted_at IS NULL),
    (SELECT COUNT(*) FROM public.tasks WHERE user_id = target_user_id AND type = 'analysis' AND status = 'completed' AND deleted_at IS NULL),
    (SELECT COUNT(*) FROM public.tasks WHERE user_id = target_user_id AND type = 'analysis' AND status IN ('pending', 'queued', 'processing') AND deleted_at IS NULL),
    (SELECT COUNT(*) FROM public.chat_conversations WHERE user_id = target_user_id AND deleted_at IS NULL),
    (SELECT COUNT(*) FROM public.notifications WHERE user_id = target_user_id AND is_read = FALSE AND deleted_at IS NULL);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================
-- BACKWARD COMPATIBILITY VIEWS
-- ============================================================

CREATE OR REPLACE VIEW public.analysis_jobs_view AS
SELECT
  id,
  user_id,
  status::public.analysis_status as status,
  CASE
    WHEN file_type LIKE 'image%' THEN 'image'::public.media_type
    WHEN file_type LIKE 'video%' THEN 'video'::public.media_type
    WHEN file_type LIKE 'audio%' THEN 'audio'::public.media_type
    WHEN file_type LIKE 'webcam%' THEN 'webcam'::public.media_type
    WHEN file_type LIKE 'batch%' THEN 'batch'::public.media_type
    ELSE 'multimodal'::public.media_type
  END as media_type,
  file_name,
  file_path,
  file_size,
  progress,
  metadata,
  error::text as error_message,
  report_code,
  started_at,
  completed_at,
  created_at,
  updated_at,
  deleted_at
FROM public.tasks
WHERE type = 'analysis';

CREATE OR REPLACE VIEW public.scans_view AS
SELECT
  id::text as id,
  user_id,
  file_name as filename,
  file_type as type,
  CASE
    WHEN result->>'is_deepfake' IS NOT NULL THEN
      CASE WHEN (result->>'is_deepfake')::boolean THEN 'Deepfake detected' ELSE 'Authentic' END
    ELSE 'Analysis complete'
  END as result,
  COALESCE((result->>'confidence')::float, 0.0) as confidence_score,
  result as detection_details,
  metadata,
  created_at,
  updated_at,
  deleted_at
FROM public.tasks
WHERE type = 'analysis';

-- ============================================================
-- FINAL COMMENT
-- ============================================================
COMMENT ON SCHEMA public IS 'SatyaAI Schema v2.6 - Tasks as Single Source of Truth';
