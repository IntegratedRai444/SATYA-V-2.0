-- Performance optimization indexes for analysis_tasks table
-- These indexes significantly improve query performance for dashboard and history queries

-- Index for user-specific queries with status filter
-- Used by: dashboard stats, recent scans, user history
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_user_status 
  ON analysis_tasks(user_id, status);

-- Index for completed tasks ordered by completion time
-- Used by: recent scans, paginated history with cursor
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_user_completed 
  ON analysis_tasks(user_id, completed_at DESC) 
  WHERE status = 'completed';

-- Index for time-based queries (last 7 days, 24 hours, etc.)
-- Used by: dashboard stats time filters
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_created_at 
  ON analysis_tasks(created_at DESC);

-- Composite index for authenticity breakdown queries
-- Used by: dashboard stats authenticity counts
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_user_authenticity 
  ON analysis_tasks(user_id, authenticity, status)
  WHERE status = 'completed';

-- Index for report code lookups
-- Used by: report retrieval by code
CREATE INDEX IF NOT EXISTS idx_analysis_tasks_report_code 
  ON analysis_tasks(report_code);

-- Analyze tables to update statistics
ANALYZE analysis_tasks;
