-- Query Optimization Migration
-- Created: 2025-11-17
-- Purpose: Additional indexes and optimizations for common queries

-- Add indexes for analysis performance
CREATE INDEX IF NOT EXISTS idx_scans_result ON scans(result);
CREATE INDEX IF NOT EXISTS idx_scans_confidence ON scans(confidence_score);

-- Composite index for dashboard queries
CREATE INDEX IF NOT EXISTS idx_scans_user_result_created 
  ON scans(user_id, result, created_at DESC);

-- Index for file type filtering
CREATE INDEX IF NOT EXISTS idx_scans_filename ON scans(filename);

-- Optimize for date range queries
CREATE INDEX IF NOT EXISTS idx_scans_created_date ON scans(DATE(created_at));

-- Add covering index for common SELECT queries
CREATE INDEX IF NOT EXISTS idx_scans_summary 
  ON scans(user_id, type, result, confidence_score, created_at);

-- Vacuum and analyze for optimization
VACUUM;
ANALYZE;
