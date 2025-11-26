-- Data Cleanup Migration
-- Created: 2025-11-17
-- Purpose: Clean up old data and optimize storage

-- Create trigger to auto-delete old scans (optional, commented out by default)
-- Uncomment if you want automatic cleanup after 90 days
/*
CREATE TRIGGER IF NOT EXISTS cleanup_old_scans
AFTER INSERT ON scans
BEGIN
  DELETE FROM scans 
  WHERE created_at < datetime('now', '-90 days')
  AND user_id = NEW.user_id;
END;
*/

-- Create trigger to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_scans_timestamp
AFTER UPDATE ON scans
FOR EACH ROW
BEGIN
  UPDATE scans SET updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_users_timestamp
AFTER UPDATE ON users
FOR EACH ROW
BEGIN
  UPDATE users SET updated_at = CURRENT_TIMESTAMP
  WHERE id = NEW.id;
END;

-- Optimize database
PRAGMA optimize;
VACUUM;
ANALYZE;
