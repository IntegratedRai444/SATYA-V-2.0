// Import the schema implementation
import { users, scans, userPreferences, tasks } from './schema';

// Re-export all types
export type { 
  InsertUser, 
  User, 
  InsertScan, 
  Scan, 
  InsertUserPreferences, 
  UserPreferences, 
  InsertTask, 
  Task 
} from './index.d';

// Export the schema implementation
export { users, scans, userPreferences, tasks };

// Export insert schemas
export { 
  insertUserSchema, 
  insertScanSchema, 
  insertUserPreferencesSchema, 
  insertTaskSchema 
} from './schema';
