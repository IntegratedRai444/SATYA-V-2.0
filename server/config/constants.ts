// Authentication constants - CRITICAL: No fallback defaults for production security
const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  throw new Error('JWT_SECRET environment variable is required for security');
}

export const JWT_SECRET_FINAL = JWT_SECRET;
export const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '1d';

// WebSocket configuration
export const WS_MAX_MESSAGES_PER_SECOND = 100;
export const WS_MAX_CONNECTIONS_PER_IP = 5;
export const WS_HEARTBEAT_INTERVAL = 30000; // 30 seconds

// Rate limiting
export const RATE_LIMIT_WINDOW_MS = 15 * 60 * 1000; // 15 minutes
export const MAX_REQUESTS_PER_WINDOW = 100;
