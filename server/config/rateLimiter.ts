import { RateLimiterMemory } from 'rate-limiter-flexible';

// Rate limiting rules
export const rateLimitRules = {
  websocket: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 100,
    message: {
      points: 100, // 100 messages
      duration: 1, // per second per connection
    },
    connection: {
      points: 5, // 5 connection attempts
      duration: 60, // per minute per IP
    },
  },
};

// Create rate limiters
export const messageRateLimiter = new RateLimiterMemory({
  points: rateLimitRules.websocket.message.points,
  duration: rateLimitRules.websocket.message.duration,
});

export const connectionRateLimiter = new RateLimiterMemory({
  points: rateLimitRules.websocket.connection.points,
  duration: rateLimitRules.websocket.connection.duration,
});

// Helper function to extract token from URL query
export function extractTokenFromQuery(url: string): string | null {
  try {
    const urlObj = new URL(url, 'http://dummy');
    return urlObj.searchParams.get('token');
  } catch (e) {
    return null;
  }
}
