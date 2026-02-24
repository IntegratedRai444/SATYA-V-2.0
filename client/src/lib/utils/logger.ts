/**
 * Production-ready Logger Utility
 * Provides environment-based logging with production noise filtering
 */

export const logger = {
  info: (...args: any[]) => {
    if (import.meta.env.DEV) {
      console.log(...args);
    }
    // Silence info logs in production
  },
  warn: (...args: any[]) => console.warn(...args),
  error: (...args: any[]) => console.error(...args),
  debug: (...args: any[]) => {
    if (import.meta.env.DEV) {
      console.debug(...args);
    }
    // Silence debug logs in production
  }
};
