/**
 * Security Utilities
 * Input sanitization, XSS prevention, and security helpers
 */

import DOMPurify from 'isomorphic-dompurify';
import logger from './logger';

// ============================================================================
// Input Sanitization
// ============================================================================

/**
 * Sanitize HTML content to prevent XSS
 */
export function sanitizeHtml(dirty: string, options?: DOMPurify.Config): string {
  const clean = DOMPurify.sanitize(dirty, {
    ALLOWED_TAGS: ['b', 'i', 'em', 'strong', 'a', 'p', 'br', 'ul', 'ol', 'li'],
    ALLOWED_ATTR: ['href', 'title', 'target'],
    ALLOW_DATA_ATTR: false,
    ...options,
  });
  
  return clean;
}

/**
 * Sanitize user input (remove all HTML)
 */
export function sanitizeInput(input: string): string {
  return DOMPurify.sanitize(input, {
    ALLOWED_TAGS: [],
    ALLOWED_ATTR: [],
  });
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  
  return text.replace(/[&<>"']/g, (char) => map[char]);
}

// ============================================================================
// File Validation
// ============================================================================

export interface FileValidationOptions {
  maxSize?: number; // in bytes
  allowedTypes?: string[];
  allowedExtensions?: string[];
}

export interface FileValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Validate file before upload
 */
export function validateFile(
  file: File,
  options: FileValidationOptions = {}
): FileValidationResult {
  const errors: string[] = [];
  const {
    maxSize = 10 * 1024 * 1024, // 10MB default
    allowedTypes = [],
    allowedExtensions = [],
  } = options;

  // Check file size
  if (file.size > maxSize) {
    errors.push(`File size exceeds maximum allowed size of ${formatBytes(maxSize)}`);
  }

  // Check MIME type
  if (allowedTypes.length > 0 && !allowedTypes.includes(file.type)) {
    errors.push(`File type ${file.type} is not allowed`);
  }

  // Check file extension
  if (allowedExtensions.length > 0) {
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (!extension || !allowedExtensions.includes(extension)) {
      errors.push(`File extension .${extension} is not allowed`);
    }
  }

  // Check for suspicious file names
  if (isSuspiciousFileName(file.name)) {
    errors.push('File name contains suspicious characters');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Check if filename is suspicious
 */
function isSuspiciousFileName(filename: string): boolean {
  const suspiciousPatterns = [
    /\.\./,  // Directory traversal
    /[<>:"|?*]/,  // Invalid characters
    /^\./, // Hidden files
    /\.(exe|bat|cmd|sh|ps1)$/i, // Executable files
  ];

  return suspiciousPatterns.some(pattern => pattern.test(filename));
}

/**
 * Format bytes to human readable
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// ============================================================================
// CSRF Token Management
// ============================================================================

/**
 * Get CSRF token from cookie or meta tag
 */
export function getCsrfToken(): string | null {
  // Try to get from meta tag
  const metaTag = document.querySelector<HTMLMetaElement>('meta[name="csrf-token"]');
  if (metaTag) {
    return metaTag.content;
  }

  // Try to get from cookie
  const cookies = document.cookie.split(';');
  for (const cookie of cookies) {
    const [name, value] = cookie.trim().split('=');
    if (name === 'XSRF-TOKEN') {
      return decodeURIComponent(value);
    }
  }

  return null;
}

/**
 * Add CSRF token to request headers
 */
export function addCsrfHeader(headers: Record<string, string>): Record<string, string> {
  const token = getCsrfToken();
  if (token) {
    return {
      ...headers,
      'X-CSRF-TOKEN': token,
      'X-XSRF-TOKEN': token,
    };
  }
  return headers;
}

// ============================================================================
// Rate Limiting
// ============================================================================

class ClientRateLimiter {
  private requests = new Map<string, number[]>();

  /**
   * Check if request is allowed
   */
  canMakeRequest(key: string, limit: number, windowMs: number): boolean {
    const now = Date.now();
    const requests = this.requests.get(key) || [];

    // Remove old requests outside the window
    const recentRequests = requests.filter(time => now - time < windowMs);

    if (recentRequests.length >= limit) {
      logger.warn('Rate limit exceeded', { key, limit, window: windowMs });
      return false;
    }

    // Add current request
    recentRequests.push(now);
    this.requests.set(key, recentRequests);

    return true;
  }

  /**
   * Clear rate limit for key
   */
  clear(key: string): void {
    this.requests.delete(key);
  }

  /**
   * Clear all rate limits
   */
  clearAll(): void {
    this.requests.clear();
  }
}

// ============================================================================
// Secure Storage
// ============================================================================

/**
 * Secure storage wrapper (uses sessionStorage with encryption placeholder)
 */
export class SecureStorage {
  private prefix = 'satyaai_secure_';

  /**
   * Set item in secure storage
   */
  set(key: string, value: string): void {
    try {
      // In production, this should use encryption
      // For now, just use sessionStorage
      sessionStorage.setItem(this.prefix + key, value);
    } catch (error) {
      logger.error('Failed to set secure storage item', error as Error);
    }
  }

  /**
   * Get item from secure storage
   */
  get(key: string): string | null {
    try {
      return sessionStorage.getItem(this.prefix + key);
    } catch (error) {
      logger.error('Failed to get secure storage item', error as Error);
      return null;
    }
  }

  /**
   * Remove item from secure storage
   */
  remove(key: string): void {
    try {
      sessionStorage.removeItem(this.prefix + key);
    } catch (error) {
      logger.error('Failed to remove secure storage item', error as Error);
    }
  }

  /**
   * Clear all secure storage
   */
  clear(): void {
    try {
      const keys = Object.keys(sessionStorage);
      keys.forEach(key => {
        if (key.startsWith(this.prefix)) {
          sessionStorage.removeItem(key);
        }
      });
    } catch (error) {
      logger.error('Failed to clear secure storage', error as Error);
    }
  }
}

// ============================================================================
// Singleton Instances
// ============================================================================

export const rateLimiter = new ClientRateLimiter();
export const secureStorage = new SecureStorage();

// ============================================================================
// Exported Functions
// ============================================================================

export default {
  sanitizeHtml,
  sanitizeInput,
  escapeHtml,
  validateFile,
  getCsrfToken,
  addCsrfHeader,
  rateLimiter,
  secureStorage,
};
