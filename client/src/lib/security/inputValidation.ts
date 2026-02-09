/**
 * Input Validation Utility
 * Provides comprehensive validation and sanitization for user inputs
 */

/**
 * Sanitizes string input by removing potentially dangerous characters
 */
export const sanitizeInput = (input: string): string => {
  if (typeof input !== 'string') {
    return '';
  }

  // Remove potentially dangerous characters
  return input
    .replace(/[<>]/g, '') // Remove HTML tags
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .replace(/data:/gi, '') // Remove data: protocol
    .replace(/vbscript:/gi, '') // Remove vbscript: protocol
    .replace(/on\w+\s*=/gi, '') // Remove event handlers
    .trim();
};

/**
 * Validates file names and extensions
 */
export const validateFileName = (fileName: string): { isValid: boolean; error?: string } => {
  if (!fileName || typeof fileName !== 'string') {
    return { isValid: false, error: 'Invalid file name' };
  }

  // Check for dangerous patterns
  const dangerousPatterns = [
    /\.\./,  // Directory traversal
    /[<>:"|?*]/,  // Special characters
    /^(con|prn|aux|nul|com[1-9])$/, // Reserved names
  ];

  const hasDangerousPattern = dangerousPatterns.some(pattern => pattern.test(fileName.toLowerCase()));

  if (hasDangerousPattern) {
    return { isValid: false, error: 'File name contains invalid characters or patterns' };
  }

  // Check length
  if (fileName.length > 255) {
    return { isValid: false, error: 'File name too long' };
  }

  return { isValid: true };
};

/**
 * Validates email addresses
 */
export const validateEmail = (email: string): { isValid: boolean; error?: string } => {
  if (!email || typeof email !== 'string') {
    return { isValid: false, error: 'Invalid email format' };
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return { isValid: false, error: 'Invalid email format' };
  }

  return { isValid: true };
};

/**
 * Validates URLs to prevent XSS attacks
 */
export const validateUrl = (url: string): { isValid: boolean; error?: string } => {
  if (!url || typeof url !== 'string') {
    return { isValid: false, error: 'Invalid URL' };
  }

  try {
    const parsedUrl = new URL(url);
    
    // Only allow http/https protocols
    if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
      return { isValid: false, error: 'Only HTTP and HTTPS protocols are allowed' };
    }

    // Check for dangerous characters
    if (url.includes('<') || url.includes('>') || url.includes('javascript:')) {
      return { isValid: false, error: 'URL contains potentially dangerous content' };
    }

    return { isValid: true };
  } catch {
    return { isValid: false, error: 'Invalid URL format' };
  }
};

/**
 * Validates and sanitizes user-generated content
 */
export const validateUserContent = (content: string): { isValid: boolean; sanitized: string; error?: string } => {
  if (!content || typeof content !== 'string') {
    return { isValid: false, error: 'Invalid content', sanitized: '' };
  }

  // Basic XSS prevention
  const sanitized = sanitizeInput(content);

  // Check for common attack patterns
  const attackPatterns = [
    /<script\b[^<]*(?:(?!.*<\/script>))*[^>]*<\/script>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi,
    /data:text\/html/gi,
    /vbscript:/gi,
  ];

  const hasAttackPattern = attackPatterns.some(pattern => pattern.test(content));

  if (hasAttackPattern) {
    return { 
      isValid: false, 
      sanitized: sanitized.replace(/[<>]/g, ''), // Remove tags as fallback
      error: 'Content contains potentially dangerous code' 
    };
  }

  // Check length limits
  if (content.length > 10000) {
    return { 
      isValid: false, 
      sanitized: sanitized.substring(0, 1000), // Truncate if too long
      error: 'Content too long' 
    };
  }

  return { isValid: true, sanitized };
};

/**
 * Rate limiting helper
 */
export const checkRateLimit = (identifier: string, maxRequests: number = 100, windowMs: number = 60000): boolean => {
  const key = `rate_limit_${identifier}`;
  const now = Date.now();
  const lastRequest = localStorage.getItem(key);
  
  if (lastRequest) {
    const lastTime = parseInt(lastRequest);
    if (now - lastTime < windowMs) {
      return false; // Rate limited
    }
  }
  
  localStorage.setItem(key, now.toString());
  return true; // Not rate limited
};
