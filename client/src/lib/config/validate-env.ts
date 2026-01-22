
// Type definitions for required environment variables
type EnvConfig = {
  // API Configuration
  VITE_AUTH_API_URL: string;
  VITE_ANALYSIS_API_URL: string;
  VITE_API_URL: string;
  VITE_WS_URL?: string;
  
  // Feature Flags
  VITE_ENABLE_ANALYTICS?: string;
  VITE_ENABLE_DEBUG?: string;
  VITE_ENABLE_ML_MODELS?: string;
};

// Required environment variables with validation rules
const ENV_CONFIG: Record<keyof EnvConfig, {
  required: boolean;
  type: 'string' | 'number' | 'boolean' | 'url';
  pattern?: RegExp;
  minLength?: number;
  description: string;
  validate?: (value: string) => { valid: boolean; message?: string };
}> = {
  VITE_API_URL: {
    required: true,
    type: 'url',
    description: 'Main API base URL',
    validate: (value: string) => {
      try {
        new URL(value);
        return { valid: true };
      } catch {
        return { valid: false, message: 'Must be a valid URL' };
      }
    }
  },
  VITE_AUTH_API_URL: {
    required: true,
    type: 'url',
    description: 'Authentication API base URL',
    validate: (value: string) => {
      try {
        new URL(value);
        return { valid: true };
      } catch {
        return { valid: false, message: 'Must be a valid URL' };
      }
    }
  },
  VITE_ANALYSIS_API_URL: {
    required: true,
    type: 'url',
    description: 'Analysis API base URL',
    validate: (value: string) => {
      try {
        new URL(value);
        return { valid: true };
      } catch {
        return { valid: false, message: 'Must be a valid URL' };
      }
    }
  },
  VITE_WS_URL: {
    required: false,
    type: 'url',
    description: 'WebSocket URL for real-time updates (defaults to VITE_ANALYSIS_API_URL/ws)',
    validate: (value: string) => ({
      valid: value.startsWith('ws://') || value.startsWith('wss://'),
      message: 'Must be a valid WebSocket URL (ws:// or wss://)'
    })
  },
  VITE_ENABLE_ANALYTICS: {
    required: false,
    type: 'boolean',
    description: 'Enable analytics tracking',
    validate: (value: string) => ({
      valid: ['true', 'false', ''].includes(value.toLowerCase()),
      message: 'Must be either "true" or "false"'
    })
  },
  VITE_ENABLE_DEBUG: {
    required: false,
    type: 'boolean',
    description: 'Enable debug mode',
    validate: (value: string) => ({
      valid: ['true', 'false', ''].includes(value.toLowerCase()),
      message: 'Must be either "true" or "false"'
    })
  },
  VITE_ENABLE_ML_MODELS: {
    required: false,
    type: 'boolean',
    description: 'Enable ML models',
    validate: (value: string) => ({
      valid: ['true', 'false', ''].includes(value.toLowerCase()),
      message: 'Must be either "true" or "false"'
    })
  }
};

/**
 * Validates that all required environment variables are set and valid
 * @throws {Error} If any required environment variables are missing or invalid
 */
export function validateEnvironment(): void {
  if (typeof window === 'undefined') {
    return; // Skip in SSR/SSG
  }

  const errors: string[] = [];
  const warnings: string[] = [];
  const env = import.meta.env;

  // Check for missing required variables
  Object.entries(ENV_CONFIG).forEach(([key, config]) => {
    const value = env[key];
    
    if (config.required && !value) {
      errors.push(`Missing required environment variable: ${key} (${config.description})`);
      return;
    }

    if (!value) return; // Skip validation for empty optional values

    // Type validation
    switch (config.type) {
      case 'number':
        if (isNaN(Number(value))) {
          errors.push(`Invalid type for ${key}: expected number, got "${value}"`);
        }
        break;
      case 'boolean':
        if (!['true', 'false', ''].includes(value.toLowerCase())) {
          errors.push(`Invalid boolean value for ${key}: must be "true" or "false"`);
        }
        break;
      case 'url':
        try {
          new URL(value);
        } catch {
          errors.push(`Invalid URL for ${key}: "${value}"`);
        }
        break;
    }

    // Custom validation
    if (config.validate) {
      const { valid, message } = config.validate(value);
      if (!valid) {
        errors.push(`Invalid value for ${key}: ${message || 'validation failed'}`);
      }
    }
  });

  // Check for hardcoded API URLs in the codebase (development only)
  if (import.meta.env.DEV) {
    checkForHardcodedUrls();
  }

  // Handle errors and warnings
  if (warnings.length > 0) {
    console.warn('Environment configuration warnings:', warnings.join('\n  - '));
  }

  if (errors.length > 0) {
    const errorMessage = `Environment configuration errors:\n  - ${errors.join('\n  - ')}`;
    console.error(errorMessage);
    showErrorToUser(errorMessage);
    throw new Error('Invalid environment configuration');
  }
}

/**
 * Checks for hardcoded API URLs in the codebase (development only)
 */
function checkForHardcodedUrls(): void {
  // Skip URL checks in development
  if (import.meta.env.DEV) {
    return;
  }

  // This is a client-side check, so we can only check the current page
  const scripts = Array.from(document.getElementsByTagName('script'));
  const hardcodedUrls = new Set<string>();

  scripts.forEach(script => {
    if (script.src) {
      const isViteDevServer = script.src.includes('@vite/client') || 
                            script.src.includes('src/main.tsx') ||
                            script.src.includes('@react-refresh');
      
      if (!isViteDevServer) {
        ['localhost', '127.0.0.1', 'http://', 'https://'].forEach(term => {
          if (script.src.includes(term) && !script.src.includes(import.meta.env.VITE_API_URL)) {
            hardcodedUrls.add(script.src);
          }
        });
      }
    }
  });

  if (hardcodedUrls.size > 0) {
    console.warn('Potential hardcoded URLs found in scripts. Use environment variables instead:', 
      Array.from(hardcodedUrls).join(', '));
  }
}

/**
 * Displays an error message to the user in the UI
 */
function showErrorToUser(message: string): void {
  const root = document.getElementById('root');
  if (!root) return;

  root.innerHTML = `
    <div style="
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      padding: 1.5rem;
      background-color: #f8d7da;
      color: #721c24;
      border-bottom: 1px solid #f5c6cb;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      z-index: 9999;
    ">
      <h2 style="margin-top: 0; margin-bottom: 1rem;">Configuration Error</h2>
      <pre style="
        background: rgba(0,0,0,0.05);
        padding: 1rem;
        border-radius: 4px;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0;
      ">${message}</pre>
      <p style="margin-top: 1rem; margin-bottom: 0.5rem;">
        Please check your environment configuration and restart the application.
      </p>
      <p style="margin: 0.5rem 0 0 0; font-size: 0.9em; color: #856404;">
        This error occurs when required environment variables are missing or invalid.
      </p>
    </div>
  `;
}

// Run validation when this module is imported
if (typeof window !== 'undefined') {
  validateEnvironment();
}

// Export a type-safe environment configuration object
export const env = new Proxy({} as EnvConfig, {
  get(_, prop: string) {
    if (import.meta.env[prop] === undefined) {
      console.warn(`Environment variable ${prop} is not defined`);
    }
    return import.meta.env[prop];
  }
});
