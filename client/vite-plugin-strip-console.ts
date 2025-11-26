/**
 * Vite plugin to strip console statements from production builds
 * Removes console.log, console.debug while preserving console.warn and console.error
 */

import type { Plugin } from 'vite';

interface StripConsoleOptions {
  include?: string[];
  exclude?: string[];
}

export function stripConsolePlugin(options: StripConsoleOptions = {}): Plugin {
  const {
    include = ['console.log', 'console.debug'],
    exclude = ['console.warn', 'console.error', 'console.info'],
  } = options;

  return {
    name: 'vite-plugin-strip-console',
    enforce: 'post',
    
    apply: 'build', // Only apply in build mode
    
    transform(code: string, id: string) {
      // Only process TypeScript and JavaScript files
      if (!/\.(tsx?|jsx?)$/.test(id)) {
        return null;
      }

      // Skip node_modules
      if (id.includes('node_modules')) {
        return null;
      }

      let modifiedCode = code;
      let hasChanges = false;

      // Remove specified console methods
      for (const method of include) {
        const regex = new RegExp(
          `\\b${method.replace('.', '\\.')}\\s*\\([^)]*\\)\\s*;?`,
          'g'
        );
        
        if (regex.test(modifiedCode)) {
          modifiedCode = modifiedCode.replace(regex, '');
          hasChanges = true;
        }
      }

      return hasChanges ? { code: modifiedCode, map: null } : null;
    },
  };
}
