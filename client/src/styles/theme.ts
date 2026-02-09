/**
 * SatyaAI Dark Theme Configuration
 * Centralized theme constants for consistent styling across the application
 */

const darkTheme = {
  colors: {
    // Background colors
    bg: {
      primary: '#0a0a0a',
      secondary: '#1a1a1a',
      tertiary: '#2a2a2a',
      card: '#1e1e1e',
      sidebar: '#141414',
    },
    // Text colors
    text: {
      primary: '#ffffff',
      secondary: '#b3b3b3',
      muted: '#666666',
    },
    // Accent colors - simplified to single primary
    accent: {
      primary: '#3b82f6', // Blue instead of cyan
      secondary: '#2c5282',
    },
    // Border colors
    border: {
      primary: '#333333',
      secondary: '#444444',
    },
  },
  typography: {
    fonts: {
      heading: "'Inter', 'Poppins', sans-serif",
      body: "'Roboto', system-ui, sans-serif",
    },
    sizes: {
      hero: '3.5rem',      // 56px
      h1: '2.5rem',        // 40px
      h2: '1.875rem',      // 30px
      h3: '1.5rem',        // 24px
      bodyLarge: '1.125rem', // 18px
      body: '1rem',        // 16px
      small: '0.875rem',   // 14px
    },
    weights: {
      normal: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
  },
  spacing: {
    xs: '0.25rem',   // 4px
    sm: '0.5rem',    // 8px
    md: '1rem',      // 16px
    lg: '1.5rem',    // 24px
    xl: '2rem',      // 32px
    '2xl': '3rem',   // 48px
    '3xl': '4rem',   // 64px
  },
  borderRadius: {
    sm: '0.375rem',  // 6px
    md: '0.5rem',    // 8px
    lg: '0.75rem',   // 12px
    xl: '1rem',      // 16px
    '2xl': '1.5rem', // 24px
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
    xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
  },
  transitions: {
    fast: '150ms',
    normal: '200ms',
    slow: '300ms',
  },
} as const;

export type Theme = typeof darkTheme;

// Helper function to get theme value
export const getThemeValue = (path: string): string => {
  const keys = path.split('.');
  let value: unknown = darkTheme;
  
  for (const key of keys) {
    value = (value as Record<string, unknown>)[key];
    if (value === undefined) {
      console.warn(`Theme value not found for path: ${path}`);
      return '';
    }
  }
  
  return typeof value === 'string' ? value : String(value);
};

export default darkTheme;
