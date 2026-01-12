import * as Sentry from '@sentry/react';
import { Integrations } from '@sentry/tracing';

// Type assertion to handle BrowserTracing type
type CompatibleIntegration = {
  name: string;
  setupOnce: any; // Using any to bypass the type checking issue
  [key: string]: any;
};

// Create a type-safe BrowserTracing integration
const createBrowserTracing = (): CompatibleIntegration => {
  return new Integrations.BrowserTracing({
    tracePropagationTargets: [
      window.location.hostname,
      /^https?:\/\/api\./,
    ],
  }) as unknown as CompatibleIntegration;
};
import { Component, ErrorInfo, ReactNode } from 'react';
import { toast } from '@/components/ui/use-toast';
import * as React from 'react';

// Fix for window type in Node.js environment
declare global {
  interface Window {
    __SENTRY__: any;
  }
}

// Environment configuration
const SENTRY_DSN = import.meta.env.VITE_SENTRY_DSN || '';
const ENVIRONMENT = import.meta.env.MODE || 'development';

// Initialize Sentry
const initSentry = (): void => {
  // Skip Sentry initialization in development or if DSN is not configured
  if (import.meta.env.DEV) {
    console.log('[Sentry] Running in development mode - Sentry will not be initialized');
    return;
  }

  if (!SENTRY_DSN) {
    console.warn('[Sentry] DSN not configured - error reporting will be disabled');
    return;
  }

  try {
    Sentry.init({
      dsn: SENTRY_DSN,
      environment: ENVIRONMENT,
      release: `satya-ai@${import.meta.env.PACKAGE_VERSION || '0.0.0'}`,
      
      // Performance Monitoring
      tracesSampleRate: 0.2, // 20% of transactions for performance monitoring
      
      // Session Replay
      replaysSessionSampleRate: 0.1, // 10% of sessions
      replaysOnErrorSampleRate: 1.0, // 100% of sessions with errors
      
      // Configure tracing
      integrations: [
        createBrowserTracing(),
      ],
      
      // Filter out common non-actionable errors
      beforeSend(event, hint) {
        const error = hint?.originalException as Error | undefined;
        
        if (error) {
          // Ignore common non-actionable errors
          const ignoredErrors = [
            'ResizeObserver',
            'Request failed',
            'Network Error',
            'Failed to fetch',
            'Network request failed',
            'Loading chunk',
          ];
          
          if (ignoredErrors.some(msg => error.message?.includes(msg))) {
            return null;
          }
        }
        
        return event;
      },
    });
    
    // Set user context when available
    try {
      const userData = localStorage.getItem('user');
      if (userData) {
        const user = JSON.parse(userData);
        if (user?.id) {
          Sentry.setUser({
            id: user.id,
            email: user.email,
            username: user.username,
          });
        }
      }
    } catch (e) {
      console.warn('Failed to set Sentry user context', e);
    }
    
    console.log('Sentry initialized successfully');
  } catch (error) {
    console.error('Failed to initialize Sentry:', error);
  }
};

// Error Boundary Component
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Report to Sentry
    Sentry.captureException(error, { extra: { componentStack: errorInfo.componentStack } });
    
    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // Show error toast
    toast({
      title: 'Something went wrong',
      description: error.message || 'An unexpected error occurred',
      variant: 'destructive',
    });
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: undefined });
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }
      
      return React.createElement(
        'div',
        { className: 'p-4 bg-red-50 dark:bg-red-900/20 rounded-lg' },
        [
          React.createElement(
            'h2',
            { className: 'text-lg font-semibold text-red-800 dark:text-red-200', key: 'title' },
            'Something went wrong'
          ),
          React.createElement(
            'p',
            { className: 'mt-2 text-sm text-red-700 dark:text-red-300', key: 'message' },
            this.state.error?.message || 'An unexpected error occurred'
          ),
          React.createElement(
            'button',
            {
              key: 'retry',
              onClick: this.handleReset,
              className: 'mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors'
            },
            'Try again'
          )
        ]
      );
    }

    return this.props.children;
  }
}

// Helper function to report errors
const reportError = (error: unknown, context?: Record<string, unknown>): void => {
  if (error instanceof Error) {
    Sentry.captureException(error, { extra: context });
  } else if (typeof error === 'string') {
    Sentry.captureMessage(error, { level: 'error', extra: context });
  } else {
    Sentry.captureException(new Error('Unknown error occurred'), { extra: { error, ...context } });
  }
};

// Export everything
export {
  Sentry,
  initSentry,
  ErrorBoundary,
  reportError,
};

export default Sentry;
