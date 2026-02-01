// Validate environment variables before anything else
import { validateEnvironment } from './lib/config/validate-env';

// Run environment validation with error handling
try {
  validateEnvironment();
} catch (error) {
  console.error('Environment validation failed:', error);
  // Continue anyway for development
}

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HelmetProvider } from 'react-helmet-async';
import * as Sentry from '@sentry/react';
import { AppProvider } from '@/contexts/AppContext';
import { NotificationProvider } from '@/contexts/NotificationContext';
import { RealtimeProvider } from '@/contexts/RealtimeContext';
import { BatchProcessingProvider } from '@/contexts/BatchProcessingContext';
import { RouterProvider } from 'react-router-dom';
import { router } from './utils/router';
import React from 'react';
// import { PerformanceMonitor, MemoryMonitor } from './utils/performanceOptimizer';
import { initSentry } from './lib/sentry';
import './index.css';

// Simple Theme Provider to replace next-themes
const SimpleThemeProvider = ({ children }: { children: React.ReactNode }) => {
  React.useEffect(() => {
    // Force dark theme
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-theme', 'dark');
  }, []);
  
  return <>{children}</>;
};

// Initialize Sentry
initSentry();

// Start performance monitoring
// PerformanceMonitor.mark('app-init-start');

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

// Get the root element
const container = document.getElementById('root');

if (!container) {
  throw new Error('Failed to find the root element');
}

const root = createRoot(container);

// Create a fallback component for errors
const FallbackComponent = () => (
  <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center bg-black text-white">
    <h1 className="text-2xl font-bold text-red-600 mb-4">🚨 SATYA AI ERROR</h1>
    <p className="mb-4">Something went wrong with the application.</p>
    <div className="mb-4 p-4 bg-gray-800 rounded text-left">
      <h3 className="text-lg font-mono mb-2">Debug Info:</h3>
      <p>URL: {window.location.href}</p>
      <p>User Agent: {navigator.userAgent}</p>
      <p>Timestamp: {new Date().toISOString()}</p>
    </div>
    <button
      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      onClick={() => window.location.reload()}
    >
      Reload Page
    </button>
  </div>
);

// Create error boundary
const SentryErrorBoundary = Sentry.ErrorBoundary || (({ children }: { children: React.ReactNode }) => <>{children}</>);

// Render the app with all context providers
// Provider order: outer to inner (general to specific)
// 1. HelmetProvider - Document head management
// 2. QueryClientProvider - React Query for data fetching
// 3. ThemeProvider - Theme management (FIXED: use dark theme)
// 4. AuthProvider - Authentication state
// 5. AppProvider - Global app state (notifications, preferences)
// 6. RealtimeProvider - Real-time updates and WebSocket management
// 7. BatchProcessingProvider - Batch processing state
// 8. Router - Application routing
root.render(
  <StrictMode>
    <SentryErrorBoundary fallback={FallbackComponent}>
      <HelmetProvider>
      <QueryClientProvider client={queryClient}>
        <SimpleThemeProvider>
          <NotificationProvider>
            <AppProvider>
              {/* 6. RealtimeProvider - Real-time updates and WebSocket management */}
              <RealtimeProvider>
                {/* 7. BatchProcessingProvider - Batch operation management */}
                <BatchProcessingProvider>
                  <RouterProvider router={router} />
                </BatchProcessingProvider>
              </RealtimeProvider>
            </AppProvider>
          </NotificationProvider>
        </SimpleThemeProvider>
      </QueryClientProvider>
    </HelmetProvider>
    </SentryErrorBoundary>
  </StrictMode>
);

// End performance monitoring
// PerformanceMonitor.mark('app-init-end');
// PerformanceMonitor.measure('app-initialization', 'app-init-start', 'app-init-end');

// Log memory usage in development
if (import.meta.env.DEV) {
  setTimeout(() => {
    // MemoryMonitor.logMemoryUsage('Initial Load');
    // logger.debug('Performance Metrics', PerformanceMonitor.getMetrics());
  }, 2000);
}
