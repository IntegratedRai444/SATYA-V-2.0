// Validate environment variables before anything else
import { validateEnvironment } from './lib/config/validate-env';

// Run environment validation with error handling
try {
  validateEnvironment();
} catch (error) {
  if (import.meta.env.DEV) {
    console.error('Environment validation failed:', error);
  }
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
import { SimpleThemeProvider } from './components/ThemeProvider';
import { FallbackComponent } from './components/ErrorFallback';

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

// Create a fallback component for errors - moved to ErrorFallback.tsx

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
