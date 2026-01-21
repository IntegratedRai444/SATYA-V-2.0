// Validate environment variables before anything else
import { validateEnvironment } from './lib/config/validate-env';

// Run environment validation
validateEnvironment();

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'next-themes';
import { HelmetProvider } from 'react-helmet-async';
import * as Sentry from '@sentry/react';
import { Toaster } from '@/components/ui/toaster';
import { SupabaseAuthProvider } from '@/contexts/SupabaseAuthProvider';
import { AppProvider } from '@/contexts/AppContext';
import { RealtimeProvider } from '@/contexts/RealtimeContext';
import { BatchProcessingProvider } from '@/contexts/BatchProcessingContext';
import { RouterProvider } from 'react-router-dom';
import { router } from './utils/router';
// import { PerformanceMonitor, MemoryMonitor } from './utils/performanceOptimizer';
import { initSentry } from './lib/sentry';
import './index.css';

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
  <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center">
    <h1 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h1>
    <p className="mb-4">We've been notified about this issue and are working on it.</p>
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
// 3. ThemeProvider - Theme management
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
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <SupabaseAuthProvider>
            <AppProvider>
              {/* 6. RealtimeProvider - Real-time updates and WebSocket management */}
              <RealtimeProvider>
                {/* 7. BatchProcessingProvider - Batch processing state */}
                <BatchProcessingProvider>
                  {/* 8. Router - Application routing */}
                  <RouterProvider router={router} />
                </BatchProcessingProvider>
              </RealtimeProvider>
            </AppProvider>
          </SupabaseAuthProvider>
        </ThemeProvider>
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
