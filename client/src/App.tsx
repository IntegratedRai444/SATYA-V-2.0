import { RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HelmetProvider } from 'react-helmet-async';
import { ThemeProvider } from 'next-themes';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import { AuthProvider } from '@/contexts/AuthContext';
import { AppProvider } from '@/contexts/AppContext';
import { RealtimeProvider } from '@/contexts/RealtimeContext';
import { router } from './utils/router';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { useEffect } from 'react';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Analytics wrapper component
const AnalyticsWrapper = ({ children }: { children: React.ReactNode }) => {
  const location = window.location;

  useEffect(() => {
    // Track page view analytics
    // TODO: Integrate with proper analytics service (Google Analytics, Mixpanel, etc.)
    if (import.meta.env.DEV) {
      console.log('Page view:', location.pathname);
    }
  }, [location.pathname]);

  return <>{children}</>;
};

function App() {
  return (
    <ErrorBoundary>
      <HelmetProvider>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false} forcedTheme="dark">
            <TooltipProvider>
              <AuthProvider>
                <AppProvider>
                  <RealtimeProvider>
                    <AnalyticsWrapper>
                      <RouterProvider router={router} />
                      <Toaster />
                    </AnalyticsWrapper>
                  </RealtimeProvider>
                </AppProvider>
              </AuthProvider>
            </TooltipProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
}

export default App;
