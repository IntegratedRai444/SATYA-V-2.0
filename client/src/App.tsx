import { RouterProvider } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import { ThemeProvider } from 'next-themes';
import { Toaster } from '@/components/ui/toaster';
import { SupabaseAuthProvider } from '@/contexts/SupabaseAuthProvider';
import { AppProvider } from '@/contexts/AppContext';
import { RealtimeProvider } from '@/contexts/RealtimeContext';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { useEffect } from 'react';
import { router } from './utils/router';
import { TooltipProvider } from '@/components/ui/tooltip';

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
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false} forcedTheme="dark">
          <TooltipProvider>
            <SupabaseAuthProvider>
              <AppProvider>
                <RealtimeProvider>
                  <AnalyticsWrapper>
                    <RouterProvider router={router} />
                    <Toaster />
                  </AnalyticsWrapper>
                </RealtimeProvider>
              </AppProvider>
            </SupabaseAuthProvider>
          </TooltipProvider>
        </ThemeProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
}

export default App;
