import { RouterProvider } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import { ThemeProvider } from 'next-themes';
import { Toaster } from '@/components/ui/toaster';
import { AppProvider } from '@/contexts/AppContext';
import { RealtimeProvider } from '@/contexts/RealtimeContext';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { useEffect } from 'react';
import { router } from './utils/router';
import { TooltipProvider } from '@/components/ui/tooltip';
import React from 'react';

// Analytics wrapper component
const AnalyticsWrapper = ({ children }: { children: React.ReactNode }) => {
  const location = window.location;

  useEffect(() => {
    // Track page view analytics
    // Analytics integration ready - using console for development, can be extended with GA4, Mixpanel, etc.
    if (import.meta.env.DEV) {
      console.log('Page view:', location.pathname);
    }
    
    // Future analytics integration points:
    // - Google Analytics 4: gtag('config', 'GA_MEASUREMENT_ID');
    // - Mixpanel: mixpanel.track('Page View', { path: location.pathname });
    // - Custom analytics: Send to backend endpoint
  }, [location.pathname]);

  return <>{children}</>;
};

function App() {
  return (
    <ErrorBoundary>
      <HelmetProvider>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false} forcedTheme="dark">
          <TooltipProvider>
            <AppProvider>
              <RealtimeProvider>
                <AnalyticsWrapper>
                  <RouterProvider router={router} />
                  <Toaster />
                </AnalyticsWrapper>
              </RealtimeProvider>
            </AppProvider>
          </TooltipProvider>
        </ThemeProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
}

export default App;
