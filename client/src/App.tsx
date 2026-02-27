import { RouterProvider } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import { Toaster } from '@/components/ui/toaster';
import { AppProvider } from '@/contexts/AppContext';
import { AuthProvider } from '@/contexts/AuthContext';
import { RealtimeProvider } from '@/contexts/RealtimeContext';
import { ErrorBoundary } from '@/components/ui/ErrorBoundary';
import { useEffect } from 'react';
import { router } from './utils/router';
import { TooltipProvider } from '@/components/ui/tooltip';
import React from 'react';

// Simple Theme Provider to replace next-themes
const SimpleThemeProvider = ({ children }: { children: React.ReactNode }) => {
  useEffect(() => {
    // Force dark theme
    document.documentElement.classList.add('dark');
    document.documentElement.setAttribute('data-theme', 'dark');
  }, []);
  
  return <>{children}</>;
};

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
        <SimpleThemeProvider>
          <TooltipProvider>
            <AppProvider>
              <AuthProvider>
                <RealtimeProvider>
                  <AnalyticsWrapper>
                    <RouterProvider router={router} />
                    <Toaster />
                  </AnalyticsWrapper>
                </RealtimeProvider>
              </AuthProvider>
            </AppProvider>
          </TooltipProvider>
        </SimpleThemeProvider>
      </HelmetProvider>
    </ErrorBoundary>
  );
}

export default App;
