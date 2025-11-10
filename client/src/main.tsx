import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from 'next-themes';
import { HelmetProvider } from 'react-helmet-async';
import { AuthProvider } from './contexts/AuthContext';
import { AppProvider } from './contexts/AppContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { RealtimeProvider } from './contexts/RealtimeContext';
import { BatchProcessingProvider } from './contexts/BatchProcessingContext';
import { RouterProvider } from 'react-router-dom';
import { router } from './utils/router';
import './index.css';

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

// Render the app with all context providers
// Provider order: outer to inner (general to specific)
// 1. HelmetProvider - Document head management
// 2. QueryClientProvider - React Query for data fetching
// 3. ThemeProvider - Theme management
// 4. AuthProvider - Authentication state
// 5. AppProvider - Global app state (notifications, preferences)
// 6. WebSocketProvider - WebSocket connection management
// 7. RealtimeProvider - Real-time updates and notifications
// 8. BatchProcessingProvider - Batch upload/processing state
root.render(
  <StrictMode>
    <HelmetProvider>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <AuthProvider>
            <AppProvider>
              <WebSocketProvider>
                <RealtimeProvider>
                  <BatchProcessingProvider>
                    <RouterProvider router={router} />
                  </BatchProcessingProvider>
                </RealtimeProvider>
              </WebSocketProvider>
            </AppProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </HelmetProvider>
  </StrictMode>
);
