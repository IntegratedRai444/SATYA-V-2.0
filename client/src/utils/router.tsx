import { createBrowserRouter, Navigate } from 'react-router-dom';
import { Suspense, lazy } from 'react';
import LoadingState from '@/components/ui/LoadingState';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import React from 'react';
import AppLayout from '@/components/layout/AppLayout';

// Lazy load page components with custom loading
const lazyWithRetry = (componentImport: () => Promise<{ default: React.ComponentType }>) =>
  lazy(async () => {
    const pageHasAlreadyBeenForceRefreshed = JSON.parse(
      localStorage.getItem('page-has-been-force-refreshed') || 'false'
    );

    try {
      const component = await componentImport();
      window.localStorage.setItem('page-has-been-force-refreshed', 'false');
      return component;
    } catch (error) {
      if (!pageHasAlreadyBeenForceRefreshed) {
        // Assuming that the user is not on the latest version of the application.
        // Let's refresh the page immediately.
        window.localStorage.setItem('page-has-been-force-refreshed', 'true');
        window.location.reload();
        // This will never resolve because the page reloads
        return new Promise<{ default: React.ComponentType }>(() => {});
      }

      // The page has already been reloaded
      // User will be redirected to the error page
      throw error;
    }
  });

// Lazy load page components with retry
const Home = lazyWithRetry(() => import('@/pages/Home'));
const Dashboard = lazyWithRetry(() => import('@/pages/Dashboard'));
const Login = lazyWithRetry(() => import('@/pages/Login'));
const Register = lazyWithRetry(() => import('@/pages/auth/Register'));
const Analytics = lazyWithRetry(() => import('@/pages/Analytics'));
const MultimodalAnalysis = lazyWithRetry(() => import('../pages/MultimodalAnalysis'));
const History = lazyWithRetry(() => import('@/pages/History'));
const Settings = lazyWithRetry(() => import('@/pages/Settings'));
const Help = lazyWithRetry(() => import('@/pages/Help'));

const AIAssistant = lazyWithRetry(() => import('@/pages/AIAssistant'));
// const BatchAnalysis = lazyWithRetry(() => import('@/pages/BatchAnalysis')); // DISABLED
const ImageAnalysis = lazyWithRetry(() => import('@/pages/ImageAnalysis'));
const VideoAnalysis = lazyWithRetry(() => import('@/pages/VideoAnalysis'));
const AudioAnalysis = lazyWithRetry(() => import('@/pages/AudioAnalysis'));
const WiringVerification = lazyWithRetry(() => import('@/pages/dev/WiringVerification'));
const NotFound = lazyWithRetry(() => import('@/pages/NotFound'));

// Layout components
// Shared components (these will be used within their respective pages)
// These are imported directly in the components that use them

// UI Components


// Create and export the router
export const router = createBrowserRouter([
  // Public routes
  // Auth routes
  {
    path: '/register',
    element: (
      <Suspense fallback={<LoadingState variant="page" message="Loading registration..." />}>
        <Register />
      </Suspense>
    ),
  },
  {
    path: '/login',
    element: (
      <Suspense fallback={<LoadingState variant="page" message="Loading login..." />}>
        <Login />
      </Suspense>
    ),
  },
  {
    path: '/forgot-password',
    element: (
      <Suspense fallback={<LoadingState variant="page" message="Loading password reset..." />}>
        <div>Forgot Password Page</div>
      </Suspense>
    ),
  },
  {
    path: '/reset-password/:token',
    element: (
      <Suspense fallback={<LoadingState variant="page" message="Loading password reset..." />}>
        <div>Reset Password Page</div>
      </Suspense>
    ),
  },
  {
    path: '/about',
    element: (
      <Suspense fallback={<LoadingState message="Loading..." isLoading={true} />}>
        <div>About Page</div>
      </Suspense>
    ),
  },

  // Protected routes with MainLayout
  {
    path: '/',
    element: <AppLayout />,
    children: [
      // Redirect root to dashboard
      {
        index: true,
        element: <Navigate to="/dashboard" replace />
      },
      // Dashboard route - Single instance with error boundary
      {
        path: 'dashboard',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading dashboard..." />}>
              <Dashboard />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'home',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading home..." />}>
              <Home />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'analytics',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading analytics..." />}>
              <Analytics />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'smart-analysis',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading Multimodal Analysis..." />}>
              <MultimodalAnalysis />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'scan',
        element: <Navigate to="/smart-analysis" replace />,
      },
      {
        path: 'scan/:id',
        element: <Navigate to="/smart-analysis" replace />,
      },
      {
        path: 'history',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading history..." />}>
              <History />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'settings',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading settings..." />}>
              <Settings />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'help',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading help..." />}>
              <Help />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'scan-history',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading scan history..." />}>
              <History />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'ai-assistant',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading AI Assistant..." />}>
              <AIAssistant />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      // {
      //   path: 'batch-analysis',
      //   element: (
      //     <ErrorBoundary level="page">
      //       <Suspense fallback={<LoadingState message="Loading batch analysis..." />}>
      //         <BatchAnalysis />
      //       </Suspense>
      //     </ErrorBoundary>
      //   ),
      // }, // DISABLED
      {
        path: 'image-analysis',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading image analysis..." />}>
              <ImageAnalysis />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'video-analysis',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading video analysis..." />}>
              <VideoAnalysis />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'audio-analysis',
        element: (
          <ErrorBoundary level="page">
            <Suspense fallback={<LoadingState message="Loading audio analysis..." />}>
              <AudioAnalysis />
            </Suspense>
          </ErrorBoundary>
        ),
      },
      {
        path: 'webcam-live',
        element: <Navigate to="/smart-analysis" replace />,
      },
      {
        path: 'multimodal-analysis',
        element: <Navigate to="/smart-analysis" replace />,
      },
    ],
  },

  // Dev Verification Route
  {
    path: '/dev-wiring',
    element: (
      <Suspense fallback={<LoadingState variant="page" message="Loading verification..." />}>
        <WiringVerification />
      </Suspense>
    ),
  },

  // 404 route - must be last
  {
    path: '/404',
    element: (
      <ErrorBoundary>
        <Suspense fallback={<LoadingState variant="page" message="Loading..." />}>
          <NotFound />
        </Suspense>
      </ErrorBoundary>
    ),
  },
  // Redirect all unknown paths to 404
  {
    path: '*',
    element: <Navigate to="/404" replace />,
  },
]);

// Router is already exported above
