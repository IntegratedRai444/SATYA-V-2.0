import { createBrowserRouter, Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Suspense, lazy, useEffect } from 'react';
import LoadingState from '@/components/ui/LoadingState';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import { PageTransition } from '@/components/ui/PageTransition';

// Lazy load page components with custom loading
const lazyWithRetry = (componentImport: any) =>
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
        return window.location.reload();
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
const Analytics = lazyWithRetry(() => import('@/pages/Analytics'));
const SmartAnalysis = lazyWithRetry(() => import('../pages/SmartAnalysis'));
const History = lazyWithRetry(() => import('@/pages/History'));
const Settings = lazyWithRetry(() => import('@/pages/Settings'));
const Help = lazyWithRetry(() => import('@/pages/Help'));

const AIAssistant = lazyWithRetry(() => import('@/pages/AIAssistant'));
const BatchAnalysis = lazyWithRetry(() => import('@/pages/BatchAnalysis'));
const ImageAnalysis = lazyWithRetry(() => import('@/pages/ImageAnalysis'));
const VideoAnalysis = lazyWithRetry(() => import('@/pages/VideoAnalysis'));
const AudioAnalysis = lazyWithRetry(() => import('@/pages/AudioAnalysis'));
const WiringVerification = lazyWithRetry(() => import('@/pages/dev/WiringVerification'));
const NotFound = lazyWithRetry(() => import('@/pages/NotFound'));

// Layout components
import MainLayout from '@/components/layout/MainLayout';

// Shared components (these will be used within their respective pages)
// These are imported directly in the components that use them

// UI Components
import { Toaster } from '@/components/ui/toaster';

// Scroll to top on route change
const ScrollToTop = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
};

// Main app layout wrapper with transitions and error boundaries
const AppLayout = () => {
  return (
    <>
      <ScrollToTop />
      <ErrorBoundary>
        <PageTransition>
          <Suspense
            fallback={
              <div className="flex-1 flex items-center justify-center">
                <LoadingState variant="section" message="Loading application..." />
              </div>
            }
          >
            <MainLayout />
          </Suspense>
        </PageTransition>
        <Toaster />
      </ErrorBoundary>
    </>
  );
};

// Protected route component with better loading state
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading, user } = useAuth();
  const location = useLocation();

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingState variant="page" message="Checking authentication..." />
      </div>
    );
  }

  // Only redirect if definitely not authenticated
  // Check both isAuthenticated and user to avoid race conditions
  if (!isAuthenticated && !user) {
    // Store the attempted URL for redirecting after login
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
};

// Public route component with redirect
const PublicRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <LoadingState variant="inline" message="Loading..." />;
  }

  // If user is authenticated and trying to access auth pages, redirect to dashboard
  if (isAuthenticated && ['/login', '/register', '/forgot-password'].includes(location.pathname)) {
    const from = location.state?.from?.pathname || '/dashboard';
    return <Navigate to={from} replace />;
  }

  return <>{children}</>;
};

// Create and export the router
export const router = createBrowserRouter([
  // Public routes
  {
    path: '/login',
    element: (
      <PublicRoute>
        <Suspense fallback={<LoadingState variant="page" message="Loading login..." />}>
          <Login />
        </Suspense>
      </PublicRoute>
    ),
  },
  {
    path: '/forgot-password',
    element: (
      <PublicRoute>
        <Suspense fallback={<LoadingState variant="page" message="Loading password reset..." />}>
          <div>Forgot Password Page</div>
        </Suspense>
      </PublicRoute>
    ),
  },
  {
    path: '/reset-password/:token',
    element: (
      <PublicRoute>
        <Suspense fallback={<LoadingState variant="page" message="Loading password reset..." />}>
          <div>Reset Password Page</div>
        </Suspense>
      </PublicRoute>
    ),
  },
  {
    path: '/about',
    element: (
      <PublicRoute>
        <Suspense fallback={<LoadingState message="Loading..." isLoading={true} />}>
          <div>About Page</div>
        </Suspense>
      </PublicRoute>
    ),
  },

  // Protected routes with MainLayout
  {
    path: '/',
    element: (
      <ProtectedRoute>
        <AppLayout />
      </ProtectedRoute>
    ),
    children: [
      // Dashboard - now with MainLayout (DEFAULT HOME PAGE)
      {
        index: true,
        element: (
          <Suspense fallback={<LoadingState message="Loading dashboard..." isLoading={true} />}>
            <Dashboard />
          </Suspense>
        ),
      },
      {
        path: 'dashboard',
        element: (
          <Suspense fallback={<LoadingState message="Loading dashboard..." isLoading={true} />}>
            <Dashboard />
          </Suspense>
        ),
      },
      {
        path: 'home',
        element: (
          <Suspense fallback={<LoadingState message="Loading..." isLoading={true} />}>
            <Home />
          </Suspense>
        ),
      },
      {
        path: 'analytics',
        element: (
          <Suspense fallback={<LoadingState message="Loading analytics..." isLoading={true} />}>
            <Analytics />
          </Suspense>
        ),
      },
      {
        path: 'smart-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading Smart Analysis..." isLoading={true} />}>
            <SmartAnalysis />
          </Suspense>
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
          <Suspense fallback={<LoadingState message="Loading history..." isLoading={true} />}>
            <History />
          </Suspense>
        ),
      },
      {
        path: 'settings',
        element: (
          <Suspense fallback={<LoadingState message="Loading settings..." isLoading={true} />}>
            <Settings />
          </Suspense>
        ),
      },
      {
        path: 'help',
        element: (
          <Suspense fallback={<LoadingState message="Loading help..." isLoading={true} />}>
            <Help />
          </Suspense>
        ),
      },
      {
        path: 'scan-history',
        element: (
          <Suspense fallback={<LoadingState message="Loading scan history..." isLoading={true} />}>
            <History />
          </Suspense>
        ),
      },
      {
        path: 'ai-assistant',
        element: (
          <Suspense fallback={<LoadingState message="Loading AI Assistant..." isLoading={true} />}>
            <AIAssistant />
          </Suspense>
        ),
      },
      {
        path: 'batch-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading batch analysis..." isLoading={true} />}>
            <BatchAnalysis />
          </Suspense>
        ),
      },
      {
        path: 'image-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading image analysis..." isLoading={true} />}>
            <ImageAnalysis />
          </Suspense>
        ),
      },
      {
        path: 'video-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading video analysis..." isLoading={true} />}>
            <VideoAnalysis />
          </Suspense>
        ),
      },
      {
        path: 'audio-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading audio analysis..." isLoading={true} />}>
            <AudioAnalysis />
          </Suspense>
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
      <PublicRoute>
        <Suspense fallback={<LoadingState variant="page" message="Loading verification..." />}>
          <WiringVerification />
        </Suspense>
      </PublicRoute>
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
