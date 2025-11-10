import { createBrowserRouter, Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Suspense, lazy, useEffect } from 'react';
import LoadingState, { SkeletonLoader } from '@/components/ui/LoadingState';
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
const DetectionTools = lazyWithRetry(() => import('@/pages/DetectionTools'));
const UploadAnalysis = lazyWithRetry(() => import('@/pages/UploadAnalysis'));
const Scan = lazyWithRetry(() => import('@/pages/Scan'));
const ImageAnalysis = lazyWithRetry(() => import('@/pages/ImageAnalysis'));
const VideoAnalysis = lazyWithRetry(() => import('@/pages/VideoAnalysis'));
const AudioAnalysis = lazyWithRetry(() => import('@/pages/AudioAnalysis'));
const History = lazyWithRetry(() => import('@/pages/History'));
const Settings = lazyWithRetry(() => import('@/pages/Settings'));
const Help = lazyWithRetry(() => import('@/pages/Help'));
const WebcamLive = lazyWithRetry(() => import('@/pages/WebcamLive'));
const LandingPage = lazyWithRetry(() => import('@/pages/LandingPage'));
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
    <MainLayout>
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
            <Outlet />
          </Suspense>
        </PageTransition>
        <Toaster />
      </ErrorBoundary>
    </MainLayout>
  );
};

// Protected route component with better loading state
const ProtectedRoute = ({ children }: { children: React.ReactNode }) => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingState variant="page" message="Checking authentication..." />
      </div>
    );
  }

  if (!isAuthenticated) {
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
    path: '/',
    element: (
      <PublicRoute>
        <LandingPage />
      </PublicRoute>
    ),
  },
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
    element: (
      <ProtectedRoute>
        <AppLayout />
      </ProtectedRoute>
    ),
    children: [
      // Dashboard - now with MainLayout (DEFAULT HOME PAGE)
      {
        path: '/',
        element: (
          <Suspense fallback={<LoadingState message="Loading dashboard..." isLoading={true} />}>
            <Dashboard />
          </Suspense>
        ),
      },
      {
        path: '/dashboard',
        element: (
          <Suspense fallback={<LoadingState message="Loading dashboard..." isLoading={true} />}>
            <Dashboard />
          </Suspense>
        ),
      },
      {
        path: '/home',
        element: (
          <Suspense fallback={<LoadingState message="Loading..." isLoading={true} />}>
            <Home />
          </Suspense>
        ),
      },
      {
        path: '/analytics',
        element: (
          <Suspense fallback={<LoadingState message="Loading analytics..." isLoading={true} />}>
            <Analytics />
          </Suspense>
        ),
      },
      {
        path: '/detection-tools',
        element: (
          <Suspense fallback={<LoadingState message="Loading tools..." isLoading={true} />}>
            <DetectionTools />
          </Suspense>
        ),
      },
      {
        path: '/upload',
        element: (
          <Suspense fallback={<LoadingState message="Preparing upload..." isLoading={true} />}>
            <UploadAnalysis />
          </Suspense>
        ),
      },
      {
        path: '/scan/:id',
        element: (
          <Suspense fallback={<LoadingState message="Loading scan..." isLoading={true} />}>
            <Scan />
          </Suspense>
        ),
      },
      {
        path: '/image-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading image analysis..." isLoading={true} />}>
            <ImageAnalysis />
          </Suspense>
        ),
      },
      {
        path: '/video-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading video analysis..." isLoading={true} />}>
            <VideoAnalysis />
          </Suspense>
        ),
      },
      {
        path: '/audio-analysis',
        element: (
          <Suspense fallback={<LoadingState message="Loading audio analysis..." isLoading={true} />}>
            <AudioAnalysis />
          </Suspense>
        ),
      },
      {
        path: '/history',
        element: (
          <Suspense fallback={<LoadingState message="Loading history..." isLoading={true} />}>
            <History />
          </Suspense>
        ),
      },
      {
        path: '/settings',
        element: (
          <Suspense fallback={<LoadingState message="Loading settings..." isLoading={true} />}>
            <Settings />
          </Suspense>
        ),
      },
      {
        path: '/help',
        element: (
          <Suspense fallback={<LoadingState message="Loading help..." isLoading={true} />}>
            <Help />
          </Suspense>
        ),
      },
      {
        path: '/webcam',
        element: (
          <Suspense fallback={<LoadingState message="Loading webcam..." isLoading={true} />}>
            <WebcamLive />
          </Suspense>
        ),
      },
      {
        path: '/webcam-live',
        element: (
          <Suspense fallback={<LoadingState message="Loading webcam..." isLoading={true} />}>
            <WebcamLive />
          </Suspense>
        ),
      },
      {
        path: '/scan-history',
        element: (
          <Suspense fallback={<LoadingState message="Loading scan history..." isLoading={true} />}>
            <History />
          </Suspense>
        ),
      },
    ],
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
