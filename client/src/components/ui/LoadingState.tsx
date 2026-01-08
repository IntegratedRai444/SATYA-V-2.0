import { Loader2, AlertCircle, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { cn } from '@/lib/utils';

interface LoadingStateProps {
  message?: string;
  showRetry?: boolean;
  onRetry?: () => void;
  error?: string;
  connectionStatus?: 'connected' | 'disconnected' | 'checking';
  isLoading?: boolean;
  className?: string;
  children?: React.ReactNode;
  variant?: 'page' | 'section' | 'inline' | 'skeleton';
  skeletonCount?: number;
}

export const Skeleton = ({ className }: { className?: string }) => (
  <div className={cn('animate-pulse bg-gray-200 dark:bg-gray-700 rounded', className)} />
);

export const SkeletonLoader = ({
  count = 1,
  className = '',
  skeletonClassName = '',
}: {
  count?: number;
  className?: string;
  skeletonClassName?: string;
}) => (
  <div className={cn('space-y-4', className)}>
    {Array.from({ length: count }).map((_, i) => (
      <Skeleton key={i} className={cn('h-4 w-full', skeletonClassName)} />
    ))}
  </div>
);

export default function LoadingState({
  message = "Loading SatyaAI...",
  showRetry = false,
  onRetry,
  error,
  connectionStatus = 'checking',
  isLoading = true,
  className = '',
  variant = 'page',
  skeletonCount = 3,
}: LoadingStateProps) {
  // Handle skeleton variant
  if (variant === 'skeleton') {
    return (
      <div className={className}>
        <SkeletonLoader count={skeletonCount} />
      </div>
    );
  }

  // Handle inline variant
  if (variant === 'inline') {
    return (
      <div className={cn('inline-flex items-center gap-2', className)}>
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="text-sm">{message}</span>
      </div>
    );
  }

  // Handle section variant
  if (variant === 'section') {
    return (
      <div className={cn('flex flex-col items-center justify-center py-12', className)}>
        <Loader2 className="h-8 w-8 animate-spin mb-4" />
        <p className="text-muted-foreground">{message}</p>
      </div>
    );
  }
  const getStatusIcon = () => {
    switch (connectionStatus) {
      case 'connected':
        return <Wifi className="h-5 w-5 text-green-500" />;
      case 'disconnected':
        return <WifiOff className="h-5 w-5 text-red-500" />;
      case 'checking':
      default:
        return <Loader2 className="h-5 w-5 animate-spin text-blue-500" />;
    }
  };

  const getStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected to SatyaAI server';
      case 'disconnected':
        return 'Disconnected from server';
      case 'checking':
      default:
        return 'Checking connection...';
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-900">
      <div className="text-center max-w-md mx-auto p-6">
        {/* Main Loading Indicator */}
        <div className="mb-6">
          {isLoading ? (
            <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
          ) : error ? (
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
          ) : (
            <div className="h-12 w-12 mx-auto mb-4 flex items-center justify-center">
              <div className="w-8 h-8 bg-green-500 rounded-full"></div>
            </div>
          )}

          <h2 className="text-xl font-semibold text-white mb-2">
            {error ? 'Connection Error' : message}
          </h2>
        </div>

        {/* Connection Status */}
        <div className="flex items-center justify-center gap-2 mb-4 p-3 bg-slate-800 rounded-lg">
          {getStatusIcon()}
          <span className="text-sm text-slate-300">{getStatusText()}</span>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-500/20 rounded-lg">
            <p className="text-red-400 text-sm mb-2">{error}</p>

            {/* Development Tips */}
            {import.meta.env.DEV && (
              <div className="text-xs text-slate-400 mt-3 space-y-1">
                <p><strong>Development Tips:</strong></p>
                <ul className="list-disc list-inside space-y-1 text-left">
                  <li>Ensure backend server is running on port 5001</li>
                  <li>Check if Python AI server is running on port 8000</li>
                  <li>Try running: <code className="bg-slate-700 px-1 rounded">npm run dev:all</code></li>
                  <li>Check browser console for detailed errors</li>
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Retry Button */}
        {showRetry && onRetry && (
          <button
            onClick={onRetry}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            Retry Connection
          </button>
        )}

        {/* Loading Progress */}
        {isLoading && !error && (
          <div className="mt-6">
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
            </div>
            <p className="text-xs text-slate-400 mt-2">Initializing SatyaAI components...</p>
          </div>
        )}


      </div>
    </div>
  );
}