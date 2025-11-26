import { Component, type ErrorInfo, type ReactNode } from 'react';
import { FiAlertTriangle, FiRefreshCw, FiHome, FiCopy, FiExternalLink } from 'react-icons/fi';
import { toast } from '@/components/ui/use-toast';
import logger from '@/lib/logger';
import { handleError, classifyError } from '@/lib/errorHandler';

interface Props {
  children: ReactNode;
  fallback?: ReactNode | ((error: Error, reset: () => void) => ReactNode);
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  level?: 'app' | 'page' | 'component';
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const level = this.props.level || 'component';
    
    // Log error using centralized logger
    logger.error(`ErrorBoundary (${level}) caught an error`, error, {
      componentStack: errorInfo.componentStack,
      level,
    });
    
    // Classify and handle error
    const classified = classifyError(error);
    handleError(error, {
      showToast: level !== 'app', // Don't show toast for app-level errors (full page error)
      logToConsole: true,
      reportToService: classified.severity === 'high' || classified.severity === 'critical',
    });
    
    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
    
    // Store error info in state
    this.setState({ errorInfo });
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined });
  };

  private handleCopyError = () => {
    if (this.state.error) {
      const errorDetails = `Error: ${this.state.error.message}\n\nStack Trace:\n${this.state.error.stack || 'No stack trace available'}`;
      navigator.clipboard.writeText(errorDetails);
      toast({
        title: 'Error details copied to clipboard',
        description: 'You can now paste this in a support ticket.',
      });
    }
  };

  private handleReportError = () => {
    if (this.state.error) {
      const subject = encodeURIComponent('Bug Report: Error in SatyaAI Application');
      const body = encodeURIComponent(
        `Error Details:\n\n` +
        `Message: ${this.state.error.message}\n\n` +
        `Stack Trace:\n${this.state.error.stack || 'No stack trace available'}\n\n` +
        `Browser: ${navigator.userAgent}\n` +
        `URL: ${window.location.href}\n`
      );
      window.open(`mailto:support@satyaai.com?subject=${subject}&body=${body}`, '_blank');
    }
  };

  public render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        if (typeof this.props.fallback === 'function') {
          return this.props.fallback(this.state.error!, this.handleRetry);
        }
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center p-6">
          <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 max-w-md w-full text-center">
            <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <FiAlertTriangle className="w-8 h-8 text-red-400" />
            </div>
            
            <h2 className="text-xl font-semibold text-white mb-2">
              Something went wrong
            </h2>
            
            <p className="text-gray-400 mb-6">
              An unexpected error occurred in the SatyaAI application. Please try the options below to resolve the issue.
            </p>
            
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="bg-gray-900 rounded p-4 mb-6 text-left">
                <p className="text-red-400 text-sm font-mono">
                  {this.state.error.message}
                </p>
                {this.state.error.stack && (
                  <pre className="text-gray-500 text-xs mt-2 overflow-auto">
                    {this.state.error.stack}
                  </pre>
                )}
              </div>
            )}
            
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={this.handleRetry}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
                >
                  <FiRefreshCw className="w-4 h-4 mr-2" />
                  Try Again
                </button>
                
                <button
                  onClick={() => window.location.reload()}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
                >
                  <FiRefreshCw className="w-4 h-4 mr-2" />
                  Refresh Page
                </button>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={this.handleCopyError}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
                >
                  <FiCopy className="w-4 h-4 mr-2" />
                  Copy Error
                </button>

                <button
                  onClick={this.handleReportError}
                  className="w-full bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
                >
                  <FiExternalLink className="w-4 h-4 mr-2" />
                  Report Issue
                </button>
              </div>

              <button
                onClick={() => window.location.href = '/'}
                className="w-full bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center mt-2"
              >
                <FiHome className="w-4 h-4 mr-2" />
                Go to Home
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Export as both default and named export
export { ErrorBoundary };
export default ErrorBoundary;