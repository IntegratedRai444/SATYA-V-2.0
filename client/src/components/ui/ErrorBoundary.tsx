import { Component, ErrorInfo, ReactNode } from 'react';
import { FiAlertTriangle, FiRefreshCw, FiCopy, FiExternalLink } from 'react-icons/fi';
import logger from '../../utils/logger';
import { classifyError, handleError } from '../../utils/errorHandling';

export interface ErrorBoundaryProps {
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

class ErrorBoundary extends Component<ErrorBoundaryProps, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const level = this.props.level || 'component';

    // Enhanced error object with additional context
    const enhancedError = {
      ...error,
      context: {
        level,
        timestamp: new Date().toISOString(),
        path: window.location.pathname,
        componentStack: errorInfo?.componentStack,
      },
    };

    // Log error using centralized logger with more context
    logger.error(`ErrorBoundary (${level}) caught an error`, enhancedError, {
      componentStack: errorInfo.componentStack,
      level,
      timestamp: enhancedError.context.timestamp,
      path: enhancedError.context.path,
    });

    // Classify and handle error
    const classified = classifyError(enhancedError);
    const shouldShowToast = level !== 'app' && classified.severity !== 'low';

    handleError(enhancedError, {
      showToast: shouldShowToast,
      logToConsole: true,
      reportToService: classified.severity === 'high' || classified.severity === 'critical',
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(enhancedError, errorInfo);
    }

    // Store error info in state with enhanced context
    this.setState({
      error: enhancedError,
      errorInfo,
      hasError: true,
    });

    // For critical errors, log to error tracking service
    if (classified.severity === 'critical') {
      this.logToErrorService(enhancedError, errorInfo);
    }
  }

  private handleCopyError = () => {
    if (this.state.error) {
      const errorDetails = `Error: ${this.state.error.message}\n\nStack Trace:\n${this.state.error.stack || 'No stack trace available'}`;
      navigator.clipboard.writeText(errorDetails);
      // Using console.log instead of toast since we don't have the toast library installed
      console.log('Error details copied to clipboard');
    }
  };

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  private logToErrorService = (error: Error, errorInfo: ErrorInfo) => {
    // This is a placeholder for actual error tracking service integration
    // In a real app, this would send the error to a service like Sentry, LogRocket, etc.
    console.group('Error Report');
    console.error('Error:', error);
    console.error('Error Info:', errorInfo);
    console.groupEnd();

    // Example of sending to an error tracking endpoint
    if (process.env.NODE_ENV === 'production') {
      try {
        // Replace with your actual error tracking endpoint
        fetch('/api/error-report', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            error: {
              name: error.name,
              message: error.message,
              stack: error.stack,
              componentStack: errorInfo?.componentStack,
            },
            context: {
              url: window.location.href,
              userAgent: navigator.userAgent,
              timestamp: new Date().toISOString(),
            },
          }),
        }).catch(console.error);
      } catch (e) {
        console.error('Failed to send error report:', e);
      }
    }
  };

  private handleReportError = () => {
    if (this.state.error) {
      const error = this.state.error as any; // Type assertion to access context
      const subject = encodeURIComponent(`[SATYA-AI ${this.props.level || 'error'}] ${error.name}: ${error.message.substring(0, 50)}`);

      let body = `Error Details:\n\n`;
      body += `Message: ${error.message}\n`;
      body += `Type: ${error.name || 'Error'}\n`;
      body += `Level: ${this.props.level || 'component'}\n`;
      body += `Timestamp: ${error.context?.timestamp || new Date().toISOString()}\n\n`;
      body += `Stack Trace:\n${error.stack || 'No stack trace available'}\n\n`;
      body += `Component Stack:\n${error.context?.componentStack || 'No component stack'}\n\n`;
      body += `Browser: ${navigator.userAgent}\n`;
      body += `URL: ${error.context?.path || window.location.href}\n`;
      body += `Environment: ${process.env.NODE_ENV}\n`;

      const mailtoUrl = `mailto:support@satyaai.com?subject=${subject}&body=${encodeURIComponent(body)}`;
      window.open(mailtoUrl, '_blank');
    }
  };

  private getErrorTitle() {
    if (!this.state.error) return 'Something went wrong';

    // Customize error titles based on error type or message
    const { error } = this.state;

    if (error.name === 'ChunkLoadError') {
      return 'Update Required';
    }

    if (error.message.includes('NetworkError')) {
      return 'Connection Error';
    }

    if (error.message.includes('timeout') || error.message.includes('Timeout')) {
      return 'Request Timed Out';
    }

    return 'Something went wrong';
  }

  private getErrorMessage() {
    if (!this.state.error) return 'An unexpected error occurred. Please try again.';
    const { error } = this.state;
    if (!error) return 'An unexpected error occurred. Please try again.';
    
    // Safe property access with optional chaining
    const hasIncludes = error && typeof error === 'object' && 'includes' in error;
    
    if (error.name === 'ChunkLoadError') {
      return 'A new version of the application is available. Please refresh your browser to update.';
    }
    
    if (error.message && hasIncludes && error.message.includes('NetworkError')) {
      return 'Unable to connect to the server. Please check your internet connection and try again.';
    }
    
    if (error.message && hasIncludes && (error.message.includes('timeout') || error.message.includes('Timeout'))) {
      return 'The request took too long to complete. Please check your connection and try again.';
    }
    
    return error.message || 'An unexpected error occurred. Please try again.';
  }

  public render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        if (typeof this.props.fallback === 'function') {
          return this.props.fallback(this.state.error!, this.handleRetry);
        }
        return this.props.fallback;
      }

      const errorTitle = this.getErrorTitle();
      const errorMessage = this.getErrorMessage();
      const isRecoverable = !['ChunkLoadError', 'TypeError', 'ReferenceError'].includes(this.state.error?.name || '');

      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center p-6">
          <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 max-w-md w-full text-center">
            <div className="w-16 h-16 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <FiAlertTriangle className="w-8 h-8 text-red-400" />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">{errorTitle}</h2>
            <p className="text-gray-300 mb-6">{errorMessage}</p>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="text-left p-3 bg-gray-900/50 rounded-md text-xs text-gray-400 font-mono overflow-auto max-h-32 mb-4">
                <div className="font-bold text-red-400 mb-1">
                  {this.state.error.name || 'Error'}
                </div>
                {this.state.error.message && (
                  <div className="text-red-300 mb-2">{this.state.error.message}</div>
                )}
                {this.state.error.stack && (
                  <pre className="text-gray-500 text-xs mt-2 overflow-auto">
                    {this.state.error.stack}
                  </pre>
                )}
              </div>
            )}

            <div className="flex flex-col space-y-3">
              {isRecoverable && (
                <button
                  onClick={this.handleRetry}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center justify-center"
                >
                  <FiRefreshCw className="w-4 h-4 mr-2" />
                  Try Again
                </button>
              )}

              <div className="flex space-x-2">
                <button
                  onClick={this.handleCopyError}
                  className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-md transition-colors flex items-center justify-center space-x-2 text-sm"
                >
                  <FiCopy className="w-4 h-4" />
                  <span>Copy Error</span>
                </button>
                <button
                  onClick={this.handleReportError}
                  className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-md transition-colors flex items-center justify-center space-x-2 text-sm"
                >
                  <FiExternalLink className="w-4 h-4" />
                  <span>Report</span>
                </button>
              </div>

              <a
                href="/"
                className="block px-4 py-2 text-blue-400 hover:text-blue-300 transition-colors text-sm"
              >
                Back to Home
              </a>

              {this.state.error?.name === 'ChunkLoadError' && (
                <button
                  onClick={() => window.location.reload()}
                  className="mt-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors flex items-center justify-center space-x-2 text-sm"
                >
                  <FiRefreshCw className="w-4 h-4 animate-spin" />
                  <span>Update Application</span>
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export { ErrorBoundary };
export default ErrorBoundary;