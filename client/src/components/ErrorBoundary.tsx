import { Component, ErrorInfo, ReactNode } from 'react';

declare const process: {
  env: {
    NODE_ENV?: string;
  };
};

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Frontend Error Boundary caught an error:', error.message);
    console.error('Error stack:', error.stack);
    console.error('Component stack:', errorInfo.componentStack);

    // In production, you might want to send this to an error reporting service
    if (process.env.NODE_ENV === 'production') {
      // TODO: Send to error monitoring service (Sentry, etc.)
      console.error('Production Error:', error, errorInfo);
    }
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{
          padding: '2rem',
          textAlign: 'center',
          backgroundColor: '#fee',
          border: '1px solid #fcc',
          borderRadius: '8px',
          margin: '1rem'
        }}>
          <h2 style={{ color: '#c33' }}>Something went wrong</h2>
          <p>We're sorry, but something unexpected happened.</p>
          <button 
            onClick={() => window.location.reload()}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#c33',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Reload Page
          </button>
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details style={{ marginTop: '1rem', textAlign: 'left' }}>
              <summary>Error Details (Development Only)</summary>
              <pre style={{ 
                backgroundColor: '#f5f5f5', 
                padding: '1rem', 
                borderRadius: '4px',
                overflow: 'auto',
                fontSize: '0.8rem'
              }}>
                {this.state.error.stack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
