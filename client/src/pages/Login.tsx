import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/SupabaseAuthProvider';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield, AlertTriangle, User, Lock, Loader2 } from 'lucide-react';

export default function Login() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { login, error, isLoading, clearError, isAuthenticated } = useAuth();

  useEffect(() => {
    console.log('Login component mounted');
    console.log('Auth state:', { error, isLoading });

    // Redirect if already authenticated
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate, error, isLoading]);

  const [showPassword, setShowPassword] = useState(false);
  const [showRoleSelection, setShowRoleSelection] = useState(false);
  const [selectedRole, setSelectedRole] = useState<'user' | 'admin'>('admin');
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [formErrors, setFormErrors] = useState<{ email?: string; password?: string; general?: string }>({});
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const validateReturnUrl = (url: string | null): string | null => {
    if (!url) return null;
    try {
      const parsed = new URL(url, window.location.origin);
      if (parsed.origin === window.location.origin) {
        return parsed.pathname + parsed.search + parsed.hash;
      }
    } catch (e) {
      console.warn('Invalid return URL:', url);
    }
    return null;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error for the field being edited
    if (formErrors[name as keyof typeof formErrors]) {
      setFormErrors(prev => ({ ...prev, [name]: undefined }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError && clearError();

    if (!formData.email || !formData.password) {
      setFormErrors({
        email: !formData.email ? 'Email is required' : undefined,
        password: !formData.password ? 'Password is required' : undefined
      });
      return;
    }

    try {
      setNetworkError(null);
      setFormErrors({});

      console.log('Attempting login with:', { email: formData.email, passwordLength: formData.password.length });

      if (login) {
        const result = await login({
          email: formData.email,
          password: formData.password
        });

        console.log('Login result:', result);

        if (result.user) {
          const returnUrl = validateReturnUrl(searchParams.get('returnUrl')) || '/dashboard';
          console.log('Login successful, navigating to:', returnUrl);
          navigate(returnUrl);
        } else {
          throw new Error('Login failed - no user returned');
        }
      }
    } catch (err: unknown) {
      console.error('Login error details:', {
        message: err instanceof Error ? err.message : 'Unknown error',
        response: err && typeof err === 'object' && 'response' in err ? (err as { response?: { status?: number } }).response : undefined,
        status: err && typeof err === 'object' && 'response' in err ? (err as { response?: { status?: number } }).response?.status : undefined,
        stack: err instanceof Error ? err.stack : undefined
      });

      const errorObj = err as { message?: string; response?: { status?: number } };

      if (errorObj.message === 'Network Error') {
        setNetworkError('Unable to connect to server. Please check your internet connection and ensure server is running on port 5001.');
      } else if (errorObj.response) {
        switch (errorObj.response.status) {
          case 401:
            setFormErrors({ ...formErrors, general: 'Invalid email or password' });
            break;
          case 403:
            setFormErrors({ ...formErrors, general: 'Access denied. Please contact support.' });
            break;
          case 500:
            setFormErrors({ ...formErrors, general: 'Server error. Please try again later.' });
            break;
          default:
            setFormErrors({ ...formErrors, general: `Server error (${errorObj.response.status}). Please try again.` });
        }
      } else {
        setFormErrors({ ...formErrors, general: 'An unexpected error occurred. Please try again.' });
      }
    }
  };

  const handleRoleSelect = (role: 'user' | 'admin') => {
    setSelectedRole(role);
  };

  const ErrorBoundary = ({ children }: { children: React.ReactNode }) => {
    const [hasError, setHasError] = useState(false);

    useEffect(() => {
      const handleError = (error: ErrorEvent) => {
        console.error('Error caught by error boundary:', error);
        setHasError(true);
      };

      window.addEventListener('error', handleError);
      return () => window.removeEventListener('error', handleError);
    }, []);

    if (hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-4">
          <div className="text-center p-6 max-w-md bg-red-900/20 rounded-lg">
            <h2 className="text-xl font-bold text-red-400 mb-2">Something went wrong</h2>
            <p className="text-red-200 mb-4">An unexpected error occurred. Please refresh the page and try again.</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return <>{children}</>;
  };

  const renderError = () => {
    const errorMessage = networkError || formErrors.general || (error ? 'An error occurred during login' : '');

    if (!errorMessage) return null;

    return (
      <div className="mb-4">
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4 mr-2" />
          <AlertDescription>
            {errorMessage}
          </AlertDescription>
        </Alert>
      </div>
    );
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-white p-4">
        <div className="w-full max-w-md mb-8">
          {/* Logo and Header */}
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full mb-6 shadow-lg">
              <Shield className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">Satya AI</h1>
            <p className="text-gray-600 text-lg font-medium">Deepfake Detection System</p>
          </div>
        </div>

        {/* Illustration */}
        <div className="w-full max-w-md mb-8">
          <div className="flex justify-center">
            <div className="relative">
              {/* Illustrated character placeholder */}
              <div className="w-32 h-32 bg-gradient-to-br from-blue-100 to-blue-200 rounded-full flex items-center justify-center">
                <div className="w-24 h-24 bg-white rounded-full flex items-center justify-center">
                  <Lock className="w-12 h-12 text-blue-500" />
                </div>
              </div>
              {/* Floating elements */}
              <div className="absolute -top-2 -right-2 w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <div className="w-4 h-4 bg-white rounded-full"></div>
              </div>
              <div className="absolute -bottom-2 -left-2 w-6 h-6 bg-blue-400 rounded-full flex items-center justify-center">
                <div className="w-3 h-3 bg-white rounded-full"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Subtitle */}
        <p className="text-center text-gray-600 text-base mb-8 max-w-md">
          Secure access to advanced deepfake detection and analysis tools
        </p>

        {/* Login Card */}
        <div className="bg-white border border-gray-200 rounded-2xl p-8 shadow-xl w-full max-w-md">
          {/* Secure Login Header */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <Lock className="w-5 h-5 text-blue-500" />
            <h2 className="text-2xl font-semibold text-gray-800">Secure Login</h2>
          </div>

          {renderError()}

          <form onSubmit={handleSubmit} className="space-y-6">
            {!showRoleSelection ? (
              <>
                {/* Email Address */}
                <div className="space-y-2">
                  <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                    Email address
                    {formErrors.email && (
                      <span className="text-red-500 text-xs ml-2">{formErrors.email}</span>
                    )}
                  </label>
                  <div className="mt-1">
                    <Input
                      id="email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      required
                      value={formData.email}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border ${formErrors.email ? 'border-red-300' : 'border-gray-300'
                        } rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors`}
                      placeholder="Enter your email address"
                      aria-invalid={!!formErrors.email}
                      aria-describedby={formErrors.email ? 'email-error' : undefined}
                    />
                  </div>
                </div>

                {/* Password */}
                <div className="space-y-2">
                  <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                    Password
                    {formErrors.password && (
                      <span className="text-red-500 text-xs ml-2">{formErrors.password}</span>
                    )}
                  </label>
                  <div className="mt-1 relative">
                    <Input
                      id="password"
                      name="password"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="current-password"
                      required
                      value={formData.password}
                      onChange={handleInputChange}
                      className={`w-full px-4 py-3 border ${formErrors.password ? 'border-red-300' : 'border-gray-300'
                        } rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors`}
                      placeholder="Enter your password"
                      aria-invalid={!!formErrors.password}
                      aria-describedby={formErrors.password ? 'password-error' : undefined}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-blue-500 transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <>
                {/* Role Selection for Rishabh Kapoor */}
                <div className="space-y-4">
                  <div className="text-center mb-4">
                    <p className="text-gray-600 text-sm mb-2">Welcome, Rishabh Kapoor!</p>
                    <p className="text-gray-500 text-xs">Select your login role:</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <button
                      type="button"
                      onClick={() => handleRoleSelect('admin')}
                      className={`p-4 rounded-lg border-2 transition-all ${selectedRole === 'admin'
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-gray-200 bg-white hover:border-gray-300'
                        }`}
                    >
                      <Shield className={`w-8 h-8 mx-auto mb-2 ${selectedRole === 'admin' ? 'text-blue-400' : 'text-gray-500'
                        }`} />
                      <p className={`font-medium ${selectedRole === 'admin' ? 'text-blue-400' : 'text-gray-600'
                        }`}>
                        Admin
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Full Access</p>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleRoleSelect('user')}
                      className={`p-4 rounded-lg border-2 transition-all ${selectedRole === 'user'
                          ? 'border-cyan-500 bg-cyan-500/10'
                          : 'border-gray-200 bg-white hover:border-gray-300'
                        }`}
                    >
                      <User className={`w-8 h-8 mx-auto mb-2 ${selectedRole === 'user' ? 'text-cyan-400' : 'text-gray-500'
                        }`} />
                      <p className={`font-medium ${selectedRole === 'user' ? 'text-cyan-400' : 'text-gray-600'
                        }`}>
                        User
                      </p>
                      <p className="text-xs text-gray-500 mt-1">Standard Access</p>
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => setShowRoleSelection(false)}
                    className="w-full text-sm text-gray-500 hover:text-blue-500 transition-colors mt-2"
                  >
                    ← Back to login
                  </button>
                </div>
              </>
            )}

            {/* Secure Login Button */}
            <div className="space-y-4">
              <Button
                type="submit"
                className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-75 transition-all duration-200"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" />
                    Signing in...
                  </>
                ) : (
                  'Sign in'
                )}
              </Button>

              <div className="text-center space-y-2">
                <div className="text-sm text-gray-600">
                  <a
                    href="/forgot-password"
                    className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
                    onClick={(e) => {
                      e.preventDefault();
                      setFormErrors({
                        ...formErrors,
                        general: 'Please contact support to reset your password.'
                      });
                    }}
                  >
                    Forgot your password?
                  </a>
                </div>
                <div className="text-sm text-gray-600">
                  Don't have an account?{' '}
                  <a
                    href="/register"
                    className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
                    onClick={(e) => {
                      e.preventDefault();
                      navigate('/register');
                    }}
                  >
                    Sign up
                  </a>
                </div>
              </div>
            </div>
          </form>
        </div>

        {/* Footer Note */}
        <div className="mt-6 space-y-2">
          <p className="text-center text-gray-500 text-xs">
            By signing in, you agree to our Terms of Service and Privacy Policy
          </p>
          <p className="text-center text-gray-500 text-xs">
            Protected by enterprise-grade security
          </p>
        </div>
      </div>
    </ErrorBoundary>
  );
}