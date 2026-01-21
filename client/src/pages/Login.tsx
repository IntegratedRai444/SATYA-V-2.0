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
  }, [isAuthenticated, navigate]);

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
    } catch (err: any) {
      console.error('Login error details:', {
        message: err.message,
        response: err.response,
        status: err.response?.status,
        stack: err.stack
      });

      if (err.message === 'Network Error') {
        setNetworkError('Unable to connect to server. Please check your internet connection and ensure the server is running on port 5001.');
      } else if (err.response) {
        switch (err.response.status) {
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
            setFormErrors({ ...formErrors, general: `Server error (${err.response.status}). Please try again.` });
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
      <div className="min-h-screen flex flex-col items-center justify-center bg-[#0f0f23] p-4">
        <div className="w-full max-w-md mb-8">
          {/* Logo and Header */}
          <div className="text-center">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mb-4 shadow-lg shadow-blue-500/50">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white mb-2">Satya</h1>
            <p className="text-slate-400 text-sm">Deepfake Detection System</p>
          </div>
        </div>

        {/* Subtitle */}
        <p className="text-center text-slate-300 text-sm mb-8">
          Secure access to advanced deepfake detection and analysis tools
        </p>

        {/* Login Card */}
        <div className="bg-[#1a1f2e] border border-slate-800 rounded-2xl p-8 shadow-2xl w-full max-w-md">
          {/* Secure Login Header */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <Lock className="w-5 h-5 text-slate-400" />
            <h2 className="text-xl font-semibold text-white">Secure Login</h2>
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
                      className={`appearance-none block w-full px-3 py-2 border ${formErrors.email ? 'border-red-300' : 'border-gray-300'
                        } rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm`}
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
                      className={`appearance-none block w-full px-3 py-2 border ${formErrors.password ? 'border-red-300' : 'border-gray-300'
                        } rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm`}
                      aria-invalid={!!formErrors.password}
                      aria-describedby={formErrors.password ? 'password-error' : undefined}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
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
                    <p className="text-slate-300 text-sm mb-2">Welcome, Rishabh Kapoor!</p>
                    <p className="text-slate-400 text-xs">Select your login role:</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <button
                      type="button"
                      onClick={() => handleRoleSelect('admin')}
                      className={`p-4 rounded-lg border-2 transition-all ${selectedRole === 'admin'
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-slate-700 bg-[#1a1f2e] hover:border-slate-600'
                        }`}
                    >
                      <Shield className={`w-8 h-8 mx-auto mb-2 ${selectedRole === 'admin' ? 'text-blue-400' : 'text-slate-400'
                        }`} />
                      <p className={`font-medium ${selectedRole === 'admin' ? 'text-blue-400' : 'text-slate-300'
                        }`}>
                        Admin
                      </p>
                      <p className="text-xs text-slate-500 mt-1">Full Access</p>
                    </button>

                    <button
                      type="button"
                      onClick={() => handleRoleSelect('user')}
                      className={`p-4 rounded-lg border-2 transition-all ${selectedRole === 'user'
                          ? 'border-cyan-500 bg-cyan-500/10'
                          : 'border-slate-700 bg-[#1a1f2e] hover:border-slate-600'
                        }`}
                    >
                      <User className={`w-8 h-8 mx-auto mb-2 ${selectedRole === 'user' ? 'text-cyan-400' : 'text-slate-400'
                        }`} />
                      <p className={`font-medium ${selectedRole === 'user' ? 'text-cyan-400' : 'text-slate-300'
                        }`}>
                        User
                      </p>
                      <p className="text-xs text-slate-500 mt-1">Standard Access</p>
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => setShowRoleSelection(false)}
                    className="w-full text-sm text-slate-400 hover:text-slate-300 transition-colors mt-2"
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
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-75"
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
                <div className="text-sm text-gray-500">
                  <a
                    href="/forgot-password"
                    className="font-medium text-blue-600 hover:text-blue-500"
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
                <div className="text-sm text-gray-500">
                  Don't have an account?{' '}
                  <a
                    href="/register"
                    className="font-medium text-blue-600 hover:text-blue-500"
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
          <p className="text-center text-slate-500 text-xs">
            By signing in, you agree to our Terms of Service and Privacy Policy
          </p>
          <p className="text-center text-slate-500 text-xs">
            Protected by enterprise-grade security
          </p>
        </div>
      </div>
    </ErrorBoundary>
  );
}