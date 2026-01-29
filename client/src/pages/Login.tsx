import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield, AlertTriangle, Lock, Loader2 } from 'lucide-react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';

export default function Login() {
  const navigate = useNavigate();
  const { signIn, user, loading, error } = useSupabaseAuth();

  // ... rest of the code remains the same ...
  useEffect(() => {
    console.log('Login component mounted');
    console.log('Auth state:', { error, loading, user });

    // Only redirect if auth is NOT loading AND user is authenticated
    if (!loading && user) {
      navigate('/dashboard');
    }
  }, [user, loading, navigate, error]);

  const [showPassword, setShowPassword] = useState(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [formErrors, setFormErrors] = useState<{ email?: string; password?: string; general?: string }>({});
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  // Add refs for input fields to fix one-character input issue
  const emailInputRef = useRef<HTMLInputElement>(null);
  const passwordInputRef = useRef<HTMLInputElement | null>(null);

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

      // Use real authentication
      await signIn(formData.email, formData.password);
      navigate('/dashboard');
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
      <div className="min-h-screen flex items-center justify-center relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        {/* Background Effects */}
        <div className="absolute inset-0">
          {/* Radial gradient highlights */}
          <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl"></div>
          <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
          
          {/* Vignette effect */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/50"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 w-full max-w-md px-4">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full mb-6 shadow-lg shadow-blue-500/25">
              <Shield className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">SatyaAI</h1>
            <p className="text-blue-200 text-lg font-medium">Deepfake Detection System</p>
            <p className="text-gray-400 text-sm mt-2 max-w-sm mx-auto">
              Secure access to AI-powered deepfake analysis and media verification tools
            </p>
          </div>

          {/* Glass Card */}
          <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-2xl shadow-black/20">
            {/* Secure Login Header */}
            <div className="flex items-center justify-center gap-2 mb-8">
              <Lock className="w-5 h-5 text-cyan-400" />
              <h2 className="text-2xl font-semibold text-white">Secure Login</h2>
            </div>

            {renderError()}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Email Address */}
              <div className="space-y-2">
                <label htmlFor="email" className="block text-sm font-medium text-gray-300">
                  Email Address
                </label>
                <div className="relative">
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    className="w-full pl-11 pr-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:border-cyan-400/50 transition-all duration-200"
                    placeholder="Enter your email"
                    value={formData.email}
                    onChange={handleInputChange}
                    ref={emailInputRef}
                  />
                </div>
                {formErrors.email && (
                  <p className="text-red-400 text-sm mt-1">{formErrors.email}</p>
                )}
              </div>

              {/* Password */}
              <div className="space-y-2">
                <label htmlFor="password" className="block text-sm font-medium text-gray-300">
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="current-password"
                    required
                    className="w-full pl-11 pr-12 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:border-cyan-400/50 transition-all duration-200"
                    placeholder="Enter your password"
                    value={formData.password}
                    onChange={handleInputChange}
                    ref={passwordInputRef}
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? <EyeOff className="w-5 h-5 text-gray-400" /> : <Eye className="w-5 h-5 text-gray-400" />}
                  </button>
                </div>
                {formErrors.password && (
                  <p className="text-red-400 text-sm mt-1">{formErrors.password}</p>
                )}
              </div>

              <div className="space-y-4">
                <Button
                  type="submit"
                  className="w-full flex justify-center py-3 px-4 border border-transparent rounded-full text-sm font-medium text-white bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:ring-offset-2 focus:ring-offset-transparent disabled:opacity-75 transition-all duration-200 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:-translate-y-0.5"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" />
                      Signing in...
                    </>
                  ) : (
                    'Secure Login'
                  )}
                </Button>

                <div className="text-center space-y-2">
                  <div className="text-sm text-gray-400">
                    <button
                      type="button"
                      className="font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                      onClick={async (e) => {
                        e.preventDefault();
                        const email = formData.email;
                        if (!email) {
                          setFormErrors({
                            ...formErrors,
                            general: 'Please enter your email address first, then click "Forgot Password".'
                          });
                          return;
                        }
                        
                        try {
                          setFormErrors({ ...formErrors, general: 'Sending reset email...' });
                          
                          const { supabase } = await import('@/lib/supabaseSingleton');
                          const { error } = await supabase.auth.resetPasswordForEmail(email, {
                            redirectTo: `${window.location.origin}/reset-password`,
                          });
                          
                          if (error) {
                            setFormErrors({
                              ...formErrors,
                              general: `Failed to send reset email: ${error.message}`
                            });
                          } else {
                            setFormErrors({
                              ...formErrors,
                              general: 'Password reset email sent! Check your inbox.'
                            });
                          }
                        } catch (error) {
                          setFormErrors({
                            ...formErrors,
                            general: 'Failed to send reset email. Please try again.'
                          });
                        }
                      }}
                    >
                      Forgot your password?
                    </button>
                  </div>
                  <div className="text-sm text-gray-400">
                    Don't have an account?{' '}
                    <a
                      href="/register"
                      className="font-medium text-cyan-400 hover:text-cyan-300 transition-colors"
                      onClick={(e) => {
                        e.preventDefault();
                        navigate('/register');
                      }}
                    >
                      Register
                    </a>
                  </div>
                </div>
              </div>
            </form>

            <p className="text-center text-gray-500 text-xs mt-6">
              By signing in, you agree to our Terms of Service and Privacy Policy
            </p>
            <p className="text-center text-gray-500 text-xs">
              Protected by enterprise-grade security
            </p>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}