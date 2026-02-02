import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield, AlertTriangle, Loader2, Mail, Key, Sparkles, Zap } from 'lucide-react';
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
      // Navigation will be handled by the useEffect hook when auth state changes
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
      <div className="min-h-screen relative overflow-hidden">
        {/* Animated gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-purple-600 via-pink-500 to-orange-400 animate-gradient-shift">
          <div className="absolute inset-0 bg-gradient-to-tr from-blue-600/20 via-transparent to-purple-600/20"></div>
          <div className="absolute inset-0 bg-gradient-to-bl from-green-400/10 via-transparent to-yellow-400/10"></div>
        </div>
        
        {/* Floating decorative elements */}
        <div className="absolute top-20 left-20 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
        <div className="absolute top-40 right-20 w-72 h-72 bg-yellow-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-40 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
        
        {/* Main content */}
        <div className="relative z-10 min-h-screen flex items-center justify-center px-4">
          <div className="w-full max-w-md">
            {/* Logo and branding */}
            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-white/20 backdrop-blur-md rounded-2xl mb-4 border border-white/30 shadow-2xl">
                <Shield className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">SatyaAI</h1>
              <p className="text-white/80 text-lg font-medium">Deepfake Detection Platform</p>
            </div>

            {/* Login form glass card */}
            <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-8 shadow-2xl">
              <div className="flex items-center justify-center mb-6">
                <Sparkles className="w-5 h-5 text-yellow-300 mr-2" />
                <h2 className="text-2xl font-bold text-white">Welcome Back</h2>
                <Zap className="w-5 h-5 text-blue-300 ml-2" />
              </div>

              {renderError()}

              <form onSubmit={handleSubmit} className="space-y-5">
                {/* Email field */}
                <div>
                  <label className="block text-white/90 text-sm font-medium mb-2">Email Address</label>
                  <div className="relative">
                    <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/60" />
                    <input
                      id="email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      required
                      className="w-full pl-12 pr-4 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
                      placeholder="Enter your email"
                      value={formData.email}
                      onChange={handleInputChange}
                      ref={emailInputRef}
                    />
                  </div>
                  {formErrors.email && (
                    <p className="text-red-300 text-sm mt-2 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-1" />
                      {formErrors.email}
                    </p>
                  )}
                </div>

                {/* Password field */}
                <div>
                  <label className="block text-white/90 text-sm font-medium mb-2">Password</label>
                  <div className="relative">
                    <Key className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/60" />
                    <input
                      id="password"
                      name="password"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="current-password"
                      required
                      className="w-full pl-12 pr-12 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-white/30 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
                      placeholder="Enter your password"
                      value={formData.password}
                      onChange={handleInputChange}
                      ref={passwordInputRef}
                    />
                    <button
                      type="button"
                      className="absolute right-4 top-1/2 -translate-y-1/2 text-white/60 hover:text-white transition-colors"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                  {formErrors.password && (
                    <p className="text-red-300 text-sm mt-2 flex items-center">
                      <AlertTriangle className="w-4 h-4 mr-1" />
                      {formErrors.password}
                    </p>
                  )}
                </div>

                {/* Submit button */}
                <Button
                  type="submit"
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-4 px-6 rounded-xl transition-all duration-300 transform hover:scale-[1.02] shadow-lg disabled:opacity-50 disabled:transform-none border border-white/20"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5" />
                      Authenticating...
                    </>
                  ) : (
                    <>
                      <Shield className="w-5 h-5 mr-2" />
                      Secure Login
                    </>
                  )}
                </Button>

                {/* Links */}
                <div className="text-center space-y-3 pt-4">
                  <button
                    type="button"
                    className="text-white/80 hover:text-white transition-colors text-sm font-medium"
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
                      } catch (err) {
                        console.warn('Password reset error:', err);
                        setFormErrors({
                          ...formErrors,
                          general: 'Failed to send reset email. Please try again.'
                        });
                      }
                    }}
                  >
                    Forgot your password?
                  </button>
                  <div className="text-white/60 text-sm">
                    Don't have an account?{' '}
                    <a
                      href="/register"
                      className="text-white hover:text-white/80 transition-colors font-medium"
                      onClick={(e) => {
                        e.preventDefault();
                        navigate('/register');
                      }}
                    >
                      Sign up
                    </a>
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}