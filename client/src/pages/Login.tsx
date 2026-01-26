import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield, AlertTriangle, User, Lock, Loader2 } from 'lucide-react';
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
  const [showRoleSelection, setShowRoleSelection] = useState(false);
  const [selectedRole, setSelectedRole] = useState<'user' | 'admin'>('admin');
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

  const handleRoleSelect = (role: 'user' | 'admin') => {
    setSelectedRole(role);
  };

  const handleSocialLogin = async (provider: 'google' | 'github') => {
    try {
      setNetworkError(null);
      setFormErrors({});
      
      const { supabase } = await import('@/lib/supabaseSingleton');
      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/dashboard`
        }
      });
      
      if (error) {
        setFormErrors({ general: `Failed to login with ${provider}: ${error.message}` });
      }
    } catch (error) {
      setFormErrors({ general: `Failed to login with ${provider}. Please try again.` });
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
              {!showRoleSelection ? (
                <>
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
                </>
              ) : (
                <>
                  {/* Role Selection for Rishabh Kapoor */}
                  <div className="space-y-4">
                    <div className="text-center mb-4">
                      <p className="text-gray-400 text-sm mb-2">Welcome, Rishabh Kapoor!</p>
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

                {/* Social Login Divider */}
                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <div className="w-full border-t border-gray-600"></div>
                  </div>
                  <div className="relative flex justify-center text-sm">
                    <span className="px-2 bg-transparent text-gray-400">Or continue with</span>
                  </div>
                </div>

                {/* Social Login Buttons */}
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => handleSocialLogin('google')}
                    className="flex items-center justify-center gap-2 px-4 py-2 border border-gray-600 rounded-lg text-sm font-medium text-gray-300 hover:bg-gray-800 hover:border-gray-500 transition-colors"
                  >
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
                      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-5.37H12V23z" fill="#34A853"/>
                      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09L3.15 9.95c-.69 1.45-1.08 3.09-1.08 4.81s.39 3.36 1.08 4.81l2.69-2.05z" fill="#FBBC05"/>
                      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.98 1 5.57 2.09 4.09 3.36l2.69 2.69C7.46 6.21 8.89 5.38 12 5.38z" fill="#EA4335"/>
                    </svg>
                    Google
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSocialLogin('github')}
                    className="flex items-center justify-center gap-2 px-4 py-2 border border-gray-600 rounded-lg text-sm font-medium text-gray-300 hover:bg-gray-800 hover:border-gray-500 transition-colors"
                  >
                    <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957.266 1.583.322 2.54.322 2.54 0 1.583-.056 2.54-.322 2.293-.908 3.301-1.23 3.301-1.23.652 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                    </svg>
                    GitHub
                  </button>
                </div>

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
      </div>
    </ErrorBoundary>
  );
}