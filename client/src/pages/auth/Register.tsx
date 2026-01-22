import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/SupabaseAuthProvider';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield, AlertTriangle, Loader2, UserPlus } from 'lucide-react';

export default function Register() {
  const navigate = useNavigate();
  const { register, error, isLoading, clearError, isAuthenticated } = useAuth();

  useEffect(() => {
    console.log('Register component mounted');
    console.log('Auth state:', { error, isLoading });

    // Redirect if already authenticated
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate, error, isLoading]);

  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [formErrors, setFormErrors] = useState<{
    name?: string;
    email?: string;
    password?: string;
    confirmPassword?: string;
    general?: string;
  }>({});
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });

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

  const validateForm = () => {
    const errors: typeof formErrors = {};

    if (!formData.name.trim()) {
      errors.name = 'Name is required';
    } else if (formData.name.trim().length < 2) {
      errors.name = 'Name must be at least 2 characters';
    }

    if (!formData.email) {
      errors.email = 'Email is required';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      errors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      errors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      errors.password = 'Password must be at least 8 characters';
    } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
      errors.password = 'Password must contain at least one uppercase letter, one lowercase letter, and one number';
    }

    if (!formData.confirmPassword) {
      errors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      errors.confirmPassword = 'Passwords do not match';
    }

    return errors;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError && clearError();

    const errors = validateForm();
    if (Object.keys(errors).length > 0) {
      setFormErrors(errors);
      return;
    }

    try {
      setNetworkError(null);
      setFormErrors({});

      console.log('Attempting registration with:', {
        username: formData.name,
        email: formData.email,
        passwordLength: formData.password.length
      });

      if (register) {
        const result = await register({
          username: formData.name,
          email: formData.email,
          password: formData.password,
          fullName: formData.name
        });

        console.log('Registration result:', result);

        if (result.user) {
          navigate('/dashboard');
        } else {
          throw new Error('Registration failed - no user returned');
        }
      }
    } catch (err: unknown) {
      console.error('Registration error details:', {
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
          case 409:
            setFormErrors({ general: 'An account with this email already exists' });
            break;
          case 400:
            setFormErrors({ general: 'Invalid registration data. Please check your input.' });
            break;
          case 500:
            setFormErrors({ general: 'Server error. Please try again later.' });
            break;
          default:
            setFormErrors({ general: `Registration error (${errorObj.response.status}). Please try again.` });
        }
      } else if (errorObj.message) {
        setFormErrors({ general: errorObj.message });
      } else {
        setFormErrors({ general: 'An unexpected error occurred during registration' });
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
    const errorMessage = networkError || formErrors.general || (error ? 'An error occurred during registration' : '');

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
                  <Shield className="w-12 h-12 text-blue-500" />
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
          Create your account to access advanced deepfake detection and analysis tools
        </p>

        {/* Registration Card */}
        <div className="bg-white border border-gray-200 rounded-2xl p-8 shadow-xl w-full max-w-md">
          {/* Create Account Header */}
          <div className="flex items-center justify-center gap-2 mb-8">
            <UserPlus className="w-5 h-5 text-blue-500" />
            <h2 className="text-2xl font-semibold text-gray-800">Create Account</h2>
          </div>

          {renderError()}

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Full Name */}
            <div className="space-y-2">
              <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                Full name
                {formErrors.name && (
                  <span className="text-red-500 text-xs ml-2">{formErrors.name}</span>
                )}
              </label>
              <div className="mt-1">
                <Input
                  id="name"
                  name="name"
                  type="text"
                  autoComplete="name"
                  required
                  value={formData.name}
                  onChange={handleInputChange}
                  className={`w-full px-4 py-3 border ${formErrors.name ? 'border-red-300' : 'border-gray-300'
                    } rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors`}
                  placeholder="Enter your full name"
                  aria-invalid={!!formErrors.name}
                  aria-describedby={formErrors.name ? 'name-error' : undefined}
                />
              </div>
            </div>

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
                  autoComplete="new-password"
                  required
                  value={formData.password}
                  onChange={handleInputChange}
                  className={`w-full px-4 py-3 border ${formErrors.password ? 'border-red-300' : 'border-gray-300'
                    } rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors`}
                  placeholder="Create a strong password"
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
              <p className="text-xs text-gray-500">
                Must contain at least 8 characters, including uppercase, lowercase, and numbers
              </p>
            </div>

            {/* Confirm Password */}
            <div className="space-y-2">
              <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                Confirm password
                {formErrors.confirmPassword && (
                  <span className="text-red-500 text-xs ml-2">{formErrors.confirmPassword}</span>
                )}
              </label>
              <div className="mt-1 relative">
                <Input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  required
                  value={formData.confirmPassword}
                  onChange={handleInputChange}
                  className={`w-full px-4 py-3 border ${formErrors.confirmPassword ? 'border-red-300' : 'border-gray-300'
                    } rounded-lg shadow-sm placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors`}
                  placeholder="Confirm your password"
                  aria-invalid={!!formErrors.confirmPassword}
                  aria-describedby={formErrors.confirmPassword ? 'confirmPassword-error' : undefined}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-blue-500 transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Register Button */}
            <div className="space-y-4">
              <Button
                type="submit"
                className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-75 transition-all duration-200"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" />
                    Creating account...
                  </>
                ) : (
                  'Create Account'
                )}
              </Button>

              <div className="text-center space-y-2">
                <div className="text-sm text-gray-600">
                  Already have an account?{' '}
                  <a
                    href="/login"
                    className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
                    onClick={(e) => {
                      e.preventDefault();
                      navigate('/login');
                    }}
                  >
                    Sign in
                  </a>
                </div>
              </div>
            </div>
          </form>
        </div>

        {/* Footer Note */}
        <div className="mt-6 space-y-2">
          <p className="text-center text-gray-500 text-xs">
            By creating an account, you agree to our Terms of Service and Privacy Policy
          </p>
          <p className="text-center text-gray-500 text-xs">
            Protected by enterprise-grade security
          </p>
        </div>
      </div>
    </ErrorBoundary>
  );
}