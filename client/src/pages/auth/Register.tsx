import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield, AlertTriangle, Loader2, User, Lock, Mail } from 'lucide-react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';

export default function Register() {
  const navigate = useNavigate();
  const { signUp } = useSupabaseAuth();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [registrationSuccess, setRegistrationSuccess] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    
    // Clear error for this field when user starts typing
    if (formErrors[name]) {
      setFormErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const validateForm = () => {
    const errors: Record<string, string> = {};

    if (!formData.name.trim()) {
      errors.name = 'Name is required';
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

    const errors = validateForm();
    if (Object.keys(errors).length > 0) {
      setFormErrors(errors);
      return;
    }

    try {
      setNetworkError(null);
      setFormErrors({});
      setLoading(true);

      console.log('Attempting registration with:', {
        username: formData.name,
        email: formData.email,
        passwordLength: formData.password.length
      });

      // Call real Supabase registration
      await signUp(formData.email, formData.password, { 
        name: formData.name,
        full_name: formData.name
      });
      
      // Show success message instead of redirecting
      setRegistrationSuccess(true);
      console.log('Registration successful - email verification required');
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
    } finally {
      setLoading(false);
    }
  };

  const renderError = () => {
    const errorMessage = networkError || formErrors.general;

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
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Content */}
      <div className="w-full max-w-md px-4">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-full mb-4">
            <Shield className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white mb-1">SatyaAI</h1>
          <p className="text-blue-300 text-sm">Cyber Intelligence Platform</p>
        </div>

        {/* Register Form */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white text-center mb-6">Secure Registration</h2>

          {renderError()}

          {/* Success Message */}
          {registrationSuccess && (
            <div className="mb-6">
              <Alert className="bg-green-900/20 border-green-500/50">
                <Shield className="h-4 w-4 mr-2 text-green-400" />
                <AlertDescription className="text-green-200">
                  Registration successful! Please check your email to verify your account before logging in.
                </AlertDescription>
              </Alert>
            </div>
          )}

          {!registrationSuccess && (
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Full Name */}
              <div>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    id="name"
                    name="name"
                    type="text"
                    autoComplete="name"
                    required
                    value={formData.name}
                    onChange={handleInputChange}
                    className="w-full pl-10 pr-4 py-2 bg-gray-700/50 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    placeholder="Full Name"
                  />
                </div>
                {formErrors.name && (
                  <p className="text-red-400 text-xs mt-1">{formErrors.name}</p>
                )}
              </div>

              {/* Email */}
              <div>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={formData.email}
                    onChange={handleInputChange}
                    className="w-full pl-10 pr-4 py-2 bg-gray-700/50 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    placeholder="Email Address"
                  />
                </div>
                {formErrors.email && (
                  <p className="text-red-400 text-xs mt-1">{formErrors.email}</p>
                )}
              </div>

              {/* Password */}
              <div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    id="password"
                    name="password"
                    type={showPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    value={formData.password}
                    onChange={handleInputChange}
                    className="w-full pl-10 pr-10 py-2 bg-gray-700/50 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    placeholder="Password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                {formErrors.password && (
                  <p className="text-red-400 text-xs mt-1">{formErrors.password}</p>
                )}
                <p className="text-xs text-gray-500 mt-1">
                  Must contain at least 8 characters, including uppercase, lowercase, and numbers
                </p>
              </div>

              {/* Confirm Password */}
              <div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <Input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    className="w-full pl-10 pr-10 py-2 bg-gray-700/50 border border-gray-600 rounded text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                    placeholder="Confirm Password"
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                  >
                    {showConfirmPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                {formErrors.confirmPassword && (
                  <p className="text-red-400 text-xs mt-1">{formErrors.confirmPassword}</p>
                )}
              </div>

              {/* Register Button */}
              <Button
                type="submit"
                className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded transition-colors disabled:opacity-50"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                    Creating account...
                  </>
                ) : (
                  'Create Account'
                )}
              </Button>

              {/* Links */}
              <div className="text-center space-y-2 pt-4">
                <div className="text-sm text-gray-400">
                  Already have an account?{' '}
                  <a
                    href="/dashboard"
                    className="text-blue-400 hover:text-blue-300 transition-colors"
                    onClick={(e) => {
                      e.preventDefault();
                      navigate('/dashboard');
                    }}
                  >
                    Go to Dashboard
                  </a>
                </div>
              </div>
            </form>
          )}

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
      </div>
    </div>
  );
}
