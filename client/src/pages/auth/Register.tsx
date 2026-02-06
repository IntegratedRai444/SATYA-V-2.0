import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle, Shield } from 'lucide-react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import AuthLayout from '@/components/auth/AuthLayout';
import AuthInput from '@/components/auth/AuthInput';
import AuthButton from '@/components/auth/AuthButton';

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

      if (import.meta.env.DEV) {
        console.log('Attempting registration with:', {
          username: formData.name,
          email: formData.email,
          passwordLength: formData.password.length
        });
      }

      // Call real Supabase registration
      await signUp(formData.email, formData.password, { 
        name: formData.name,
        full_name: formData.name
      });
      
      // Show success message instead of redirecting
      setRegistrationSuccess(true);
      if (import.meta.env.DEV) {
        console.log('Registration successful - email verification required');
      }
    } catch (err: unknown) {
      if (import.meta.env.DEV) {
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
  } finally {
    setLoading(false);
  }
  };

  const renderError = () => {
    const errorMessage = networkError || formErrors.general;

    if (!errorMessage) return null;

    return (
      <div className="mb-6 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
        <div className="flex items-center text-red-400">
          <AlertTriangle className="w-4 h-4 mr-2 flex-shrink-0" />
          <span className="text-sm">{errorMessage}</span>
        </div>
      </div>
    );
  };

  return (
    <AuthLayout 
      title="Create Account"
      subtitle="Secure access to advanced deepfake detection and media analysis tools"
    >
      {renderError()}

      {/* Success Message */}
      {registrationSuccess && (
        <div className="mb-6 p-3 bg-green-900/20 border border-green-500/30 rounded-lg">
          <div className="flex items-center text-green-400">
            <Shield className="w-4 h-4 mr-2 flex-shrink-0" />
            <span className="text-sm">Registration successful! Please check your email to verify your account before logging in.</span>
          </div>
        </div>
      )}

      {!registrationSuccess && (
        <form onSubmit={handleSubmit} className="space-y-6">
          <AuthInput
            label="Full Name"
            name="name"
            type="text"
            placeholder="Enter your full name"
            value={formData.name}
            onChange={handleInputChange}
            error={formErrors.name}
            autoComplete="name"
            icon="name"
          />

          <AuthInput
            label="Email Address"
            name="email"
            type="email"
            placeholder="Enter your email"
            value={formData.email}
            onChange={handleInputChange}
            error={formErrors.email}
            autoComplete="email"
            icon="email"
          />

          <AuthInput
            label="Password"
            name="password"
            type="password"
            placeholder="Create a strong password"
            value={formData.password}
            onChange={handleInputChange}
            error={formErrors.password}
            autoComplete="new-password"
            icon="password"
            showPasswordToggle
            showPassword={showPassword}
            onTogglePassword={() => setShowPassword(!showPassword)}
          />
          <p className="text-xs text-gray-500 -mt-4 mb-2">
            Must contain at least 8 characters, including uppercase, lowercase, and numbers
          </p>

          <AuthInput
            label="Confirm Password"
            name="confirmPassword"
            type="password"
            placeholder="Confirm your password"
            value={formData.confirmPassword}
            onChange={handleInputChange}
            error={formErrors.confirmPassword}
            autoComplete="new-password"
            icon="password"
            showPasswordToggle
            showPassword={showConfirmPassword}
            onTogglePassword={() => setShowConfirmPassword(!showConfirmPassword)}
          />

          <AuthButton loading={loading}>
            {loading ? 'Creating account...' : 'Create Account'}
          </AuthButton>

          <div className="text-center space-y-3 pt-4">
            <div className="text-blue-400/60 text-sm">
              Already have an account?{' '}
              <button
                type="button"
                className="text-blue-300 hover:text-blue-200 transition-colors font-medium hover:underline"
                onClick={(e) => {
                  e.preventDefault();
                  navigate('/login');
                }}
              >
                Login
              </button>
            </div>
          </div>
        </form>
      )}

      {/* Footer Note */}
      {!registrationSuccess && (
        <div className="mt-6 space-y-2">
          <p className="text-center text-gray-500/80 text-xs">
            By creating an account, you agree to our Terms of Service and Privacy Policy
          </p>
          <p className="text-center text-gray-500/80 text-xs">
            Protected by enterprise-grade security
          </p>
        </div>
      )}
    </AuthLayout>
  );
}
