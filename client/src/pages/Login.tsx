import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import AuthLayout from '@/components/auth/AuthLayout';
import AuthInput from '@/components/auth/AuthInput';
import AuthButton from '@/components/auth/AuthButton';

export default function Login() {
  const navigate = useNavigate();
  const { signIn, user, loading, error } = useSupabaseAuth();

  useEffect(() => {
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

      // Use real authentication
      await signIn(formData.email, formData.password);
      // Navigation will be handled by the useEffect hook when auth state changes
    } catch (err: unknown) {

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

  const renderError = () => {
    const errorMessage = networkError || formErrors.general || (error ? 'An error occurred during login' : '');

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
      title="Secure Login"
      subtitle="Secure access to advanced deepfake detection and media analysis tools"
    >
      {renderError()}

      <form onSubmit={handleSubmit} className="space-y-6">
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
          inputRef={emailInputRef}
        />

        <AuthInput
          label="Password"
          name="password"
          type="password"
          placeholder="Enter your password"
          value={formData.password}
          onChange={handleInputChange}
          error={formErrors.password}
          autoComplete="current-password"
          icon="password"
          showPasswordToggle
          showPassword={showPassword}
          onTogglePassword={() => setShowPassword(!showPassword)}
          inputRef={passwordInputRef}
        />

        <AuthButton loading={loading}>
          {loading ? 'Authenticating...' : 'Secure Login'}
        </AuthButton>

        <div className="text-center space-y-3 pt-4">
          <button
            type="button"
            className="text-blue-300/80 hover:text-blue-200 transition-colors text-sm font-medium hover:underline"
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
              } catch {
                setFormErrors({
                  ...formErrors,
                  general: 'Failed to send reset email. Please try again.'
                });
              }
            }}
          >
            Forgot your password?
          </button>
          <div className="text-blue-400/60 text-sm">
            Don't have an account?{' '}
            <button
              type="button"
              className="text-blue-300 hover:text-blue-200 transition-colors font-medium hover:underline"
              onClick={(e) => {
                e.preventDefault();
                navigate('/register');
              }}
            >
              Sign up
            </button>
          </div>
        </div>
      </form>
    </AuthLayout>
  );
}