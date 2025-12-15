import React, { useState, useEffect } from 'react';
import { getAuthToken } from '@/services/auth';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Alert, AlertDescription } from '../components/ui/alert';
import { Loader2, Eye, EyeOff, Shield, Mail, Lock, User } from 'lucide-react';

export default function Login() {
  const navigate = useNavigate();
  const { login, error, isLoading, clearError } = useAuth();

  // Debug: Log auth state
  useEffect(() => {
    console.log('Login component mounted');
    console.log('Auth state:', { error, isLoading });

    // Check if already authenticated
    const checkAuth = async () => {
      try {
        const token = await getAuthToken().catch(() => null);
        console.log('Current auth token exists:', !!token);
      } catch (err) {
        console.error('Auth check error:', err);
      }
    };

    checkAuth();
  }, []);

  const [showPassword, setShowPassword] = useState(false);
  const [showRoleSelection, setShowRoleSelection] = useState(false);
  const [selectedRole, setSelectedRole] = useState<'user' | 'admin'>('admin');
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    clearError();
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Check if this is Rishabh Kapoor's account
    const isRishabhKapoor = formData.email === 'rishabhkapoor@atomicmail.io' ||
      formData.email === 'rishabhkapoor';

    if (isRishabhKapoor && !showRoleSelection) {
      // Show role selection for Rishabh Kapoor
      setShowRoleSelection(true);
      return;
    }

    // Use email as username for login
    const success = await login(formData.email, formData.password, isRishabhKapoor ? selectedRole : undefined);
    if (success) {
      // Add small delay to ensure state updates complete before navigation
      // This fixes the race condition where ProtectedRoute checks auth before user state is set
      setTimeout(() => {
        navigate('/dashboard');
      }, 100);
    }
  };

  const handleRoleSelect = (role: 'user' | 'admin') => {
    setSelectedRole(role);
  };

  // Add a simple error boundary wrapper
  const ErrorBoundary = ({ children }: { children: React.ReactNode }) => {
    const [hasError, setHasError] = useState(false);

    if (hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-red-900/10 p-4">
          <div className="text-center p-6 max-w-md bg-red-900/20 rounded-lg">
            <h2 className="text-xl font-bold text-red-400 mb-2">Something went wrong</h2>
            <p className="text-red-300 mb-4">Please refresh the page or try again later.</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return <>{children}</>;
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen flex items-center justify-center bg-[#0a0e1a] p-4">
        <div className="w-full max-w-md">
          {/* Logo and Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mb-4 shadow-lg shadow-blue-500/50">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-white mb-2">SatyaAI</h1>
            <p className="text-slate-400 text-sm">Cyber Intelligence Platform</p>
          </div>

          {/* Subtitle */}
          <p className="text-center text-slate-300 text-sm mb-8">
            Secure access to advanced cybersecurity intelligence and analysis tools
          </p>

          {/* Login Card */}
          <div className="bg-[#0f1420] border border-slate-800 rounded-2xl p-8 shadow-2xl">
            {/* Secure Login Header */}
            <div className="flex items-center justify-center gap-2 mb-8">
              <Lock className="w-5 h-5 text-slate-400" />
              <h2 className="text-xl font-semibold text-white">Secure Login</h2>
            </div>

            {error && (
              <Alert className="bg-red-900/20 border-red-800 text-red-400 mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {!showRoleSelection ? (
                <>
                  {/* Email Address */}
                  <div className="space-y-2">
                    <label htmlFor="email" className="text-sm font-medium text-slate-300 block">
                      Email Address
                    </label>
                    <div className="relative">
                      <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                      <Input
                        id="email"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleInputChange}
                        placeholder="Enter your email"
                        required
                        className="w-full bg-[#1a1f2e] border-slate-700 text-white placeholder:text-slate-500 pl-11 h-12 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                      />
                    </div>
                  </div>

                  {/* Password */}
                  <div className="space-y-2">
                    <label htmlFor="password" className="text-sm font-medium text-slate-300 block">
                      Password
                    </label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                      <Input
                        id="password"
                        name="password"
                        type={showPassword ? 'text' : 'password'}
                        value={formData.password}
                        onChange={handleInputChange}
                        placeholder="Enter your password"
                        required
                        className="w-full bg-[#1a1f2e] border-slate-700 text-white placeholder:text-slate-500 pl-11 pr-11 h-12 rounded-lg focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
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
                          }`}>Admin</p>
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
                          }`}>User</p>
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
              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-12 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white font-medium rounded-lg shadow-lg shadow-blue-500/30 transition-all duration-200"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Authenticating...
                  </>
                ) : showRoleSelection ? (
                  <>
                    <Shield className="mr-2 h-5 w-5" />
                    Login as {selectedRole === 'admin' ? 'Admin' : 'User'}
                  </>
                ) : (
                  <>
                    <Shield className="mr-2 h-5 w-5" />
                    Secure Login
                  </>
                )}
              </Button>
            </form>
          </div>

          {/* Footer Note */}
          <p className="text-center text-slate-500 text-xs mt-6">
            Protected by enterprise-grade security
          </p>
        </div>
      </div>
    </ErrorBoundary>
  );
}