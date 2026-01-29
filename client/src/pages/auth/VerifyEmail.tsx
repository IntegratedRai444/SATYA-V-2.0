import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle, Loader2, Mail, CheckCircle } from 'lucide-react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import { supabase } from '@/lib/supabaseSingleton';

export default function VerifyEmail() {
  const navigate = useNavigate();
  const { user, loading } = useSupabaseAuth();
  const [resending, setResending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // If user is already verified, redirect to dashboard
    if (user && user.email_confirmed_at) {
      navigate('/dashboard');
    }
  }, [user, navigate]);

  const handleResendVerification = async () => {
    try {
      setResending(true);
      setError(null);
      setMessage(null);

      // Resend verification email using Supabase
      const { error } = await supabase.auth.resend({
        type: 'signup',
        email: user?.email || '',
      });

      if (error) throw error;

      setMessage('Verification email sent! Please check your inbox.');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to resend verification email');
    } finally {
      setResending(false);
    }
  };

  const handleCheckVerification = async () => {
    try {
      // Refresh user session to check email verification status
      const { data: { user }, error } = await supabase.auth.getUser();
      
      if (error) throw error;
      
      if (user?.email_confirmed_at) {
        navigate('/dashboard');
      } else {
        setError('Email not yet verified. Please check your inbox and click the verification link.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to check verification status');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-4" />
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Background Effects */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-black/50"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 w-full max-w-md px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full mb-6 shadow-lg shadow-blue-500/25">
            <Mail className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">Verify Email</h1>
          <p className="text-blue-200 text-lg font-medium">SatyaAI Deepfake Detection</p>
        </div>

        {/* Glass Card */}
        <div className="backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-2xl shadow-black/20">
          <div className="text-center mb-6">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-yellow-500/20 rounded-full mb-4">
              <AlertTriangle className="w-8 h-8 text-yellow-400" />
            </div>
            <h2 className="text-2xl font-semibold text-white mb-2">Email Verification Required</h2>
            <p className="text-gray-300">
              We've sent a verification email to:<br />
              <span className="font-mono text-cyan-400">{user?.email}</span>
            </p>
          </div>

          {/* Instructions */}
          <div className="space-y-4 mb-6">
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
              <p className="text-gray-300 text-sm">Check your email inbox for the verification link</p>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
              <p className="text-gray-300 text-sm">Click the link to verify your account</p>
            </div>
            <div className="flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
              <p className="text-gray-300 text-sm">Return here and click "I've Verified" to continue</p>
            </div>
          </div>

          {/* Messages */}
          {message && (
            <div className="mb-6">
              <Alert className="bg-green-900/20 border-green-500/50">
                <CheckCircle className="h-4 w-4 mr-2 text-green-400" />
                <AlertDescription className="text-green-200">
                  {message}
                </AlertDescription>
              </Alert>
            </div>
          )}

          {error && (
            <div className="mb-6">
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4 mr-2" />
                <AlertDescription>
                  {error}
                </AlertDescription>
              </Alert>
            </div>
          )}

          {/* Actions */}
          <div className="space-y-4">
            <Button
              onClick={handleCheckVerification}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-full text-sm font-medium text-white bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:ring-offset-2 focus:ring-offset-transparent transition-all duration-200 shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:-translate-y-0.5"
            >
              I've Verified My Email
            </Button>

            <Button
              onClick={handleResendVerification}
              variant="outline"
              disabled={resending}
              className="w-full flex justify-center py-3 px-4 border border-cyan-500/50 rounded-full text-sm font-medium text-cyan-400 hover:bg-cyan-500/10 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 focus:ring-offset-2 focus:ring-offset-transparent transition-all duration-200"
            >
              {resending ? (
                <>
                  <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                  Resending...
                </>
              ) : (
                'Resend Verification Email'
              )}
            </Button>

            <div className="text-center">
              <button
                onClick={() => navigate('/login')}
                className="text-sm text-gray-400 hover:text-cyan-400 transition-colors"
              >
                Back to Login
              </button>
            </div>
          </div>
        </div>

        {/* Footer Note */}
        <div className="mt-6 text-center">
          <p className="text-gray-500 text-xs">
            Didn't receive the email? Check your spam folder or click "Resend Verification Email"
          </p>
        </div>
      </div>
    </div>
  );
}
