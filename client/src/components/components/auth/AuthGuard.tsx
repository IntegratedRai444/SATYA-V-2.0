import React, { useEffect, useState } from 'react';
import { useLocation } from 'wouter';
import { checkAuth } from '../../lib/auth';
import LoginForm from './LoginForm';

interface AuthGuardProps {
  children: React.ReactNode;
}

const AuthGuard: React.FC<AuthGuardProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [, setLocation] = useLocation();

  useEffect(() => {
    const verifyAuth = async () => {
      try {
        const { isAuthenticated } = await checkAuth();
        setIsAuthenticated(isAuthenticated);
        if (!isAuthenticated) {
          setLocation('/login');
        } else if (window.location.pathname === '/login') {
          setLocation('/dashboard');
        }
      } catch (error) {
        console.error('Auth verification failed:', error);
        setIsAuthenticated(false);
        setLocation('/login');
      }
    };
    verifyAuth();
  }, [setLocation]);

  if (isAuthenticated === null) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-lg font-medium">Verifying authentication...</p>
        </div>
      </div>
    );
  }

  // If not authenticated, block rendering children (redirect handled above)
  if (!isAuthenticated) {
    return null;
  }

  // If authenticated, show children
  return <>{children}</>;
};

export default AuthGuard;