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
<<<<<<< HEAD
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

=======
      } catch (error) {
        console.error('Auth verification failed:', error);
        setIsAuthenticated(false);
      }
    };

    verifyAuth();
  }, []);

  // Show loading while checking authentication
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
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

<<<<<<< HEAD
  // If not authenticated, block rendering children (redirect handled above)
  if (!isAuthenticated) {
    return null;
=======
  // If not authenticated, show login form
  if (!isAuthenticated) {
    return (
      <div className="container mx-auto py-8 px-4">
        <div className="max-w-md mx-auto">
          <LoginForm onLoginSuccess={() => setIsAuthenticated(true)} />
        </div>
      </div>
    );
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
  }

  // If authenticated, show children
  return <>{children}</>;
};

export default AuthGuard;