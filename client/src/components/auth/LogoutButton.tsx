import React from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut } from 'lucide-react';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';

interface LogoutButtonProps {
  className?: string;
  variant?: 'button' | 'dropdown';
}

export const LogoutButton: React.FC<LogoutButtonProps> = ({ 
  className = '', 
  variant = 'button' 
}) => {
  const navigate = useNavigate();
  const { signOut, loading } = useSupabaseAuth();

  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      console.error('Logout failed:', error);
      // Even if logout fails, redirect to login
      navigate('/login');
    }
  };

  if (variant === 'dropdown') {
    return (
      <button
        onClick={handleLogout}
        disabled={loading}
        className={`w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900 flex items-center gap-2 ${className}`}
      >
        <LogOut className="w-4 h-4" />
        Sign out
      </button>
    );
  }

  return (
    <button
      onClick={handleLogout}
      disabled={loading}
      className={`inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-red-600 hover:text-red-700 hover:bg-red-50 rounded-md transition-colors ${className}`}
    >
      <LogOut className="w-4 h-4" />
      {loading ? 'Signing out...' : 'Sign out'}
    </button>
  );
};

export default LogoutButton;
