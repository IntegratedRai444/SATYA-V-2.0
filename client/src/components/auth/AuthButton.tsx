import React from 'react';
import { Shield, Loader2 } from 'lucide-react';

interface AuthButtonProps {
  type?: 'submit' | 'button';
  children: React.ReactNode;
  loading?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  className?: string;
}

export default function AuthButton({
  type = 'submit',
  children,
  loading = false,
  disabled = false,
  onClick,
  className = ''
}: AuthButtonProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`w-full h-12 bg-gradient-to-r from-[#2563eb] to-[#06b6d4] hover:brightness-110 text-white font-semibold px-4 rounded-[10px] transition-all duration-200 disabled:opacity-50 disabled:brightness-100 ${className}`}
    >
      {loading ? (
        <>
          <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
          {children}
        </>
      ) : (
        <>
          <Shield className="w-4 h-4 mr-2" />
          {children}
        </>
      )}
    </button>
  );
}
