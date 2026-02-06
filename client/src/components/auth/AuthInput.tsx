import React from 'react';
import { AlertTriangle, Eye, EyeOff, Mail, Key, User } from 'lucide-react';

interface AuthInputProps {
  label: string;
  name: string;
  type: string;
  placeholder: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  error?: string;
  required?: boolean;
  autoComplete?: string;
  icon?: 'email' | 'password' | 'name';
  showPasswordToggle?: boolean;
  showPassword?: boolean;
  onTogglePassword?: () => void;
  disabled?: boolean;
  inputRef?: React.RefObject<HTMLInputElement>;
}

export default function AuthInput({
  label,
  name,
  type,
  placeholder,
  value,
  onChange,
  error,
  required = false,
  autoComplete,
  icon = 'email',
  showPasswordToggle = false,
  showPassword = false,
  onTogglePassword,
  disabled = false,
  inputRef
}: AuthInputProps) {
  const getIcon = () => {
    switch (icon) {
      case 'email':
        return <Mail className="w-5 h-5 text-[#8fb3ff]" />;
      case 'password':
        return <Key className="w-5 h-5 text-[#8fb3ff]" />;
      case 'name':
        return <User className="w-5 h-5 text-[#8fb3ff]" />;
      default:
        return <Mail className="w-5 h-5 text-[#8fb3ff]" />;
    }
  };

  return (
    <div>
      <label className="block text-blue-200/90 text-sm font-medium mb-2">
        {label}
      </label>
      <div className="relative">
        <div className="absolute left-4 top-1/2 -translate-y-1/2">
          {getIcon()}
        </div>
        <input
          ref={inputRef}
          id={name}
          name={name}
          type={showPasswordToggle && showPassword ? 'text' : type}
          autoComplete={autoComplete}
          required={required}
          disabled={disabled}
          className={`w-full h-12 pl-[44px] ${showPasswordToggle ? 'pr-[44px]' : 'pr-4'} bg-[#0b1629] border border-[rgba(140,180,240,0.18)] rounded-[10px] text-[#eaf2ff] placeholder-[#7f97b8] focus:outline-none focus:border-[#4da3ff] transition-colors disabled:opacity-50 disabled:cursor-not-allowed`}
          placeholder={placeholder}
          value={value}
          onChange={onChange}
        />
        {showPasswordToggle && (
          <button
            type="button"
            className="absolute right-[11px] top-1/2 -translate-y-1/2 text-[#8fb3ff] hover:text-[#4da3ff] transition-colors disabled:opacity-50"
            onClick={onTogglePassword}
            disabled={disabled}
          >
            {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
          </button>
        )}
      </div>
      {error && (
        <p className="text-red-400 text-sm mt-2 flex items-center">
          <AlertTriangle className="w-4 h-4 mr-1" />
          {error}
        </p>
      )}
    </div>
  );
}
