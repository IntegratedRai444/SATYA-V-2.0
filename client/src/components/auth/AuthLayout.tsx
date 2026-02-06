import React from 'react';
import { Shield } from 'lucide-react';

interface AuthLayoutProps {
  children: React.ReactNode;
  title: string;
  subtitle?: string;
}

export default function AuthLayout({ children, title, subtitle }: AuthLayoutProps) {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Background gradient: #050b16 → #0a1628 → #020712 */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#050b16] via-[#0a1628] to-[#020712]"></div>
      
      {/* Main content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center px-4">
        <div className="w-full max-w-[420px]">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-10 h-10 mb-4">
              <Shield className="w-10 h-10 text-[#8fb3ff]" />
            </div>
            <h1 className="text-[32px] font-bold text-[#e6f0ff] mb-2">SatyaAI</h1>
            <p className="text-[15px] font-medium text-[#8fa6c9]">Deepfake Detection System</p>
            {subtitle && (
              <p className="text-[#6b85a6] text-sm mt-1">{subtitle}</p>
            )}
          </div>

          {/* Card */}
          <div className="bg-[rgba(9,18,34,0.92)] border border-[rgba(120,160,220,0.15)] rounded-[14px] p-8 shadow-[0_20px_60px_rgba(0,0,0,0.55)]">
            {/* Card Header */}
            <div className="flex items-center justify-center mb-6">
              <Shield className="w-5 h-5 text-[#8fb3ff] mr-2" />
              <h2 className="text-xl font-semibold text-[#e6f0ff]">{title}</h2>
            </div>

            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
