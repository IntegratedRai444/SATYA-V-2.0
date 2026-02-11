import React, { useState, useEffect, useCallback } from 'react';
import {
  Upload,
  ArrowRight,
  Info,
  Camera,
  Eye,
  Zap,
  Image,
  Video,
  Mic,
  Check,
  FileText,
  AlertTriangle,
  Clock,
  CheckCircle,
  Activity,
  TrendingUp,
  Book,
  ExternalLink,
  Loader2,
  RefreshCw,
  Loader,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { DashboardBackground } from '@/components/ui/background-paths';
import RecentActivity from '@/components/analysis/RecentActivity';
import AnalysisProgress from '@/components/analysis/AnalysisProgress';
import AnalysisResults from '@/components/analysis/AnalysisResults';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useDashboardStats } from '@/hooks/useDashboardStats';
import { useDashboardWebSocket } from '@/hooks/useDashboardWebSocket';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import { useSystemHealth } from '@/hooks/useSystemHealth';
import SystemStatus from '@/components/system/SystemStatus';

/**
 * Dashboard Component - Full Featured Version
 */
const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { signOut } = useSupabaseAuth();
  useSystemHealth(); // Initialize system health monitoring

  const logout = useCallback(async () => {
    try {
      await signOut();
      navigate('/login');
    } catch (error) {
      if (import.meta.env.DEV) {
        console.error('Logout failed:', error);
      }
      // Force redirect even if logout fails
      navigate('/login');
    }
  }, [signOut, navigate]);

  // Listen for System Status events
  useEffect(() => {
    const handleOpenSystemStatus = () => {
      setShowSystemStatus(true);
    };

    window.addEventListener('openSystemStatus', handleOpenSystemStatus);
    return () => window.removeEventListener('openSystemStatus', handleOpenSystemStatus);
  }, []);

  // Listen for route-based System Status trigger
  useEffect(() => {
    const currentPath = window.location.pathname;
    if (currentPath === '/system-status') {
      setShowSystemStatus(true);
      // Navigate back to dashboard to prevent staying on system-status route
      navigate('/dashboard', { replace: true });
    }
  }, [navigate]);

  // State management
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showAnalyzeModal, setShowAnalyzeModal] = useState(false);
  const [showHowItWorksModal, setShowHowItWorksModal] = useState(false);
  const [showSystemStatus, setShowSystemStatus] = useState(false);
  const [currentModelIndex, setCurrentModelIndex] = useState(0);
  const [progressItems, setProgressItems] = useState<
    Array<{
      fileId: string;
      filename: string;
      progress: number;
      status: 'uploading' | 'processing' | 'completed' | 'error' | 'queued';
      message?: string;
      result?: Record<string, unknown>;
    }>
  >([]);

  const [analysisResults] = useState<
    Array<{
      id: string;
      filename: string;
      fileType: 'image' | 'video' | 'audio';
      confidence: number;
      authenticity: 'AUTHENTIC MEDIA' | 'MANIPULATED MEDIA' | 'UNCERTAIN';
      processingTime: number;
      keyFindings: string[];
      timestamp: Date;
    }>
  >([]);

  // AI Models data for rotation - One-line descriptions only
  const aiModels = [
    {
      title: "Image Analysis",
      subtitle: "Xception & EfficientNet face forgery detection",
      icon: Image
    },
    {
      title: "Video Analysis", 
      subtitle: "Temporal CNN + frame consistency checks",
      icon: Video
    },
    {
      title: "Audio Analysis",
      subtitle: "CNN-LSTM synthetic voice detection", 
      icon: Mic
    },
    {
      title: "Text Analysis",
      subtitle: "RoBERTa-based AI text detection",
      icon: FileText
    }
  ];

  // Rotate models every 15 seconds (reduced frequency)
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentModelIndex((prev) => (prev + 1) % aiModels.length);
    }, 15000); // 15 seconds (reduced from 8)

    return () => clearInterval(interval);
  }, [aiModels.length]);

  const { isConnected } = useDashboardWebSocket({
    autoConnect: true,
    onStatsUpdate: () => {
      setError(null); // Clear error on successful update
    },
    onActivityUpdate: () => {
      // Handle activity updates
    },
  });

  // Dashboard hooks with error handling
  const statsQuery = useDashboardStats();

  // Handle errors from hooks
  useEffect(() => {
    if (statsQuery.error) {
      if (import.meta.env.DEV) {
        console.error('Stats error:', statsQuery.error);
      }
      setError((prev) => prev || 'Failed to load statistics. Some features may be limited.');
    }
  }, [statsQuery.error]);

  // Handle initial load and error states
  useEffect(() => {
    setIsLoading(statsQuery.isLoading);

    // Set error if any of the hooks report an error
    if (statsQuery.error) {
      setError(statsQuery.error instanceof Error ? statsQuery.error.message : 'An error occurred');
    }
  }, [statsQuery.isLoading, statsQuery.error]);

  // Handle removing an item from progress
  const handleRemoveProgress = (fileId: string) => {
    setProgressItems((prev) => prev.filter((item) => item.fileId !== fileId));
  };

  // Handle refresh action
  const handleRefresh = async () => {
    try {
      setIsRefreshing(true);
      setError(null);
      // Trigger a refresh of all data
      // Note: You'll need to implement refresh methods in your hooks
      window.location.reload(); // Simple refresh for now
    } catch (err) {
      if (import.meta.env.DEV) {
        console.error('Refresh failed:', err);
      }
      setError('Failed to refresh data. Please try again.');
    } finally {
      setIsRefreshing(false);
    }
  };

  // Handle authentication errors
  useEffect(() => {
    if (error?.includes('401') || error?.includes('403')) {
      if (import.meta.env.DEV) {
        console.warn('Authentication error detected, redirecting to login');
      }
      logout();
    }
  }, [error, logout]);

  // Render error state
  const renderError = () => {
    if (!error) return null;

    return (
      <div className="mb-6">
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4 mr-2" />
          <AlertDescription className="flex justify-between items-center">
            <span>{error}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="ml-4"
            >
              {isRefreshing ? (
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-2" />
              )}
              Retry
            </Button>
          </AlertDescription>
        </Alert>
      </div>
    );
  };

  // Render loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 text-blue-500 animate-spin mx-auto" />
          <p className="text-lg text-slate-600">Loading dashboard...</p>
          <p className="text-sm text-slate-500">This may take a moment</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full">
      {renderError()}
      {/* Connection Status Badge */}
      <div className="flex justify-end mb-4 px-6">
        <div
          className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
            isConnected ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full mr-2 ${
              isConnected ? 'bg-green-500' : 'bg-yellow-500'
            }`}
          ></span>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Hero Banner Section - Full Width */}
      <div className="px-6 mb-14">
        <div className="relative bg-gradient-to-br from-[#0B1220] via-[#020617] to-[#0F172A] rounded-3xl p-12 overflow-hidden border border-cyan-400/25 shadow-xl">
          {/* Background Paths Animation */}
          <DashboardBackground />
          
          {/* Futuristic Star Background - Enhanced & Visible */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none z-[1]">
            <div className="absolute inset-0">
              {/* Primary Star Field - Subtle */}
              <div className="absolute top-[8%] left-[12%] w-1 h-1 bg-cyan-300/12 rounded-full"></div>
              <div className="absolute top-[15%] left-[28%] w-0.5 h-0.5 bg-blue-200/8 rounded-full"></div>
              <div className="absolute top-[6%] left-[45%] w-1 h-1 bg-cyan-200/10 rounded-full"></div>
              <div className="absolute top-[22%] left-[8%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[18%] left-[38%] w-1 h-1 bg-cyan-300/8 rounded-full"></div>
              <div className="absolute top-[11%] left-[58%] w-0.5 h-0.5 bg-blue-200/6 rounded-full"></div>
              <div className="absolute top-[25%] left-[72%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[4%] left-[85%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[32%] left-[18%] w-1 h-1 bg-cyan-300/8 rounded-full"></div>
              <div className="absolute top-[38%] left-[35%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[28%] left-[62%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[42%] left-[22%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[48%] left-[48%] w-1 h-1 bg-cyan-300/8 rounded-full"></div>
              <div className="absolute top-[35%] left-[78%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[55%] left-[12%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[52%] left-[32%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[58%] left-[52%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[62%] left-[68%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[68%] left-[25%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[72%] left-[42%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[78%] left-[82%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[82%] left-[15%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[88%] left-[38%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              <div className="absolute top-[92%] left-[58%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[96%] left-[75%] w-1 h-1 bg-cyan-200/8 rounded-full"></div>
              
              {/* Secondary Star Layer - Subtle */}
              <div className="absolute top-[10%] left-[20%] w-0.5 h-0.5 bg-white/6 rounded-full"></div>
              <div className="absolute top-[25%] left-[15%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[40%] left-[30%] w-0.5 h-0.5 bg-white/5 rounded-full"></div>
              <div className="absolute top-[55%] left-[45%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[70%] left-[60%] w-0.5 h-0.5 bg-white/5 rounded-full"></div>
              <div className="absolute top-[85%] left-[75%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[15%] left-[80%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[30%] left-[10%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[45%] left-[70%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[60%] left-[25%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[75%] left-[40%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[90%] left-[55%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[20%] left-[50%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[35%] left-[65%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[50%] left-[80%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[65%] left-[15%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[80%] left-[30%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[95%] left-[45%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[12%] left-[35%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[27%] left-[55%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[42%] left-[75%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[57%] left-[20%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[72%] left-[40%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[87%] left-[60%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[18%] left-[85%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[33%] left-[25%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[48%] left-[45%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[63%] left-[65%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[78%] left-[85%] w-0.5 h-0.5 bg-white/3 rounded-full"></div>
              <div className="absolute top-[93%] left-[15%] w-0.5 h-0.5 bg-white/4 rounded-full"></div>
              <div className="absolute top-[45%] left-[45%] w-0.5 h-0.5 bg-cyan-300/6 rounded-full"></div>
              <div className="absolute top-[52%] left-[68%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[58%] left-[85%] w-0.5 h-0.5 bg-cyan-300/6 rounded-full"></div>
              <div className="absolute top-[65%] left-[55%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[72%] left-[78%] w-0.5 h-0.5 bg-cyan-300/6 rounded-full"></div>
              <div className="absolute top-[78%] left-[25%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[85%] left-[65%] w-0.5 h-0.5 bg-cyan-300/6 rounded-full"></div>
              <div className="absolute top-[92%] left-[88%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              <div className="absolute top-[98%] left-[75%] w-0.5 h-0.5 bg-blue-300/6 rounded-full"></div>
              
              {/* Enhanced Minimal Streaks - Subtle */}
              <div className="absolute top-[12%] left-[15%] w-10 h-px bg-gradient-to-r from-transparent via-cyan-400/6 to-transparent transform rotate-12"></div>
              <div className="absolute top-[28%] left-[52%] w-8 h-px bg-gradient-to-r from-transparent via-blue-300/4 to-transparent transform -rotate-6"></div>
              <div className="absolute top-[45%] left-[25%] w-12 h-px bg-gradient-to-r from-transparent via-cyan-300/5 to-transparent transform rotate-8"></div>
              <div className="absolute top-[67%] left-[68%] w-10 h-px bg-gradient-to-r from-transparent via-blue-400/4 to-transparent transform -rotate-15"></div>
              <div className="absolute top-[83%] left-[35%] w-9 h-px bg-gradient-to-r from-transparent via-cyan-300/3 to-transparent transform rotate-5"></div>
              
              {/* Distant Star Field - Enhanced Visibility */}
              <div className="absolute top-[5%] left-[20%] w-0.5 h-0.5 bg-cyan-100/8 rounded-full"></div>
              <div className="absolute top-[17%] left-[35%] w-0.5 h-0.5 bg-blue-100/6 rounded-full"></div>
              <div className="absolute top-[31%] left-[48%] w-0.5 h-0.5 bg-cyan-100/8 rounded-full"></div>
              <div className="absolute top-[43%] left-[65%] w-0.5 h-0.5 bg-blue-100/6 rounded-full"></div>
              <div className="absolute top-[57%] left-[18%] w-0.5 h-0.5 bg-cyan-100/8 rounded-full"></div>
              <div className="absolute top-[69%] left-[82%] w-0.5 h-0.5 bg-blue-100/6 rounded-full"></div>
              <div className="absolute top-[79%] left-[28%] w-0.5 h-0.5 bg-cyan-100/8 rounded-full"></div>
              <div className="absolute top-[91%] left-[55%] w-0.5 h-0.5 bg-blue-100/6 rounded-full"></div>
            </div>
          </div>
          
          <div className="grid grid-cols-[1.2fr_320px] gap-16 relative z-10 items-center">
            {/* Left Section - Text Content */}
            <div className="space-y-8">
              {/* Hero Heading */}
              <div className="space-y-4">
                <h1 className="text-[48px] font-bold leading-[1.1] text-white tracking-tight mb-6">
                  Detect <span className="text-cyan-400">deepfakes</span> with the
                  <br />
                  <span className="text-white text-[56px]">power of SatyaAI</span>
                </h1>

                {/* Subtitle */}
                <p className="text-gray-300 text-[17px] leading-relaxed max-w-2xl font-light mb-4">
                  Our advanced detection system helps you authenticate media with 
                  <span className="text-cyan-400 font-medium">unprecedented accuracy</span>, exposing manipulated content across images, videos, and audio.
                </p>
                <p className="text-gray-400 text-[15px] leading-relaxed max-w-2xl mb-8">
                  Upload your files or use your webcam for real-time analysis and get detailed authenticity reports instantly.
                </p>

                {/* Feature Highlights */}
                <div className="flex items-center gap-8 text-[14px] text-gray-400">
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-500/30 rounded-full"></div>
                    <span>Real-time Analysis</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-2 h-2 bg-blue-400/30 rounded-full"></div>
                    <span>ML-Powered Detection</span>
                  </div>
                </div>

                {/* CTA Button */}
                <div className="flex gap-4 pt-6">
                  <Button 
                    onClick={() => setShowAnalyzeModal(true)}
                    className="group relative bg-cyan-400 hover:bg-cyan-500 text-white px-8 py-3 text-[16px] font-semibold rounded-xl shadow-lg shadow-cyan-400/20 transition-all duration-300 hover:scale-105 overflow-hidden"
                  >
                    <Upload className="w-5 h-5 mr-3 relative z-10" />
                    <span className="relative z-10 font-medium">Analyze Media</span>
                    <ArrowRight className="w-4 h-4 ml-2 relative z-10 group-hover:translate-x-1 transition-transform duration-300" />
                  </Button>
                  <Button 
                    onClick={() => setShowHowItWorksModal(true)}
                    variant="outline" className="group relative border-cyan-400 hover:bg-cyan-400/10 text-cyan-400 px-8 py-3 text-[16px] font-semibold rounded-xl transition-all duration-300 hover:scale-105 overflow-hidden"
                  >
                    <Info className="w-5 h-5 mr-3 relative z-10" />
                    <span className="relative z-10 font-medium">How It Works</span>
                  </Button>
                </div>
              </div>
            </div>

            {/* Right Section - AI Engine Status - Fully Transparent Design */}
            <div className="relative flex items-center justify-center">
              <div className="w-full max-w-[380px]">
                <div className="bg-cyan-400/3 backdrop-blur-3xl rounded-3xl p-8 relative overflow-hidden border border-cyan-400/15 shadow-2xl shadow-cyan-400/10">
                  {/* UI-matching glassmorphism gradient overlay */}
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/8 via-transparent to-blue-400/5 opacity-40"></div>
                  <div className="absolute inset-0 rounded-3xl border border-cyan-400/20 shadow-inner shadow-cyan-400/10"></div>
                  
                  {/* UI-matching animated glass particles */}
                  <div className="absolute inset-0 overflow-hidden pointer-events-none">
                    <div className="absolute top-4 left-4 w-1 h-1 bg-cyan-300/25 rounded-full animate-pulse"></div>
                    <div className="absolute top-8 right-8 w-0.5 h-0.5 bg-blue-300/20 rounded-full animate-pulse delay-75"></div>
                    <div className="absolute bottom-6 left-12 w-1 h-1 bg-cyan-200/22 rounded-full animate-pulse delay-150"></div>
                    <div className="absolute bottom-8 right-6 w-0.5 h-0.5 bg-blue-200/20 rounded-full animate-pulse delay-300"></div>
                  </div>
                  
                  <div className="flex flex-col items-center text-center space-y-8 relative z-10">
                    {/* Static Header - Transparent */}
                    <div className="space-y-3 text-center">
                      <div className="text-[10px] uppercase tracking-[0.15em] font-medium text-cyan-300/80 flex items-center justify-center gap-3">
                        <div className="w-1.5 h-1.5 bg-cyan-400/50 rounded-full animate-pulse"></div>
                        AI ENGINE STATUS
                        <div className="w-1.5 h-1.5 bg-cyan-400/50 rounded-full animate-pulse"></div>
                      </div>
                      <div className="flex items-center justify-center gap-3">
                        <div className="relative">
                          <div className="w-2.5 h-2.5 bg-cyan-400/70 rounded-full animate-pulse"></div>
                          <div className="absolute inset-0 bg-cyan-400/40 rounded-full blur-md animate-pulse"></div>
                        </div>
                        <span className="text-xl font-light text-white tracking-wide">ONLINE</span>
                      </div>
                      <p className="text-xs text-cyan-200/70 font-medium">Models loaded · Real-time analysis ready</p>
                    </div>

                    {/* Enhanced Rotating Content - Transparent */}
                    <div className="relative h-48 flex items-center justify-center w-full">
                      {aiModels.map((model, index) => (
                        <div
                          key={index}
                          className={`absolute inset-0 flex flex-col items-center justify-center space-y-6 transition-all duration-1000 ease-in-out w-full px-6 ${
                            index === currentModelIndex 
                              ? 'opacity-100 scale-100' 
                              : 'opacity-0 scale-95'
                          }`}
                        >
                          {/* UI-matching Glassmorphism Model Icon */}
                          <div className="w-16 h-16 rounded-2xl border border-cyan-400/20 bg-cyan-400/2 flex items-center justify-center relative group backdrop-blur-xl">
                            {/* UI-matching glassmorphism glow */}
                            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-cyan-400/10 via-transparent to-blue-400/5 blur-lg opacity-60 group-hover:opacity-80 transition-opacity"></div>
                            <div className="absolute -inset-1 rounded-2xl border border-cyan-400/30 shadow-lg shadow-cyan-400/20"></div>
                            <div className="absolute inset-0 rounded-2xl bg-gradient-to-t from-transparent via-cyan-400/8 to-transparent opacity-25"></div>
                            <model.icon className="w-8 h-8 text-cyan-300/90 relative z-10" strokeWidth={1.5} />
                          </div>
                          
                          {/* Enhanced Model Title */}
                          <div className="space-y-3 text-center w-full">
                            <h3 className="text-lg font-semibold text-white tracking-tight">
                              {model.title}
                            </h3>
                            <p className="text-sm text-cyan-200/80 leading-relaxed text-center font-medium">
                              {model.subtitle}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* UI-matching Glassmorphism Pagination */}
                    <div className="flex gap-2 pt-4 justify-center">
                      {aiModels.map((_, index) => (
                        <div
                          key={index}
                          className={`h-2 w-8 rounded-full transition-all duration-700 relative ${
                            index === currentModelIndex
                              ? 'bg-cyan-400/50 shadow-lg shadow-cyan-400/30 border border-cyan-400/25'
                              : 'bg-cyan-400/15 border border-cyan-400/10'
                          }`}
                        >
                          {/* UI-matching glassmorphism active indicator */}
                          {index === currentModelIndex && (
                            <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-sm"></div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Detection Tools Section */}
      <div className="px-6">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-[28px] font-bold text-white mb-2">Deepfake Detection Tools</h2>
            <p className="text-gray-500 text-[14px]">
              Choose your media type for comprehensive analysis
            </p>
          </div>
          <div className="flex items-center gap-2 text-[13px]">
            <span className="text-gray-500">Using</span>
            <span className="text-cyan-400 font-semibold">Neural Vision v4.2</span>
            <span className="text-gray-500">+ models</span>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Image Card */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-5 rounded-lg hover:border-blue-500/40 transition-all cursor-pointer group relative">
            {/* Accuracy Badge - Top Right */}
            <div className="absolute top-3 right-3 text-[10px] text-gray-500">
              Accuracy: <span className="text-white font-semibold">94.7%</span>
            </div>

            {/* Icon - Centered at Top */}
            <div className="flex justify-start mb-4">
              <div className="w-10 h-10 rounded-md bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <Image className="w-5 h-5 text-blue-400" strokeWidth={2} />
              </div>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h3 className="text-[15px] font-bold text-white">Image Analysis</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                Detect manipulated photos & generated images
              </p>

              {/* Features List */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Photoshop Detection</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>GAN Detection</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Metadata Analysis</span>
                </div>
              </div>

              {/* Button */}
              <button
                onClick={() => navigate('/image-analysis')}
                className="w-full py-2 text-[10px] font-bold text-cyan-400 hover:text-cyan-300 rounded-md transition-all flex items-center justify-center gap-1 mt-4"
              >
                START ANALYSIS →
              </button>
            </div>
          </Card>

          {/* Video Card */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-5 rounded-lg hover:border-green-500/40 transition-all cursor-pointer group relative">
            {/* Accuracy Badge - Top Right */}
            <div className="absolute top-3 right-3 text-[10px] text-gray-500">
              Accuracy: <span className="text-white font-semibold">89.3%</span>
            </div>

            {/* Icon - Centered at Top */}
            <div className="flex justify-start mb-4">
              <div className="w-10 h-10 rounded-md bg-green-500/10 border border-green-500/20 flex items-center justify-center">
                <Video className="w-5 h-5 text-green-400" strokeWidth={2} />
              </div>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h3 className="text-[15px] font-bold text-white">Video Verification</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                Identify deepfake videos & facial manipulations
              </p>

              {/* Features List */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Facial Inconsistencies</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Temporal Analysis</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Lip-Sync Verification</span>
                </div>
              </div>

              {/* Button */}
              <button
                onClick={() => navigate('/video-analysis')}
                className="w-full py-2 text-[10px] font-bold text-cyan-400 hover:text-cyan-300 rounded-md transition-all flex items-center justify-center gap-1 mt-4"
              >
                START ANALYSIS →
              </button>
            </div>
          </Card>

          {/* Audio Card */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-5 rounded-lg hover:border-purple-500/40 transition-all cursor-pointer group relative">
            {/* Accuracy Badge - Top Right */}
            <div className="absolute top-3 right-3 text-[10px] text-gray-500">
              Accuracy: <span className="text-white font-semibold">91.7%</span>
            </div>

            {/* Icon - Centered at Top */}
            <div className="flex justify-start mb-4">
              <div className="w-10 h-10 rounded-md bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
                <Mic className="w-5 h-5 text-purple-400" strokeWidth={2} />
              </div>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h3 className="text-[15px] font-bold text-white">Audio Detection</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                Uncover voice cloning & synthetic speech
              </p>

              {/* Features List */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Voice Cloning Detection</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Natural Patterns Analysis</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Neural Voice Filter</span>
                </div>
              </div>

              {/* Button */}
              <button
                onClick={() => navigate('/audio-analysis')}
                className="w-full py-2 text-[10px] font-bold text-cyan-400 hover:text-cyan-300 rounded-md transition-all flex items-center justify-center gap-1 mt-4"
              >
                START ANALYSIS →
              </button>
            </div>
          </Card>

          {/* Text Analysis Card */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-5 rounded-lg hover:border-blue-500/40 transition-all cursor-pointer group relative">
            {/* Accuracy Badge - Top Right */}
            <div className="absolute top-3 right-3 text-[10px] text-gray-500">
              Accuracy: <span className="text-white font-semibold">94.2%</span>
            </div>

            {/* Icon - Centered at Top */}
            <div className="flex justify-start mb-4">
              <div className="w-10 h-10 rounded-md bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
                <FileText className="w-5 h-5 text-blue-400" strokeWidth={2} />
              </div>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h3 className="text-[15px] font-bold text-white">Text Authenticity Analysis</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                AI-generated text detection with NLP models
              </p>

              {/* Features List */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>AI Pattern Detection</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Stylometric Analysis</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Perplexity Scoring</span>
                </div>
              </div>

              {/* Button */}
              <button
                onClick={() => navigate('/text-analysis')}
                className="w-full py-2 text-[10px] font-bold text-blue-400 hover:text-blue-300 rounded-md transition-all flex items-center justify-center gap-1 mt-4"
              >
                START ANALYSIS →
              </button>
            </div>
          </Card>

          {/* Batch Analysis Card - DISABLED */}
        </div>
      </div>

      {/* Analysis Progress Section */}
      {progressItems.length > 0 && (
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Analysis in Progress</h2>
            <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isRefreshing}>
              {isRefreshing ? (
                <Loader className="h-4 w-4 animate-spin mr-2" />
              ) : (
                <RefreshCw className="h-4 w-4 mr-2" />
              )}
              Refresh
            </Button>
          </div>
          <ErrorBoundary
            fallback={
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4 mr-2" />
                <AlertDescription>
                  Failed to load analysis progress. Please try refreshing the page.
                </AlertDescription>
              </Alert>
            }
          >
            <AnalysisProgress progressItems={progressItems} onRemove={handleRemoveProgress} />
          </ErrorBoundary>
        </div>
      )}

      {/* Analysis Results Section */}
      {analysisResults.length > 0 && (
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Recent Analysis Results</h2>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                // Handle view all results
                navigate('/analysis/history');
              }}
            >
              View All
            </Button>
          </div>
          <ErrorBoundary
            fallback={
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4 mr-2" />
                <AlertDescription>
                  Failed to load analysis results. Please try again later.
                </AlertDescription>
              </Alert>
            }
          >
            <AnalysisResults results={analysisResults} isLoading={isLoading} />
          </ErrorBoundary>
        </div>
      )}

      {/* Analytics & Insights Section */}
      <div className="mt-12">
        {/* Section Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-[20px] font-bold text-white mb-1 flex items-center gap-2">
              Analytics & Insights
              <Activity className="w-5 h-5 text-cyan-400" />
            </h2>
            <p className="text-gray-400 text-[13px]">System performance and detection tips</p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              className="border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/10 px-4 py-2 text-[11px] font-semibold rounded-md"
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              Statistics
            </Button>
            <Button
              variant="outline"
              className="border-gray-700 text-gray-400 hover:bg-gray-800/50 px-4 py-2 text-[11px] font-semibold rounded-md"
            >
              <Eye className="w-4 h-4 mr-2" />
              Insights
            </Button>
          </div>
        </div>

        {/* Stats Grid - 2x2 on mobile, 4 columns on desktop */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Analyzed Media */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-6 rounded-xl">
            <div className="flex items-start gap-3 mb-4">
              <div className="w-12 h-12 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <FileText className="w-6 h-6 text-blue-400" />
              </div>
              <div className="flex-1">
                <p className="text-[12px] text-gray-400 font-medium">Analyzed Media</p>
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-[48px] font-bold text-white leading-none">147</span>
              <span className="text-[13px] text-green-400 font-semibold">+23% ↑</span>
            </div>
          </Card>

          {/* Detected Deepfakes */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-6 rounded-xl">
            <div className="flex items-start gap-3 mb-4">
              <div className="w-12 h-12 rounded-lg bg-red-500/10 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-red-400" />
              </div>
              <div className="flex-1">
                <p className="text-[12px] text-gray-400 font-medium">Detected Deepfakes</p>
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-[48px] font-bold text-white leading-none">36</span>
              <span className="text-[13px] text-red-400 font-semibold">+12% ↓</span>
            </div>
          </Card>

          {/* Avg Detection Time */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-6 rounded-xl">
            <div className="flex items-start gap-3 mb-4">
              <div className="w-12 h-12 rounded-lg bg-yellow-500/10 flex items-center justify-center">
                <Clock className="w-6 h-6 text-yellow-400" />
              </div>
              <div className="flex-1">
                <p className="text-[12px] text-gray-400 font-medium">Avg. Detection Time</p>
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-[48px] font-bold text-white leading-none">4.2s</span>
              <span className="text-[13px] text-green-400 font-semibold">+8% ↑</span>
            </div>
          </Card>

          {/* Detection Accuracy */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-6 rounded-xl">
            <div className="flex items-start gap-3 mb-4">
              <div className="w-12 h-12 rounded-lg bg-green-500/10 flex items-center justify-center">
                <CheckCircle className="w-6 h-6 text-green-400" />
              </div>
              <div className="flex-1">
                <p className="text-[12px] text-gray-400 font-medium">Detection Accuracy</p>
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-[48px] font-bold text-white leading-none">96%</span>
              <span className="text-[13px] text-green-400 font-semibold">+3% ↑</span>
            </div>
          </Card>
        </div>

        {/* Two Column Layout - Chart and Activity */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-6">
          {/* Detection Activity Chart */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-6 rounded-xl">
            <h3 className="text-[17px] font-bold text-white mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-gray-400" />
              Detection Activity
            </h3>
            <div className="h-64 flex items-end justify-between gap-1">
              {[
                15, 22, 28, 35, 30, 38, 33, 42, 38, 45, 40, 48, 44, 52, 48, 55, 50, 58, 54, 60, 56,
                62, 58, 65, 60, 68, 63, 70, 65, 72, 68, 75, 70, 78, 72, 80, 75, 82, 78, 85, 80, 88,
                82, 90, 85, 92, 88, 95, 90, 98,
              ].map((height, i) => (
                <div
                  key={i}
                  className={`flex-1 rounded-t-sm transition-all cursor-pointer ${
                    i === 16
                      ? 'bg-cyan-400 shadow-lg shadow-cyan-400/50'
                      : 'bg-cyan-600/50 hover:bg-cyan-500/70'
                  }`}
                  style={{ height: `${height}%` }}
                  title={i === 16 ? '16 Alerts' : `${i} detections`}
                />
              ))}
            </div>
            <div className="flex justify-between mt-4 text-[11px] text-gray-500 font-medium">
              <span>00:00</span>
              <span>06:00</span>
              <span>12:00</span>
              <span>18:00</span>
              <span>23:59</span>
            </div>

            {/* Chart Legend */}
            <div className="flex items-center justify-between mt-5 pt-4 border-t border-gray-800/50">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-cyan-600/50"></div>
                <span className="text-[11px] text-gray-400">Normal Activity</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-cyan-400"></div>
                <span className="text-[11px] text-gray-400">Peak Detection</span>
              </div>
              <div className="text-[11px] text-gray-500">
                <span className="text-white font-semibold">2,847</span> total scans today
              </div>
            </div>
          </Card>

          {/* Recent Activity */}
          <ErrorBoundary>
            <RecentActivity />
          </ErrorBoundary>
        </div>

        {/* Detection Guide - Full Width */}
        <Card key="detection-guide-section" id="detection-guide" className="bg-[#0f1419] border border-gray-800/50 p-8 rounded-xl mt-6">
          <div className="flex items-center justify-between mb-7">
            <h3 className="text-[18px] font-bold text-white flex items-center gap-2">
              <Book className="w-5 h-5 text-cyan-400" />
              Detection Guide
            </h3>
            <Badge className="bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-4 py-2 text-[12px] font-semibold rounded-lg">
              Expert Tips
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-7">
            {/* Guide Item 1 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-[14px] font-bold text-cyan-400">
                1
              </div>
              <p className="text-[14px] text-gray-300 leading-relaxed">
                Look for unnatural eye blinking patterns and inconsistent eye reflections in
                suspected videos.
              </p>
            </div>

            {/* Guide Item 2 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-[14px] font-bold text-cyan-400">
                2
              </div>
              <p className="text-[14px] text-gray-300 leading-relaxed">
                Check for unnatural hair movement, unusual skin texture, or blurry face boundaries
                in images.
              </p>
            </div>

            {/* Guide Item 3 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-[14px] font-bold text-cyan-400">
                3
              </div>
              <p className="text-[14px] text-gray-300 leading-relaxed">
                Watch for inconsistencies in audio-visual synchronization, especially in speech
                videos.
              </p>
            </div>

            {/* Guide Item 4 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-[14px] font-bold text-cyan-400">
                4
              </div>
              <p className="text-[14px] text-gray-300 leading-relaxed">
                Analyze visual artifacts around the edges of faces, which often indicate
                manipulation.
              </p>
            </div>
          </div>

          {/* View Complete Guide Button */}
          <Button 
            key="view-complete-guide-button"
            onClick={() => {
              console.log('Navigating to detection guide...');
              navigate('/detection-guide');
            }}
            className="w-full py-4 bg-cyan-500 hover:bg-cyan-600 text-white text-[14px] font-bold rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20"
          >
            <Zap className="w-5 h-5" />
            View Complete Deepfake Guide
            <ExternalLink className="w-5 h-5" />
          </Button>

          {/* Additional Info */}
          <p className="text-[12px] text-gray-500 text-center mt-5">
            New techniques added • Voice pattern analysis, metadata verification
          </p>
        </Card>

        {/* Analyze Media Modal */}
        {showAnalyzeModal && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50" onClick={() => setShowAnalyzeModal(false)}>
            <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-white">Choose Analysis Type</h3>
                <button 
                  onClick={() => setShowAnalyzeModal(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-3">
                <button 
                  onClick={() => {
                    setShowAnalyzeModal(false);
                    navigate('/image-analysis');
                  }}
                  className="w-full flex items-center gap-4 p-4 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-left"
                >
                  <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Image className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium">Image Analysis</div>
                    <div className="text-gray-400 text-sm">Analyze photos and static images</div>
                  </div>
                </button>

                <button 
                  onClick={() => {
                    setShowAnalyzeModal(false);
                    navigate('/video-analysis');
                  }}
                  className="w-full flex items-center gap-4 p-4 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-left"
                >
                  <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <Video className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium">Video Analysis</div>
                    <div className="text-gray-400 text-sm">Analyze video files and recordings</div>
                  </div>
                </button>

                <button 
                  onClick={() => {
                    setShowAnalyzeModal(false);
                    navigate('/audio-analysis');
                  }}
                  className="w-full flex items-center gap-4 p-4 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-left"
                >
                  <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <Mic className="w-5 h-5 text-green-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium">Audio Analysis</div>
                    <div className="text-gray-400 text-sm">Analyze audio files and voice recordings</div>
                  </div>
                </button>

                <button 
                  onClick={() => {
                    setShowAnalyzeModal(false);
                    navigate('/smart-analysis');
                  }}
                  className="w-full flex items-center gap-4 p-4 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors text-left"
                >
                  <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                    <Camera className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div>
                    <div className="text-white font-medium">Webcam / Multimodal</div>
                    <div className="text-gray-400 text-sm">Real-time analysis with webcam</div>
                  </div>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* How It Works Modal */}
        {showHowItWorksModal && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50" onClick={() => setShowHowItWorksModal(false)}>
            <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-white">How SatyaAI Works</h3>
                <button 
                  onClick={() => setShowHowItWorksModal(false)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              
              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Upload className="w-4 h-4 text-blue-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-medium mb-2">Media Upload & Capture</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Simply upload your media files or use your webcam for real-time capture. Our system accepts images, videos, and audio files in all common formats.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <Zap className="w-4 h-4 text-purple-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-medium mb-2">AI Model Analysis</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Our advanced AI models analyze visual patterns, audio frequencies, and metadata signals. Multiple specialized models work together to detect manipulation across different media types.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <Activity className="w-4 h-4 text-green-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-medium mb-2">Signal Fusion</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Individual detection signals are combined using sophisticated fusion algorithms. This creates a comprehensive authenticity score that's more accurate than any single indicator.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center">
                    <CheckCircle className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div>
                    <h4 className="text-white font-medium mb-2">Confidence Reports</h4>
                    <p className="text-gray-300 text-sm leading-relaxed">
                      Receive detailed reports with confidence scores, key findings, and explanations. Each analysis includes cryptographic proof for verification and audit trails.
                    </p>
                  </div>
                </div>

                <div className="border-t border-gray-700 pt-4">
                  <p className="text-gray-400 text-sm">
                    Want to learn more about manual detection techniques? Check out our 
                    <button 
                      onClick={() => {
                        setShowHowItWorksModal(false);
                        // Scroll to detection guide section
                        const element = document.getElementById('detection-guide');
                        element?.scrollIntoView({ behavior: 'smooth' });
                      }}
                      className="text-cyan-400 hover:text-cyan-300 underline ml-1"
                    >
                      Detection Guide
                    </button>
                    {' '}for expert tips.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* System Status Modal */}
      <SystemStatus 
        isOpen={showSystemStatus}
        onClose={() => setShowSystemStatus(false)}
      />
    </div>
  );
};

export default Dashboard;
