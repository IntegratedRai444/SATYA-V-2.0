import React, { useState, useEffect, useCallback } from 'react';
import {
  Cloud,
  HelpCircle,
  Camera,
  Eye,
  Lock,
  Shield,
  Zap,
  Image,
  Video,
  Mic,
  Check,
  Layers,
  AlertTriangle,
  Clock,
  CheckCircle,
  Activity,
  TrendingUp,
  Book,
  ExternalLink,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import RecentActivity from '@/components/analysis/RecentActivity';
import AnalysisProgress from '@/components/analysis/AnalysisProgress';
import AnalysisResults from '@/components/analysis/AnalysisResults';
import ErrorBoundary from '@/components/ui/ErrorBoundary';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, RefreshCw, Loader } from 'lucide-react';
import { useDashboardStats } from '@/hooks/useDashboardStats';
import { useDashboardWebSocket } from '@/hooks/useDashboardWebSocket';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';

/**
 * Dashboard Component - Full Featured Version
 */
const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { signOut } = useSupabaseAuth();

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

  // State management
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
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

  // Dashboard hooks with error handling
  const statsQuery = useDashboardStats();
  const { isConnected } = useDashboardWebSocket({
    autoConnect: true,
    onStatsUpdate: () => {
      setError(null); // Clear error on successful update
    },
    onActivityUpdate: () => {
      // Handle activity updates
    },
  });

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
        <div className="relative bg-gradient-to-br from-[#1e3a5f] via-[#1a2f4a] to-[#152238] rounded-2xl p-10 overflow-hidden border border-gray-800/30">
          <div className="grid grid-cols-[1.2fr_320px] gap-16 relative z-10">
            {/* Left Section - Text Content */}
            <div className="space-y-5">
              {/* Badges */}
              <div className="flex items-center gap-2.5">
                <Badge className="bg-gradient-to-r from-purple-600 to-blue-600 text-white border-0 px-3.5 py-1.5 text-[11px] font-semibold rounded-md shadow-lg">
                  <Zap className="w-3 h-3 mr-1.5 inline" />
                  New AI Models Released
                </Badge>
                <Badge className="bg-transparent text-cyan-400 border border-cyan-500/60 px-3.5 py-1.5 text-[11px] font-semibold rounded-md">
                  Protection
                </Badge>
              </div>

              {/* Hero Heading */}
              <h1 className="text-[52px] font-bold leading-[1.1] text-white">
                Detect <span className="text-cyan-400">deepfakes</span> with the
                <br />
                power of SatyaAI
              </h1>

              {/* Description */}
              <p className="text-gray-200 text-[16px] leading-relaxed max-w-xl">
                Our advanced detection system helps you authenticate media with unprecedented
                accuracy, exposing manipulated content across images, videos, and audio.
              </p>

              {/* Secondary Text */}
              <p className="text-gray-400 text-[14px] leading-relaxed max-w-xl">
                Upload your files for analysis and get detailed authenticity reports instantly.
              </p>

              {/* CTA Buttons */}
              <div className="flex items-center gap-3.5 pt-4">
                <Button className="bg-cyan-500 hover:bg-cyan-600 text-white px-6 py-3 text-[14px] font-semibold rounded-lg shadow-lg shadow-cyan-500/30 transition-all">
                  <Cloud className="w-4 h-4 mr-2" />
                  Analyze Media
                  <span className="ml-2">→</span>
                </Button>
                <Button
                  variant="outline"
                  className="border-white/30 text-white hover:bg-white/10 hover:border-white/40 px-6 py-3 text-[14px] font-medium rounded-lg transition-all"
                >
                  <HelpCircle className="w-5 h-5" /> 
                  How It Works
                </Button>
              </div>
            </div>

            {/* Right Section - Authenticity Score Card */}
            <div className="relative flex items-start justify-end pt-4">
              <div className="w-full max-w-[240px]">
                <div className="flex flex-col items-center text-center space-y-4">
                  {/* Icon */}
                  <div className="w-14 h-14 rounded-lg border border-cyan-500/20 flex items-center justify-center bg-transparent">
                    <Camera className="w-5 h-5 text-cyan-400" /> 
                  </div>

                  {/* Score Display */}
                  <div>
                    <div className="text-[9px] text-cyan-400/70 uppercase tracking-[0.15em] font-semibold mb-1.5">
                      AUTHENTICITY SCORE
                    </div>
                    <div className="text-5xl font-bold text-cyan-400">75%</div>
                  </div>

                  {/* Feature List */}
                  <div className="w-full space-y-2 pt-2">
                    <div className="flex items-center gap-2.5 px-2 py-1.5">
                      <div className="w-5 h-5 rounded-md bg-transparent flex items-center justify-center">
                        <Eye className="w-3.5 h-3.5 text-cyan-400/70" />
                      </div>
                      <span className="text-[11px] text-gray-300/80 font-normal">
                        Real-time Analysis
                      </span>
                    </div>

                    <div className="flex items-center gap-2.5 px-2 py-1.5">
                      <div className="w-5 h-5 rounded-md bg-transparent flex items-center justify-center">
                        <Lock className="w-3.5 h-3.5 text-cyan-400/70" />
                      </div>
                      <span className="text-[11px] text-gray-300/80 font-normal">
                        Secure Processing
                      </span>
                    </div>

                    <div className="flex items-center gap-2.5 px-2 py-1.5">
                      <div className="w-5 h-5 rounded-md bg-transparent flex items-center justify-center">
                        <Shield className="w-3.5 h-3.5 text-cyan-400/70" />
                      </div>
                      <span className="text-[11px] text-gray-300/80 font-normal">
                        Verified Protection
                      </span>
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
              Accuracy: <span className="text-white font-semibold">98.2%</span>
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
              Accuracy: <span className="text-white font-semibold">96.8%</span>
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
              Accuracy: <span className="text-white font-semibold">95.3%</span>
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

          {/* Multimodal Card */}
          <Card className="bg-[#0f1419] border border-gray-800/50 p-5 rounded-lg hover:border-orange-500/40 transition-all cursor-pointer group relative">
            {/* Accuracy Badge - Top Right */}
            <div className="absolute top-3 right-3 text-[10px] text-gray-500">
              Accuracy: <span className="text-white font-semibold">99.1%</span>
            </div>

            {/* Icon - Centered at Top */}
            <div className="flex justify-start mb-4">
              <div className="w-10 h-10 rounded-md bg-orange-500/10 border border-orange-500/20 flex items-center justify-center">
                <Layers className="w-5 h-5 text-orange-400" strokeWidth={2} />
              </div>
            </div>

            {/* Content */}
            <div className="space-y-3">
              <h3 className="text-[15px] font-bold text-white">Multimodal Analysis</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed">
                Cross-media deepfake detection with AI fusion
              </p>

              {/* Features List */}
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Cross-Modal Fusion</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Multi-Source Analysis</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500">
                  <Check className="w-3 h-3 text-gray-600 flex-shrink-0" />
                  <span>Confidence Boost</span>
                </div>
              </div>

              {/* Button */}
              <button
                onClick={() => navigate('/smart-analysis')}
                className="w-full py-2 text-[10px] font-bold text-cyan-400 hover:text-cyan-300 rounded-md transition-all flex items-center justify-center gap-1 mt-4"
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
                <Layers className="w-6 h-6 text-blue-400" />
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
        <Card className="bg-[#0f1419] border border-gray-800/50 p-8 rounded-xl mt-6">
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
          <button className="w-full py-4 bg-cyan-500 hover:bg-cyan-600 text-white text-[14px] font-bold rounded-lg transition-all flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20">
            <Zap className="w-5 h-5" />
            View Complete Deepfake Guide
            <ExternalLink className="w-5 h-5" />
          </button>

          {/* Additional Info */}
          <p className="text-[12px] text-gray-500 text-center mt-5">
            New techniques added • Voice pattern analysis, metadata verification
          </p>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
