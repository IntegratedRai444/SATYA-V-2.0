import React, { useState, useEffect } from 'react';
import { Image, Video, Mic, FileText, Clock, ArrowRight, Activity } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AnalysisResult {
  id: string;
  filename: string;
  confidenceScore: number;
  status: 'Authentic' | 'Deepfake' | 'Suspicious';
  timestamp: Date;
  type: 'image' | 'video' | 'audio';
  processingTime?: string;
  thumbnailUrl?: string;
  reportCode?: string; // Case ID
}

const FileIcon: React.FC<{ type: string }> = ({ type }) => {
  const iconClass = "w-5 h-5";

  const getIconColor = (type: string) => {
    switch (type) {
      case 'image':
        return 'text-blue-400';
      case 'video':
        return 'text-purple-400';
      case 'audio':
        return 'text-green-400';
      default:
        return 'text-gray-400';
    }
  };

  switch (type) {
    case 'image':
      return <Image className={`${iconClass} ${getIconColor(type)}`} />;
    case 'video':
      return <Video className={`${iconClass} ${getIconColor(type)}`} />;
    case 'audio':
      return <Mic className={`${iconClass} ${getIconColor(type)}`} />;
    default:
      return <FileText className={`${iconClass} ${getIconColor(type)}`} />;
  }
};

interface RecentActivityItemProps {
  result: AnalysisResult;
  onClick: (id: string) => void;
}

const RecentActivityItem: React.FC<RecentActivityItemProps> = ({ result, onClick }) => {
  const getStatusColor = (status: string, score: number) => {
    if (status === 'Authentic' && score >= 90) return 'text-green-400';
    if (status === 'Deepfake' || score < 60) return 'text-red-400';
    return 'text-yellow-400';
  };

  const getStatusBgColor = (status: string, score: number) => {
    if (status === 'Authentic' && score >= 90) return 'bg-green-400/10 border-green-400/20';
    if (status === 'Deepfake' || score < 60) return 'bg-red-400/10 border-red-400/20';
    return 'bg-yellow-400/10 border-yellow-400/20';
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <div
      onClick={() => onClick(result.id)}
      className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg hover:bg-gray-700/50 transition-all duration-200 cursor-pointer group border border-gray-600/30 hover:border-gray-500/50"
    >
      <div className="flex items-center space-x-4 flex-1 min-w-0">
        {/* File Icon with Background */}
        <div className="w-10 h-10 bg-gray-600/30 rounded-lg flex items-center justify-center group-hover:bg-gray-600/50 transition-colors">
          <FileIcon type={result.type} />
        </div>

        {/* File Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-2 mb-1">
            <p className="text-white text-sm font-medium truncate group-hover:text-blue-100 transition-colors">
              {result.filename}
            </p>
            <span className="text-gray-500 text-xs flex-shrink-0">
              {result.type.toUpperCase()}
            </span>
          </div>
          <div className="flex items-center space-x-3 text-xs text-gray-400">
            <span>{result.confidenceScore}% confidence</span>
            {result.processingTime && (
              <>
                <span>•</span>
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{result.processingTime}</span>
                </div>
              </>
            )}
            <span>•</span>
            <span>{formatTimestamp(result.timestamp)}</span>
            {result.reportCode && (
              <>
                <span>•</span>
                <span className="text-gray-500 font-mono text-[10px]">{result.reportCode}</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Status Badge and Arrow */}
      <div className="flex items-center space-x-3">
        <div className={`px-3 py-1 rounded-full border ${getStatusBgColor(result.status, result.confidenceScore)}`}>
          <span className={`text-xs font-semibold ${getStatusColor(result.status, result.confidenceScore)}`}>
            {result.status}
          </span>
        </div>
        <ArrowRight className="w-4 h-4 text-gray-500 group-hover:text-gray-400 group-hover:translate-x-1 transition-all" />
      </div>
    </div>
  );
};

interface RecentActivityProps {
  loading?: boolean;
}

const RecentActivity: React.FC<RecentActivityProps> = ({ loading = false }) => {
  const [recentResults, setRecentResults] = useState<AnalysisResult[]>([]);
  const navigate = useNavigate();

  const handleActivityClick = (id: string) => {
    navigate(`/history/${id}`);
  };

  const handleViewAllClick = () => {
    navigate('/history');
  };

  useEffect(() => {
    // Fetch real recent activity from API
    const fetchRecentActivity = async () => {
      try {
        const response = await fetch('/api/dashboard/recent-activity', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('satyaai_auth_token')}`,
            'Content-Type': 'application/json'
          }
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data.recentScans) {
            // Map API response to component format
            const mappedResults: AnalysisResult[] = data.data.recentScans.map((scan: any) => ({
              id: scan.id.toString(),
              filename: scan.filename,
              confidenceScore: Math.round(scan.confidenceScore * 100),
              status: scan.result === 'authentic' ? 'Authentic' : 
                      scan.result === 'deepfake' ? 'Deepfake' : 'Suspicious',
              timestamp: new Date(scan.createdAt),
              type: scan.type,
              processingTime: scan.processingTime || undefined,
              reportCode: scan.reportCode // Include Case ID
            }));
            setRecentResults(mappedResults);
          }
        } else {
          // If API fails, show empty state
          setRecentResults([]);
        }
      } catch (error) {
        console.error('Failed to fetch recent activity:', error);
        // Show empty state on error
        setRecentResults([]);
      }
    };

    fetchRecentActivity();
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6 animate-pulse">
        <div className="flex items-center justify-between mb-6">
          <div className="h-6 bg-gray-700 rounded w-32"></div>
          <div className="w-2 h-2 bg-gray-700 rounded-full"></div>
        </div>
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg">
              <div className="flex items-center space-x-4 flex-1">
                <div className="w-10 h-10 bg-gray-700 rounded-lg"></div>
                <div className="flex-1">
                  <div className="h-4 bg-gray-700 rounded w-32 mb-2"></div>
                  <div className="h-3 bg-gray-700 rounded w-48"></div>
                </div>
              </div>
              <div className="h-6 bg-gray-700 rounded w-20"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-6 hover:bg-gray-800/70 transition-all duration-300">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <h3 className="text-lg font-semibold text-white">Recent Activity</h3>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-gray-500">Live</span>
          </div>
        </div>
        <Activity className="w-5 h-5 text-gray-400" />
      </div>

      <div className="space-y-3">
        {recentResults.length > 0 ? (
          recentResults.map((result) => (
            <RecentActivityItem
              key={result.id}
              result={result}
              onClick={handleActivityClick}
            />
          ))
        ) : (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-700/50 rounded-full flex items-center justify-center mx-auto mb-4">
              <FileText className="w-8 h-8 text-gray-400" />
            </div>
            <p className="text-gray-400 text-sm font-medium mb-1">No recent activity</p>
            <p className="text-gray-500 text-xs">Upload files to see analysis results</p>
          </div>
        )}
      </div>

      {recentResults.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-700/50">
          <button
            onClick={handleViewAllClick}
            className="w-full text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors flex items-center justify-center space-x-2 py-2 hover:bg-blue-400/5 rounded-lg"
          >
            <span>View All History</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};

export default RecentActivity;