import { useLocation, Link } from 'react-router-dom';
import { 
  Image, 
  Video, 
  Mic, 
  Layers, 
  Settings, 
  HelpCircle, 
  ChevronLeft,
  ChevronRight,
  Grid,
  TrendingUp,
  Clock,
  Zap
} from 'lucide-react';
import clsx from 'clsx';
import { useMediaQuery } from '@/hooks/use-media-query';
import { useChat } from '@/contexts/ChatContext';

interface SidebarProps {
  isExpanded: boolean;
  onToggle: () => void;
}

const Sidebar = ({ isExpanded, onToggle }: SidebarProps) => {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;
  const isMobile = useMediaQuery('(max-width: 768px)');
  const { openChat } = useChat();

  const handleOpenChat = () => {
    openChat();
  };

  const handleQuickAction = (action: string) => {
    openChat(action);
  };

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="fixed left-0 top-20 z-50 bg-[#0f1419] border border-[#333333] rounded-r-lg p-2 hover:bg-[#1a2a3a] transition-all duration-300"
        style={{ left: isExpanded ? (isMobile ? '240px' : '280px') : '0' }}
        aria-label={isExpanded ? 'Close sidebar' : 'Open sidebar'}
      >
        {isExpanded ? (
          <ChevronLeft className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Sidebar - Flex positioning */}
      <aside 
        className={clsx(
          'bg-[#0f1419] border-r border-[#333333] flex flex-col h-full transition-all duration-300 flex-shrink-0 overflow-hidden',
          isExpanded 
            ? (isMobile ? 'w-[240px]' : 'w-[280px]') 
            : 'w-0'
        )}
      >
        {/* Scrollable Navigation Area */}
        <nav 
          className="flex-1 overflow-y-auto sidebar-scroll px-4 py-8"
          aria-label="Main navigation"
        >
          
          {/* Detection Tools Section */}
          <div className="mb-10" role="group" aria-labelledby="detection-tools-heading">
            <h3 
              id="detection-tools-heading"
              className="text-[12px] font-bold text-white uppercase tracking-wider mb-4 px-4"
            >
              Detection Tools
            </h3>
            <div className="space-y-2">
              
              {/* Dashboard - Active */}
              <Link
                to="/dashboard"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/dashboard')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/dashboard') ? 'page' : undefined}
                aria-label="Dashboard - Navigate to dashboard page"
              >
                <Grid className="w-5 h-5" />
                <span className="text-[15px] font-medium">Dashboard</span>
              </Link>
              
              {/* Image Analysis */}
              <Link
                to="/image-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/image-analysis')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/image-analysis') ? 'page' : undefined}
              >
                <Image className="w-5 h-5" />
                <span className="text-[15px] font-medium">Image Analysis</span>
              </Link>
              
              {/* Video Analysis */}
              <Link
                to="/video-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/video-analysis')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/video-analysis') ? 'page' : undefined}
              >
                <Video className="w-5 h-5" />
                <span className="text-[15px] font-medium">Video Analysis</span>
              </Link>
              
              {/* Audio Analysis */}
              <Link
                to="/audio-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/audio-analysis')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/audio-analysis') ? 'page' : undefined}
              >
                <Mic className="w-5 h-5" />
                <span className="text-[15px] font-medium">Audio Analysis</span>
              </Link>
              
              {/* Multimodal Analysis */}
              <Link
                to="/multimodal-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/multimodal-analysis')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/multimodal-analysis') ? 'page' : undefined}
              >
                <Layers className="w-5 h-5" />
                <span className="text-[15px] font-medium">Multimodal</span>
              </Link>

              {/* Batch Analysis - DISABLED */}
              {/* <Link
                to="/batch-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/batch-analysis')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/batch-analysis') ? 'page' : undefined}
              >
                <Upload className="w-5 h-5" />
                <span className="text-[15px] font-medium">Batch Analysis</span>
              </Link> */}
              
            </div>
          </div>

          {/* Management Section */}
          <div className="mb-10" role="group" aria-labelledby="management-heading">
            <h3 
              id="management-heading"
              className="text-[12px] font-bold text-white uppercase tracking-wider mb-4 px-4"
            >
              Management
            </h3>
            <div className="space-y-2">
              
              {/* Analytics */}
              <Link
                to="/analytics"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/analytics')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/analytics') ? 'page' : undefined}
              >
                <TrendingUp className="w-5 h-5" />
                <span className="text-[15px] font-medium">Analytics</span>
              </Link>

              {/* Scan History */}
              <Link
                to="/scan-history"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/scan-history')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/scan-history') ? 'page' : undefined}
              >
                <Clock className="w-5 h-5" />
                <span className="text-[15px] font-medium">Scan History</span>
              </Link>
              
              {/* Settings */}
              <Link
                to="/settings"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/settings')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/settings') ? 'page' : undefined}
              >
                <Settings className="w-5 h-5" />
                <span className="text-[15px] font-medium">Settings</span>
              </Link>
              
              {/* Help & Support */}
              <Link
                to="/help"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-[#00a8ff] focus:ring-offset-2 focus:ring-offset-[#0f1419]",
                  isActive('/help')
                    ? "bg-[#00a8ff]/10 text-[#00a8ff] border-l-3 border-[#00a8ff]"
                    : "text-gray-400 hover:bg-[#1a2a3a] hover:text-white"
                )}
                aria-current={isActive('/help') ? 'page' : undefined}
              >
                <HelpCircle className="w-5 h-5" />
                <span className="text-[15px] font-medium">Help & Support</span>
              </Link>
              
            </div>
          </div>

          {/* AI Assistant Mini Chatbox - Inside Scrollable Area */}
          <div className="mt-6 mb-4">
            <div className="bg-[#0a0a0a] border-2 border-[#00a8ff]/30 rounded-xl p-4 space-y-3 hover:border-[#00a8ff]/50 transition-all duration-300">
            
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00a8ff] to-[#0088cc] flex items-center justify-center">
                    <Zap className="w-4 h-4" />
                  </div>
                  <div className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-[#00a8ff] rounded-full border-2 border-[#0f1419] animate-pulse"></div>
                </div>
                <div>
                  <h4 className="text-[13px] font-bold text-white">AI Assistant</h4>
                  <p className="text-[10px] text-[#00a8ff]">Online</p>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="space-y-2">
              <button 
                onClick={() => handleQuickAction('How does deepfake detection work?')}
                className="w-full text-left px-3 py-2 rounded-lg bg-[#0f1419] hover:bg-[#1a2a3a] border border-[#00a8ff]/20 hover:border-[#00a8ff]/40 transition-all duration-200 group"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#00a8ff]"></div>
                  <span className="text-[12px] text-gray-300 group-hover:text-white">Ask about detection</span>
                </div>
              </button>
              
              <button 
                onClick={() => handleQuickAction('Can you help me understand my analysis results?')}
                className="w-full text-left px-3 py-2 rounded-lg bg-[#0f1419] hover:bg-[#1a2a3a] border border-[#00a8ff]/20 hover:border-[#00a8ff]/40 transition-all duration-200 group"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#0088cc]"></div>
                  <span className="text-[12px] text-gray-300 group-hover:text-white">Get help with results</span>
                </div>
              </button>
              
              <button 
                onClick={() => handleQuickAction('What are the best practices for deepfake detection?')}
                className="w-full text-left px-3 py-2 rounded-lg bg-[#0f1419] hover:bg-[#1a2a3a] border border-[#00a8ff]/20 hover:border-[#00a8ff]/40 transition-all duration-200 group"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#0066cc]"></div>
                  <span className="text-[12px] text-gray-300 group-hover:text-white">Learn best practices</span>
                </div>
              </button>
            </div>

            {/* Open Chat Button */}
            <button 
              onClick={handleOpenChat}
              className="w-full py-2.5 rounded-lg bg-gradient-to-r from-[#00a8ff] to-[#0088cc] text-white text-[13px] font-semibold hover:shadow-lg hover:shadow-[#00a8ff]/30 transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
            >
              Open Chat
            </button>
          </div>
          </div>
          
        </nav>
        
      </aside>
    </>
  );
};

export default Sidebar;
