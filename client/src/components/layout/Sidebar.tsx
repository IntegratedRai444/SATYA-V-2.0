import { useLocation, Link, useNavigate } from 'react-router-dom';
import { 
  // FiHome, // Not used
  FiImage, 
  FiVideo, 
  FiMic, 
  FiLayers, 
  // FiUpload, // DISABLED - batch processing
  // FiBarChart2, // Not used
  FiSettings, 
  FiHelpCircle, 
  // FiChevronDown, // Not used
  FiChevronLeft,
  FiChevronRight,
  FiGrid,
  FiTrendingUp,
  FiClock,
  FiZap
} from 'react-icons/fi';
import clsx from 'clsx';
import { useMediaQuery } from '@/hooks/use-media-query';

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
}

const Sidebar = ({ isOpen, onToggle }: SidebarProps) => {
  const location = useLocation();
  const navigate = useNavigate();
  const isActive = (path: string) => location.pathname === path;
  const isMobile = useMediaQuery('(max-width: 768px)');

  const handleOpenChat = () => {
    navigate('/ai-assistant');
  };

  const handleQuickAction = (action: string) => {
    navigate('/ai-assistant', { state: { quickPrompt: action } });
  };

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={onToggle}
        className="fixed left-0 top-20 z-50 bg-[#141414] border border-[#21262d] rounded-r-lg p-2 hover:bg-[#1c1c1c] transition-all duration-300"
        style={{ left: isOpen ? (isMobile ? '240px' : '280px') : '0' }}
        aria-label={isOpen ? 'Close sidebar' : 'Open sidebar'}
      >
        {isOpen ? (
          <FiChevronLeft className="w-5 h-5 text-gray-400" />
        ) : (
          <FiChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {/* Sidebar */}
      <aside 
        className={clsx(
          'bg-[#141414] border-r border-[#21262d] flex flex-col h-[calc(100vh-4rem)] fixed left-0 top-16 z-40 transition-transform duration-300',
          isMobile ? 'w-[240px]' : 'w-[280px]',
          isOpen ? 'translate-x-0' : '-translate-x-full'
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
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/dashboard')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/dashboard') ? 'page' : undefined}
                aria-label="Dashboard - Navigate to dashboard page"
              >
                <FiGrid className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Dashboard</span>
              </Link>
              
              {/* Image Analysis */}
              <Link
                to="/image-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/image-analysis')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/image-analysis') ? 'page' : undefined}
              >
                <FiImage className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Image Analysis</span>
              </Link>
              
              {/* Video Analysis */}
              <Link
                to="/video-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/video-analysis')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/video-analysis') ? 'page' : undefined}
              >
                <FiVideo className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Video Analysis</span>
              </Link>
              
              {/* Audio Analysis */}
              <Link
                to="/audio-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/audio-analysis')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/audio-analysis') ? 'page' : undefined}
              >
                <FiMic className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Audio Analysis</span>
              </Link>
              
              {/* Multimodal Analysis */}
              <Link
                to="/multimodal-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/multimodal-analysis')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/multimodal-analysis') ? 'page' : undefined}
              >
                <FiLayers className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Multimodal</span>
              </Link>

              {/* Batch Analysis - DISABLED */}
              {/* <Link
                to="/batch-analysis"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/batch-analysis')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/batch-analysis') ? 'page' : undefined}
              >
                <FiUpload className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
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
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/analytics')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/analytics') ? 'page' : undefined}
              >
                <FiTrendingUp className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Analytics</span>
              </Link>

              {/* Scan History */}
              <Link
                to="/scan-history"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/scan-history')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/scan-history') ? 'page' : undefined}
              >
                <FiClock className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Scan History</span>
              </Link>
              
              {/* Settings */}
              <Link
                to="/settings"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/settings')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/settings') ? 'page' : undefined}
              >
                <FiSettings className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Settings</span>
              </Link>
              
              {/* Help & Support */}
              <Link
                to="/help"
                className={clsx(
                  "w-full flex items-center gap-4 px-4 py-3.5 rounded-lg transition-all duration-150",
                  "focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-[#141414]",
                  isActive('/help')
                    ? "bg-cyan-500/10 text-cyan-400 border-l-3 border-cyan-500"
                    : "text-gray-400 hover:bg-[#1c1c1c] hover:text-white"
                )}
                aria-current={isActive('/help') ? 'page' : undefined}
              >
                <FiHelpCircle className="w-6 h-6 flex-shrink-0" strokeWidth={2} />
                <span className="text-[15px] font-medium">Help & Support</span>
              </Link>
              
            </div>
          </div>

          {/* AI Assistant Mini Chatbox - Inside Scrollable Area */}
          <div className="mt-6 mb-4">
            <div className="bg-[#1a1a1a] border-2 border-emerald-500/30 rounded-xl p-4 space-y-3 hover:border-emerald-500/50 transition-all duration-300">
            
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center">
                    <FiZap className="w-4 h-4 text-white" strokeWidth={2.5} />
                  </div>
                  <div className="absolute -top-0.5 -right-0.5 w-3 h-3 bg-emerald-400 rounded-full border-2 border-[#141414] animate-pulse"></div>
                </div>
                <div>
                  <h4 className="text-[13px] font-bold text-white">AI Assistant</h4>
                  <p className="text-[10px] text-emerald-400">Online</p>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="space-y-2">
              <button 
                onClick={() => handleQuickAction('How does deepfake detection work?')}
                className="w-full text-left px-3 py-2 rounded-lg bg-[#1a1a1a] hover:bg-[#222222] border border-emerald-500/20 hover:border-emerald-500/40 transition-all duration-200 group"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-400"></div>
                  <span className="text-[12px] text-gray-300 group-hover:text-white">Ask about detection</span>
                </div>
              </button>
              
              <button 
                onClick={() => handleQuickAction('Can you help me understand my analysis results?')}
                className="w-full text-left px-3 py-2 rounded-lg bg-[#1a1a1a] hover:bg-[#222222] border border-emerald-500/20 hover:border-emerald-500/40 transition-all duration-200 group"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                  <span className="text-[12px] text-gray-300 group-hover:text-white">Get help with results</span>
                </div>
              </button>
              
              <button 
                onClick={() => handleQuickAction('What are the best practices for deepfake detection?')}
                className="w-full text-left px-3 py-2 rounded-lg bg-[#1a1a1a] hover:bg-[#222222] border border-emerald-500/20 hover:border-emerald-500/40 transition-all duration-200 group"
              >
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 rounded-full bg-teal-400"></div>
                  <span className="text-[12px] text-gray-300 group-hover:text-white">Learn best practices</span>
                </div>
              </button>
            </div>

            {/* Open Chat Button */}
            <button 
              onClick={handleOpenChat}
              className="w-full py-2.5 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white text-[13px] font-semibold hover:shadow-lg hover:shadow-emerald-500/30 transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
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
