import { useLocation } from 'react-router-dom';
import {
  FiHome,
  FiZap,
  FiClock,
  FiSettings,
  FiHelpCircle,
} from 'react-icons/fi';
import { NotificationBell } from '@/components/notifications/NotificationBell';
import { useUser } from '@/hooks/useUser';

const Navbar = () => {
  const location = useLocation();
  const { user } = useUser();

  const navItems = [
    { icon: FiHome, label: 'Home', path: '/dashboard' },
    { icon: FiZap, label: 'Scan', path: '/scan' },
    { icon: FiClock, label: 'History', path: '/history' },
    { icon: FiSettings, label: 'Settings', path: '/settings' },
    { icon: FiHelpCircle, label: 'Help', path: '/help' },
  ];

  const isActive = (path: string) => location.pathname === path;

  return (
    <header className="h-16 border-b border-white/5 bg-[#161B22] fixed top-0 right-0 left-0 z-50 shadow-[0_1px_0_rgba(255,255,255,0.05)]">
      <div className="flex items-center justify-between h-full px-8">
        
        {/* Logo Section */}
        <div className="flex items-center gap-2.5 pl-2">
          {/* Circular Logo with glow */}
          <div className="w-9 h-9 rounded-full bg-[#00BFFF] flex items-center justify-center shadow-lg shadow-cyan-500/30">
            <span className="text-white font-bold text-xl">S</span>
          </div>
          
          {/* Brand Name */}
          <div className="text-xl font-bold">
            Satya<span className="text-[#00BFFF]">AI</span>
          </div>
          
          {/* Tagline - Larger and more visible */}
          <span className="text-[13px] text-gray-400 ml-2 font-normal">
            Synthetic Authentication Technology for Your Analysis
          </span>
        </div>

        {/* Navigation Items */}
        <nav className="flex items-center gap-8">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.path);
            
            return (
              <button
                key={item.path}
                className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-[15px] font-medium transition-all duration-200 ${
                  active
                    ? 'bg-[#00BFFF] text-white shadow-lg shadow-cyan-500/30'
                    : 'text-[#E2E8F0] hover:text-[#00BFFF] hover:bg-[#00BFFF]/10'
                }`}
              >
                <Icon className="w-[18px] h-[18px]" strokeWidth={2} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Side Actions */}
        <div className="flex items-center gap-5">
          {/* Notification Bell with real-time updates */}
          <NotificationBell />
          
          {/* User Avatar */}
          <div className="w-9 h-9 rounded-full bg-[#00BFFF] flex items-center justify-center text-white font-bold text-[14px] shadow-lg shadow-cyan-500/30 cursor-pointer hover:shadow-cyan-500/50 transition-shadow">
            U
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
