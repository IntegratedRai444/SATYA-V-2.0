import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  ChevronDown,
  LogOut,
  Settings,
  Home,
  Activity,
  Clock,
  HelpCircle
} from 'lucide-react';
import { NotificationBell } from '@/components/notifications/NotificationBell';
import { useSupabaseAuth } from '../../hooks/useSupabaseAuth';

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, signOut } = useSupabaseAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Handle System Status click
  const handleSystemStatusClick = () => {
    if (location.pathname === '/dashboard') {
      // If already on dashboard, emit custom event to open modal
      window.dispatchEvent(new CustomEvent('openSystemStatus'));
    } else {
      // Navigate to dashboard (will trigger modal via route)
      navigate('/dashboard');
    }
  };

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    await signOut();
    window.location.href = '/login';
  };

  // Get first letter of user's name (from email or metadata)
  const getUserInitial = () => {
    if (!user) return 'U';
    // Use email first character or metadata name
    const name = user.user_metadata?.full_name || user.user_metadata?.name || user.email || 'User';
    return name.charAt(0).toUpperCase();
  };

  // Get display name (from metadata or email)
  const getDisplayName = () => {
    if (!user) return 'User';
    return user.user_metadata?.full_name || user.user_metadata?.name || user.email?.split('@')[0] || 'User';
  };

  const navItems = [
    { icon: Home, label: 'Home', path: '/dashboard' },
    { icon: Activity, label: 'System Status', path: '/system-status' },
    { icon: Clock, label: 'History', path: '/history' },
    { icon: Settings, label: 'Settings', path: '/settings' },
    { icon: HelpCircle, label: 'Help', path: '/help' },
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
                onClick={() => item.label === 'System Status' ? handleSystemStatusClick() : navigate(item.path)}
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
          
          {/* User Profile Dropdown */}
          <div className="relative" ref={menuRef}>
            <button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center gap-2 hover:opacity-80 transition-opacity"
            >
              {/* User Avatar */}
              <div className="w-9 h-9 rounded-full bg-[#00BFFF] flex items-center justify-center text-white font-bold text-[14px] shadow-lg shadow-cyan-500/30 cursor-pointer hover:shadow-cyan-500/50 transition-shadow">
                {getUserInitial()}
              </div>
              <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
            </button>

            {/* Dropdown Menu */}
            {showUserMenu && (
              <div className="absolute right-0 mt-2 w-64 bg-[#1C2128] border border-white/10 rounded-lg shadow-xl overflow-hidden z-50">
                {/* User Info Section */}
                <div className="px-4 py-3 border-b border-white/10">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-full bg-[#00BFFF] flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-cyan-500/30">
                      {getUserInitial()}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-white font-semibold text-sm truncate">
                        {getDisplayName()}
                      </p>
                      {user?.email && (
                        <p className="text-gray-400 text-xs truncate">
                          {user.email}
                        </p>
                      )}
                      <p className="text-[#00BFFF] text-xs font-medium mt-0.5 capitalize">
                        {user?.role || 'User'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Menu Items */}
                <div className="py-1">
                  <button
                    onClick={() => {
                      setShowUserMenu(false);
                      navigate('/settings');
                    }}
                    className="w-full px-4 py-2.5 flex items-center gap-3 text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
                  >
                    <Settings className="w-4 h-4" />
                    <span className="text-sm">Profile Settings</span>
                  </button>
                  
                  <button
                    onClick={handleLogout}
                    className="w-full px-4 py-2.5 flex items-center gap-3 text-red-400 hover:bg-red-500/10 hover:text-red-300 transition-colors"
                  >
                    <LogOut className="w-4 h-4" />
                    <span className="text-sm">Logout</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
