import { Outlet, useLocation } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';

const MainLayout = () => {
  const location = useLocation();

  // Don't show layout wrapper on auth pages and landing page
  const isAuthPage = ['/login', '/register', '/forgot-password', '/reset-password'].includes(location.pathname);

  if (isAuthPage) {
    return (
      <div className="min-h-screen bg-black">
        <Outlet />
        <Toaster />
      </div>
    );
  }

  // Full layout with Navbar, Sidebar, and Footer for authenticated pages
  return (
    <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
      {/* Top Navbar */}
      <div className="bg-gray-900 border-b border-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="text-xl font-bold text-white">SatyaAI Dashboard</div>
        <div className="text-sm text-gray-300">Logged in as: {location.pathname}</div>
      </div>
      
      {/* Main Content Area with Sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <div className="w-64 bg-gray-900 border-r border-gray-800 p-4">
          <div className="text-white mb-4">Navigation</div>
          <div className="space-y-2">
            <div className="text-gray-300 hover:text-white cursor-pointer p-2 rounded">Dashboard</div>
            <div className="text-gray-300 hover:text-white cursor-pointer p-2 rounded">Analysis</div>
            <div className="text-gray-300 hover:text-white cursor-pointer p-2 rounded">History</div>
            <div className="text-gray-300 hover:text-white cursor-pointer p-2 rounded">Settings</div>
          </div>
        </div>
        
        {/* Main Content */}
        <div className="flex-1 bg-black p-6 overflow-auto">
          <div className="text-white text-2xl font-bold mb-4">Welcome to SatyaAI Dashboard</div>
          <div className="text-gray-300 mb-4">This is a simplified version to test rendering.</div>
          <Outlet />
        </div>
      </div>
      
      <Toaster />
    </div>
  );
};

export default MainLayout;
