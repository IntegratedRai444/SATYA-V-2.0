import { Outlet, useLocation } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import Footer from './Footer';
import { useState, useEffect } from 'react';

const MainLayout = () => {
  const location = useLocation();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Don't show layout wrapper on auth pages and landing page
  const isAuthPage = ['/login', '/register', '/forgot-password', '/reset-password'].includes(location.pathname);

  // Load sidebar state from localStorage
  useEffect(() => {
    const savedState = localStorage.getItem('sidebar-open');
    if (savedState !== null) {
      setIsSidebarOpen(savedState === 'true');
    }
  }, []);

  // Save sidebar state to localStorage
  const toggleSidebar = () => {
    setIsSidebarOpen(prev => {
      const newState = !prev;
      localStorage.setItem('sidebar-open', String(newState));
      return newState;
    });
  };

  if (isAuthPage) {
    return (
      <div className="min-h-screen bg-bg-primary">
        <Outlet />
        <Toaster />
      </div>
    );
  }

  // Full layout with Navbar, Sidebar, and Footer for authenticated pages
  return (
    <div className="h-screen bg-[#0a0a0a] flex flex-col overflow-hidden">
      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area with Sidebar */}
      <div className="flex flex-1 pt-16 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar isOpen={isSidebarOpen} onToggle={toggleSidebar} />
        
        {/* Main Content */}
        <main 
          className="flex-1 flex flex-col transition-all duration-300 overflow-y-auto"
          style={{ marginLeft: isSidebarOpen ? '280px' : '0' }}
        >
          <div className="flex-1 p-8">
            <Outlet />
          </div>
          
          {/* Footer */}
          <Footer />
        </main>
      </div>
      
      <Toaster />
    </div>
  );
};

export default MainLayout;
