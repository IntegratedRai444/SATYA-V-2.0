import { Outlet, useLocation } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import Footer from './Footer';

const MainLayout = () => {
  const location = useLocation();

  // Don't show layout wrapper on auth pages and landing page
  const isAuthPage = ['/login', '/register', '/forgot-password', '/reset-password'].includes(location.pathname);

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
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area with Sidebar */}
      <div className="flex pt-16">
        {/* Left Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <main className="flex-1 ml-[280px] min-h-[calc(100vh-4rem)]">
          <div className="p-8">
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
