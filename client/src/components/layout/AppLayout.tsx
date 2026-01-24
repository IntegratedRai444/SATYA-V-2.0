import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import Footer from './Footer';

const AppLayout = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  return (
    <div className="h-screen bg-black text-white flex flex-col overflow-hidden">
      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area with Sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar */}
        <Sidebar isOpen={isSidebarOpen} onToggle={toggleSidebar} />
        
        {/* Main Content */}
        <main className="flex-1 overflow-y-auto bg-gray-950">
          <div className="p-6">
            <Outlet />
          </div>
        </main>
      </div>
      
      {/* Footer */}
      <Footer />
      
      {/* Global Toaster */}
      <Toaster />
    </div>
  );
};

export default AppLayout;