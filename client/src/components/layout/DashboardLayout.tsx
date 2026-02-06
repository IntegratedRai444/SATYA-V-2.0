import React, { useState, useEffect } from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';
import { DashboardBackground } from '@/components/ui/background-paths';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import Footer from './Footer';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import { Loader2 } from 'lucide-react';
import { ChatProvider, useChat } from '@/contexts/ChatContext';
import ChatOverlay from '@/components/chat/ChatOverlay';
import FloatingChatButton from '@/components/chat/FloatingChatButton';

interface DashboardLayoutProps {
  children?: React.ReactNode;
}

const DashboardLayoutContent = ({ children }: DashboardLayoutProps) => {
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const { user, loading } = useSupabaseAuth();
  const { isChatOpen, closeChat, initialPrompt } = useChat();

  // Responsive sidebar behavior
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        // Mobile: sidebar collapsed by default
        setIsSidebarExpanded(false);
      } else if (window.innerWidth < 1024) {
        // Tablet: sidebar collapsed by default
        setIsSidebarExpanded(false);
      } else {
        // Desktop: sidebar expanded by default
        setIsSidebarExpanded(true);
      }
    };

    // Set initial state
    handleResize();

    // Listen for resize events
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Show loading spinner while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-4" />
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  const toggleSidebar = () => {
    setIsSidebarExpanded(!isSidebarExpanded);
  };

  return (
    <div className="h-screen bg-black text-white flex flex-col overflow-hidden relative">
      {/* Animated Background Paths */}
      <DashboardBackground />

      {/* Top Navbar */}
      <Navbar />
      
      {/* Main Content Area - Flex layout with sidebar */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Sidebar - Part of flex layout */}
        <Sidebar 
          isExpanded={isSidebarExpanded} 
          onToggle={toggleSidebar} 
        />
        
        {/* Main Content - Takes remaining space */}
        <main className="flex-1 overflow-y-auto overflow-x-hidden bg-gray-950/80 backdrop-blur-sm transition-all duration-300 ease-in-out scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900 hover:scrollbar-thumb-gray-600">
          <div className="p-6 min-h-full">
            {children || <Outlet />}
          </div>
        </main>
      </div>
      
      {/* Footer */}
      <Footer />
      
      {/* Global Toaster */}
      <Toaster />
      
      {/* Chat Overlay */}
      <ChatOverlay 
        isOpen={isChatOpen} 
        onClose={closeChat} 
        initialPrompt={initialPrompt}
      />
      
      {/* Floating Chat Button */}
      <FloatingChatButton />
    </div>
  );
};

const DashboardLayout = (props: DashboardLayoutProps) => {
  return (
    <ChatProvider>
      <DashboardLayoutContent {...props} />
    </ChatProvider>
  );
};

export default DashboardLayout;
