import { useState } from 'react';
import { Outlet, Navigate } from 'react-router-dom';
import { Toaster } from '@/components/ui/toaster';
import Navbar from './Navbar';
import Sidebar from './Sidebar';
import Footer from './Footer';
import { useSupabaseAuth } from '@/hooks/useSupabaseAuth';
import { Loader2 } from 'lucide-react';
import { ChatProvider, useChat } from '@/contexts/ChatContext';
import ChatOverlay from '@/components/chat/ChatOverlay';
import FloatingChatButton from '@/components/chat/FloatingChatButton';

const AppLayoutContent = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const { user, loading } = useSupabaseAuth();
  const { isChatOpen, closeChat, initialPrompt } = useChat();

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

const AppLayout = () => {
  return (
    <ChatProvider>
      <AppLayoutContent />
    </ChatProvider>
  );
};

export default AppLayout;