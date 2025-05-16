import { useState, useEffect } from "react";
import Header from "./Header";
import Sidebar from "./Sidebar";
import Footer from "./Footer";
import { useLocation } from "wouter";
import { cn } from "@/lib/utils";

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [location] = useLocation();

  // Close sidebar when changing routes on mobile
  useEffect(() => {
    setSidebarOpen(false);
  }, [location]);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header toggleSidebar={() => setSidebarOpen(!sidebarOpen)} />

      <div className="flex flex-1 overflow-hidden relative pt-[57px]">
        <Sidebar open={sidebarOpen} />

        {/* Overlay that appears when sidebar is open on mobile */}
        {sidebarOpen && (
          <div 
            className="fixed inset-0 bg-black/50 z-20 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <main 
          className={cn(
            "flex-1 overflow-y-auto transition-all duration-300 pt-6 px-6 pb-6",
            "md:pl-[272px]" // 16px padding + 256px sidebar
          )}
        >
          {children}
        </main>
      </div>

      <div className={cn("md:pl-64")}>
        <Footer />
      </div>
    </div>
  );
}
