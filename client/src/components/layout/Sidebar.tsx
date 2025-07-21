import { useState } from "react";
import { Link, useLocation } from "wouter";
import { cn } from "../../lib/utils";
import AIAssistant from "../shared/AIAssistant";
import {
  LayoutDashboard,
  Image,
  Video,
  Mic,
  Camera,
  History,
  Settings,
  HelpCircle,
} from "lucide-react";

interface SidebarProps {
  open: boolean;
  className?: string;
}

export default function Sidebar({ open, className }: SidebarProps) {
  const [location] = useLocation();

  const detectionTools = [
    {
      name: "Dashboard",
      path: "/",
      icon: LayoutDashboard,
    },
    {
      name: "Image Analysis",
      path: "/scan?type=image",
      icon: Image,
    },
    {
      name: "Video Analysis",
      path: "/scan?type=video",
      icon: Video,
    },
    {
      name: "Audio Analysis",
      path: "/scan?type=audio",
      icon: Mic,
    },
    {
      name: "Webcam Live",
      path: "/scan?type=webcam",
      icon: Camera,
    },
  ];

  const managementTools = [
    {
      name: "Scan History",
      path: "/history",
      icon: History,
    },
    {
      name: "Settings",
      path: "/settings",
      icon: Settings,
    },
    {
      name: "Help & Support",
      path: "/help",
      icon: HelpCircle,
    },
  ];

  return (
    <aside
      className={cn(
        "bg-sidebar w-64 border-r border-primary/20 fixed inset-y-0 left-0 transform transition-transform duration-300 ease-in-out z-30 md:translate-x-0 top-[57px] pt-0 md:pt-0 md:top-[57px] overflow-y-auto",
        open ? "translate-x-0" : "-translate-x-full",
        className
      )}
    >
      <div className="p-6">
        <h2 className="text-lg font-poppins font-medium text-foreground mb-4">
          Detection Tools
        </h2>
        <ul className="space-y-2">
          {detectionTools.map((tool) => {
            const isActive = 
              tool.path === "/" 
                ? location === tool.path 
                : location.startsWith(tool.path.split("?")[0]) && 
                  (location.includes(tool.path.split("?")[1]) || !tool.path.includes("?"));
            
            return (
              <li key={tool.path}>
                <Link href={tool.path}>
                  <a
                    className={cn(
                      "flex items-center space-x-3 p-3 rounded-lg transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-muted text-muted-foreground"
                    )}
                  >
                    <tool.icon className="text-xl" size={20} />
                    <span className={cn(isActive && "font-medium")}>
                      {tool.name}
                    </span>
                  </a>
                </Link>
              </li>
            );
          })}
        </ul>

        <h2 className="text-lg font-poppins font-medium text-foreground mt-8 mb-4">
          Management
        </h2>
        <ul className="space-y-2">
          {managementTools.map((tool) => {
            const isActive = location === tool.path;
            
            return (
              <li key={tool.path}>
                <Link href={tool.path}>
                  <a
                    className={cn(
                      "flex items-center space-x-3 p-3 rounded-lg transition-colors",
                      isActive
                        ? "bg-primary/10 text-primary"
                        : "hover:bg-muted text-muted-foreground"
                    )}
                  >
                    <tool.icon className="text-xl" size={20} />
                    <span className={cn(isActive && "font-medium")}>
                      {tool.name}
                    </span>
                  </a>
                </Link>
              </li>
            );
          })}
        </ul>
      </div>

      {/* AI Assistant Widget */}
      <div className="px-6 mt-4 pb-6">
        <AIAssistant />
      </div>
    </aside>
  );
}
