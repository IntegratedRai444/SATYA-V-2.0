import { Link, useLocation } from "wouter";
import { Menu, Bell, Shield, Sparkles, Scan, Clock, Settings, HelpCircle, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useState, useEffect, useRef } from "react";

interface HeaderProps {
  toggleSidebar: () => void;
}

export default function Header({ toggleSidebar }: HeaderProps) {
  const [location] = useLocation();
  const [hoverItem, setHoverItem] = useState<string | null>(null);
  const [ripple, setRipple] = useState({ active: false, x: 0, y: 0 });
  const [logoHovered, setLogoHovered] = useState(false);
  const headerRef = useRef<HTMLDivElement>(null);
  
  // Generate random floating particles for the logo
  const [particles] = useState(
    Array.from({ length: 15 }, () => ({
      x: Math.random() * 40 - 20, // Position around logo
      y: Math.random() * 40 - 20,
      size: 1 + Math.random() * 2,
      speed: 0.2 + Math.random() * 0.3,
      angle: Math.random() * Math.PI * 2,
      opacity: 0.1 + Math.random() * 0.3,
      delay: Math.random() * 5
    }))
  );
  
  // Create ripple effect when clicking on header
  const handleHeaderClick = (e: React.MouseEvent) => {
    if (!headerRef.current) return;
    
    const rect = headerRef.current.getBoundingClientRect();
    setRipple({
      active: true,
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    
    // Reset ripple after animation
    setTimeout(() => {
      setRipple({ active: false, x: 0, y: 0 });
    }, 1000);
  };

  const navItems = [
    { name: "Home", path: "/", icon: Shield },
    { name: "Scan", path: "/scan", icon: Scan },
    { name: "History", path: "/history", icon: Clock },
    { name: "Settings", path: "/settings", icon: Settings },
    { name: "Help", path: "/help", icon: HelpCircle },
  ];

  return (
    <header 
      ref={headerRef}
      className="bg-card bg-opacity-80 backdrop-blur-sm border-b border-primary/20 px-6 py-3 flex items-center justify-between relative overflow-hidden"
      onClick={handleHeaderClick}
    >
      {/* Interactive background grid */}
      <div className="absolute inset-0 opacity-5 pointer-events-none">
        <div className="w-full h-full" style={{
          backgroundImage: `radial-gradient(circle, rgba(0,200,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '20px 20px'
        }}/>
      </div>
      
      {/* Ripple animation */}
      {ripple.active && (
        <div 
          className="absolute rounded-full bg-primary/10 pointer-events-none"
          style={{
            left: ripple.x,
            top: ripple.y,
            width: '10px',
            height: '10px',
            transform: 'translate(-50%, -50%) scale(0)',
            animation: 'ripple-expand 1s forwards'
          }}
        />
      )}
      
      <div className="flex items-center space-x-4">
        {/* Logo and Brand with 3D hover effect */}
        <div 
          className="flex items-center relative"
          onMouseEnter={() => setLogoHovered(true)}
          onMouseLeave={() => setLogoHovered(false)}
        >
          {/* Logo background with glow */}
          <div 
            className={cn(
              "w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center mr-3 relative overflow-hidden transition-all duration-300 transform-gpu",
              logoHovered ? "shadow-[0_0_15px_rgba(0,200,255,0.5)] scale-110" : ""
            )}
            style={{
              transform: logoHovered ? 'rotate3d(0, 1, 0.2, 15deg)' : 'none',
              transition: 'transform 0.3s ease, box-shadow 0.3s ease, scale 0.3s ease'
            }}
          >
            {/* Logo particles */}
            {logoHovered && particles.map((particle, i) => (
              <div 
                key={i}
                className="absolute rounded-full bg-primary pointer-events-none"
                style={{
                  left: '50%',
                  top: '50%',
                  width: `${particle.size}px`,
                  height: `${particle.size}px`,
                  opacity: particle.opacity,
                  transform: `translate(-50%, -50%) 
                    translate(
                      ${particle.x + Math.sin(Date.now() * 0.001 * particle.speed + particle.delay) * 5}px, 
                      ${particle.y + Math.cos(Date.now() * 0.001 * particle.speed + particle.delay) * 5}px
                    )`,
                  animation: `pulse 2s infinite ${particle.delay}s`
                }}
              />
            ))}
            
            {/* Shine effect */}
            {logoHovered && (
              <div className="absolute inset-0 opacity-100 pointer-events-none overflow-hidden">
                <div className="absolute -inset-[100%] animate-shine bg-gradient-to-r from-transparent via-white/30 to-transparent" />
              </div>
            )}
            
            <span className={cn(
              "text-primary text-2xl font-bold transition-all duration-300",
              logoHovered ? "text-white" : ""
            )}>S</span>
          </div>
          
          <h1 
            className="text-2xl font-bold text-foreground font-poppins relative"
            style={{
              textShadow: logoHovered ? '0 0 10px rgba(255, 255, 255, 0.2)' : 'none',
              transition: 'text-shadow 0.3s ease'
            }}
          >
            Satya
            <span className={cn(
              "text-primary relative",
              logoHovered ? "animate-pulse" : ""
            )}>
              AI
              {logoHovered && (
                <Sparkles 
                  size={14} 
                  className="absolute -right-4 -top-2 text-primary animate-pulse" 
                />
              )}
            </span>
          </h1>
        </div>
        
        <p className="text-muted-foreground text-sm hidden md:block">
          Synthetic Authentication Technology for Your Analysis
        </p>
      </div>

      {/* Navigation with hover effects */}
      <nav className="hidden md:flex relative z-10">
        <ul className="flex space-x-6">
          {navItems.map((item) => (
            <li key={item.path} className="relative">
              <Link href={item.path}>
                <a
                  className={cn(
                    "font-poppins transition-all duration-300 py-1 px-2 rounded-md flex items-center gap-1.5 relative overflow-hidden",
                    location === item.path
                      ? "text-primary font-medium bg-primary/10" 
                      : "text-muted-foreground hover:text-foreground",
                    hoverItem === item.path && location !== item.path ? "bg-card-foreground/5" : ""
                  )}
                  onMouseEnter={() => setHoverItem(item.path)}
                  onMouseLeave={() => setHoverItem(null)}
                >
                  <item.icon 
                    size={14} 
                    className={cn(
                      "transition-transform duration-300",
                      hoverItem === item.path || location === item.path ? "scale-125" : "scale-100"
                    )} 
                  />
                  <span>{item.name}</span>
                  
                  {/* Active indicator line with animation */}
                  {location === item.path && (
                    <span className="absolute bottom-0 left-0 h-0.5 bg-primary rounded-full w-full transform-gpu" style={{
                      background: 'linear-gradient(to right, transparent, rgba(0,200,255,0.8), transparent)',
                      animation: 'glow 1.5s infinite'
                    }}></span>
                  )}
                  
                  {/* Hover indicator with shine effect */}
                  {hoverItem === item.path && location !== item.path && (
                    <div className="absolute inset-0 opacity-100 pointer-events-none overflow-hidden">
                      <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                    </div>
                  )}
                </a>
              </Link>
            </li>
          ))}
        </ul>
      </nav>

      {/* Profile and Mobile Menu with animations */}
      <div className="flex items-center">
        <Button 
          variant="ghost" 
          size="icon" 
          className="rounded-full mr-2 relative overflow-hidden group"
        >
          {/* Notification indicator */}
          <div className="absolute top-1 right-1 w-2 h-2 rounded-full bg-primary animate-pulse"></div>
          
          {/* Shine effect on hover */}
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
            <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/10 to-transparent" />
          </div>
          
          <Bell className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
        </Button>
        
        <div 
          className="w-9 h-9 rounded-full bg-primary/20 flex items-center justify-center cursor-pointer group relative overflow-hidden hover:bg-primary/30 transition-colors"
        >
          {/* Profile glow effect */}
          <div 
            className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
            style={{
              background: 'radial-gradient(circle at center, rgba(0, 200, 255, 0.3) 0%, transparent 70%)'
            }}
          />
          
          {/* Shine effect */}
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
            <div className="absolute -inset-[100%] animate-shine bg-gradient-to-r from-transparent via-white/30 to-transparent" />
          </div>
          
          <span className="text-primary font-bold group-hover:text-white transition-colors">U</span>
        </div>
        
        <Button
          variant="ghost"
          size="icon"
          className="md:hidden ml-4 relative group"
          onClick={toggleSidebar}
        >
          {/* Button glow on hover */}
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none rounded-md" style={{
            background: 'radial-gradient(circle at center, rgba(0, 200, 255, 0.2) 0%, transparent 70%)'
          }}></div>
          
          <Menu className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
        </Button>
      </div>
      
      {/* Add CSS for custom animations */}
      <style jsx>{`
        @keyframes ripple-expand {
          to {
            transform: translate(-50%, -50%) scale(100);
            opacity: 0;
          }
        }
        
        @keyframes glow {
          0%, 100% { opacity: 0.5; filter: blur(1px); }
          50% { opacity: 1; filter: blur(0px); }
        }
      `}</style>
    </header>
  );
}
