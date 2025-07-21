<<<<<<< HEAD
import React from 'react';
import { UploadCloud, Info, Shield, ArrowRight, FileVideo, Camera, Headphones, Sparkles, Lock, Eye } from "lucide-react";
import { Button } from "../ui/button";
import { useNavigation } from "../../hooks/useNavigation";
import { useState, useEffect, useRef } from "react";
import { cn } from "../../lib/utils";
=======
import { UploadCloud, Info, Shield, ArrowRight, FileVideo, Camera, Headphones, Sparkles, Lock, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigation } from "@/hooks/useNavigation";
import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f

export default function WelcomeBanner() {
  const { navigate } = useNavigation();
  const [glowPosition, setGlowPosition] = useState({ x: 0, y: 0 });
  const [confidence, setConfidence] = useState(78);
  const [animationPhase, setAnimationPhase] = useState(0);
  const [isHovering, setIsHovering] = useState(false);
  const [stars, setStars] = useState<{ x: number, y: number, size: number, opacity: number, delay: number }[]>([]);
  const bannerRef = useRef<HTMLDivElement>(null);
  
  // Generate stars on first render
  useEffect(() => {
    const newStars = Array.from({ length: 20 }, () => ({
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: 1 + Math.random() * 2,
      opacity: 0.1 + Math.random() * 0.6,
      delay: Math.random() * 3
    }));
    setStars(newStars);
  }, []);
  
  // Animate the confidence score
  useEffect(() => {
    const interval = setInterval(() => {
      setConfidence(prev => {
        const newValue = prev + (Math.random() > 0.5 ? 1 : -1);
        return Math.min(Math.max(newValue, 75), 98); // Keep between 75-98%
      });
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Cycle through animation phases
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimationPhase(prev => (prev + 1) % 3);
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);

  const handleUploadClick = () => {
    navigate("/scan");
  };
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!bannerRef.current) return;
    
    const rect = bannerRef.current.getBoundingClientRect();
    
    // Calculate normalized position (0 to 1) for 3D effect
    const normalizedX = (e.clientX - rect.left) / rect.width;
    const normalizedY = (e.clientY - rect.top) / rect.height;
    
    // Update tilt effect on the banner (subtle 3D rotation)
    if (bannerRef.current) {
      bannerRef.current.style.transform = `perspective(1000px) rotateX(${(normalizedY - 0.5) * 2}deg) rotateY(${(normalizedX - 0.5) * -2}deg)`;
    }
    
    // Update glow position
    setGlowPosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };
  
  const handleMouseEnter = () => setIsHovering(true);
  const handleMouseLeave = () => {
    setIsHovering(false);
    if (bannerRef.current) {
      bannerRef.current.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg)';
    }
  };

  return (
    <div 
      ref={bannerRef}
      className="mb-8 bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl p-8 relative overflow-hidden border border-primary/10 transform-gpu transition-transform duration-300"
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{ transformStyle: 'preserve-3d' }}
    >
      {/* Interactive glow effect */}
      <div 
        className="absolute w-[150px] h-[150px] rounded-full blur-[80px] bg-primary/30 pointer-events-none transition-all duration-200"
        style={{ 
          left: `${glowPosition.x - 75}px`, 
          top: `${glowPosition.y - 75}px`,
          opacity: isHovering ? 0.8 : 0.5
        }}
      />
      
      {/* Digital grid pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="w-full h-full" style={{
          backgroundImage: `radial-gradient(circle, rgba(0,200,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '20px 20px'
        }}/>
      </div>
      
      {/* Animated stars */}
      {stars.map((star, i) => (
        <div 
          key={i}
          className="absolute rounded-full bg-white pointer-events-none animate-twinkle"
          style={{
            left: `${star.x}%`,
            top: `${star.y}%`,
            width: `${star.size}px`,
            height: `${star.size}px`,
            opacity: star.opacity,
            animationDelay: `${star.delay}s`
          }}
        />
      ))}
      
      {/* 3D pop-up alert badge */}
      <div 
        className="absolute top-6 left-6 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-xs py-2 px-3 shadow-xl transform-gpu transition-all duration-300 group cursor-pointer hidden md:flex items-center gap-2"
        style={{ 
          transform: `perspective(1000px) translateZ(${isHovering ? 20 : 0}px)`,
          transformStyle: 'preserve-3d'
        }}
      >
        <Sparkles size={12} className="animate-pulse" />
        <span>New AI Models Released</span>
        <div className="absolute -bottom-1 left-3 w-2 h-2 bg-indigo-600 rotate-45"></div>
      </div>
      
      {/* Animated elements with 3D effect */}
      <div 
        className="absolute top-8 right-8 opacity-80 hidden lg:block"
        style={{ 
          transform: `perspective(1000px) translateZ(${isHovering ? 40 : 0}px)`,
          transformStyle: 'preserve-3d',
          transition: 'transform 300ms ease'
        }}
      >
        <div className="relative h-32 w-32">
          <div className={`absolute inset-0 rounded-lg border border-primary/40 flex items-center justify-center transition-all duration-1000 ${animationPhase === 0 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
            <FileVideo size={28} className="text-primary" />
          </div>
          <div className={`absolute inset-0 rounded-lg border border-primary/40 flex items-center justify-center transition-all duration-1000 ${animationPhase === 1 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
            <Camera size={28} className="text-primary" />
          </div>
          <div className={`absolute inset-0 rounded-lg border border-primary/40 flex items-center justify-center transition-all duration-1000 ${animationPhase === 2 ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`}>
            <Headphones size={28} className="text-primary" />
          </div>
        </div>
        <div 
          className="mt-4 bg-primary/10 rounded-lg p-2 text-center transform-gpu hover:scale-105 transition-transform cursor-pointer"
          style={{ boxShadow: '0 0 15px rgba(0,200,255,0.2)' }}
        >
          <div className="text-xs text-primary/80">AUTHENTICITY SCORE</div>
          <div className="text-xl font-bold text-primary">{confidence}%</div>
        </div>
      </div>
      
      {/* 3D floating security indicators */}
      <div className="absolute right-10 bottom-10 space-y-3 hidden xl:block">
        {[
          { icon: Eye, label: "Real-time Analysis", delay: 0 },
          { icon: Lock, label: "Secure Processing", delay: 0.5 },
          { icon: Shield, label: "Verified Protection", delay: 1 }
        ].map((item, index) => (
          <div 
            key={index}
            className="flex items-center gap-2 text-xs text-slate-300/80 transition-all duration-500 group"
            style={{ 
              transform: isHovering ? `perspective(1000px) translateZ(${20 + index * 10}px)` : 'none',
              transformStyle: 'preserve-3d',
              transitionDelay: `${item.delay}s`
            }}
          >
            <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
              <item.icon size={12} className="text-primary" />
            </div>
            <span className="group-hover:text-white transition-colors">{item.label}</span>
          </div>
        ))}
      </div>
      
      <div 
        className="relative z-10 max-w-3xl"
        style={{ 
          transform: `perspective(1000px) translateZ(${isHovering ? 15 : 0}px)`,
          transformStyle: 'preserve-3d',
          transition: 'transform 400ms ease'
        }}
      >
        <div 
          className="inline-flex items-center gap-2 bg-primary/20 py-1 px-3 rounded-full text-sm text-primary mb-4 hover:bg-primary/30 transition-colors"
          style={{ 
            boxShadow: isHovering ? '0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -4px rgba(0,0,0,0.1), 0 0 0 1px rgba(0,200,255,0.2)' : 'none',
            transition: 'box-shadow 300ms ease'
          }}
        >
          <Shield size={14} className="animate-pulse" />
          <span>Advanced AI-powered Protection</span>
        </div>
        
        <h1 
          className="text-4xl md:text-5xl font-bold font-poppins text-white mb-4 leading-tight"
          style={{ textShadow: isHovering ? '0 0 20px rgba(255,255,255,0.15)' : 'none' }}
        >
          Detect <span className="text-primary relative group">
            deepfakes
            <span className={cn(
              "absolute bottom-1 left-0 w-full h-1 bg-primary/30 rounded-full",
              isHovering ? "animate-pulse" : ""
            )}></span>
            
            {/* Pop-up 3D explanation on hover */}
            <div className="absolute opacity-0 group-hover:opacity-100 top-0 -mt-16 left-1/2 transform -translate-x-1/2 transition-opacity duration-300 pointer-events-none">
              <div className="bg-slate-800 text-white text-xs p-2 rounded shadow-xl border border-primary/30 w-60">
                <div className="font-medium text-primary mb-1">What are deepfakes?</div>
                <div className="text-slate-300 text-[10px]">AI-generated media that realistically replace a person's likeness with someone else's</div>
                <div className="absolute bottom-0 left-1/2 transform translate-y-1/2 -translate-x-1/2 rotate-45 w-2 h-2 bg-slate-800 border-r border-b border-primary/30"></div>
              </div>
            </div>
          </span> with the power of SatyaAI
        </h1>
        
        <p className="text-lg text-slate-300 max-w-2xl mb-3">
          Our advanced detection system helps you authenticate media with unprecedented accuracy, 
          exposing manipulated content across images, videos, and audio.
        </p>
        
        <p className="text-slate-400 mb-8">
          Upload your files or use your webcam for real-time analysis and get detailed authenticity reports instantly.
        </p>
        
        <div className="flex flex-wrap gap-4">
          <Button 
            className="bg-primary hover:bg-primary/90 text-white shadow-[0_0_20px_rgba(0,200,255,0.4)] flex items-center gap-2 text-base px-6 py-6 hover:shadow-[0_0_30px_rgba(0,200,255,0.6)] transition-shadow relative overflow-hidden group"
            onClick={handleUploadClick}
            style={{ 
              transform: isHovering ? 'scale(1.03)' : 'scale(1)',
              transition: 'transform 300ms ease, box-shadow 300ms ease'
            }}
          >
            {/* Animated shine effect */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
              <div className="absolute -inset-[100%] animate-shine bg-gradient-to-r from-transparent via-white/10 to-transparent" />
            </div>
            
            <UploadCloud size={20} />
            <span>Analyze Media</span>
            <ArrowRight size={16} className="ml-1 animate-pulse" />
          </Button>
          
          <Button 
            variant="outline" 
            className="border-slate-600 text-slate-300 hover:text-white hover:bg-slate-700/50 flex items-center gap-2 text-base py-6 relative overflow-hidden group"
            style={{ 
              transform: isHovering ? 'scale(1.01)' : 'scale(1)',
              transition: 'transform 300ms ease'
            }}
          >
            {/* Subtle shine effect on hover */}
            <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
              <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/5 to-transparent" />
            </div>
            
            <Info size={20} />
            <span>How It Works</span>
          </Button>
        </div>
      </div>

      {/* Animated tech pattern on the right - visible on larger screens */}
      <div 
        className="hidden lg:block absolute bottom-0 right-0 w-60 h-60 opacity-20"
        style={{ 
          transform: isHovering ? 'rotate(15deg) scale(1.1)' : 'rotate(0deg) scale(1)',
          transition: 'transform 700ms ease', 
          transformOrigin: 'bottom right'
        }}
      >
        <div className="w-full h-full relative">
          <div className="absolute inset-0 border-l border-t border-primary/30 rounded-tl-full"></div>
          <div className="absolute inset-0 scale-90 border-l border-t border-primary/20 rounded-tl-full"></div>
          <div className="absolute inset-0 scale-80 border-l border-t border-primary/10 rounded-tl-full"></div>
          <div className="absolute top-1/2 left-1/2 w-4 h-4 -translate-x-1/2 -translate-y-1/2 bg-primary/30 rounded-full animate-pulse"></div>
        </div>
      </div>
    </div>
  );
}
