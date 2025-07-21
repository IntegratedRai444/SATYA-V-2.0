<<<<<<< HEAD
import React from 'react';
import { cn } from "../../lib/utils";
import { Image as ImageIcon, Video, Mic, Camera, ArrowRight, Check, CircleAlert, FileWarning, Sparkles, Lock } from "lucide-react";
import { useNavigation } from "../../hooks/useNavigation";
=======
import { cn } from "@/lib/utils";
import { Image as ImageIcon, Video, Mic, Camera, ArrowRight, Check, CircleAlert, FileWarning, Sparkles, Lock } from "lucide-react";
import { useNavigation } from "@/hooks/useNavigation";
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
import { useState, useRef } from "react";

export default function QuickAccessTiles() {
  const { navigate } = useNavigation();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [rotations, setRotations] = useState<{x: number, y: number}[]>([
    {x: 0, y: 0}, {x: 0, y: 0}, {x: 0, y: 0}, {x: 0, y: 0}
  ]);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);

  const mediaTypes = [
    {
      title: "Image Analysis",
      description: "Detect manipulated photos & generated images",
      icon: ImageIcon,
      colorClass: "text-blue-400 bg-blue-400/10 group-hover:bg-blue-400/20",
      hoverGradient: "from-blue-500/20 to-blue-700/5",
      activeGradient: "from-blue-600/20 to-blue-700/10",
      borderColor: "border-blue-400/30",
      path: "/scan?type=image",
      accuracy: "98.2%",
      glowColor: "0, 122, 255", // RGB for blue glow
      stars: Array.from({ length: 4 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: 1 + Math.random() * 1.5,
        delay: Math.random() * 3
      })),
      featuredStats: [
        { label: "Photoshop Detection", icon: Check },
        { label: "GAN Detection", icon: Check },
        { label: "Metadata Analysis", icon: Check }
      ]
    },
    {
      title: "Video Verification",
      description: "Identify deepfake videos & facial manipulations",
      icon: Video,
      colorClass: "text-emerald-400 bg-emerald-400/10 group-hover:bg-emerald-400/20",
      hoverGradient: "from-emerald-500/20 to-emerald-700/5",
      activeGradient: "from-emerald-600/20 to-emerald-700/10",
      borderColor: "border-emerald-400/30",
      path: "/scan?type=video",
      accuracy: "96.8%",
      glowColor: "16, 185, 129", // RGB for emerald glow
      stars: Array.from({ length: 3 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: 1 + Math.random() * 1.5,
        delay: Math.random() * 3
      })),
      featuredStats: [
        { label: "Facial Inconsistencies", icon: Check },
        { label: "Temporal Analysis", icon: Check },
        { label: "Lip-Sync Verification", icon: Check }
      ]
    },
    {
      title: "Audio Detection",
      description: "Uncover voice cloning & synthetic speech",
      icon: Mic,
      colorClass: "text-purple-400 bg-purple-400/10 group-hover:bg-purple-400/20",
      hoverGradient: "from-purple-500/20 to-purple-700/5",
      activeGradient: "from-purple-600/20 to-purple-700/10",
      borderColor: "border-purple-400/30",
      path: "/scan?type=audio",
      accuracy: "95.3%",
      glowColor: "168, 85, 247", // RGB for purple glow
      stars: Array.from({ length: 5 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: 1 + Math.random() * 1.5,
        delay: Math.random() * 3
      })),
      featuredStats: [
        { label: "Voice Cloning Detection", icon: Check },
        { label: "Natural Patterns Analysis", icon: Check },
        { label: "Neural Voice Filter", icon: CircleAlert }
      ]
    },
    {
      title: "Live Webcam",
      description: "Real-time deepfake analysis & verification",
      icon: Camera,
      colorClass: "text-rose-400 bg-rose-400/10 group-hover:bg-rose-400/20",
      hoverGradient: "from-rose-500/20 to-rose-700/5",
      activeGradient: "from-rose-600/20 to-rose-700/10",
      borderColor: "border-rose-400/30",
      path: "/scan?type=webcam",
      accuracy: "92.7%",
      glowColor: "244, 63, 94", // RGB for rose glow
      stars: Array.from({ length: 4 }, () => ({
        x: Math.random() * 100,
        y: Math.random() * 100,
        size: 1 + Math.random() * 1.5,
        delay: Math.random() * 3
      })),
      featuredStats: [
        { label: "Live Deepfake Alert", icon: Check },
        { label: "Facial Authentication", icon: Check },
        { label: "Low-Light Analysis", icon: FileWarning }
      ]
    }
  ];

  const handleMouseMove = (e: React.MouseEvent, index: number) => {
    if (!cardRefs.current[index]) return;
    
    const card = cardRefs.current[index];
    const rect = card!.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Calculate distance from center (0 to 1)
    const distanceX = (e.clientX - centerX) / (rect.width / 2);
    const distanceY = (e.clientY - centerY) / (rect.height / 2);
    
    // Set rotation (max 8 degrees)
    const newRotations = [...rotations];
    newRotations[index] = {
      x: -distanceY * 8,
      y: distanceX * 8
    };
    setRotations(newRotations);
  };

  const handleMouseLeave = (index: number) => {
    setHoveredIndex(null);
    
    // Reset rotation
    const newRotations = [...rotations];
    newRotations[index] = { x: 0, y: 0 };
    setRotations(newRotations);
  };

  return (
    <div className="mb-12">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div className="relative">
          <h2 className="text-2xl font-bold mb-1 text-foreground relative inline-flex items-center">
            Deepfake Detection Tools
            <div className="ml-2 w-1 h-1 rounded-full bg-primary animate-pulse"></div>
            <div className="ml-1 w-1.5 h-1.5 rounded-full bg-primary animate-pulse" style={{ animationDelay: '0.5s' }}></div>
            <div className="ml-1 w-1 h-1 rounded-full bg-primary animate-pulse" style={{ animationDelay: '1s' }}></div>
          </h2>
          <p className="text-muted-foreground">Choose your media type for comprehensive analysis</p>
        </div>
        <div className="bg-card/60 rounded-lg border border-border/50 px-3 py-2 text-sm text-muted-foreground mt-3 md:mt-0 relative overflow-hidden group/badge">
          <div className="absolute -inset-[100%] group-hover/badge:animate-shine-slow bg-gradient-to-r from-transparent via-white/5 to-transparent"></div>
          Using <span className="font-semibold text-primary inline-flex items-center">
            Neural Vision v4.2
            <Sparkles size={10} className="ml-1 text-primary animate-pulse opacity-70" />
          </span> models
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 lg:gap-6">
        {mediaTypes.map((type, index) => (
          <div 
            key={type.title}
            ref={el => cardRefs.current[index] = el}
            className={cn(
              "bg-gradient-to-br rounded-xl border transition-all duration-300 cursor-pointer group relative overflow-hidden h-full transform-gpu will-change-transform",
              hoveredIndex === index ? `${type.hoverGradient} shadow-lg ${type.borderColor}` : 
              "from-transparent to-transparent shadow border-border/50"
            )}
            style={{
              transform: hoveredIndex === index 
                ? `perspective(1000px) rotateX(${rotations[index].x}deg) rotateY(${rotations[index].y}deg) scale(1.02)` 
                : 'perspective(1000px) rotateX(0deg) rotateY(0deg)',
              transformStyle: 'preserve-3d',
              boxShadow: hoveredIndex === index 
                ? `0 10px 30px -10px rgba(${type.glowColor}, 0.2), 0 0 10px rgba(${type.glowColor}, 0.1)` 
                : 'none'
            }}
            onClick={() => navigate(type.path)}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseMove={(e) => handleMouseMove(e, index)}
            onMouseLeave={() => handleMouseLeave(index)}
          >
            {/* Glow effect on hover */}
            {hoveredIndex === index && (
              <div 
                className="absolute w-full h-full opacity-20 pointer-events-none blur-[80px] rounded-full transform-gpu transition-opacity"
                style={{
                  background: `radial-gradient(circle at center, rgba(${type.glowColor}, 0.8) 0%, transparent 70%)`,
                  top: '-50%',
                  left: '-50%',
                  width: '200%',
                  height: '200%'
                }}
              />
            )}
            
            {/* Animated stars */}
            {hoveredIndex === index && type.stars.map((star, i) => (
              <div 
                key={i}
                className="absolute rounded-full bg-white transform-gpu pointer-events-none animate-twinkle"
                style={{
                  left: `${star.x}%`,
                  top: `${star.y}%`,
                  width: `${star.size}px`,
                  height: `${star.size}px`,
                  opacity: 0.6,
                  animationDelay: `${star.delay}s`,
                  zIndex: 1
                }}
              />
            ))}
            
            {/* Decorative elements */}
            <div 
              className="absolute top-0 right-0 w-32 h-32 opacity-5 -translate-y-16 translate-x-16 transform-gpu transition-transform duration-700"
              style={{
                transform: hoveredIndex === index 
                  ? 'translateY(-12px) translateX(12px) rotate(15deg)' 
                  : 'translateY(-16px) translateX(16px) rotate(0deg)'
              }}
            >
              <div className="w-full h-full rounded-full border-4 border-current"></div>
            </div>
            
            {/* 3D Content */}
            <div className="p-5 flex flex-col h-full relative transform-gpu" style={{ transformStyle: 'preserve-3d' }}>
              <div 
                className="flex justify-between items-start mb-3"
                style={{ 
                  transform: hoveredIndex === index ? 'translateZ(20px)' : 'none',
                  transition: 'transform 300ms ease-out' 
                }}
              >
                <div className={cn(
                  "w-12 h-12 rounded-lg flex items-center justify-center transition-colors transform-gpu will-change-transform",
                  type.colorClass
                )}
                style={{ 
                  transform: hoveredIndex === index ? 'scale(1.1)' : 'scale(1)',
                  transition: 'transform 300ms ease-out, background-color 300ms ease'
                }}
                >
                  <type.icon 
                    size={22} 
                    className={hoveredIndex === index ? "animate-float" : ""} 
                  />
                </div>
                
                <div className={cn(
                  "text-xs font-medium px-2 py-1 rounded-full transition-colors relative overflow-hidden",
                  hoveredIndex === index ? type.colorClass : "bg-muted/50 text-muted-foreground"
                )}
                style={{ 
                  boxShadow: hoveredIndex === index ? `0 0 10px rgba(${type.glowColor}, 0.3)` : 'none'
                }}
                >
                  {/* Subtle shine animation */}
                  {hoveredIndex === index && (
                    <div className="absolute inset-0 opacity-100 pointer-events-none">
                      <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                    </div>
                  )}
                  <div className="flex items-center gap-1">
                    <Lock size={8} className={hoveredIndex === index ? "opacity-100" : "opacity-0"} />
                    <span>Accuracy: {type.accuracy}</span>
                  </div>
                </div>
              </div>
              
              <h3 
                className="font-bold text-lg mb-1 text-foreground group-hover:text-foreground/90"
                style={{ 
                  transform: hoveredIndex === index ? 'translateZ(30px)' : 'none',
                  transition: 'transform 300ms ease-out',
                  textShadow: hoveredIndex === index ? '0 0 15px rgba(255,255,255,0.1)' : 'none'
                }}
              >
                {type.title}
              </h3>
              
              <p 
                className="text-sm text-muted-foreground mb-4"
                style={{ 
                  transform: hoveredIndex === index ? 'translateZ(20px)' : 'none',
                  transition: 'transform 300ms ease-out' 
                }}
              >
                {type.description}
              </p>
              
              <div 
                className="mt-auto"
                style={{ 
                  transform: hoveredIndex === index ? 'translateZ(25px)' : 'none',
                  transition: 'transform 300ms ease-out' 
                }}
              >
                <div className="space-y-1.5 mb-4">
                  {type.featuredStats.map((stat, statIndex) => (
                    <div 
                      key={statIndex} 
                      className="flex items-center gap-2 text-xs text-muted-foreground"
                      style={{ 
                        transform: hoveredIndex === index ? `translateZ(${30 + statIndex * 5}px)` : 'none',
                        transition: `transform 300ms ease-out ${statIndex * 0.1}s` 
                      }}
                    >
                      <stat.icon 
                        size={12} 
                        className={cn(
                          stat.icon === Check ? "text-green-500" : 
                          stat.icon === FileWarning ? "text-amber-500" : 
                          "text-slate-500",
                          hoveredIndex === index ? "animate-pulse" : ""
                        )} 
                      />
                      <span className={hoveredIndex === index ? "text-white/90" : ""}>{stat.label}</span>
                    </div>
                  ))}
                </div>
                
                <div className={cn(
                  "flex items-center text-xs font-medium transition-all",
                  hoveredIndex === index ? type.colorClass : "text-muted-foreground"
                )}
                style={{ 
                  transform: hoveredIndex === index ? 'translateZ(40px) scale(1.05)' : 'scale(1)',
                  transformOrigin: 'left',
                  transition: 'transform 300ms ease-out, color 300ms ease' 
                }}
                >
                  <span>START ANALYSIS</span>
                  <ArrowRight 
                    size={12} 
                    className={cn(
                      "ml-1 transition-transform",
                      hoveredIndex === index ? "translate-x-1" : ""
                    )} 
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
