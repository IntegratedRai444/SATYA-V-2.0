import { ArrowUp, ArrowDown, CheckCircle, Info, BookOpen, AlertTriangle, Clock, PieChart, Layers, Shield, Activity, Zap, Sparkles, ExternalLink, LucideIcon, Lock, Eye, Lightbulb } from "lucide-react";
import { cn } from "../../lib/utils";
import { useState, useRef, useEffect } from "react";

export default function InformativeSection() {
  const [activeTab, setActiveTab] = useState<'stats' | 'insights'>('stats');
  const [isHoveringPanel, setIsHoveringPanel] = useState(false);
  const [activeTip, setActiveTip] = useState<number | null>(null);
  const [rotations, setRotations] = useState<{x: number, y: number}>({ x: 0, y: 0 });
  const [particleCount, setParticleCount] = useState(0);
  const panelRef = useRef<HTMLDivElement>(null);
  
  // Generate particles on tab change for a flourish effect
  useEffect(() => {
    setParticleCount(prev => prev + 15); // Add 15 more particles
    
    // Clean up particles after animation completes
    const timer = setTimeout(() => {
      setParticleCount(0);
    }, 3000);
    
    return () => clearTimeout(timer);
  }, [activeTab]);
  
  const handlePanelMouseMove = (e: React.MouseEvent) => {
    if (!panelRef.current || !isHoveringPanel) return;
    
    const rect = panelRef.current.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    // Calculate distance from center (0 to 1)
    const distanceX = (e.clientX - centerX) / (rect.width / 2);
    const distanceY = (e.clientY - centerY) / (rect.height / 2);
    
    // Set rotation (max 5 degrees)
    setRotations({
      x: -distanceY * 3,
      y: distanceX * 3
    });
  };
  
  const handlePanelMouseLeave = () => {
    setIsHoveringPanel(false);
    setRotations({ x: 0, y: 0 });
  };
  
  const stats = [
    {
      label: "Analyzed Media",
      value: "147",
      change: "+23%",
      isPositive: true,
      icon: Layers,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10",
      glowColor: "59, 130, 246", // RGB for blue
      animationDelay: 0
    },
    {
      label: "Detected Deepfakes",
      value: "36",
      change: "+12%",
      isPositive: false,
      isWarning: true,
      icon: AlertTriangle,
      color: "text-rose-500",
      bgColor: "bg-rose-500/10",
      glowColor: "244, 63, 94", // RGB for rose
      animationDelay: 0.1
    },
    {
      label: "Avg. Detection Time",
      value: "4.2s",
      change: "-18%",
      isPositive: true,
      icon: Clock,
      color: "text-amber-500",
      bgColor: "bg-amber-500/10",
      glowColor: "245, 158, 11", // RGB for amber
      animationDelay: 0.2
    },
    {
      label: "Detection Accuracy",
      value: "96%",
      change: "+3%",
      isPositive: true,
      icon: PieChart,
      color: "text-emerald-500",
      bgColor: "bg-emerald-500/10",
      glowColor: "16, 185, 129", // RGB for emerald
      animationDelay: 0.3
    }
  ];

  const insights = [
    {
      title: "AI Detection Trends",
      description: "Our neural networks showed a 12% improvement in detecting GAN-based face manipulations over the last month.",
      icon: Activity,
      color: "text-indigo-500",
      bgColor: "bg-indigo-500/10",
      glowColor: "99, 102, 241", // RGB for indigo
      animationDelay: 0
    },
    {
      title: "Security Analysis",
      description: "Multi-modal detection combining audio, video, and metadata analysis improves accuracy by up to 15%.",
      icon: Shield,
      color: "text-emerald-500",
      bgColor: "bg-emerald-500/10",
      glowColor: "16, 185, 129", // RGB for emerald
      animationDelay: 0.15
    }
  ];

  const tips = [
    {
      text: "Look for unnatural eye blinking patterns and inconsistent eye reflections in suspected videos.",
      icon: Eye,
      color: "text-sky-400",
      animationDelay: 0
    },
    {
      text: "Check for unnatural hair movement, unusual skin texture, or blurry face boundaries in images.",
      icon: Zap,
      color: "text-amber-400",
      animationDelay: 0.1
    },
    {
      text: "Watch for inconsistencies in audio-visual synchronization, especially in speech videos.",
      icon: Activity,
      color: "text-purple-400",
      animationDelay: 0.2
    },
    {
      text: "Analyze visual artifacts around the edges of faces, which often indicate manipulation.",
      icon: Lightbulb,
      color: "text-emerald-400",
      animationDelay: 0.3
    }
  ];
  
  // Generate particles for tab switching animation
  const particles = Array.from({ length: particleCount }, (_, i) => ({
    id: i,
    x: 50 + (Math.random() - 0.5) * 80, // Centered with some spread
    y: 0,
    size: 3 + Math.random() * 5,
    speed: 1 + Math.random() * 3,
    color: activeTab === 'stats' ? 
      `rgba(${stats[Math.floor(Math.random() * stats.length)].glowColor}, ${0.5 + Math.random() * 0.5})` :
      `rgba(${insights[Math.floor(Math.random() * insights.length)].glowColor}, ${0.5 + Math.random() * 0.5})`,
    angle: -Math.PI/2 + (Math.random() - 0.5) * Math.PI/4 // Mostly downward with some spread
  }));

  return (
    <div className="mt-8 mb-12">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div className="relative">
          <h2 className="text-2xl font-bold mb-1 text-foreground flex items-center">
            Analytics & Insights
            <div className="ml-2 relative h-5 w-5">
              <div className="absolute inset-0 bg-primary/20 rounded-full animate-ping"></div>
              <div className="absolute inset-0 bg-primary/30 rounded-full"></div>
              <div className="absolute inset-1 bg-primary/50 rounded-full"></div>
            </div>
          </h2>
          <p className="text-muted-foreground">System performance and detection tips</p>
        </div>
        
        <div className="relative mt-3 md:mt-0">
          {/* Animation particles when switching tabs */}
          {particles.map(particle => (
            <div 
              key={particle.id}
              className="absolute rounded-full"
              style={{
                left: `${particle.x}%`,
                top: `${particle.y}px`,
                width: `${particle.size}px`,
                height: `${particle.size}px`,
                background: particle.color,
                opacity: 0,
                transform: `translateY(0px)`,
                animation: `fadeUpAndOut 2s forwards`,
                zIndex: 10
              }}
            />
          ))}
          
          <div className="flex bg-card rounded-lg border border-border p-1 relative overflow-hidden z-0 shadow-lg">
            {/* Subtle glow around active tab */}
            <div 
              className="absolute rounded-md transition-all duration-300" 
              style={{
                left: activeTab === 'stats' ? '0%' : '50%',
                width: '50%',
                height: '100%',
                background: `radial-gradient(circle at center, rgba(0, 200, 255, 0.15) 0%, transparent 70%)`,
                filter: 'blur(10px)',
                zIndex: -1
              }}
            />
            
            <button 
              className={cn(
                "px-4 py-2 text-sm font-medium rounded-md transition-all duration-300 relative overflow-hidden",
                activeTab === 'stats' 
                  ? "bg-primary text-white shadow-[0_0_10px_rgba(0,200,255,0.3)]" 
                  : "hover:bg-muted text-muted-foreground"
              )}
              onClick={() => setActiveTab('stats')}
            >
              <div className="flex items-center">
                <PieChart size={14} className="mr-1.5" />
                <span>Statistics</span>
              </div>
              
              {/* Shine effect on hover */}
              <div className="absolute inset-0 opacity-0 hover:opacity-100 pointer-events-none">
                <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              </div>
            </button>
            
            <button 
              className={cn(
                "px-4 py-2 text-sm font-medium rounded-md transition-all duration-300 relative overflow-hidden",
                activeTab === 'insights' 
                  ? "bg-primary text-white shadow-[0_0_10px_rgba(0,200,255,0.3)]" 
                  : "hover:bg-muted text-muted-foreground"
              )}
              onClick={() => setActiveTab('insights')}
            >
              <div className="flex items-center">
                <Sparkles size={14} className="mr-1.5" />
                <span>Insights</span>
              </div>
              
              {/* Shine effect on hover */}
              <div className="absolute inset-0 opacity-0 hover:opacity-100 pointer-events-none">
                <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/10 to-transparent" />
              </div>
            </button>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Panel with 3D effect */}
        <div 
          ref={panelRef}
          className="lg:col-span-2 bg-gradient-to-br from-slate-950 to-slate-900 rounded-xl overflow-hidden border border-slate-800 relative transform-gpu transition-transform duration-300"
          style={{
            transform: `perspective(1000px) rotateX(${rotations.x}deg) rotateY(${rotations.y}deg)`,
            transformStyle: 'preserve-3d',
            boxShadow: isHoveringPanel ? '0 25px 50px -12px rgba(0, 0, 0, 0.4)' : 'none'
          }}
          onMouseEnter={() => setIsHoveringPanel(true)}
          onMouseMove={handlePanelMouseMove}
          onMouseLeave={handlePanelMouseLeave}
        >
          {/* Digital grid pattern */}
          <div className="absolute inset-0 opacity-10">
            <div className="w-full h-full" style={{
              backgroundImage: `radial-gradient(circle, rgba(0,200,255,0.05) 1px, transparent 1px)`,
              backgroundSize: '20px 20px'
            }}/>
          </div>
          
          {/* Content based on active tab */}
          <div 
            className="p-6 relative z-10"
            style={{ 
              transform: 'translateZ(0px)',
              transition: 'transform 400ms ease' 
            }}
          >
            {activeTab === 'stats' ? (
              <div className="space-y-6 transform-gpu" style={{ transformStyle: 'preserve-3d' }}>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  {stats.map((stat, index) => (
                    <div 
                      key={index} 
                      className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex items-start relative overflow-hidden group transform-gpu hover:shadow-lg transition-all duration-500"
                      style={{ 
                        transform: isHoveringPanel ? 'translateZ(20px)' : 'translateZ(0)',
                        transformStyle: 'preserve-3d',
                        transition: `transform 500ms ease ${stat.animationDelay}s`,
                        boxShadow: isHoveringPanel ? `0 0 15px rgba(${stat.glowColor}, 0.1)` : 'none'
                      }}
                    >
                      {/* Background decoration */}
                      <div className="absolute -right-12 -bottom-12 w-40 h-40 rounded-full opacity-5 border-8 border-slate-400 group-hover:opacity-10 transition-opacity"></div>
                      
                      {/* Pulsing glow on hover */}
                      <div 
                        className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-700"
                        style={{
                          background: `radial-gradient(circle at center, rgba(${stat.glowColor}, 0.15) 0%, transparent 70%)`,
                          animation: 'pulse 3s infinite'
                        }}
                      />
                      
                      <div 
                        className={cn(
                          "w-12 h-12 rounded-lg flex items-center justify-center mr-4 transition-all group-hover:scale-110 relative overflow-hidden",
                          stat.bgColor
                        )}
                        style={{ 
                          transform: 'translateZ(30px)',
                          transition: 'transform 300ms ease, background-color 300ms ease',
                          boxShadow: `0 0 15px rgba(${stat.glowColor}, 0.2)`
                        }}
                      >
                        {/* Shine effect */}
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
                          <div className="absolute -inset-[100%] animate-shine bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                        </div>
                        
                        <stat.icon className={cn(stat.color, "group-hover:animate-pulse")} size={20} />
                      </div>
                      
                      <div style={{ transform: 'translateZ(25px)' }}>
                        <p className="text-muted-foreground text-sm mb-1 group-hover:text-white/80 transition-colors">{stat.label}</p>
                        <div className="flex items-end gap-2">
                          <span 
                            className={cn(
                              "text-3xl font-bold transition-all duration-700",
                              stat.isWarning ? "text-rose-500" : "text-white"
                            )}
                            style={{
                              textShadow: isHoveringPanel ? `0 0 10px rgba(${stat.glowColor}, 0.4)` : 'none'
                            }}
                          >
                            {stat.value}
                          </span>
                          <span className={cn(
                            "text-sm font-medium flex items-center transition-transform group-hover:scale-110",
                            stat.isPositive && !stat.isWarning ? "text-emerald-500" : "text-rose-500"
                          )}>
                            {stat.change}{" "}
                            {stat.isPositive ? (
                              <ArrowUp size={14} className="ml-1 group-hover:animate-float" />
                            ) : (
                              <ArrowDown size={14} className="ml-1 group-hover:animate-float" />
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                
                <div 
                  className="mt-6 bg-slate-900/50 border border-slate-800 rounded-xl p-5 relative transform-gpu group"
                  style={{ 
                    transform: isHoveringPanel ? 'translateZ(30px)' : 'translateZ(0)',
                    transformStyle: 'preserve-3d',
                    transition: 'transform 500ms ease 0.2s'
                  }}
                >
                  {/* Animated highlight line that moves across chart */}
                  <div className="absolute inset-y-0 w-[1px] bg-primary/40 left-0 opacity-0 group-hover:opacity-100" style={{
                    animation: 'scanRight 5s linear infinite',
                    boxShadow: '0 0 8px rgba(0, 200, 255, 0.6)',
                    zIndex: 5
                  }}></div>
                  
                  <div 
                    className="flex items-center mb-4"
                    style={{ transform: 'translateZ(10px)' }}
                  >
                    <Activity size={18} className="text-primary mr-2 group-hover:animate-pulse" />
                    <h3 className="text-lg font-bold text-white">Detection Activity</h3>
                  </div>
                  
                  <div 
                    className="relative h-32 w-full"
                    style={{ transform: 'translateZ(5px)' }}
                  >
                    {/* Simulated activity chart with HTML/CSS */}
                    <div className="absolute inset-x-0 bottom-0 h-full flex items-end px-2">
                      {Array.from({ length: 24 }).map((_, i) => {
                        const height = 20 + Math.random() * 60;
                        const isHighlighted = i === 16;
                        return (
                          <div 
                            key={i} 
                            className="flex-1 mx-0.5 group/bar cursor-pointer transition-all duration-500"
                            style={{ 
                              height: `${height}%`,
                              transform: isHoveringPanel && isHighlighted ? 'translateZ(15px) scale(1.1)' : 'translateZ(0) scale(1)'
                            }}
                          >
                            <div 
                              className={cn(
                                "w-full h-full rounded-sm transition-all duration-300",
                                isHighlighted 
                                  ? "bg-primary" 
                                  : "bg-primary/30 group-hover/bar:bg-primary/60"
                              )}
                              style={{
                                boxShadow: isHighlighted ? '0 0 10px rgba(0, 200, 255, 0.4)' : 'none'
                              }}
                            />
                            {isHighlighted && (
                              <div 
                                className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-primary text-white text-xs py-1 px-2 rounded shadow-lg z-20 transition-all duration-300"
                                style={{
                                  transform: isHoveringPanel ? 'translate(-50%, -2px) translateZ(20px)' : 'translate(-50%, 0) translateZ(0)',
                                  boxShadow: '0 0 15px rgba(0, 200, 255, 0.3)'
                                }}
                              >
                                <div className="flex items-center">
                                  <AlertTriangle size={10} className="mr-1 animate-pulse" />
                                  <span>16 Alerts</span>
                                </div>
                                <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-2 h-2 bg-primary rotate-45"></div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                    
                    {/* X-axis labels */}
                    <div className="absolute inset-x-0 bottom-0 flex justify-between text-xs text-slate-500 pt-2 border-t border-slate-800">
                      <span>00:00</span>
                      <span>06:00</span>
                      <span>12:00</span>
                      <span>18:00</span>
                      <span>23:59</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-5 transform-gpu" style={{ transformStyle: 'preserve-3d' }}>
                {insights.map((insight, index) => (
                  <div 
                    key={index}
                    className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 mb-5 last:mb-0 relative overflow-hidden group transform-gpu"
                    style={{ 
                      transform: isHoveringPanel ? `translateZ(${20 + index * 5}px)` : 'translateZ(0)',
                      transformStyle: 'preserve-3d',
                      transition: `transform 500ms ease ${insight.animationDelay}s`,
                      boxShadow: isHoveringPanel ? `0 0 15px rgba(${insight.glowColor}, 0.1)` : 'none' 
                    }}
                  >
                    {/* Background decoration */}
                    <div className="absolute top-0 right-0 w-40 h-40 opacity-5 -translate-y-20 translate-x-20 group-hover:opacity-8 transition-opacity">
                      <div className="w-full h-full rounded-full border-4 border-current"></div>
                    </div>
                    
                    {/* Animated highlight */}
                    <div 
                      className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity duration-700"
                      style={{
                        background: `radial-gradient(circle at center, rgba(${insight.glowColor}, 0.1) 0%, transparent 70%)`
                      }}
                    />
                    
                    <div className="flex items-start relative z-10">
                      <div 
                        className={cn(
                          "w-10 h-10 rounded-lg flex items-center justify-center mr-4 relative overflow-hidden group-hover:scale-110 transition-transform duration-500",
                          insight.bgColor
                        )}
                        style={{ 
                          transform: 'translateZ(15px)',
                          boxShadow: `0 0 10px rgba(${insight.glowColor}, 0.3)` 
                        }}
                      >
                        {/* Shine effect */}
                        <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
                          <div className="absolute -inset-[100%] animate-shine bg-gradient-to-r from-transparent via-white/30 to-transparent" />
                        </div>
                        
                        <insight.icon className={cn(insight.color, "group-hover:animate-pulse")} size={20} />
                      </div>
                      
                      <div style={{ transform: 'translateZ(10px)' }}>
                        <h3 className="font-bold text-white text-lg mb-1 group-hover:text-primary transition-colors">{insight.title}</h3>
                        <p className="text-slate-400 group-hover:text-slate-300 transition-colors">{insight.description}</p>
                      </div>
                    </div>
                    
                    {index === 0 && (
                      <div 
                        className="mt-4 grid grid-cols-3 gap-2 pl-14"
                        style={{ transform: 'translateZ(8px)' }}
                      >
                        {[
                          { label: "Face Swap", value: "78%" },
                          { label: "Voice Clone", value: "92%" },
                          { label: "Lip-sync", value: "65%" }
                        ].map((item, i) => (
                          <div 
                            key={i} 
                            className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg p-3 text-center group/stat relative overflow-hidden"
                            style={{ 
                              transform: isHoveringPanel ? `translateZ(${10 + i * 5}px)` : 'translateZ(0)',
                              transition: `transform 500ms ease ${i * 0.1}s`,
                              boxShadow: isHoveringPanel ? '0 10px 20px -5px rgba(79, 70, 229, 0.15)' : 'none'
                            }}
                          >
                            {/* Shine effect */}
                            <div className="absolute inset-0 opacity-0 group-hover/stat:opacity-100 pointer-events-none">
                              <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                            </div>
                            
                            <div 
                              className="text-xl font-bold text-white transition-all duration-300 group-hover/stat:text-indigo-400"
                              style={{
                                textShadow: isHoveringPanel ? '0 0 10px rgba(79, 70, 229, 0.4)' : 'none'
                              }}
                            >{item.value}</div>
                            <div className="text-xs text-slate-400 group-hover/stat:text-slate-300">{item.label}</div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {index === 1 && (
                      <div 
                        className="mt-5 grid grid-cols-1 gap-3 pl-14"
                        style={{ transform: 'translateZ(8px)' }}
                      >
                        {[
                          { label: "Single-modal", value: 82, color: "from-emerald-500 to-emerald-400", textColor: "text-emerald-900" },
                          { label: "Multi-modal", value: 97, color: "from-emerald-600 to-emerald-400", textColor: "text-emerald-900" }
                        ].map((item, i) => (
                          <div 
                            key={i} 
                            className="relative h-6"
                            style={{ 
                              transform: isHoveringPanel ? `translateZ(${12 - i * 3}px)` : 'translateZ(0)',
                              transition: `transform 500ms ease ${i * 0.15}s`
                            }}
                          >
                            {/* Background bar */}
                            <div className="absolute inset-0 bg-slate-800 rounded-full overflow-hidden"></div>
                            
                            {/* Animated progress with growing animation */}
                            <div 
                              className={`absolute inset-y-0 left-0 bg-gradient-to-r ${item.color} rounded-full transition-all duration-1000 group-hover:shadow-[0_0_8px_rgba(16,185,129,0.5)]`} 
                              style={{ 
                                width: isHoveringPanel ? `${item.value}%` : '0%',
                                transition: 'width 1.5s ease-out, box-shadow 0.3s ease'
                              }}
                            >
                              <span className={`absolute right-2 top-1/2 transform -translate-y-1/2 text-xs font-semibold ${item.textColor} animate-fadeIn`}>
                                {isHoveringPanel ? `${item.value}%` : ''}
                              </span>
                            </div>
                            
                            <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-xs font-medium text-white">
                              {item.label}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                
                <div 
                  className="bg-primary/10 border border-primary/20 rounded-xl p-4 mt-6 relative overflow-hidden group transform-gpu hover:shadow-[0_0_15px_rgba(0,200,255,0.2)] transition-all"
                  style={{ 
                    transform: isHoveringPanel ? 'translateZ(25px)' : 'translateZ(0)',
                    transition: 'transform 500ms ease 0.3s'
                  }}
                >
                  {/* Shine effect */}
                  <div className="absolute inset-0 opacity-0 group-hover:opacity-100 pointer-events-none">
                    <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/10 to-transparent" />
                  </div>
                  
                  <div className="flex items-center">
                    <Info size={16} className="text-primary mr-2 group-hover:animate-pulse" />
                    <p className="text-sm text-primary">Coming soon: Quarterly trend analysis and predictive deepfake detection models</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Side Panel - Detection Tips */}
        <div className="lg:col-span-1">
          <div 
            className="bg-slate-900 rounded-xl p-6 border border-slate-800 h-full relative transform-gpu overflow-hidden"
            style={{ 
              transform: isHoveringPanel ? 'translateZ(10px) scale(1.01)' : 'translateZ(0) scale(1)',
              transformStyle: 'preserve-3d',
              transition: 'transform 500ms ease',
              boxShadow: '0 10px 30px -15px rgba(0, 0, 0, 0.3)'
            }}
          >
            {/* Subtle animated background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-slate-950 to-slate-900 opacity-50"></div>
            
            {/* Digital grid pattern */}
            <div className="absolute inset-0 opacity-5">
              <div className="w-full h-full" style={{
                backgroundImage: `radial-gradient(circle, rgba(0,200,255,0.2) 1px, transparent 1px)`,
                backgroundSize: '15px 15px'
              }}/>
            </div>
            
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-bold text-white flex items-center">
                  <BookOpen size={18} className="mr-2 text-primary animate-pulse" />
                  <span>Detection Guide</span>
                </h3>
                <div className="text-xs bg-primary/20 text-primary px-2 py-1 rounded-full flex items-center gap-1 relative overflow-hidden group/badge">
                  {/* Subtle shine effect */}
                  <div className="absolute inset-0 opacity-0 group-hover/badge:opacity-100 pointer-events-none">
                    <div className="absolute -inset-[100%] animate-shine-slow bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                  </div>
                  <Lock size={9} />
                  <span>Expert Tips</span>
                </div>
              </div>
              
              <div className="space-y-4">
                {tips.map((tip, index) => (
                  <div 
                    key={index} 
                    className={cn(
                      "flex items-start p-3 rounded-lg border transition-all duration-300 relative overflow-hidden transform-gpu",
                      activeTip === index
                        ? "bg-slate-800/80 border-primary/40 shadow-[0_0_15px_rgba(0,200,255,0.2)]"
                        : "bg-slate-800/30 border-slate-700/30 hover:border-primary/20"
                    )}
                    style={{ 
                      transform: activeTip === index ? 'translateZ(30px) scale(1.02)' : 'translateZ(0) scale(1)',
                      transformStyle: 'preserve-3d',
                      transition: `all 400ms ease ${tip.animationDelay}s`
                    }}
                    onMouseEnter={() => setActiveTip(index)}
                    onMouseLeave={() => setActiveTip(null)}
                  >
                    {/* Animated background glow */}
                    {activeTip === index && (
                      <div className="absolute inset-0 opacity-20 pointer-events-none" style={{
                        background: `radial-gradient(circle at center, rgba(0, 200, 255, 0.8) 0%, transparent 70%)`
                      }} />
                    )}
                    
                    {/* Number indicator with icon */}
                    <div 
                      className={cn(
                        "w-6 h-6 rounded-full flex items-center justify-center mr-3 mt-0.5 transition-all",
                        activeTip === index
                          ? `bg-${tip.color.split('-')[1]}-400/30 ${tip.color}`
                          : "bg-primary/10 text-primary/80"
                      )}
                      style={{ 
                        transform: activeTip === index ? 'translateZ(10px) scale(1.1)' : 'scale(1)',
                        boxShadow: activeTip === index ? '0 0 10px rgba(0, 200, 255, 0.3)' : 'none'
                      }}
                    >
                      {activeTip === index ? (
                        <tip.icon size={12} className="animate-pulse" />
                      ) : (
                        <span>{index + 1}</span>
                      )}
                    </div>
                    
                    {/* Tip text */}
                    <p 
                      className={cn(
                        "text-sm transition-colors",
                        activeTip === index ? "text-white" : "text-slate-300"
                      )}
                      style={{ 
                        transform: activeTip === index ? 'translateZ(15px)' : 'none',
                        textShadow: activeTip === index ? '0 0 8px rgba(255, 255, 255, 0.1)' : 'none'
                      }}
                    >
                      {tip.text}
                    </p>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 pt-5 border-t border-slate-800">
                <button 
                  className="w-full py-3 px-4 bg-gradient-to-r from-primary to-primary/80 text-white rounded-lg relative overflow-hidden group/btn transform-gpu"
                  style={{ 
                    transform: 'translateZ(20px)',
                    boxShadow: '0 0 20px rgba(0, 200, 255, 0.2)'
                  }}
                >
                  {/* Button glow effect on hover */}
                  <div className="absolute inset-0 opacity-0 group-hover/btn:opacity-100 transition-opacity duration-300" style={{
                    background: 'linear-gradient(to right, rgba(0, 200, 255, 0.1), rgba(0, 120, 255, 0.1))'
                  }}></div>
                  
                  {/* Shine effect */}
                  <div className="absolute inset-0 opacity-0 group-hover/btn:opacity-100 pointer-events-none">
                    <div className="absolute -inset-[100%] animate-shine bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                  </div>
                  
                  <div className="relative flex items-center justify-center text-sm font-medium gap-2">
                    <BookOpen size={16} className="group-hover/btn:animate-pulse" />
                    <span>View Complete Deepfake Guide</span>
                    <ExternalLink size={12} className="ml-1 group-hover/btn:translate-x-0.5 transition-transform" />
                  </div>
                </button>
                
                <div className="mt-4 text-xs text-center text-slate-500 relative">
                  <div className="absolute left-0 right-0 h-px bg-gradient-to-r from-transparent via-slate-700/50 to-transparent"></div>
                  <div className="mt-2 flex items-center justify-center gap-1">
                    <Sparkles size={10} className="text-primary/40" />
                    <span>New techniques added: Voice pattern analysis, metadata verification</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Add CSS for custom animations */}
      <style>{`
        @keyframes fadeUpAndOut {
          0% { opacity: 0; transform: translateY(0px); }
          40% { opacity: 0.8; transform: translateY(-40px) rotate(${Math.random() * 180 - 90}deg); }
          100% { opacity: 0; transform: translateY(-80px) rotate(${Math.random() * 360 - 180}deg); }
        }
        
        @keyframes scanRight {
          0% { left: 0; }
          100% { left: 100%; }
        }
        
        @keyframes animate-fadeIn {
          0% { opacity: 0; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}
