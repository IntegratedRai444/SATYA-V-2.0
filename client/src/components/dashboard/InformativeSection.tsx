import { ArrowUp, ArrowDown, CheckCircle, Info, BookOpen, AlertTriangle, Clock, PieChart, Layers, Shield, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

export default function InformativeSection() {
  const [activeTab, setActiveTab] = useState<'stats' | 'insights'>('stats');
  
  const stats = [
    {
      label: "Analyzed Media",
      value: "147",
      change: "+23%",
      isPositive: true,
      icon: Layers,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10"
    },
    {
      label: "Detected Deepfakes",
      value: "36",
      change: "+12%",
      isPositive: false,
      isWarning: true,
      icon: AlertTriangle,
      color: "text-rose-500",
      bgColor: "bg-rose-500/10"
    },
    {
      label: "Avg. Detection Time",
      value: "4.2s",
      change: "-18%",
      isPositive: true,
      icon: Clock,
      color: "text-amber-500",
      bgColor: "bg-amber-500/10"
    },
    {
      label: "Detection Accuracy",
      value: "96%",
      change: "+3%",
      isPositive: true,
      icon: PieChart,
      color: "text-emerald-500",
      bgColor: "bg-emerald-500/10"
    }
  ];

  const insights = [
    {
      title: "AI Detection Trends",
      description: "Our neural networks showed a 12% improvement in detecting GAN-based face manipulations over the last month.",
      icon: Activity,
      color: "text-indigo-500",
      bgColor: "bg-indigo-500/10"
    },
    {
      title: "Security Analysis",
      description: "Multi-modal detection combining audio, video, and metadata analysis improves accuracy by up to 15%.",
      icon: Shield,
      color: "text-emerald-500",
      bgColor: "bg-emerald-500/10"
    }
  ];

  const tips = [
    "Look for unnatural eye blinking patterns and inconsistent eye reflections in suspected videos.",
    "Check for unnatural hair movement, unusual skin texture, or blurry face boundaries in images.",
    "Watch for inconsistencies in audio-visual synchronization, especially in speech videos.",
    "Analyze visual artifacts around the edges of faces, which often indicate manipulation."
  ];

  return (
    <div className="mt-8 mb-12">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold mb-1 text-foreground">Analytics & Insights</h2>
          <p className="text-muted-foreground">System performance and detection tips</p>
        </div>
        <div className="flex bg-card rounded-lg border border-border p-1 mt-3 md:mt-0">
          <button 
            className={cn(
              "px-4 py-2 text-sm font-medium rounded-md transition-colors",
              activeTab === 'stats' 
                ? "bg-primary text-white" 
                : "hover:bg-muted text-muted-foreground"
            )}
            onClick={() => setActiveTab('stats')}
          >
            Statistics
          </button>
          <button 
            className={cn(
              "px-4 py-2 text-sm font-medium rounded-md transition-colors",
              activeTab === 'insights' 
                ? "bg-primary text-white" 
                : "hover:bg-muted text-muted-foreground"
            )}
            onClick={() => setActiveTab('insights')}
          >
            Insights
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Panel */}
        <div className="lg:col-span-2 bg-gradient-to-br from-slate-950 to-slate-900 rounded-xl overflow-hidden border border-slate-800">
          {activeTab === 'stats' ? (
            <div className="p-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                {stats.map((stat, index) => (
                  <div 
                    key={index} 
                    className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 flex items-start relative overflow-hidden group"
                  >
                    {/* Background decoration */}
                    <div className="absolute -right-12 -bottom-12 w-40 h-40 rounded-full opacity-5 border-8 border-slate-400 group-hover:opacity-10 transition-opacity"></div>
                    
                    <div className={cn(
                      "w-12 h-12 rounded-lg flex items-center justify-center mr-4 transition-transform group-hover:scale-110",
                      stat.bgColor
                    )}>
                      <stat.icon className={stat.color} size={20} />
                    </div>
                    
                    <div>
                      <p className="text-muted-foreground text-sm mb-1">{stat.label}</p>
                      <div className="flex items-end gap-2">
                        <span className={cn(
                          "text-3xl font-bold",
                          stat.isWarning ? "text-rose-500" : "text-white"
                        )}>
                          {stat.value}
                        </span>
                        <span className={cn(
                          "text-sm font-medium flex items-center",
                          stat.isPositive && !stat.isWarning ? "text-emerald-500" : "text-rose-500"
                        )}>
                          {stat.change}{" "}
                          {stat.isPositive ? (
                            <ArrowUp size={14} className="ml-1" />
                          ) : (
                            <ArrowDown size={14} className="ml-1" />
                          )}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 bg-slate-900/50 border border-slate-800 rounded-xl p-5">
                <div className="flex items-center mb-4">
                  <Activity size={18} className="text-primary mr-2" />
                  <h3 className="text-lg font-bold text-white">Detection Activity</h3>
                </div>
                
                <div className="relative h-32 w-full">
                  {/* Simulated activity chart with HTML/CSS */}
                  <div className="absolute inset-x-0 bottom-0 h-full flex items-end px-2">
                    {Array.from({ length: 24 }).map((_, i) => {
                      const height = 20 + Math.random() * 60;
                      const isHighlighted = i === 16;
                      return (
                        <div 
                          key={i} 
                          className="flex-1 mx-0.5 group cursor-pointer"
                          style={{ height: `${height}%` }}
                        >
                          <div 
                            className={cn(
                              "w-full h-full rounded-sm transition-all",
                              isHighlighted 
                                ? "bg-primary" 
                                : "bg-primary/30 group-hover:bg-primary/60"
                            )}
                          />
                          {isHighlighted && (
                            <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-primary text-white text-xs py-1 px-2 rounded">
                              16 Alerts
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
            <div className="p-6">
              {insights.map((insight, index) => (
                <div 
                  key={index}
                  className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 mb-5 last:mb-0"
                >
                  <div className="flex items-start">
                    <div className={cn(
                      "w-10 h-10 rounded-lg flex items-center justify-center mr-4",
                      insight.bgColor
                    )}>
                      <insight.icon className={insight.color} size={20} />
                    </div>
                    
                    <div>
                      <h3 className="font-bold text-white text-lg mb-1">{insight.title}</h3>
                      <p className="text-slate-400">{insight.description}</p>
                    </div>
                  </div>
                  
                  {index === 0 && (
                    <div className="mt-4 grid grid-cols-3 gap-2 pl-14">
                      <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg p-3 text-center">
                        <div className="text-xl font-bold text-white">78%</div>
                        <div className="text-xs text-slate-400">Face Swap</div>
                      </div>
                      <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg p-3 text-center">
                        <div className="text-xl font-bold text-white">92%</div>
                        <div className="text-xs text-slate-400">Voice Clone</div>
                      </div>
                      <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg p-3 text-center">
                        <div className="text-xl font-bold text-white">65%</div>
                        <div className="text-xs text-slate-400">Lip-sync</div>
                      </div>
                    </div>
                  )}
                  
                  {index === 1 && (
                    <div className="mt-5 grid grid-cols-1 gap-3 pl-14">
                      <div className="relative h-6 bg-slate-800 rounded-full overflow-hidden">
                        <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full" style={{ width: '82%' }}>
                          <span className="absolute right-2 top-1/2 transform -translate-y-1/2 text-xs font-semibold text-emerald-900">82%</span>
                        </div>
                        <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-xs font-medium text-white">Single-modal</span>
                      </div>
                      
                      <div className="relative h-6 bg-slate-800 rounded-full overflow-hidden">
                        <div className="absolute inset-y-0 left-0 bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full" style={{ width: '97%' }}>
                          <span className="absolute right-2 top-1/2 transform -translate-y-1/2 text-xs font-semibold text-emerald-900">97%</span>
                        </div>
                        <span className="absolute left-3 top-1/2 transform -translate-y-1/2 text-xs font-medium text-white">Multi-modal</span>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              
              <div className="bg-primary/10 border border-primary/20 rounded-xl p-4 mt-6">
                <div className="flex items-center">
                  <Info size={16} className="text-primary mr-2" />
                  <p className="text-sm text-primary">Coming soon: Quarterly trend analysis and predictive deepfake detection models</p>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Side Panel */}
        <div className="lg:col-span-1">
          <div className="bg-slate-900 rounded-xl p-6 border border-slate-800 h-full">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-white flex items-center">
                <BookOpen size={18} className="mr-2 text-primary" />
                Detection Guide
              </h3>
              <span className="text-xs bg-primary/20 text-primary px-2 py-1 rounded-full">Expert Tips</span>
            </div>
            
            <div className="space-y-4">
              {tips.map((tip, index) => (
                <div 
                  key={index} 
                  className="flex items-start bg-slate-800/50 p-3 rounded-lg border border-slate-700/50 hover:border-primary/30 transition-colors group"
                >
                  <div className="bg-primary/20 text-primary rounded-full w-6 h-6 flex items-center justify-center mr-3 mt-0.5 group-hover:bg-primary/30 transition-colors">
                    {index + 1}
                  </div>
                  <p className="text-slate-300 text-sm">{tip}</p>
                </div>
              ))}
            </div>
            
            <div className="mt-6 pt-5 border-t border-slate-800">
              <button className="w-full py-3 px-4 bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-white rounded-lg transition-colors flex items-center justify-center text-sm font-medium group">
                <BookOpen size={16} className="mr-2 group-hover:animate-pulse" />
                <span>View Complete Deepfake Guide</span>
              </button>
              
              <div className="mt-4 text-xs text-center text-slate-500">
                New techniques added: Voice pattern analysis, metadata verification
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
