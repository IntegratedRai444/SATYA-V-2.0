import { cn } from "@/lib/utils";
import { Image as ImageIcon, Video, Mic, Camera, ArrowRight, Check, CircleAlert, FileWarning } from "lucide-react";
import { useNavigation } from "@/hooks/useNavigation";
import { useState } from "react";

export default function QuickAccessTiles() {
  const { navigate } = useNavigation();
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

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
      featuredStats: [
        { label: "Live Deepfake Alert", icon: Check },
        { label: "Facial Authentication", icon: Check },
        { label: "Low-Light Analysis", icon: FileWarning }
      ]
    }
  ];

  return (
    <div className="mb-12">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold mb-1 text-foreground">Deepfake Detection Tools</h2>
          <p className="text-muted-foreground">Choose your media type for comprehensive analysis</p>
        </div>
        <div className="bg-card/60 rounded-lg border border-border/50 px-3 py-2 text-sm text-muted-foreground mt-3 md:mt-0">
          Using <span className="font-semibold text-primary">Neural Vision v4.2</span> models
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 lg:gap-6">
        {mediaTypes.map((type, index) => (
          <div 
            key={type.title}
            className={cn(
              "bg-gradient-to-br rounded-xl border transition-all duration-300 cursor-pointer group relative overflow-hidden h-full",
              hoveredIndex === index ? `${type.hoverGradient} shadow-lg scale-[1.02] ${type.borderColor}` : 
              "from-transparent to-transparent shadow border-border/50",
            )}
            onClick={() => navigate(type.path)}
            onMouseEnter={() => setHoveredIndex(index)}
            onMouseLeave={() => setHoveredIndex(null)}
          >
            {/* Decorative elements */}
            <div className="absolute top-0 right-0 w-32 h-32 opacity-5 -translate-y-16 translate-x-16">
              <div className="w-full h-full rounded-full border-4 border-current"></div>
            </div>
            
            <div className="p-5 flex flex-col h-full">
              <div className="flex justify-between items-start mb-3">
                <div className={cn(
                  "w-12 h-12 rounded-lg flex items-center justify-center transition-colors",
                  type.colorClass
                )}>
                  <type.icon size={22} />
                </div>
                
                <div className={cn(
                  "text-xs font-medium px-2 py-1 rounded-full transition-colors",
                  hoveredIndex === index ? type.colorClass : "bg-muted/50 text-muted-foreground"
                )}>
                  Accuracy: {type.accuracy}
                </div>
              </div>
              
              <h3 className="font-bold text-lg mb-1 text-foreground group-hover:text-foreground/90">
                {type.title}
              </h3>
              
              <p className="text-sm text-muted-foreground mb-4">
                {type.description}
              </p>
              
              <div className="mt-auto">
                <div className="space-y-1.5 mb-4">
                  {type.featuredStats.map((stat, statIndex) => (
                    <div key={statIndex} className="flex items-center gap-2 text-xs text-muted-foreground">
                      <stat.icon size={12} className={stat.icon === Check ? "text-green-500" : stat.icon === FileWarning ? "text-amber-500" : "text-slate-500"} />
                      <span>{stat.label}</span>
                    </div>
                  ))}
                </div>
                
                <div className={cn(
                  "flex items-center text-xs font-medium transition-colors",
                  hoveredIndex === index ? type.colorClass : "text-muted-foreground"
                )}>
                  <span>START ANALYSIS</span>
                  <ArrowRight size={12} className="ml-1 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
