import { useNavigate } from "wouter";
import { cn } from "@/lib/utils";
import { Image as ImageIcon, Video, Mic, Camera } from "lucide-react";

export default function QuickAccessTiles() {
  const navigate = useNavigate();

  const mediaTypes = [
    {
      title: "Image",
      description: "Analyze still images for manipulation",
      icon: ImageIcon,
      colorClass: "text-primary bg-primary/20 group-hover:bg-primary/30",
      shadowClass: "group-hover:shadow-[0_0_10px_rgba(0,200,255,0.7)]",
      path: "/scan?type=image"
    },
    {
      title: "Video",
      description: "Detect deepfakes in video content",
      icon: Video,
      colorClass: "text-secondary bg-secondary/20 group-hover:bg-secondary/30",
      shadowClass: "group-hover:shadow-[0_0_10px_rgba(6,214,160,0.7)]",
      path: "/scan?type=video"
    },
    {
      title: "Audio",
      description: "Identify voice synthesis & cloning",
      icon: Mic,
      colorClass: "text-accent bg-accent/20 group-hover:bg-accent/30",
      shadowClass: "group-hover:shadow-[0_0_10px_rgba(131,255,51,0.7)]",
      path: "/scan?type=audio"
    },
    {
      title: "Webcam",
      description: "Real-time live analysis",
      icon: Camera,
      colorClass: "text-primary bg-primary/20 group-hover:bg-primary/30",
      shadowClass: "group-hover:shadow-[0_0_10px_rgba(0,200,255,0.7)]",
      path: "/scan?type=webcam"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      {mediaTypes.map((type) => (
        <div 
          key={type.title}
          className={cn(
            "bg-card rounded-xl p-5 transition-all duration-300 cursor-pointer group",
            type.shadowClass
          )}
          onClick={() => navigate(type.path)}
        >
          <div className={cn(
            "mb-3 w-12 h-12 rounded-full flex items-center justify-center transition-colors",
            type.colorClass
          )}>
            <type.icon className="text-2xl" size={24} />
          </div>
          <h3 className="font-poppins font-medium text-lg text-foreground mb-1">
            {type.title}
          </h3>
          <p className="text-sm text-muted-foreground">
            {type.description}
          </p>
        </div>
      ))}
    </div>
  );
}
