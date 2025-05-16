import { ArrowUp, ArrowDown, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export default function InformativeSection() {
  const stats = [
    {
      label: "Analyzed Media",
      value: "147",
      change: "+23%",
      isPositive: true
    },
    {
      label: "Detected Deepfakes",
      value: "36",
      change: "+12%",
      isPositive: false,
      isWarning: true
    },
    {
      label: "Avg. Detection Time",
      value: "4.2s",
      change: "-18%",
      isPositive: true
    },
    {
      label: "Detection Accuracy",
      value: "96%",
      change: "+3%",
      isPositive: true
    }
  ];

  const tips = [
    "Look for unnatural eye blinking patterns and inconsistent eye reflections.",
    "Check for unnatural hair movement, unusual skin texture, or blurry face boundaries.",
    "Watch for inconsistencies in audio-visual synchronization, especially in videos.",
    "Use multiple detection methods for increased accuracy and confidence."
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
      {/* Stats Card */}
      <div className="bg-card rounded-xl p-6">
        <h2 className="text-xl font-poppins font-semibold text-foreground mb-4">Detection Stats</h2>
        
        <div className="grid grid-cols-2 gap-4">
          {stats.map((stat, index) => (
            <div key={index} className="p-4 bg-muted rounded-lg">
              <p className="text-muted-foreground text-sm">{stat.label}</p>
              <div className="flex items-end justify-between">
                <span className={cn(
                  "text-2xl font-bold",
                  stat.isWarning ? "text-destructive" : "text-foreground"
                )}>
                  {stat.value}
                </span>
                <span className={cn(
                  "text-sm flex items-center",
                  stat.isPositive && !stat.isWarning ? "text-secondary" : "text-destructive"
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
          ))}
        </div>
      </div>
      
      {/* Tips Card */}
      <div className="bg-card rounded-xl p-6">
        <h2 className="text-xl font-poppins font-semibold text-foreground mb-4">Deepfake Detection Tips</h2>
        
        <ul className="space-y-3">
          {tips.map((tip, index) => (
            <li key={index} className="flex items-start">
              <CheckCircle className="text-secondary mt-1 mr-2 flex-shrink-0" size={16} />
              <p className="text-muted-foreground text-sm">{tip}</p>
            </li>
          ))}
        </ul>
        
        <button className="mt-4 py-2 px-4 bg-card hover:bg-muted border border-secondary/50 text-secondary rounded-lg transition-colors flex items-center text-sm">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="mr-2"
          >
            <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
            <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
          </svg>
          <span>View Complete Guide</span>
        </button>
      </div>
    </div>
  );
}
