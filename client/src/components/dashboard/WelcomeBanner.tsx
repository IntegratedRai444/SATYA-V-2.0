import { UploadCloud, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigation } from "@/hooks/useNavigation";

export default function WelcomeBanner() {
  const { navigate } = useNavigation();

  const handleUploadClick = () => {
    navigate("/scan");
  };

  return (
    <div className="mb-8 bg-muted rounded-xl p-6 relative overflow-hidden">
      {/* Abstract digital background with reduced opacity */}
      <div className="absolute inset-0 opacity-20 bg-cover bg-center" />
      
      <div className="relative z-10">
        <h1 className="text-3xl font-bold font-poppins text-foreground mb-2">
          Welcome to <span className="text-primary">SatyaAI</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-2xl">
          Advanced deepfake detection system that helps you authenticate media with confidence. 
          Upload files or use your webcam to get started.
        </p>
        
        <div className="mt-6 flex flex-wrap gap-4">
          <Button 
            className="shadow-[0_0_10px_rgba(0,200,255,0.3)] flex items-center gap-2"
            onClick={handleUploadClick}
          >
            <UploadCloud size={18} />
            <span>Upload Media</span>
          </Button>
          
          <Button 
            variant="outline" 
            className="border-primary/50 text-primary flex items-center gap-2"
          >
            <Info size={18} />
            <span>Learn More</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
