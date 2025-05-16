import { Bot } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function AIAssistant() {
  return (
    <div className="p-4 rounded-lg bg-muted border border-secondary/30 relative overflow-hidden">
      <div className="absolute -top-6 -right-6 w-12 h-12 rounded-full bg-secondary/20 flex items-center justify-center">
        <Bot className="text-secondary" size={20} />
      </div>
      <h3 className="font-poppins font-medium text-secondary mb-2">AI Assistant</h3>
      <p className="text-xs text-muted-foreground mb-3">
        Need help with deepfake detection? I can guide you through the process.
      </p>
      <Button
        variant="outline"
        className="w-full py-2 px-3 rounded bg-secondary/20 text-secondary text-sm hover:bg-secondary/30 border-secondary/30"
      >
        Ask for help
      </Button>
    </div>
  );
}
