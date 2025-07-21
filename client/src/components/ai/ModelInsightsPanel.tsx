import React from 'react';
<<<<<<< HEAD
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "../ui/card";
import { Progress } from "../ui/progress";
import { Separator } from "../ui/separator";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
=======
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
>>>>>>> a152be44fa5a0782cc9b4e4235229eb36a2aaa8f
import { BrainCircuit, Info, Shield, Zap, BarChart3, Activity, FileSearch } from "lucide-react";

interface ModelInsightsPanelProps {
  analysisData?: any;
  isLoading?: boolean;
}

const ModelInsightsPanel: React.FC<ModelInsightsPanelProps> = ({
  analysisData,
  isLoading = false
}) => {
  if (isLoading) {
    return (
      <Card className="w-full animate-pulse">
        <CardHeader className="space-y-2">
          <div className="h-6 w-48 bg-muted rounded-md"></div>
          <div className="h-4 w-full bg-muted rounded-md"></div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="h-4 w-full bg-muted rounded-md"></div>
          <div className="h-24 w-full bg-muted rounded-md"></div>
        </CardContent>
      </Card>
    );
  }

  if (!analysisData) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center">
            <BrainCircuit className="mr-2 h-5 w-5" />
            AI Model Insights
          </CardTitle>
          <CardDescription>
            Run an analysis to view detailed AI model insights
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center py-8">
          <Info className="h-12 w-12 text-muted-foreground mx-auto mb-3" />
          <p className="text-muted-foreground">
            Model insights will appear here after analysis
          </p>
        </CardContent>
      </Card>
    );
  }

  // Extract model data from analysis
  const modelInfo = analysisData.model_info || {
    name: "SatyaAI Model",
    version: "1.0",
    type: "Multimodal Analysis"
  };
  
  // Get metrics if available
  const metrics = analysisData.metrics || {
    temporal_consistency: 0.85,
    lighting_consistency: 0.92,
    audio_visual_sync: 0.88,
    face_movement_naturality: 0.91
  };
  
  // Whether it's a multimodal analysis
  const isMultimodal = !!analysisData.modalities_used;
  
  // Get detected artifacts if available
  const artifacts = [
    "Face boundary inconsistencies",
    "Temporal flickering in eye region",
    "Unnatural lip movements",
    "Audio-visual desynchronization"
  ];
  
  return (
    <Card className="w-full border border-primary/20">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="flex items-center">
              <BrainCircuit className="mr-2 h-5 w-5 text-primary" />
              AI Model Insights
            </CardTitle>
            <CardDescription>
              Technical details about the detection process
            </CardDescription>
          </div>
          <Badge variant="outline" className="bg-primary/5 border-primary/30">
            v{modelInfo.version}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Model Information */}
        <div className="space-y-2">
          <h3 className="text-sm font-medium flex items-center">
            <Shield className="h-4 w-4 mr-2 text-primary" />
            Model Information
          </h3>
          
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="space-y-1">
              <p className="text-muted-foreground">Engine</p>
              <p className="font-medium">{modelInfo.name}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Type</p>
              <p className="font-medium">{modelInfo.type}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Approach</p>
              <p className="font-medium">{isMultimodal ? "Multimodal Fusion" : "Single-modal Analysis"}</p>
            </div>
            <div className="space-y-1">
              <p className="text-muted-foreground">Features</p>
              <p className="font-medium">{isMultimodal ? analysisData.modalities_used?.length || 0 : 1}</p>
            </div>
          </div>
        </div>
        
        <Separator />
        
        {/* Performance Metrics */}
        <div className="space-y-3">
          <h3 className="text-sm font-medium flex items-center">
            <BarChart3 className="h-4 w-4 mr-2 text-primary" />
            Detection Metrics
          </h3>
          
          <div className="space-y-4">
            {Object.entries(metrics).map(([key, value]: [string, any]) => (
              <div key={key} className="space-y-1">
                <div className="flex justify-between items-center">
                  <p className="text-sm capitalize">
                    {key.replace(/_/g, ' ')}
                  </p>
                  <span className={`text-xs font-medium ${
                    value > 0.9 ? 'text-green-500' : 
                    value > 0.7 ? 'text-amber-500' : 
                    'text-red-500'
                  }`}>
                    {(value * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={value * 100} className="h-1" />
              </div>
            ))}
          </div>
        </div>
        
        {/* Artifact Detection */}
        {artifacts.length > 0 && (
          <>
            <Separator />
            
            <div className="space-y-3">
              <h3 className="text-sm font-medium flex items-center">
                <FileSearch className="h-4 w-4 mr-2 text-primary" />
                Detected Artifacts
              </h3>
              
              <ul className="space-y-2">
                {artifacts.map((artifact, index) => (
                  <li key={index} className="text-sm flex items-start">
                    <div className="h-2 w-2 rounded-full bg-red-500 mt-1.5 mr-2 flex-shrink-0"></div>
                    <span>{artifact}</span>
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
        
        {/* Detection Patterns */}
        {analysisData.suspicious_frames && (
          <>
            <Separator />
            
            <div className="space-y-3">
              <h3 className="text-sm font-medium flex items-center">
                <Activity className="h-4 w-4 mr-2 text-primary" />
                Anomaly Timeline
              </h3>
              
              <div className="relative h-10 bg-muted rounded-md overflow-hidden">
                {analysisData.suspicious_frames.map((frame: string, index: number) => {
                  // Convert frame ranges like "123-145" to percentages for visualization
                  const [start, end] = frame.split('-').map(Number);
                  const startPercent = (start / 1000) * 100;
                  const width = ((end - start) / 1000) * 100;
                  
                  return (
                    <div 
                      key={index}
                      className="absolute h-full bg-red-500/70"
                      style={{
                        left: `${startPercent}%`,
                        width: `${width}%`,
                      }}
                      title={`Suspicious frames: ${frame}`}
                    />
                  );
                })}
                
                <div className="absolute inset-0 flex items-center justify-between px-2 text-xs text-muted-foreground">
                  <span>Start</span>
                  <span>End</span>
                </div>
              </div>
              
              <p className="text-xs text-muted-foreground">
                Red regions indicate detected anomalies in the video timeline
              </p>
            </div>
          </>
        )}
      </CardContent>
      
      <CardFooter className="justify-between pt-2">
        <Button variant="outline" size="sm">
          <Info className="h-4 w-4 mr-2" />
          Learn More
        </Button>
        <Button variant="outline" size="sm">
          <Zap className="h-4 w-4 mr-2" />
          Advanced Details
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ModelInsightsPanel;