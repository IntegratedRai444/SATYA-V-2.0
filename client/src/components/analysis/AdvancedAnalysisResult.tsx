import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { AlertTriangle, CheckCircle, ChevronDown, Download, ExternalLink, Eye, FileCheck, Info, Share2, ShieldAlert, Zap } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface AdvancedAnalysisResultProps {
  result: any; // The detailed analysis result
  isLoading?: boolean;
  onShare?: () => void;
  onExport?: () => void;
}

const AdvancedAnalysisResult: React.FC<AdvancedAnalysisResultProps> = ({
  result,
  isLoading = false,
  onShare,
  onExport
}) => {
  const { toast } = useToast();
  const [expandedSection, setExpandedSection] = useState<string | null>(null);
  
  // Handle loading state
  if (isLoading) {
    return (
      <Card className="w-full animate-pulse">
        <CardHeader>
          <div className="h-6 w-48 bg-muted rounded-md"></div>
          <div className="h-4 w-full bg-muted rounded-md mt-2"></div>
        </CardHeader>
        <CardContent>
          <div className="h-32 w-full bg-muted rounded-md"></div>
        </CardContent>
      </Card>
    );
  }
  
  // Handle missing result
  if (!result) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>No Analysis Data</CardTitle>
          <CardDescription>No analysis results available to display</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center justify-center py-8">
            <InfoIcon size={48} className="text-muted-foreground mb-4" />
            <p className="text-center text-muted-foreground">
              Please run a new analysis to view detailed results
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  // Extract key information from results
  const isAuthentic = result.authenticity === "AUTHENTIC MEDIA";
  const confidence = typeof result.confidence === 'number' ? Math.round(result.confidence) : 0;
  const analysisDate = result.analysis_date || new Date().toISOString();
  const formattedDate = new Date(analysisDate).toLocaleString();
  const keyFindings = result.key_findings || [];
  const modelInfo = result.model_info || { name: "Advanced AI Model", version: "1.0" };
  
  // Extract multimodal information if available
  const isMultimodal = !!result.modalities_used;
  const modalities = result.modalities_used || [];
  const modalityResults = result.modality_results || {};
  const crossModalConsistency = result.cross_modal_consistency || 1.0;
  
  // Extract suspicious frames if available (for video)
  const suspiciousFrames = result.suspicious_frames || [];
  
  // Handle sharing
  const handleShare = () => {
    if (onShare) {
      onShare();
    } else {
      // Fallback implementation
      toast({
        title: "Share Link Generated",
        description: "Analysis results link copied to clipboard",
      });
    }
  };
  
  // Handle export
  const handleExport = () => {
    if (onExport) {
      onExport();
    } else {
      // Fallback implementation - export as JSON
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(result, null, 2));
      const downloadAnchorNode = document.createElement('a');
      downloadAnchorNode.setAttribute("href", dataStr);
      downloadAnchorNode.setAttribute("download", `satya-analysis-${new Date().getTime()}.json`);
      document.body.appendChild(downloadAnchorNode);
      downloadAnchorNode.click();
      downloadAnchorNode.remove();
      
      toast({
        title: "Analysis Exported",
        description: "Results saved as JSON file",
      });
    }
  };
  
  // Toggle section expansion
  const toggleSection = (section: string) => {
    if (expandedSection === section) {
      setExpandedSection(null);
    } else {
      setExpandedSection(section);
    }
  };
  
  return (
    <Card className="w-full border border-primary/20 overflow-hidden">
      <CardHeader className={isAuthentic ? "bg-green-500/10" : "bg-red-500/10"}>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-xl flex items-center">
              {isAuthentic ? (
                <CheckCircle className="mr-2 text-green-500" />
              ) : (
                <AlertTriangle className="mr-2 text-red-500" />
              )}
              {isAuthentic ? "Authentic Media" : "Manipulated Media Detected"}
            </CardTitle>
            <CardDescription>
              Analysis completed on {formattedDate}
            </CardDescription>
          </div>
          <Badge 
            variant={isAuthentic ? "outline" : "destructive"}
            className={`${isAuthentic ? 'border-green-500 text-green-500 bg-green-500/10' : ''} px-3 py-1 text-sm font-medium`}
          >
            {isAuthentic ? "Authentic" : "Deepfake"}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="p-6">
        {/* Confidence Score */}
        <div className="mb-6">
          <div className="flex justify-between mb-2">
            <span className="text-sm font-medium">Confidence Score</span>
            <span className={`text-sm font-bold ${confidence > 80 ? 'text-green-500' : confidence > 60 ? 'text-amber-500' : 'text-red-500'}`}>
              {confidence}%
            </span>
          </div>
          <Progress 
            value={confidence} 
            className={`h-2 ${confidence > 80 ? 'bg-muted' : confidence > 60 ? 'bg-amber-500/20' : 'bg-red-500/20'}`}
          />
          <p className="text-xs text-muted-foreground mt-2">
            {confidence > 90 ? (
              "Very high confidence in the analysis result"
            ) : confidence > 80 ? (
              "High confidence in the analysis result"
            ) : confidence > 70 ? (
              "Good confidence in the analysis result"
            ) : confidence > 60 ? (
              "Moderate confidence - review findings carefully"
            ) : (
              "Lower confidence - exercise caution when interpreting"
            )}
          </p>
        </div>
        
        <Separator className="my-4" />
        
        {/* Tabs for different analysis views */}
        <Tabs defaultValue="findings" className="mt-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="findings">Key Findings</TabsTrigger>
            {isMultimodal && <TabsTrigger value="modalities">Modality Analysis</TabsTrigger>}
            <TabsTrigger value="technical">Technical Details</TabsTrigger>
          </TabsList>
          
          {/* Key Findings Tab */}
          <TabsContent value="findings" className="py-4">
            <h3 className="text-lg font-semibold mb-3">Analysis Findings</h3>
            <ul className="space-y-2">
              {keyFindings.map((finding: string, index: number) => (
                <li 
                  key={index} 
                  className={`flex items-start p-3 text-sm rounded-md ${
                    finding.includes("No") || finding.includes("natural") || finding.includes("consistent")
                      ? "bg-green-500/10 text-green-700 dark:text-green-300"
                      : "bg-red-500/10 text-red-700 dark:text-red-300"
                  }`}
                >
                  {finding.includes("No") || finding.includes("natural") || finding.includes("consistent") ? (
                    <CheckCircle className="mr-2 h-4 w-4 mt-0.5" />
                  ) : (
                    <ShieldAlert className="mr-2 h-4 w-4 mt-0.5" />
                  )}
                  <span>{finding}</span>
                </li>
              ))}
            </ul>
            
            {suspiciousFrames.length > 0 && (
              <div className="mt-4">
                <h4 className="text-md font-semibold mb-2 text-red-500">Suspicious Frame Sequences</h4>
                <div className="bg-red-500/10 p-3 rounded-md">
                  <p className="text-sm text-red-700 dark:text-red-300">
                    Potential manipulation detected in the following frame ranges:
                  </p>
                  <ul className="mt-2 space-y-1">
                    {suspiciousFrames.map((frameRange: string, index: number) => (
                      <li key={index} className="text-sm">
                        <span className="font-mono">{frameRange}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </TabsContent>
          
          {/* Modality Analysis Tab (for multimodal analysis) */}
          {isMultimodal && (
            <TabsContent value="modalities" className="py-4">
              <h3 className="text-lg font-semibold mb-3">Multimodal Analysis</h3>
              
              <div className="mb-4">
                <div className="flex justify-between mb-2">
                  <span className="text-sm font-medium">Cross-Modal Consistency</span>
                  <span className={`text-sm font-bold ${
                    crossModalConsistency > 0.8 ? 'text-green-500' : 
                    crossModalConsistency > 0.6 ? 'text-amber-500' : 
                    'text-red-500'
                  }`}>
                    {Math.round(crossModalConsistency * 100)}%
                  </span>
                </div>
                <Progress 
                  value={crossModalConsistency * 100} 
                  className="h-2 bg-muted"
                />
                <p className="text-xs text-muted-foreground mt-2">
                  {crossModalConsistency > 0.8 ? (
                    "High consistency across all analyzed modalities"
                  ) : crossModalConsistency > 0.6 ? (
                    "Some inconsistencies detected between modalities"
                  ) : (
                    "Significant inconsistencies between modalities (strong indicator of manipulation)"
                  )}
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
                {Object.entries(modalityResults).map(([modality, result]: [string, any]) => (
                  <Card key={modality} className="overflow-hidden">
                    <CardHeader className={`py-3 px-4 ${
                      result.result === 'authentic' ? 'bg-green-500/10' : 'bg-red-500/10'
                    }`}>
                      <CardTitle className="text-sm flex items-center">
                        {modality === 'image' && <Eye size={16} className="mr-2" />}
                        {modality === 'audio' && <FileCheck size={16} className="mr-2" />}
                        {modality === 'video' && <FileCheck size={16} className="mr-2" />}
                        {modality.charAt(0).toUpperCase() + modality.slice(1)} Analysis
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium">Result</span>
                        <Badge variant={result.result === 'authentic' ? 'outline' : 'destructive'}>
                          {result.result === 'authentic' ? 'Authentic' : 'Manipulated'}
                        </Badge>
                      </div>
                      <div className="mt-2">
                        <div className="flex justify-between mb-1">
                          <span className="text-xs">Confidence</span>
                          <span className="text-xs font-medium">{Math.round(result.confidence)}%</span>
                        </div>
                        <Progress value={result.confidence} className="h-1" />
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
              
              {modalities.length > 0 && (
                <div className="mt-4 bg-blue-500/10 rounded-md p-3">
                  <p className="text-sm text-blue-700 dark:text-blue-300">
                    <Info size={16} className="inline mr-1" />
                    Analysis performed using {modalities.length} modalities: {modalities.join(', ')}
                  </p>
                </div>
              )}
            </TabsContent>
          )}
          
          {/* Technical Details Tab */}
          <TabsContent value="technical" className="py-4">
            <h3 className="text-lg font-semibold mb-3">Technical Details</h3>
            
            <div className="space-y-4">
              <div className="bg-muted/30 rounded-md p-4">
                <h4 className="text-sm font-medium mb-2">Analysis Engine</h4>
                <div className="flex flex-col space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Model</span>
                    <span className="text-sm">{modelInfo.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Version</span>
                    <span className="text-sm">{modelInfo.version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Type</span>
                    <span className="text-sm">{modelInfo.type || "Advanced AI"}</span>
                  </div>
                </div>
              </div>
              
              {/* Additional metrics if available */}
              {result.metrics && (
                <Collapsible
                  open={expandedSection === 'metrics'}
                  onOpenChange={() => toggleSection('metrics')}
                >
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" className="flex w-full justify-between items-center p-0 h-auto">
                      <span className="text-sm font-medium">Advanced Metrics</span>
                      <ChevronDown
                        className={`h-4 w-4 transition-transform ${
                          expandedSection === 'metrics' ? 'rotate-180' : ''
                        }`}
                      />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <div className="mt-2 p-3 bg-muted/30 rounded-md space-y-2">
                      {Object.entries(result.metrics).map(([key, value]: [string, any]) => (
                        <div key={key} className="flex justify-between">
                          <span className="text-sm text-muted-foreground">
                            {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                          </span>
                          <span className="text-sm">
                            {typeof value === 'number' ? value.toFixed(2) : value.toString()}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              )}
              
              {/* Case ID and timestamp */}
              <div className="text-xs text-muted-foreground mt-4">
                <div className="flex justify-between">
                  <span>Case ID:</span>
                  <span className="font-mono">{result.case_id || "N/A"}</span>
                </div>
                <div className="flex justify-between mt-1">
                  <span>Analysis Date:</span>
                  <span>{formattedDate}</span>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
      
      <CardFooter className="flex justify-between bg-muted/20 p-4">
        <div className="flex space-x-2">
          <Button size="sm" variant="outline" onClick={handleShare}>
            <Share2 className="mr-2 h-4 w-4" />
            Share
          </Button>
          <Button size="sm" variant="outline" onClick={handleExport}>
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
        <div>
          <Button size="sm" variant="ghost">
            <Zap className="mr-2 h-4 w-4" />
            Advanced Analysis
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
};

// Utility components
const InfoIcon: React.FC<{ size?: number, className?: string }> = ({ size = 24, className = "" }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <circle cx="12" cy="12" r="10" />
    <path d="M12 16v-4" />
    <path d="M12 8h.01" />
  </svg>
);

export default AdvancedAnalysisResult;