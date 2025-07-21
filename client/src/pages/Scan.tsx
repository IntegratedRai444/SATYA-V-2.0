import React from 'react';
import { useEffect, useState } from "react";
import { Helmet } from 'react-helmet';
import { useLocation } from "wouter";
import UploadSection from "../components/upload/UploadSection";
import AdvancedScanSection from "../components/upload/AdvancedScanSection";
import AnalysisResults from "../components/results/AnalysisResults";
import { Card, CardContent } from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Button } from "../components/ui/button";
import { Zap } from "lucide-react";

export default function Scan() {
  const [location] = useLocation();
  const [activeTab, setActiveTab] = useState("upload");
  const [scanId, setScanId] = useState<string | undefined>(undefined);
  const [useAdvancedMode, setUseAdvancedMode] = useState(false);

  // Parse query parameters
  const searchParams = new URLSearchParams(location.split("?")[1] || "");
  const mediaType = searchParams.get("type") || "image";
  const result = searchParams.get("result") || undefined;
  const advancedParam = searchParams.get("advanced") || undefined;

  // Check if advanced mode is requested
  useEffect(() => {
    if (advancedParam === "true") {
      setUseAdvancedMode(true);
    }
  }, [advancedParam]);

  // If result ID is provided, show results tab
  useEffect(() => {
    if (result) {
      setScanId(result);
      setActiveTab("results");
    }
  }, [result]);

  return (
    <>
      <Helmet>
        <title>SatyaAI - Media Scan & Analysis</title>
        <meta name="description" content="Upload and analyze media for deepfakes. SatyaAI's advanced algorithms can detect manipulated images, videos, and audio with high accuracy." />
      </Helmet>
      
      <div className="mb-6 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold mb-2">Media Scan</h1>
          <p className="text-muted-foreground">
            Upload media for deepfake detection or analyze in real-time using your webcam.
          </p>
        </div>
        <Button 
          className={`${useAdvancedMode ? 'bg-secondary' : 'bg-primary'} flex items-center gap-2`}
          onClick={() => setUseAdvancedMode(!useAdvancedMode)}
        >
          <Zap size={16} />
          {useAdvancedMode ? 'Using Advanced Mode' : 'Switch to Advanced Mode'}
        </Button>
      </div>
      
      <Card>
        <CardContent className="p-0">
          <Tabs 
            defaultValue={activeTab} 
            value={activeTab} 
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList className="w-full grid grid-cols-2">
              <TabsTrigger value="upload">Upload & Scan</TabsTrigger>
              <TabsTrigger value="results" disabled={!scanId}>View Results</TabsTrigger>
            </TabsList>
            
            <TabsContent value="upload" className="p-6">
              {useAdvancedMode ? (
                <AdvancedScanSection />
              ) : (
                <UploadSection />
              )}
            </TabsContent>
            
            <TabsContent value="results" className="p-6">
              {scanId ? (
                <AnalysisResults scanId={scanId} />
              ) : (
                <div className="text-center py-12">
                  <p className="text-muted-foreground">
                    No analysis results to display. Please upload and scan media first.
                  </p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </>
  );
}
