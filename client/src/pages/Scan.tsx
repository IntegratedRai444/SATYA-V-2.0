import { useEffect, useState } from "react";
import { Helmet } from 'react-helmet';
import { useLocation } from "wouter";
import UploadSection from "@/components/upload/UploadSection";
import AnalysisResults from "@/components/results/AnalysisResults";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Scan() {
  const [location] = useLocation();
  const [activeTab, setActiveTab] = useState("upload");
  const [scanId, setScanId] = useState<string | undefined>(undefined);

  // Parse query parameters
  const searchParams = new URLSearchParams(location.split("?")[1] || "");
  const mediaType = searchParams.get("type") || "image";
  const result = searchParams.get("result") || undefined;

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
      
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Media Scan</h1>
        <p className="text-muted-foreground">
          Upload media for deepfake detection or analyze in real-time using your webcam.
        </p>
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
              <UploadSection />
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
