import React from 'react';
import { useState, useRef } from "react";
import { FileDown, FileText, Share2, Download, FileCheck, AlertCircle } from "lucide-react";
import { Button } from "../ui/button";
import { useQuery } from "@tanstack/react-query";
import { ScanResult } from "../../lib/types";
import AdvancedAnalysisCard from "../shared/AdvancedAnalysisCard";
import { useToast } from "../../hooks/use-toast";
import { createApiUrl } from "../../lib/config";

interface AnalysisResultsProps {
  scanId?: string;
}

export default function AnalysisResults({ scanId }: AnalysisResultsProps) {
  const { data: result, isLoading, error } = useQuery<ScanResult>({
    queryKey: scanId ? [`/api/scans/${scanId}`] : [],
    enabled: !!scanId,
  });
  
  const [isPdfDownloading, setIsPdfDownloading] = useState(false);
  const [isCsvDownloading, setIsCsvDownloading] = useState(false);
  const { toast } = useToast();
  const pdfLinkRef = useRef<HTMLAnchorElement>(null);

  const handleExportPdf = async () => {
    if (!result || !scanId) return;
    
    setIsPdfDownloading(true);
    
    // Create a direct link to the PDF report route
    const reportUrl = await createApiUrl(`/api/scans/${scanId}/report`);
    
    // Option 1: Show download progress with iframe
    try {
      // Create an iframe to trigger the download
      const downloadFrame = document.createElement('iframe');
      downloadFrame.style.display = 'none';
      document.body.appendChild(downloadFrame);
      
      // Set up a load handler to know when the download begins
      downloadFrame.onload = () => {
        toast({
          title: "Report generated successfully!",
          description: "Your PDF report is downloading...",
          variant: "default",

        });
        
        // Clean up the iframe after a delay
        setTimeout(() => {
          document.body.removeChild(downloadFrame);
          setIsPdfDownloading(false);
        }, 1000);
      };
      
      // Set the src to trigger the download
      downloadFrame.src = reportUrl;
    } catch (error) {
      console.error('PDF download error:', error);
      setIsPdfDownloading(false);
      
      toast({
        title: "Download failed",
        description: "There was an issue generating your PDF report. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleExportCsv = () => {
    if (!result) return;
    
    setIsCsvDownloading(true);
    
    try {
      // Generate CSV content
      const csvContent = `
"Filename","Type","Result","Confidence Score","Timestamp"
"${result.filename}","${result.type}","${result.result}","${result.confidenceScore}%","${result.timestamp}"
      `;
      
      // Create download link
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `SatyaAI-Analysis-${result.filename.split('.')[0]}.csv`);
      document.body.appendChild(link);
      
      // Trigger download
      link.click();
      
      // Clean up
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      toast({
        title: "CSV exported successfully",
        description: "Your data has been exported to CSV format",
      });
    } catch (error) {
      console.error('CSV export error:', error);
      
      toast({
        title: "Export failed",
        description: "There was an issue exporting your data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsCsvDownloading(false);
    }
  };

  const handleShare = () => {
    if (!result) return;
    
    try {
      if (navigator.share) {
        navigator.share({
          title: `SatyaAI Analysis: ${result.filename}`,
          text: `SatyaAI detected this media as ${result.result.toUpperCase()} with ${result.confidenceScore}% confidence.`,
          url: window.location.href
        }).catch(err => {
          console.error('Share failed:', err);
        });
      } else {
        // Fallback for browsers that don't support navigator.share
        navigator.clipboard.writeText(window.location.href);
        
        toast({
          title: "Link copied to clipboard",
          description: "You can now share this analysis with others",
          duration: 3000,
        });
      }
    } catch (error) {
      console.error('Share error:', error);
      
      toast({
        title: "Share failed",
        description: "There was an issue sharing this analysis. Please try again.",
        variant: "destructive",
      });
    }
  };

  // Use our advanced card component for the analysis
  return (
    <div className="space-y-6">
      <AdvancedAnalysisCard 
        result={result} 
        isLoading={isLoading} 
      />
      
      {/* Hidden link element for PDF downloads */}
      <a 
        ref={pdfLinkRef}
        className="hidden"
        target="_blank"
        rel="noopener noreferrer"
      />
      
      {/* Action Buttons - only show when we have results */}
      {result && (
        <div className="flex flex-wrap gap-3 justify-end">
          <Button 
            variant="outline" 
            className="border-primary/50 text-primary shadow-[0_0_10px_rgba(0,200,255,0.1)] hover:shadow-[0_0_15px_rgba(0,200,255,0.3)] relative overflow-hidden group"
            onClick={handleExportPdf}
            disabled={isPdfDownloading}
          >
            {/* Animated glow effect */}
            <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-primary/0 via-primary/30 to-primary/0 -translate-x-[100%] group-hover:animate-shimmer" />
            
            {isPdfDownloading ? (
              <>
                <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                <span>Generating PDF...</span>
              </>
            ) : (
              <>
                <FileDown className="mr-2 group-hover:animate-bounce-light" size={16} />
                <span>Export PDF Report</span>
              </>
            )}
          </Button>
          
          <Button 
            variant="outline" 
            className="border-secondary/50 text-secondary shadow-[0_0_10px_rgba(6,214,160,0.1)] hover:shadow-[0_0_15px_rgba(6,214,160,0.3)] relative overflow-hidden group"
            onClick={handleExportCsv}
            disabled={isCsvDownloading}
          >
            {/* Animated glow effect */}
            <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-secondary/0 via-secondary/30 to-secondary/0 -translate-x-[100%] group-hover:animate-shimmer" />
            
            {isCsvDownloading ? (
              <>
                <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                <span>Exporting CSV...</span>
              </>
            ) : (
              <>
                <FileText className="mr-2 group-hover:animate-bounce-light" size={16} />
                <span>Export CSV Data</span>
              </>
            )}
          </Button>
          
          <Button 
            variant="outline" 
            className="border-accent/50 text-accent shadow-[0_0_10px_rgba(131,255,51,0.1)] hover:shadow-[0_0_15px_rgba(131,255,51,0.3)] relative overflow-hidden group"
            onClick={handleShare}
          >
            {/* Animated glow effect */}
            <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-accent/0 via-accent/30 to-accent/0 -translate-x-[100%] group-hover:animate-shimmer" />
            
            <Share2 className="mr-2 group-hover:animate-bounce-light" size={16} />
            <span>Share Results</span>
          </Button>
        </div>
      )}
    </div>
  );
}