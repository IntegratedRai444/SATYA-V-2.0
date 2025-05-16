import { useState } from "react";
import { FileDown, FileText, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { ScanResult } from "@/lib/types";
import AdvancedAnalysisCard from "@/components/shared/AdvancedAnalysisCard";

interface AnalysisResultsProps {
  scanId?: string;
}

export default function AnalysisResults({ scanId }: AnalysisResultsProps) {
  const { data: result, isLoading, error } = useQuery<ScanResult>({
    queryKey: scanId ? [`/api/scans/${scanId}`] : [],
    enabled: !!scanId,
  });

  const handleExportPdf = () => {
    window.print();
  };

  const handleExportCsv = () => {
    // In a full implementation, this would generate and download a CSV
    if (!result) return;
    
    const csvContent = `
    "Filename","Type","Result","Confidence Score","Timestamp"
    "${result.filename}","${result.type}","${result.result}","${result.confidenceScore}%","${result.timestamp}"
    `;
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `${result.filename.split('.')[0]}_analysis.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleShare = () => {
    // In a full implementation, this would open a share dialog
    if (!result) return;
    
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
      // Copy current URL to clipboard
      navigator.clipboard.writeText(window.location.href);
      alert('URL copied to clipboard!');
    }
  };

  // Use our advanced card component for the analysis
  return (
    <div className="space-y-6">
      <AdvancedAnalysisCard 
        result={result} 
        isLoading={isLoading} 
      />
      
      {/* Action Buttons - only show when we have results */}
      {result && (
        <div className="flex flex-wrap gap-3 justify-end">
          <Button 
            variant="outline" 
            className="border-primary/50 text-primary shadow-[0_0_10px_rgba(0,200,255,0.1)] hover:shadow-[0_0_15px_rgba(0,200,255,0.3)]" 
            onClick={handleExportPdf}
          >
            <FileDown className="mr-2" size={16} />
            <span>Export PDF Report</span>
          </Button>
          <Button 
            variant="outline" 
            className="border-secondary/50 text-secondary shadow-[0_0_10px_rgba(6,214,160,0.1)] hover:shadow-[0_0_15px_rgba(6,214,160,0.3)]"
            onClick={handleExportCsv}
          >
            <FileText className="mr-2" size={16} />
            <span>Export CSV Data</span>
          </Button>
          <Button 
            variant="outline" 
            className="border-accent/50 text-accent shadow-[0_0_10px_rgba(131,255,51,0.1)] hover:shadow-[0_0_15px_rgba(131,255,51,0.3)]"
            onClick={handleShare}
          >
            <Share2 className="mr-2" size={16} />
            <span>Share Results</span>
          </Button>
        </div>
      )}
    </div>
  );
}