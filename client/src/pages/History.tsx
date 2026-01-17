import { useRef, useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { useAnalysisHistory } from '../hooks/useApi';

export default function History() {
  const { history, clearHistory } = useAnalysisHistory();
  const containerRef = useRef<HTMLDivElement>(null);
  const [listHeight, setListHeight] = useState(600);

  // Calculate list height based on container
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const availableHeight = window.innerHeight - rect.top - 200; // Leave space for stats
        setListHeight(Math.max(400, availableHeight));
      }
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => {
      window.removeEventListener('resize', updateHeight);
    };
  }, []);

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const getResultColor = (authenticity: string) => {
    switch (authenticity) {
      case 'AUTHENTIC MEDIA':
        return 'text-green-500';
      case 'MANIPULATED MEDIA':
        return 'text-red-500';
      default:
        return 'text-yellow-500';
    }
  };

  // Map API response to UI format
  const renderHistoryItem = (result: any, index: number) => {
    // Map API response to UI format
    const uiItem = {
      id: result.id || result.jobId,
      case_id: result.reportCode || result.jobId,
      analysis_date: result.timestamp || result.created_at,
      authenticity: result.is_deepfake ? 'MANIPULATED MEDIA' : 'AUTHENTIC MEDIA',
      confidence: result.confidence || 0,
      analysis_type: result.modality || result.type || 'Unknown',
      key_findings: result.key_findings || result.analysis_data?.findings || []
    };

    return (
      <Card key={uiItem.id} className="mb-4">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">
              Analysis #{history.length - index}
            </CardTitle>
            <span className="text-sm text-muted-foreground">
              {formatDate(uiItem.analysis_date)}
            </span>
          </div>
          <CardDescription>
            Case ID: {uiItem.case_id}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-medium mb-1">Result</h4>
              <p className={`font-semibold ${getResultColor(uiItem.authenticity)}`}>
                {uiItem.authenticity}
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Confidence</h4>
              <p className="font-semibold">
                {(uiItem.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Type</h4>
              <p className="font-semibold capitalize">
                {uiItem.analysis_type}
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Case ID</h4>
              <p className="font-semibold">
                {uiItem.case_id}
              </p>
            </div>
          </div>

          {uiItem.key_findings && uiItem.key_findings.length > 0 && (
            <div className="mt-4">
              <h4 className="font-medium mb-2">Key Findings</h4>
              <div className="bg-muted rounded-lg p-3">
                <ul className="space-y-1 text-sm">
                  {uiItem.key_findings.slice(0, 3).map((finding: string, findingIndex: number) => (
                    <li key={findingIndex} className="flex items-start gap-2">
                      <span className="text-primary">•</span>
                      <span>{finding}</span>
                    </li>
                  ))}
                  {uiItem.key_findings.length > 3 && (
                    <li className="text-muted-foreground text-xs">
                      +{uiItem.key_findings.length - 3} more findings...
                    </li>
                  )}
                </ul>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-6 bg-bg-primary min-h-screen">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Analysis History</h1>
          <p className="text-text-secondary">Review past deepfake detection results</p>
        </div>
        {history.length > 0 && (
          <Button variant="outline" onClick={clearHistory}>
            Clear History
          </Button>
        )}
      </div>

      {history.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center p-12">
            <div className="text-6xl mb-4">📊</div>
            <h3 className="text-xl font-semibold mb-2">No Analysis History</h3>
            <p className="text-muted-foreground text-center mb-4">
              Your analysis results will appear here after you start analyzing media files.
            </p>
            <Button onClick={() => window.location.href = '/scan'}>
              Start First Analysis
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div 
          ref={containerRef}
          style={{ maxHeight: `${listHeight}px` }}
          className="overflow-y-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-200 pr-2"
        >
          {history.map((result: any, index: number) => renderHistoryItem(result, index))}
        </div>
      )}
      <Card>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-primary">{history.length}</p>
              <p className="text-sm text-muted-foreground">Total Analyses</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-red-500">
                {history.filter((h: any) => h.authenticity === 'MANIPULATED MEDIA').length}
              </p>
              <p className="text-sm text-muted-foreground">Manipulated</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}