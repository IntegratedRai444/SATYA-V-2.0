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
    return () => window.removeEventListener('resize', updateHeight);
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

  // Render individual history item
  const renderHistoryItem = (result: any, index: number) => {
    return (
      <Card key={result.case_id || index} className="mb-4">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg">
              Analysis #{history.length - index}
            </CardTitle>
            <span className="text-sm text-muted-foreground">
              {formatDate(result.analysis_date)}
            </span>
          </div>
          <CardDescription>
            Case ID: {result.case_id}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-medium mb-1">Result</h4>
              <p className={`font-semibold ${getResultColor(result.authenticity)}`}>
                {result.authenticity}
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Confidence</h4>
              <p className="font-semibold">
                {result.confidence.toFixed(1)}%
              </p>
            </div>
            <div>
              <h4 className="font-medium mb-1">Type</h4>
              <p className="font-semibold capitalize">
                {result.analysis_type || 'Unknown'}
              </p>
            </div>
          </div>

          {result.key_findings && result.key_findings.length > 0 && (
            <div className="mt-4">
              <h4 className="font-medium mb-2">Key Findings</h4>
              <div className="bg-muted rounded-lg p-3">
                <ul className="space-y-1 text-sm">
                  {result.key_findings.slice(0, 3).map((finding: string, findingIndex: number) => (
                    <li key={findingIndex} className="flex items-start gap-2">
                      <span className="text-primary">•</span>
                      <span>{finding}</span>
                    </li>
                  ))}
                  {result.key_findings.length > 3 && (
                    <li className="text-muted-foreground text-xs">
                      +{result.key_findings.length - 3} more findings...
                    </li>
                  )}
                </ul>
              </div>
            </div>
          )}

          {result.metrics && (
            <div className="mt-4">
              <h4 className="font-medium mb-2">Metrics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                {Object.entries(result.metrics).map(([key, value]: [string, any]) => (
                  <div key={key} className="bg-muted rounded p-2">
                    <p className="text-xs text-muted-foreground capitalize">
                      {key.replace(/_/g, ' ')}
                    </p>
                    <p className="font-medium">
                      {typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : value}
                    </p>
                  </div>
                ))}
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

      {/* Statistics */}
      {history.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Statistics</CardTitle>
            <CardDescription>Analysis summary</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{history.length}</p>
                <p className="text-sm text-muted-foreground">Total Analyses</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-green-500">
                  {history.filter((h: any) => h.authenticity === 'AUTHENTIC MEDIA').length}
                </p>
                <p className="text-sm text-muted-foreground">Authentic</p>
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
      )}
    </div>
  );
}