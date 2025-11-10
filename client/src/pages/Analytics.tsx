import { useDashboardStats } from '../hooks/useDashboardStats';
import { useAnalytics } from '../hooks/useAnalytics';
import { Button } from '../components/ui/button';
import { Download } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';

export default function Analytics() {
  const { data: stats, error } = useDashboardStats();
  const { exportData } = useAnalytics('30d');

  if (error) {
    return (
      <div className="p-6 bg-bg-primary min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-text-primary mb-2">Failed to load analytics</h2>
          <p className="text-text-secondary">Please try refreshing the page</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 bg-bg-primary min-h-screen">
      {/* Header */}
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-text-primary mb-2">Analytics & Insights</h1>
          <p className="text-text-secondary">System performance and detection tips</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline" 
            onClick={() => exportData('json')}
            className="flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export JSON
          </Button>
          <Button 
            variant="outline" 
            onClick={() => exportData('csv')}
            className="flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Total Scans</CardTitle>
            <CardDescription>All time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.analyzedMedia?.count || 0}</div>
            <p className="text-sm text-green-500">{stats?.analyzedMedia?.growth || '+0%'}</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Deepfakes Detected</CardTitle>
            <CardDescription>This month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.detectedDeepfakes?.count || 0}</div>
            <p className="text-sm text-red-500">{stats?.detectedDeepfakes?.growth || '+0%'}</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Avg Detection Time</CardTitle>
            <CardDescription>Per scan</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.avgDetectionTime?.time || '0s'}</div>
            <p className="text-sm text-green-500">{stats?.avgDetectionTime?.improvement || '0%'}</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Accuracy</CardTitle>
            <CardDescription>Overall</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.detectionAccuracy?.percentage || 0}%</div>
            <p className="text-sm text-green-500">{stats?.detectionAccuracy?.improvement || '+0%'}</p>
          </CardContent>
        </Card>
      </div>

      {/* Activity Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Detection Activity</CardTitle>
          <CardDescription>Daily analysis trends</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-end justify-between gap-2">
            {stats?.dailyActivity?.slice(0, 20).map((item, i) => (
              <div 
                key={i}
                className="flex-1 bg-blue-500 rounded-t hover:bg-blue-600 transition-colors"
                style={{ height: `${(item.analyses / 70) * 100}%` }}
                title={`${item.date}: ${item.analyses} analyses`}
              />
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}