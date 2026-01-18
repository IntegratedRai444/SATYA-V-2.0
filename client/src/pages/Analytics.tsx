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
            <div className="text-3xl font-bold">{stats?.totalAnalyses || 0}</div>
            <p className="text-sm text-green-500">+0%</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Deepfakes Detected</CardTitle>
            <CardDescription>This month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.manipulatedMedia || 0}</div>
            <p className="text-sm text-red-500">+0%</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Authentic Media</CardTitle>
            <CardDescription>This month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.authenticMedia || 0}</div>
            <p className="text-sm text-green-500">+0%</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Uncertain Scans</CardTitle>
            <CardDescription>This month</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.uncertainScans || 0}</div>
            <p className="text-sm text-yellow-500">+0%</p>
          </CardContent>
        </Card>
      </div>

      {/* Activity Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>Latest analysis results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {stats?.recentActivity?.slice(0, 10).map((item: any, i: number) => (
              <div 
                key={item.id || i}
                className="flex items-center justify-between p-2 border rounded"
              >
                <span className="text-sm">{item.type}</span>
                <span className="text-xs text-gray-500">{item.date}</span>
                <span className={`text-xs px-2 py-1 rounded ${
                  item.status === 'completed' ? 'bg-green-100 text-green-800' :
                  item.status === 'failed' ? 'bg-red-100 text-red-800' :
                  'bg-yellow-100 text-yellow-800'
                }`}>
                  {item.status}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}