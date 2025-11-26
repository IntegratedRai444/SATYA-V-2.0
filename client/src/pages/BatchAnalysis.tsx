import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Upload, Download, History, Settings, BarChart3 } from 'lucide-react';
import BatchUploader from '@/components/batch/BatchUploader';
import { toast } from 'react-hot-toast';

const BatchAnalysis = () => {
  const [batchHistory] = useState([
    {
      id: 'batch_001',
      date: '2024-01-07 14:30',
      filesCount: 25,
      authentic: 18,
      manipulated: 7,
      avgConfidence: 87.3,
      status: 'completed'
    },
    {
      id: 'batch_002', 
      date: '2024-01-06 09:15',
      filesCount: 12,
      authentic: 9,
      manipulated: 3,
      avgConfidence: 91.2,
      status: 'completed'
    },
    {
      id: 'batch_003',
      date: '2024-01-05 16:45',
      filesCount: 8,
      authentic: 5,
      manipulated: 2,
      avgConfidence: 84.7,
      status: 'processing'
    }
  ]);



  const exportBatchResults = (batchId: string) => {
    toast.success(`Exporting results for ${batchId}`);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Batch Analysis</h1>
        <p className="text-muted-foreground">
          Process multiple files simultaneously for efficient deepfake detection
        </p>
      </div>

      <Tabs defaultValue="upload" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4 lg:w-[400px]">
          <TabsTrigger value="upload" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Upload
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center gap-2">
            <History className="h-4 w-4" />
            History
          </TabsTrigger>
          <TabsTrigger value="analytics" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Analytics
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex items-center gap-2">
            <Settings className="h-4 w-4" />
            Settings
          </TabsTrigger>
        </TabsList>

        {/* Upload Tab */}
        <TabsContent value="upload">
          <BatchUploader />
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <History className="h-5 w-5" />
                Batch Processing History
              </CardTitle>
              <CardDescription>
                View and manage your previous batch analysis sessions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {batchHistory.map((batch) => (
                  <div key={batch.id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <h3 className="font-semibold">{batch.id}</h3>
                        <p className="text-sm text-muted-foreground">{batch.date}</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge 
                          variant={batch.status === 'completed' ? 'default' : 'secondary'}
                        >
                          {batch.status}
                        </Badge>
                        {batch.status === 'completed' && (
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => exportBatchResults(batch.id)}
                          >
                            <Download className="h-4 w-4 mr-2" />
                            Export
                          </Button>
                        )}
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Total Files</p>
                        <p className="font-semibold">{batch.filesCount}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Authentic</p>
                        <p className="font-semibold text-green-600">{batch.authentic}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Manipulated</p>
                        <p className="font-semibold text-red-600">{batch.manipulated}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Avg Confidence</p>
                        <p className="font-semibold">{batch.avgConfidence.toFixed(1)}%</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Analytics Tab */}
        <TabsContent value="analytics" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Batch Processing Stats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>Total Batches Processed</span>
                    <span className="font-semibold">{batchHistory.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Total Files Analyzed</span>
                    <span className="font-semibold">
                      {batchHistory.reduce((acc, batch) => acc + batch.filesCount, 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Average Batch Size</span>
                    <span className="font-semibold">
                      {(batchHistory.reduce((acc, batch) => acc + batch.filesCount, 0) / batchHistory.length).toFixed(1)} files
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Overall Accuracy</span>
                    <span className="font-semibold">
                      {(batchHistory.reduce((acc, batch) => acc + batch.avgConfidence, 0) / batchHistory.length).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Processing Efficiency</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>Avg Processing Time</span>
                    <span className="font-semibold">2.3s per file</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Success Rate</span>
                    <span className="font-semibold text-green-600">98.7%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Queue Wait Time</span>
                    <span className="font-semibold">&lt; 1 minute</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Concurrent Processing</span>
                    <span className="font-semibold">Up to 5 files</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Batch Processing Settings</CardTitle>
              <CardDescription>
                Configure how batch processing works for your account
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Processing Options</h4>
                  <div className="space-y-2">
                    <label className="flex items-center space-x-2">
                      <input type="checkbox" defaultChecked className="rounded" />
                      <span className="text-sm">Enable parallel processing</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input type="checkbox" defaultChecked className="rounded" />
                      <span className="text-sm">Auto-generate reports</span>
                    </label>
                    <label className="flex items-center space-x-2">
                      <input type="checkbox" className="rounded" />
                      <span className="text-sm">Email notifications on completion</span>
                    </label>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">File Limits</h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-muted-foreground">Max files per batch</label>
                      <input 
                        type="number" 
                        defaultValue={50} 
                        className="w-full mt-1 px-3 py-2 border rounded-md" 
                      />
                    </div>
                    <div>
                      <label className="text-sm text-muted-foreground">Max file size (MB)</label>
                      <input 
                        type="number" 
                        defaultValue={50} 
                        className="w-full mt-1 px-3 py-2 border rounded-md" 
                      />
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Output Format</h4>
                  <select className="w-full px-3 py-2 border rounded-md">
                    <option value="json">JSON Report</option>
                    <option value="csv">CSV Export</option>
                    <option value="pdf">PDF Report</option>
                    <option value="all">All Formats</option>
                  </select>
                </div>

                <Button className="w-full">
                  Save Settings
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BatchAnalysis;
