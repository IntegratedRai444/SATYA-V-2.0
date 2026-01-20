import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Upload, Download, History, Settings, BarChart3, AlertTriangle } from 'lucide-react';
import { toast } from 'react-hot-toast';

const BatchAnalysis = () => {
  // Feature disabled - show message instead of hardcoded data
  const [isFeatureEnabled] = useState(false);

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

      {!isFeatureEnabled && (
        <Card className="mb-6 border-orange-200 bg-orange-50">
          <CardContent className="pt-6">
            <div className="flex items-center space-x-3">
              <AlertTriangle className="h-5 w-5 text-orange-600" />
              <div>
                <h3 className="font-semibold text-orange-800">Feature Under Development</h3>
                <p className="text-sm text-orange-600 mt-1">
                  Batch analysis is currently being optimized. Please use individual file analysis for now.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="upload" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upload" disabled={!isFeatureEnabled}>
            <Upload className="h-4 w-4 mr-2" />
            Upload Files
          </TabsTrigger>
          <TabsTrigger value="history" disabled={!isFeatureEnabled}>
            <History className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle>Upload Multiple Files</CardTitle>
              <CardDescription>
                Select and upload multiple files for batch processing
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <Upload className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Batch upload feature is currently disabled</p>
                <p className="text-sm">Please use individual file analysis from the main menu</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Batch Processing History</CardTitle>
              <CardDescription>
                View and manage your previous batch analysis sessions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-muted-foreground">
                <History className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No batch history available</p>
                <p className="text-sm">Start with individual analyses to track your progress</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default BatchAnalysis;
