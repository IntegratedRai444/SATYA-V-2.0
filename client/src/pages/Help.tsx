import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import ChatInterface from '@/components/chat/ChatInterface';
import { MessageCircle, X } from 'lucide-react';

export default function Help() {
  const [showChat, setShowChat] = useState(false);
  
  return (
    <div className="container mx-auto p-6 space-y-6 bg-bg-primary min-h-screen">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-text-primary">Help & Documentation</h1>
          <p className="text-text-secondary">Learn how to use SatyaAI effectively</p>
        </div>
        <Button 
          onClick={() => setShowChat(!showChat)}
          className="flex items-center gap-2"
        >
          {showChat ? <X className="w-4 h-4" /> : <MessageCircle className="w-4 h-4" />}
          {showChat ? 'Close Chat' : 'AI Assistant'}
        </Button>
      </div>
      
      {/* Chat Interface */}
      {showChat && (
        <Card className="border-2 border-blue-500/50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageCircle className="w-5 h-5 text-blue-500" />
              AI Assistant
            </CardTitle>
            <CardDescription>Ask me anything about SatyaAI</CardDescription>
          </CardHeader>
          <CardContent>
            <ChatInterface />
          </CardContent>
        </Card>
      )}

      {/* Quick Start Guide */}
      <Card>
        <CardHeader>
          <CardTitle>🚀 Quick Start Guide</CardTitle>
          <CardDescription>Get started with SatyaAI in minutes</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Step 1: Upload Media</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Go to the Scan page</li>
                <li>• Drag and drop your file or click to browse</li>
                <li>• Supported: Images, Videos, Audio files</li>
                <li>• Maximum file size: 50MB</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Step 2: Analyze</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Click the "Analyze" button</li>
                <li>• Wait for processing to complete</li>
                <li>• Review the detection results</li>
                <li>• Check confidence scores and findings</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Features Overview */}
      <Card>
        <CardHeader>
          <CardTitle>🔍 Features Overview</CardTitle>
          <CardDescription>Understanding SatyaAI's capabilities</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold mb-2">Image Analysis</h4>
              <p className="text-sm text-muted-foreground">
                Detect manipulated images using advanced CNN models. Analyzes facial features, 
                lighting consistency, and metadata.
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold mb-2">Video Detection</h4>
              <p className="text-sm text-muted-foreground">
                Frame-by-frame analysis with temporal consistency checks. Detects unnatural 
                movements and transitions.
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold mb-2">Audio Verification</h4>
              <p className="text-sm text-muted-foreground">
                Voice cloning detection using spectral analysis. Identifies synthetic speech 
                and audio manipulation.
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold mb-2">Multimodal Analysis</h4>
              <p className="text-sm text-muted-foreground">
                Combines multiple detection methods for comprehensive analysis. Cross-validates 
                results across modalities.
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold mb-2">Real-time Detection</h4>
              <p className="text-sm text-muted-foreground">
                Advanced analysis algorithms for instant deepfake detection. Get immediate 
                feedback and detailed authenticity reports.
              </p>
            </div>
            <div className="border rounded-lg p-4">
              <h4 className="font-semibold mb-2">Advanced Features</h4>
              <p className="text-sm text-muted-foreground">
                Blockchain verification, dark web checks, and emotion conflict analysis 
                for comprehensive verification.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* FAQ */}
      <Card>
        <CardHeader>
          <CardTitle>❓ Frequently Asked Questions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div>
              <h4 className="font-semibold mb-2">What file formats are supported?</h4>
              <p className="text-sm text-muted-foreground">
                Images: JPG, PNG, GIF, BMP, WebP<br/>
                Videos: MP4, AVI, MOV, WebM<br/>
                Audio: MP3, WAV, AAC, OGG
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">How accurate is the detection?</h4>
              <p className="text-sm text-muted-foreground">
                SatyaAI uses state-of-the-art AI models with high accuracy rates. However, 
                results should be interpreted alongside other evidence and expert analysis.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Is my data secure?</h4>
              <p className="text-sm text-muted-foreground">
                All uploads are processed securely and are not stored permanently on our servers. 
                Analysis results are kept locally in your browser.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">What do confidence scores mean?</h4>
              <p className="text-sm text-muted-foreground">
                Confidence scores indicate how certain the AI model is about its prediction. 
                Higher scores (above 80%) indicate stronger confidence in the result.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Technical Information */}
      <Card>
        <CardHeader>
          <CardTitle>🔧 Technical Information</CardTitle>
          <CardDescription>For developers and technical users</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">API Endpoints</h4>
              <div className="space-y-2 text-sm font-mono bg-muted p-3 rounded">
                <p>POST /api/analyze/image</p>
                <p>POST /api/analyze/video</p>
                <p>POST /api/analyze/audio</p>
                <p>POST /api/analyze/multimodal</p>
                <p>GET /health</p>
              </div>
            </div>
            <div>
              <h4 className="font-semibold mb-3">Model Information</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• CNN-based image analysis</li>
                <li>• LSTM+CNN for video processing</li>
                <li>• WaveNet for audio analysis</li>
                <li>• Transformer-based fusion</li>
                <li>• Real-time processing optimized</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Contact & Support */}
      <Card>
        <CardHeader>
          <CardTitle>📞 Contact & Support</CardTitle>
          <CardDescription>Get help when you need it</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-3">Getting Help</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Check this help documentation first</li>
                <li>• Review the FAQ section above</li>
                <li>• Check browser console for error messages</li>
                <li>• Ensure stable internet connection</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3">System Requirements</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Modern web browser (Chrome, Firefox, Safari)</li>
                <li>• JavaScript enabled</li>
                <li>• Stable internet connection</li>
                <li>• Minimum 4GB RAM recommended</li>
              </ul>
            </div>
          </div>
          <div className="mt-6 pt-6 border-t">
            <div className="flex gap-4">
              <Button variant="outline">
                View Documentation
              </Button>
              <Button variant="outline">
                Report Issue
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}