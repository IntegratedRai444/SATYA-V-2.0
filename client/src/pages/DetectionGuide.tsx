import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Eye, Mic, Video, Shield, Zap, CheckCircle, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

const DetectionGuide: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#050a0a] pt-20">
      {/* Header */}
      <div className="border-b border-gray-800/50 bg-black/20 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button 
                variant="ghost" 
                onClick={() => navigate('/dashboard')}
                className="text-gray-400 hover:text-white"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back to Dashboard
              </Button>
              <h1 className="text-2xl font-bold text-white">Complete Deepfake Detection Guide</h1>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Overview Section */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-white mb-4">Understanding Deepfakes</h2>
          <p className="text-gray-300 text-lg leading-relaxed max-w-4xl">
            Deepfakes are synthetic media that have been digitally manipulated to replace one person's likeness with another. 
            While the technology can be used for creative purposes, it's increasingly used for misinformation and fraud. 
            This guide helps you identify manipulated content using both human observation and AI-powered analysis.
          </p>
        </div>

        {/* Visual Indicators */}
        <Card className="bg-[#0f1419] border border-gray-800/50 p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Eye className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-2xl font-bold text-white">Visual Indicators</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Face Boundaries
                </h4>
                <p className="text-gray-300 text-sm">
                  Look for unnatural edges around faces, especially near the hairline and jaw. 
                  Blurry or inconsistent boundaries often indicate digital manipulation.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Eye Movements
                </h4>
                <p className="text-gray-300 text-sm">
                  Unnatural blinking patterns, lack of eye contact, or inconsistent eye movement 
                  can indicate deepfake manipulation.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Lighting Inconsistencies
                </h4>
                <p className="text-gray-300 text-sm">
                  Mismatched lighting between the face and background, or inconsistent shadows, 
                  may signal digital alteration.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Skin Texture
                </h4>
                <p className="text-gray-300 text-sm">
                  Unusually smooth or inconsistent skin texture, especially around the cheeks 
                  and forehead, can indicate AI-generated content.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Facial Symmetry
                </h4>
                <p className="text-gray-300 text-sm">
                  While natural faces have asymmetries, deepfakes often create perfect symmetry 
                  or unnatural proportions.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Hair and Accessories
                </h4>
                <p className="text-gray-300 text-sm">
                  Unnatural hair movement, inconsistent rendering of accessories, or 
                  blurry details around hairlines.
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Audio Indicators */}
        <Card className="bg-[#0f1419] border border-gray-800/50 p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-green-500/20 rounded-lg flex items-center justify-center">
              <Mic className="w-5 h-5 text-green-400" />
            </div>
            <h3 className="text-2xl font-bold text-white">Audio Indicators</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Voice Cloning Artifacts
                </h4>
                <p className="text-gray-300 text-sm">
                  Robotic or metallic sounds, unnatural pitch variations, or inconsistent 
                  voice tone can indicate synthetic audio.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Breathing Patterns
                </h4>
                <p className="text-gray-300 text-sm">
                  Missing or unnatural breathing sounds, inconsistent pauses, or lack of 
                  natural speech rhythm.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Background Noise
                </h4>
                <p className="text-gray-300 text-sm">
                  Unusually clean audio, missing background ambiance, or inconsistent 
                  environmental sounds.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Audio-Visual Sync
                </h4>
                <p className="text-gray-300 text-sm">
                  Lip movements that don't match the audio, timing delays between 
                  speech and mouth movements.
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Video Indicators */}
        <Card className="bg-[#0f1419] border border-gray-800/50 p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-purple-500/20 rounded-lg flex items-center justify-center">
              <Video className="w-5 h-5 text-purple-400" />
            </div>
            <h3 className="text-2xl font-bold text-white">Video Indicators</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Lip-Sync Issues
                </h4>
                <p className="text-gray-300 text-sm">
                  Mismatched lip movements, delayed speech synchronization, or 
                  unnatural mouth shapes.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Temporal Inconsistencies
                </h4>
                <p className="text-gray-300 text-sm">
                  Jerky movements, sudden changes in appearance, or inconsistent 
                  motion patterns between frames.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Flickering Artifacts
                </h4>
                <p className="text-gray-300 text-sm">
                  Subtle flickering around the face, especially during movement, 
                  or digital artifacts in transitions.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Unnatural Physics
                </h4>
                <p className="text-gray-300 text-sm">
                  Hair or clothing that doesn't move naturally, inconsistent 
                  shadows, or impossible lighting scenarios.
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Metadata and Context */}
        <Card className="bg-[#0f1419] border border-gray-800/50 p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-cyan-500/20 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-cyan-400" />
            </div>
            <h3 className="text-2xl font-bold text-white">Metadata and Context Analysis</h3>
          </div>

          <div className="space-y-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-white font-semibold mb-2">Digital Footprints</h4>
              <p className="text-gray-300 text-sm">
                Check creation dates, modification history, and source credibility. 
                Legitimate content typically has consistent metadata and verifiable origins.
              </p>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <h4 className="text-white font-semibold mb-2">Source Verification</h4>
              <p className="text-gray-300 text-sm">
                Cross-reference with original sources, check for watermarks, and verify 
                the content's context within the broader narrative.
              </p>
            </div>
          </div>
        </Card>

        {/* How SatyaAI Helps */}
        <Card className="bg-[#0f1419] border border-gray-800/50 p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-blue-400" />
            </div>
            <h3 className="text-2xl font-bold text-white">How SatyaAI Enhances Detection</h3>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Multi-Modal Analysis
                </h4>
                <p className="text-gray-300 text-sm">
                  SatyaAI combines visual, audio, and metadata analysis using multiple 
                  specialized AI models for comprehensive detection.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Signal Fusion
                </h4>
                <p className="text-gray-300 text-sm">
                  Individual detection signals are fused using advanced algorithms 
                  to create highly accurate authenticity scores.
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Real-Time Processing
                </h4>
                <p className="text-gray-300 text-sm">
                  Get instant analysis results with confidence scores and detailed 
                  explanations of detected anomalies.
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <h4 className="text-white font-semibold mb-2 flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  Cryptographic Proof
                </h4>
                <p className="text-gray-300 text-sm">
                  Each analysis includes cryptographic verification for audit trails 
                  and evidence preservation.
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Call to Action */}
        <div className="text-center py-8">
          <h3 className="text-2xl font-bold text-white mb-4">Ready to Analyze Media?</h3>
          <p className="text-gray-300 mb-6">
            Put your knowledge into practice with our advanced AI-powered detection tools.
          </p>
          <Button 
            onClick={() => navigate('/dashboard')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-semibold"
          >
            Go to Dashboard
          </Button>
        </div>
      </div>
    </div>
  );
};

export default DetectionGuide;
