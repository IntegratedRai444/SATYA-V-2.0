import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTextAnalysis } from '@/hooks/useApi';
import { pollAnalysisResult, AnalysisJobStatus } from '@/lib/analysis/pollResult';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, FileText, CheckCircle, AlertCircle, Brain } from 'lucide-react';

interface TextAnalysisResult {
  success: boolean;
  is_ai_generated: boolean;
  confidence: number;
  explanation: string;
  model_name: string;
}

const TextAnalysis: React.FC = () => {
  const navigate = useNavigate();
  const [text, setText] = useState('');
  const [result, setResult] = useState<TextAnalysisResult | null>(null);
  const [error, setError] = useState('');
  const [jobId, setJobId] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<'idle' | 'processing' | 'completed' | 'failed'>('idle');
  
  const { analyzeText, isAnalyzing } = useTextAnalysis();

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    try {
      setError('');
      setResult(null);
      setAnalysisStatus('processing');
      
      const jobResult = await analyzeText({ text });
      setJobId(jobResult.jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
      setAnalysisStatus('failed');
    }
  };

  // Poll for results when jobId is available
  useEffect(() => {
    if (jobId && analysisStatus === 'processing') {
      const polling = pollAnalysisResult(jobId, {
        onProgress: () => {
          // Update progress if needed
        }
      });
      
      // Handle polling promise
      polling.promise
        .then((job: AnalysisJobStatus) => {
          if (job.status === 'completed' && job.result) {
            const analysisResult: TextAnalysisResult = {
              success: true,
              is_ai_generated: !job.result.isAuthentic, // For text, AI-generated = not authentic
              confidence: job.result.confidence,
              explanation: `Analysis completed using ${job.result.details.modelInfo?.modelName || 'Text Analysis Model'}. ${job.result.details.features ? `Key indicators: ${Object.keys(job.result.details.features).join(', ')}` : ''}`,
              model_name: job.result.metrics.modelVersion || 'Text Analysis Model'
            };
            setResult(analysisResult);
            setAnalysisStatus('completed');
          } else if (job.status === 'failed') {
            setError(job.error || 'Analysis failed');
            setAnalysisStatus('failed');
          }
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : 'Analysis failed');
          setAnalysisStatus('failed');
        });
      
      return polling.cancel; // Return proper cleanup function
    }
  }, [jobId, analysisStatus]);

  const handleClear = () => {
    setText('');
    setResult(null);
    setError('');
    setJobId(null);
    setAnalysisStatus('idle');
  };

  return (
    <div className="min-h-screen bg-[#0f1419] text-white p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Button
            variant="ghost"
            onClick={() => navigate('/dashboard')}
            className="mb-4 text-gray-400 hover:text-white"
          >
            ← Back to Dashboard
          </Button>
          
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-md bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
              <FileText className="w-5 h-5 text-blue-400" strokeWidth={2} />
            </div>
            <h1 className="text-2xl font-bold text-white">Text Authenticity Analysis</h1>
          </div>
          
          <p className="text-gray-400">
            Analyze text content for AI-generated patterns using advanced NLP models
          </p>
        </div>

        {/* Input Section */}
        <Card className="bg-[#1e2128] border border-gray-800/50 p-6 mb-6">
          <div className="space-y-4">
            <div>
              <label htmlFor="text-input" className="block text-sm font-medium text-white mb-2">
                Enter Text for Analysis
              </label>
              <textarea
                id="text-input"
                value={text}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setText(e.target.value)}
                placeholder="Paste or type of text you want to analyze for AI-generated content..."
                className="w-full min-h-[200px] bg-[#0f1419] border border-gray-700 text-white placeholder-gray-500 p-3 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isAnalyzing}
              />
              <div className="mt-2 text-sm text-gray-400">
                {text.length} characters • {text.split(/\s+/).filter(word => word.length > 0).length} words
              </div>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="flex gap-3">
              <Button
                onClick={handleAnalyze}
                disabled={isAnalyzing || !text.trim()}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed text-white flex items-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    Analyze Text
                  </>
                )}
              </Button>
              
              <Button
                variant="outline"
                onClick={handleClear}
                disabled={isAnalyzing}
                className="border-gray-600 text-gray-300 hover:bg-gray-700"
              >
                Clear
              </Button>
            </div>
          </div>
        </Card>

        {/* Results Section */}
        {result && (
          <Card className="bg-[#2a2e39] border border-gray-700/50 p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                result.is_ai_generated ? 'bg-red-500/10' : 'bg-green-500/10'
              }`}>
                {result.is_ai_generated ? (
                  <AlertCircle className="w-6 h-6 text-red-500" />
                ) : (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                )}
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">
                  {result.is_ai_generated ? 'AI-Generated Text Detected' : 'Human-Written Text Detected'}
                </h3>
                <p className="text-gray-400 text-sm">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4 text-blue-400" />
                  Analysis Details
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Model:</span>
                    <span className="text-white">{result.model_name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">AI Probability:</span>
                    <span className="text-white">
                      {result.is_ai_generated ? `${(result.confidence * 100).toFixed(1)}%` : `${((1 - result.confidence) * 100).toFixed(1)}%`}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Human Probability:</span>
                    <span className="text-white">
                      {result.is_ai_generated ? `${((1 - result.confidence) * 100).toFixed(1)}%` : `${(result.confidence * 100).toFixed(1)}%`}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-white font-medium mb-3 flex items-center gap-2">
                  <Brain className="w-4 h-4 text-blue-400" />
                  Explanation
                </h4>
                <p className="text-gray-300 text-sm leading-relaxed">
                  {result.explanation}
                </p>
              </div>
            </div>

            <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <p className="text-blue-400 text-sm">
                ℹ️ This analysis uses transformer-based NLP models to detect patterns commonly found in AI-generated text. 
                Results should be used as guidance and not as definitive proof.
              </p>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
};

export default TextAnalysis;
