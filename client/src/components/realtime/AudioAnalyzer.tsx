import React, { useState, useEffect, useRef } from 'react';
import { Mic, Pause, Activity, AlertCircle } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';

interface AudioAnalysisResult {
  is_deepfake: boolean;
  confidence: number;
  timestamp: string;
}

const AudioAnalyzer: React.FC = () => {
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [results, setResults] = useState<AudioAnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sensitivity, setSensitivity] = useState<number>(0.7);

  const { isConnected, sendMessage } = useWebSocket({
    autoConnect: true
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize audio context and analyzer
  const initAudioContext = async () => {
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Create media recorder
      mediaRecorderRef.current = new MediaRecorder(stream);

      // Set up audio processing
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      analyserRef.current = analyser;
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

      // Start visualization
      drawVisualizer();

      return stream;
    } catch (err) {
      console.error('Error initializing audio:', err);
      setError('Could not access microphone. Please ensure you have granted microphone permissions.');
      setIsAnalyzing(false);
      return null;
    }
  };

  // Visualize audio
  const drawVisualizer = () => {
    if (!canvasRef.current || !analyserRef.current || !dataArrayRef.current) {
      animationFrameRef.current = requestAnimationFrame(drawVisualizer);
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    if (!ctx) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = dataArrayRef.current;

    analyser.getByteFrequencyData(dataArray as any);

    ctx.clearRect(0, 0, width, height);

    // Draw frequency bars
    const barWidth = (width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * height;

      // Gradient based on frequency
      const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
      gradient.addColorStop(0, '#3b82f6'); // blue-500
      gradient.addColorStop(1, '#22c55e'); // green-500

      ctx.fillStyle = gradient;
      ctx.fillRect(x, height - barHeight, barWidth, barHeight);

      x += barWidth + 1;
    }

    animationFrameRef.current = requestAnimationFrame(drawVisualizer);
  };

  // Start/stop analysis
  const toggleAnalysis = async () => {
    if (isAnalyzing) {
      // Stop analysis
      if (mediaRecorderRef.current?.state !== 'inactive') {
        mediaRecorderRef.current?.stop();
      }
      if (audioContextRef.current?.state === 'running') {
        await audioContextRef.current.suspend();
      }
      setIsAnalyzing(false);
    } else {
      // Start analysis
      setError(null);
      setResults([]);

      try {
        const stream = await initAudioContext();
        if (!stream) return;

        if (audioContextRef.current?.state === 'suspended') {
          await audioContextRef.current.resume();
        }

        // Start recording and sending data
        if (mediaRecorderRef.current) {
          mediaRecorderRef.current.ondataavailable = (event) => {
            if (event.data.size > 0 && isConnected) {
              // Convert blob to base64 or array buffer and send via WebSocket
              const reader = new FileReader();
              reader.onloadend = () => {
                const base64data = reader.result;
                sendMessage({
                  type: 'audio_chunk',
                  data: base64data,
                  config: {
                    sensitivity
                  }
                });
              };
              reader.readAsDataURL(event.data);
            }
          };

          mediaRecorderRef.current.start(100); // Collect 100ms chunks
          setIsAnalyzing(true);
        }
      } catch (err) {
        console.error('Error starting analysis:', err);
        setError('Failed to start audio analysis');
        setIsAnalyzing(false);
      }
    }
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      if (mediaRecorderRef.current?.state !== 'inactive') {
        mediaRecorderRef.current?.stop();
      }

      // Close audio context
      if (audioContextRef.current?.state !== 'closed') {
        audioContextRef.current?.close();
      }
    };
  }, []);

  // Calculate overall confidence score
  const getOverallConfidence = () => {
    if (results.length === 0) return 0;

    const fakeResults = results.filter(r => r.is_deepfake);
    if (fakeResults.length === 0) return 0;

    return (fakeResults.reduce((sum, r) => sum + r.confidence, 0) / fakeResults.length) * 100;
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="w-full max-w-3xl mx-auto bg-[#1e2128] border border-gray-700/50 rounded-xl overflow-hidden">
      <div className="p-6 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-500/10 rounded-lg flex items-center justify-center">
            <Mic className="w-5 h-5 text-blue-400" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Real-time Audio Analysis</h3>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'}`}></div>
              <span className="text-xs text-gray-400">{isConnected ? 'Connected' : 'Connecting...'}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 mb-6 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        <div className="mb-6">
          <canvas
            ref={canvasRef}
            width={600}
            height={150}
            className="w-full h-[150px] bg-gray-900 rounded-lg mb-4"
          />

          <div className="flex flex-col sm:flex-row items-center gap-6">
            <button
              onClick={toggleAnalysis}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all ${isAnalyzing
                ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20'
                : 'bg-blue-500 hover:bg-blue-600 text-white'
                }`}
            >
              {isAnalyzing ? (
                <>
                  <Pause className="w-5 h-5" />
                  Stop Analysis
                </>
              ) : (
                <>
                  <Mic className="w-5 h-5" />
                  Start Analysis
                </>
              )}
            </button>

            <div className="flex items-center gap-4 flex-1 w-full">
              <span className="text-sm text-gray-400 whitespace-nowrap">Sensitivity: {Math.round(sensitivity * 100)}%</span>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={sensitivity}
                onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                disabled={isAnalyzing}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {results.length > 0 ? (
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-white">Overall Deepfake Confidence</span>
                <span className="text-sm text-blue-400">{getOverallConfidence().toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${getOverallConfidence() > 50 ? 'bg-red-500' : 'bg-green-500'
                    }`}
                  style={{ width: `${getOverallConfidence()}%` }}
                ></div>
              </div>
            </div>

            <div className="border border-gray-700/50 rounded-lg overflow-hidden max-h-[300px] overflow-y-auto">
              {results.map((result, index) => (
                <div
                  key={index}
                  className={`p-3 border-b border-gray-700/50 flex items-center justify-between ${result.is_deepfake ? 'bg-red-500/5' : 'bg-green-500/5'
                    }`}
                >
                  <span className="text-xs text-gray-500">{formatTime(result.timestamp)}</span>
                  <span className={`text-sm font-medium ${result.is_deepfake ? 'text-red-400' : 'text-green-400'
                    }`}>
                    {result.is_deepfake ? 'Potential Deepfake' : 'Authentic'}
                  </span>
                  <span className="text-xs text-gray-400">
                    Confidence: {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500 border border-dashed border-gray-700 rounded-lg">
            <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No analysis results yet. Click "Start Analysis" to begin.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AudioAnalyzer;
