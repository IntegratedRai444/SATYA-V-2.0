import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button, Card, Progress, Alert, Slider, Typography, Space, Row, Col } from 'antd';
import { AudioOutlined, PauseOutlined, LoadingOutlined } from '@ant-design/icons';
import { useAudioRecorder } from 'react-audio-voice-recorder';
import { wsClient } from '../../lib/wsClient';
import { API_URL } from '../../config';

const { Title, Text } = Typography;

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
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Initialize WebSocket connection
  const initWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setConnectionStatus('connecting');
    
    const wsUrl = API_URL.replace('http', 'ws') + '/api/v1/ws/audio-analysis';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnectionStatus('connected');
      
      // Send configuration
      ws.send(JSON.stringify({
        sample_rate: 16000,
        buffer_size: 1024,
        threshold: sensitivity
      }));
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'analysis_result') {
          setResults(prev => [{
            is_deepfake: data.data.is_deepfake,
            confidence: data.data.confidence,
            timestamp: data.timestamp
          }, ...prev].slice(0, 50)); // Keep only last 50 results
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnectionStatus('disconnected');
      if (isAnalyzing) {
        // Try to reconnect after a delay
        setTimeout(initWebSocket, 3000);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Failed to connect to audio analysis service');
      setConnectionStatus('disconnected');
    };
    
    wsRef.current = ws;
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [sensitivity, isAnalyzing]);

  // Initialize audio context and analyzer
  const initAudioContext = async () => {
    try {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Create media recorder
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      
      // Set up audio processing
      const source = audioContextRef.current.createMediaStreamSource(stream);
      const analyser = audioContextRef.current.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      
      analyserRef.current = analyser;
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
      
      // Start visualization
      drawVisualizer();
      
    } catch (err) {
      console.error('Error initializing audio:', err);
      setError('Could not access microphone. Please ensure you have granted microphone permissions.');
      setIsAnalyzing(false);
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
    
    analyser.getByteFrequencyData(dataArray);
    
    ctx.clearRect(0, 0, width, height);
    
    // Draw frequency bars
    const barWidth = (width / bufferLength) * 2.5;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * height;
      
      // Gradient based on frequency
      const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
      gradient.addColorStop(0, '#1890ff');
      gradient.addColorStop(1, '#52c41a');
      
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
      if (wsRef.current) {
        wsRef.current.close();
      }
      setIsAnalyzing(false);
    } else {
      // Start analysis
      setError(null);
      setResults([]);
      
      try {
        await initAudioContext();
        initWebSocket();
        
        if (audioContextRef.current?.state === 'suspended') {
          await audioContextRef.current.resume();
        }
        
        // Start recording
        if (mediaRecorderRef.current) {
          mediaRecorderRef.current.ondataavailable = (event) => {
            if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
              event.data.arrayBuffer().then(buffer => {
                wsRef.current?.send(buffer);
              });
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
      
      if (wsRef.current) {
        wsRef.current.close();
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

  // Format timestamp
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <Card 
      title={
        <Space>
          <AudioOutlined />
          <span>Real-time Audio Analysis</span>
          {connectionStatus === 'connected' && (
            <span style={{ color: '#52c41a', fontSize: '0.8em' }}>
              ● Connected
            </span>
          )}
          {connectionStatus === 'connecting' && (
            <span style={{ color: '#faad14', fontSize: '0.8em' }}>
              <LoadingOutlined spin /> Connecting...
            </span>
          )}
        </Space>
      }
      style={{ width: '100%', maxWidth: '800px', margin: '0 auto' }}
    >
      {error && (
        <Alert
          message="Error"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          closable
          onClose={() => setError(null)}
        />
      )}
      
      <div style={{ marginBottom: 24 }}>
        <canvas
          ref={canvasRef}
          width={600}
          height={150}
          style={{
            width: '100%',
            height: '150px',
            backgroundColor: '#f0f2f5',
            borderRadius: '4px',
            marginBottom: '16px'
          }}
        />
        
        <Button
          type={isAnalyzing ? 'default' : 'primary'}
          danger={isAnalyzing}
          icon={isAnalyzing ? <PauseOutlined /> : <AudioOutlined />}
          onClick={toggleAnalysis}
          size="large"
          style={{ width: '200px', marginBottom: '16px' }}
        >
          {isAnalyzing ? 'Stop Analysis' : 'Start Analysis'}
        </Button>
        
        <div style={{ marginBottom: '16px' }}>
          <Text>Sensitivity: </Text>
          <Slider
            min={0.1}
            max={0.9}
            step={0.1}
            value={sensitivity}
            onChange={setSensitivity}
            style={{ width: '200px', display: 'inline-block', marginLeft: '16px' }}
            disabled={isAnalyzing}
          />
          <Text style={{ marginLeft: '16px' }}>{Math.round(sensitivity * 100)}%</Text>
        </div>
      </div>
      
      <div style={{ marginTop: '24px' }}>
        <Title level={4} style={{ marginBottom: '16px' }}>Analysis Results</Title>
        
        {results.length > 0 ? (
          <>
            <div style={{ marginBottom: '16px' }}>
              <Text strong>Overall Deepfake Confidence: </Text>
              <Progress 
                percent={getOverallConfidence()} 
                status={getOverallConfidence() > 50 ? 'exception' : 'success'}
                style={{ width: '200px', display: 'inline-block', marginLeft: '16px' }}
                showInfo={false}
              />
              <Text style={{ marginLeft: '16px' }}>{getOverallConfidence().toFixed(1)}%</Text>
            </div>
            
            <div style={{ maxHeight: '300px', overflowY: 'auto', border: '1px solid #f0f0f0', borderRadius: '4px' }}>
              {results.map((result, index) => (
                <div 
                  key={index}
                  style={{
                    padding: '8px 16px',
                    borderBottom: '1px solid #f0f0f0',
                    backgroundColor: result.is_deepfake ? '#fff1f0' : '#f6ffed',
                  }}
                >
                  <Row justify="space-between" align="middle">
                    <Col>
                      <Text>{formatTime(result.timestamp)}</Text>
                    </Col>
                    <Col>
                      <Text strong style={{ color: result.is_deepfake ? '#ff4d4f' : '#52c41a' }}>
                        {result.is_deepfake ? 'Potential Deepfake' : 'Authentic'}
                      </Text>
                    </Col>
                    <Col>
                      <Text>Confidence: {(result.confidence * 100).toFixed(1)}%</Text>
                    </Col>
                  </Row>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div style={{ textAlign: 'center', padding: '24px', color: '#8c8c8c' }}>
            <p>No analysis results yet. Click "Start Analysis" to begin.</p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default AudioAnalyzer;
