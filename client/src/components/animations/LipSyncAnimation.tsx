import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface LipSyncAnimationProps {
  language?: 'english' | 'hindi' | 'tamil';
  isAnalyzing?: boolean;
  confidence?: number;
}

const LipSyncAnimation: React.FC<LipSyncAnimationProps> = ({
  language = 'english',
  isAnalyzing = false,
  confidence = 0
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>(0);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const context = canvas.getContext('2d');
    if (!context) return;
    
    // Set canvas dimensions
    const setCanvasDimensions = () => {
      if (containerRef.current) {
        const { width, height } = containerRef.current.getBoundingClientRect();
        canvas.width = width;
        canvas.height = height;
      }
    };
    
    setCanvasDimensions();
    window.addEventListener('resize', setCanvasDimensions);
    
    // Animation variables
    let time = 0;
    let waveSpeed = 0.01;
    const waveAmplitude = isAnalyzing ? 20 : 10;
    const waveFrequency = 0.02;
    const lineCount = 8;
    
    // Color themes for different languages
    const colorThemes = {
      english: {
        background: 'rgba(10, 20, 50, 0.9)',
        primary: 'rgba(41, 121, 255, 1)',
        secondary: 'rgba(0, 212, 255, 0.8)',
        text: '#ffffff'
      },
      hindi: {
        background: 'rgba(50, 10, 40, 0.9)',
        primary: 'rgba(255, 41, 121, 1)',
        secondary: 'rgba(255, 0, 128, 0.8)',
        text: '#ffffff'
      },
      tamil: {
        background: 'rgba(40, 50, 10, 0.9)',
        primary: 'rgba(121, 255, 41, 1)',
        secondary: 'rgba(173, 255, 0, 0.8)',
        text: '#ffffff'
      }
    };
    
    const theme = colorThemes[language];
    
    // Draw waveform
    const drawWaveform = () => {
      if (!context) return;
      
      // Clear canvas
      context.fillStyle = theme.background;
      context.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw title
      context.font = 'bold 14px Arial';
      context.fillStyle = theme.text;
      context.textAlign = 'center';
      context.fillText(`${language.toUpperCase()} LIP-SYNC ANALYZER`, canvas.width / 2, 30);
      
      // Draw language indicator
      const indicatorRadius = 5;
      context.beginPath();
      context.arc(canvas.width / 2 - 100, 30, indicatorRadius, 0, Math.PI * 2);
      context.fillStyle = theme.primary;
      context.fill();
      
      // Calculate center point
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      // Draw waveform
      for (let i = 0; i < lineCount; i++) {
        const lineWidth = 2;
        const opacity = 1 - i * (0.7 / lineCount);
        const hue = i * (30 / lineCount);
        
        context.beginPath();
        
        // Create gradient
        const gradient = context.createLinearGradient(0, centerY - 50, 0, centerY + 50);
        gradient.addColorStop(0, theme.secondary);
        gradient.addColorStop(1, theme.primary);
        
        context.strokeStyle = i === 0 ? theme.primary : gradient;
        context.lineWidth = lineWidth;
        
        // Start point
        context.moveTo(0, centerY);
        
        // Draw wave
        for (let x = 0; x < canvas.width; x += 5) {
          // Basic sine wave
          const y = centerY + 
            Math.sin(x * waveFrequency + time + i * 0.3) * waveAmplitude + 
            Math.sin(x * waveFrequency * 2 + time * 1.5) * (waveAmplitude / 2);
          
          context.lineTo(x, y);
        }
        
        context.stroke();
      }
      
      // Draw mouth visualization
      if (isAnalyzing || confidence > 0) {
        const mouthWidth = 120;
        const mouthHeight = isAnalyzing ? 
          30 + Math.sin(time * 5) * 20 : 
          30 * (confidence / 100);
        
        context.beginPath();
        context.ellipse(
          centerX,
          centerY + 80,
          mouthWidth,
          mouthHeight,
          0,
          0,
          Math.PI * 2
        );
        context.fillStyle = 'rgba(0, 0, 0, 0.7)';
        context.fill();
        
        // Inner mouth
        context.beginPath();
        context.ellipse(
          centerX,
          centerY + 80,
          mouthWidth - 20,
          mouthHeight - 10,
          0,
          0,
          Math.PI * 2
        );
        context.fillStyle = 'rgba(150, 0, 0, 0.7)';
        context.fill();
        
        // Teeth
        if (mouthHeight > 15) {
          context.beginPath();
          context.rect(centerX - 50, centerY + 65, 100, 10);
          context.fillStyle = 'rgba(240, 240, 240, 0.9)';
          context.fill();
        }
      }
      
      // Draw phoneme indicators
      if (isAnalyzing) {
        const phonemes = ["AA", "AE", "AH", "AO", "EH", "IH", "IY", "UH"];
        const phonemeRadius = 25;
        const angleStep = (Math.PI * 2) / phonemes.length;
        
        for (let i = 0; i < phonemes.length; i++) {
          const angle = i * angleStep;
          const x = centerX + Math.cos(angle) * 160;
          const y = centerY + Math.sin(angle) * 160;
          
          // Phoneme circle
          const isActive = Math.random() > 0.7;
          context.beginPath();
          context.arc(x, y, phonemeRadius, 0, Math.PI * 2);
          context.fillStyle = isActive ? 
            `rgba(${parseInt(theme.primary.slice(5))}, 0.8)` : 
            'rgba(255, 255, 255, 0.2)';
          context.fill();
          
          // Phoneme text
          context.font = 'bold 12px Arial';
          context.fillStyle = isActive ? '#ffffff' : 'rgba(255, 255, 255, 0.5)';
          context.textAlign = 'center';
          context.textBaseline = 'middle';
          context.fillText(phonemes[i], x, y);
        }
        
        // Draw connecting lines
        context.beginPath();
        for (let i = 0; i < phonemes.length; i++) {
          const angle = i * angleStep;
          const x = centerX + Math.cos(angle) * 160;
          const y = centerY + Math.sin(angle) * 160;
          
          if (i === 0) {
            context.moveTo(x, y);
          } else {
            context.lineTo(x, y);
          }
        }
        context.closePath();
        context.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        context.lineWidth = 1;
        context.stroke();
      }
      
      // Draw confidence indicator if not analyzing
      if (!isAnalyzing && confidence > 0) {
        const radius = 80;
        const startAngle = -Math.PI / 2;
        const endAngle = startAngle + (Math.PI * 2 * (confidence / 100));
        
        // Draw confidence arc
        context.beginPath();
        context.arc(centerX, centerY - 80, radius, startAngle, endAngle);
        context.lineWidth = 10;
        context.strokeStyle = confidence > 70 ? 
          'rgba(41, 255, 121, 0.8)' : 
          confidence > 40 ? 
            'rgba(255, 204, 41, 0.8)' : 
            'rgba(255, 61, 41, 0.8)';
        context.stroke();
        
        // Draw confidence text
        context.font = 'bold 24px Arial';
        context.fillStyle = '#ffffff';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(`${confidence}%`, centerX, centerY - 80);
        
        context.font = '14px Arial';
        context.fillText('Sync Confidence', centerX, centerY - 50);
      }
      
      // Update time for animation
      time += waveSpeed;
      if (isAnalyzing) {
        waveSpeed = 0.05;
      } else {
        waveSpeed = 0.01;
      }
      
      // Continue animation
      animationFrameRef.current = requestAnimationFrame(drawWaveform);
    };
    
    // Start animation
    drawWaveform();
    
    // Cleanup
    return () => {
      cancelAnimationFrame(animationFrameRef.current);
      window.removeEventListener('resize', setCanvasDimensions);
    };
  }, [language, isAnalyzing, confidence]);
  
  return (
    <div 
      ref={containerRef} 
      className="w-full h-64 md:h-80 relative rounded-lg overflow-hidden"
    >
      <canvas 
        ref={canvasRef} 
        className="w-full h-full"
      />
      
      {isAnalyzing && (
        <div className="absolute bottom-4 left-0 right-0 text-center text-white text-sm">
          <div className="inline-flex items-center bg-black/50 px-3 py-1 rounded-full">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse mr-2"></div>
            Analyzing {language} lip synchronization
          </div>
        </div>
      )}
    </div>
  );
};

export default LipSyncAnimation;