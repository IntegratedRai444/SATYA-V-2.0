import { useEffect, useState } from 'react';

interface CircularProgressProps {
  value: number;
  size?: number;
  strokeWidth?: number;
}

const CircularProgress = ({ value, size = 200, strokeWidth = 8 }: CircularProgressProps) => {
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    // Animate progress on mount
    const timer = setTimeout(() => {
      setProgress(value);
    }, 100);
    
    return () => clearTimeout(timer);
  }, [value]);
  
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (progress / 100) * circumference;
  
  // Determine color based on score
  const getColor = () => {
    if (progress >= 90) return '#00ff88'; // Green
    if (progress >= 70) return '#ff8800'; // Orange
    return '#ff4444'; // Red
  };
  
  return (
    <svg
      width={size}
      height={size}
      className="transform -rotate-90"
    >
      {/* Background circle */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke="#2a2a2a"
        strokeWidth={strokeWidth}
        fill="none"
      />
      
      {/* Progress circle */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        stroke={getColor()}
        strokeWidth={strokeWidth}
        fill="none"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        className="transition-all duration-1000 ease-out"
        style={{
          filter: `drop-shadow(0 0 8px ${getColor()})`
        }}
      />
    </svg>
  );
};

export default CircularProgress;
