import React from 'react';
import { CheckCircle, AlertTriangle, XCircle, Clock, FileText, Image, Video, Mic } from 'lucide-react';

interface AnalysisResult {
  id: string;
  filename: string;
  fileType: 'image' | 'video' | 'audio';
  confidence: number;
  authenticity: 'AUTHENTIC MEDIA' | 'MANIPULATED MEDIA' | 'UNCERTAIN';
  processingTime: number;
  keyFindings: string[];
  details?: {
    modelVersion: string;
    analysisMethod: string;
    technicalDetails?: any;
  };
  timestamp: Date;
}

interface AnalysisResultsProps {
  results: AnalysisResult[];
  isLoading?: boolean;
}

const FileTypeIcon: React.FC<{ type: string }> = ({ type }) => {
  const iconClass = "w-5 h-5";
  
  switch (type) {
    case 'image':
      return <Image className={`${iconClass} text-blue-400`} />;
    case 'video':
      return <Video className={`${iconClass} text-purple-400`} />;
    case 'audio':
      return <Mic className={`${iconClass} text-green-400`} />;
    default:
      return <FileText className={`${iconClass} text-gray-400`} />;
  }
};

const AuthenticityBadge: React.FC<{ authenticity: string; score: number }> = ({ authenticity, score }) => {
  const getColors = () => {
    if (authenticity === 'AUTHENTIC MEDIA' && score >= 90) {
      return {
        bg: 'bg-green-400/10',
        border: 'border-green-400/30',
        text: 'text-green-400',
        icon: CheckCircle
      };
    }
    if (authenticity === 'MANIPULATED MEDIA' || score < 60) {
      return {
        bg: 'bg-red-400/10',
        border: 'border-red-400/30',
        text: 'text-red-400',
        icon: XCircle
      };
    }
    return {
      bg: 'bg-yellow-400/10',
      border: 'border-yellow-400/30',
      text: 'text-yellow-400',
      icon: AlertTriangle
    };
  };

  const colors = getColors();
  const Icon = colors.icon;

  return (
    <div className={`inline-flex items-center px-3 py-1 rounded-full border ${colors.bg} ${colors.border}`}>
      <Icon className={`w-4 h-4 mr-2 ${colors.text}`} />
      <span className={`text-sm font-semibold ${colors.text}`}>
        {authenticity === 'AUTHENTIC MEDIA' ? 'Authentic' : 
         authenticity === 'MANIPULATED MEDIA' ? 'Manipulated' : 'Uncertain'}
      </span>
    </div>
  );
};

const ConfidenceBar: React.FC<{ score: number }> = ({ score }) => {
  const getColor = () => {
    if (score >= 90) return 'bg-green-400';
    if (score >= 60) return 'bg-yellow-400';
    return 'bg-red-400';
  };

  return (
    <div className="w-full bg-gray-700 rounded-full h-2">
      <div 
        className={`h-2 rounded-full transition-all duration-500 ${getColor()}`}
        style={{ width: `${score}%` }}
      ></div>
    </div>
  );
};

const AnalysisResultCard: React.FC<{ result: AnalysisResult }> = ({ result }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6 border border-gray-700 hover:border-gray-600 transition-colors">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <FileTypeIcon type={result.fileType} />
          <div>
            <h3 className="text-white font-semibold">{result.filename}</h3>
            <p className="text-gray-400 text-sm">
              Analyzed {result.timestamp.toLocaleTimeString()}
            </p>
          </div>
        </div>
        <AuthenticityBadge authenticity={result.authenticity} score={result.confidence} />
      </div>

      <div className="space-y-4">
        {/* Confidence Score */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300 text-sm">Confidence Score</span>
            <span className="text-white font-semibold">{result.confidence}%</span>
          </div>
          <ConfidenceBar score={result.confidence} />
        </div>

        {/* Processing Time */}
        <div className="flex items-center text-gray-400 text-sm">
          <Clock className="w-4 h-4 mr-2" />
          <span>Processed in {result.processingTime}s</span>
        </div>

        {/* Detection Details */}
        <div className="space-y-2">
          <h4 className="text-white text-sm font-medium">Analysis Details</h4>
          <div className="text-gray-400 text-sm">
            <p>Model: {result.details?.modelVersion || 'Unknown'}</p>
            <p>Method: {result.details?.analysisMethod || 'Standard'}</p>
            {result.keyFindings && result.keyFindings.length > 0 && (
              <div className="mt-2">
                <p className="text-white text-xs font-medium mb-1">Key Findings:</p>
                <ul className="text-xs space-y-1">
                  {result.keyFindings.slice(0, 3).map((finding, index) => (
                    <li key={index} className="text-gray-300">• {finding}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results, isLoading = false }) => {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="bg-gray-800 rounded-lg p-6 border border-gray-700 animate-pulse">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-5 h-5 bg-gray-600 rounded"></div>
              <div className="space-y-2 flex-1">
                <div className="h-4 bg-gray-600 rounded w-1/3"></div>
                <div className="h-3 bg-gray-600 rounded w-1/4"></div>
              </div>
            </div>
            <div className="space-y-3">
              <div className="h-2 bg-gray-600 rounded"></div>
              <div className="h-3 bg-gray-600 rounded w-1/2"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
          <FileText className="w-8 h-8 text-gray-400" />
        </div>
        <h3 className="text-white text-lg font-semibold mb-2">No Analysis Results</h3>
        <p className="text-gray-400">Upload files to see analysis results here</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Analysis Results</h2>
        <span className="text-gray-400 text-sm">{results.length} result{results.length !== 1 ? 's' : ''}</span>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {results.map((result) => (
          <AnalysisResultCard key={result.id} result={result} />
        ))}
      </div>
    </div>
  );
};

export default AnalysisResults;