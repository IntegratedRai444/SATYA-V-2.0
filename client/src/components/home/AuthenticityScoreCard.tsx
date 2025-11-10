import { FiZap, FiShield, FiCheckCircle } from 'react-icons/fi';
import CircularProgress from './CircularProgress';

interface AuthenticityScoreCardProps {
  score: number;
}

interface FeatureIndicatorProps {
  icon: React.ReactNode;
  label: string;
  active: boolean;
}

const FeatureIndicator = ({ icon, label, active }: FeatureIndicatorProps) => {
  return (
    <div className="flex items-center gap-3">
      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
        active ? 'bg-accent-cyan/20 text-accent-cyan' : 'bg-bg-tertiary text-text-muted'
      }`}>
        {icon}
      </div>
      <span className={`text-sm font-medium ${
        active ? 'text-text-primary' : 'text-text-muted'
      }`}>
        {label}
      </span>
    </div>
  );
};

const AuthenticityScoreCard = ({ score }: AuthenticityScoreCardProps) => {
  return (
    <div className="relative w-full max-w-md">
      {/* Card with gradient border effect */}
      <div className="bg-bg-card border border-border-primary rounded-2xl p-8 backdrop-blur-sm relative z-10">
        {/* Score display */}
        <div className="text-center mb-8">
          <div className="relative inline-block">
            <CircularProgress value={score} size={200} />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-6xl font-bold text-accent-cyan">
                  {score}%
                </div>
                <div className="text-sm text-text-secondary uppercase tracking-wider mt-2">
                  Authenticity Score
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Feature indicators */}
        <div className="space-y-4">
          <FeatureIndicator
            icon={<FiZap className="w-4 h-4" />}
            label="Real-time Analysis"
            active={true}
          />
          <FeatureIndicator
            icon={<FiShield className="w-4 h-4" />}
            label="Secure Processing"
            active={true}
          />
          <FeatureIndicator
            icon={<FiCheckCircle className="w-4 h-4" />}
            label="Verified Protection"
            active={true}
          />
        </div>
      </div>
      
      {/* Glow effect */}
      <div className="absolute inset-0 bg-accent-cyan/20 blur-3xl -z-10 rounded-full" />
    </div>
  );
};

export default AuthenticityScoreCard;
