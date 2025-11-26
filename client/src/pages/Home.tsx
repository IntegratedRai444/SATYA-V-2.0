import ParticleBackground from '@/components/home/ParticleBackground';
import AuthenticityScoreCard from '@/components/home/AuthenticityScoreCard';


const Home = () => {
  return (
    <div className="min-h-screen bg-[#0a0a0a] relative">
      {/* Particle Background */}
      <ParticleBackground />

      <div className="relative z-10 container mx-auto px-6 py-20">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Welcome Content */}
          <div className="space-y-6">
            <h1 className="text-5xl font-bold text-white">
              Welcome to <span className="text-cyan-400">SatyaAI</span>
            </h1>
            <p className="text-xl text-gray-300">
              Your trusted platform for deepfake detection and media authentication.
            </p>
            <p className="text-gray-400">
              Get started by analyzing your media files or exploring our detection tools.
            </p>
          </div>

          {/* Right Side - Authenticity Score Card */}
          <div className="flex justify-center">
            <AuthenticityScoreCard score={97} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
