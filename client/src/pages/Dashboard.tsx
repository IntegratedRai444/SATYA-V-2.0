import { Helmet } from 'react-helmet';
import WelcomeBanner from "@/components/dashboard/WelcomeBanner";
import QuickAccessTiles from "@/components/dashboard/QuickAccessTiles";
import RecentActivity from "@/components/dashboard/RecentActivity";
import UploadSection from "@/components/upload/UploadSection";
import InformativeSection from "@/components/dashboard/InformativeSection";

export default function Dashboard() {
  return (
    <>
      <Helmet>
        <title>SatyaAI - Deepfake Detection Dashboard</title>
        <meta name="description" content="Authenticate media with confidence using SatyaAI's advanced deepfake detection technology. Upload images, videos, audio or use your webcam for real-time analysis." />
      </Helmet>
      
      <WelcomeBanner />
      
      <QuickAccessTiles />
      
      {/* Upload & Results Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Section - Takes 2/3 of the space on large screens */}
        <div className="lg:col-span-2">
          <UploadSection />
        </div>
        
        {/* Recent Activity Section - Takes 1/3 of the space on large screens */}
        <div className="lg:col-span-1">
          <RecentActivity />
        </div>
      </div>
      
      <InformativeSection />
    </>
  );
}
