import React, { useEffect, useState } from 'react';
import { Helmet } from 'react-helmet';
import WelcomeBanner from "../components/dashboard/WelcomeBanner";
import QuickAccessTiles from "../components/dashboard/QuickAccessTiles";
import RecentActivity from "../components/dashboard/RecentActivity";
import UploadSection from "../components/upload/UploadSection";
import InformativeSection from "../components/dashboard/InformativeSection";
import { Bar } from 'react-chartjs-2';
import { Chart, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

Chart.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

export default function Dashboard() {
  // Analytics state
  const [analytics, setAnalytics] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [analyticsError, setAnalyticsError] = useState(null);

  // Recent activity state
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState(null);

  useEffect(() => {
    // Fetch analytics
    setAnalyticsLoading(true);
    fetch('/api/analytics')
      .then(res => res.json())
      .then(data => {
        setAnalytics(data);
        setAnalyticsLoading(false);
      })
      .catch(err => {
        setAnalyticsError('Failed to load analytics');
        setAnalyticsLoading(false);
      });
    // Fetch recent activity
    setHistoryLoading(true);
    fetch('/api/detections/history')
      .then(res => res.json())
      .then(data => {
        setHistory(data.history || []);
        setHistoryLoading(false);
      })
      .catch(err => {
        setHistoryError('Failed to load recent activity');
        setHistoryLoading(false);
      });
  }, []);

  // Chart.js data
  const chartData = analytics ? {
    labels: analytics.trends?.map(t => t.date) || [],
    datasets: [
      {
        label: 'Scans',
        data: analytics.trends?.map(t => t.count) || [],
        backgroundColor: '#2563eb',
      },
    ],
  } : { labels: [], datasets: [] };

  return (
    <>
      <Helmet>
        <title>SatyaAI - Deepfake Detection Dashboard</title>
        <meta name="description" content="Authenticate media with confidence using SatyaAI's advanced deepfake detection technology. Upload images, videos, audio or use your webcam for real-time analysis." />
      </Helmet>
      <WelcomeBanner />
      <QuickAccessTiles />
      {/* Analytics Section */}
      <div className="bg-white rounded shadow p-4 mb-6">
        <h3 className="text-lg font-semibold mb-2">Scan Analytics</h3>
        {analyticsLoading ? (
          <div className="h-32 flex items-center justify-center text-gray-400 animate-pulse">Loading analytics...</div>
        ) : analyticsError ? (
          <div className="h-32 flex items-center justify-center text-red-500">{analyticsError}</div>
        ) : (
          <>
            <div className="flex gap-6 mb-4">
              <div>
                <div className="text-2xl font-bold">{analytics.total_scans ?? '-'}</div>
                <div className="text-xs text-gray-500">Total Scans</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{analytics.avg_confidence ?? '-'}</div>
                <div className="text-xs text-gray-500">Avg. Confidence</div>
              </div>
              <div>
                <div className="text-2xl font-bold">{analytics.most_common_type ?? '-'}</div>
                <div className="text-xs text-gray-500">Most Common Type</div>
              </div>
            </div>
            <Bar data={chartData} options={{ responsive: true, plugins: { legend: { display: false } } }} height={100} />
          </>
        )}
      </div>
      {/* Recent Activity Section */}
      <div className="bg-white rounded shadow p-4 mb-6 overflow-x-auto">
        <h3 className="text-lg font-semibold mb-2">Recent Activity</h3>
        {historyLoading ? (
          <div className="h-24 flex items-center justify-center text-gray-400 animate-pulse">Loading recent activity...</div>
        ) : historyError ? (
          <div className="h-24 flex items-center justify-center text-red-500">{historyError}</div>
        ) : (
          <table className="min-w-full text-sm">
            <thead>
              <tr>
                <th className="px-2 py-1">Type</th>
                <th className="px-2 py-1">Filename</th>
                <th className="px-2 py-1">Date</th>
                <th className="px-2 py-1">Result</th>
              </tr>
            </thead>
            <tbody>
              {history.map((scan, idx) => (
                <tr key={scan.id || idx} className="border-t">
                  <td className="px-2 py-1">{scan.type}</td>
                  <td className="px-2 py-1">{scan.filename}</td>
                  <td className="px-2 py-1">{scan.date}</td>
                  <td className={`px-2 py-1 font-bold ${scan.result === 'FAKE' ? 'text-red-500' : 'text-green-600'}`}>{scan.result}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
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
