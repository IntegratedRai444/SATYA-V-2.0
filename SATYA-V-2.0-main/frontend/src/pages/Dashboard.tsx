import React from 'react';
import { Helmet } from 'react-helmet';
import WelcomeBanner from "../components/dashboard/WelcomeBanner";
import QuickAccessTiles from "../components/dashboard/QuickAccessTiles";
import RecentActivity from "../components/dashboard/RecentActivity";
import UploadSection from "../components/upload/UploadSection";
import InformativeSection from "../components/dashboard/InformativeSection";
import { useState } from "react";

// Placeholder scan history data
const scanHistory = [
  { id: 1, type: "image", filename: "img1.jpg", date: "2024-07-17", result: "FAKE" },
  { id: 2, type: "video", filename: "vid1.mp4", date: "2024-07-16", result: "REAL" },
  { id: 3, type: "audio", filename: "audio1.wav", date: "2024-07-15", result: "FAKE" },
];

// Simple analytics chart placeholder (could use chart.js or similar for real data)
function AnalyticsChart() {
  return (
    <div className="bg-white rounded shadow p-4 mb-6">
      <h3 className="text-lg font-semibold mb-2">Scan Analytics</h3>
      <div className="h-32 flex items-center justify-center text-gray-400">
        [Analytics Chart Placeholder]
      </div>
    </div>
  );
}

function ScanHistoryTable() {
  return (
    <div className="bg-white rounded shadow p-4 mb-6 overflow-x-auto">
      <h3 className="text-lg font-semibold mb-2">Scan History</h3>
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
          {scanHistory.map(scan => (
            <tr key={scan.id} className="border-t">
              <td className="px-2 py-1">{scan.type}</td>
              <td className="px-2 py-1">{scan.filename}</td>
              <td className="px-2 py-1">{scan.date}</td>
              <td className={`px-2 py-1 font-bold ${scan.result === 'FAKE' ? 'text-red-500' : 'text-green-600'}`}>{scan.result}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Dashboard() {
  return (
    <>
      <Helmet>
        <title>SatyaAI - Deepfake Detection Dashboard</title>
        <meta name="description" content="Authenticate media with confidence using SatyaAI's advanced deepfake detection technology. Upload images, videos, audio or use your webcam for real-time analysis." />
      </Helmet>
      <WelcomeBanner />
      <QuickAccessTiles />
      <AnalyticsChart />
      <ScanHistoryTable />
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
