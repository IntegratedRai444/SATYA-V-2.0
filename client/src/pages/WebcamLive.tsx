import React from 'react';

const WebcamLive: React.FC = () => {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Webcam Live Analysis</h1>
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="aspect-video bg-gray-100 dark:bg-gray-700 rounded-lg flex items-center justify-center">
          <p className="text-gray-500 dark:text-gray-400">Webcam feed will appear here</p>
        </div>
        <div className="mt-6 flex justify-center space-x-4">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
            Start Analysis
          </button>
          <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
            Stop Camera
          </button>
        </div>
      </div>
    </div>
  );
};

export default WebcamLive;
