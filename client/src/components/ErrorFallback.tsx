export const FallbackComponent = () => (
  <div className="flex flex-col items-center justify-center min-h-screen p-4 text-center bg-black text-white">
    <h1 className="text-2xl font-bold text-red-600 mb-4">🚨 SATYA AI ERROR</h1>
    <p className="mb-4">Something went wrong with the application.</p>
    <div className="mb-4 p-4 bg-gray-800 rounded text-left">
      <h3 className="text-lg font-mono mb-2">Debug Info:</h3>
      <p>URL: {window.location.href}</p>
      <p>User Agent: {navigator.userAgent}</p>
      <p>Timestamp: {new Date().toISOString()}</p>
    </div>
    <button
      className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      onClick={() => window.location.reload()}
    >
      Reload Page
    </button>
  </div>
);
