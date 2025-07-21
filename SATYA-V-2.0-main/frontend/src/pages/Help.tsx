import React from "react";

export default function Help() {
  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded shadow mt-8">
      <h2 className="text-2xl font-bold mb-4">Help & Support</h2>
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Frequently Asked Questions</h3>
        <ul className="list-disc pl-6 space-y-2">
          <li><strong>How do I upload media for analysis?</strong> Use the Dashboard to upload images, videos, audio, or use your webcam.</li>
          <li><strong>What file types are supported?</strong> JPEG, PNG, MP4, WAV, and more. See Settings for details.</li>
          <li><strong>How do I interpret the results?</strong> The dashboard and reports provide clear labels and explanations for each scan.</li>
          <li><strong>Is my data private?</strong> Yes, your uploads and results are processed securely and not shared.</li>
        </ul>
      </div>
      <div>
        <h3 className="text-lg font-semibold mb-2">Contact Support</h3>
        <p>If you need further assistance, email <a href="mailto:support@satyaai.com" className="text-blue-600 underline">support@satyaai.com</a> or use the in-app chat.</p>
      </div>
    </div>
  );
}
