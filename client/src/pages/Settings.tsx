import React, { useState } from "react";

export default function Settings() {
  const [theme, setTheme] = useState("light");
  const [email, setEmail] = useState("");

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded shadow mt-8">
      <h2 className="text-2xl font-bold mb-4">Settings</h2>
      <div className="mb-4">
        <label className="block font-semibold mb-1">Theme</label>
        <select
          className="border rounded px-2 py-1"
          value={theme}
          onChange={e => setTheme(e.target.value)}
          title="Theme Mode"
        >
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
      </div>
      <div className="mb-4">
        <label className="block font-semibold mb-1">Email</label>
        <input
          className="border rounded px-2 py-1 w-full"
                  type="email" 
                  value={email} 
          onChange={e => setEmail(e.target.value)}
          placeholder="user@example.com"
                />
              </div>
      <button className="bg-blue-600 text-white px-4 py-2 rounded">Save Preferences</button>
              </div>
  );
}
