"use client"

import { Bell, Camera, Cloud, Eye, Grid2X2, HelpCircle, History, Home, Image as ImageIcon, Lock, Mic, Settings, Shield, Sparkles, Video, Check } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

export default function Page() {
  return (
    <div className="flex min-h-screen bg-[#0a0e1a]">
      {/* Left Sidebar */}
      <aside className="w-56 bg-[#0d1117] border-r border-white/5 flex flex-col">
        <div className="p-6">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center">
              <span className="text-white font-bold text-lg">S</span>
            </div>
            <div>
              <div className="text-white font-bold">Satya<span className="text-cyan-400">AI</span></div>
            </div>
          </div>
        </div>

        <nav className="flex-1 px-3">
          <div className="mb-6">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 px-3">Detection Tools</h3>
            <div className="space-y-1">
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg bg-[#0a3a42] text-cyan-400 border-l-2 border-cyan-400 hover:bg-[#0a3a42]/80 transition-colors relative">
                <Grid2X2 className="w-4 h-4" />
                <span className="text-sm font-medium">Dashboard</span>
              </button>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <ImageIcon className="w-4 h-4" />
                <span className="text-sm">Image Analysis</span>
              </button>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <Video className="w-4 h-4" />
                <span className="text-sm">Video Analysis</span>
              </button>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <Mic className="w-4 h-4" />
                <span className="text-sm">Audio Analysis</span>
              </button>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <Camera className="w-4 h-4" />
                <span className="text-sm">Webcam Live</span>
              </button>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 px-3">Management</h3>
            <div className="space-y-1">
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <History className="w-4 h-4" />
                <span className="text-sm">Scan History</span>
              </button>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <Settings className="w-4 h-4" />
                <span className="text-sm">Settings</span>
              </button>
              <button className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-gray-400 hover:bg-white/5 hover:text-gray-300 transition-colors">
                <HelpCircle className="w-4 h-4" />
                <span className="text-sm">Help & Support</span>
              </button>
            </div>
          </div>
        </nav>

        <div className="p-3 pb-6">
          <button className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white font-medium hover:opacity-90 transition-opacity shadow-lg shadow-cyan-500/20">
            <Sparkles className="w-4 h-4" />
            <span className="text-sm">AI Assistant</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Navigation */}
        <header className="border-b border-white/10 bg-[#0d1117]">
          <div className="flex items-center justify-between px-8 py-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-cyan-500 flex items-center justify-center">
                <span className="text-white font-bold text-sm">S</span>
              </div>
              <div>
                <div className="text-white font-bold text-base">Satya<span className="text-cyan-400">AI</span></div>
                <div className="text-[10px] text-gray-500">Synthetic Authentication Technology for Your Analysis</div>
              </div>
            </div>

            <nav className="flex items-center gap-8">
              <button className="flex items-center gap-2 text-cyan-400 text-sm font-medium hover:text-cyan-300 transition-colors">
                <Home className="w-4 h-4" />
                <span>Home</span>
              </button>
              <button className="flex items-center gap-2 text-gray-400 text-sm hover:text-gray-300 transition-colors">
                <Sparkles className="w-4 h-4" />
                <span>Scan</span>
              </button>
              <button className="flex items-center gap-2 text-gray-400 text-sm hover:text-gray-300 transition-colors">
                <History className="w-4 h-4" />
                <span>History</span>
              </button>
              <button className="flex items-center gap-2 text-gray-400 text-sm hover:text-gray-300 transition-colors">
                <Settings className="w-4 h-4" />
                <span>Settings</span>
              </button>
              <button className="flex items-center gap-2 text-gray-400 text-sm hover:text-gray-300 transition-colors">
                <HelpCircle className="w-4 h-4" />
                <span>Help</span>
              </button>
              <button className="relative text-gray-400 hover:text-gray-300 transition-colors ml-2">
                <Bell className="w-5 h-5" />
                <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
              </button>
              <div className="w-8 h-8 rounded-full bg-cyan-500 flex items-center justify-center text-white font-bold text-sm">
                U
              </div>
            </nav>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto bg-gradient-to-br from-[#0a0e1a] via-[#0a0e1a] to-[#0d1520]">
          <div className="max-w-[1600px] mx-auto px-8 py-16">
            {/* Hero Section */}
            <div className="grid grid-cols-[1fr_420px] gap-16 mb-24">
              <div className="space-y-8 pt-12">
                <div className="flex items-center gap-3">
                  <Badge className="bg-gradient-to-r from-purple-500 to-blue-500 text-white border-0 px-4 py-1.5 text-xs font-medium shadow-lg shadow-purple-500/20">
                    <Sparkles className="w-3.5 h-3.5 mr-2" />
                    New AI Models Released
                  </Badge>
                  <Badge className="bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-4 py-1.5 text-xs font-medium">
                    Protection
                  </Badge>
                </div>

                <h1 className="text-7xl font-bold leading-[1.1] text-white tracking-tight">
                  Detect <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">deepfakes</span> with the<br />
                  power of <span className="text-white">SatyaAI</span>
                </h1>

                <p className="text-gray-300 text-xl leading-relaxed max-w-2xl font-light">
                  Our advanced detection system helps you authenticate media with unprecedented<br />
                  accuracy, exposing manipulated content across images, videos, and audio.
                </p>

                <p className="text-gray-400 text-base max-w-2xl leading-relaxed">
                  Upload your files or use your webcam for real-time analysis and get detailed authenticity reports instantly.
                </p>

                <div className="flex items-center gap-5 pt-6">
                  <Button className="bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-600 hover:to-cyan-700 text-white px-8 py-7 text-base font-semibold rounded-xl shadow-xl shadow-cyan-500/25 hover:shadow-cyan-500/40 transition-all">
                    <Cloud className="w-5 h-5 mr-2.5" />
                    Analyze Media
                    <span className="ml-3 text-lg">→</span>
                  </Button>
                  <Button variant="outline" className="border-white/20 text-white hover:bg-white/10 hover:border-white/30 px-8 py-7 text-base font-medium rounded-xl backdrop-blur-sm transition-all">
                    <HelpCircle className="w-5 h-5 mr-2.5" />
                    How It Works
                  </Button>
                </div>
              </div>

              {/* Right Feature Card */}
              <div className="relative pt-12">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 via-blue-500/10 to-purple-500/10 rounded-3xl blur-3xl opacity-60"></div>
                <Card className="relative bg-gradient-to-br from-[#0d1117] to-[#0a0e14] backdrop-blur-xl border border-white/10 p-10 rounded-3xl shadow-2xl">
                  <div className="flex flex-col items-center text-center space-y-8">
                    <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                      <Camera className="w-12 h-12 text-cyan-400" />
                    </div>

                    <div>
                      <div className="text-xs text-cyan-400 uppercase tracking-widest font-semibold mb-3">AUTHENTICITY SCORE</div>
                      <div className="text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-400">77%</div>
                    </div>

                    <div className="w-full space-y-3 pt-4">
                      <div className="flex items-center gap-4 px-5 py-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm hover:bg-white/10 hover:border-cyan-500/30 transition-all group">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/10 flex items-center justify-center border border-cyan-500/20 group-hover:border-cyan-500/40 transition-all">
                          <Eye className="w-5 h-5 text-cyan-400" />
                        </div>
                        <span className="text-sm text-gray-300 font-medium">Real-time Analysis</span>
                      </div>
                      <div className="flex items-center gap-4 px-5 py-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm hover:bg-white/10 hover:border-cyan-500/30 transition-all group">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/10 flex items-center justify-center border border-cyan-500/20 group-hover:border-cyan-500/40 transition-all">
                          <Lock className="w-5 h-5 text-cyan-400" />
                        </div>
                        <span className="text-sm text-gray-300 font-medium">Secure Processing</span>
                      </div>
                      <div className="flex items-center gap-4 px-5 py-4 rounded-xl bg-white/5 border border-white/10 backdrop-blur-sm hover:bg-white/10 hover:border-cyan-500/30 transition-all group">
                        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/10 flex items-center justify-center border border-cyan-500/20 group-hover:border-cyan-500/40 transition-all">
                          <Shield className="w-5 h-5 text-cyan-400" />
                        </div>
                        <span className="text-sm text-gray-300 font-medium">Verified Protection</span>
                      </div>
                    </div>
                  </div>
                </Card>
              </div>
            </div>

            {/* Detection Tools Section */}
            <div>
              <div className="flex items-center justify-between mb-10">
                <div>
                  <h2 className="text-4xl font-bold text-white mb-3 flex items-center gap-3">
                    Deepfake Detection Tools
                    <span className="text-cyan-400 text-2xl">•••</span>
                  </h2>
                  <p className="text-gray-400 text-base">Choose your media type for comprehensive analysis</p>
                </div>
                <div className="flex items-center gap-3 text-sm">
                  <span className="text-gray-400">Using</span>
                  <span className="text-cyan-400 font-semibold">Neural Vision v4.2</span>
                  <span className="text-gray-400">models</span>
                  <button className="ml-3 px-5 py-2.5 rounded-lg bg-gradient-to-r from-purple-500 to-cyan-500 text-white text-xs font-semibold hover:opacity-90 transition-opacity shadow-lg shadow-purple-500/20">
                    Explore
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-7">
                {/* Image Card */}
                <Card className="bg-gradient-to-br from-[#0d1117] to-[#0a0e14] backdrop-blur-sm border border-white/10 p-7 rounded-2xl hover:border-cyan-500/40 hover:shadow-xl hover:shadow-cyan-500/10 transition-all cursor-pointer group">
                  <div className="flex flex-col space-y-5">
                    <div className="flex items-center justify-between">
                      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/10 border border-blue-500/30 flex items-center justify-center group-hover:border-blue-500/50 group-hover:shadow-lg group-hover:shadow-blue-500/20 transition-all">
                        <ImageIcon className="w-7 h-7 text-blue-400" />
                      </div>
                      <div className="text-xs text-gray-400">
                        Accuracy: <span className="text-white font-bold">98.2%</span>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2.5 group-hover:text-cyan-400 transition-colors">Image Analysis</h3>
                      <p className="text-sm text-gray-400 mb-5 leading-relaxed">Detect manipulated photos & generated images</p>
                      
                      <div className="space-y-2.5 mb-5">
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Photoshop Detection</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>GAN Detection</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Metadata Analysis</span>
                        </div>
                      </div>
                      
                      <button className="w-full py-3 text-xs font-bold text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-all flex items-center justify-center gap-2 border border-transparent hover:border-cyan-500/30">
                        START ANALYSIS
                        <span className="text-base">→</span>
                      </button>
                    </div>
                  </div>
                </Card>

                {/* Video Card */}
                <Card className="bg-gradient-to-br from-[#0d1117] to-[#0a0e14] backdrop-blur-sm border border-white/10 p-7 rounded-2xl hover:border-cyan-500/40 hover:shadow-xl hover:shadow-cyan-500/10 transition-all cursor-pointer group">
                  <div className="flex flex-col space-y-5">
                    <div className="flex items-center justify-between">
                      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 border border-cyan-500/30 flex items-center justify-center group-hover:border-cyan-500/50 group-hover:shadow-lg group-hover:shadow-cyan-500/20 transition-all">
                        <Video className="w-7 h-7 text-cyan-400" />
                      </div>
                      <div className="text-xs text-gray-400">
                        Accuracy: <span className="text-white font-bold">96.8%</span>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2.5 group-hover:text-cyan-400 transition-colors">Video Verification</h3>
                      <p className="text-sm text-gray-400 mb-5 leading-relaxed">Identify deepfake videos & facial manipulations</p>
                      
                      <div className="space-y-2.5 mb-5">
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Facial Inconsistencies</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Temporal Analysis</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Lip-Sync Verification</span>
                        </div>
                      </div>
                      
                      <button className="w-full py-3 text-xs font-bold text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-all flex items-center justify-center gap-2 border border-transparent hover:border-cyan-500/30">
                        START ANALYSIS
                        <span className="text-base">→</span>
                      </button>
                    </div>
                  </div>
                </Card>

                {/* Audio Card */}
                <Card className="bg-gradient-to-br from-[#0d1117] to-[#0a0e14] backdrop-blur-sm border border-white/10 p-7 rounded-2xl hover:border-cyan-500/40 hover:shadow-xl hover:shadow-purple-500/10 transition-all cursor-pointer group">
                  <div className="flex flex-col space-y-5">
                    <div className="flex items-center justify-between">
                      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-600/10 border border-purple-500/30 flex items-center justify-center group-hover:border-purple-500/50 group-hover:shadow-lg group-hover:shadow-purple-500/20 transition-all">
                        <Mic className="w-7 h-7 text-purple-400" />
                      </div>
                      <div className="text-xs text-gray-400">
                        Accuracy: <span className="text-white font-bold">95.3%</span>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2.5 group-hover:text-cyan-400 transition-colors">Audio Detection</h3>
                      <p className="text-sm text-gray-400 mb-5 leading-relaxed">Uncover voice cloning & synthetic speech</p>
                      
                      <div className="space-y-2.5 mb-5">
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Voice Cloning Detection</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Natural Patterns Analysis</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
                          <span>Neural Voice Filter</span>
                        </div>
                      </div>
                      
                      <button className="w-full py-3 text-xs font-bold text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-all flex items-center justify-center gap-2 border border-transparent hover:border-cyan-500/30">
                        START ANALYSIS
                        <span className="text-base">→</span>
                      </button>
                    </div>
                  </div>
                </Card>

                {/* Webcam Card */}
                <Card className="bg-gradient-to-br from-[#0d1117] to-[#0a0e14] backdrop-blur-sm border border-white/10 p-7 rounded-2xl hover:border-cyan-500/40 hover:shadow-xl hover:shadow-pink-500/10 transition-all cursor-pointer group">
                  <div className="flex flex-col space-y-5">
                    <div className="flex items-center justify-between">
                      <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-pink-500/20 to-pink-600/10 border border-pink-500/30 flex items-center justify-center group-hover:border-pink-500/50 group-hover:shadow-lg group-hover:shadow-pink-500/20 transition-all">
                        <Camera className="w-7 h-7 text-pink-400" />
                      </div>
                      <div className="text-xs text-gray-400">
                        Accuracy: <span className="text-white font-bold">92.7%</span>
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-xl font-bold text-white mb-2.5 group-hover:text-cyan-400 transition-colors">Live Webcam</h3>
                      <p className="text-sm text-gray-400 mb-5 leading-relaxed">Real-time deepfake analysis & verification</p>
                      
                      <div className="space-y-2.5 mb-5">
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Live Deepfake Alert</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                          <span>Facial Authentication</span>
                        </div>
                        <div className="flex items-center gap-2.5 text-xs text-gray-300">
                          <Check className="w-3.5 h-3.5 text-orange-400 flex-shrink-0" />
                          <span>Low-Light Analysis</span>
                        </div>
                      </div>
                      
                      <button className="w-full py-3 text-xs font-bold text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10 rounded-lg transition-all flex items-center justify-center gap-2 border border-transparent hover:border-cyan-500/30">
                        START ANALYSIS
                        <span className="text-base">→</span>
                      </button>
                    </div>
                  </div>
                </Card>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}