/**
 * Python Bridge - Connects Node.js/Express with Python deepfake detection
 */

const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');
const axios = require('axios');

let pythonServer = null;
let serverPort = 5001; // Python server port (different from Express)
const SERVER_URL = `http://localhost:${serverPort}`;
const ANIMATION_CACHE = {};

/**
 * Start the Python server as a child process
 */
async function startPythonServer() {
    if (pythonServer) {
        console.log('Python server already running');
        return true;
    }

    // Path to Python script
    const scriptPath = path.join(__dirname, 'python', 'main.py');
    
    // Check if script exists
    if (!fs.existsSync(scriptPath)) {
        console.error(`Python script not found at: ${scriptPath}`);
        return false;
    }
    
    // Determine Python executable (python3 for Unix, python for Windows)
    const pythonExe = os.platform() === 'win32' ? 'python' : 'python3';
    
    try {
        // Start the Python server as a child process
        console.log(`Starting Python server: ${pythonExe} ${scriptPath} --port ${serverPort}`);
        pythonServer = spawn(pythonExe, [scriptPath, '--port', serverPort.toString()]);
        
        // Handle stdout
        pythonServer.stdout.on('data', (data) => {
            console.log(`[Python] ${data.toString().trim()}`);
        });
        
        // Handle stderr
        pythonServer.stderr.on('data', (data) => {
            console.error(`[Python Error] ${data.toString().trim()}`);
        });
        
        // Handle process exit
        pythonServer.on('exit', (code, signal) => {
            console.log(`Python server exited with code ${code} and signal ${signal}`);
            pythonServer = null;
        });
        
        // Wait for server to be ready
        const isReady = await waitForServerReady();
        
        return isReady;
    } catch (error) {
        console.error('Failed to start Python server:', error);
        return false;
    }
}

/**
 * Stop the Python server
 */
function stopPythonServer() {
    if (pythonServer) {
        pythonServer.kill();
        pythonServer = null;
        console.log('Python server stopped');
        return true;
    }
    return false;
}

/**
 * Wait for the Python server to be ready
 */
async function waitForServerReady(maxAttempts = 30, interval = 1000) {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            const response = await axios.get(`${SERVER_URL}/health`);
            if (response.data.status === 'ready') {
                console.log('Python server is ready');
                return true;
            }
        } catch (error) {
            // Server not ready yet, try again after interval
        }
        
        await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    console.error('Timed out waiting for Python server to be ready');
    return false;
}

/**
 * Login to the Python server
 */
async function login(username, password) {
    try {
        const response = await axios.post(`${SERVER_URL}/api/login`, {
            username,
            password
        });
        
        return response.data;
    } catch (error) {
        console.error('Login error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Logout from the Python server
 */
async function logout() {
    try {
        const response = await axios.post(`${SERVER_URL}/api/logout`);
        return response.data;
    } catch (error) {
        console.error('Logout error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Validate user session
 */
async function validateSession(token) {
    try {
        const response = await axios.get(`${SERVER_URL}/api/user`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        
        return response.data;
    } catch (error) {
        console.error('Session validation error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Send an image to the Python server for analysis
 */
async function analyzeImage(imageData, token = null) {
    try {
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/analyze/image`, {
            image_data: imageData
        }, { headers });
        
        return response.data;
    } catch (error) {
        console.error('Image analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Send a video to the Python server for analysis
 */
async function analyzeVideo(videoBuffer, filename, token = null) {
    try {
        const formData = new FormData();
        formData.append('video', new Blob([videoBuffer]), filename);
        
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/analyze/video`, formData, {
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        
        return response.data;
    } catch (error) {
        console.error('Video analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Send audio to the Python server for analysis
 */
async function analyzeAudio(audioBuffer, filename, token = null) {
    try {
        const formData = new FormData();
        formData.append('audio', new Blob([audioBuffer]), filename);
        
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/analyze/audio`, formData, {
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        
        return response.data;
    } catch (error) {
        console.error('Audio analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Send multiple media types to the Python server for multimodal analysis
 */
async function analyzeMultimodal(imageBuffer, audioBuffer, videoBuffer, token = null) {
    try {
        const formData = new FormData();
        
        if (imageBuffer) {
            formData.append('image', new Blob([imageBuffer]), 'image.jpg');
        }
        
        if (audioBuffer) {
            formData.append('audio', new Blob([audioBuffer]), 'audio.wav');
        }
        
        if (videoBuffer) {
            formData.append('video', new Blob([videoBuffer]), 'video.mp4');
        }
        
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/analyze/multimodal`, formData, {
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        
        return response.data;
    } catch (error) {
        console.error('Multimodal analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Send webcam image data to the Python server for analysis
 */
async function analyzeWebcam(imageData, token = null) {
    return analyzeImage(imageData, token);
}

/**
 * Generate 3D neural network animation
 */
async function generateNeuralNetworkAnimation(authenticity = "AUTHENTIC MEDIA", confidence = 85, token = null) {
    try {
        // Check if animation is cached
        const cacheKey = `nn_${authenticity}_${confidence}`;
        if (ANIMATION_CACHE[cacheKey]) {
            return ANIMATION_CACHE[cacheKey];
        }
        
        // Execute Python animation script directly
        const pythonExe = os.platform() === 'win32' ? 'python' : 'python3';
        const animationScript = `
import sys
sys.path.append('${path.join(__dirname, 'python')}')
from animations import NeuralNetworkAnimation
import json

# Create animation
animation = NeuralNetworkAnimation()
frames = animation.create_animation(frames=30, fps=15, authenticity="${authenticity}", confidence=${confidence})

# Output as JSON
print(json.dumps(frames))
`;
        
        const tempScriptPath = path.join(os.tmpdir(), `anim_nn_${Date.now()}.py`);
        fs.writeFileSync(tempScriptPath, animationScript);
        
        const frames = await new Promise((resolve, reject) => {
            exec(`${pythonExe} ${tempScriptPath}`, {
                maxBuffer: 10 * 1024 * 1024 // 10MB buffer
            }, (error, stdout, stderr) => {
                fs.unlinkSync(tempScriptPath);
                
                if (error) {
                    console.error('Animation error:', stderr);
                    reject(error);
                } else {
                    try {
                        const framesData = JSON.parse(stdout);
                        resolve(framesData);
                    } catch (e) {
                        reject(new Error('Failed to parse animation frames'));
                    }
                }
            });
        });
        
        // Cache the animation
        ANIMATION_CACHE[cacheKey] = frames;
        
        return frames;
    } catch (error) {
        console.error('Neural network animation error:', error);
        throw error;
    }
}

/**
 * Generate 3D blockchain verification animation
 */
async function generateBlockchainAnimation(isVerified = true, token = null) {
    try {
        // Check if animation is cached
        const cacheKey = `bc_${isVerified}`;
        if (ANIMATION_CACHE[cacheKey]) {
            return ANIMATION_CACHE[cacheKey];
        }
        
        // Execute Python animation script directly
        const pythonExe = os.platform() === 'win32' ? 'python' : 'python3';
        const animationScript = `
import sys
sys.path.append('${path.join(__dirname, 'python')}')
from animations import BlockchainAnimation
import json

# Create animation
animation = BlockchainAnimation()
frames = animation.create_animation(frames=30, fps=15, is_verified=${isVerified ? 'True' : 'False'})

# Output as JSON
print(json.dumps(frames))
`;
        
        const tempScriptPath = path.join(os.tmpdir(), `anim_bc_${Date.now()}.py`);
        fs.writeFileSync(tempScriptPath, animationScript);
        
        const frames = await new Promise((resolve, reject) => {
            exec(`${pythonExe} ${tempScriptPath}`, {
                maxBuffer: 10 * 1024 * 1024 // 10MB buffer
            }, (error, stdout, stderr) => {
                fs.unlinkSync(tempScriptPath);
                
                if (error) {
                    console.error('Animation error:', stderr);
                    reject(error);
                } else {
                    try {
                        const framesData = JSON.parse(stdout);
                        resolve(framesData);
                    } catch (e) {
                        reject(new Error('Failed to parse animation frames'));
                    }
                }
            });
        });
        
        // Cache the animation
        ANIMATION_CACHE[cacheKey] = frames;
        
        return frames;
    } catch (error) {
        console.error('Blockchain animation error:', error);
        throw error;
    }
}

/**
 * Generate 3D waveform animation for lip-sync analysis
 */
async function generateWaveformAnimation(language = "english", syncScore = 85, token = null) {
    try {
        // Check if animation is cached
        const cacheKey = `wf_${language}_${syncScore}`;
        if (ANIMATION_CACHE[cacheKey]) {
            return ANIMATION_CACHE[cacheKey];
        }
        
        // Execute Python animation script directly
        const pythonExe = os.platform() === 'win32' ? 'python' : 'python3';
        const animationScript = `
import sys
sys.path.append('${path.join(__dirname, 'python')}')
from animations import WaveformAnimation
import json

# Create animation
animation = WaveformAnimation(language="${language}")
frames = animation.create_animation(frames=30, fps=15, sync_score=${syncScore})

# Output as JSON
print(json.dumps(frames))
`;
        
        const tempScriptPath = path.join(os.tmpdir(), `anim_wf_${Date.now()}.py`);
        fs.writeFileSync(tempScriptPath, animationScript);
        
        const frames = await new Promise((resolve, reject) => {
            exec(`${pythonExe} ${tempScriptPath}`, {
                maxBuffer: 10 * 1024 * 1024 // 10MB buffer
            }, (error, stdout, stderr) => {
                fs.unlinkSync(tempScriptPath);
                
                if (error) {
                    console.error('Animation error:', stderr);
                    reject(error);
                } else {
                    try {
                        const framesData = JSON.parse(stdout);
                        resolve(framesData);
                    } catch (e) {
                        reject(new Error('Failed to parse animation frames'));
                    }
                }
            });
        });
        
        // Cache the animation
        ANIMATION_CACHE[cacheKey] = frames;
        
        return frames;
    } catch (error) {
        console.error('Waveform animation error:', error);
        throw error;
    }
}

/**
 * Verify media on SatyaChain blockchain
 */
async function verifySatyaChain(mediaHash, token = null) {
    try {
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/verify/blockchain`, {
            media_hash: mediaHash
        }, { headers });
        
        return response.data;
    } catch (error) {
        console.error('Blockchain verification error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Check media on darkweb
 */
async function checkDarkweb(mediaHash, token = null) {
    try {
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/check/darkweb`, {
            media_hash: mediaHash
        }, { headers });
        
        return response.data;
    } catch (error) {
        console.error('Darkweb check error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Analyze lip sync for a specific language
 */
async function analyzeLanguageLipSync(videoBuffer, language = 'english', token = null) {
    try {
        const formData = new FormData();
        formData.append('video', new Blob([videoBuffer]), 'video.mp4');
        formData.append('language', language);
        
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/analyze/lip-sync`, formData, {
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        
        return response.data;
    } catch (error) {
        console.error('Lip sync analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Analyze emotion conflict
 */
async function analyzeEmotionConflict(videoBuffer, token = null) {
    try {
        const formData = new FormData();
        formData.append('video', new Blob([videoBuffer]), 'video.mp4');
        
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.post(`${SERVER_URL}/api/analyze/emotion-conflict`, formData, {
            headers,
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });
        
        return response.data;
    } catch (error) {
        console.error('Emotion conflict analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Get information about available models
 */
async function getModelsInfo(token = null) {
    try {
        const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
        
        const response = await axios.get(`${SERVER_URL}/api/models/info`, { headers });
        
        return response.data;
    } catch (error) {
        console.error('Models info error:', error.response?.data || error.message);
        throw error;
    }
}

// Export the functions
module.exports = {
    startPythonServer,
    stopPythonServer,
    waitForServerReady,
    login,
    logout,
    validateSession,
    analyzeImage,
    analyzeVideo,
    analyzeAudio,
    analyzeMultimodal,
    analyzeWebcam,
    generateNeuralNetworkAnimation,
    generateBlockchainAnimation,
    generateWaveformAnimation,
    verifySatyaChain,
    checkDarkweb,
    analyzeLanguageLipSync,
    analyzeEmotionConflict,
    getModelsInfo
};