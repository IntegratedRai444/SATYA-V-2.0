// Python-Node.js Bridge for SatyaAI Deepfake Detection System
const { spawn } = require('child_process');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');

let pythonProcess = null;
const PYTHON_SERVER_PORT = 5000;
const BASE_URL = `http://localhost:${PYTHON_SERVER_PORT}`;
let apiToken = null;

/**
 * Start the Python server as a child process
 */
async function startPythonServer() {
  if (pythonProcess) {
    console.log('Python server is already running');
    return true;
  }

  try {
    const pythonPath = 'python3';
    const scriptPath = path.join(__dirname, 'python', 'app.py');
    
    // Check if the script exists
    if (!fs.existsSync(scriptPath)) {
      console.error(`Python script not found at: ${scriptPath}`);
      return false;
    }
    
    // Spawn the Python process
    pythonProcess = spawn(pythonPath, [scriptPath]);
    
    // Handle process output and errors
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python server: ${data.toString()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python server error: ${data.toString()}`);
    });
    
    pythonProcess.on('close', (code) => {
      console.log(`Python server exited with code ${code}`);
      pythonProcess = null;
    });
    
    // Wait for the server to be ready
    return await waitForServerReady();
  } catch (error) {
    console.error('Failed to start Python server:', error);
    return false;
  }
}

/**
 * Stop the Python server
 */
function stopPythonServer() {
  if (!pythonProcess) {
    console.log('Python server is not running');
    return;
  }
  
  pythonProcess.kill();
  pythonProcess = null;
  console.log('Python server stopped');
}

/**
 * Wait for the Python server to be ready
 */
async function waitForServerReady(maxAttempts = 30, interval = 1000) {
  console.log('Waiting for Python server to be ready...');
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const response = await axios.get(`${BASE_URL}/health`);
      if (response.status === 200 && response.data.status === 'ready') {
        console.log('Python server is ready');
        return true;
      }
    } catch (error) {
      // Server not ready yet, retry after interval
    }
    
    // Wait for the specified interval
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  console.error(`Python server not ready after ${maxAttempts} attempts`);
  return false;
}

/**
 * Login to the Python server
 */
async function login(username, password) {
  try {
    const response = await axios.post(`${BASE_URL}/login`, {
      username,
      password
    });
    
    if (response.status === 200 && response.data.session_token) {
      // Store token for future requests
      apiToken = response.data.session_token;
      return {
        success: true,
        user: {
          id: response.data.user_id,
          username: response.data.username
        },
        token: response.data.session_token
      };
    }
    
    return {
      success: false,
      message: 'Authentication failed'
    };
  } catch (error) {
    console.error('Login error:', error.message);
    return {
      success: false,
      message: error.response?.data?.error || 'Authentication failed'
    };
  }
}

/**
 * Logout from the Python server
 */
async function logout() {
  if (!apiToken) {
    return { success: false, message: 'Not logged in' };
  }
  
  try {
    const response = await axios.post(
      `${BASE_URL}/logout`,
      {},
      {
        headers: {
          'Authorization': `Bearer ${apiToken}`
        }
      }
    );
    
    apiToken = null;
    
    return {
      success: true,
      message: 'Successfully logged out'
    };
  } catch (error) {
    console.error('Logout error:', error.message);
    return {
      success: false,
      message: error.response?.data?.error || 'Logout failed'
    };
  }
}

/**
 * Validate user session
 */
async function validateSession(token) {
  try {
    const response = await axios.get(
      `${BASE_URL}/session`,
      {
        headers: {
          'Authorization': `Bearer ${token || apiToken}`
        }
      }
    );
    
    return {
      valid: true,
      user: response.data.user
    };
  } catch (error) {
    return {
      valid: false,
      message: error.response?.data?.error || 'Invalid session'
    };
  }
}

/**
 * Send an image to the Python server for analysis
 */
async function analyzeImage(imageData, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/image`,
      { image_data: imageData },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing image:', error.message);
    return {
      error: true,
      message: 'Failed to analyze image',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Send a video to the Python server for analysis
 */
async function analyzeVideo(videoBuffer, filename, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Create a form data object for the file upload
    const formData = new FormData();
    formData.append('video', videoBuffer, {
      filename: filename || 'video.mp4',
      contentType: 'video/mp4'
    });
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/video`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing video:', error.message);
    return {
      error: true,
      message: 'Failed to analyze video',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Send audio to the Python server for analysis
 */
async function analyzeAudio(audioBuffer, filename, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Create a form data object for the file upload
    const formData = new FormData();
    formData.append('audio', audioBuffer, {
      filename: filename || 'audio.wav',
      contentType: 'audio/wav'
    });
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/audio`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing audio:', error.message);
    return {
      error: true,
      message: 'Failed to analyze audio',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Send multiple media types to the Python server for multimodal analysis
 */
async function analyzeMultimodal(imageBuffer, audioBuffer, videoBuffer, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Create a form data object for the file uploads
    const formData = new FormData();
    
    if (imageBuffer) {
      formData.append('image', imageBuffer, {
        filename: 'image.jpg',
        contentType: 'image/jpeg'
      });
    }
    
    if (audioBuffer) {
      formData.append('audio', audioBuffer, {
        filename: 'audio.wav',
        contentType: 'audio/wav'
      });
    }
    
    if (videoBuffer) {
      formData.append('video', videoBuffer, {
        filename: 'video.mp4',
        contentType: 'video/mp4'
      });
    }
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/multimodal`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing multimodal data:', error.message);
    return {
      error: true,
      message: 'Failed to analyze multimodal data',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Send webcam image data to the Python server for analysis
 */
async function analyzeWebcam(imageData, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/webcam`,
      { image_data: imageData },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing webcam data:', error.message);
    return {
      error: true,
      message: 'Failed to analyze webcam data',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Verify media on SatyaChain blockchain
 */
async function verifySatyaChain(mediaHash, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/verify/blockchain`,
      { media_hash: mediaHash },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error verifying on blockchain:', error.message);
    return {
      error: true,
      message: 'Failed to verify on blockchain',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Check media on darkweb
 */
async function checkDarkweb(mediaHash, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/check/darkweb`,
      { media_hash: mediaHash },
      {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error checking darkweb:', error.message);
    return {
      error: true,
      message: 'Failed to check darkweb',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Analyze lip sync for a specific language
 */
async function analyzeLanguageLipSync(videoBuffer, language = 'english', token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Create a form data object for the file upload
    const formData = new FormData();
    formData.append('video', videoBuffer, {
      filename: 'video.mp4',
      contentType: 'video/mp4'
    });
    formData.append('language', language);
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/lip-sync`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing lip sync:', error.message);
    return {
      error: true,
      message: 'Failed to analyze lip sync',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Analyze emotion conflict
 */
async function analyzeEmotionConflict(videoBuffer, token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Create a form data object for the file upload
    const formData = new FormData();
    formData.append('video', videoBuffer, {
      filename: 'video.mp4',
      contentType: 'video/mp4'
    });
    
    // Send request to the Python server
    const response = await axios.post(
      `${BASE_URL}/analyze/emotion-conflict`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing emotion conflict:', error.message);
    return {
      error: true,
      message: 'Failed to analyze emotion conflict',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

/**
 * Get information about available models
 */
async function getModelsInfo(token = null) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Check authorization
    const sessionToken = token || apiToken;
    if (!sessionToken) {
      throw new Error('Authentication required');
    }
    
    // Send request to the Python server
    const response = await axios.get(
      `${BASE_URL}/models/info`,
      {
        headers: {
          'Authorization': `Bearer ${sessionToken}`
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error getting models info:', error.message);
    return {
      error: true,
      message: 'Failed to get models info',
      details: error.message,
      code: error.response?.data?.code
    };
  }
}

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
  verifySatyaChain,
  checkDarkweb,
  analyzeLanguageLipSync,
  analyzeEmotionConflict,
  getModelsInfo
};