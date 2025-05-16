// Python-NodeJS Bridge for SatyaAI
const { spawn } = require('child_process');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const FormData = require('form-data');

let pythonProcess = null;
const PYTHON_SERVER_PORT = 5000;
const BASE_URL = `http://localhost:${PYTHON_SERVER_PORT}`;

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
 * Send an image to the Python server for analysis
 */
async function analyzeImage(imageData) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Send request to the Python server
    const response = await axios.post(`${BASE_URL}/analyze/image`, {
      image_data: imageData
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    return {
      error: true,
      message: 'Failed to analyze image',
      details: error.message
    };
  }
}

/**
 * Send a video to the Python server for analysis
 */
async function analyzeVideo(videoBuffer, filename) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Create a form data object for the file upload
    const formData = new FormData();
    formData.append('video', videoBuffer, {
      filename: filename || 'video.mp4',
      contentType: 'video/mp4'
    });
    
    // Send request to the Python server
    const response = await axios.post(`${BASE_URL}/analyze/video`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing video:', error);
    return {
      error: true,
      message: 'Failed to analyze video',
      details: error.message
    };
  }
}

/**
 * Send audio to the Python server for analysis
 */
async function analyzeAudio(audioBuffer, filename) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Create a form data object for the file upload
    const formData = new FormData();
    formData.append('audio', audioBuffer, {
      filename: filename || 'audio.wav',
      contentType: 'audio/wav'
    });
    
    // Send request to the Python server
    const response = await axios.post(`${BASE_URL}/analyze/audio`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing audio:', error);
    return {
      error: true,
      message: 'Failed to analyze audio',
      details: error.message
    };
  }
}

/**
 * Send multiple media types to the Python server for multimodal analysis
 */
async function analyzeMultimodal(imageBuffer, audioBuffer, videoBuffer) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
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
    const response = await axios.post(`${BASE_URL}/analyze/multimodal`, formData, {
      headers: {
        ...formData.getHeaders()
      }
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing multimodal data:', error);
    return {
      error: true,
      message: 'Failed to analyze multimodal data',
      details: error.message
    };
  }
}

/**
 * Send webcam image data to the Python server for analysis
 */
async function analyzeWebcam(imageData) {
  try {
    // Check if the server is running
    if (!pythonProcess) {
      console.log('Python server not running, starting...');
      const serverStarted = await startPythonServer();
      if (!serverStarted) {
        throw new Error('Failed to start Python server');
      }
    }
    
    // Send request to the Python server
    const response = await axios.post(`${BASE_URL}/analyze/webcam`, {
      image_data: imageData
    });
    
    return response.data;
  } catch (error) {
    console.error('Error analyzing webcam data:', error);
    return {
      error: true,
      message: 'Failed to analyze webcam data',
      details: error.message
    };
  }
}

module.exports = {
  startPythonServer,
  stopPythonServer,
  analyzeImage,
  analyzeVideo,
  analyzeAudio,
  analyzeMultimodal,
  analyzeWebcam,
  waitForServerReady
};