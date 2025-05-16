const { spawn } = require('child_process');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

// Configuration for the Python server
const PYTHON_PORT = process.env.PYTHON_API_PORT || 5001;
const PYTHON_HOST = process.env.PYTHON_API_HOST || 'localhost';
const PYTHON_API_URL = `http://${PYTHON_HOST}:${PYTHON_PORT}`;

// Path to Python executable and script
const PYTHON_PATH = 'python3'; // or 'python' depending on your system
const SCRIPT_PATH = path.join(__dirname, 'python', 'app.py');

// Variable to track if the Python server is running
let pythonProcess = null;

/**
 * Start the Python server as a child process
 */
async function startPythonServer() {
  if (pythonProcess) {
    console.log('Python server is already running');
    return;
  }

  console.log('Starting Python server...');
  
  // Make sure the script exists
  if (!fs.existsSync(SCRIPT_PATH)) {
    throw new Error(`Python script not found at ${SCRIPT_PATH}`);
  }

  // Set environment variables for the Python process
  const env = {
    ...process.env,
    PYTHON_API_PORT: PYTHON_PORT,
  };

  // Spawn Python process
  pythonProcess = spawn(PYTHON_PATH, [SCRIPT_PATH], { env });

  // Handle process output
  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python server: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python server error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python server exited with code ${code}`);
    pythonProcess = null;
  });

  // Wait for the server to start
  await waitForServerReady();
}

/**
 * Stop the Python server
 */
function stopPythonServer() {
  if (pythonProcess) {
    console.log('Stopping Python server...');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

/**
 * Wait for the Python server to be ready
 */
async function waitForServerReady(maxAttempts = 30, interval = 1000) {
  console.log(`Waiting for Python server at ${PYTHON_API_URL}/api/python/status...`);
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const response = await axios.get(`${PYTHON_API_URL}/api/python/status`);
      if (response.status === 200) {
        console.log('Python server is ready');
        return true;
      }
    } catch (error) {
      console.log(`Attempt ${attempt + 1}/${maxAttempts}: Python server not ready yet...`);
    }
    
    // Wait before next attempt
    await new Promise(resolve => setTimeout(resolve, interval));
  }
  
  throw new Error('Failed to start Python server after maximum attempts');
}

/**
 * Send an image to the Python server for analysis
 */
async function analyzeImage(imageData) {
  if (!pythonProcess) {
    await startPythonServer();
  }
  
  try {
    const response = await axios.post(`${PYTHON_API_URL}/api/python/analyze/image`, {
      imageData
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing image:', error.message);
    throw error;
  }
}

/**
 * Send a video to the Python server for analysis
 */
async function analyzeVideo(videoBuffer, filename) {
  if (!pythonProcess) {
    await startPythonServer();
  }
  
  try {
    // Create form data for file upload
    const FormData = require('form-data');
    const form = new FormData();
    form.append('video', videoBuffer, {
      filename,
      contentType: 'video/mp4', // Adjust content type as needed
    });
    
    const response = await axios.post(`${PYTHON_API_URL}/api/python/analyze/video`, form, {
      headers: form.getHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing video:', error.message);
    throw error;
  }
}

/**
 * Send audio to the Python server for analysis
 */
async function analyzeAudio(audioBuffer, filename) {
  if (!pythonProcess) {
    await startPythonServer();
  }
  
  try {
    // Create form data for file upload
    const FormData = require('form-data');
    const form = new FormData();
    form.append('audio', audioBuffer, {
      filename,
      contentType: 'audio/mpeg', // Adjust content type as needed
    });
    
    const response = await axios.post(`${PYTHON_API_URL}/api/python/analyze/audio`, form, {
      headers: form.getHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing audio:', error.message);
    throw error;
  }
}

/**
 * Send multiple media types to the Python server for multimodal analysis
 */
async function analyzeMultimodal(imageBuffer, audioBuffer, videoBuffer) {
  if (!pythonProcess) {
    await startPythonServer();
  }
  
  try {
    // Create form data for file upload
    const FormData = require('form-data');
    const form = new FormData();
    
    // Add available media types
    if (imageBuffer) {
      form.append('image', imageBuffer, {
        filename: 'image.jpg',
        contentType: 'image/jpeg',
      });
    }
    
    if (audioBuffer) {
      form.append('audio', audioBuffer, {
        filename: 'audio.mp3',
        contentType: 'audio/mpeg',
      });
    }
    
    if (videoBuffer) {
      form.append('video', videoBuffer, {
        filename: 'video.mp4',
        contentType: 'video/mp4',
      });
    }
    
    const response = await axios.post(`${PYTHON_API_URL}/api/python/analyze/multimodal`, form, {
      headers: form.getHeaders(),
    });
    return response.data;
  } catch (error) {
    console.error('Error performing multimodal analysis:', error.message);
    throw error;
  }
}

/**
 * Send webcam image data to the Python server for analysis
 */
async function analyzeWebcam(imageData) {
  if (!pythonProcess) {
    await startPythonServer();
  }
  
  try {
    const response = await axios.post(`${PYTHON_API_URL}/api/python/analyze/webcam`, {
      imageData
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing webcam image:', error.message);
    throw error;
  }
}

// Export functions for use in Express routes
module.exports = {
  startPythonServer,
  stopPythonServer,
  analyzeImage,
  analyzeVideo,
  analyzeAudio,
  analyzeMultimodal,
  analyzeWebcam
};