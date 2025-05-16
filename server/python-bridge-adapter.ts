/**
 * SatyaAI - Python Bridge Adapter
 * Provides compatibility between ES modules and the Python bridge
 * 
 * ███████╗ █████╗ ████████╗██╗   ██╗ █████╗      █████╗ ██╗
 * ██╔════╝██╔══██╗╚══██╔══╝╚██╗ ██╔╝██╔══██╗    ██╔══██╗██║
 * ███████╗███████║   ██║    ╚████╔╝ ███████║    ███████║██║
 * ╚════██║██╔══██║   ██║     ╚██╔╝  ██╔══██║    ██╔══██║██║
 * ███████║██║  ██║   ██║      ██║   ██║  ██║    ██║  ██║██║
 * ╚══════╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝
 * 
 * SYNTHETIC AUTHENTICATION TECHNOLOGY FOR YOUR ANALYSIS
 */

import { spawn } from 'child_process';
import axios from 'axios';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import FormData from 'form-data';

// Get current file directory (equivalent to __dirname in CommonJS)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Default configuration
const PYTHON_SERVER_PORT = process.env.PYTHON_SERVER_PORT || 5001;
const PYTHON_SERVER_URL = `http://localhost:${PYTHON_SERVER_PORT}`;

// References to the Python server process
let pythonProcess: any = null;

/**
 * Start the Python server as a child process
 */
export async function startPythonServer(): Promise<boolean> {
    try {
        // Check if server is already running
        const serverRunning = await checkServerRunning();
        if (serverRunning) {
            console.log('Python server is already running');
            return true;
        }

        console.log('Starting Python server...');
        
        // Path to the Python script
        const scriptPath = path.join(__dirname, '..', 'run_satyaai.py');
        
        // Check if script exists
        if (!fs.existsSync(scriptPath)) {
            console.error(`Python script not found: ${scriptPath}`);
            return false;
        }
        
        // Start the Python process
        pythonProcess = spawn('python', [scriptPath], {
            stdio: 'pipe',
            detached: false
        });
        
        // Handle stdout
        pythonProcess.stdout.on('data', (data: Buffer) => {
            console.log(`Python server: ${data.toString().trim()}`);
        });
        
        // Handle stderr
        pythonProcess.stderr.on('data', (data: Buffer) => {
            console.error(`Python server error: ${data.toString().trim()}`);
        });
        
        // Handle exit
        pythonProcess.on('exit', (code: number) => {
            console.log(`Python server exited with code: ${code}`);
            pythonProcess = null;
        });
        
        // Handle errors
        pythonProcess.on('error', (err: Error) => {
            console.error(`Failed to start Python server: ${err.message}`);
            pythonProcess = null;
        });
        
        // Wait for server to be ready
        const serverReady = await waitForServerReady();
        if (serverReady) {
            console.log('Python server is ready');
            return true;
        } else {
            console.error('Timed out waiting for Python server to start');
            stopPythonServer();
            return false;
        }
    } catch (error) {
        console.error(`Error starting Python server: ${(error as Error).message}`);
        return false;
    }
}

/**
 * Stop the Python server
 */
export function stopPythonServer(): void {
    if (pythonProcess) {
        console.log('Stopping Python server...');
        
        // Kill the process
        if (process.platform === 'win32') {
            // Windows
            spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
        } else {
            // Unix/Linux/Mac
            pythonProcess.kill('SIGTERM');
        }
        
        pythonProcess = null;
    }
}

/**
 * Check if the Python server is running
 */
export async function checkServerRunning(): Promise<boolean> {
    try {
        const response = await axios.get(`${PYTHON_SERVER_URL}/health`, { 
            timeout: 1000 
        });
        return response.status === 200;
    } catch (error) {
        return false;
    }
}

/**
 * Wait for the Python server to be ready
 */
export async function waitForServerReady(maxAttempts = 30, interval = 1000): Promise<boolean> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const isRunning = await checkServerRunning();
        if (isRunning) {
            return true;
        }
        
        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, interval));
    }
    
    return false;
}

/**
 * Send an image to the Python server for analysis
 */
export async function analyzeImage(imageBuffer: Buffer, token: string | null = null): Promise<any> {
    try {
        // Create a multipart form
        const form = new FormData();
        form.append('image', imageBuffer, {
            filename: 'image.jpg',
            contentType: 'image/jpeg'
        });
        
        // Add token if provided
        if (token) {
            form.append('token', token);
        }
        
        // Send request to Python server
        const response = await axios.post(`${PYTHON_SERVER_URL}/api/analyze/image`, form, {
            headers: form.getHeaders(),
            timeout: 30000  // 30 seconds timeout
        });
        
        return response.data;
    } catch (error: any) {
        console.error(`Error analyzing image: ${error.message}`);
        if (error.response) {
            console.error(`Server response: ${JSON.stringify(error.response.data)}`);
        }
        
        // Return error information
        return {
            error: 'Failed to analyze image',
            details: error.message,
            authenticity: 'ANALYSIS FAILED',
            confidence: 0
        };
    }
}

/**
 * Send a video to the Python server for analysis
 */
export async function analyzeVideo(videoBuffer: Buffer, filename = 'video.mp4', token: string | null = null): Promise<any> {
    try {
        // Create a multipart form
        const form = new FormData();
        form.append('video', videoBuffer, {
            filename,
            contentType: 'video/mp4'
        });
        
        // Add token if provided
        if (token) {
            form.append('token', token);
        }
        
        // Send request to Python server
        const response = await axios.post(`${PYTHON_SERVER_URL}/api/analyze/video`, form, {
            headers: form.getHeaders(),
            timeout: 60000  // 60 seconds timeout
        });
        
        return response.data;
    } catch (error: any) {
        console.error(`Error analyzing video: ${error.message}`);
        if (error.response) {
            console.error(`Server response: ${JSON.stringify(error.response.data)}`);
        }
        
        // Return error information
        return {
            error: 'Failed to analyze video',
            details: error.message,
            authenticity: 'ANALYSIS FAILED',
            confidence: 0
        };
    }
}

/**
 * Send audio to the Python server for analysis
 */
export async function analyzeAudio(audioBuffer: Buffer, filename = 'audio.mp3', token: string | null = null): Promise<any> {
    try {
        // Create a multipart form
        const form = new FormData();
        form.append('audio', audioBuffer, {
            filename,
            contentType: 'audio/mpeg'
        });
        
        // Add token if provided
        if (token) {
            form.append('token', token);
        }
        
        // Send request to Python server
        const response = await axios.post(`${PYTHON_SERVER_URL}/api/analyze/audio`, form, {
            headers: form.getHeaders(),
            timeout: 30000  // 30 seconds timeout
        });
        
        return response.data;
    } catch (error: any) {
        console.error(`Error analyzing audio: ${error.message}`);
        if (error.response) {
            console.error(`Server response: ${JSON.stringify(error.response.data)}`);
        }
        
        // Return error information
        return {
            error: 'Failed to analyze audio',
            details: error.message,
            authenticity: 'ANALYSIS FAILED',
            confidence: 0
        };
    }
}

/**
 * Send multiple media types to the Python server for multimodal analysis
 */
export async function analyzeMultimodal(
    imageBuffer: Buffer | null = null, 
    audioBuffer: Buffer | null = null, 
    videoBuffer: Buffer | null = null, 
    token: string | null = null
): Promise<any> {
    try {
        // Create a multipart form
        const form = new FormData();
        
        // Add media files if provided
        if (imageBuffer) {
            form.append('image', imageBuffer, {
                filename: 'image.jpg',
                contentType: 'image/jpeg'
            });
        }
        
        if (audioBuffer) {
            form.append('audio', audioBuffer, {
                filename: 'audio.mp3',
                contentType: 'audio/mpeg'
            });
        }
        
        if (videoBuffer) {
            form.append('video', videoBuffer, {
                filename: 'video.mp4',
                contentType: 'video/mp4'
            });
        }
        
        // Add token if provided
        if (token) {
            form.append('token', token);
        }
        
        // Send request to Python server
        const response = await axios.post(`${PYTHON_SERVER_URL}/api/analyze/multimodal`, form, {
            headers: form.getHeaders(),
            timeout: 60000  // 60 seconds timeout
        });
        
        return response.data;
    } catch (error: any) {
        console.error(`Error performing multimodal analysis: ${error.message}`);
        if (error.response) {
            console.error(`Server response: ${JSON.stringify(error.response.data)}`);
        }
        
        // Return error information
        return {
            error: 'Failed to perform multimodal analysis',
            details: error.message,
            authenticity: 'ANALYSIS FAILED',
            confidence: 0
        };
    }
}

/**
 * Send webcam image data to the Python server for analysis
 */
export async function analyzeWebcam(imageData: string, token: string | null = null): Promise<any> {
    try {
        // Send request to Python server
        const response = await axios.post(`${PYTHON_SERVER_URL}/api/analyze/webcam`, {
            imageData,
            token
        }, {
            timeout: 30000  // 30 seconds timeout
        });
        
        return response.data;
    } catch (error: any) {
        console.error(`Error analyzing webcam image: ${error.message}`);
        if (error.response) {
            console.error(`Server response: ${JSON.stringify(error.response.data)}`);
        }
        
        // Return error information
        return {
            error: 'Failed to analyze webcam image',
            details: error.message,
            authenticity: 'ANALYSIS FAILED',
            confidence: 0
        };
    }
}