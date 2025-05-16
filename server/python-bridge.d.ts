// Type definitions for python-bridge.js

export function startPythonServer(): Promise<boolean>;
export function stopPythonServer(): void;
export function analyzeImage(imageData: string): Promise<any>;
export function analyzeVideo(videoBuffer: Buffer, filename: string): Promise<any>;
export function analyzeAudio(audioBuffer: Buffer, filename: string): Promise<any>;
export function analyzeMultimodal(imageBuffer?: Buffer, audioBuffer?: Buffer, videoBuffer?: Buffer): Promise<any>;
export function analyzeWebcam(imageData: string): Promise<any>;
export function waitForServerReady(maxAttempts?: number, interval?: number): Promise<boolean>;