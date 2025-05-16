/**
 * Type definitions for Python Bridge
 */

export function startPythonServer(): Promise<boolean>;
export function stopPythonServer(): void;
export function checkServerRunning(): Promise<boolean>;
export function waitForServerReady(maxAttempts?: number, interval?: number): Promise<boolean>;
export function analyzeImage(imageBuffer: Buffer, token?: string | null): Promise<any>;
export function analyzeVideo(videoBuffer: Buffer, filename?: string, token?: string | null): Promise<any>;
export function analyzeAudio(audioBuffer: Buffer, filename?: string, token?: string | null): Promise<any>;
export function analyzeMultimodal(imageBuffer?: Buffer | null, audioBuffer?: Buffer | null, videoBuffer?: Buffer | null, token?: string | null): Promise<any>;
export function analyzeWebcam(imageData: string, token?: string | null): Promise<any>;