"""
Real-time Audio Processing for Deepfake Detection
Handles live audio stream processing and analysis.
"""
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import sounddevice as sd
import torch
import torchaudio
import webrtcvad
from scipy import signal


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    frames_per_buffer: int = 1024
    vad_aggressiveness: int = 3


class RealTimeAudioProcessor:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[AudioConfig] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the real-time audio processor.

        Args:
            model: Pre-trained audio model for inference
            config: Audio configuration
            callback: Function to call with analysis results
        """
        self.config = config or AudioConfig()
        self.model = model
        self.callback = callback
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)

        # Initialize audio stream
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.frames_per_buffer,
            callback=self._audio_callback,
        )

        # Initialize model and move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        # Initialize preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=64,
        ).to(self.device)

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio stream status: {status}")

        # Convert to mono if needed
        if len(indata.shape) > 1:
            indata = np.mean(indata, axis=1)

        # Add to processing queue
        self.audio_queue.put(indata.copy())

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Preprocess audio for model input."""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Compute mel spectrogram
        with torch.no_grad():
            mel = self.mel_transform(audio_tensor.unsqueeze(0))
            log_mel = torch.log(mel + 1e-6)

        return log_mel.unsqueeze(0)  # Add batch dimension

    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """Process a single audio chunk."""
        try:
            # Preprocess
            input_tensor = self._preprocess_audio(audio_chunk)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs["logits"], dim=1)
                fake_prob = probs[0, 1].item()

            return {
                "is_deepfake": fake_prob > 0.5,
                "confidence": fake_prob,
                "attention_weights": outputs.get("attention_weights", None),
                "timestamp": time.time(),
            }

        except Exception as e:
            print(f"Error processing audio: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def start(self):
        """Start processing audio."""
        if self.is_running:
            print("Audio processor is already running")
            return

        self.is_running = True
        self.stream.start()
        print("Audio processor started")

        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self.processing_thread.start()

    def stop(self):
        """Stop processing audio."""
        if not self.is_running:
            return

        self.is_running = False
        self.stream.stop()
        if hasattr(self, "processing_thread"):
            self.processing_thread.join(timeout=1.0)
        print("Audio processor stopped")

    def _processing_loop(self):
        """Main processing loop."""
        buffer = []
        min_buffer_size = self.config.sample_rate  # 1 second of audio

        while self.is_running:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get_nowait()
                    buffer.append(chunk)

                    # Process when we have enough data
                    if len(buffer) * self.config.frames_per_buffer >= min_buffer_size:
                        # Concatenate chunks
                        audio_chunk = np.concatenate(buffer)

                        # Process chunk
                        result = self._process_audio_chunk(audio_chunk)

                        # Call callback if provided
                        if self.callback and "error" not in result:
                            self.callback(result)

                        # Clear buffer
                        buffer = []
                else:
                    # Sleep briefly if queue is empty
                    time.sleep(0.01)

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.1)


# Example usage
if __name__ == "__main__":
    # Example model (replace with your actual model)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 16, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(16, 2)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).squeeze(-1).squeeze(-1)
            return {"logits": self.fc(x)}

    # Create processor
    model = DummyModel()
    config = AudioConfig(sample_rate=16000, frames_per_buffer=1024)

    def callback(result):
        print(
            f"Deepfake: {result['is_deepfake']} (Confidence: {result['confidence']:.2f})"
        )

    processor = RealTimeAudioProcessor(model, config, callback)

    try:
        print("Starting audio processing... (Press Ctrl+C to stop)")
        processor.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        processor.stop()
