"""
Real-time Audio Processing for Deepfake Detection
Handles live audio stream processing and analysis.
"""
import queue
import threading
import time
import logging
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import numpy as np
import sounddevice as sd
import torch
import torchaudio
# Try to import webrtcvad with fallback
try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False
    logger.warning("webrtcvad not available, using alternative VAD")
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcessingState(Enum):
    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    ERROR = auto()

class AudioChunk:
    def __init__(self, data: np.ndarray, timestamp: float, sample_rate: int):
        self.data = data
        self.timestamp = timestamp
        self.sample_rate = sample_rate
        self.duration = len(data) / sample_rate

class ProcessingMetrics:
    def __init__(self):
        self.start_time = 0.0
        self.processed_chunks = 0
        self.total_audio_seconds = 0.0
        self.avg_processing_time = 0.0
        self.last_update = 0.0
        self.vad_speech_frames = 0
        self.vad_total_frames = 0


@dataclass
class AudioConfig:
    # Audio stream configuration
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    frames_per_buffer: int = 1024
    
    # VAD configuration
    vad_aggressiveness: int = 3
    vad_frame_duration: int = 30  # ms
    
    # Processing parameters
    min_voice_activity_ratio: float = 0.3
    max_chunk_duration: float = 3.0  # seconds
    min_chunk_duration: float = 0.5  # seconds
    
    # Performance settings
    use_gpu: bool = True
    batch_size: int = 1
    
    # Callback intervals (in seconds)
    progress_interval: float = 0.5
    metrics_interval: float = 1.0
    
    def validate(self):
        """Validate configuration parameters."""
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Unsupported sample rate: {self.sample_rate}")
        if self.channels not in [1, 2]:
            raise ValueError(f"Unsupported number of channels: {self.channels}")
        if self.vad_aggressiveness not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid VAD aggressiveness: {self.vad_aggressiveness}")


class AudioProcessorEvent:
    """Base class for audio processor events."""
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ProcessingStarted(AudioProcessorEvent):
    """Event emitted when processing starts."""
    def __init__(self, timestamp: float):
        self.timestamp = timestamp
        self.event_type = "processing_started"


class ProcessingStopped(AudioProcessorEvent):
    """Event emitted when processing stops."""
    def __init__(self, timestamp: float, reason: str = "normal"):
        self.timestamp = timestamp
        self.reason = reason
        self.event_type = "processing_stopped"


class ChunkProcessed(AudioProcessorEvent):
    """Event emitted when an audio chunk is processed."""
    def __init__(self, chunk: AudioChunk, result: Dict[str, Any]):
        self.timestamp = time.time()
        self.duration = chunk.duration
        self.result = result
        self.event_type = "chunk_processed"


class VoiceActivityDetected(AudioProcessorEvent):
    """Event emitted when voice activity is detected."""
    def __init__(self, timestamp: float, duration: float, is_speech: bool):
        self.timestamp = timestamp
        self.duration = duration
        self.is_speech = is_speech
        self.event_type = "voice_activity"


class ProcessingMetricsUpdate(AudioProcessorEvent):
    """Event containing processing metrics."""
    def __init__(self, metrics: 'ProcessingMetrics'):
        self.timestamp = time.time()
        self.metrics = {
            'processed_chunks': metrics.processed_chunks,
            'total_audio_seconds': metrics.total_audio_seconds,
            'avg_processing_time': metrics.avg_processing_time,
            'vad_speech_ratio': (metrics.vad_speech_frames / metrics.vad_total_frames 
                               if metrics.vad_total_frames > 0 else 0.0)
        }
        self.event_type = "metrics_update"


class ProcessingError(AudioProcessorEvent):
    """Event emitted when an error occurs during processing."""
    def __init__(self, error: Exception, context: str = "processing"):
        self.timestamp = time.time()
        self.error_type = type(error).__name__
        self.error_message = str(error)
        self.context = context
        self.event_type = "error"


class RealTimeAudioProcessor:
    """
    Enhanced real-time audio processor with VAD, progress tracking, and event callbacks.
    
    Features:
    - Real-time audio streaming with configurable buffer sizes
    - Voice Activity Detection (VAD) for efficient processing
    - Progress tracking and metrics
    - Event-based callback system
    - Graceful error handling and recovery
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[AudioConfig] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        event_callback: Optional[Callable[[AudioProcessorEvent], None]] = None,
    ):
        """
        Initialize the enhanced real-time audio processor.

        Args:
            model: Pre-trained audio model for inference
            config: Audio configuration (uses defaults if None)
            callback: Legacy callback for backward compatibility
            event_callback: New event-based callback system
        """
        # Initialize configuration
        self.config = config or AudioConfig()
        self.config.validate()
        
        # Model setup
        self.model = model
        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu"
        )
        self.model = self.model.to(self.device).eval()
        
        # State management
        self._state = ProcessingState.IDLE
        self._state_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.clear()
        
        # Callbacks
        self._legacy_callback = callback
        self._event_callbacks = []
        if event_callback:
            self._event_callbacks.append(event_callback)
            
        # Audio processing
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.active_chunks = []
        self._last_metrics_update = 0.0
        self._last_progress_update = 0.0
        self.metrics = ProcessingMetrics()
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)
        self.vad_frame_size = self.config.sample_rate * self.config.vad_frame_duration // 1000
        
        # Initialize preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=512,
            n_mels=64,
        ).to(self.device)
        
        # Initialize audio stream (will be started explicitly)
        self.stream = None
        self._init_audio_stream()
        
        logger.info(f"Audio processor initialized on device: {self.device}")
    
    def _init_audio_stream(self):
        """Initialize the audio input stream."""
        if self.stream is not None:
            self.stream.close()
            
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=self.config.dtype,
            blocksize=self.config.frames_per_buffer,
            callback=self._audio_callback,
            latency='high'  # Better for real-time processing
        )
    
    def register_event_callback(self, callback: Callable[[AudioProcessorEvent], None]):
        """Register a callback for audio processing events."""
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[AudioProcessorEvent], None]):
        """Remove a registered event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    def _notify_event(self, event: AudioProcessorEvent):
        """Notify all registered event callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}", exc_info=True)
        
        # For backward compatibility with legacy callback
        if (isinstance(event, ChunkProcessed) and 
            self._legacy_callback is not None and 
            'error' not in event.result):
            try:
                self._legacy_callback(event.result)
            except Exception as e:
                logger.error(f"Error in legacy callback: {e}", exc_info=True)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Dict, status: sd.CallbackFlags):
        """
        Callback for audio stream with VAD and error handling.
        
        Args:
            indata: Input audio data
            frames: Number of frames in the buffer
            time_info: Dictionary containing timestamps
            status: Status flags for the stream
        """
        try:
            if status:
                logger.warning(f"Audio stream status: {status}")
                if status.input_overflow:
                    self._notify_event(ProcessingError(
                        Exception("Input overflow in audio stream"),
                        context="audio_input"
                    ))
            
            # Skip if paused
            if self._paused.is_set():
                return
                
            # Convert to mono if needed
            if len(indata.shape) > 1:
                indata = np.mean(indata, axis=1)
            
            # Create audio chunk with timestamp
            chunk = AudioChunk(
                data=indata.copy(),
                timestamp=time.time(),
                sample_rate=self.config.sample_rate
            )
            
            # Add to processing queue if not full
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
                self._notify_event(ProcessingError(
                    Exception("Audio processing queue overflow"),
                    context="audio_processing"
                ))
                
        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=True)
            self._notify_event(ProcessingError(e, context="audio_callback"))
    
    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio for model input with validation and normalization.
        
        Args:
            audio: Input audio as numpy array
            
        Returns:
            Processed audio tensor ready for model input
        """
        try:
            # Input validation
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                raise ValueError("Invalid audio input")
                
            # Normalize audio if needed
            if np.abs(audio).max() > 1.0:
                audio = audio / (np.abs(audio).max() + 1e-8)
                
            # Convert to tensor and move to device
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            
            # Compute mel spectrogram with gradient disabled
            with torch.no_grad():
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim
                mel = self.mel_transform(audio_tensor)
                log_mel = torch.log(mel + 1e-6)  # Log scale with small offset
                
            return log_mel
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}", exc_info=True)
            raise
    
    def _process_audio_chunk(self, chunk: AudioChunk) -> Dict[str, Any]:
        """
        Process a single audio chunk with VAD and model inference.
        
        Args:
            chunk: AudioChunk to process
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        result = {
            "is_deepfake": False,
            "confidence": 0.0,
            "timestamp": chunk.timestamp,
            "duration": chunk.duration,
            "vad_decision": False,
            "processing_time": 0.0
        }
        
        try:
            # Apply VAD if enabled
            if hasattr(self, 'vad'):
                result["vad_decision"] = self._check_voice_activity(chunk.data)
                
                # Skip silent chunks based on VAD
                if not result["vad_decision"] and self.config.min_voice_activity_ratio > 0:
                    result["processing_time"] = time.time() - start_time
                    result["skipped"] = True
                    return result
            
            # Preprocess and run model inference
            with torch.no_grad():
                input_tensor = self._preprocess_audio(chunk.data)
                outputs = self.model(input_tensor)
                
                # Handle different model output formats
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs
                
                # Get probabilities
                if logits is not None:
                    probs = torch.softmax(logits, dim=-1)
                    fake_prob = probs[0, 1].item() if probs.dim() > 1 else probs[1].item()
                    
                    result.update({
                        "is_deepfake": fake_prob > 0.5,
                        "confidence": fake_prob,
                        "raw_outputs": outputs,
                    })
            
            # Update metrics
            result["processing_time"] = time.time() - start_time
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing audio chunk: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            result.update({
                "error": error_msg,
                "exception": str(e),
                "processing_time": time.time() - start_time
            })
            
            self._notify_event(ProcessingError(e, context="audio_processing"))
            return result
    
    def _check_voice_activity(self, audio: np.ndarray) -> bool:
        """
        Check for voice activity in audio using VAD.
        
        Args:
            audio: Input audio data
            
        Returns:
            True if voice activity is detected, False otherwise
        """
        if not hasattr(self, 'vad') or self.vad is None:
            return True  # Always process if VAD is disabled
            
        try:
            # Convert to 16-bit PCM for VAD
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)
            
            # Split into VAD frames
            frame_size = self.vad_frame_size
            frames = [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]
            
            # Check each frame for speech
            speech_frames = 0
            for frame in frames:
                if len(frame) < frame_size:
                    continue  # Skip incomplete frames
                    
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.config.sample_rate)
                    speech_frames += int(is_speech)
                except:
                    continue
            
            # Update metrics
            self.metrics.vad_speech_frames += speech_frames
            self.metrics.vad_total_frames += len(frames)
            
            # Check if enough speech is detected
            if len(frames) == 0:
                return False
                
            speech_ratio = speech_frames / len(frames)
            return speech_ratio >= self.config.min_voice_activity_ratio
            
        except Exception as e:
            logger.error(f"VAD check failed: {e}", exc_info=True)
            return True  # Default to processing on VAD error

    def _update_metrics(self, result: Dict[str, Any]):
        """
        Update processing metrics with the latest result.
        
        Args:
            result: Dictionary containing processing results
        """
        with self._state_lock:
            # Update basic metrics
            self.metrics.processed_chunks += 1
            self.metrics.total_audio_seconds += result.get('duration', 0)
            
            # Update average processing time using Welford's algorithm
            processing_time = result.get('processing_time', 0)
            if self.metrics.processed_chunks == 1:
                self.metrics.avg_processing_time = processing_time
            else:
                self.metrics.avg_processing_time = (
                    (self.metrics.avg_processing_time * (self.metrics.processed_chunks - 1) + processing_time) 
                    / self.metrics.processed_chunks
                )
            
            # Update last update time
            self.metrics.last_update = time.time()
    
    def _should_update_metrics(self) -> bool:
        """Check if metrics should be updated based on configured interval."""
        if self.config.metrics_interval <= 0:
            return False
            
        now = time.time()
        if now - self._last_metrics_update >= self.config.metrics_interval:
            self._last_metrics_update = now
            return True
        return False
    
    def _should_update_progress(self) -> bool:
        """Check if progress should be updated based on configured interval."""
        if self.config.progress_interval <= 0:
            return False
            
        now = time.time()
        if now - self._last_progress_update >= self.config.progress_interval:
            self._last_progress_update = now
            return True
        return False
    
    def get_state(self) -> ProcessingState:
        """Get the current processing state."""
        with self._state_lock:
            return self._state
    
    def is_running(self) -> bool:
        """Check if the processor is currently running."""
        return self.get_state() == ProcessingState.RUNNING
    
    def is_paused(self) -> bool:
        """Check if the processor is currently paused."""
        return self._paused.is_set()
    
    def start(self):
        """
        Start the audio processing pipeline.
        
        This will start the audio stream and processing thread.
        """
        with self._state_lock:
            if self._state in [ProcessingState.RUNNING, ProcessingState.STARTING]:
                logger.warning("Audio processor is already running")
                return
                
            self._state = ProcessingState.STARTING
            self._stop_event.clear()
            
            try:
                # Reset metrics
                self.metrics = ProcessingMetrics()
                self.metrics.start_time = time.time()
                
                # Start audio stream
                self.stream.start()
                
                # Start processing thread
                self.processing_thread = threading.Thread(
                    target=self._processing_loop,
                    name="AudioProcessorThread",
                    daemon=True
                )
                self.processing_thread.start()
                
                self._state = ProcessingState.RUNNING
                logger.info("Audio processor started")
                self._notify_event(ProcessingStarted(time.time()))
                
            except Exception as e:
                self._state = ProcessingState.ERROR
                logger.error(f"Failed to start audio processor: {e}", exc_info=True)
                self._notify_event(ProcessingError(e, context="start"))
                raise
    
    def stop(self, reason: str = "user_request"):
        """
        Stop the audio processing pipeline.
        
        Args:
            reason: Reason for stopping (for logging)
        """
        with self._state_lock:
            if self._state in [ProcessingState.STOPPING, ProcessingState.IDLE]:
                return
                
            self._state = ProcessingState.STOPPING
            logger.info(f"Stopping audio processor (reason: {reason})")
            
            # Signal threads to stop
            self._stop_event.set()
            
            try:
                # Stop audio stream
                if self.stream is not None and self.stream.active:
                    self.stream.stop()
                
                # Wait for processing thread to finish
                if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                    self.processing_thread.join(timeout=2.0)
                    
            except Exception as e:
                logger.error(f"Error during processor shutdown: {e}", exc_info=True)
                
            finally:
                self._state = ProcessingState.IDLE
                self._notify_event(ProcessingStopped(time.time(), reason))
                logger.info("Audio processor stopped")
    
    def pause(self):
        """Pause audio processing."""
        if not self.is_running() or self.is_paused():
            return
            
        self._paused.set()
        logger.info("Audio processing paused")
        
    def resume(self):
        """Resume audio processing."""
        if not self.is_paused():
            return
            
        self._paused.clear()
        logger.info("Audio processing resumed")
    
    def _processing_loop(self):
        """
        Main processing loop that handles audio chunks from the queue.
        
        This runs in a separate thread and processes audio chunks as they arrive.
        """
        buffer = []
        buffer_duration = 0.0
        last_chunk_time = time.time()
        
        logger.info("Audio processing thread started")
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Check if we should pause
                    if self._paused.is_set():
                        time.sleep(0.1)
                        continue
                    
                    # Get audio data from queue with timeout
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        buffer.append(chunk)
                        buffer_duration += chunk.duration
                        last_chunk_time = time.time()
                        
                    except queue.Empty:
                        # Process any remaining audio if buffer isn't empty and we've waited long enough
                        current_time = time.time()
                        if (buffer and 
                            (buffer_duration >= self.config.min_chunk_duration or 
                             current_time - last_chunk_time > 0.5)):
                            self._process_buffer(buffer)
                            buffer = []
                            buffer_duration = 0.0
                        continue
                    
                    # Check if we have enough data to process
                    if buffer_duration >= self.config.max_chunk_duration:
                        self._process_buffer(buffer)
                        buffer = []
                        buffer_duration = 0.0
                    
                    # Update metrics periodically
                    if self._should_update_metrics():
                        self._notify_event(ProcessingMetricsUpdate(self.metrics))
                    
                except Exception as e:
                    logger.error(f"Error in processing loop: {e}", exc_info=True)
                    self._notify_event(ProcessingError(e, context="processing_loop"))
                    time.sleep(0.1)  # Prevent tight loop on errors
            
            # Process any remaining audio before exiting
            if buffer:
                self._process_buffer(buffer)
                
        except Exception as e:
            logger.critical(f"Fatal error in processing loop: {e}", exc_info=True)
            self._notify_event(ProcessingError(e, context="processing_loop_fatal"))
            
        finally:
            logger.info("Audio processing thread stopped")
    
    def _process_buffer(self, chunks: List[AudioChunk]):
        """
        Process a buffer of audio chunks.
        
        Args:
            chunks: List of AudioChunk objects to process
        """
        if not chunks:
            return
            
        try:
            # Concatenate chunks
            audio_data = np.concatenate([chunk.data for chunk in chunks])
            
            # Use the timestamp of the first chunk
            chunk = AudioChunk(
                data=audio_data,
                timestamp=chunks[0].timestamp,
                sample_rate=self.config.sample_rate
            )
            
            # Process the chunk
            result = self._process_audio_chunk(chunk)
            
            # Notify listeners
            if 'error' not in result:
                self._notify_event(ChunkProcessed(chunk, result))
                
            # Update progress if needed
            if self._should_update_progress():
                self._notify_event(ProcessingMetricsUpdate(self.metrics))
                
        except Exception as e:
            logger.error(f"Error processing buffer: {e}", exc_info=True)
            self._notify_event(ProcessingError(e, context="process_buffer"))


# Example usage with enhanced features
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
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create configuration with custom settings
    config = AudioConfig(
        sample_rate=16000,
        frames_per_buffer=1024,
        vad_aggressiveness=3,
        min_voice_activity_ratio=0.2,  # Lower threshold for more sensitive VAD
        max_chunk_duration=2.0,        # Process up to 2 seconds of audio at once
        min_chunk_duration=0.5,        # Minimum chunk duration to process
        progress_interval=0.5,          # Update progress every 0.5 seconds
        metrics_interval=1.0            # Update metrics every 1 second
    )
    
    # Create model
    model = DummyModel()
    
    # Event handler for audio processing events
    def handle_audio_event(event: AudioProcessorEvent):
        """Handle all audio processing events."""
        try:
            if isinstance(event, ProcessingStarted):
                logger.info(f"Processing started at {event.timestamp}")
                
            elif isinstance(event, ProcessingStopped):
                logger.info(f"Processing stopped: {event.reason}")
                
            elif isinstance(event, ChunkProcessed):
                # Only log detailed info for non-skipped chunks
                if not event.result.get('skipped', False):
                    result = event.result
                    logger.info(
                        f"Processed {event.duration:.2f}s audio: "
                        f"Deepfake: {result['is_deepfake']} "
                        f"(Confidence: {result['confidence']:.2%}, "
                        f"Processed in {result['processing_time']*1000:.1f}ms)"
                    )
                
            elif isinstance(event, VoiceActivityDetected):
                activity = "Speech" if event.is_speech else "Silence"
                logger.debug(f"VAD: {activity} detected for {event.duration:.2f}s")
                
            elif isinstance(event, ProcessingMetricsUpdate):
                metrics = event.metrics
                logger.info(
                    f"Metrics: {metrics['processed_chunks']} chunks, "
                    f"{metrics['total_audio_seconds']:.1f}s processed, "
                    f"Avg: {metrics['avg_processing_time']*1000:.1f}ms/chunk, "
                    f"VAD: {metrics['vad_speech_ratio']:.1%} speech"
                )
                
            elif isinstance(event, ProcessingError):
                logger.error(
                    f"Error in {event.context}: {event.error_type}: {event.error_message}",
                    exc_info=isinstance(event.error, Exception) and event.error or None
                )
                
        except Exception as e:
            logger.error(f"Error in event handler: {e}", exc_info=True)
    
    # Create processor with event callback
    processor = RealTimeAudioProcessor(
        model=model,
        config=config,
        event_callback=handle_audio_event
    )
    
    # Register additional event callback (you can have multiple)
    def log_all_events(event: AudioProcessorEvent):
        """Log all events for debugging."""
        logger.debug(f"Event: {event.event_type} - {event.to_dict()}")
    
    processor.register_event_callback(log_all_events)
    
    try:
        logger.info("Starting audio processing... (Press Ctrl+C to stop)")
        
        # Start processing
        processor.start()
        
        # Keep the main thread alive and handle user input
        while True:
            try:
                cmd = input("\nEnter command [pause/resume/stop/status/help]: ").strip().lower()
                
                if cmd == 'pause':
                    processor.pause()
                    logger.info("Processing paused")
                    
                elif cmd == 'resume':
                    processor.resume()
                    logger.info("Processing resumed")
                    
                elif cmd == 'status':
                    status = {
                        'state': processor.get_state().name,
                        'paused': processor.is_paused(),
                        'processed_chunks': processor.metrics.processed_chunks,
                        'total_audio_seconds': processor.metrics.total_audio_seconds,
                        'avg_processing_time': processor.metrics.avg_processing_time
                    }
                    logger.info(f"Status: {json.dumps(status, indent=2, default=str)}")
                    
                elif cmd in ['stop', 'quit', 'exit']:
                    logger.info("Stopping processor...")
                    break
                    
                elif cmd in ['h', 'help']:
                    print("\nAvailable commands:")
                    print("  pause   - Pause audio processing")
                    print("  resume  - Resume audio processing")
                    print("  status  - Show current status")
                    print("  stop    - Stop processing and exit")
                    print("  help    - Show this help")
                
            except (KeyboardInterrupt, EOFError):
                logger.info("\nStopping processor...")
                break
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        
    finally:
        # Ensure processor is properly stopped
        processor.stop(reason="application_exit")
        logger.info("Application stopped")
