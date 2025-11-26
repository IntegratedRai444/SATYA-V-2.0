""
Optimized Video Processing for Deepfake Detection
Implements GPU acceleration, frame batching, and memory optimizations.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import time
import logging

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Optimized frame processing with GPU acceleration and batching."""
    
    def __init__(self, model, device: str = 'cuda', batch_size: int = 16, num_workers: int = 4):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model.to(device)
        self.model.eval()
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with dummy data."""
        dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        for _ in range(3):  # Run a few iterations
            _ = self.model(dummy_input.unsqueeze(0))  # Add batch dim
        torch.cuda.synchronize()  # Wait for all kernels to finish
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a single frame for model input."""
        # Convert BGR to RGB and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        
        # Convert to tensor and add batch dim
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # HWC to CHW
        return frame_tensor
    
    def process_batch(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Process a batch of frames."""
        with torch.no_grad():
            # Preprocess frames
            frames_tensor = torch.stack([self.preprocess_frame(f) for f in frames])
            frames_tensor = frames_tensor.to(self.device)
            
            # Run inference
            outputs = self.model(frames_tensor)
            
            # Get probabilities
            if isinstance(outputs, dict) and 'logits' in outputs:
                probs = F.softmax(outputs['logits'], dim=1)
            else:
                probs = F.softmax(outputs, dim=1)
            
            return probs.cpu()  # Move to CPU to free GPU memory

class VideoAnalyzer:
    """High-performance video analysis with parallel processing."""
    
    def __init__(self, model, device: str = 'cuda', batch_size: int = 16, 
                 num_workers: int = 4, max_queue_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        
        # Initialize frame processor
        self.processor = FrameProcessor(model, device, batch_size, num_workers)
        
        # Thread-safe queue for frames
        self.frame_queue = Queue(maxsize=max_queue_size)
        self.result_queue = Queue()
        
        # Thread control
        self._stop_event = threading.Event()
        self._workers = []
        
    def start_workers(self):
        """Start worker threads for parallel processing."""
        self._stop_event.clear()
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            worker.start()
            self._workers.append(worker)
    
    def stop_workers(self):
        """Stop all worker threads."""
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers = []
    
    def _worker_loop(self):
        """Worker thread main loop."""
        batch = []
        batch_indices = []
        
        while not self._stop_event.is_set():
            try:
                # Get frame from queue with timeout
                try:
                    idx, frame = self.frame_queue.get(timeout=0.1)
                except:
                    continue
                
                # Add to current batch
                batch.append(frame)
                batch_indices.append(idx)
                
                # Process batch if full or queue is empty
                if len(batch) >= self.batch_size or self.frame_queue.empty():
                    if batch:  # Ensure batch is not empty
                        try:
                            # Process batch
                            probs = self.processor.process_batch(batch)
                            
                            # Put results in result queue
                            for i, idx in enumerate(batch_indices):
                                self.result_queue.put((idx, {
                                    'frame_idx': idx,
                                    'real_prob': probs[i][0].item(),
                                    'fake_prob': probs[i][1].item(),
                                    'prediction': 'Fake' if probs[i][1] > 0.5 else 'Real',
                                    'confidence': max(probs[i][0].item(), probs[i][1].item())
                                }))
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                        finally:
                            batch = []
                            batch_indices = []
                
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def analyze_video(
        self, 
        video_path: str, 
        frame_interval: int = 1,
        max_frames: int = 300
    ) -> Dict[str, Any]:
        """Analyze video with optimized processing."""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Determine frames to process
            frame_indices = list(range(0, min(total_frames, max_frames), frame_interval))
            
            # Start worker threads
            self.start_workers()
            
            # Process frames
            results = {}
            frame_count = 0
            
            # Read and queue frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Add frame to processing queue
                    self.frame_queue.put((idx, frame))
                    frame_count += 1
            
            # Wait for all frames to be processed
            processed_count = 0
            with tqdm(total=frame_count, desc="Processing frames") as pbar:
                while processed_count < frame_count:
                    try:
                        # Get result with timeout
                        idx, result = self.result_queue.get(timeout=1.0)
                        results[idx] = result
                        processed_count += 1
                        pbar.update(1)
                    except:
                        # Check if workers are still alive
                        if not any(worker.is_alive() for worker in self._workers):
                            break
            
            # Stop workers
            self.stop_workers()
            
            # Aggregate results
            if not results:
                raise RuntimeError("No frames were processed successfully")
            
            # Calculate overall prediction
            fake_probs = [r['fake_prob'] for r in results.values()]
            avg_fake_prob = sum(fake_probs) / len(fake_probs)
            
            return {
                'video_path': video_path,
                'num_frames_processed': len(results),
                'avg_fake_prob': avg_fake_prob,
                'prediction': 'Fake' if avg_fake_prob > 0.5 else 'Real',
                'confidence': max(avg_fake_prob, 1 - avg_fake_prob),
                'frame_results': results,
                'processing_time': time.time() - start_time
            }
            
        finally:
            # Ensure resources are cleaned up
            cap.release()
            self.stop_workers()

def optimize_model(model):
    """Optimize model for inference."""
    model.eval()
    
    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Use torch.jit for optimization if available
    if hasattr(torch.jit, 'script'):
        try:
            model = torch.jit.script(model)
            logger.info("Applied TorchScript optimization")
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
    
    return model
