""
Benchmarking Script for Video Analysis
Compares performance of different video processing approaches.
"""
import os
import time
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from server.python.models.video_model_enhanced import VideoProcessor as BasicProcessor
from server.python.models.video_optimized import VideoAnalyzer, optimize_model

def load_sample_videos(directory: str, max_videos: int = 5):
    """Load sample videos from directory."""
    video_exts = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_exts:
        video_files.extend(list(Path(directory).rglob(f'*{ext}')))
    
    return video_files[:max_videos]

def benchmark_basic_processor(video_path: str, device: str):
    """Benchmark the basic video processor."""
    processor = BasicProcessor(device=device)
    
    start_time = time.time()
    try:
        result = processor.analyze_video(
            str(video_path),
            use_temporal=True,
            frame_analysis=True
        )
        return {
            'success': True,
            'time': time.time() - start_time,
            'num_frames': result.get('num_frames', 0),
            'prediction': result.get('final_prediction', 'Unknown'),
            'confidence': result.get('confidence', 0.0)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time': time.time() - start_time
        }

def benchmark_optimized_processor(video_path: str, model, device: str):
    """Benchmark the optimized video analyzer."""
    analyzer = VideoAnalyzer(model, device=device, batch_size=16, num_workers=4)
    
    start_time = time.time()
    try:
        result = analyzer.analyze_video(
            str(video_path),
            frame_interval=1,
            max_frames=300
        )
        return {
            'success': True,
            'time': result['processing_time'],
            'num_frames': len(result['frame_results']),
            'prediction': result['prediction'],
            'confidence': result['confidence']
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'time': time.time() - start_time
        }

def main():
    parser = argparse.ArgumentParser(description='Benchmark video processing performance')
    parser.add_argument('--video-dir', type=str, required=True, help='Directory containing test videos')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run benchmarks on (cuda/cpu)')
    parser.add_argument('--num-runs', type=int, default=3, help='Number of runs per video')
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f"Benchmarking on {args.device.upper()}")
    print(f"Device: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*50}\n")
    
    # Load sample videos
    video_files = load_sample_videos(args.video_dir)
    if not video_files:
        print(f"No video files found in {args.video_dir}")
        return
    
    print(f"Found {len(video_files)} videos for benchmarking\n")
    
    # Load and optimize model
    print("Loading and optimizing model...")
    try:
        from server.python.models.temporal_models import VideoDeepfakeDetector
        model = VideoDeepfakeDetector()
        model = optimize_model(model)
        model = model.to(args.device)
        print("Model loaded and optimized successfully!\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Benchmark each video
    results = []
    
    for video_path in video_files:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {video_path.name}")
        print(f"Size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        
        # Benchmark basic processor
        print("\nBenchmarking Basic Processor...")
        basic_times = []
        for i in range(args.num_runs):
            print(f"Run {i+1}/{args.num_runs}...", end=' ', flush=True)
            result = benchmark_basic_processor(str(video_path), args.device)
            if result['success']:
                basic_times.append(result['time'])
                print(f"{result['time']:.2f}s")
            else:
                print(f"Failed: {result['error']}")
        
        # Benchmark optimized processor
        print("\nBenchmarking Optimized Processor...")
        optimized_times = []
        for i in range(args.num_runs):
            print(f"Run {i+1}/{args.num_runs}...", end=' ', flush=True)
            result = benchmark_optimized_processor(str(video_path), model, args.device)
            if result['success']:
                optimized_times.append(result['time'])
                print(f"{result['time']:.2f}s")
            else:
                print(f"Failed: {result['error']}")
        
        # Calculate statistics
        if basic_times and optimized_times:
            avg_basic = sum(basic_times) / len(basic_times)
            avg_optimized = sum(optimized_times) / len(optimized_times)
            speedup = (avg_basic - avg_optimized) / avg_basic * 100
            
            results.append({
                'video': video_path.name,
                'basic_avg': avg_basic,
                'optimized_avg': avg_optimized,
                'speedup': speedup
            })
            
            print(f"\nResults for {video_path.name}:")
            print(f"Basic Processor: {avg_basic:.2f}s (avg)")
            print(f"Optimized Processor: {avg_optimized:.2f}s (avg)")
            print(f"Speedup: {speedup:.1f}% faster")
        
        print(f"{'='*50}\n")
    
    # Print summary
    if results:
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        for result in results:
            print(f"\n{result['video']}:")
            print(f"  Basic: {result['basic_avg']:.2f}s")
            print(f"  Optimized: {result['optimized_avg']:.2f}s")
            print(f"  Speedup: {result['speedup']:.1f}%")
        
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\nAverage Speedup: {avg_speedup:.1f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
