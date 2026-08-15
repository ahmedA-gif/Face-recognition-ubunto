#!/usr/bin/env python3
"""
FACE-RECOGNITION-UBUNTO: Performance Benchmark Script

This script benchmarks the inference speed of different models and backends.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.person_yolo import PersonDetector


def load_test_image(size=(640, 640)):
    """Load or create a test image."""
    # Try to load a sample image
    test_image_path = ROOT / "tests" / "test_image.jpg"
    if test_image_path.exists():
        frame = cv2.imread(str(test_image_path))
        if frame is not None:
            frame = cv2.resize(frame, size)
            return frame
    
    # Create a synthetic test image
    frame = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return frame


def benchmark_model(model_path, device="cpu", backend="onnx", iterations=100, imgsz=640):
    """Benchmark a model with the given configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {Path(model_path).name}")
    print(f"Device: {device}, Backend: {backend}, Size: {imgsz}x{imgsz}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}")
    
    try:
        # Initialize detector
        init_start = time.time()
        detector = PersonDetector(
            weights=str(model_path),
            conf=0.45,
            iou=0.5,
            imgsz=imgsz,
            device=device,
            backend=backend,
        )
        init_time = time.time() - init_start
        print(f"[Init] Model loaded in {init_time:.3f}s")
        
        # Load test image
        frame = load_test_image((imgsz, imgsz))
        
        # Warmup
        print("Warming up...")
        for _ in range(5):
            _ = detector.detect(frame)
        
        # Run benchmark
        print("Running benchmark...")
        start_time = time.time()
        for i in range(iterations):
            _ = detector.detect(frame)
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations}...")
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_time = total_time / iterations * 1000  # ms
        fps = iterations / total_time
        
        print(f"\n[Results]")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average latency: {avg_time:.2f}ms")
        print(f"  Average FPS: {fps:.2f}")
        
        return {
            "model": Path(model_path).name,
            "device": device,
            "backend": backend,
            "imgsz": imgsz,
            "iterations": iterations,
            "total_time": total_time,
            "avg_latency_ms": avg_time,
            "avg_fps": fps,
            "init_time": init_time,
        }
        
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_models(model_dir, extensions=[".onnx", ".pt", ".engine", ".xml"]):
    """Find all model files in the directory."""
    model_dir = Path(model_dir)
    models = []
    
    if not model_dir.exists():
        return models
    
    for ext in extensions:
        models.extend(model_dir.glob(f"*{ext}"))
    
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark face recognition models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a specific model file to benchmark",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/yolo",
        help="Directory containing models to benchmark (default: models/yolo)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: cuda:0, cuda:1, GPU, cpu (default: cpu)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnx",
        help="Backend to use: onnx, tensorrt, openvino, tflite (default: onnx)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations to run (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all models in the model directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare CPU vs GPU performance",
    )
    
    args = parser.parse_args()
    
    # Check environment variables
    env_device = os.environ.get("VMS_DEVICE", "")
    env_backend = os.environ.get("VMS_BACKEND", "")
    
    if env_device:
        args.device = env_device
    if env_backend:
        args.backend = env_backend
    
    print("FACE-RECOGNITION-UBUNTO BENCHMARK")
    print(f"Device: {args.device}, Backend: {args.backend}")
    print()
    
    # Get model path
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    
    results = []
    
    if args.all:
        # Benchmark all models
        models = find_models(model_dir)
        if not models:
            print(f"No models found in {model_dir}")
            return
        
        print(f"Found {len(models)} models to benchmark")
        for model_path in models:
            result = benchmark_model(
                model_path,
                device=args.device,
                backend=args.backend,
                iterations=args.iterations,
                imgsz=args.imgsz,
            )
            if result:
                results.append(result)
    elif args.compare:
        # Compare CPU vs GPU
        models = find_models(model_dir)
        if not models:
            models = [model_dir / "yolo11n.onnx"]
        
        for model_path in models[:1]:  # Just test one model
            print("\n" + "="*60)
            print("CPU BENCHMARK")
            print("="*60)
            cpu_result = benchmark_model(
                model_path,
                device="cpu",
                backend="onnx",
                iterations=args.iterations,
                imgsz=args.imgsz,
            )
            
            print("\n" + "="*60)
            print("GPU BENCHMARK")
            print("="*60)
            gpu_result = benchmark_model(
                model_path,
                device="cuda:0",
                backend="onnx",
                iterations=args.iterations,
                imgsz=args.imgsz,
            )
            
            if cpu_result and gpu_result:
                speedup = gpu_result["avg_fps"] / cpu_result["avg_fps"]
                print(f"\n[COMPARISON]")
                print(f"  GPU Speedup: {speedup:.2f}x faster")
                print(f"  CPU: {cpu_result['avg_fps']:.2f} FPS")
                print(f"  GPU: {gpu_result['avg_fps']:.2f} FPS")
            
            return
    else:
        # Benchmark specific model
        if args.model:
            model_path = Path(args.model)
            if not model_path.is_absolute():
                model_path = ROOT / model_path
            
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                return
            
            result = benchmark_model(
                model_path,
                device=args.device,
                backend=args.backend,
                iterations=args.iterations,
                imgsz=args.imgsz,
            )
            if result:
                results.append(result)
        else:
            # Default: benchmark yolo11n
            default_model = model_dir / "yolo11n.onnx"
            if default_model.exists():
                result = benchmark_model(
                    default_model,
                    device=args.device,
                    backend=args.backend,
                    iterations=args.iterations,
                    imgsz=args.imgsz,
                )
                if result:
                    results.append(result)
            else:
                print(f"Default model not found: {default_model}")
                return
    
    # Print summary
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'Backend':<12} {'Device':<10} {'FPS':>8} {'Latency':>10}")
        print("-"*60)
        for r in results:
            print(f"{r['model']:<20} {r['backend']:<12} {r['device']:<10} {r['avg_fps']:>8.2f} {r['avg_latency_ms']:>10.2f}ms")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
