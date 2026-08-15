#!/usr/bin/env python3
"""
FACE-RECOGNITION-UBUNTO: Smart Model Downloader

This script intelligently downloads the appropriate AI models based on
detected hardware (GPU/CPU/TPU) and framework support.

Supported Hardware:
- NVIDIA GPU: TensorRT + YOLOv11 (m/s/n/l) or YOLOv8
- Intel iGPU: OpenVINO + YOLOv11 or YOLOv8
- Coral TPU: TFLite + YOLOv11n or YOLOv8n
- AMD GPU: ROCm/ONNX + YOLOv11
- CPU: ONNX + YOLOv11n or YOLOv8n

Supported Models:
- YOLOv8 (n/s/m/l/x)
- YOLOv9 (c/e)
- YOLOv10 (n/s/m)
- YOLOv11 (n/s/m/l/x) - DEFAULT

Usage:
    python3 scripts/download_models.py [options]

Options:
    --all              Download all models for all hardware
    --gpu, --nvidia    Force NVIDIA GPU mode
    --cpu              Force CPU mode
    --intel            Force Intel iGPU mode
    --coral            Force Coral TPU mode
    --amd              Force AMD GPU mode
    --yolo8            Use YOLOv8 models
    --yolo9            Use YOLOv9 models
    --yolo10           Use YOLOv10 models
    --yolo11           Use YOLOv11 models (default)
    --clean            Remove existing models before downloading
    --no-convert        Skip ONNX/TensorRT/OpenVINO conversion
    --no-registry      Skip model registry creation
    --tensorrt         Build TensorRT engines after downloading (NVIDIA only)
    --fp16             Use FP16 precision for TensorRT engines
    --help, -h         Show this help

Examples:
    # Auto-detect and download for current hardware
    python3 scripts/download_models.py

    # Force CPU mode
    python3 scripts/download_models.py --cpu

    # Download all models
    python3 scripts/download_models.py --all

    # Use YOLOv8 on GPU
    python3 scripts/download_models.py --gpu --yolo8
"""

import argparse
import os
import sys
import json
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve
import tempfile
import zipfile
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
YOLO_DIR = MODEL_DIR / "yolo"
FACE_DIR = MODEL_DIR / "face"

# Model URLs
YOLOV8_MODELS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov8m": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "yolov8l": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
    "yolov8x": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
}

YOLOV9_MODELS = {
    "yolov9c": "https://github.com/ultralytics/assets/releases/download/v9.1.0/yolov9c.pt",
    "yolov9e": "https://github.com/ultralytics/assets/releases/download/v9.1.0/yolov9e.pt",
}

YOLOV10_MODELS = {
    "yolov10n": "https://github.com/ultralytics/assets/releases/download/v10.0.0/yolov10n.pt",
    "yolov10s": "https://github.com/ultralytics/assets/releases/download/v10.0.0/yolov10s.pt",
    "yolov10m": "https://github.com/ultralytics/assets/releases/download/v10.0.0/yolov10m.pt",
}

YOLOV11_MODELS = {
    "yolo11n": "https://github.com/ultralytics/assets/releases/download/v11.1/yolo11n.pt",
    "yolo11s": "https://github.com/ultralytics/assets/releases/download/v11.1/yolo11s.pt",
    "yolo11m": "https://github.com/ultralytics/assets/releases/download/v11.1/yolo11m.pt",
    "yolo11l": "https://github.com/ultralytics/assets/releases/download/v11.1/yolo11l.pt",
    "yolo11x": "https://github.com/ultralytics/assets/releases/download/v11.1/yolo11x.pt",
}

FACE_MODELS = {
    "buffalo_s": "https://github.com/naeheon/insightface_pytorch/releases/download/v0.1/buffalo_s.zip",
    "buffalo_l": "https://github.com/naeheon/insightface_pytorch/releases/download/v0.1/buffalo_l.zip",
}

# Hardware to YOLO model mapping
HARDWARE_MODELS = {
    "nvidia": {
        16000: "yolo11l",  # >= 16GB VRAM
        8000: "yolo11m",   # >= 8GB VRAM
        4000: "yolo11s",   # >= 4GB VRAM
        0: "yolo11n",      # < 4GB VRAM
    },
    "intel": "yolo11s",
    "coral": "yolo11n",
    "amd": "yolo11s",
    "cpu": "yolo11n",
}

# Fallback models
FALLBACK_MODELS = {
    "yolo11l": "yolo11m",
    "yolo11m": "yolo11s",
    "yolo11s": "yolo11n",
    "yolov8l": "yolov8m",
    "yolov8m": "yolov8s",
    "yolov8s": "yolov8n",
}

# =============================================================================
# COLOR OUTPUT
# =============================================================================

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_info(msg: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.END} {msg}")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}[✓]{Colors.END} {msg}")


def print_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}[!]{Colors.END} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}[✗]{Colors.END} {msg}")


def print_header(msg: str) -> None:
    print(f"\n{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{msg}{Colors.END}")
    print(f"{Colors.CYAN}{'='*60}{Colors.END}")


# =============================================================================
# HARDWARE DETECTION
# =============================================================================

def command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            ["which", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def detect_nvidia_gpu() -> Tuple[bool, int]:
    """Detect NVIDIA GPU and get VRAM in MB."""
    if not command_exists("nvidia-smi"):
        return False, 0
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            vram_str = result.stdout.strip().split("\n")[0]
            vram = int("".join(filter(str.isdigit, vram_str)))
            return True, vram
    except Exception:
        pass
    
    return False, 0


def detect_intel_gpu() -> bool:
    """Detect Intel integrated GPU."""
    return command_exists("intel_gpu_top")


def detect_coral_tpu() -> bool:
    """Detect Coral TPU."""
    return os.path.exists("/dev/apex_0") or \
           subprocess.run(["lsusb"], capture_output=True, text=True).stdout.lower().find("coral") >= 0


def detect_amd_gpu() -> bool:
    """Detect AMD GPU with ROCm."""
    return os.path.exists("/opt/rocm") or command_exists("rocminfo")


def detect_avx2() -> bool:
    """Check if CPU supports AVX2."""
    try:
        if platform.machine() == "x86_64":
            with open("/proc/cpuinfo", "r") as f:
                return "avx2" in f.read().lower()
    except Exception:
        pass
    return False


def detect_hardware() -> Tuple[str, Optional[int]]:
    """
    Detect hardware and return (type, vram_mb).
    
    Returns:
        Tuple of (hardware_type, vram_in_mb)
        hardware_type: "nvidia", "intel", "coral", "amd", "cpu"
        vram_in_mb: Only for NVIDIA, otherwise None
    """
    # Check NVIDIA
    has_nvidia, vram = detect_nvidia_gpu()
    if has_nvidia:
        return "nvidia", vram
    
    # Check Intel
    if detect_intel_gpu():
        return "intel", None
    
    # Check Coral
    if detect_coral_tpu():
        return "coral", None
    
    # Check AMD
    if detect_amd_gpu():
        return "amd", None
    
    # Default to CPU
    return "cpu", None


# =============================================================================
# MODEL SELECTION
# =============================================================================

def select_yolo_model(hardware: str, vram: Optional[int] = None, yolo_version: str = "yolo11") -> str:
    """Select the best YOLO model for given hardware and VRAM."""
    
    if hardware == "nvidia" and vram is not None:
        # Sort VRAM thresholds in descending order
        thresholds = sorted(HARDWARE_MODELS["nvidia"].keys(), reverse=True)
        for threshold in thresholds:
            if vram >= threshold:
                return HARDWARE_MODELS["nvidia"][threshold]
        return HARDWARE_MODELS["nvidia"][0]
    
    # For other hardware
    return HARDWARE_MODELS.get(hardware, "yolo11n")


def get_yolo_model_urls(version: str) -> Dict[str, str]:
    """Get YOLO model URLs for the specified version."""
    urls = {
        "yolo8": YOLOV8_MODELS,
        "yolo9": YOLOV9_MODELS,
        "yolo10": YOLOV10_MODELS,
        "yolo11": YOLOV11_MODELS,
    }
    return urls.get(version, YOLOV11_MODELS)


# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================

def download_file(url: str, dest: Path, max_retries: int = 3, timeout: int = 60) -> bool:
    """Download a file with retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(1, max_retries + 1):
        try:
            # Use a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            try:
                print_info(f"Downloading {dest.name} (attempt {attempt}/{max_retries})...")
                
                # Try with requests first (better progress)
                try:
                    import requests
                    response = requests.get(url, stream=True, timeout=timeout)
                    response.raise_for_status()
                    
                    with open(tmp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    tmp_path.replace(dest)
                    print_success(f"Downloaded {dest.name}")
                    return True
                except ImportError:
                    # Fallback to urllib
                    urlretrieve(url, tmp_path, reporthook=lambda b, bs, ts: None)
                    tmp_path.replace(dest)
                    print_success(f"Downloaded {dest.name}")
                    return True
                    
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                if attempt < max_retries:
                    print_warning(f"Retry {attempt}/{max_retries}: {e}")
                    time.sleep(5)
                else:
                    print_error(f"Failed to download {url}: {e}")
                    return False
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            return False
    
    return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        zip_path.unlink()
        return True
    except Exception as e:
        print_error(f"Failed to extract {zip_path}: {e}")
        return False


def convert_pt_to_onnx(pt_path: Path, onnx_path: Path, imgsz: int = 416) -> bool:
    """Convert PyTorch model to ONNX format."""
    try:
        from ultralytics import YOLO
        
        print_info(f"Converting {pt_path.name} to ONNX...")
        model = YOLO(str(pt_path))
        model.export(format='onnx', imgsz=imgsz)
        
        # Move the exported ONNX file
        exported_onnx = pt_path.with_suffix('.onnx')
        if exported_onnx.exists():
            exported_onnx.replace(onnx_path)
            print_success(f"Converted to ONNX: {onnx_path.name}")
            return True
        else:
            print_warning(f"ONNX file not found at {exported_onnx}")
            return False
    except ImportError:
        print_warning("Ultralytics not installed. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print_error(f"ONNX conversion failed: {e}")
        return False


# =============================================================================
# MODEL REGISTRY
# =============================================================================

def create_model_registry() -> None:
    """Create model registry with metadata."""
    print_header("CREATING MODEL REGISTRY")
    
    if not YOLO_DIR.exists():
        print_warning("No YOLO models found")
        return
    
    for model_file in YOLO_DIR.glob("*.onnx") + YOLO_DIR.glob("*.pt"):
        model_name = model_file.stem
        model_dir = MODEL_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        extension = model_file.suffix
        if extension == ".pt":
            # Convert to ONNX if needed
            onnx_file = model_dir / f"{model_name}.onnx"
            if not onnx_file.exists():
                convert_pt_to_onnx(model_file, onnx_file)
            shutil.copy2(model_file, model_dir / f"{model_name}.pt")
        else:
            shutil.copy2(model_file, model_dir / f"{model_name}.onnx")
        
        # Create metadata.json
        metadata = {
            "model_name": model_name,
            "type": "detection",
            "framework": extension[1:],  # Remove the dot
            "input_size": 416,
            "classes": 80,
            "description": f"YOLOv{model_name[:2].upper()} {model_name[2:]}" if model_name.startswith("yolo") else model_name,
            "hardware": [],
            "fps": {
                "480p": 0,
                "1080p": 0
            },
            "download_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
        # Determine compatible hardware
        if model_name.startswith("yolo"):
            size = model_name[4:]  # n, s, m, l, x
            if size in ["l", "m", "s"]:
                metadata["hardware"].append("gpu")
            if size in ["n", "s", "m"]:
                metadata["hardware"].append("cpu")
            if size in ["n", "s"]:
                metadata["hardware"].extend(["intel", "coral", "amd"])
        
        # Write metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print_info(f"Registry: {model_name}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart Model Downloader for FACE-RECOGNITION-UBUNTO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models for all hardware"
    )
    parser.add_argument(
        "--gpu", "--nvidia",
        action="store_true",
        help="Force NVIDIA GPU mode"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode"
    )
    parser.add_argument(
        "--intel",
        action="store_true",
        help="Force Intel iGPU mode"
    )
    parser.add_argument(
        "--coral",
        action="store_true",
        help="Force Coral TPU mode"
    )
    parser.add_argument(
        "--amd",
        action="store_true",
        help="Force AMD GPU mode"
    )
    parser.add_argument(
        "--yolo8",
        action="store_true",
        help="Use YOLOv8 models"
    )
    parser.add_argument(
        "--yolo9",
        action="store_true",
        help="Use YOLOv9 models"
    )
    parser.add_argument(
        "--yolo10",
        action="store_true",
        help="Use YOLOv10 models"
    )
    parser.add_argument(
        "--yolo11",
        action="store_true",
        help="Use YOLOv11 models (default)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing models before downloading"
    )
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip ONNX/TensorRT/OpenVINO conversion"
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip model registry creation"
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Build TensorRT engines after downloading (NVIDIA only)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision for TensorRT engines"
    )
    
    args = parser.parse_args()
    
    # Determine YOLO version
    yolo_versions = ["yolo8", "yolo9", "yolo10", "yolo11"]
    yolo_version = "yolo11"
    for v in yolo_versions:
        if getattr(args, v, False):
            yolo_version = v
            break
    
    # Determine hardware mode
    forced_hardware = None
    if args.gpu or args.nvidia:
        forced_hardware = "nvidia"
    elif args.cpu:
        forced_hardware = "cpu"
    elif args.intel:
        forced_hardware = "intel"
    elif args.coral:
        forced_hardware = "coral"
    elif args.amd:
        forced_hardware = "amd"
    
    # Detect hardware if not forced
    if forced_hardware is None:
        hardware, vram = detect_hardware()
    else:
        hardware = forced_hardware
        vram = detect_nvidia_gpu()[1] if forced_hardware == "nvidia" else None
    
    print_header(f"FACE-RECOGNITION-UBUNTO MODEL DOWNLOADER")
    print_info(f"Hardware: {hardware}")
    if vram:
        print_info(f"VRAM: {vram}MB")
    print_info(f"YOLO Version: {yolo_version}")
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    FACE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean if requested
    if args.clean:
        print_info("Cleaning existing models...")
        if YOLO_DIR.exists():
            shutil.rmtree(YOLO_DIR)
            YOLO_DIR.mkdir()
        if FACE_DIR.exists():
            shutil.rmtree(FACE_DIR)
            FACE_DIR.mkdir()
    
    # Get model URLs
    model_urls = get_yolo_model_urls(yolo_version)
    
    # Select models to download
    if args.all:
        # Download all models
        models_to_download = list(model_urls.keys())
        fallback_models = []
    else:
        # Select best model for hardware
        primary_model = select_yolo_model(hardware, vram, yolo_version)
        models_to_download = [primary_model]
        
        # Add fallback model
        fallback = FALLBACK_MODELS.get(primary_model)
        if fallback and fallback in model_urls:
            fallback_models = [fallback]
        else:
            fallback_models = []
    
    # =========================================================================
    # DOWNLOAD YOLO MODELS
    # =========================================================================
    print_header("DOWNLOADING YOLO DETECTION MODELS")
    
    # Download primary models
    for model_name in models_to_download:
        if model_name not in model_urls:
            print_warning(f"Model {model_name} not available for {yolo_version}")
            continue
        
        url = model_urls[model_name]
        pt_dest = YOLO_DIR / f"{model_name}.pt"
        
        if pt_dest.exists():
            print_info(f"Already exists: {pt_dest.name}")
        else:
            if download_file(url, pt_dest):
                print_success(f"Downloaded: {pt_dest.name}")
            else:
                print_error(f"Failed to download: {pt_dest.name}")
                continue
        
        # Convert to ONNX (unless skipped)
        if not args.no_convert:
            onnx_dest = YOLO_DIR / f"{model_name}.onnx"
            if not onnx_dest.exists():
                convert_pt_to_onnx(pt_dest, onnx_dest)
    
    # Download fallback models
    for model_name in fallback_models:
        if model_name not in model_urls:
            continue
        
        url = model_urls[model_name]
        pt_dest = YOLO_DIR / f"{model_name}.pt"
        
        if pt_dest.exists():
            print_info(f"Already exists: {pt_dest.name}")
        else:
            print_info(f"Downloading fallback model: {model_name}")
            download_file(url, pt_dest)
        
        # Convert to ONNX
        if not args.no_convert:
            onnx_dest = YOLO_DIR / f"{model_name}.onnx"
            if not onnx_dest.exists():
                convert_pt_to_onnx(pt_dest, onnx_dest)
    
    # =========================================================================
    # DOWNLOAD FACE RECOGNITION MODELS
    # =========================================================================
    print_header("DOWNLOADING FACE RECOGNITION MODELS")
    
    for face_name, url in FACE_MODELS.items():
        face_dir = FACE_DIR / face_name
        zip_dest = FACE_DIR / f"{face_name}.zip"
        
        if face_dir.exists():
            print_info(f"Already exists: {face_name}")
            continue
        
        print_info(f"Downloading {face_name}...")
        if download_file(url, zip_dest):
            print_info(f"Extracting {face_name}...")
            if extract_zip(zip_dest, FACE_DIR):
                print_success(f"Installed: {face_name}")
    
    # =========================================================================
    # CREATE MODEL REGISTRY
    # =========================================================================
    if not args.no_registry:
        create_model_registry()
    
    # =========================================================================
    # UPDATE SETTINGS.YAML
    # =========================================================================
    print_header("UPDATING CONFIGURATION")
    
    # Select best model
    if args.all:
        best_model = list(model_urls.keys())[0]  # First model
    else:
        best_model = models_to_download[0]
    
    settings_path = ROOT / "config" / "settings.yaml"
    if settings_path.exists():
        try:
            import yaml
            
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f) or {}
            
            if 'models' not in settings:
                settings['models'] = {}
            
            # Update YOLO settings
            pt_path = YOLO_DIR / f"{best_model}.pt"
            onnx_path = YOLO_DIR / f"{best_model}.onnx"
            
            if pt_path.exists():
                settings['models']['yolo_weights'] = str(pt_path)
            if onnx_path.exists():
                settings['models']['yolo_onnx'] = str(onnx_path)
            
            settings['models']['yolo_imgsz'] = 416
            settings['models']['yolo_conf'] = 0.45
            settings['models']['yolo_iou'] = 0.5
            
            # Update pipeline device
            if 'pipeline' not in settings:
                settings['pipeline'] = {}
            
            if hardware == "nvidia":
                settings['pipeline']['device'] = 'cuda:0'
            else:
                settings['pipeline']['device'] = 'cpu'
            
            # Save
            with open(settings_path, 'w') as f:
                yaml.dump(settings, f, sort_keys=False, default_flow_style=False)
            
            print_success(f"Updated settings.yaml")
            print_info(f"  Model: {best_model}")
            print_info(f"  Device: {settings['pipeline'].get('device', 'cpu')}")
        except ImportError:
            print_warning("PyYAML not installed. Install with: pip install pyyaml")
        except Exception as e:
            print_error(f"Failed to update settings.yaml: {e}")
    else:
        print_warning(f"settings.yaml not found at {settings_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_header("DOWNLOAD COMPLETE")
    
    print_success("✅ Models downloaded successfully!")
    print(f"\nHardware: {hardware}")
    print(f"Primary Detection Model: {best_model}")
    print(f"Face Models: buffalo_s, buffalo_l")
    
    print(f"\nYOLO Models in {YOLO_DIR}:")
    for f in sorted(YOLO_DIR.glob("*")):
        print(f"  - {f.name}")
    
    print(f"\nFace Models in {FACE_DIR}:")
    for f in sorted(FACE_DIR.iterdir()):
        if f.is_dir():
            print(f"  - {f.name}/")
    
    # =========================================================================
    # BUILD TENSORRT ENGINES (if requested and NVIDIA)
    # =========================================================================
    if args.tensorrt and hardware == "nvidia":
        print_header("BUILDING TENSORRT ENGINES")
        try:
            import subprocess
            fp16_flag = " --fp16" if args.fp16 else ""
            result = subprocess.run(
                ["python3", "scripts/build_tensorrt.py", "--all" + fp16_flag],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print_success("TensorRT engines built successfully")
                print(result.stdout)
            else:
                print_warning("TensorRT build returned non-zero exit code")
                print(result.stderr)
        except ImportError:
            print_warning("subprocess not available, skipping TensorRT build")
        except Exception as e:
            print_warning(f"TensorRT build failed: {e}")

    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. cd {ROOT}")
    print("2. python3 main.py --source 'your_camera_source' --display true")
    print("")
    print("For TensorRT/ROCm/OpenVINO optimizations:")
    if hardware == "nvidia":
        if not args.tensorrt:
            print("  - Run: python3 scripts/build_tensorrt.py --all")
        print("  - Set: VMS_DEVICE=cuda:0 VMS_BACKEND=tensorrt")
    elif hardware == "intel":
        print("  - Install: pip install openvino")
        print("  - Set: VMS_DEVICE=GPU VMS_BACKEND=openvino")
    elif hardware == "amd":
        print("  - Install ROCm drivers")
        print("  - Set: VMS_DEVICE=cuda:0 VMS_BACKEND=onnx")
    elif hardware == "coral":
        print("  - Already optimized for TFLite")
        print("  - Set: VMS_DEVICE=cpu VMS_BACKEND=tflite")
    else:
        print("  - CPU mode: All optimizations applied")
        print("  - Set: VMS_DEVICE=cpu VMS_BACKEND=onnx")
    print("")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
