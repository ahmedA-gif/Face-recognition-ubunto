#!/bin/bash

# =============================================================================
# FACE-RECOGNITION-UBUNTO: SMART MODEL DOWNLOADER
# =============================================================================
#
# This script intelligently downloads the appropriate AI models based on
# detected hardware (GPU/CPU/TPU) and framework support.
#
# Supported Hardware:
# - NVIDIA GPU: TensorRT + YOLOv11 (m/s/n/l) or YOLOv8
# - Intel iGPU: OpenVINO + YOLOv11 or YOLOv8
# - Coral TPU: TFLite + YOLOv11n or YOLOv8n
# - AMD GPU: ROCm + YOLOv11 or ONNX + YOLOv11
# - CPU: ONNX + YOLOv11n or YOLOv8n
#
# Usage:
#   chmod +x scripts/download_models.sh
#   ./scripts/download_models.sh [options]
#
# Options:
#   --all          Download all models for all hardware
#   --gpu          Force GPU mode (NVIDIA)
#   --cpu          Force CPU mode
#   --intel        Force Intel iGPU mode
#   --coral        Force Coral TPU mode
#   --yolo8        Use YOLOv8 instead of YOLOv11
#   --yolo9        Use YOLOv9
#   --yolo10       Use YOLOv10
#   --clean        Remove existing models before downloading
#   --help         Show this help
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_DIR="/Users/macbookpro/Desktop/Face-recognition-ubunto/models"
YOLO_DIR="$MODEL_DIR/yolo"
FACE_DIR="$MODEL_DIR/face"
MAX_RETRIES=3
TIMEOUT=60

# YOLO Model URLs (Official Ultralytics)
declare -A YOLOV8_URLS=(
    [yolov8n]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    [yolov8s]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt"
    [yolov8m]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    [yolov8l]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt"
    [yolov8x]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt"
)

declare -A YOLOV9_URLS=(
    [yolov9c]="https://github.com/ultralytics/assets/releases/download/v9.1.0/yolov9c.pt"
    [yolov9e]="https://github.com/ultralytics/assets/releases/download/v9.1.0/yolov9e.pt"
)

declare -A YOLOV10_URLS=(
    [yolov10n]="https://github.com/ultralytics/assets/releases/download/v10.0.0/yolov10n.pt"
    [yolov10s]="https://github.com/ultralytics/assets/releases/download/v10.0.0/yolov10s.pt"
    [yolov10m]="https://github.com/ultralytics/assets/releases/download/v10.0.0/yolov10m.pt"
)

declare -A YOLOV11_URLS=(
    [yolo11n]="https://github.com/ultralytics/assets/releases/download/v11.1/yolo11n.pt"
    [yolo11s]="https://github.com/ultralytics/assets/releases/download/v11.1/yolo11s.pt"
    [yolo11m]="https://github.com/ultralytics/assets/releases/download/v11.1/yolo11m.pt"
    [yolo11l]="https://github.com/ultralytics/assets/releases/download/v11.1/yolo11l.pt"
    [yolo11x]="https://github.com/ultralytics/assets/releases/download/v11.1/yolo11x.pt"
)

# Face Recognition Models (InsightFace)
BUFFALO_S_URL="https://github.com/naeheon/insightface_pytorch/releases/download/v0.1/buffalo_s.zip"
BUFFALO_L_URL="https://github.com/naeheon/insightface_pytorch/releases/download/v0.1/buffalo_l.zip"

# Default to YOLOv11
YOLO_VERSION="yolo11"
MODEL_PREFIX="yolo11"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create directory if not exists
create_dir() {
    mkdir -p "$1"
}

# Download with retry
download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"
    
    for ((i=1; i<=$MAX_RETRIES; i++)); do
        if command_exists wget; then
            if wget --quiet --timeout=$TIMEOUT --tries=1 --output-document="$dest" "$url"; then
                log_success "Downloaded $name"
                return 0
            fi
        elif command_exists curl; then
            if curl -sS --fail --max-time $TIMEOUT -o "$dest" "$url"; then
                log_success "Downloaded $name"
                return 0
            fi
        else
            log_error "Neither wget nor curl found. Please install one."
            return 1
        fi
        
        if [ $i -lt $MAX_RETRIES ]; then
            log_warning "Retry $i/$MAX_RETRIES for $name..."
            sleep 5
        fi
    done
    
    log_error "Failed to download $name after $MAX_RETRIES attempts"
    return 1
}

# Check GPU/TPU hardware
detect_hardware() {
    # Check NVIDIA GPU
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        echo "nvidia"
        return
    fi
    
    # Check Intel iGPU
    if command_exists intel_gpu_top && intel_gpu_top >/dev/null 2>&1; then
        echo "intel"
        return
    fi
    
    # Check Coral TPU
    if [ -e "/dev/apex_0" ] || lsusb | grep -q "Coral"; then
        echo "coral"
        return
    fi
    
    # Check AMD GPU (ROCm)
    if [ -e "/opt/rocm" ] || command_exists rocminfo; then
        echo "amd"
        return
    fi
    
    # Default to CPU
    echo "cpu"
}

# Get VRAM for NVIDIA GPU
get_nvidia_vram() {
    if command_exists nvidia-smi; then
        local vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | sed 's/[^0-9]//g')
        echo "$vram"
    else
        echo "0"
    fi
}

# Select YOLO model based on hardware and VRAM
select_yolo_model() {
    local hardware="$1"
    local vram="$2"
    
    case "$hardware" in
        nvidia)
            if [ "$vram" -ge 16000 ]; then
                echo "yolo11l"
            elif [ "$vram" -ge 8000 ]; then
                echo "yolo11m"
            elif [ "$vram" -ge 4000 ]; then
                echo "yolo11s"
            else
                echo "yolo11n"
            fi
            ;;
        intel)
            echo "yolo11s"
            ;;
        coral)
            echo "yolo11n"
            ;;
        amd)
            echo "yolo11s"
            ;;
        cpu)
            # Check CPU capabilities
            if uname -m | grep -q "x86_64"; then
                # Check for AVX2
                if grep -q "avx2" /proc/cpuinfo 2>/dev/null || sysctl -a 2>/dev/null | grep -q "avx2"; then
                    echo "yolo11n"
                else
                    echo "yolo8n"
                fi
            else
                echo "yolo8n"
            fi
            ;;
        *)
            echo "yolo11n"
            ;;
    esac
}

# Convert PyTorch to ONNX
convert_to_onnx() {
    local pt_path="$1"
    local onnx_path="$2"
    
    if [ ! -f "$pt_path" ]; then
        return 1
    fi
    
    log_info "Converting $pt_path to ONNX..."
    python3 -c "
from ultralytics import YOLO
import os
model = YOLO('$pt_path')
model.export(format='onnx', imgsz=416)
if os.path.exists('$onnx_path'):
    os.remove('$pt_path')
    print('Converted and removed PT file')
" 2>/dev/null && \
    log_success "Converted to ONNX: $onnx_path"
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================

# Parse arguments
FORCE_MODE=""
CLEAN=false
YOLOV="yolo11"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            FORCE_MODE="all"
            shift
            ;;
        --gpu|--nvidia)
            FORCE_MODE="nvidia"
            shift
            ;;
        --cpu)
            FORCE_MODE="cpu"
            shift
            ;;
        --intel)
            FORCE_MODE="intel"
            shift
            ;;
        --coral)
            FORCE_MODE="coral"
            shift
            ;;
        --amd)
            FORCE_MODE="amd"
            shift
            ;;
        --yolo8)
            YOLOV="yolo8"
            MODEL_PREFIX="yolov8"
            shift
            ;;
        --yolo9)
            YOLOV="yolo9"
            MODEL_PREFIX="yolov9"
            shift
            ;;
        --yolo10)
            YOLOV="yolo10"
            MODEL_PREFIX="yolov10"
            shift
            ;;
        --yolo11)
            YOLOV="yolo11"
            MODEL_PREFIX="yolo11"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --all          Download all models for all hardware"
            echo "  --gpu          Force NVIDIA GPU mode"
            echo "  --cpu          Force CPU mode"
            echo "  --intel        Force Intel iGPU mode"
            echo "  --coral        Force Coral TPU mode"
            echo "  --amd          Force AMD GPU mode"
            echo "  --yolo8        Use YOLOv8 models"
            echo "  --yolo9        Use YOLOv9 models"
            echo "  --yolo10       Use YOLOv10 models"
            echo "  --yolo11       Use YOLOv11 models (default)"
            echo "  --clean        Remove existing models before downloading"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
log_info "Creating model directories..."
create_dir "$MODEL_DIR"
create_dir "$YOLO_DIR"
create_dir "$FACE_DIR"

# Clean if requested
if [ "$CLEAN" = true ]; then
    log_info "Cleaning existing models..."
    rm -rf "$YOLO_DIR"/*
    rm -rf "$FACE_DIR"/*
fi

# =============================================================================
# DETECT HARDWARE OR USE FORCE MODE
# =============================================================================

if [ -n "$FORCE_MODE" ] && [ "$FORCE_MODE" != "all" ]; then
    HARDWARE="$FORCE_MODE"
    log_info "Forced hardware mode: $HARDWARE"
else
    HARDWARE=$(detect_hardware)
    log_info "Detected hardware: $HARDWARE"
fi

# Get VRAM for NVIDIA
VRAM=0
if [ "$HARDWARE" = "nvidia" ] || [ "$FORCE_MODE" = "all" ]; then
    VRAM=$(get_nvidia_vram)
    log_info "NVIDIA VRAM: ${VRAM}MB"
fi

# =============================================================================
# SELECT AND DOWNLOAD YOLO MODELS
# =============================================================================

log_info ""
log_info "=========================================="
log_info "DOWNLOADING YOLO DETECTION MODELS"
log_info "=========================================="

# Select models based on hardware
if [ "$FORCE_MODE" = "all" ]; then
    # Download all models
    log_info "Downloading ALL YOLOv${YOLOV} models..."
    
    # Download all sizes for the selected YOLO version
    case "$YOLOV" in
        yolo8)
            MODELS=("${!YOLOV8_URLS[@]}")
            ;;
        yolo9)
            MODELS=("${!YOLOV9_URLS[@]}")
            ;;
        yolo10)
            MODELS=("${!YOLOV10_URLS[@]}")
            ;;
        *)
            MODELS=("${!YOLOV11_URLS[@]}")
            ;;
    esac
    
    for model in "${MODELS[@]}"; do
        # Get URL
        declare -n urls="YOLOV${YOLOV^^}_URLS"
        url="${urls[$model]}"
        dest="$YOLO_DIR/${model}.pt"
        
        if [ -f "$dest" ]; then
            log_info "Already exists: $dest"
        else
            download_file "$url" "$dest" "${model}.pt"
        fi
        
        # Convert to ONNX
        onnx_dest="$YOLO_DIR/${model}.onnx"
        if [ ! -f "$onnx_dest" ]; then
            convert_to_onnx "$dest" "$onnx_dest"
        fi
    done
    
    # For NVIDIA, also download TensorRT engines (placeholder)
    if command_exists trtexec || [ -n "$FORCE_MODE" ]; then
        log_info ""
        log_info "TensorRT engines will be built from ONNX files..."
        log_info "Run: python3 scripts/build_tensorrt.py"
    fi
    
elif [ "$HARDWARE" = "nvidia" ]; then
    # NVIDIA GPU - Download based on VRAM
    MODEL=$(select_yolo_model "nvidia" "$VRAM")
    log_info "Selected model for NVIDIA (${VRAM}MB VRAM): $MODEL"
    
    # Download PyTorch model
    url="${YOLOV11_URLS[$MODEL]}"
    pt_dest="$YOLO_DIR/${MODEL}.pt"
    download_file "$url" "$pt_dest" "${MODEL}.pt"
    
    # Convert to ONNX (for CPU fallback)
    onnx_dest="$YOLO_DIR/${MODEL}.onnx"
    convert_to_onnx "$pt_dest" "$onnx_dest"
    
    # Convert to TensorRT
    log_info "Building TensorRT engine for $MODEL..."
    python3 -c "
import tensorrt as trt
from ultralytics import YOLO
import os

# Load model
model = YOLO('$pt_dest')

# Export to TensorRT
try:
    model.export(format='engine', device=0, imgsz=416, half=True)
    engine_path = '$YOLO_DIR/${MODEL}_engine_fp16.tensorrt'
    if os.path.exists(engine_path):
        print(f'TensorRT engine saved: {engine_path}')
    else:
        # Try manual conversion
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='$pt_dest')
        model.to('cuda').eval()
        print('Manual conversion may be needed')
except Exception as e:
    print(f'TensorRT conversion failed: {e}')
    print('You may need to build TensorRT engines manually')
" 2>&1 || log_warning "TensorRT conversion requires CUDA and TensorRT installed"
    
    # Also download a lighter model for fallback
    if [ "$MODEL" != "yolo11n" ]; then
        log_info "Downloading fallback model: yolo11n"
        url="${YOLOV11_URLS[yolo11n]}"
        pt_dest="$YOLO_DIR/yolo11n.pt"
        download_file "$url" "$pt_dest" "yolo11n.pt"
        onnx_dest="$YOLO_DIR/yolo11n.onnx"
        convert_to_onnx "$pt_dest" "$onnx_dest"
    fi
    
elif [ "$HARDWARE" = "intel" ]; then
    # Intel iGPU - Use OpenVINO
    MODEL="yolo11s"
    log_info "Selected model for Intel iGPU: $MODEL (OpenVINO)"
    
    # Download PyTorch model
    url="${YOLOV11_URLS[$MODEL]}"
    pt_dest="$YOLO_DIR/${MODEL}.pt"
    download_file "$url" "$pt_dest" "${MODEL}.pt"
    
    # Convert to ONNX
    onnx_dest="$YOLO_DIR/${MODEL}.onnx"
    convert_to_onnx "$pt_dest" "$onnx_dest"
    
    # Convert to OpenVINO
    log_info "Converting to OpenVINO format..."
    python3 -c "
from ultralytics import YOLO
import os

model = YOLO('$pt_dest')
try:
    model.export(format='openvino', imgsz=416)
    xml_path = '$YOLO_DIR/${MODEL}.xml'
    bin_path = '$YOLO_DIR/${MODEL}.bin'
    if os.path.exists(xml_path) and os.path.exists(bin_path):
        print(f'OpenVINO model saved: {xml_path} and {bin_path}')
    else:
        print('OpenVINO conversion may need OpenVINO toolkit installed')
except Exception as e:
    print(f'OpenVINO conversion failed: {e}')
    print('Install OpenVINO: pip install openvino')
" 2>&1 || log_warning "OpenVINO conversion requires OpenVINO toolkit"
    
    # Also download yolo11n for lighter option
    log_info "Downloading fallback model: yolo11n"
    url="${YOLOV11_URLS[yolo11n]}"
    pt_dest="$YOLO_DIR/yolo11n.pt"
    download_file "$url" "$pt_dest" "yolo11n.pt"
    onnx_dest="$YOLO_DIR/yolo11n.onnx"
    convert_to_onnx "$pt_dest" "$onnx_dest"
    
elif [ "$HARDWARE" = "coral" ]; then
    # Coral TPU - Use TFLite
    MODEL="yolo11n"
    log_info "Selected model for Coral TPU: $MODEL (TFLite INT8)"
    
    # Download PyTorch model
    url="${YOLOV11_URLS[$MODEL]}"
    pt_dest="$YOLO_DIR/${MODEL}.pt"
    download_file "$url" "$pt_dest" "${MODEL}.pt"
    
    # Convert to TFLite
    log_info "Converting to TensorFlow Lite INT8..."
    python3 -c "
from ultralytics import YOLO
import os

model = YOLO('$pt_dest')
try:
    model.export(format='tflite', imgsz=416, int8=True)
    tflite_path = '$YOLO_DIR/${MODEL}_int8.tflite'
    if os.path.exists(tflite_path):
        print(f'TFLite INT8 model saved: {tflite_path}')
    else:
        print('TFLite conversion may need additional setup')
except Exception as e:
    print(f'TFLite conversion failed: {e}')
    print('Install: pip install tensorflow')
" 2>&1 || log_warning "TFLite conversion requires TensorFlow"
    
elif [ "$HARDWARE" = "amd" ]; then
    # AMD GPU - Try ROCm, fallback to ONNX
    MODEL="yolo11s"
    log_info "Selected model for AMD GPU: $MODEL (ROCm or ONNX)"
    
    # Download PyTorch model
    url="${YOLOV11_URLS[$MODEL]}"
    pt_dest="$YOLO_DIR/${MODEL}.pt"
    download_file "$url" "$pt_dest" "${MODEL}.pt"
    
    # Convert to ONNX
    onnx_dest="$YOLO_DIR/${MODEL}.onnx"
    convert_to_onnx "$pt_dest" "$onnx_dest"
    
    # Try ROCm conversion
    log_info "Attempting ROCm conversion..."
    python3 -c "
import torch
if torch.cuda.is_available() and 'rocm' in torch.version.hip:
    from ultralytics import YOLO
    model = YOLO('$pt_dest')
    model.to('cuda')
    model.export(format='torchscript', device=0, imgsz=416)
    print('ROCm model converted')
else:
    print('ROCm not available, using ONNX')
" 2>&1 || log_warning "ROCm may require additional setup"
    
    # Also download yolo11n
    log_info "Downloading fallback model: yolo11n"
    url="${YOLOV11_URLS[yolo11n]}"
    pt_dest="$YOLO_DIR/yolo11n.pt"
    download_file "$url" "$pt_dest" "yolo11n.pt"
    onnx_dest="$YOLO_DIR/yolo11n.onnx"
    convert_to_onnx "$pt_dest" "$onnx_dest"
    
else # CPU
    # CPU - Use ONNX
    log_info "Selected model for CPU: Checking capabilities..."
    
    # Check for AVX2
    HAS_AVX2=false
    if uname -m | grep -q "x86_64"; then
        if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
            HAS_AVX2=true
        elif sysctl -a 2>/dev/null | grep -q "avx2"; then
            HAS_AVX2=true
        fi
    fi
    
    if [ "$HAS_AVX2" = true ]; then
        MODEL="yolo11n"
        log_info "CPU with AVX2: Using $MODEL"
    else
        MODEL="yolo8n"
        log_info "CPU without AVX2: Using $MODEL (lighter)"
    fi
    
    # Download model
    declare -n urls="YOLOV${YOLOV^^}_URLS"
    url="${urls[$MODEL]}"
    pt_dest="$YOLO_DIR/${MODEL}.pt"
    download_file "$url" "$pt_dest" "${MODEL}.pt"
    
    # Convert to ONNX
    onnx_dest="$YOLO_DIR/${MODEL}.onnx"
    convert_to_onnx "$pt_dest" "$onnx_dest"
    
    # Also download a fallback model
    if [ "$MODEL" = "yolo11n" ]; then
        FALLBACK="yolo8n"
    else
        FALLBACK="yolo8n"
    fi
    
    if [ "$FALLBACK" != "$MODEL" ]; then
        declare -n fallback_urls="YOLOV8_URLS"
        url="${fallback_urls[$FALLBACK]}"
        pt_dest="$YOLO_DIR/${FALLBACK}.pt"
        download_file "$url" "$pt_dest" "${FALLBACK}.pt"
        onnx_dest="$YOLO_DIR/${FALLBACK}.onnx"
        convert_to_onnx "$pt_dest" "$onnx_dest"
    fi
fi

# =============================================================================
# DOWNLOAD FACE RECOGNITION MODELS
# =============================================================================

log_info ""
log_info "=========================================="
log_info "DOWNLOADING FACE RECOGNITION MODELS"
log_info "=========================================="

# Check if buffalo models already exist
if [ ! -d "$FACE_DIR/buffalo_s" ] || [ ! -d "$FACE_DIR/buffalo_l" ]; then
    log_info "Downloading buffalo_s (fast, recommended)..."
    download_file "$BUFFALO_S_URL" "$FACE_DIR/buffalo_s.zip" "buffalo_s.zip"
    
    log_info "Extracting buffalo_s..."
    cd "$FACE_DIR"
    unzip -q buffalo_s.zip -d buffalo_s
    rm buffalo_s.zip
    cd - >/dev/null
    log_success "buffalo_s installed"
    
    log_info "Downloading buffalo_l (accurate)..."
    download_file "$BUFFALO_L_URL" "$FACE_DIR/buffalo_l.zip" "buffalo_l.zip"
    
    log_info "Extracting buffalo_l..."
    cd "$FACE_DIR"
    unzip -q buffalo_l.zip -d buffalo_l
    rm buffalo_l.zip
    cd - >/dev/null
    log_success "buffalo_l installed"
else
    log_info "buffalo_s and buffalo_l already exist"
fi

# =============================================================================
# CREATE MODEL REGISTRY FOR HOT-SWAP
# =============================================================================

log_info ""
log_info "=========================================="
log_info "CREATING MODEL REGISTRY"
log_info "=========================================="

# Create metadata for each model
for model_file in "$YOLO_DIR"/*.onnx "$YOLO_DIR"/*.pt; do
    if [ -f "$model_file" ]; then
        model_name=$(basename "$model_file" | sed 's/\..*//')
        model_dir="$MODEL_DIR/${model_name}"
        
        create_dir "$model_dir"
        
        # Copy/Link model file
        extension="${model_file##*.}"
        if [ "$extension" = "pt" ]; then
            # Convert to ONNX if not exists
            onnx_file="$model_dir/${model_name}.onnx"
            if [ ! -f "$onnx_file" ]; then
                convert_to_onnx "$model_file" "$onnx_file"
            fi
            cp "$model_file" "$model_dir/${model_name}.pt"
        else
            cp "$model_file" "$model_dir/${model_name}.onnx"
        fi
        
        # Create metadata.json
        cat > "$model_dir/metadata.json" << EOF
{
  "model_name": "$model_name",
  "type": "detection",
  "framework": "$(echo $model_file | grep -o 'onnx\|pt\|engine' | head -1)",
  "input_size": 416,
  "classes": 80,
  "description": "YOLO${YOLOV^^} ${model_name^}",
  "hardware": [
    $(if [ "$HARDWARE" = "nvidia" ] || [ "$FORCE_MODE" = "all" ]; then echo '"gpu"'; fi)
    $(if [ "$HARDWARE" = "cpu" ] || [ "$FORCE_MODE" = "all" ]; then echo '"cpu"'; fi)
    $(if [ "$HARDWARE" = "intel" ] || [ "$FORCE_MODE" = "all" ]; then echo '"intel"'; fi)
    $(if [ "$HARDWARE" = "coral" ] || [ "$FORCE_MODE" = "all" ]; then echo '"coral"'; fi)
    $(if [ "$HARDWARE" = "amd" ] || [ "$FORCE_MODE" = "all" ]; then echo '"amd"'; fi)
  ],
  "fps": {
    "480p": 0,
    "1080p": 0
  },
  "download_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
        
        log_info "Registry: $model_name"
    fi
done

# =============================================================================
# UPDATE SETTINGS.YAML
# =============================================================================

log_info ""
log_info "=========================================="
log_info "UPDATING CONFIGURATION"
log_info "=========================================="

# Determine best model for detected hardware
if [ "$HARDWARE" = "nvidia" ]; then
    if [ "$VRAM" -ge 16000 ]; then
        BEST_MODEL="yolo11l"
    elif [ "$VRAM" -ge 8000 ]; then
        BEST_MODEL="yolo11m"
    elif [ "$VRAM" -ge 4000 ]; then
        BEST_MODEL="yolo11s"
    else
        BEST_MODEL="yolo11n"
    fi
elif [ "$HARDWARE" = "intel" ]; then
    BEST_MODEL="yolo11s"
elif [ "$HARDWARE" = "coral" ]; then
    BEST_MODEL="yolo11n"
elif [ "$HARDWARE" = "amd" ]; then
    BEST_MODEL="yolo11s"
else # CPU
    BEST_MODEL="yolo11n"
fi

# Update settings.yaml if it exists
SETTINGS_FILE="$ROOT/config/settings.yaml"
if [ -f "$SETTINGS_FILE" ]; then
    log_info "Updating settings.yaml with detected hardware settings..."
    
    # Use Python to update YAML properly
    python3 -c "
import yaml
import os

settings_path = '$SETTINGS_FILE'

with open(settings_path, 'r') as f:
    settings = yaml.safe_load(f)

# Update models section
if 'models' not in settings:
    settings['models'] = {}

settings['models']['yolo_weights'] = 'models/yolo/${BEST_MODEL}.pt'
settings['models']['yolo_onnx'] = 'models/yolo/${BEST_MODEL}.onnx'
settings['models']['yolo_imgsz'] = 416

# Update pipeline device
if 'pipeline' not in settings:
    settings['pipeline'] = {}

if '$HARDWARE' == 'nvidia':
    settings['pipeline']['device'] = 'cuda:0'
else:
    settings['pipeline']['device'] = 'cpu'

# Save
with open(settings_path, 'w') as f:
    yaml.dump(settings, f, sort_keys=False, default_flow_style=False)

print(f'Updated settings.yaml with model: {BEST_MODEL}')
print(f'Device: {settings[\"pipeline\"].get(\"device\", \"cpu\")}')
"
else
    log_warning "settings.yaml not found, skipping configuration update"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================

log_info ""
log_info "=========================================="
log_info "DOWNLOAD COMPLETE"
log_info "=========================================="

ls -lh "$YOLO_DIR"
ls -lh "$FACE_DIR"

echo ""
log_success "✅ Models downloaded successfully!"
echo ""
echo "Hardware: $HARDWARE"
echo "Primary Model: $BEST_MODEL"
echo "Face Models: buffalo_s, buffalo_l"
echo ""
echo "Next steps:"
echo "  1. cd /Users/macbookpro/Desktop/Face-recognition-ubunto"
echo "  2. python3 main.py --source 'your_camera_source' --display true"
echo ""
echo "For TensorRT/ROCm/OpenVINO optimizations:"
echo "  - NVIDIA: Run scripts/build_tensorrt.py"
echo "  - Intel: pip install openvino"
echo "  - AMD: Install ROCm drivers"
echo "  - Coral: Already optimized for TFLite"
echo ""
