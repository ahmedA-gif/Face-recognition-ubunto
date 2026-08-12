#!/bin/bash
# Hardware Detection Script for FACE-RECOGNITION-UBUNTO
# Auto-detects GPU/CPU/TPU and selects optimal model/framework
# Runs at system startup and on hardware changes

set -e

LOG_FILE="/var/log/vms/hardware_detect.log"
CONFIG_DIR="/opt/vms/configs"
MODELS_DIR="/opt/vms/models"

# Ensure directories exist
mkdir -p "$CONFIG_DIR" "$MODELS_DIR" /var/log/vms

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting hardware detection..."

# Detect hardware and set environment variables
detect_hardware() {
    local hardware="cpu"
    local framework="onnx"
    local model="yolo11n"
    local fps_480p="10"
    local fps_1080p="4"
    
    # 1. Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected"
        hardware="nvidia"
        framework="tensorrt"
        
        # Get VRAM
        local vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | sed 's/[^0-9.]//g')
        vram=${vram:-0}
        
        log "VRAM: ${vram}GB"
        
        if (( $(echo "$vram >= 8" | bc -l) )); then
            model="yolo11l"
            fps_480p="60"
            fps_1080p="30"
        elif (( $(echo "$vram >= 4" | bc -l) )); then
            model="yolo11m"
            fps_480p="50"
            fps_1080p="25"
        else
            model="yolo11s"
            fps_480p="40"
            fps_1080p="15"
        fi
        
        # Check if TensorRT is available
        if ! python3 -c "import tensorrt" 2>/dev/null; then
            log "TensorRT not found, falling back to ONNX"
            framework="onnx"
        fi
        
    # 2. Check for Intel iGPU
    elif command -v intel_gpu_top &> /dev/null; then
        log "Intel iGPU detected"
        hardware="intel"
        framework="openvino"
        model="yolo11s"
        fps_480p="30"
        fps_1080p="10"
        
        # Check if OpenVINO is available
        if ! python3 -c "from openvino.runtime import Core" 2>/dev/null; then
            log "OpenVINO not found, falling back to ONNX"
            framework="onnx"
            model="yolo11n"
            fps_480p="10"
            fps_1080p="4"
        fi
        
    # 3. Check for Coral TPU
    elif [ -e "/dev/apex_0" ]; then
        log "Coral TPU detected"
        hardware="coral"
        framework="tflite"
        model="yolo11n"
        fps_480p="20"
        fps_1080p="8"
        
        # Check if TFLite is available
        if ! python3 -c "import tflite_runtime" 2>/dev/null; then
            log "TFLite not found, falling back to ONNX"
            framework="onnx"
            model="yolo11n"
            fps_480p="10"
            fps_1080p="4"
        fi
        
    # 4. Check for AMD GPU (ROCm)
    elif [ -f "/opt/rocm/bin/rocminfo" ]; then
        log "AMD GPU detected"
        hardware="amd"
        framework="onnx"
        model="yolo11s"
        fps_480p="15"
        fps_1080p="5"
        
        # Try ROCm
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
            framework="rocm"
        fi
        
    # 5. CPU Only
    else
        log "CPU only detected"
        hardware="cpu"
        framework="onnx"
        model="yolo11n"
        fps_480p="10"
        fps_1080p="4"
        
        # Check CPU features
        if grep -q "avx2" /proc/cpuinfo 2>/dev/null; then
            log "AVX2 supported"
        elif grep -q "avx512" /proc/cpuinfo 2>/dev/null; then
            log "AVX-512 supported"
            # Can use larger model
            model="yolo11s"
        else
            log "No AVX, using fallback model"
            model="yolo8n"
        fi
    fi
    
    # Set environment variables
    export VMS_HARDWARE="$hardware"
    export VMS_FRAMEWORK="$framework"
    export VMS_MODEL="$model"
    export VMS_FPS_480P="$fps_480p"
    export VMS_FPS_1080P="$fps_1080p"
    
    # Generate hardware config
    cat > "$CONFIG_DIR/hardware.yaml" <<EOF
hardware:
  type: $hardware
  framework: $framework
  model: $model
  performance:
    fps_480p: $fps_480p
    fps_1080p: $fps_1080p
  detection:
    backend: $framework
    model_path: $MODELS_DIR/$model
EOF
    
    log "Hardware detection complete:"
    log "  Hardware: $hardware"
    log "  Framework: $framework"
    log "  Model: $model"
    log "  FPS (480p): $fps_480p"
    log "  FPS (1080p): $fps_1080p"
}

# Main execution
detect_hardware

log "Hardware detection script completed successfully."
exit 0
