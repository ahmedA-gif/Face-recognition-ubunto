#!/bin/bash

# FACE-RECOGNITION-UBUNTO: GPU-FIRST SETUP SCRIPT
# This script automates the setup of GPU acceleration for the face recognition system
# It detects hardware, installs dependencies, and configures the system for optimal performance

set -e

echo "=========================================="
echo "FACE-RECOGNITION-UBUNTO GPU-FIRST SETUP"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
echo ""

# Function to detect OS
DetectOS() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
    elif [ -f /etc/lsb-release ]; then
        . /etc/lsb-release
        OS=$DISTRIB_ID
        VER=$DISTRIB_RELEASE
    else
        OS=$(uname -s)
        VER=$(uname -r)
    fi
    echo "Detected OS: $OS $VER"
}

# Function to detect hardware
detect_hardware() {
    echo "Detecting hardware..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | sed 's/[^0-9]//g')
        if [ -n "$VRAM" ]; then
            echo "Detected: NVIDIA GPU with ${VRAM}MB VRAM"
            echo "nvidia"
            return
        fi
    fi
    
    # Check for Intel iGPU
    if command -v intel_gpu_top &> /dev/null; then
        echo "Detected: Intel iGPU"
        echo "intel"
        return
    fi
    
    # Check for AMD GPU (ROCm)
    if [ -d /opt/rocm ] || command -v rocminfo &> /dev/null; then
        echo "Detected: AMD GPU (ROCm)"
        echo "amd"
        return
    fi
    
    # Check for Coral TPU
    if [ -e /dev/apex_0 ] || lsusb | grep -i coral &> /dev/null; then
        echo "Detected: Coral TPU"
        echo "coral"
        return
    fi
    
    echo "Detected: CPU only"
    echo "cpu"
}

# Function to install NVIDIA dependencies
install_nvidia() {
    echo ""
    echo "=========================================="
    echo "Installing NVIDIA GPU dependencies"
    echo "=========================================="
    
    # Check for CUDA
    if ! command -v nvcc &> /dev/null; then
        echo "Installing CUDA Toolkit..."
        sudo apt update
        sudo apt install -y nvidia-cuda-toolkit
    else
        echo "CUDA Toolkit already installed"
    fi
    
    # Check for TensorRT
    if ! dpkg -l | grep -q tensorrt; then
        echo "Installing TensorRT..."
        sudo apt install -y nvidia-tensorrt
    else
        echo "TensorRT already installed"
    fi
    
    # Install Python GPU packages
    echo "Installing Python GPU packages..."
    pip install --upgrade onnxruntime-gpu
    pip install --upgrade nvidia-tensorrt nvidia-cuda-runtime nvidia-cudnn
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install --upgrade pycuda
    
    echo "NVIDIA GPU dependencies installed successfully"
}

# Function to install Intel dependencies
install_intel() {
    echo ""
    echo "=========================================="
    echo "Installing Intel iGPU dependencies"
    echo "=========================================="
    
    # Install OpenCL drivers
    if ! dpkg -l | grep -q intel-opencl-icd; then
        echo "Installing Intel OpenCL drivers..."
        sudo apt update
        sudo apt install -y intel-opencl-icd intel-media-va-driver-non-free
    else
        echo "Intel OpenCL drivers already installed"
    fi
    
    # Install OpenVINO
    echo "Installing OpenVINO..."
    pip install --upgrade openvino openvino-dev
    
    echo "Intel iGPU dependencies installed successfully"
}

# Function to install AMD dependencies
install_amd() {
    echo ""
    echo "=========================================="
    echo "Installing AMD GPU dependencies"
    echo "=========================================="
    
    # Install ROCm
    if [ ! -d /opt/rocm ]; then
        echo "Installing ROCm..."
        sudo apt update
        sudo apt install -y rocm-opencl-runtime rocm-hip-sdk
        sudo usermod -aG video $USER
    else
        echo "ROCm already installed"
    fi
    
    # Install PyTorch for ROCm
    echo "Installing PyTorch for ROCm..."
    pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.6
    
    echo "AMD GPU dependencies installed successfully"
}

# Function to install Coral dependencies
install_coral() {
    echo ""
    echo "=========================================="
    echo "Installing Coral TPU dependencies"
    echo "=========================================="
    
    # Add Coral repository
    if [ ! -f /etc/apt/sources.list.d/coral-edgetpu-stable.list ]; then
        echo "Adding Coral repository..."
        echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu-stable.list
        curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
        sudo apt update
    fi
    
    # Install Coral runtime
    if ! dpkg -l | grep -q libedgetpu; then
        echo "Installing Coral runtime..."
        sudo apt install -y libedgetpu1-std python3-pycoral
    else
        echo "Coral runtime already installed"
    fi
    
    # Install Python packages
    echo "Installing Python packages for Coral..."
    pip install --upgrade tflite-runtime pycoral
    
    echo "Coral TPU dependencies installed successfully"
}

# Function to install CPU dependencies
install_cpu() {
    echo ""
    echo "=========================================="
    echo "Installing CPU dependencies"
    echo "=========================================="
    
    # Base requirements are already installed via requirements.txt
    echo "Installing base CPU dependencies..."
    pip install -r "$PROJECT_DIR/requirements.txt"
    
    echo "CPU dependencies installed successfully"
}

# Function to download models
download_models() {
    local hardware=$1
    echo ""
    echo "=========================================="
    echo "Downloading models for $hardware"
    echo "=========================================="
    
    cd "$PROJECT_DIR"
    
    case $hardware in
        "nvidia")
            python3 scripts/download_models.py --gpu
            ;;
        "intel")
            python3 scripts/download_models.py --intel
            ;;
        "amd")
            python3 scripts/download_models.py --amd
            ;;
        "coral")
            python3 scripts/download_models.py --coral
            ;;
        "cpu")
            python3 scripts/download_models.py --cpu
            ;;
    esac
    
    echo "Models downloaded successfully"
}

# Function to build TensorRT engines
build_tensorrt() {
    local hardware=$1
    
    if [ "$hardware" = "nvidia" ]; then
        echo ""
        echo "=========================================="
        echo "Building TensorRT engines"
        echo "=========================================="
        
        cd "$PROJECT_DIR"
        python3 scripts/build_tensorrt.py --all --fp16
        
        echo "TensorRT engines built successfully"
    fi
}

# Function to update configuration
update_config() {
    local hardware=$1
    local device=""
    local backend=""
    
    case $hardware in
        "nvidia")
            device="cuda:0"
            backend="tensorrt"
            ;;
        "intel")
            device="GPU"
            backend="openvino"
            ;;
        "amd")
            device="cuda:0"
            backend="onnx"
            ;;
        "coral")
            device="cpu"
            backend="tflite"
            ;;
        "cpu")
            device="cpu"
            backend="onnx"
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "Updating configuration for $hardware"
    echo "=========================================="
    
    cd "$PROJECT_DIR"
    
    # Update settings.yaml
    if [ -f config/settings.yaml ]; then
        # Backup original
        cp config/settings.yaml config/settings.yaml.bak
        
        # Update device and backend
        if command -v yq &> /dev/null; then
            yq e -i '.pipeline.device = "'"$device"'"' config/settings.yaml
            yq e -i '.pipeline.backend = "'"$backend"'"' config/settings.yaml
        else
            # Use sed for simple replacement
            sed -i "s/device:.*/device: \"$device\"/g" config/settings.yaml
            sed -i "s/backend:.*/backend: \"$backend\"/g" config/settings.yaml
        fi
        
        echo "Configuration updated:"
        echo "  Device: $device"
        echo "  Backend: $backend"
    else
        echo "settings.yaml not found. Using defaults."
    fi
    
    echo "Configuration updated successfully"
}

# Main execution
main() {
    # Detect OS
    DetectOS
    echo ""
    
    # Detect hardware
    HARDWARE=$(detect_hardware)
    echo ""
    
    # Install base requirements
    echo "Installing base requirements..."
    cd "$PROJECT_DIR"
    pip install -r requirements.txt
    echo ""
    
    # Install hardware-specific dependencies
    case $HARDWARE in
        "nvidia")
            install_nvidia
            ;;
        "intel")
            install_intel
            ;;
        "amd")
            install_amd
            ;;
        "coral")
            install_coral
            ;;
        "cpu")
            install_cpu
            ;;
    esac
    
    # Download models
    download_models $HARDWARE
    
    # Build TensorRT engines (NVIDIA only)
    build_tensorrt $HARDWARE
    
    # Update configuration
    update_config $HARDWARE
    
    # Final verification
    echo ""
    echo "=========================================="
    echo "VERIFICATION"
    echo "=========================================="
    
    case $HARDWARE in
        "nvidia")
            echo "Checking NVIDIA GPU..."
            nvidia-smi
            echo ""
            echo "Checking CUDA..."
            nvcc --version
            echo ""
            echo "Checking TensorRT..."
            dpkg -l | grep tensorrt
            ;;
        "intel")
            echo "Checking Intel GPU..."
            intel_gpu_top
            ;;
        "amd")
            echo "Checking AMD GPU..."
            rocm-smi
            ;;
        "coral")
            echo "Checking Coral TPU..."
            ls /dev/apex_0
            ;;
    esac
    
    echo ""
    echo "Checking Python packages..."
    python3 -c "import onnxruntime; print('ONNX Runtime:', onnxruntime.__version__)"
    
    case $HARDWARE in
        "nvidia")
            python3 -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())" 2>/dev/null || true
            python3 -c "import tensorrt; print('TensorRT:', tensorrt.__version__)" 2>/dev/null || true
            ;;
        "intel")
            python3 -c "from openvino.runtime import Core; print('OpenVINO available')" 2>/dev/null || true
            ;;
    esac
    
    echo ""
    echo "=========================================="
    echo "SETUP COMPLETE!"
    echo "=========================================="
    echo ""
    echo "Hardware: $HARDWARE"
    echo "Project directory: $PROJECT_DIR"
    echo ""
    echo "To run the pipeline:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 main.py --source 'rtsp://127.0.0.1:8554/cam_01' --display true"
    echo ""
    echo "Or with explicit device:"
    case $HARDWARE in
        "nvidia")
            echo "  VMS_DEVICE=cuda:0 VMS_BACKEND=tensorrt python3 main.py --source 'rtsp://127.0.0.1:8554/cam_01'"
            ;;
        "intel")
            echo "  VMS_DEVICE=GPU VMS_BACKEND=openvino python3 main.py --source 'rtsp://127.0.0.1:8554/cam_01'"
            ;;
        "cpu")
            echo "  VMS_DEVICE=cpu VMS_BACKEND=onnx python3 main.py --source 'rtsp://127.0.0.1:8554/cam_01'"
            ;;
    esac
    echo ""
}

# Run main
main
