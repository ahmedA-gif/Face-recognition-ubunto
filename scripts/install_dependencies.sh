#!/bin/bash

# FACE-RECOGNITION-UBUNTO: Dependencies Installer
# This script installs dependencies based on hardware type

set -e

echo "=========================================="
echo "FACE-RECOGNITION-UBUNTO DEPENDENCIES INSTALLER"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
HARDWARE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu|--nvidia)
            HARDWARE="nvidia"
            shift
            ;;
        --intel)
            HARDWARE="intel"
            shift
            ;;
        --amd)
            HARDWARE="amd"
            shift
            ;;
        --coral)
            HARDWARE="coral"
            shift
            ;;
        --cpu)
            HARDWARE="cpu"
            shift
            ;;
        --all)
            HARDWARE="all"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo ""
            echo "Options:"
            echo "  --gpu, --nvidia    Install NVIDIA GPU dependencies"
            echo "  --intel            Install Intel iGPU dependencies"
            echo "  --amd              Install AMD GPU dependencies"
            echo "  --coral            Install Coral TPU dependencies"
            echo "  --cpu              Install CPU-only dependencies"
            echo "  --all              Install all dependencies"
            echo ""
            echo "If no option is provided, auto-detect hardware and install appropriate dependencies."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-detect hardware if not specified
detect_hardware() {
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "nvidia"
        return
    fi
    
    # Check for Intel iGPU
    if command -v intel_gpu_top &> /dev/null; then
        echo "intel"
        return
    fi
    
    # Check for AMD GPU (ROCm)
    if [ -d /opt/rocm ] || command -v rocminfo &> /dev/null; then
        echo "amd"
        return
    fi
    
    # Check for Coral TPU
    if [ -e /dev/apex_0 ] || lsusb | grep -i coral &> /dev/null; then
        echo "coral"
        return
    fi
    
    echo "cpu"
}

# If hardware not specified, auto-detect
if [ -z "$HARDWARE" ]; then
    echo "Auto-detecting hardware..."
    HARDWARE=$(detect_hardware)
    echo "Detected hardware: $HARDWARE"
    echo ""
fi

# Install base dependencies
echo "Installing base dependencies..."
pip install -r "$PROJECT_DIR/requirements.txt"
echo ""

# Function to install NVIDIA dependencies
install_nvidia() {
    echo "=========================================="
    echo "Installing NVIDIA GPU dependencies"
    echo "=========================================="
    
    # Install system packages
    echo "Installing system packages..."
    sudo apt update
    sudo apt install -y nvidia-cuda-toolkit nvidia-tensorrt
    echo ""
    
    # Install Python packages
    echo "Installing Python packages..."
    pip install --upgrade onnxruntime-gpu
    pip install --upgrade nvidia-tensorrt nvidia-cuda-runtime nvidia-cudnn
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install --upgrade pycuda
    
    echo "NVIDIA dependencies installed successfully"
    echo ""
}

# Function to install Intel dependencies
install_intel() {
    echo "=========================================="
    echo "Installing Intel iGPU dependencies"
    echo "=========================================="
    
    # Install system packages
    echo "Installing system packages..."
    sudo apt update
    sudo apt install -y intel-opencl-icd intel-media-va-driver-non-free
    echo ""
    
    # Install Python packages
    echo "Installing Python packages..."
    pip install --upgrade openvino openvino-dev
    
    echo "Intel dependencies installed successfully"
    echo ""
}

# Function to install AMD dependencies
install_amd() {
    echo "=========================================="
    echo "Installing AMD GPU dependencies"
    echo "=========================================="
    
    # Install system packages
    echo "Installing system packages..."
    sudo apt update
    sudo apt install -y rocm-opencl-runtime rocm-hip-sdk
    sudo usermod -aG video $USER
    echo ""
    
    # Install Python packages
    echo "Installing Python packages..."
    pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.6
    
    echo "AMD dependencies installed successfully"
    echo ""
}

# Function to install Coral dependencies
install_coral() {
    echo "=========================================="
    echo "Installing Coral TPU dependencies"
    echo "=========================================="
    
    # Add Coral repository
    echo "Adding Coral repository..."
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu-stable.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update
    echo ""
    
    # Install Coral runtime
    echo "Installing Coral runtime..."
    sudo apt install -y libedgetpu1-std python3-pycoral
    echo ""
    
    # Install Python packages
    echo "Installing Python packages..."
    pip install --upgrade tflite-runtime pycoral
    
    echo "Coral dependencies installed successfully"
    echo ""
}

# Function to install CPU dependencies
install_cpu() {
    echo "=========================================="
    echo "CPU-only dependencies already installed via requirements.txt"
    echo "=========================================="
    echo ""
}

# Install based on hardware
case "$HARDWARE" in
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
    "all")
        echo "Installing dependencies for all hardware types..."
        echo ""
        install_nvidia || true
        install_intel || true
        install_amd || true
        install_coral || true
        ;;
esac

echo "=========================================="
echo "DEPENDENCIES INSTALLATION COMPLETE"
echo "=========================================="
echo ""
echo "Installed dependencies for: $HARDWARE"
echo ""
echo "Next steps:"
echo "1. Run: python3 scripts/download_models.py --$HARDWARE"
echo "2. Run: python3 main.py --source 'your_camera_source'"
echo ""
