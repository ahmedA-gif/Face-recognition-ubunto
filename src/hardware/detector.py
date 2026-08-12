"""Hardware Detection Module.

Auto-detects available hardware (NVIDIA GPU, Intel iGPU, Coral TPU, AMD GPU, CPU)
and selects the optimal model/framework combination.
"""

from __future__ import annotations

import os
import subprocess
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class HardwareType(Enum):
    """Supported hardware types."""
    NVIDIA = "nvidia"
    INTEL = "intel"
    CORAL = "coral"
    AMD = "amd"
    CPU = "cpu"


class FrameworkType(Enum):
    """Supported AI frameworks."""
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    TFLITE = "tflite"
    ONNX = "onnx"
    ROCM = "rocm"


@dataclass
class HardwareConfig:
    """Configuration for detected hardware."""
    hardware_type: HardwareType
    framework: FrameworkType
    model_name: str
    fps_480p: int
    fps_1080p: int
    vram_gb: Optional[float] = None
    cpu_features: Dict[str, bool] = field(default_factory=dict)
    
    @property
    def is_gpu(self) -> bool:
        return self.hardware_type in (HardwareType.NVIDIA, HardwareType.AMD, HardwareType.INTEL)
    
    @property
    def is_accelerated(self) -> bool:
        return self.hardware_type in (HardwareType.NVIDIA, HardwareType.INTEL, HardwareType.CORAL, HardwareType.AMD)
    
    def to_dict(self) -> Dict:
        return {
            "hardware_type": self.hardware_type.value,
            "framework": self.framework.value,
            "model_name": self.model_name,
            "fps_480p": self.fps_480p,
            "fps_1080p": self.fps_1080p,
            "vram_gb": self.vram_gb,
            "cpu_features": self.cpu_features,
            "is_gpu": self.is_gpu,
            "is_accelerated": self.is_accelerated,
        }


class HardwareDetector:
    """Detects available hardware and selects optimal configuration."""
    
    # Model recommendations per hardware
    MODEL_RECOMMENDATIONS = {
        HardwareType.NVIDIA: {
            "default": ("yolo11m", FrameworkType.TENSORRT, 50, 25),
            "high_vram": ("yolo11l", FrameworkType.TENSORRT, 60, 30),
            "low_vram": ("yolo11s", FrameworkType.TENSORRT, 40, 15),
        },
        HardwareType.INTEL: {
            "default": ("yolo11s", FrameworkType.OPENVINO, 30, 10),
            "fallback": ("yolo11n", FrameworkType.ONNX, 10, 4),
        },
        HardwareType.CORAL: {
            "default": ("yolo11n", FrameworkType.TFLITE, 20, 8),
            "fallback": ("yolo11n", FrameworkType.ONNX, 10, 4),
        },
        HardwareType.AMD: {
            "default": ("yolo11s", FrameworkType.ONNX, 15, 5),
            "rocm": ("yolo11s", FrameworkType.ROCM, 15, 5),
        },
        HardwareType.CPU: {
            "avx512": ("yolo11s", FrameworkType.ONNX, 10, 4),
            "avx2": ("yolo11n", FrameworkType.ONNX, 10, 4),
            "fallback": ("yolo8n", FrameworkType.ONNX, 5, 2),
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[HardwareConfig] = None
    
    def detect(self) -> HardwareConfig:
        """Detect hardware and return optimal configuration."""
        logger.info("Starting hardware detection...")
        
        # Try NVIDIA first
        if self._check_nvidia():
            return self._config
        
        # Try Intel
        if self._check_intel():
            return self._config
        
        # Try Coral
        if self._check_coral():
            return self._config
        
        # Try AMD
        if self._check_amd():
            return self._config
        
        # Fallback to CPU
        self._check_cpu()
        return self._config
    
    def _check_nvidia(self) -> bool:
        """Check for NVIDIA GPU and TensorRT support."""
        try:
            # Check nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                # Extract VRAM
                vram_str = result.stdout.strip().split()[0]
                vram = float(''.join(c for c in vram_str if c.isdigit() or c == '.'))
                
                # Select model based on VRAM
                if vram >= 8:
                    model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.NVIDIA]["high_vram"]
                elif vram >= 4:
                    model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.NVIDIA]["default"]
                else:
                    model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.NVIDIA]["low_vram"]
                
                # Verify TensorRT
                if not self._check_python_module("tensorrt"):
                    framework = FrameworkType.ONNX
                    logger.warning("TensorRT not available, falling back to ONNX")
                
                self._config = HardwareConfig(
                    hardware_type=HardwareType.NVIDIA,
                    framework=framework,
                    model_name=model,
                    fps_480p=fps_480p,
                    fps_1080p=fps_1080p,
                    vram_gb=vram,
                )
                logger.info(f"NVIDIA GPU detected: VRAM={vram}GB, Model={model}, Framework={framework.value}")
                return True
        except Exception as e:
            logger.debug(f"NVIDIA detection failed: {e}")
        
        return False
    
    def _check_intel(self) -> bool:
        """Check for Intel iGPU and OpenVINO support."""
        try:
            # Check intel_gpu_top
            result = subprocess.run(
                ["intel_gpu_top"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.INTEL]["default"]
                
                # Verify OpenVINO
                if not self._check_python_module("openvino"):
                    framework = FrameworkType.ONNX
                    model, _, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.INTEL]["fallback"]
                    logger.warning("OpenVINO not available, falling back to ONNX")
                
                self._config = HardwareConfig(
                    hardware_type=HardwareType.INTEL,
                    framework=framework,
                    model_name=model,
                    fps_480p=fps_480p,
                    fps_1080p=fps_1080p,
                )
                logger.info(f"Intel iGPU detected: Model={model}, Framework={framework.value}")
                return True
        except Exception as e:
            logger.debug(f"Intel detection failed: {e}")
        
        return False
    
    def _check_coral(self) -> bool:
        """Check for Coral TPU."""
        try:
            if Path("/dev/apex_0").exists():
                model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.CORAL]["default"]
                
                # Verify TFLite
                if not self._check_python_module("tflite_runtime"):
                    framework = FrameworkType.ONNX
                    model, _, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.CORAL]["fallback"]
                    logger.warning("TFLite not available, falling back to ONNX")
                
                self._config = HardwareConfig(
                    hardware_type=HardwareType.CORAL,
                    framework=framework,
                    model_name=model,
                    fps_480p=fps_480p,
                    fps_1080p=fps_1080p,
                )
                logger.info(f"Coral TPU detected: Model={model}, Framework={framework.value}")
                return True
        except Exception as e:
            logger.debug(f"Coral detection failed: {e}")
        
        return False
    
    def _check_amd(self) -> bool:
        """Check for AMD GPU and ROCm support."""
        try:
            if Path("/opt/rocm/bin/rocminfo").exists():
                model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.AMD]["rocm"]
                
                # Verify ROCm
                if not self._check_python_module("torch") or not self._check_rocm():
                    framework = FrameworkType.ONNX
                    model, _, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.AMD]["default"]
                    logger.warning("ROCm not available, falling back to ONNX")
                
                self._config = HardwareConfig(
                    hardware_type=HardwareType.AMD,
                    framework=framework,
                    model_name=model,
                    fps_480p=fps_480p,
                    fps_1080p=fps_1080p,
                )
                logger.info(f"AMD GPU detected: Model={model}, Framework={framework.value}")
                return True
        except Exception as e:
            logger.debug(f"AMD detection failed: {e}")
        
        return False
    
    def _check_cpu(self) -> bool:
        """Check CPU features and select optimal model."""
        try:
            cpu_features = {"avx2": False, "avx512": False}
            
            # Check /proc/cpuinfo on Linux
            if Path("/proc/cpuinfo").exists():
                cpuinfo = Path("/proc/cpuinfo").read_text()
                cpu_features["avx2"] = "avx2" in cpuinfo.lower()
                cpu_features["avx512"] = "avx512" in cpuinfo.lower()
            
            # Select model based on CPU features
            if cpu_features["avx512"]:
                model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.CPU]["avx512"]
            elif cpu_features["avx2"]:
                model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.CPU]["avx2"]
            else:
                model, framework, fps_480p, fps_1080p = self.MODEL_RECOMMENDATIONS[HardwareType.CPU]["fallback"]
            
            self._config = HardwareConfig(
                hardware_type=HardwareType.CPU,
                framework=framework,
                model_name=model,
                fps_480p=fps_480p,
                fps_1080p=fps_1080p,
                cpu_features=cpu_features,
            )
            logger.info(f"CPU detected: Features={cpu_features}, Model={model}, Framework={framework.value}")
            return True
        except Exception as e:
            logger.debug(f"CPU detection failed: {e}")
            # Ultimate fallback
            self._config = HardwareConfig(
                hardware_type=HardwareType.CPU,
                framework=FrameworkType.ONNX,
                model_name="yolo8n",
                fps_480p=5,
                fps_1080p=2,
            )
            return True
    
    def _check_python_module(self, module_name: str) -> bool:
        """Check if a Python module is importable."""
        try:
            import importlib
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def _check_rocm(self) -> bool:
        """Check if ROCm is working."""
        try:
            result = subprocess.run(
                ["python3", "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True, text=True, timeout=5
            )
            return "True" in result.stdout
        except Exception:
            return False
    
    def get_config(self) -> HardwareConfig:
        """Return the detected hardware configuration."""
        if self._config is None:
            self.detect()
        return self._config
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to YAML file."""
        import yaml
        from pathlib import Path
        
        save_path = Path(path) if path else self.config_path or Path("/opt/vms/configs/hardware.yaml")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.get_config().to_dict()
        config_dict["detection"] = {
            "backend": self.get_config().framework.value,
            "model_path": f"/opt/vms/models/{self.get_config().model_name}",
        }
        
        with open(save_path, 'w') as f:
            yaml.dump({"hardware": config_dict}, f, indent=2)
        
        logger.info(f"Hardware configuration saved to {save_path}")
    
    @staticmethod
    def load_config(path: str) -> HardwareConfig:
        """Load configuration from YAML file."""
        import yaml
        from pathlib import Path
        
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path) as f:
            data = yaml.safe_load(f)
        
        hardware_data = data.get("hardware", {})
        return HardwareConfig(
            hardware_type=HardwareType(hardware_data.get("type", "cpu")),
            framework=FrameworkType(hardware_data.get("framework", "onnx")),
            model_name=hardware_data.get("model_name", "yolo11n"),
            fps_480p=hardware_data.get("fps_480p", 10),
            fps_1080p=hardware_data.get("fps_1080p", 4),
            vram_gb=hardware_data.get("vram_gb"),
            cpu_features=hardware_data.get("cpu_features", {}),
        )


# Global detector instance
_detector: Optional[HardwareDetector] = None


def get_detector() -> HardwareDetector:
    """Get or create the global hardware detector."""
    global _detector
    if _detector is None:
        _detector = HardwareDetector()
    return _detector


def detect_hardware() -> HardwareConfig:
    """Detect hardware and return configuration."""
    return get_detector().detect()


def get_hardware_config() -> HardwareConfig:
    """Get the current hardware configuration."""
    return get_detector().get_config()
