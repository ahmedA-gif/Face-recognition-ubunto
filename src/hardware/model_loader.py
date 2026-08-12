"""Model Loader Module.

Handles loading, caching, and hot-swapping of AI models.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import importlib

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    framework: str
    path: str
    input_size: int = 640
    classes: List[str] = field(default_factory=list)
    is_loaded: bool = False
    module: Any = None  # Loaded model module
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRegistry:
    """Registry of available models."""
    models_dir: Path = Path("/opt/vms/models")
    models: Dict[str, ModelInfo] = field(default_factory=dict)
    loaded_models: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.models_dir = Path(self.models_dir)
        self._scan_models()
    
    def _scan_models(self) -> None:
        """Scan the models directory for available models."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                self.models[model_name] = self._load_model_info(model_dir)
    
    def _load_model_info(self, model_dir: Path) -> ModelInfo:
        """Load model information from directory."""
        metadata_path = model_dir / "metadata.json"
        
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                return ModelInfo(
                    name=model_dir.name,
                    framework=metadata.get("framework", "onnx"),
                    path=str(model_dir),
                    input_size=metadata.get("input_size", 640),
                    classes=metadata.get("classes", []),
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Error loading metadata for {model_dir.name}: {e}")
        
        # Default metadata
        return ModelInfo(
            name=model_dir.name,
            framework="onnx",
            path=str(model_dir),
        )
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model information by name."""
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self.models.keys())
    
    def get_best_model(self, hardware_type: str, framework: str) -> Optional[str]:
        """Get the best model for given hardware/framework."""
        # Model priority per framework
        framework_models = {
            "tensorrt": ["yolo11l", "yolo11m", "yolo11s", "yolo11n"],
            "openvino": ["yolo11s", "yolo11n", "yolo8n"],
            "tflite": ["yolo11n", "yolo8n"],
            "onnx": ["yolo11n", "yolo11s", "yolo8n"],
            "rocm": ["yolo11s", "yolo11n", "yolo8n"],
        }
        
        # Hardware model preferences
        hardware_preferences = {
            "nvidia": ["yolo11l", "yolo11m", "yolo11s", "yolo11n"],
            "intel": ["yolo11s", "yolo11n", "yolo8n"],
            "coral": ["yolo11n", "yolo8n"],
            "amd": ["yolo11s", "yolo11n", "yolo8n"],
            "cpu": ["yolo11n", "yolo8n"],
        }
        
        preferred = hardware_preferences.get(hardware_type, [])
        available = self.models.keys()
        
        for model in preferred:
            if model in available:
                return model
        
        # Fallback to first available model
        if available:
            return available[0]
        
        return None


class ModelLoader:
    """Loads and manages AI models with hot-swapping support."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self._loaded_models: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}
        self._load_hooks: Dict[str, Callable] = {}
        self._unload_hooks: Dict[str, Callable] = {}
        
        # Register default loaders
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register default model loaders for each framework."""
        self.register_loader("onnx", self._load_onnx)
        self.register_loader("tensorrt", self._load_tensorrt)
        self.register_loader("openvino", self._load_openvino)
        self.register_loader("tflite", self._load_tflite)
        self.register_loader("rocm", self._load_rocm)
    
    def register_loader(self, framework: str, loader: Callable) -> None:
        """Register a custom loader for a framework."""
        self._load_hooks[framework] = loader
    
    def register_unloader(self, framework: str, unloader: Callable) -> None:
        """Register a custom unloader for a framework."""
        self._unload_hooks[framework] = unloader
    
    def load(self, model_name: str, framework: Optional[str] = None) -> Any:
        """Load a model by name."""
        model_info = self.registry.get_model(model_name)
        if not model_info:
            raise ValueError(f"Model not found: {model_name}")
        
        if framework is None:
            framework = model_info.framework
        
        # Check if already loaded
        cache_key = f"{model_name}:{framework}"
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        # Use specific loader if available
        loader = self._load_hooks.get(framework, self._load_generic)
        model = loader(model_info)
        
        # Cache the model
        self._model_cache[cache_key] = model
        model_info.is_loaded = True
        model_info.module = model
        
        logger.info(f"Loaded model: {model_name} ({framework})")
        return model
    
    def unload(self, model_name: str, framework: Optional[str] = None) -> bool:
        """Unload a model by name."""
        if framework is None:
            model_info = self.registry.get_model(model_name)
            if model_info:
                framework = model_info.framework
        
        cache_key = f"{model_name}:{framework}"
        if cache_key in self._model_cache:
            # Call unloader if available
            unloader = self._unload_hooks.get(framework)
            if unloader:
                unloader(self._model_cache[cache_key])
            
            del self._model_cache[cache_key]
            
            # Update model info
            model_info = self.registry.get_model(model_name)
            if model_info:
                model_info.is_loaded = False
                model_info.module = None
            
            logger.info(f"Unloaded model: {model_name} ({framework})")
            return True
        return False
    
    def switch_model(self, old_model: str, new_model: str) -> Any:
        """Hot-swap from one model to another."""
        # Unload old model
        self.unload(old_model)
        
        # Load new model
        return self.load(new_model)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return list(self._model_cache.keys())
    
    def _load_onnx(self, model_info: ModelInfo) -> Any:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            model_path = str(Path(model_info.path) / f"{model_info.name}.onnx")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            return session
        except ImportError:
            raise ImportError("ONNX Runtime not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _load_tensorrt(self, model_info: ModelInfo) -> Any:
        """Load TensorRT model."""
        try:
            import tensorrt as trt
            logger = trt.Logger(trt.Logger.WARNING)
            
            model_path = str(Path(model_info.path) / f"{model_info.name}_engine_fp16.tensorrt")
            if not Path(model_path).exists():
                model_path = str(Path(model_info.path) / f"{model_info.name}.plan")
            
            with open(model_path, 'rb') as model, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(model.read())
            
            return engine
        except ImportError:
            raise ImportError("TensorRT not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT model: {e}")
    
    def _load_openvino(self, model_info: ModelInfo) -> Any:
        """Load OpenVINO model."""
        try:
            from openvino.runtime import Core
            core = Core()
            
            xml_path = str(Path(model_info.path) / f"{model_info.name}.xml")
            bin_path = str(Path(model_info.path) / f"{model_info.name}.bin")
            
            model = core.read_model(model=xml_path, weights=bin_path)
            compiled_model = core.compile_model(model, "AUTO")
            
            return compiled_model
        except ImportError:
            raise ImportError("OpenVINO not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO model: {e}")
    
    def _load_tflite(self, model_info: ModelInfo) -> Any:
        """Load TFLite model."""
        try:
            import tflite_runtime.interpreter as tflite
            
            model_path = str(Path(model_info.path) / f"{model_info.name}.tflite")
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            return interpreter
        except ImportError:
            raise ImportError("TFLite not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load TFLite model: {e}")
    
    def _load_rocm(self, model_info: ModelInfo) -> Any:
        """Load ROCm model."""
        try:
            import torch
            model_path = str(Path(model_info.path) / f"{model_info.name}.pt")
            model = torch.jit.load(model_path)
            model.eval()
            return model
        except ImportError:
            raise ImportError("PyTorch/ROCm not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to load ROCm model: {e}")
    
    def _load_generic(self, model_info: ModelInfo) -> Any:
        """Generic model loader (fallback)."""
        # Try ONNX first
        try:
            return self._load_onnx(model_info)
        except Exception:
            pass
        
        # Try other frameworks
        loaders = [
            self._load_tensorrt,
            self._load_openvino,
            self._load_tflite,
            self._load_rocm,
        ]
        
        for loader in loaders:
            try:
                return loader(model_info)
            except Exception:
                continue
        
        raise RuntimeError(f"No loader available for model: {model_info.name}")


# Global instances
_registry: Optional[ModelRegistry] = None
_loader: Optional[ModelLoader] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_loader() -> ModelLoader:
    """Get or create the global model loader."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
