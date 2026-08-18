from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray  # (4,) x1,y1,x2,y2
    conf: float
    cls_id: int = 0


def _letterbox(
    img: np.ndarray,
    target: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    th, tw = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, scale, dx, dy


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


class PersonDetector:
    """YOLO nano person-only detector (GPU/CPU via ONNX Runtime or TensorRT)."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 416,
        device: str = "cpu",
        person_class_id: int = 0,
        backend: str = "onnx",
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.person_class_id = person_class_id
        self.device = device
        self.backend = backend.lower()
        
        # Try to use environment variable for device
        env_device = os.environ.get("VMS_DEVICE", "").lower()
        if env_device:
            self.device = env_device
        
        # Try to use environment variable for backend
        env_backend = os.environ.get("VMS_BACKEND", "").lower()
        if env_backend:
            self.backend = env_backend
        
        # Initialize the appropriate backend
        self._init_backend(weights)

        # Get model metadata (ONNX Runtime only — TensorRT/OpenVINO skip this)
        if self.session is not None:
            meta = self.session.get_modelmeta()
            inp = self.session.get_inputs()[0]
            self._inp_name = inp.name
            self._inp_shape = inp.shape
            self._num_classes = 80 if "v8" in meta.description.lower() or "yolo" in meta.description.lower() else 80
    
    def _init_backend(self, weights: str) -> None:
        """Initialize the appropriate inference backend."""
        weight_path = str(weights)
        
        # Check if it's a TensorRT engine
        if self.backend == "tensorrt" or weight_path.endswith(".engine"):
            self._init_tensorrt(weight_path)
        # Check if it's OpenVINO
        elif self.backend == "openvino" or weight_path.endswith(".xml"):
            self._init_openvino(weight_path)
        # Default to ONNX Runtime
        else:
            self._init_onnxruntime(weight_path)
    
    def _init_tensorrt(self, engine_path: str) -> None:
        """Initialize TensorRT backend."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as model:
                self.engine = trt.Runtime(logger).deserialize_cuda_engine(model.read())
            
            self.context = self.engine.create_execution_context()
            
            # Allocate buffers
            self.stream = cuda.Stream()
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                if self.engine.binding_is_input(binding):
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.outputs.append({'host': host_mem, 'device': device_mem})
                self.bindings.append(int(device_mem))
            
            self._inp_name = "input"
            self._is_tensorrt = True
            self.session = None  # TensorRT backend has no onnxruntime session
            
        except ImportError as e:
            print(f"[WARNING] TensorRT not available: {e}. Falling back to ONNX Runtime.")
            self._init_onnxruntime(engine_path)
    
    def _init_openvino(self, xml_path: str) -> None:
        """Initialize OpenVINO backend."""
        try:
            from openvino.runtime import Core
            
            core = Core()
            model = core.read_model(xml_path)
            
            # Get device
            ov_device = "GPU" if self.device in ["gpu", "cuda:0", "cuda"] else "CPU"
            compiled_model = core.compile_model(model, ov_device)
            
            self.session = compiled_model
            self.output_layer = compiled_model.output(0)
            self._inp_name = compiled_model.input(0).name if len(compiled_model.inputs) > 0 else "input"
            self._is_openvino = True
            self._is_tensorrt = False
            
        except ImportError as e:
            print(f"[WARNING] OpenVINO not available: {e}. Falling back to ONNX Runtime.")
            self._init_onnxruntime(xml_path)
    
    def _init_onnxruntime(self, onnx_path: str) -> None:
        """Initialize ONNX Runtime backend with GPU/CPU support."""
        try:
            import onnxruntime as ort
            
            # Determine providers based on device
            providers = []
            device_str = self.device.lower()
            
            if device_str.startswith("cuda:") or device_str == "cuda" or device_str == "gpu":
                # GPU with CUDA
                try:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except:
                    providers = ["CPUExecutionProvider"]
            elif device_str == "cpu":
                providers = ["CPUExecutionProvider"]
            else:
                # Auto-detect: try GPU first, then CPU
                try:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                except:
                    providers = ["CPUExecutionProvider"]
            
            # Set provider options for GPU (must align 1:1 with providers)
            provider_options = [
                {"device_id": 0} if p == "CUDAExecutionProvider" else {} for p in providers
            ] if providers else None
            
            self.session = ort.InferenceSession(
                onnx_path, 
                providers=providers,
                provider_options=provider_options if provider_options else None
            )
            
            self._is_tensorrt = False
            self._is_openvino = False
            
        except ImportError as e:
            raise ImportError(f"ONNX Runtime not available: {e}. Please install with: pip install onnxruntime")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, scale, dx, dy = _letterbox(frame, (self.imgsz, self.imgsz))
        blob = letterboxed.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

        # Handle different backends
        if getattr(self, '_is_tensorrt', False):
            pred = self._infer_tensorrt(blob)
        elif getattr(self, '_is_openvino', False):
            pred = self._infer_openvino(blob)
        else:
            outputs = self.session.run(None, {self._inp_name: blob})
            pred = outputs[0][0]

        num_classes = pred.shape[0] - 4
        boxes = pred[:4, :]
        scores = pred[4:, :]

        cls_ids = scores.argmax(axis=0)
        confs = scores[cls_ids, np.arange(scores.shape[1])]

        mask = (cls_ids == self.person_class_id) & (confs >= self.conf)
        if not mask.any():
            return []

        boxes = boxes[:, mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        cx, cy, w, h = boxes
        x1 = (cx - w / 2 - dx) / scale
        y1 = (cy - h / 2 - dy) / scale
        x2 = (cx + w / 2 - dx) / scale
        y2 = (cy + h / 2 - dy) / scale

        stacked = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(stacked, confs, self.iou)

        out: List[Detection] = []
        for i in keep:
            out.append(
                Detection(
                    xyxy=stacked[i].astype(float),
                    conf=float(confs[i]),
                    cls_id=int(cls_ids[i]),
                )
            )
        return out
    
    def _infer_tensorrt(self, blob: np.ndarray) -> np.ndarray:
        """Run inference using TensorRT."""
        import pycuda.driver as cuda
        
        np.copyto(self.inputs[0]['host'], blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        return self.outputs[0]['host'].reshape([1, -1, *blob.shape[2:]])
    
    def _infer_openvino(self, blob: np.ndarray) -> np.ndarray:
        """Run inference using OpenVINO."""
        input_tensor = blob.astype(np.float32)
        result = self.session(input_tensor)[self.output_layer]
        return result
