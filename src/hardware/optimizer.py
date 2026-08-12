"""Dynamic Optimizer Module.

Monitors system metrics and automatically adjusts inference parameters
to maintain optimal performance.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from threading import Thread, Lock
from typing import Dict, List, Optional, Callable
import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Current system metrics."""
    cpu_usage: float = 0.0        # Percentage
    gpu_usage: float = 0.0        # Percentage
    gpu_memory: float = 0.0      # Percentage
    memory_usage: float = 0.0    # Percentage
    dropped_frames: int = 0
    inference_queue: int = 0
    fps: float = 0.0
    camera_count: int = 0
    motion_detected: bool = False
    timestamp: float = 0.0


@dataclass
class OptimizationAction:
    """Action to take for optimization."""
    action_type: str
    parameter: str
    old_value: float
    new_value: float
    reason: str


@dataclass
class OptimizerConfig:
    """Configuration for the dynamic optimizer."""
    check_interval: float = 5.0       # Seconds between checks
    cpu_threshold: float = 90.0      # % CPU usage to trigger action
    gpu_threshold: float = 95.0      # % GPU usage to trigger action
    gpu_memory_threshold: float = 90.0  # % GPU memory to trigger action
    dropped_frames_threshold: float = 10.0  # % dropped frames
    queue_threshold: int = 100       # Inference queue size
    idle_timeout: float = 5.0       # Seconds of no motion before reducing FPS
    recovery_timeout: float = 30.0   # Seconds to wait before reverting
    
    # Action parameters
    fps_reduction_pct: float = 20.0  # % to reduce FPS by
    batch_size_reduction: float = 0.5  # Factor to reduce batch size by
    min_fps: float = 1.0            # Minimum FPS
    max_fps: float = 30.0           # Maximum FPS


class DynamicOptimizer:
    """Dynamically optimizes inference parameters based on system metrics."""
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self._metrics: SystemMetrics = SystemMetrics()
        self._last_action: Optional[OptimizationAction] = None
        self._action_history: List[OptimizationAction] = []
        self._last_motion_time: float = 0.0
        self._last_recovery_time: Dict[str, float] = {}
        self._lock = Lock()
        self._running = False
        self._thread: Optional[Thread] = None
        
        # Callbacks for parameter changes
        self._callbacks: List[Callable[[str, float], None]] = []
        
        # Current parameters
        self._params: Dict[str, float] = {
            "fps": 30.0,
            "batch_size": 1.0,
            "model_size": 1.0,  # 0=small, 1=medium, 2=large
        }
    
    def start(self) -> None:
        """Start the optimizer thread."""
        if self._running:
            return
        self._running = True
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Dynamic optimizer started")
    
    def stop(self) -> None:
        """Stop the optimizer thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Dynamic optimizer stopped")
    
    def _run_loop(self) -> None:
        """Main optimization loop."""
        while self._running:
            self._collect_metrics()
            self._analyze_and_optimize()
            time.sleep(self.config.check_interval)
    
    def _collect_metrics(self) -> None:
        """Collect current system metrics."""
        with self._lock:
            # CPU usage
            self._metrics.cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            mem = psutil.virtual_memory()
            self._metrics.memory_usage = mem.percent
            
            # Try to get GPU metrics if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._metrics.gpu_usage = info.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._metrics.gpu_memory = (mem_info.used / mem_info.total) * 100
                pynvml.nvmlShutdown()
            except Exception:
                self._metrics.gpu_usage = 0.0
                self._metrics.gpu_memory = 0.0
            
            self._metrics.timestamp = time.time()
    
    def update_metrics(
        self,
        dropped_frames: int = 0,
        inference_queue: int = 0,
        fps: float = 0.0,
        camera_count: int = 0,
        motion_detected: bool = False,
    ) -> None:
        """Update custom metrics from the pipeline."""
        with self._lock:
            self._metrics.dropped_frames = dropped_frames
            self._metrics.inference_queue = inference_queue
            self._metrics.fps = fps
            self._metrics.camera_count = camera_count
            if motion_detected:
                self._last_motion_time = time.time()
            self._metrics.motion_detected = motion_detected
    
    def _analyze_and_optimize(self) -> None:
        """Analyze metrics and apply optimizations."""
        actions = []
        
        # Check CPU usage
        if self._metrics.cpu_usage > self.config.cpu_threshold:
            action = self._handle_high_cpu()
            if action:
                actions.append(action)
        
        # Check GPU usage
        if self._metrics.gpu_usage > self.config.gpu_threshold:
            action = self._handle_high_gpu()
            if action:
                actions.append(action)
        
        # Check GPU memory
        if self._metrics.gpu_memory > self.config.gpu_memory_threshold:
            action = self._handle_high_gpu_memory()
            if action:
                actions.append(action)
        
        # Check dropped frames
        if self._metrics.fps > 0:
            drop_rate = (self._metrics.dropped_frames / self._metrics.fps) * 100
            if drop_rate > self.config.dropped_frames_threshold:
                action = self._handle_high_dropped_frames()
                if action:
                    actions.append(action)
        
        # Check inference queue
        if self._metrics.inference_queue > self.config.queue_threshold:
            action = self._handle_high_queue()
            if action:
                actions.append(action)
        
        # Check for idle (no motion)
        if not self._metrics.motion_detected:
            idle_time = time.time() - self._last_motion_time
            if idle_time > self.config.idle_timeout:
                action = self._handle_idle()
                if action:
                    actions.append(action)
        
        # Check for recovery (revert optimizations)
        for param, last_time in list(self._last_recovery_time.items()):
            if time.time() - last_time > self.config.recovery_timeout:
                action = self._handle_recovery(param)
                if action:
                    actions.append(action)
                    del self._last_recovery_time[param]
        
        # Apply actions
        for action in actions:
            self._apply_action(action)
    
    def _handle_high_cpu(self) -> Optional[OptimizationAction]:
        """Handle high CPU usage."""
        current_fps = self._params.get("fps", 30.0)
        new_fps = max(self.config.min_fps, current_fps * (1 - self.config.fps_reduction_pct / 100))
        
        if new_fps < current_fps:
            return OptimizationAction(
                action_type="reduce_fps",
                parameter="fps",
                old_value=current_fps,
                new_value=new_fps,
                reason=f"CPU usage {self._metrics.cpu_usage}% > {self.config.cpu_threshold}%"
            )
        return None
    
    def _handle_high_gpu(self) -> Optional[OptimizationAction]:
        """Handle high GPU usage."""
        # Switch to lighter model
        current_model = self._params.get("model_size", 1.0)
        if current_model > 0:
            return OptimizationAction(
                action_type="reduce_model",
                parameter="model_size",
                old_value=current_model,
                new_value=0.0,
                reason=f"GPU usage {self._metrics.gpu_usage}% > {self.config.gpu_threshold}%"
            )
        return None
    
    def _handle_high_gpu_memory(self) -> Optional[OptimizationAction]:
        """Handle high GPU memory usage."""
        current_batch = self._params.get("batch_size", 1.0)
        new_batch = max(1, int(current_batch * self.config.batch_size_reduction))
        
        if new_batch < current_batch:
            return OptimizationAction(
                action_type="reduce_batch",
                parameter="batch_size",
                old_value=current_batch,
                new_value=new_batch,
                reason=f"GPU memory {self._metrics.gpu_memory}% > {self.config.gpu_memory_threshold}%"
            )
        return None
    
    def _handle_high_dropped_frames(self) -> Optional[OptimizationAction]:
        """Handle high dropped frames."""
        # Increase batch size to reduce overhead
        current_batch = self._params.get("batch_size", 1.0)
        new_batch = int(current_batch * 2)
        
        return OptimizationAction(
            action_type="increase_batch",
            parameter="batch_size",
            old_value=current_batch,
            new_value=new_batch,
            reason=f"Dropped frames {self._metrics.dropped_frames} > threshold"
        )
    
    def _handle_high_queue(self) -> Optional[OptimizationAction]:
        """Handle high inference queue."""
        # Prioritize high-priority cameras
        # For now, just reduce FPS
        current_fps = self._params.get("fps", 30.0)
        new_fps = max(self.config.min_fps, current_fps * (1 - self.config.fps_reduction_pct / 100))
        
        if new_fps < current_fps:
            return OptimizationAction(
                action_type="reduce_fps",
                parameter="fps",
                old_value=current_fps,
                new_value=new_fps,
                reason=f"Queue size {self._metrics.inference_queue} > {self.config.queue_threshold}"
            )
        return None
    
    def _handle_idle(self) -> Optional[OptimizationAction]:
        """Handle idle state (no motion)."""
        current_fps = self._params.get("fps", 30.0)
        if current_fps > 1.0:
            return OptimizationAction(
                action_type="reduce_fps_idle",
                parameter="fps",
                old_value=current_fps,
                new_value=1.0,
                reason="No motion detected"
            )
        return None
    
    def _handle_recovery(self, param: str) -> Optional[OptimizationAction]:
        """Handle recovery (revert optimization)."""
        if param == "fps":
            # Restore FPS to max
            current_fps = self._params.get("fps", 30.0)
            if current_fps < self.config.max_fps:
                return OptimizationAction(
                    action_type="restore_fps",
                    parameter="fps",
                    old_value=current_fps,
                    new_value=self.config.max_fps,
                    reason="Recovery after cooldown"
                )
        elif param == "batch_size":
            # Restore batch size
            current_batch = self._params.get("batch_size", 1.0)
            if current_batch < 1.0:
                return OptimizationAction(
                    action_type="restore_batch",
                    parameter="batch_size",
                    old_value=current_batch,
                    new_value=1.0,
                    reason="Recovery after cooldown"
                )
        return None
    
    def _apply_action(self, action: OptimizationAction) -> None:
        """Apply an optimization action."""
        logger.info(f"Optimization: {action.action_type} - {action.parameter} {action.old_value} -> {action.new_value} ({action.reason})")
        
        # Update parameter
        self._params[action.parameter] = action.new_value
        
        # Track last action
        self._last_action = action
        self._action_history.append(action)
        
        # Track recovery time for revert
        if action.action_type.startswith("reduce"):
            self._last_recovery_time[action.parameter] = time.time()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(action.parameter, action.new_value)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def get_current_params(self) -> Dict[str, float]:
        """Get current optimization parameters."""
        with self._lock:
            return dict(self._params)
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        with self._lock:
            return SystemMetrics(
                cpu_usage=self._metrics.cpu_usage,
                gpu_usage=self._metrics.gpu_usage,
                gpu_memory=self._metrics.gpu_memory,
                memory_usage=self._metrics.memory_usage,
                dropped_frames=self._metrics.dropped_frames,
                inference_queue=self._metrics.inference_queue,
                fps=self._metrics.fps,
                camera_count=self._metrics.camera_count,
                motion_detected=self._metrics.motion_detected,
                timestamp=self._metrics.timestamp,
            )
    
    def on_parameter_change(self, callback: Callable[[str, float], None]) -> None:
        """Register a callback for parameter changes."""
        self._callbacks.append(callback)
    
    def get_fps(self, resolution: str = "1080p") -> float:
        """Get recommended FPS based on current optimization."""
        base_fps = self._params.get("fps", 30.0)
        if resolution == "480p":
            return min(base_fps * 2, self.config.max_fps)
        return base_fps
