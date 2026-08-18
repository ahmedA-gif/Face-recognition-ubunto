"""GPU memory monitor — auto-switches to CPU when VRAM is exhausted.

Usage in pipeline runner:
    from src.hardware.gpu_monitor import gpu_monitor
    if not gpu_monitor.try_use_gpu():
        device = "cpu"
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class GPUMonitor:
    """Tracks GPU VRAM usage and auto-switches to CPU when threshold exceeded.

    Parameters
    ----------
    vram_threshold_gb : float
        Maximum VRAM usage (GB) before auto-switching to CPU.
    check_interval_sec : float
        How often to poll GPU memory (seconds).
    cooldown_sec : float
        How long to stay on CPU before retrying GPU.
    """

    def __init__(
        self,
        vram_threshold_gb: float = 2.5,
        check_interval_sec: float = 5.0,
        cooldown_sec: float = 60.0,
    ) -> None:
        self.vram_threshold_gb = vram_threshold_gb
        self.check_interval_sec = check_interval_sec
        self.cooldown_sec = cooldown_sec
        self._use_gpu = True
        self._switch_time: float = 0.0
        self._lock = threading.Lock()
        self._gpu_available = self._detect_gpu()

    @staticmethod
    def _detect_gpu() -> bool:
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    @staticmethod
    def _get_vram_usage_gb() -> Optional[float]:
        """Return current GPU VRAM usage in GB, or None if unavailable."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mb = float(result.stdout.strip().split("\n")[0])
                return mb / 1024.0
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return None

    def try_use_gpu(self) -> bool:
        """Check if GPU is safe to use. Returns True if GPU OK, False = use CPU."""
        if not self._gpu_available:
            return False
        with self._lock:
            if not self._use_gpu:
                if time.time() - self._switch_time > self.cooldown_sec:
                    usage = self._get_vram_usage_gb()
                    if usage is None or usage < self.vram_threshold_gb * 0.7:
                        self._use_gpu = True
                        print(f"[GPUMonitor] Retrying GPU (usage={usage:.2f}GB)")
                        return True
                return False
            usage = self._get_vram_usage_gb()
            if usage is not None and usage > self.vram_threshold_gb:
                self._use_gpu = False
                self._switch_time = time.time()
                print(
                    f"[GPUMonitor] VRAM {usage:.2f}GB > threshold {self.vram_threshold_gb}GB — "
                    f"switching to CPU for {self.cooldown_sec}s"
                )
                return False
            return True

    @property
    def is_gpu_mode(self) -> bool:
        with self._lock:
            return self._use_gpu and self._gpu_available

    @property
    def gpu_available(self) -> bool:
        return self._gpu_available

    def get_status(self) -> dict:
        usage = self._get_vram_usage_gb()
        return {
            "gpu_available": self._gpu_available,
            "gpu_mode": self.is_gpu_mode,
            "vram_used_gb": round(usage, 2) if usage else None,
            "vram_threshold_gb": self.vram_threshold_gb,
            "cooldown_remaining": max(0, self.cooldown_sec - (time.time() - self._switch_time)) if not self._use_gpu else 0,
        }


# Singleton instance
gpu_monitor = GPUMonitor()
