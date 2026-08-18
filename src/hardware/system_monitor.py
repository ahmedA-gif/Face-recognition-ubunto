"""System monitor — live CPU/GPU temperature, RAM, disk, network, uptime metrics.

Usage:
    from src.hardware.system_monitor import system_monitor
    metrics = system_monitor.get_all()
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class SystemMonitor:
    """Collects live system metrics: CPU temp, GPU temp, RAM, disk, network, uptime."""

    def __init__(self, poll_interval: float = 2.0) -> None:
        self.poll_interval = poll_interval
        self._cache: Dict[str, Any] = {}
        self._last_poll: float = 0.0
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._prev_net = self._get_net_io()

    # ── CPU Temperature ───────────────────────────────────────────────────

    def _get_cpu_temp_windows(self) -> Optional[float]:
        """Get CPU temperature on Windows via WMI or OpenHardwareMonitor."""
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace 'root/wmi' "
                 "| Select-Object -First 1 -ExpandProperty CurrentTemperature"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                raw = float(result.stdout.strip())
                return (raw / 10.0) - 273.15  # decikelvin to celsius
        except Exception:
            pass
        # Fallback: try OpenHardwareMonitor
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "(Get-WmiObject -Namespace root\\OpenHardwareMonitor -Class Sensor "
                 "| Where-Object {$_.SensorType -eq 'Temperature' -and $_.Name -like '*CPU*'} "
                 "| Select-Object -First 1).Value"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def _get_cpu_temp_linux(self) -> Optional[float]:
        """Get CPU temperature on Linux from thermal zones."""
        try:
            for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
                type_file = zone / "type"
                temp_file = zone / "temp"
                if temp_file.exists():
                    raw = int(temp_file.read_text().strip())
                    return raw / 1000.0
        except Exception:
            pass
        # Try lm-sensors
        try:
            result = subprocess.run(["sensors", "-u"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n"):
                if "temp1_input" in line:
                    return float(line.split(":")[-1].strip())
        except Exception:
            pass
        return None

    def get_cpu_temp(self) -> Optional[float]:
        if platform.system() == "Windows":
            return self._get_cpu_temp_windows()
        return self._get_cpu_temp_linux()

    # ── GPU Temperature ───────────────────────────────────────────────────

    def _get_gpu_temp_nvidia(self) -> Optional[float]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0])
        except Exception:
            pass
        return None

    def get_gpu_temp(self) -> Optional[float]:
        return self._get_gpu_temp_nvidia()

    # ── GPU VRAM ──────────────────────────────────────────────────────────

    def get_gpu_vram(self) -> Dict[str, Optional[float]]:
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,memory.total,memory.free,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 4:
                    return {
                        "used_mb": float(parts[0]),
                        "total_mb": float(parts[1]),
                        "free_mb": float(parts[2]),
                        "utilization_pct": float(parts[3]),
                    }
        except Exception:
            pass
        return {"used_mb": None, "total_mb": None, "free_mb": None, "utilization_pct": None}

    # ── RAM ───────────────────────────────────────────────────────────────

    def get_ram(self) -> Dict[str, float]:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent": mem.percent,
            }
        except ImportError:
            pass
        # Fallback: platform-specific
        if platform.system() == "Windows":
            return self._get_ram_windows()
        return self._get_ram_linux()

    def _get_ram_windows(self) -> Dict[str, float]:
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "$m=Get-CimInstance Win32_OperatingSystem; "
                 "[math]::Round($m.TotalVisibleMemorySize/1MB,2)|%%{"
                 "[math]::Round(($m.TotalVisibleMemorySize-$m.FreePhysicalMemory)/1MB,2)|%%{"
                 "[math]::Round($m.FreePhysicalMemory/1MB,2)"],
                capture_output=True, text=True, timeout=5
            )
            # Simpler approach
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-CimInstance Win32_OperatingSystem | Select-Object "
                 "@{N='Total';E={[math]::Round($_.TotalVisibleMemorySize/1MB,2)}}, "
                 "@{N='Used';E={[math]::Round(($_.TotalVisibleMemorySize-$_.FreePhysicalMemory)/1MB,2)}}, "
                 "@{N='Free';E={[math]::Round($_.FreePhysicalMemory/1MB,2)}} | ConvertTo-Json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout.strip())
                total = float(data["Total"])
                used = float(data["Used"])
                free = float(data["Free"])
                return {
                    "total_gb": total,
                    "used_gb": used,
                    "available_gb": free,
                    "percent": round((used / total) * 100, 1) if total > 0 else 0,
                }
        except Exception:
            pass
        return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0}

    def _get_ram_linux(self) -> Dict[str, float]:
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = int(parts[1].strip().split()[0]) / (1024**2)  # kB to GB
                    info[key] = val
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            used = total - available
            return {
                "total_gb": round(total, 2),
                "used_gb": round(used, 2),
                "available_gb": round(available, 2),
                "percent": round((used / total) * 100, 1) if total > 0 else 0,
            }
        except Exception:
            return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0}

    # ── CPU Usage ─────────────────────────────────────────────────────────

    def get_cpu_usage(self) -> Dict[str, Any]:
        try:
            import psutil
            return {
                "percent": psutil.cpu_percent(interval=0.1),
                "cores": psutil.cpu_count(logical=False) or 0,
                "threads": psutil.cpu_count(logical=True) or 0,
                "freq_mhz": round(psutil.cpu_freq().current, 0) if psutil.cpu_freq() else 0,
            }
        except ImportError:
            pass
        return self._get_cpu_usage_fallback()

    def _get_cpu_usage_fallback(self) -> Dict[str, Any]:
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command",
                     "(Get-CimInstance Win32_Processor).LoadPercentage"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return {"percent": float(result.stdout.strip()),
                            "cores": os.cpu_count() or 0, "threads": os.cpu_count() or 0, "freq_mhz": 0}
        except Exception:
            pass
        return {"percent": 0, "cores": os.cpu_count() or 0, "threads": os.cpu_count() or 0, "freq_mhz": 0}

    # ── Disk ──────────────────────────────────────────────────────────────

    def get_disk(self) -> Dict[str, Any]:
        try:
            import psutil
            disk = psutil.disk_usage("/")
            return {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent,
            }
        except ImportError:
            pass
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}

    # ── Network ───────────────────────────────────────────────────────────

    def _get_net_io(self) -> Dict[str, int]:
        try:
            import psutil
            net = psutil.net_io_counters()
            return {"bytes_sent": net.bytes_sent, "bytes_recv": net.bytes_recv}
        except ImportError:
            pass
        return {"bytes_sent": 0, "bytes_recv": 0}

    def get_network(self) -> Dict[str, Any]:
        current = self._get_net_io()
        dt = time.time() - self._last_poll if self._last_poll > 0 else 1
        if dt < 0.1:
            dt = 0.1
        sent_rate = (current["bytes_sent"] - self._prev_net.get("bytes_sent", 0)) / dt
        recv_rate = (current["bytes_recv"] - self._prev_net.get("bytes_recv", 0)) / dt
        self._prev_net = current
        return {
            "bytes_sent": current["bytes_sent"],
            "bytes_recv": current["bytes_recv"],
            "sent_mb": round(current["bytes_sent"] / (1024**2), 2),
            "recv_mb": round(current["bytes_recv"] / (1024**2), 2),
            "sent_rate_kb": round(sent_rate / 1024, 1),
            "recv_rate_kb": round(recv_rate / 1024, 1),
        }

    # ── Uptime ────────────────────────────────────────────────────────────

    def get_uptime(self) -> Dict[str, Any]:
        uptime_sec = time.time() - self._start_time
        days = int(uptime_sec // 86400)
        hours = int((uptime_sec % 86400) // 3600)
        mins = int((uptime_sec % 3600) // 60)
        return {
            "seconds": round(uptime_sec),
            "formatted": f"{days}d {hours}h {mins}m",
        }

    # ── Process Count ─────────────────────────────────────────────────────

    def get_process_count(self) -> int:
        try:
            import psutil
            return len(psutil.pids())
        except ImportError:
            pass
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=5)
                return len(result.stdout.strip().split("\n")) - 3
        except Exception:
            pass
        return 0

    # ── Get All ───────────────────────────────────────────────────────────

    def get_all(self) -> Dict[str, Any]:
        now = time.time()
        if now - self._last_poll < self.poll_interval and self._cache:
            return self._cache
        with self._lock:
            now = time.time()
            if now - self._last_poll < self.poll_interval and self._cache:
                return self._cache
            data = {
                "timestamp": datetime.now().isoformat(),
                "cpu": self.get_cpu_usage(),
                "cpu_temp": self.get_cpu_temp(),
                "gpu_temp": self.get_gpu_temp(),
                "gpu_vram": self.get_gpu_vram(),
                "ram": self.get_ram(),
                "disk": self.get_disk(),
                "network": self.get_network(),
                "uptime": self.get_uptime(),
                "processes": self.get_process_count(),
                "hostname": platform.node(),
                "os": f"{platform.system()} {platform.release()}",
                "python": platform.python_version(),
            }
            self._cache = data
            self._last_poll = now
            return data


# Singleton
system_monitor = SystemMonitor()
