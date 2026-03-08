"""Hardware detection: RAM, VRAM, CPU, GPU."""

from __future__ import annotations

import platform
import subprocess

import psutil
from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    """Information about a detected GPU."""

    name: str
    vram_gb: float
    vendor: str  # "nvidia", "amd", "apple"


class HardwareInfo(BaseModel):
    """Detected hardware capabilities."""

    ram_gb: float
    cpu_cores: int
    cpu_name: str
    gpus: list[GPUInfo] = Field(default_factory=list)
    os: str  # "darwin", "linux", "windows"
    apple_silicon: bool = False
    unified_memory: bool = False

    @property
    def available_vram_gb(self) -> float:
        """Effective VRAM available for models.

        On Apple Silicon with unified memory, most of RAM is usable as VRAM.
        On discrete GPUs, use the GPU VRAM.
        Falls back to a conservative fraction of RAM for CPU-only inference.
        """
        if self.unified_memory:
            # Apple Silicon can use ~75% of unified memory for ML
            return self.ram_gb * 0.75
        if self.gpus:
            return max(gpu.vram_gb for gpu in self.gpus)
        # CPU-only: models load into RAM, leave room for OS
        return max(0, self.ram_gb - 4)


def detect_hardware() -> HardwareInfo:
    """Detect hardware capabilities of the current machine."""
    system = platform.system().lower()
    os_name = {"darwin": "darwin", "linux": "linux", "windows": "windows"}.get(system, system)

    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    cpu_name = _detect_cpu_name()

    apple_silicon = False
    unified_memory = False
    gpus: list[GPUInfo] = []

    if os_name == "darwin":
        apple_silicon, unified_memory, gpus = _detect_macos_gpu(ram_gb)
    elif os_name == "linux":
        gpus = _detect_linux_gpu()

    return HardwareInfo(
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        cpu_name=cpu_name,
        gpus=gpus,
        os=os_name,
        apple_silicon=apple_silicon,
        unified_memory=unified_memory,
    )


def _detect_cpu_name() -> str:
    """Detect the CPU model name."""
    system = platform.system().lower()

    if system == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        # Apple Silicon doesn't have brand_string, use chip name
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.chip"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    elif system == "linux":
        try:
            result = subprocess.run(
                ["grep", "-m1", "model name", "/proc/cpuinfo"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and ":" in result.stdout:
                return result.stdout.split(":", 1)[1].strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return platform.processor() or "Unknown CPU"


def _is_apple_silicon() -> bool:
    """Detect Apple Silicon, even under Rosetta emulation."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return True
    # Under Rosetta, platform.machine() returns x86_64.
    # Check sysctl for the actual hardware.
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False


def _detect_macos_gpu(ram_gb: float) -> tuple[bool, bool, list[GPUInfo]]:
    """Detect GPU on macOS. Returns (apple_silicon, unified_memory, gpus)."""
    apple_silicon = _is_apple_silicon()

    if apple_silicon:
        # Apple Silicon has unified memory — RAM is shared with GPU
        chip_name = _get_apple_chip_name()
        return (
            True,
            True,
            [
                GPUInfo(
                    name=chip_name,
                    vram_gb=ram_gb,  # Unified — all RAM is addressable
                    vendor="apple",
                )
            ],
        )

    # Intel Mac — try to detect discrete GPU via system_profiler
    gpus = _detect_macos_discrete_gpu()
    return False, False, gpus


def _get_apple_chip_name() -> str:
    """Get the Apple Silicon chip name (e.g., 'Apple M2 Pro')."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "Apple Silicon"


def _detect_macos_discrete_gpu() -> list[GPUInfo]:
    """Detect discrete GPUs on Intel Macs via system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-detailLevel", "basic"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus: list[GPUInfo] = []
        current_name = ""
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Chipset Model:"):
                current_name = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("VRAM") and current_name:
                vram_str = stripped.split(":", 1)[1].strip()
                vram_gb = _parse_vram_string(vram_str)
                vendor = _guess_gpu_vendor(current_name)
                gpus.append(GPUInfo(name=current_name, vram_gb=vram_gb, vendor=vendor))
                current_name = ""
        return gpus
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def _detect_linux_gpu() -> list[GPUInfo]:
    """Detect NVIDIA GPUs on Linux via nvidia-smi."""
    gpus: list[GPUInfo] = []

    # Try nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    vram_mb = float(parts[1].strip())
                    gpus.append(
                        GPUInfo(name=name, vram_gb=round(vram_mb / 1024, 1), vendor="nvidia")
                    )
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    # Try AMD ROCm
    if not gpus:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines()[1:]:  # skip header
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            vram_bytes = int(parts[1].strip())
                            gpus.append(
                                GPUInfo(
                                    name="AMD GPU",
                                    vram_gb=round(vram_bytes / (1024**3), 1),
                                    vendor="amd",
                                )
                            )
                        except ValueError:
                            pass
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return gpus


def _parse_vram_string(vram_str: str) -> float:
    """Parse VRAM strings like '8 GB', '4096 MB' into GB."""
    vram_str = vram_str.strip().upper()
    try:
        if "GB" in vram_str:
            return float(vram_str.replace("GB", "").strip())
        if "MB" in vram_str:
            return round(float(vram_str.replace("MB", "").strip()) / 1024, 1)
    except ValueError:
        pass
    return 0.0


def _guess_gpu_vendor(name: str) -> str:
    """Guess GPU vendor from the chipset name."""
    name_lower = name.lower()
    if "nvidia" in name_lower or "geforce" in name_lower or "quadro" in name_lower:
        return "nvidia"
    if "amd" in name_lower or "radeon" in name_lower:
        return "amd"
    if "intel" in name_lower:
        return "intel"
    return "unknown"
