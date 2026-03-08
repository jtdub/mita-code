"""Tests for hardware detection."""

from mita.models.hardware import (
    GPUInfo,
    HardwareInfo,
    _guess_gpu_vendor,
    _parse_vram_string,
)


class TestHardwareInfo:
    def test_available_vram_unified_memory(self) -> None:
        """Apple Silicon unified memory should use 75% of RAM."""
        hw = HardwareInfo(
            ram_gb=32,
            cpu_cores=10,
            cpu_name="Apple M2 Pro",
            gpus=[GPUInfo(name="Apple M2 Pro", vram_gb=32, vendor="apple")],
            os="darwin",
            apple_silicon=True,
            unified_memory=True,
        )
        assert hw.available_vram_gb == 24.0  # 32 * 0.75

    def test_available_vram_discrete_gpu(self) -> None:
        """Discrete GPU should use the GPU VRAM."""
        hw = HardwareInfo(
            ram_gb=32,
            cpu_cores=8,
            cpu_name="Intel i7",
            gpus=[GPUInfo(name="RTX 4090", vram_gb=24, vendor="nvidia")],
            os="linux",
        )
        assert hw.available_vram_gb == 24.0

    def test_available_vram_multi_gpu(self) -> None:
        """Multiple GPUs should use the largest VRAM."""
        hw = HardwareInfo(
            ram_gb=64,
            cpu_cores=16,
            cpu_name="AMD Threadripper",
            gpus=[
                GPUInfo(name="RTX 3090", vram_gb=24, vendor="nvidia"),
                GPUInfo(name="RTX 3080", vram_gb=10, vendor="nvidia"),
            ],
            os="linux",
        )
        assert hw.available_vram_gb == 24.0

    def test_available_vram_no_gpu(self) -> None:
        """No GPU should use RAM minus 4GB reserve."""
        hw = HardwareInfo(
            ram_gb=16,
            cpu_cores=4,
            cpu_name="Intel i5",
            os="linux",
        )
        assert hw.available_vram_gb == 12.0  # 16 - 4

    def test_available_vram_low_ram_no_gpu(self) -> None:
        """Very low RAM with no GPU should not go negative."""
        hw = HardwareInfo(
            ram_gb=2,
            cpu_cores=2,
            cpu_name="Intel Celeron",
            os="linux",
        )
        assert hw.available_vram_gb == 0


class TestParseVramString:
    def test_gb(self) -> None:
        assert _parse_vram_string("8 GB") == 8.0

    def test_mb(self) -> None:
        assert _parse_vram_string("4096 MB") == 4.0

    def test_no_unit(self) -> None:
        assert _parse_vram_string("unknown") == 0.0

    def test_empty(self) -> None:
        assert _parse_vram_string("") == 0.0


class TestGuessGpuVendor:
    def test_nvidia(self) -> None:
        assert _guess_gpu_vendor("NVIDIA GeForce RTX 4090") == "nvidia"

    def test_amd(self) -> None:
        assert _guess_gpu_vendor("AMD Radeon RX 7900") == "amd"

    def test_intel(self) -> None:
        assert _guess_gpu_vendor("Intel UHD Graphics 630") == "intel"

    def test_unknown(self) -> None:
        assert _guess_gpu_vendor("Some GPU") == "unknown"
