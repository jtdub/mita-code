"""Tests for hardware detection functions (subprocess-based)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mita.models.hardware import (
    _detect_cpu_name,
    _detect_linux_gpu,
    _detect_macos_discrete_gpu,
    _detect_macos_gpu,
    _get_apple_chip_name,
    _is_apple_silicon,
    detect_hardware,
)


class TestDetectHardware:
    @patch("mita.models.hardware.platform")
    @patch("mita.models.hardware.psutil")
    @patch("mita.models.hardware._detect_cpu_name", return_value="Test CPU")
    @patch("mita.models.hardware._detect_macos_gpu")
    def test_darwin(
        self,
        mock_gpu: MagicMock,
        mock_cpu: MagicMock,
        mock_psutil: MagicMock,
        mock_platform: MagicMock,
    ) -> None:
        mock_platform.system.return_value = "Darwin"
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3)
        mock_psutil.cpu_count.return_value = 8
        mock_gpu.return_value = (True, True, [])
        hw = detect_hardware()
        assert hw.os == "darwin"
        assert hw.ram_gb == 16.0
        assert hw.cpu_cores == 8

    @patch("mita.models.hardware.platform")
    @patch("mita.models.hardware.psutil")
    @patch("mita.models.hardware._detect_cpu_name", return_value="AMD CPU")
    @patch("mita.models.hardware._detect_linux_gpu", return_value=[])
    def test_linux(
        self,
        mock_gpu: MagicMock,
        mock_cpu: MagicMock,
        mock_psutil: MagicMock,
        mock_platform: MagicMock,
    ) -> None:
        mock_platform.system.return_value = "Linux"
        mock_psutil.virtual_memory.return_value = MagicMock(total=32 * 1024**3)
        mock_psutil.cpu_count.return_value = 16
        hw = detect_hardware()
        assert hw.os == "linux"
        assert hw.cpu_cores == 16

    @patch("mita.models.hardware.platform")
    @patch("mita.models.hardware.psutil")
    @patch("mita.models.hardware._detect_cpu_name", return_value="Unknown CPU")
    def test_windows(
        self, mock_cpu: MagicMock, mock_psutil: MagicMock, mock_platform: MagicMock
    ) -> None:
        mock_platform.system.return_value = "Windows"
        mock_psutil.virtual_memory.return_value = MagicMock(total=8 * 1024**3)
        mock_psutil.cpu_count.return_value = 4
        hw = detect_hardware()
        assert hw.os == "windows"


class TestDetectCpuName:
    @patch("mita.models.hardware.platform")
    @patch("mita.models.hardware.subprocess.run")
    def test_darwin_brand_string(self, mock_run: MagicMock, mock_platform: MagicMock) -> None:
        mock_platform.system.return_value = "Darwin"
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M2 Pro\n")
        assert _detect_cpu_name() == "Apple M2 Pro"

    @patch("mita.models.hardware.platform")
    @patch("mita.models.hardware.subprocess.run")
    def test_darwin_chip_fallback(self, mock_run: MagicMock, mock_platform: MagicMock) -> None:
        mock_platform.system.return_value = "Darwin"
        mock_run.side_effect = [
            MagicMock(returncode=1, stdout=""),  # brand_string fails
            MagicMock(returncode=0, stdout="Apple M1\n"),  # hw.chip succeeds
        ]
        assert _detect_cpu_name() == "Apple M1"

    @patch("mita.models.hardware.platform")
    @patch("mita.models.hardware.subprocess.run")
    def test_darwin_all_fail(self, mock_run: MagicMock, mock_platform: MagicMock) -> None:
        mock_platform.system.return_value = "Darwin"
        mock_platform.processor.return_value = "arm"
        mock_run.side_effect = FileNotFoundError
        assert _detect_cpu_name() == "arm"

    @patch("mita.models.hardware.platform")
    @patch("builtins.open")
    def test_linux_proc_cpuinfo(self, mock_open: MagicMock, mock_platform: MagicMock) -> None:
        mock_platform.system.return_value = "Linux"
        mock_open.return_value.__enter__ = lambda s: iter(
            ["processor\t: 0\n", "model name\t: Intel Core i7-12700K\n"]
        )
        mock_open.return_value.__exit__ = lambda *a: None
        assert _detect_cpu_name() == "Intel Core i7-12700K"

    @patch("mita.models.hardware.platform")
    def test_fallback(self, mock_platform: MagicMock) -> None:
        mock_platform.system.return_value = "Haiku"
        mock_platform.processor.return_value = ""
        assert _detect_cpu_name() == "Unknown CPU"


class TestIsAppleSilicon:
    @patch("mita.models.hardware.platform.machine", return_value="arm64")
    def test_arm64(self, _: MagicMock) -> None:
        assert _is_apple_silicon() is True

    @patch("mita.models.hardware.platform.machine", return_value="x86_64")
    @patch("mita.models.hardware.subprocess.run")
    def test_rosetta(self, mock_run: MagicMock, _: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="1\n")
        assert _is_apple_silicon() is True

    @patch("mita.models.hardware.platform.machine", return_value="x86_64")
    @patch("mita.models.hardware.subprocess.run")
    def test_intel(self, mock_run: MagicMock, _: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="0\n")
        assert _is_apple_silicon() is False

    @patch("mita.models.hardware.platform.machine", return_value="x86_64")
    @patch("mita.models.hardware.subprocess.run", side_effect=FileNotFoundError)
    def test_no_sysctl(self, _run: MagicMock, _mach: MagicMock) -> None:
        assert _is_apple_silicon() is False


class TestDetectMacosGpu:
    @patch("mita.models.hardware._is_apple_silicon", return_value=True)
    @patch("mita.models.hardware._get_apple_chip_name", return_value="Apple M2 Max")
    def test_apple_silicon(self, _name: MagicMock, _is_as: MagicMock) -> None:
        is_as, unified, gpus = _detect_macos_gpu(32.0)
        assert is_as is True
        assert unified is True
        assert len(gpus) == 1
        assert gpus[0].vendor == "apple"
        assert gpus[0].vram_gb == 32.0

    @patch("mita.models.hardware._is_apple_silicon", return_value=False)
    @patch("mita.models.hardware._detect_macos_discrete_gpu", return_value=[])
    def test_intel_mac(self, _disc: MagicMock, _is_as: MagicMock) -> None:
        is_as, unified, gpus = _detect_macos_gpu(16.0)
        assert is_as is False
        assert unified is False


class TestGetAppleChipName:
    @patch("mita.models.hardware.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M3 Pro\n")
        assert _get_apple_chip_name() == "Apple M3 Pro"

    @patch("mita.models.hardware.subprocess.run", side_effect=FileNotFoundError)
    def test_fallback(self, _: MagicMock) -> None:
        assert _get_apple_chip_name() == "Apple Silicon"


class TestDetectMacosDiscreteGpu:
    @patch("mita.models.hardware.subprocess.run")
    def test_with_gpu(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "Graphics/Displays:\n\n"
                "    Chipset Model: AMD Radeon Pro 5500M\n"
                "      VRAM (Total): 8 GB\n"
            ),
        )
        gpus = _detect_macos_discrete_gpu()
        assert len(gpus) == 1
        assert gpus[0].name == "AMD Radeon Pro 5500M"
        assert gpus[0].vram_gb == 8.0

    @patch("mita.models.hardware.subprocess.run", side_effect=FileNotFoundError)
    def test_no_profiler(self, _: MagicMock) -> None:
        assert _detect_macos_discrete_gpu() == []


class TestDetectLinuxGpu:
    @patch("mita.models.hardware.subprocess.run")
    def test_nvidia(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 4090, 24576\n")
        gpus = _detect_linux_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "nvidia"
        assert gpus[0].vram_gb == 24.0

    @patch("mita.models.hardware.subprocess.run")
    def test_amd_rocm(self, mock_run: MagicMock) -> None:
        # nvidia-smi fails, rocm-smi works
        mock_run.side_effect = [
            MagicMock(returncode=1),  # nvidia-smi fails
            MagicMock(returncode=0, stdout="header\nGPU0, 8589934592\n"),
        ]
        gpus = _detect_linux_gpu()
        assert len(gpus) == 1
        assert gpus[0].vendor == "amd"

    @patch("mita.models.hardware.subprocess.run", side_effect=FileNotFoundError)
    def test_no_gpu(self, _: MagicMock) -> None:
        assert _detect_linux_gpu() == []
