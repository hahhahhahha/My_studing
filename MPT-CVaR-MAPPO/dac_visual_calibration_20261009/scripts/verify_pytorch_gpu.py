"""CPU/CUDA/torchvision/NVML diagnostics, not visual-workload measurements."""
from __future__ import annotations

from common import BASE, WINDOWS_RUNTIME_RESTORED, save

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
import platform
import subprocess
import sys
import traceback


def verify():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default=str(BASE / "protocol/PYTORCH_GPU_INITIALIZATION.json"))
    options = parser.parse_args()
    result = {
        "datetime": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "restored_process_variables": WINDOWS_RUNTIME_RESTORED,
        "required_process_variables": {key: os.environ.get(key) for key in
                                       ("SystemRoot", "windir", "ProgramFiles", "ProgramW6432",
                                        "USERNAME", "TEMP", "TMP")},
        "persistent_environment_modified": False,
        "packages_or_drivers_modified": False,
        "visual_service_measurements": False,
        "passed": False,
    }
    try:
        import socket
        import asyncio
        with socket.socket():
            pass
        result["socket_asyncio_ok"] = True

        import torch
        import torchvision
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261010)
        result["torch"] = torch.__version__
        result["torchvision"] = torchvision.__version__
        result["pytorch_cuda_runtime"] = torch.version.cuda

        a = torch.randn(64, 64, dtype=torch.float32)
        torch.testing.assert_close(a @ torch.eye(64), a)
        result["cpu_kernel_ok"] = True
        if not torch.cuda.is_available() or not torch.cuda.device_count():
            raise RuntimeError("CUDA device is not available")
        device = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": device.name, "total_memory_bytes": device.total_memory,
                         "compute_capability": [device.major, device.minor],
                         "device_count": torch.cuda.device_count()}
        reference = a @ a.T
        gpu_a = a.to("cuda")
        actual = gpu_a @ gpu_a.T
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
        # Include autograd, not just device enumeration or allocation.
        train_a = gpu_a.detach().clone().requires_grad_(True)
        train_a.square().sum().backward()
        torch.cuda.synchronize()
        torch.testing.assert_close(train_a.grad.cpu(), 2 * a)
        result["cuda_matrix_and_autograd_ok"] = True

        # Random initialization here is only a runtime witness; no pretrained
        # accuracy or visual service-time claim is made from this test.
        model = torchvision.models.resnet18(weights=None).eval().to("cuda")
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 224, 224, device="cuda"))
        torch.cuda.synchronize()
        if output.shape != (1, 1000) or not torch.isfinite(output).all():
            raise RuntimeError("TorchVision CUDA forward witness failed")
        result["torchvision_resnet18_cuda_forward_ok"] = True

        smi = os.path.join(os.environ["SystemRoot"], "System32", "nvidia-smi.exe")
        probe = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20)
        result["nvidia_smi"] = {"returncode": probe.returncode,
                                "stdout": probe.stdout.strip(), "stderr": probe.stderr.strip()}
        if probe.returncode:
            raise RuntimeError("NVIDIA NVML verification failed")
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    save(options.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(verify())
