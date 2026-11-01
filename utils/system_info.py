"""System information and hardware acceleration detector."""

import platform
import psutil
from typing import Dict, Any

def get_system_metrics() -> Dict[str, Any]:
    """Inspect system architecture, RAM, and hardware acceleration support."""
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    proc_mem = proc.memory_info().rss / (1024 * 1024)  # MB

    metrics = {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "processor": platform.processor() or "Apple Silicon",
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
        "ram_usage_percent": vm.percent,
        "process_ram_mb": round(proc_mem, 1),
        "mps_available": False,
        "mps_built": False,
        "cuda_available": False,
        "torch_version": "Not loaded",
        "recommended_device": "cpu"
    }

    try:
        import torch
        metrics["torch_version"] = torch.__version__
        metrics["cuda_available"] = torch.cuda.is_available()
        metrics["mps_built"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        metrics["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        
        # On Apple Silicon M2 with 8GB RAM, CPU is ultra stable and reliable.
        # MPS can be enabled optionally for batch/larger inference, but for 8GB RAM
        # CPU provides consistent, zero-overhead memory paging.
        if metrics["mps_available"]:
            metrics["recommended_device"] = "mps"
        else:
            metrics["recommended_device"] = "cpu"
    except Exception as e:
        metrics["torch_error"] = str(e)

    return metrics

def check_memory_headroom(min_free_mb: float = 400.0) -> bool:
    """Check if system has sufficient free memory to proceed with inference."""
    vm = psutil.virtual_memory()
    free_mb = vm.available / (1024 * 1024)
    return free_mb >= min_free_mb
