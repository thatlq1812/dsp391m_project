"""Console reporting helpers shared across scripts."""

from __future__ import annotations

import platform
from typing import Dict, Optional

import torch

from .config_loader import TrainingConfig


def print_section(title: str) -> None:
    """Print a formatted section header to stdout."""

    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


def log_device_summary(device: torch.device, training_cfg: TrainingConfig) -> Optional[str]:
    """Describe the compute resource in use and return the GPU name if available."""

    if device.type == "cuda":
        cuda_index = device.index or 0
        gpu_name = torch.cuda.get_device_name(cuda_index)
        gpu_capability = torch.cuda.get_device_capability(cuda_index)
        total_mem_gb = torch.cuda.get_device_properties(cuda_index).total_memory / (1024 ** 3)
        allow_tf32 = training_cfg.matmul_precision.lower() != "high"
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        print(
            f"Using CUDA device: {gpu_name} ({total_mem_gb:.1f} GB, CC {gpu_capability[0]}.{gpu_capability[1]})"
        )
        print(f"TF32 enabled: {allow_tf32}")
        return gpu_name
    else:
        cpu_name = platform.processor() or "generic CPU"
        print(f"Using CPU device: {cpu_name}")
        return None


def log_dataloader_settings(training_cfg: TrainingConfig) -> None:
    """Emit a concise summary of dataloader tuning parameters."""

    print(
        "Resolved dataloader settings -> "
        f"workers={training_cfg.num_workers}, pin_memory={training_cfg.pin_memory}, "
        f"persistent_workers={training_cfg.persistent_workers}, prefetch_factor={training_cfg.prefetch_factor}"
    )


def format_metric_line(metrics: Dict[str, float]) -> str:
    """Create a human-friendly string for metric dictionaries."""

    parts = [f"{key}: {value:.4f}" for key, value in metrics.items()]
    return " | ".join(parts)
