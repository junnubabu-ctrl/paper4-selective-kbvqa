from __future__ import annotations
import json, os, platform, shutil
from pathlib import Path

def collect_environment() -> dict:
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "storage": dict(zip(("total", "used", "free"), shutil.disk_usage("/"))),
    }
    try:
        import psutil
        info["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        info["ram_bytes"] = None
    try:
        import torch
        info.update({
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu_count": int(torch.cuda.device_count()),
            "gpus": [],
        })
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": p.name, "vram_bytes": int(p.total_memory)})
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info

def write_environment(path: str | Path) -> dict:
    info = collect_environment()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
