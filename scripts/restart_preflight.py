#!/usr/bin/env python3
"""Read-only restart readiness gate. Does not download data/models or substitute fixtures."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from paper4_kbvqa.data.restart_pilot import validate_prepared_pilot


def readiness(pilot_dir: Path, min_free_gib: float = 20.0, quantized: bool = True) -> dict:
    pilot_dir = Path(pilot_dir).resolve()
    checks = []

    def record(name, passed, details):
        checks.append({"check": name, "passed": bool(passed), "details": details})

    distributions = [("numpy", "numpy"), ("PIL", "Pillow"), ("sklearn", "scikit-learn"),
                     ("yaml", "PyYAML"), ("torch", "torch"), ("transformers", "transformers"),
                     ("accelerate", "accelerate")]
    if quantized:
        distributions.append(("bitsandbytes", "bitsandbytes"))
    modules = {}
    for module_name, distribution in distributions:
        try:
            modules[module_name] = importlib.import_module(module_name)
            version = importlib.metadata.version(distribution)
            record("dependency_" + module_name, True, {"version": version})
        except Exception as error:
            record("dependency_" + module_name, False, {"error": f"{type(error).__name__}: {error}"})
    torch = modules.get("torch")
    devices = []
    try:
        available = bool(torch and torch.cuda.is_available())
        count = torch.cuda.device_count() if available else 0
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": properties.name,
                            "total_memory_bytes": int(properties.total_memory),
                            "compute_capability": [properties.major, properties.minor]})
        record("cuda", available and count > 0, {"available": available, "device_count": count,
               "devices": devices, "note": "Capacity recorded; one-image execution decides actual model feasibility. No speculative VRAM minimum."})
    except Exception as error:
        record("cuda", False, {"error": f"{type(error).__name__}: {error}"})
    try:
        transformers = modules.get("transformers")
        if transformers is None:
            raise RuntimeError("transformers unavailable")
        for name in ["Qwen2_5_VLForConditionalGeneration", "AutoProcessor", "AutoModelForCausalLM", "AutoTokenizer"]:
            getattr(transformers, name)
        if quantized:
            getattr(transformers, "BitsAndBytesConfig")
        record("qwen_model_classes", True, {"checked": "import availability only; no model loaded/downloaded"})
    except Exception as error:
        record("qwen_model_classes", False, {"error": f"{type(error).__name__}: {error}"})
    try:
        pins = json.loads((ROOT / "configs/models_20260915.json").read_text(encoding="utf-8"))
        expected = {"generator": "Qwen/Qwen2.5-VL-3B-Instruct", "qwen": "Qwen/Qwen2.5-1.5B-Instruct"}
        for key, model_id in expected.items():
            entry = pins[key]
            if entry.get("model_id") != model_id or not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", "")):
                raise ValueError(f"Missing immutable pin for {key}")
        record("model_pins", True, {key: {field: pins[key][field] for field in ["model_id", "revision"]} for key in expected})
    except Exception as error:
        record("model_pins", False, {"error": f"{type(error).__name__}: {error}"})
    try:
        validated = validate_prepared_pilot(pilot_dir, require_images=True)
        record("pilot_data_and_images", True, {"questions": len(validated["rows"]), "source_split": "train",
               "partition_role": "development_only", "source_manifest_sha256": validated["source_manifest_sha256"],
               "image_inventory_sha256": validated["image_inventory_sha256"], "images_decoded_and_hash_checked": True})
    except Exception as error:
        record("pilot_data_and_images", False, {"error": f"{type(error).__name__}: {error}"})
    disk_path = pilot_dir
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    try:
        usage = shutil.disk_usage(disk_path)
        record("disk", usage.free >= min_free_gib * 1024**3,
               {"free_bytes": usage.free, "minimum_free_gib": min_free_gib,
                "note": "Explicit operational planning threshold, not a measured complete-study storage requirement"})
    except Exception as error:
        record("disk", False, {"error": f"{type(error).__name__}: {error}"})
    blocked = [check["check"] for check in checks if not check["passed"]]
    return {"schema_version": 1, "status": "BLOCKED" if blocked else "READY_FOR_ONE_IMAGE_GATE",
            "checked_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "pilot_dir": str(pilot_dir), "quantized": quantized,
            "checks": checks, "blockers": blocked, "model_weights_loaded": False,
            "downloads_performed": False, "benchmark_results": False, "calibration_performed": False,
            "next_gate": "One-image genuine CUDA generator/critic run; then fixed-candidate 50-question development pilot"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=20.0, help="Explicit disk planning gate; override recorded in output")
    parser.add_argument("--unquantized", action="store_true", help="Do not require bitsandbytes; weights still not loaded")
    args = parser.parse_args()
    if args.min_free_gib < 0:
        parser.error("min-free-gib cannot be negative")
    report = readiness(args.pilot_dir, args.min_free_gib, not args.unquantized)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.out)
    print(json.dumps({"status": report["status"], "blockers": report["blockers"], "out": str(args.out.resolve())}, indent=2))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
