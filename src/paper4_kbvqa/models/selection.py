from __future__ import annotations

def select_model(environment: dict) -> dict:
    """Conservative policy; measured peak VRAM must be recorded during real GPU execution."""
    cuda=bool(environment.get("cuda_available")); gpus=environment.get("gpus",[]) or []
    if not cuda or not gpus:
        return {"model":"Qwen/Qwen2.5-VL-3B-Instruct","execution":"NOT_EXECUTED_NO_GPU","quantization":"4bit_on_supported_gpu","rationale":"Provisional compact research VLM; current session has no GPU."}
    min_vram=min(g.get("vram_bytes",0) for g in gpus)
    if min_vram and min_vram < 12*1024**3:
        return {"model":"Qwen/Qwen2.5-VL-3B-Instruct","execution":"gpu","quantization":"4bit","rationale":"Compact model for limited-VRAM free GPU."}
    return {"model":"Qwen/Qwen2.5-VL-3B-Instruct","execution":"gpu","quantization":"bf16_or_4bit","rationale":"Primary reproducible research model; benchmark peak VRAM before locking precision."}
