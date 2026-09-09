from __future__ import annotations

import shutil
import subprocess
from typing import Any


def gpu_snapshot() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"ok": False, "available": False, "error": "nvidia-smi not found"}
    try:
        query = "name,memory.total,memory.used,utilization.gpu,driver_version"
        result = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10, check=True)
        values = [v.strip() for v in result.stdout.strip().split(",")]
        if len(values) < 5:
            raise RuntimeError("Unexpected nvidia-smi output")
        return {"ok": True, "available": True, "name": values[0], "memory_total_mb": int(float(values[1])), "memory_used_mb": int(float(values[2])), "utilization_percent": float(values[3]), "driver": values[4]}
    except Exception as exc:
        return {"ok": False, "available": False, "error": str(exc)}
