# -*- coding: utf-8 -*-
"""B4Q-V1 capability probe — no network installs; create-only matrix."""
from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Docs/results/ms_sta_v6_b4q/B4Q_V1_CAPABILITY_MATRIX.json"


def has_mod(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"refuse_overwrite:{OUT}")
    if not torch.cuda.is_available():
        raise SystemExit("cuda_required")
    props = torch.cuda.get_device_properties(0)
    compile_ok = False
    compile_error = None
    try:
        if hasattr(torch, "compile"):

            @torch.compile
            def _micro(x: torch.Tensor) -> torch.Tensor:
                return x * 2.0 + 1.0

            x = torch.randn(8, 8, device="cuda")
            y = _micro(x)
            torch.cuda.synchronize()
            compile_ok = bool(torch.isfinite(y).all().item())
    except Exception as exc:  # noqa: BLE001 — probe must record failures
        compile_error = repr(exc)

    amp_ok = False
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            a = torch.randn(64, 64, device="cuda")
            b = a @ a.T
            torch.cuda.synchronize()
            amp_ok = bool(torch.isfinite(b).all().item())
    except Exception as exc:  # noqa: BLE001
        amp_ok = False
        amp_error = repr(exc)
    else:
        amp_error = None

    bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    matrix = {
        "schema": "ms_sta_v6_b4q_v1_capability_matrix_v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": [int(props.major), int(props.minor)],
            "total_vram_mb": round(props.total_memory / (1024 * 1024), 1),
        },
        "native": {
            "inference_mode": True,
            "autocast_fp16_matmul_finite": amp_ok,
            "autocast_fp16_error": amp_error,
            "bf16_supported": bf16,
            "channels_last": True,
            "torch_compile_micrograph_ok": compile_ok,
            "torch_compile_error": compile_error,
        },
        "optional_modules_present_in_current_env": {
            "onnx": has_mod("onnx"),
            "onnxruntime": has_mod("onnxruntime"),
            "tensorrt": has_mod("tensorrt"),
            "torch_tensorrt": has_mod("torch_tensorrt"),
            "torchao": has_mod("torchao"),
            "modelopt": has_mod("modelopt") or has_mod("nvidia_modelopt"),
        },
        "decisions": {
            "continue_native_fp16_arms": True,
            "tensorrt_fp16": "NOT_RUN_ENVIRONMENT_UNAVAILABLE"
            if not (has_mod("tensorrt") or has_mod("torch_tensorrt"))
            else "CAPABILITY_PRESENT_NEEDS_APPROVAL",
            "int8_ptq": "NOT_RUN_ENVIRONMENT_UNAVAILABLE_PENDING_PROBE",
            "forbid_network_install_without_user_approval": True,
            "proposed_optional_venv": "_venvs/b4q_trt",
        },
        "e4_ready": False,
        "note": "Probe only; does not authorize Q4/Q5 runs or dependency installs.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("CAPABILITY_OK", OUT)
    print("compile", compile_ok, "amp", amp_ok, "trt", matrix["optional_modules_present_in_current_env"]["tensorrt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
