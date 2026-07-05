from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from quadrotor_rl.evaluation.checkpoint_selector import resolve_checkpoint_path
from quadrotor_rl.export.export_onnx import (
    EXPORT_CHECKPOINT_NAMES,
    build_export_module,
    load_policy_io_for_checkpoint,
    torch_load,
)


def compare_torch_onnx(
    checkpoint: str | Path,
    onnx_path: str | Path,
    policy_io_path: str | Path | None = None,
    batch_size: int = 8,
) -> Dict[str, float | str | bool]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is required for Torch/ONNX comparison") from exc

    checkpoint_path = resolve_checkpoint_path(checkpoint, preferred_names=EXPORT_CHECKPOINT_NAMES)
    checkpoint_data = torch_load(checkpoint_path)
    policy_io = load_policy_io_for_checkpoint(checkpoint_path, policy_io_path)
    export_module, export_info = build_export_module(checkpoint_path, checkpoint_data, policy_io)

    obs_dim = int(export_info["actor_obs_dim"])
    obs = torch.randn(int(batch_size), obs_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_out = export_module(obs).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {session.get_inputs()[0].name: obs.cpu().numpy()})[0]
    diff = np.asarray(torch_out - onnx_out, dtype=np.float64)
    return {
        "task_name": str(export_info["task_name"]),
        "obs_dim": float(obs_dim),
        "action_dim": float(export_info["action_dim"]),
        "normalizer_embedded": bool(export_info["normalizer_embedded"]),
        "onnx_input_convention": str(export_info["onnx_input_convention"]),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX quadrotor policy outputs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--policy-io", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    report = compare_torch_onnx(args.checkpoint, args.onnx, policy_io_path=args.policy_io or None, batch_size=args.batch_size)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
