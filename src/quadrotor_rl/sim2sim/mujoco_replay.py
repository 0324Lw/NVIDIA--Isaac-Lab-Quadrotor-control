from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from quadrotor_rl.sim2sim.sim2sim_report import write_sim2sim_report


def run_open_loop_replay(onnx_path: str | Path, rollout_path: str | Path | None = None, report_path: str | Path = "sim2sim_report.md", max_steps: int = 500) -> Dict[str, Any]:
    onnx_path = Path(onnx_path).expanduser().resolve()
    metrics: Dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "max_steps": int(max_steps),
        "status": "PASS",
    }
    if not onnx_path.exists():
        metrics["status"] = "FAIL"
        metrics["reason"] = "ONNX file does not exist"
        write_sim2sim_report(report_path, metrics)
        return metrics

    try:
        import onnxruntime as ort
    except Exception:
        metrics["status"] = "WARN"
        metrics["reason"] = "onnxruntime is not installed; file-level sim2sim check only"
        metrics["onnx_size_bytes"] = onnx_path.stat().st_size
        write_sim2sim_report(report_path, metrics)
        return metrics

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape
    obs_dim = int(input_shape[-1])

    if rollout_path:
        data = np.load(Path(rollout_path), allow_pickle=False)
        obs = np.asarray(data["obs"], dtype=np.float32).reshape(-1, obs_dim)
        obs = obs[: int(max_steps)]
    else:
        obs = np.zeros((min(int(max_steps), 8), obs_dim), dtype=np.float32)

    actions = session.run(None, {session.get_inputs()[0].name: obs})[0]
    metrics.update(
        {
            "obs_dim": int(obs_dim),
            "num_replay_steps": int(obs.shape[0]),
            "action_dim": int(actions.shape[-1]),
            "action_mean": float(np.mean(actions)),
            "action_abs_mean": float(np.mean(np.abs(actions))),
            "action_max_abs": float(np.max(np.abs(actions))),
            "contains_nan": bool(np.isnan(actions).any()),
        }
    )
    if metrics["contains_nan"]:
        metrics["status"] = "FAIL"
        metrics["reason"] = "ONNX policy produced NaN action"
    write_sim2sim_report(report_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal quadrotor ONNX sim2sim replay check")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--rollout", default="")
    parser.add_argument("--report", default="sim2sim_report.md")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    metrics = run_open_loop_replay(args.onnx, rollout_path=args.rollout or None, report_path=args.report, max_steps=args.max_steps)
    print(metrics)


if __name__ == "__main__":
    main()
