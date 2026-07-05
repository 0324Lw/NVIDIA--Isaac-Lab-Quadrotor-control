from __future__ import annotations

import json
from pathlib import Path

import torch

from quadrotor_rl.export.export_onnx import ObservationNormalizedPolicy, export_policy_onnx


def _task1_policy_state() -> dict[str, torch.Tensor]:
    return {
        "net.0.weight": torch.randn(256, 108) * 0.01,
        "net.0.bias": torch.zeros(256),
        "net.2.weight": torch.randn(256, 256) * 0.01,
        "net.2.bias": torch.zeros(256),
        "net.4.weight": torch.randn(128, 256) * 0.01,
        "net.4.bias": torch.zeros(128),
        "net.6.weight": torch.randn(4, 128) * 0.01,
        "net.6.bias": torch.zeros(4),
        "log_std_parameter": torch.zeros(4),
    }


def test_export_prefers_eval_checkpoint_and_ignores_log_std(tmp_path: Path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    policy_state = _task1_policy_state()
    torch.save({"policy": {"unsupported.weight": torch.ones(1)}}, checkpoint_dir / "quadrotor_task1_skrl_model.pt")
    torch.save(
        {
            "policy": policy_state,
            "actor_obs_norm": {
                "running_mean": torch.zeros(108),
                "running_variance": torch.ones(108),
            },
        },
        checkpoint_dir / "quadrotor_task1_model.pt",
    )
    (checkpoint_dir / "policy_io.json").write_text(
        json.dumps(
            {
                "task_name": "quadrotor_task1_hover_stabilization",
                "actor_obs_dim": 108,
                "critic_obs_dim": 108,
                "action_dim": 4,
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_export(model, dummy, output_path, **kwargs):
        captured["model_type"] = type(model).__name__
        captured["dummy_shape"] = tuple(dummy.shape)
        captured["is_normalized_policy"] = isinstance(model, ObservationNormalizedPolicy)
        Path(output_path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    output = tmp_path / "policy.onnx"
    export_policy_onnx(checkpoint_dir, output)

    sidecar = json.loads(output.with_suffix(".export.json").read_text(encoding="utf-8"))
    assert output.exists()
    assert sidecar["checkpoint"].endswith("quadrotor_task1_model.pt")
    assert sidecar["export"]["dropped_tensors"] == ["log_std_parameter"]
    assert sidecar["export"]["normalizer_embedded"] is True
    assert sidecar["export"]["onnx_input_convention"] == "raw_observation"
    assert captured["dummy_shape"] == (1, 108)
    assert captured["is_normalized_policy"] is True
