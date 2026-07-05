from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from quadrotor_rl.evaluation.checkpoint_selector import resolve_checkpoint_path
from quadrotor_rl.export.policy_io import _json_safe


EXPORT_CHECKPOINT_NAMES = (
    "quadrotor_task1_model.pt",
    "quadrotor_task2_model.pt",
    "quadrotor_task3_model.pt",
    "quadrotor_task4_model.pt",
    "model.pt",
    "checkpoint.pt",
    "quadrotor_task1_skrl_model.pt",
    "quadrotor_task2_skrl_model.pt",
    "quadrotor_task3_skrl_model.pt",
    "quadrotor_task4_skrl_model.pt",
)


class MlpDeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ELU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(action_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = torch.clamp(obs, -10.0, 10.0)
        out = self.net(obs)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(out, -5.0, 5.0)


class TaskVisionEncoder(nn.Module):
    def __init__(
        self,
        depth_dim: int = 4096,
        compact_dim: int = 32,
        cnn_output_dim: int = 256,
        compact_output_dim: int = 128,
    ):
        super().__init__()
        self.depth_dim = int(depth_dim)
        self.compact_dim = int(compact_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            flat = int(self.cnn(dummy).shape[1])
        self.cnn_linear = nn.Sequential(nn.Linear(flat, int(cnn_output_dim)), nn.ELU())
        self.compact_mlp = nn.Sequential(
            nn.Linear(self.compact_dim, 128),
            nn.ELU(),
            nn.Linear(128, int(compact_output_dim)),
            nn.ELU(),
        )
        self.output_dim = int(cnn_output_dim) + int(compact_output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = torch.clamp(x, -10.0, 10.0)
        depth = x[:, : self.depth_dim].reshape(-1, 1, 64, 64)
        compact = x[:, self.depth_dim : self.depth_dim + self.compact_dim]
        depth = torch.clamp(depth, 0.0, 1.0)
        compact = torch.clamp(compact, -10.0, 10.0)
        return torch.cat([self.cnn_linear(self.cnn(depth)), self.compact_mlp(compact)], dim=-1)


class VisionDeterministicPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        del obs_dim
        self.encoder = TaskVisionEncoder(depth_dim=4096, compact_dim=32)
        self.net = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, int(action_dim)),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(self.encoder(obs))
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(out, -5.0, 5.0)


class ObservationNormalizedPolicy(nn.Module):
    def __init__(self, policy: nn.Module, mean: torch.Tensor, variance: torch.Tensor, clip_value: float = 10.0):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", mean.view(1, -1).float())
        self.register_buffer("variance", variance.view(1, -1).float())
        self.clip_value = float(clip_value)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.nan_to_num(obs, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        obs = (obs - self.mean) / torch.sqrt(torch.clamp(self.variance, min=1.0e-8))
        obs = torch.clamp(obs, -self.clip_value, self.clip_value)
        return self.policy(obs)


def torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def extract_policy_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("policy"), dict):
            return dict(checkpoint["policy"])
        if isinstance(checkpoint.get("models"), dict) and isinstance(checkpoint["models"].get("policy"), dict):
            return dict(checkpoint["models"]["policy"])
        if all(isinstance(k, str) for k in checkpoint.keys()) and any(
            k.startswith("net.") or k.startswith("encoder.") for k in checkpoint.keys()
        ):
            return dict(checkpoint)
    raise RuntimeError("checkpoint does not contain a supported policy state_dict")


def remove_unused_policy_tensors(policy_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ignored = {"log_std_parameter"}
    return {key: value for key, value in policy_state.items() if key not in ignored}


def infer_dims(policy_state: Dict[str, torch.Tensor], policy_io: Dict[str, Any] | None) -> Tuple[int, int]:
    obs_dim = int(policy_io.get("actor_obs_dim", 0)) if policy_io else 0
    action_dim = int(policy_io.get("action_dim", 0)) if policy_io else 0
    if obs_dim <= 0:
        first_linear = next((v for k, v in policy_state.items() if k.endswith("0.weight") and v.ndim == 2), None)
        if first_linear is not None:
            obs_dim = int(first_linear.shape[1])
    if action_dim <= 0:
        last_weight = None
        for key, value in policy_state.items():
            if key.endswith("weight") and value.ndim == 2:
                last_weight = value
        if last_weight is not None:
            action_dim = int(last_weight.shape[0])
    if obs_dim <= 0 or action_dim <= 0:
        raise RuntimeError("failed to infer policy input/output dimensions")
    return obs_dim, action_dim


def infer_task_name(checkpoint_path: Path, policy_io: Dict[str, Any] | None, policy_state: Dict[str, torch.Tensor]) -> str:
    if policy_io and policy_io.get("task_name"):
        return str(policy_io["task_name"])
    name = checkpoint_path.name.lower() + " " + str(checkpoint_path.parent).lower()
    for task_id in ("task1", "task2", "task3", "task4"):
        if task_id in name:
            return task_id
    if any(key.startswith("encoder.") for key in policy_state.keys()):
        return "task4"
    return "task"


def build_policy(task_name: str, obs_dim: int, action_dim: int) -> nn.Module:
    task = task_name.lower()
    if "task4" in task or obs_dim >= 4096:
        return VisionDeterministicPolicy(obs_dim, action_dim)
    if "task3" in task:
        return MlpDeterministicPolicy(obs_dim, action_dim, [512, 512, 256, 128])
    if "task2" in task:
        return MlpDeterministicPolicy(obs_dim, action_dim, [512, 256, 128])
    return MlpDeterministicPolicy(obs_dim, action_dim, [256, 256, 128])


def load_policy_io_for_checkpoint(checkpoint_path: Path, policy_io_path: str | Path | None = None) -> Dict[str, Any] | None:
    if policy_io_path:
        return json.loads(Path(policy_io_path).read_text(encoding="utf-8"))
    candidate = checkpoint_path.parent / "policy_io.json"
    if candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def find_normalizer_tensors(norm_state: Dict[str, Any] | None) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(norm_state, dict):
        return None, None
    mean = None
    variance = None
    mean_keys = ("running_mean", "_running_mean", "mean", "_mean", "obs_mean")
    var_keys = ("running_variance", "_running_variance", "variance", "_variance", "var", "_var", "obs_var")
    for key in mean_keys:
        if key in norm_state:
            mean = norm_state[key]
            break
    for key in var_keys:
        if key in norm_state:
            variance = norm_state[key]
            break
    if mean is None or variance is None:
        return None, None
    return torch.as_tensor(mean, dtype=torch.float32).view(-1), torch.as_tensor(variance, dtype=torch.float32).view(-1)


def extract_observation_normalizer(checkpoint_data: Any, checkpoint_path: Path, obs_dim: int) -> Tuple[torch.Tensor | None, torch.Tensor | None, str]:
    candidates: List[Tuple[str, Any]] = []
    if isinstance(checkpoint_data, dict):
        candidates.append(("checkpoint.actor_obs_norm", checkpoint_data.get("actor_obs_norm")))
        candidates.append(("checkpoint.observation_preprocessor", checkpoint_data.get("observation_preprocessor")))
    preprocessor_path = checkpoint_path.parent / "_observation_preprocessor.pt"
    if preprocessor_path.exists():
        try:
            candidates.append((str(preprocessor_path), torch_load(preprocessor_path)))
        except Exception:
            pass
    for source, candidate in candidates:
        mean, variance = find_normalizer_tensors(candidate)
        if mean is None or variance is None:
            continue
        if mean.numel() == int(obs_dim) and variance.numel() == int(obs_dim):
            return mean, variance, source
    return None, None, "none"


def build_export_module(
    checkpoint_path: Path,
    checkpoint_data: Any,
    policy_io: Dict[str, Any] | None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    original_policy_state = extract_policy_state(checkpoint_data)
    policy_state = remove_unused_policy_tensors(original_policy_state)
    obs_dim, action_dim = infer_dims(policy_state, policy_io)
    task_name = infer_task_name(checkpoint_path, policy_io, policy_state)
    policy = build_policy(task_name, obs_dim, action_dim)
    missing, unexpected = policy.load_state_dict(policy_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected policy tensors: {unexpected}")
    policy.eval()

    mean, variance, normalizer_source = extract_observation_normalizer(checkpoint_data, checkpoint_path, obs_dim)
    normalizer_embedded = mean is not None and variance is not None
    export_module: nn.Module
    if normalizer_embedded:
        export_module = ObservationNormalizedPolicy(policy, mean=mean, variance=variance)
        input_convention = "raw_observation"
    else:
        export_module = policy
        input_convention = "normalized_observation"
    export_module.eval()
    export_info = {
        "task_name": task_name,
        "actor_obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "missing_tensors": list(missing),
        "dropped_tensors": sorted(set(original_policy_state.keys()) - set(policy_state.keys())),
        "normalizer_embedded": bool(normalizer_embedded),
        "normalizer_source": normalizer_source,
        "onnx_input_convention": input_convention,
    }
    return export_module, export_info


def export_policy_onnx(checkpoint: str | Path, output: str | Path, policy_io_path: str | Path | None = None, opset: int = 17) -> Path:
    checkpoint_path = resolve_checkpoint_path(checkpoint, preferred_names=EXPORT_CHECKPOINT_NAMES)
    checkpoint_data = torch_load(checkpoint_path)
    policy_io = load_policy_io_for_checkpoint(checkpoint_path, policy_io_path)
    export_module, export_info = build_export_module(checkpoint_path, checkpoint_data, policy_io)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obs_dim = int(export_info["actor_obs_dim"])
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        export_module,
        dummy,
        str(output_path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=int(opset),
    )
    sidecar = output_path.with_suffix(".export.json")
    sidecar.write_text(
        json.dumps(
            _json_safe({
                "checkpoint": str(checkpoint_path),
                "policy_io": policy_io,
                "export": export_info,
            }),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export quadrotor policy checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-io", default="")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    out = export_policy_onnx(args.checkpoint, args.output, policy_io_path=args.policy_io or None, opset=args.opset)
    print(f"exported: {out}")


if __name__ == "__main__":
    main()
