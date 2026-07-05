# Quadrotor RL Refactor Implementation

## Scope

This refactor keeps the original four task entry points and extracts shared logic into reusable modules. The goal is to preserve the trainable task structure while adding a clearer base layer for configuration, action semantics, rotor wrench computation, evaluation, export, and sim2sim validation.

## Implemented structure

- `quadrotor_rl.core.config`: base configuration schema.
- `quadrotor_rl.core.math`: quaternion and frame utilities.
- `quadrotor_rl.core.physics`: action semantics and rotor wrench conversion.
- `quadrotor_rl.core.env`: environment helper utilities.
- `quadrotor_rl.core.scene`: scene helper utilities.
- `quadrotor_rl.training`: reusable training helper modules.
- `quadrotor_rl.evaluation`: checkpoint selection, rollout recording, and metrics helpers.
- `quadrotor_rl.export`: policy I/O metadata, ONNX export, and Torch/ONNX comparison.
- `quadrotor_rl.sim2sim`: trajectory format, MuJoCo replay entry, and report helpers.

## Compatibility decisions

Task1 historically used `filtered_actions` as a raw-action-space smoothed buffer, and both observation and reward terms read that buffer directly. This refactor therefore preserves Task1 raw-action buffer semantics and converts to motor-delta space only when computing rotor wrench.

Task2, Task3, and Task4 already used scaled motor-delta filtered actions in the control path, so they continue to use the canonical shared action pipeline. Their `prev_raw_actions` fields now store sanitized raw actions after NaN/Inf handling and clipping, which keeps action smoothness terms consistent with environment-side action safety.

## ONNX export behavior

The ONNX export path now prioritizes task eval checkpoint files such as `quadrotor_task1_model.pt` before skrl agent checkpoint files. This matches the training scripts that save a clearer `policy` state dictionary and normalizer metadata in eval checkpoints.

Gaussian actor checkpoints may include `log_std_parameter`. Deployment uses deterministic mean actions, so `log_std_parameter` is intentionally dropped during export.

If an observation normalizer is found in the checkpoint or in `_observation_preprocessor.pt`, it is embedded into the exported ONNX graph. In that case the exported model accepts raw observations. If no normalizer is available, the exported model accepts normalized observations and the export sidecar records this explicitly.

## Validation

Run:

```bash
python -m compileall -q src tests
PYTHONPATH=src pytest -q tests/core/test_action_semantics.py tests/core/test_legacy_action_equivalence.py tests/export/test_policy_io.py tests/export/test_export_onnx_checkpoint.py
```

Expected result:

```text
6 passed
```

Isaac Lab smoke tests should still be run in the target machine because this environment cannot execute full Isaac simulation.
