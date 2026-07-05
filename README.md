# 🚁 Quadrotor Reinforcement Learning Control with NVIDIA Isaac Lab

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-supported-brightgreen)
![RL](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![Platform](https://img.shields.io/badge/platform-Ubuntu%20%7C%20Windows-green)

本项目是一个基于 NVIDIA Isaac Lab 的四旋翼无人机强化学习控制工程，面向 Crazyflie / Quadrotor 类平台，提供从基础悬停、航点跟踪、动态障碍导航到视觉窄门竞速的多任务训练、评估、导出与验证流程。

项目目标不是单纯提供一个可运行的 PPO demo，而是建立一套可复用的机器人策略训练工程框架：统一任务配置、统一动作语义、统一环境测试、统一训练入口、统一 checkpoint 与 policy metadata 管理，并为后续 ONNX 导出、跨仿真验证和 sim2sim 流程预留标准接口。

---

## 🎬 Demonstrations

| Scenario | Preview |
|---|---|
| Hovering / Waypoint Tracking | ![Quadrotor hover demo](assets/gifs/quadrotor_hover_demo.gif) |
| Obstacle Avoidance / Gate Racing | ![Quadrotor racing demo](assets/gifs/quadrotor_gate_racing_demo.gif) |

> GIF 文件仅用于展示训练目标和典型效果。实际结果会受到 Isaac Lab 版本、物理设置、随机种子、训练步数、并发环境数和硬件资源影响。

---

## ✨ Highlights

- 基于 NVIDIA Isaac Lab 构建四旋翼无人机强化学习控制任务。
- 支持四个递进任务：悬停稳定、三维航点跟踪、动态障碍导航、视觉窄门竞速。
- 使用 `skrl` PPO 作为默认强化学习训练框架。
- 采用统一动作语义：`raw action -> scale -> filter -> motor multiplier -> body wrench`。
- 新增 `core/` 公共层，集中管理配置、姿态数学、动作语义、电机模型和环境通用逻辑。
- 新增 `training/`、`evaluation/`、`export/`、`sim2sim/` 模块，支持训练工具复用和后续部署验证扩展。
- 保留任务级入口，兼容原有 Task1–Task4 训练、测试、评估工作流。
- 提供 Ubuntu / Windows 两套脚本，脚本命名采用统一的工程化风格。
- 区分 standalone IsaacLab 环境测试和纯 Python 单元测试，避免测试入口混用。
- 提供 `policy_io.json`，记录 observation、action、normalizer、checkpoint 和导出相关元信息。
- 支持后续扩展 ONNX 导出、Torch/ONNX 对齐检查和 MuJoCo sim2sim 回放。

---

## 🧭 Project Scope

本仓库聚焦四旋翼无人机的纯强化学习控制与工程化验证，适合用于以下方向：

- Isaac Lab 多任务强化学习环境构建；
- quadrotor root wrench 控制接口验证；
- PPO 训练流程、日志与 checkpoint 管理；
- 多场景 reward、termination、observation 调试；
- ONNX 导出和 policy IO 一致性检查；
- 跨仿真 sim2sim 验证流程设计；
- 机器人 RL 项目工程结构参考。

本项目不直接提供真实无人机部署接口，不包含飞控固件、不替代真实飞行控制器，也不保证未经安全验证的策略可在真实无人机上运行。

---

## 📁 Repository Structure

```text
quadrotor_isaaclab_rl/
├── assets/
│   ├── gifs/
│   ├── motions/
│   └── usd/
├── configs/
│   ├── task1_hover_stabilization.yaml
│   ├── task2_waypoint_tracking.yaml
│   ├── task3_obstacle_navigation.yaml
│   └── task4_vision_gate_racing.yaml
├── docs/
│   ├── project_overview.md
│   ├── refactor_implementation.md
│   ├── results_and_checkpoints.md
│   ├── task1_design.md
│   ├── task2_design.md
│   ├── task3_design.md
│   ├── task4_design.md
│   ├── troubleshooting.md
│   ├── ubuntu_training.md
│   └── windows_training.md
├── scripts/
│   ├── ubuntu/
│   │   ├── checkProjectStructure.sh
│   │   ├── testQuadrotorAssetControl.sh
│   │   ├── testTask1Environment.sh
│   │   ├── testTask2Environment.sh
│   │   ├── testTask3Environment.sh
│   │   ├── testTask3World.sh
│   │   ├── testTask4Environment.sh
│   │   ├── testTask4World.sh
│   │   ├── trainTask1Hover.sh
│   │   ├── trainTask1HoverSmoke.sh
│   │   ├── trainTask2Waypoint.sh
│   │   ├── trainTask2WaypointSmoke.sh
│   │   ├── trainTask3Obstacle.sh
│   │   ├── trainTask3ObstacleSmoke.sh
│   │   ├── trainTask4Gate.sh
│   │   ├── trainTask4GateSmoke.sh
│   │   ├── evaluateTask1Hover.sh
│   │   ├── evaluateTask2Waypoint.sh
│   │   ├── evaluateTask3Obstacle.sh
│   │   ├── evaluateTask4Gate.sh
│   │   └── visual/
│   └── windows/
│       ├── checkProjectStructure.ps1
│       ├── testTask1Environment.ps1
│       ├── trainTask1Hover.ps1
│       ├── trainTask1HoverSmoke.ps1
│       ├── evaluateTask1Hover.ps1
│       └── visual/
├── src/
│   └── quadrotor_rl/
│       ├── common/
│       ├── core/
│       │   ├── config/
│       │   ├── env/
│       │   ├── math/
│       │   ├── physics/
│       │   └── scene/
│       ├── data/
│       ├── evaluation/
│       ├── export/
│       ├── sim2sim/
│       ├── tasks/
│       │   ├── task1/
│       │   ├── task2/
│       │   ├── task3/
│       │   └── task4/
│       └── training/
├── tests/
│   ├── asset/
│   ├── core/
│   ├── export/
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── pytest.ini
└── README.md
```

| Path | Description |
|---|---|
| `src/quadrotor_rl/core/` | 底层公共模块，包括基础配置、姿态数学、动作语义、电机模型、环境工具和场景工具。 |
| `src/quadrotor_rl/tasks/` | Task1–Task4 的任务定义，每个任务保留独立的 config、scene、env、train 和 model test。 |
| `src/quadrotor_rl/training/` | 训练工具层，包括日志、PPO 配置、数值安全、训练 runner 和 checkpoint 保存辅助逻辑。 |
| `src/quadrotor_rl/evaluation/` | 评估工具层，包括评估指标、rollout 记录、checkpoint 选择和 eval runner。 |
| `src/quadrotor_rl/export/` | 导出工具层，包括 `policy_io.json`、ONNX 导出、输入输出检查和 Torch/ONNX 对齐。 |
| `src/quadrotor_rl/sim2sim/` | 跨仿真验证接口，包括轨迹格式、MuJoCo replay 和 sim2sim report。 |
| `scripts/ubuntu/` | Ubuntu 环境下的测试、训练、评估和可视化脚本。 |
| `scripts/windows/` | Windows 环境下的测试、训练、评估和可视化脚本。 |
| `tests/core/` | 纯 Python 单元测试，可用 pytest 运行。 |
| `tests/export/` | policy IO、ONNX export 和 checkpoint 相关单元测试。 |
| `tests/taskX/` | IsaacLab standalone 环境测试，不建议直接用 pytest 收集。 |
| `logs/` | 默认训练日志、TensorBoard 文件和 checkpoint 输出目录。 |
| `outputs/` | 默认评估结果、曲线、rollout 和诊断文件输出目录。 |

---

## 🧩 Task Overview

| Task | Name | Objective | Observation | Action |
|---|---|---|---|---|
| Task1 | Hover Stabilization | 固定高度悬停与姿态稳定 | stacked proprioceptive state | 4 rotor corrections |
| Task2 | Waypoint Tracking | 三维航点跟踪与连续目标切换 | stacked state + waypoint features | 4 rotor corrections |
| Task3 | Obstacle Navigation | 动态障碍环境中的导航避障 | stacked state + lidar/risk features | 4 rotor corrections |
| Task4 | Vision Gate Racing | 基于深度图的窄门穿越 | depth image + compact state | 4 rotor corrections |

---

## 🛠️ Requirements

本项目需要在可运行 Isaac Lab 的 Python 环境中使用。建议先完成 NVIDIA Isaac Sim / Isaac Lab 的官方安装流程，并确认以下组件可正常导入：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import isaaclab; print('Isaac Lab import ok')"
```

常用 Python 依赖包括：

```bash
pip install skrl tensorboard tqdm numpy matplotlib
```

如使用 ONNX 导出与对齐检查，可按需安装：

```bash
pip install onnx onnxruntime
```

> Isaac Lab、Isaac Sim、PyTorch、CUDA 和驱动版本之间存在兼容要求。建议以 NVIDIA 官方文档和当前工作站环境为准，不在 README 中绑定特定个人配置。

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/0324Lw/NVIDIA--Isaac-Lab-Quadrotor-control quadrotor_isaaclab_rl
cd quadrotor_isaaclab_rl
```

### 2. Set Python Path

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Ubuntu / Windows 脚本会自动设置项目根目录和 `PYTHONPATH`，通常无需手动配置。

### 3. Check Project Structure

```bash
bash scripts/ubuntu/checkProjectStructure.sh
```

### 4. Run Asset Control Test

```bash
bash scripts/ubuntu/testQuadrotorAssetControl.sh
```

该测试用于检查 Crazyflie / Quadrotor 资产加载、质量估计和 root wrench 控制接口是否正常。

---

## ✅ Testing

本项目包含两类测试，需要区分运行方式。

### Pure Unit Tests

`tests/core/` 和 `tests/export/` 属于纯 Python 单元测试，可以使用 pytest 运行：

```bash
cd /path/to/quadrotor_isaaclab_rl

PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/core/test_action_semantics.py \
  tests/core/test_legacy_action_equivalence.py \
  tests/export/test_policy_io.py \
  tests/export/test_export_onnx_checkpoint.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免系统中 ROS / launch testing 等第三方 pytest 插件污染当前 conda 环境。

### IsaacLab Standalone Environment Tests

Task 环境测试需要启动 IsaacLab / SimulationApp，应通过脚本运行，而不是直接交给 pytest 自动收集：

```bash
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/testTask2Environment.sh
bash scripts/ubuntu/testTask3World.sh
bash scripts/ubuntu/testTask3Environment.sh
bash scripts/ubuntu/testTask4World.sh
bash scripts/ubuntu/testTask4Environment.sh
```

> `tests/taskX/taskX_env_test.py` 是 standalone IsaacLab 测试程序。不要直接执行 `pytest tests/taskX/taskX_env_test.py`，否则 pytest 会把内部函数参数误识别为 fixture。

---

## 🏃 Training

### Smoke Training

Smoke training 用于确认训练入口、环境 step、日志写入和 checkpoint 保存是否正常，不用于评价最终策略效果。

```bash
bash scripts/ubuntu/trainTask1HoverSmoke.sh
bash scripts/ubuntu/trainTask2WaypointSmoke.sh
bash scripts/ubuntu/trainTask3ObstacleSmoke.sh
bash scripts/ubuntu/trainTask4GateSmoke.sh
```

可通过环境变量覆盖默认训练规模：

```bash
NUM_ENVS=1024 TOTAL_ENV_STEPS=20000 bash scripts/ubuntu/trainTask2WaypointSmoke.sh
```

### Full Training

```bash
bash scripts/ubuntu/trainTask1Hover.sh
bash scripts/ubuntu/trainTask2Waypoint.sh
bash scripts/ubuntu/trainTask3Obstacle.sh
bash scripts/ubuntu/trainTask4Gate.sh
```

推荐训练顺序：

```text
Task1 Hover
→ Task2 Waypoint Tracking
→ Task3 Obstacle Navigation
→ Task4 Vision Gate Racing
```

Task2–Task4 可以根据实验设计从前序任务 checkpoint warm-start，也可以从零训练。

---

## 🔍 Evaluation

评估脚本默认查找对应任务日志目录下最近的 `final_checkpoint`。也可以通过 `CHECKPOINT` 显式指定模型目录。

```bash
bash scripts/ubuntu/evaluateTask1Hover.sh
bash scripts/ubuntu/evaluateTask2Waypoint.sh
bash scripts/ubuntu/evaluateTask3Obstacle.sh
bash scripts/ubuntu/evaluateTask4Gate.sh
```

指定 checkpoint：

```bash
CHECKPOINT=logs/task2/<run_name>/final_checkpoint bash scripts/ubuntu/evaluateTask2Waypoint.sh
```

评估输出通常包括：

- success rate；
- crash rate；
- episode length；
- position error；
- trajectory error；
- action magnitude；
- motor saturation ratio；
- obstacle collision rate；
- gate pass count；
- rollout summary。

---

## 🖥️ Visualization

```bash
CHECKPOINT=logs/task1/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask1Hover.sh
CHECKPOINT=logs/task2/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask2Waypoint.sh
CHECKPOINT=logs/task3/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask3Obstacle.sh
CHECKPOINT=logs/task4/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask4Gate.sh
```

可视化模式通常需要 GUI、图形驱动和 Isaac Sim 渲染环境支持。无显示环境下建议先运行 headless smoke training 和 eval。

---

## ➡️ Task 1: Hover Stabilization

Task1 是基础飞行控制任务，用于验证无人机资产、root wrench 控制、动作语义和 PPO 训练管线。

### Objective

- 在目标高度附近保持稳定悬停。
- 限制 roll / pitch / yaw 发散。
- 学习旋翼推力修正与机体姿态、速度、高度之间的关系。
- 为后续 waypoint、navigation 和 gate racing 任务提供基础控制能力。

### Design

- 使用 Crazyflie / Quadrotor USD 资产。
- action 为 4 维旋翼修正量。
- 环境将 policy action 映射为 motor multiplier，再映射为机体系 wrench。
- observation 包含高度误差、速度、角速度、重力投影、姿态角和历史动作。
- reward 重点关注高度稳定、姿态稳定、速度约束和动作平滑。

### Commands

```bash
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/trainTask1HoverSmoke.sh
bash scripts/ubuntu/trainTask1Hover.sh
bash scripts/ubuntu/evaluateTask1Hover.sh
```

---

## ➡️ Task 2: Waypoint Tracking

Task2 在悬停稳定基础上加入三维航点目标，要求无人机在保持姿态稳定的同时向目标点移动。

### Objective

- 跟踪随机生成的三维 waypoint。
- 接近当前目标后切换下一个目标。
- 学习目标方向、速度、姿态和动作之间的关系。
- 为动态障碍导航提供基础移动能力。

### Design

- observation 包含目标相对位置、目标距离、机体速度、角速度、姿态和历史动作。
- reward 包含距离减少、朝向目标速度、目标到达、高度保持和动作平滑。
- 支持从 Task1 checkpoint 进行 warm-start。

### Commands

```bash
bash scripts/ubuntu/testTask2Environment.sh
bash scripts/ubuntu/trainTask2WaypointSmoke.sh
bash scripts/ubuntu/trainTask2Waypoint.sh
bash scripts/ubuntu/evaluateTask2Waypoint.sh
```

---

## ➡️ Task 3: Obstacle Navigation

Task3 在真实无人机物理控制基础上加入解析障碍物世界。无人机需要根据目标点、lidar 和风险特征完成导航避障。

### Objective

- 根据目标点进行自主导航。
- 使用 lidar / risk features 感知障碍物。
- 避免静态和动态障碍物碰撞。
- 在保持飞行稳定的同时向目标前进。

### Design

- `task3_world.py` 提供纯 tensor 解析障碍物世界。
- 静态障碍、动态障碍、lidar、碰撞和目标点由 GPU tensor 计算。
- `task3_env.py` 接入 IsaacLab 物理环境，继续使用统一 root wrench 控制。
- 解析 world 不生成大量 USD prim，有利于保持训练并发效率。

### Commands

```bash
bash scripts/ubuntu/testTask3World.sh
bash scripts/ubuntu/testTask3Environment.sh
bash scripts/ubuntu/trainTask3ObstacleSmoke.sh
bash scripts/ubuntu/trainTask3Obstacle.sh
bash scripts/ubuntu/evaluateTask3Obstacle.sh
```

---

## ➡️ Task 4: Vision Gate Racing

Task4 面向视觉控制任务。无人机需要基于解析深度图和 compact state 依次穿过随机窄门。

### Objective

- 在 gate racing 场景中依次通过多个随机窄门。
- 使用 depth image 作为主要视觉输入。
- 学习中心线跟踪、姿态对齐、门框避障和穿门策略。
- 为后续视觉域随机化和 sim2real 研究提供基础任务。

### Design

- `task4_world.py` 生成 gate layout、centerline 和解析 depth observation。
- observation 由 depth image 和 compact state 组成。
- policy 可采用 CNN encoder + MLP compact encoder。
- reward 包含中心线跟踪、门中心对齐、穿门奖励、深度安全和动作平滑。
- smoke checkpoint 只验证训练管线和模型 forward，不代表最终策略性能。

### Commands

```bash
bash scripts/ubuntu/testTask4World.sh
bash scripts/ubuntu/testTask4Environment.sh
bash scripts/ubuntu/trainTask4GateSmoke.sh
bash scripts/ubuntu/trainTask4Gate.sh
bash scripts/ubuntu/evaluateTask4Gate.sh
```

---

## 📦 Checkpoints and Policy Metadata

训练结果默认保存到：

```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```

典型 checkpoint 目录包含：

```text
checkpoint_<env_steps>/
final_checkpoint/
  quadrotor_taskX_model.pt
  quadrotor_taskX_skrl_model.pt
  _observation_preprocessor.pt
  _state_preprocessor.pt
  _value_preprocessor.pt
  train_metadata.pt
  policy_io.json
```

`policy_io.json` 用于记录模型输入输出和导出相关信息，例如：

```json
{
  "task_name": "task2_waypoint",
  "obs_dim": 100,
  "action_dim": 4,
  "action_semantics": "motor_multiplier_delta",
  "action_scale": 0.25,
  "control_dt": 0.02,
  "normalizer": "skrl_running_standard_scaler"
}
```

该文件是 ONNX 导出、sim2sim replay 和部署一致性检查的基础。

---

## 🔄 ONNX Export and Sim2Sim

### Policy IO Check

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.export.check_policy_io \
  --policy-io logs/task1/<run_name>/final_checkpoint/policy_io.json
```

### ONNX Export

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.export.export_onnx \
  --checkpoint logs/task1/<run_name>/final_checkpoint \
  --output exports/task1/policy.onnx
```

### Torch / ONNX Comparison

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.export.compare_torch_onnx \
  --checkpoint logs/task1/<run_name>/final_checkpoint \
  --onnx exports/task1/policy.onnx
```

### Sim2Sim Replay

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.sim2sim.mujoco_replay \
  --task task1 \
  --onnx exports/task1/policy.onnx
```

当前 sim2sim 模块提供标准接口和最小 replay/report 结构，可继续扩展为更完整的 MuJoCo closed-loop 验证。

---

## 📊 Logging

训练过程中默认记录：

- reward components；
- event flags；
- telemetry；
- PPO optimization metrics；
- action statistics；
- checkpoint metadata；
- evaluation summary。

可使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir logs
```

建议重点关注：

| Task | Key Metrics |
|---|---|
| Task1 | height error, roll/pitch error, crash rate, action magnitude |
| Task2 | goal distance, progress, waypoint success, height stability |
| Task3 | goal distance, min lidar distance, obstacle collision rate, deviation rate |
| Task4 | passed gates, centerline distance, depth safety, gate collision rate |

---

## 🧪 Troubleshooting

### `ModuleNotFoundError: No module named quadrotor_rl`

确认位于项目根目录，并设置：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

或使用 `scripts/ubuntu/` 下的脚本运行。

### pytest loads ROS plugins and reports missing `lark`

在混合 ROS / IsaacLab / conda 环境中，pytest 可能自动加载 ROS 插件。运行纯单元测试时使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ...
```

### `fixture 'env' not found`

不要用 pytest 直接收集 `tests/taskX/taskX_env_test.py`。这些文件是 standalone IsaacLab 测试程序，应使用：

```bash
bash scripts/ubuntu/testTask1Environment.sh
```

### Checkpoint saving fails with JSON serialization error

确认当前代码使用了 JSON-safe policy metadata writer。`policy_io.json` 会将 tensor、numpy array、path、dataclass 等对象转换为可序列化格式。

### GPU memory is insufficient

降低并发环境数：

```bash
NUM_ENVS=128 bash scripts/ubuntu/trainTask1HoverSmoke.sh
```

或关闭 GUI，仅运行 headless smoke training。

### Smoke training performance is poor

这是正常现象。Smoke training 只用于验证工程链路，不用于判断最终策略质量。

---

## 🧱 Development Notes

建议按照以下顺序修改或扩展代码：

1. 先确认 asset test 和 Task1 environment test 通过。
2. 再修改 action semantics、reward 或 observation。
3. 每次修改后先运行 smoke training。
4. 训练前保存配置和 commit hash。
5. 导出 checkpoint 前检查 `policy_io.json`。
6. sim2sim 前检查 Torch / ONNX 输出差异。
7. 不直接用 smoke checkpoint 判断最终策略效果。

---

## 📌 Current Limitations

- 当前策略主要用于仿真训练与工程研究，不提供真实无人机部署安全保证。
- Task3 / Task4 使用解析 world，障碍物和 gate 不一定生成真实 USD prim。
- Task4 使用解析 depth observation，后续可替换为真实 Isaac camera rendering。
- sim2sim 模块目前以标准接口和最小 replay 为主，仍可扩展更完整的 dynamics alignment。
- 不同 Isaac Lab / Isaac Sim 环境可能存在 API 差异，必要时需要少量适配。
- 训练结果受随机种子、并发环境数、训练步数、物理参数和 reward 权重影响。

---

## 📄 License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

This project is built on top of the following open-source tools and communities:

- NVIDIA Isaac Sim / Isaac Lab
- Crazyflie / Bitcraze ecosystem
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- Open-source robotics and reinforcement learning communities