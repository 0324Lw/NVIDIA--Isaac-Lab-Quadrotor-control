# 🚁 基于 NVIDIA Isaac Lab 的 Crazyflie 四旋翼无人机强化学习控制项目

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.x-brightgreen)
![skrl](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![OS](https://img.shields.io/badge/OS-Ubuntu%20%7C%20Windows-green)

本项目是一个基于 NVIDIA Isaac Lab 的 Crazyflie / Quadrotor 四旋翼无人机强化学习控制项目。项目包含 4 个递进任务：悬停稳定、航点跟踪、动态障碍导航、视觉窄门竞速。

这个仓库最开始来自传统 Gazebo / PyBullet 无人机控制代码的迁移需求。后来我参考前面整理过的 Isaac Lab 机器狗、无人车和灵巧手工程，对无人机项目进行了重新设计和工程化重构：统一了项目目录结构，统一采用 `skrl` 的 PPO 训练流程，增加了 Crazyflie 资产接口测试、环境测试、世界模型测试、模型测试、Ubuntu / Windows 脚本、训练进度条、日志与 checkpoint 管理。

项目重点不是追求一次性得到完美策略，而是把每个任务从模型接口、世界场景、环境、测试、训练到评估尽量拆清楚。代码中仍然会有可以继续改进的地方，欢迎大家根据自己的 Isaac Lab 版本、显卡配置和研究目标继续修改。

---

## 🎬 训练效果展示

| Scene | Preview |
|---|---|
| 悬停 / 航点跟踪 | ![Quadrotor hover demo](assets/gifs/quadrotor_hover_demo.gif) |
| 避障 / 窄门竞速 | ![Quadrotor racing demo](assets/gifs/quadrotor_gate_racing_demo.gif) |

---

## ✨ 项目特点

- 基于 NVIDIA Isaac Lab 和 Crazyflie 四旋翼无人机 USD 资产。
- 包含 4 个递进任务，从基础悬停到航点跟踪、动态避障和视觉窄门竞速。
- 所有任务统一使用 `skrl` PPO 训练框架。
- 每个任务提供独立的环境测试、训练脚本和模型测试脚本。
- Task3 / Task4 将 world 逻辑与 Isaac Lab 物理环境分离，方便单独测试障碍物、窄门、解析深度图和课程逻辑。
- 提供 Crazyflie asset/control smoke test，用于验证无人机模型接口、质量估计和 root wrench 控制。
- 支持 Ubuntu / Windows 本地开发、测试和训练。
- 训练采用 `tqdm` 进度条，方便查看实时进度和日志信息。

---

## 📁 项目结构

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
│   └── task4_sim2real_robust_flight.yaml
├── docs/
│   ├── project_overview.md
│   ├── results_and_checkpoints.md
│   ├── task1_design.md
│   ├── task2_design.md
│   ├── task3_design.md
│   ├── task4_design.md
│   ├── troubleshooting.md
│   ├── ubuntu_training.md
│   ├── windows_path_config.md
│   └── windows_training.md
├── scripts/
│   ├── ubuntu/
│   │   ├── test_quadrotor_asset_control.sh
│   │   ├── test_task1_env.sh
│   │   ├── test_task2_env.sh
│   │   ├── test_task3_world.sh
│   │   ├── test_task3_env.sh
│   │   ├── test_task4_world.sh
│   │   ├── test_task4_env.sh
│   │   ├── train_task1_skrl_smoke.sh
│   │   ├── train_task2_skrl_smoke.sh
│   │   ├── train_task3_skrl_smoke.sh
│   │   ├── train_task4_skrl_smoke.sh
│   │   ├── eval_task1_skrl.sh
│   │   ├── eval_task2_skrl.sh
│   │   ├── eval_task3_skrl.sh
│   │   ├── eval_task4_skrl.sh
│   │   └── visual/
│   └── windows/
├── src/
│   └── quadrotor_rl/
│       ├── common/
│       │   ├── eval_curriculum_utils.py
│       │   ├── info_utils.py
│       │   ├── model_eval_utils.py
│       │   ├── paths.py
│       │   ├── progress.py
│       │   ├── quadrotor_skrl_models.py
│       │   ├── quadrotor_skrl_wrappers.py
│       │   ├── running_mean_std.py
│       │   ├── skrl_models.py
│       │   └── vec_wrappers.py
│       ├── data/
│       └── tasks/
│           ├── task1/
│           │   ├── task1_config.py
│           │   ├── task1_scene.py
│           │   ├── task1_env.py
│           │   ├── task1_train.py
│           │   └── task1_model_test.py
│           ├── task2/
│           │   ├── task2_config.py
│           │   ├── task2_scene.py
│           │   ├── task2_env.py
│           │   ├── task2_train.py
│           │   └── task2_model_test.py
│           ├── task3/
│           │   ├── task3_config.py
│           │   ├── task3_world.py
│           │   ├── task3_scene.py
│           │   ├── task3_env.py
│           │   ├── task3_train.py
│           │   └── task3_model_test.py
│           └── task4/
│               ├── task4_config.py
│               ├── task4_world.py
│               ├── task4_scene.py
│               ├── task4_env.py
│               ├── task4_train.py
│               └── task4_model_test.py
├── tests/
│   ├── asset/
│   │   └── quadrotor_asset_control_test.py
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
└── README.md
```

| 目录 | 说明 |
|---|---|
| `configs/` | 每个任务的配置文件，便于统一管理任务参数。 |
| `src/quadrotor_rl/common/` | 通用网络模型、日志工具、路径工具、评估工具和 skrl wrapper。 |
| `src/quadrotor_rl/tasks/taskX/` | 每个任务的配置、场景、环境、训练脚本和模型测试脚本。 |
| `tests/asset/` | Crazyflie 资产接口和 root wrench 控制测试。 |
| `tests/taskX/` | 环境测试和世界模型测试脚本。 |
| `scripts/ubuntu/` | Ubuntu 下的测试、训练、评估和可视化脚本。 |
| `scripts/windows/` | Windows / RTX 3090 下的准备检查、训练、评估脚本。 |
| `logs/` | 默认训练日志和 checkpoint 输出目录。 |
| `outputs/` | 默认评估曲线、npz 诊断和可视化输出目录。 |
| `assets/` | README 图片、GIF、USD 说明和其他展示素材。 |

---

## 🛠️ 建议硬件与系统配置

### 最低测试配置

用于模型接口测试、环境测试、world 测试、smoke training 和低并发调试：

- Ubuntu 22.04 / 24.04
- NVIDIA GPU，显存 16GB 以上
- Python 3.11
- PyTorch 2.x
- Isaac Sim / Isaac Lab
- `skrl`, `tensorboard`, `tqdm`, `numpy`

无人机任务的单个资产比四足机器人轻，但 Task4 使用 `64 x 64` 深度视觉 CNN，显存和算力压力会明显增加。在 16GB 显存设备上，建议从小并发开始：

```bash
--num-envs 1
--num-envs 2
--num-envs 4
--num-envs 8
--num-envs 16
```

### 推荐训练配置

用于较大规模训练和长时间实验：

- NVIDIA RTX 3090 / 4090 或同级别 GPU
- 显存 24GB 或更高
- Windows 或 Ubuntu 均可，但需要保证 Isaac Lab 环境可正常运行

较大显存设备可以尝试：

```bash
--num-envs 32
--num-envs 64
--num-envs 128
--num-envs 256
```

具体并发数需要根据任务复杂度、显存占用、Isaac Lab 版本和是否启用可视化调整。不要一开始直接使用最大并发，建议先运行 smoke training。

---

## 🚀 基础准备

### 1. 安装 Isaac Lab 环境

请先按照 NVIDIA Isaac Lab 官方文档安装 Isaac Sim / Isaac Lab，并确认 Isaac Lab 的 Python 环境可以正常导入：

```bash
python -c "import isaaclab; print('isaaclab ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. 克隆项目

```bash
git clone https://github.com/0324Lw/NVIDIA--Isaac-Lab-Quadrotor-control quadrotor_isaaclab_rl
cd quadrotor_isaaclab_rl
```


### 3. 设置 PYTHONPATH

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

也可以直接使用 `scripts/ubuntu/` 下的脚本，这些脚本会自动设置项目路径。

### 4. 安装 Python 依赖

在 Isaac Lab 对应的 Python 环境中安装必要依赖：

```bash
pip install skrl tensorboard tqdm numpy matplotlib
```

如果你的 Isaac Lab 安装方式已经包含部分依赖，可以按需跳过。

### 5. 先测试 Crazyflie 资产

无人机项目从传统 Gazebo / PyBullet 迁移到 Isaac Lab 后，最重要的第一步是确认 Crazyflie 模型能被 Isaac Lab 正常加载，并且 root wrench 控制可以产生上升和偏航响应。

```bash
bash scripts/ubuntu/test_quadrotor_asset_control.sh
```

如果本地 `isaaclab_assets` 中没有内置 Crazyflie 配置，代码会回退到 Isaac 官方 Crazyflie USD 路径。

---

## ⚡ 快速开始

### 1. 环境测试

建议先从资产测试开始，再进入 Task1 到 Task4。

```bash
bash scripts/ubuntu/test_quadrotor_asset_control.sh
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/test_task4_world.sh
bash scripts/ubuntu/test_task4_env.sh
```

如果显存不足，可以打开对应脚本，降低 `--num-envs`。

### 2. Smoke 训练

Smoke training 用于确认训练管线可以启动、日志可以写入、checkpoint 可以保存，不用于评估最终效果。

```bash
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
```

### 3. 模型测试

训练完成后，可以使用 eval 脚本加载 checkpoint 做推理测试。

```bash
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint fixed_easy
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint
```

### 4. GUI 可视化

```bash
bash scripts/ubuntu/visual/visualize_task1.sh logs/task1/<run_name>/final_checkpoint 1.0
bash scripts/ubuntu/visual/visualize_task2.sh logs/task2/<run_name>/final_checkpoint 1.0
bash scripts/ubuntu/visual/visualize_task3.sh logs/task3/<run_name>/final_checkpoint 1.0 fixed_easy
bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run_name>/final_checkpoint 1.0
```

---

## 🧩 任务设计总览

| Task | 目标 | 环境特点 | 训练重点 | 主要脚本 |
|---|---|---|---|---|
| Task1 | 悬停稳定 | Crazyflie、root wrench 控制、定高悬停 | 稳定高度、姿态稳定、动作平滑 | `task1_env.py`, `task1_train.py`, `task1_model_test.py` |
| Task2 | 航点跟踪 | 随机航点、目标向量、速度引导 | 到达航点、保持高度、控制速度 | `task2_env.py`, `task2_train.py`, `task2_model_test.py` |
| Task3 | 动态障碍导航 | 解析障碍物世界、24 维 lidar、静态/动态障碍 | 避障、朝目标前进、保持飞行稳定 | `task3_world.py`, `task3_env.py`, `task3_train.py` |
| Task4 | 视觉窄门竞速 | 5 个随机窄门、64x64 深度图、CNN policy | 穿门、沿中心线飞行、避免门框碰撞 | `task4_world.py`, `task4_env.py`, `task4_train.py` |

---

## ➡️ Task 1：悬停稳定

Task1 是最基础的无人机控制任务，用于训练 Crazyflie 在固定高度附近稳定悬停。

### 任务目标

- 无人机在目标高度附近保持稳定。
- 控制 roll / pitch / yaw 不发散。
- 学习 root wrench 四旋翼动作接口，为后续任务提供基础控制能力。
- 验证 Isaac Lab 中 Crazyflie 资产、质量估计和外力控制链路。

### 环境设计

- 使用 Isaac Lab 中的 Crazyflie USD 资产。
- 如果没有内置 `CRAZYFLIE_CFG`，自动使用 fallback USD 路径。
- 动作是 4 个旋翼的归一化 thrust correction。
- 环境内部将动作映射为总推力和机体系力矩，再通过 root wrench 作用到机体。
- 观测包含相对高度、线速度、角速度、重力投影、姿态角、历史动作等。
- 训练代码统一采用 `skrl` PPO。

### 常用命令

```bash
bash scripts/ubuntu/test_task1_env.sh
bash scripts/ubuntu/train_task1_skrl_smoke.sh
bash scripts/ubuntu/train_task1_skrl_laptop.sh
bash scripts/ubuntu/eval_task1_skrl.sh logs/task1/<run_name>/final_checkpoint
```

### 训练时重点观察

- `Pos_Z` 是否稳定在目标高度附近。
- `Height_Error` 是否逐步下降。
- `RollPitchAbs` 是否保持较小。
- `Action_Abs` 是否不过大。
- `Crash_Rate` 是否接近 0。
- PPO 的 KL、loss、学习率是否稳定。

---

## ➡️ Task 2：航点跟踪

Task2 在悬停稳定的基础上加入空间航点目标。无人机需要根据目标相对位置进行移动，并在接近目标后刷新下一个 waypoint。

### 任务目标

- 从悬停扩展到三维航点跟踪。
- 在保持高度和姿态稳定的同时向目标点移动。
- 学习目标方向、速度、姿态和动作之间的关系。
- 为 Task3 的导航避障提供基础移动能力。

### 环境设计

- `task2_env.py` 直接在 Isaac Lab Crazyflie 环境中生成 waypoint。
- 观测包含目标相对位置、目标距离、机体速度、角速度、姿态和历史动作。
- 奖励包含接近目标、速度方向、定高、安全姿态和动作平滑。
- 支持从 Task1 checkpoint warm-start。

### 常用命令

```bash
bash scripts/ubuntu/test_task2_env.sh
bash scripts/ubuntu/train_task2_skrl_smoke.sh
bash scripts/ubuntu/train_task2_skrl_laptop.sh logs/task1/<run_name>/final_checkpoint
bash scripts/ubuntu/eval_task2_skrl.sh logs/task2/<run_name>/final_checkpoint
```

### 训练时重点观察

- `Goal_Dist` 是否逐步下降。
- `Progress` 是否为正。
- `Success_Rate` 是否逐步上升。
- `Pos_Z` 是否稳定。
- `RollPitchAbs` 是否不过大。
- `Crash_Rate` / `Deviation_Rate` 是否接近 0。

---

## ➡️ Task 3：动态障碍导航

Task3 在真实 Crazyflie 物理控制的基础上加入解析障碍物世界。无人机需要根据目标点、lidar 和风险特征，在存在静态/动态障碍物的环境中到达目标。

### 任务目标

- 根据虚拟目标点进行自主导航。
- 使用 24 维 lidar 与 risk features 感知障碍物。
- 避免静态和动态障碍物碰撞。
- 在保持飞行稳定的同时尽量向目标前进。

### 环境设计

Task3 使用“真实无人机物理 + 解析导航世界”的结构：

- `task3_world.py` 是纯 torch 解析世界，不依赖 Isaac Lab。
- 静态障碍、动态障碍、lidar、碰撞、目标点、risk features 都由 GPU tensor 计算。
- `task3_env.py` 接入 Isaac Lab 的 Crazyflie 物理环境，动作仍然控制真实无人机 root wrench。
- 障碍物不生成大量真实 prim，这样可以保持较高并发和较低仿真开销。

### 观测结构

- 单帧 actor observation：75 维。
- 4 帧堆叠后 actor input：300 维。
- action dim：4。
- lidar：24 rays。
- critic input 当前与 actor input 对齐，后续可扩展 privileged obstacle features。

### 常用命令

```bash
bash scripts/ubuntu/test_task3_world.sh
bash scripts/ubuntu/test_task3_env.sh
bash scripts/ubuntu/train_task3_skrl_smoke.sh
bash scripts/ubuntu/train_task3_skrl_laptop.sh logs/task2/<run_name>/final_checkpoint
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint fixed_easy
bash scripts/ubuntu/eval_task3_skrl.sh logs/task3/<run_name>/final_checkpoint fixed_hard
```

### 训练时重点观察

- `Goal_Dist`
- `Progress`
- `Min_Lidar`
- `Heading_Align`
- `Success_Rate`
- `Crash_Rate`
- `Obstacle_Collision_Rate`
- `Deviation_Rate`

Task3 训练难度比 Task1 / Task2 更高，建议优先使用 Task1 或 Task2 checkpoint warm-start。

---

## ➡️ Task 4：视觉窄门竞速

Task4 面向更高难度的视觉控制任务。无人机需要仅依靠解析深度图和少量本体状态，依次穿过随机窄门。

### 任务目标

- 在 `30m x 10m x 5m` 场地中依次通过 5 个随机窄门。
- 使用 `64 x 64` 深度图作为主要视觉输入。
- 学习沿中心线飞行、对准门姿态、避免门框碰撞和漏门。
- 为后续真实视觉、域随机化和 sim2real 鲁棒飞行做准备。

### 环境设计

Task4 使用“真实无人机物理 + 解析 gate racing world + CNN policy”的结构：

- `task4_world.py` 生成 5 个随机窄门、中心线和解析深度图。
- `task4_env.py` 接入 Isaac Lab Crazyflie 物理环境。
- 观测由 `64 x 64` 深度图和 32 维 compact state 组成。
- 训练策略使用 CNN encoder + MLP compact encoder。
- 奖励包含中心线跟踪、姿态对齐、穿门奖励、深度安全、动作平滑和终止惩罚。

### 观测结构

```text
depth image        1 x 64 x 64 = 4096
compact state      32
actor obs total    4128
action dim         4
```

### 常用命令

```bash
bash scripts/ubuntu/test_task4_world.sh
bash scripts/ubuntu/test_task4_env.sh
bash scripts/ubuntu/train_task4_skrl_smoke.sh
bash scripts/ubuntu/train_task4_skrl_laptop.sh logs/task3/<run_name>/final_checkpoint
bash scripts/ubuntu/eval_task4_skrl.sh logs/task4/<run_name>/final_checkpoint
bash scripts/ubuntu/visual/visualize_task4.sh logs/task4/<run_name>/final_checkpoint 1.0
```

### 训练时重点观察

- `Passed_Gates`
- `Target_Gate_Idx`
- `Centerline_Dist`
- `Depth_Min`
- `Pose_Align`
- `Gate_Pass_Rate`
- `Gate_Collision_Rate`
- `Missed_Gate_Rate`
- `Success_Rate`

Task4 的 smoke checkpoint 只要求加载、forward 和 rollout 正常，不要求立即学会穿门。视觉窄门竞速通常需要更长训练、更稳定的奖励和更合理的课程设计。

---

## 📊 日志与模型保存

训练日志默认保存在：

```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```

每个训练 run 通常包含：

```text
checkpoint_<env_steps>/
final_checkpoint/
train_metadata.pt
quadrotor_taskX_model.pt
quadrotor_taskX_skrl_model.pt
```

可以使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir logs
```

训练过程中会记录以下类型的信息：

- `reward_components`：各奖励项。
- `events`：成功、碰撞、坠机、漏门、偏离、超时等事件。
- `telemetry`：高度、距离、中心线误差、深度最小值、动作强度等训练指标。
- `debug`：观测维度、reward 范围、资产来源、质量估计等信息。
- `ppo`：PPO 更新信息，例如 KL、loss、学习率等。

---

## 💻 Ubuntu / Windows 使用说明

### Ubuntu

Ubuntu 用于：

- 代码开发
- Crazyflie asset/control smoke test
- 环境测试
- world 测试
- smoke training
- 训练验证

常用脚本在：

```text
scripts/ubuntu/
```

### Windows

Windows 脚本在：

```text
scripts/windows/
```

建议先运行 readiness check：

```powershell
.\scripts\windows\check_task1_windows_ready.ps1
.\scripts\windows\check_task2_windows_ready.ps1
.\scripts\windows\check_task3_windows_ready.ps1
.\scripts\windows\check_task4_windows_ready.ps1
```

Windows 训练脚本通常带有审批环境变量，避免误启动长时间训练。例如：

```powershell
$env:QUADROTOR_TASK3_WINDOWS_SMOKE_APPROVED = "1"
.\scripts\windows\train_task3_skrl_smoke_3090.ps1
```

正式训练前建议先运行 smoke 版本，确认路径、Isaac Lab Python、显卡和日志输出都正常。

---

## 🧭 推荐训练顺序

推荐顺序：

1. 先运行 `test_quadrotor_asset_control.sh`，确认 Crazyflie 资产和 root wrench 控制正常。
2. 训练 Task1，获得基础悬停稳定 checkpoint。
3. Task2 从 Task1 warm-start，训练航点跟踪。
4. Task3 从 Task2 warm-start，训练导航与避障。
5. Task4 从 Task3 或 Task2 warm-start，训练视觉窄门竞速。

也可以每个任务从零开始训练，但训练时间会更长，早期调参也会更困难。

---

## 📌 当前状态与限制

- 本项目主要用于学习、复现实验和开源交流。
- 当前代码完成了四个任务的 Isaac Lab 环境、测试、`skrl` PPO 训练和模型测试脚本。
- Task3 / Task4 的障碍物和窄门主要采用解析 world，不会生成大量真实 USD prim。
- Task4 当前使用解析深度图，后续可以替换为真实 Isaac 相机渲染或加入更多视觉域随机化。
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异，需要根据本地环境做少量适配。
- 训练效果会受到 GPU、并发数、随机种子、训练步数和超参数影响。
- Windows 脚本中的默认路径可能需要根据自己的机器修改。
- 本项目不是官方 Bitcraze、Crazyflie 或 NVIDIA 项目，只是个人学习和开源整理。

---

## ❓ 常见问题

### 1. `ModuleNotFoundError: No module named torch`

通常是没有进入 Isaac Lab 对应的 Python / conda 环境。请先确认：

```bash
which python
python -c "import torch; print(torch.__version__)"
```

### 2. Isaac Lab / `pxr` 导入报错

涉及 Isaac Lab、USD、`pxr` 的文件需要在 Isaac Sim / Isaac Lab 环境中运行。测试脚本中如果需要 AppLauncher，应保证先启动 AppLauncher，再导入依赖 Isaac Lab 的环境文件。

### 3. 找不到 `isaaclab_assets.robots.crazyflie`

如果本地 Isaac Lab 没有内置 Crazyflie 配置，代码会自动 fallback 到 Isaac 官方 Crazyflie USD 路径。只要 `test_quadrotor_asset_control.sh` 能通过，说明资产和控制接口可以继续使用。

### 4. 训练启动后显存不足怎么办?

先降低并发数：

```bash
--num-envs 1
--num-envs 2
--num-envs 4
--num-envs 8
--num-envs 16
```

确认能跑通后再逐步增加。

### 5. Smoke training 效果很差怎么办?

这是正常的。Smoke training 只用于检查训练流程是否能启动和保存模型，不代表最终策略效果。

### 6. Task3 / Task4 为什么推荐 warm-start?

Task3 加入导航和障碍物，Task4 加入视觉窄门竞速，直接从零训练会更难。使用 Task1 / Task2 / Task3 checkpoint 可以先继承基础悬停和移动能力，再学习更复杂的任务。

### 7. 为什么要先跑环境测试?

无人机训练中的很多问题不是 PPO 本身造成的，而是 reset、观测维度、坐标系、动作映射、奖励项或终止条件有问题。先跑测试可以减少后续训练调参的时间。

### 8. Task4 配置文件名为什么还是 `task4_sim2real_robust_flight.yaml`?

如果你的仓库已经把 Task4 最终确定为视觉窄门竞速，建议在开源前将该配置文件重命名为：

```text
configs/task4_vision_gate_racing.yaml
```

如果暂时不重命名，也要确保配置文件内容和 README 中的 Task4 描述一致。

---

## 📄 License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

感谢以下开源项目和工具：

- NVIDIA Isaac Sim / Isaac Lab
- Crazyflie / Bitcraze quadrotor asset
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- 机器人强化学习和 Isaac Lab 开源社区

如果这个项目对你有帮助，欢迎参考、修改和继续完善。也欢迎指出代码或文档中的问题。

联系邮箱：2559906288@qq.com  
小红书账号：574661219
