# 🚁 基于 NVIDIA Isaac Lab 的四旋翼无人机强化学习控制框架

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-orange)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-supported-brightgreen)
![RL](https://img.shields.io/badge/RL-skrl%20PPO-purple)
![Platform](https://img.shields.io/badge/platform-Ubuntu%20%7C%20Windows-green)

## 项目摘要

本项目是一个面向四旋翼无人机控制任务的强化学习训练与验证框架，基于 NVIDIA Isaac Lab 构建物理仿真环境，使用 Crazyflie / Quadrotor 类无人机资产作为主要控制对象，围绕悬停稳定、三维航点跟踪、动态障碍导航和视觉窄门穿越四类任务，形成从环境建模、动作映射、观测构造、奖励设计、终止条件、训练流程、模型评估、策略导出到跨仿真验证的完整工程链路。

框架的核心目标是提供一个结构清晰、任务递进、接口统一、便于扩展的无人机强化学习工程模板。项目不是单一任务脚本集合，而是将无人机控制任务拆分为公共动力学层、任务 MDP 层、训练工具层、评估工具层、策略导出层和 sim2sim 验证层。四个任务共享底层动作语义、姿态数学、推力分配、checkpoint 元信息和训练评估流程，同时保留各自独立的场景、观测、奖励和终止逻辑，便于针对不同任务进行调试和扩展。

本项目中的四个任务按照难度逐级递进。Task1 关注基础悬停稳定，主要验证无人机资产加载、root wrench 控制、推力与姿态响应、动作平滑和高度稳定能力；Task2 在悬停基础上加入随机三维航点，使策略学习目标方向、速度控制和连续目标切换；Task3 将解析障碍物世界与真实无人机物理控制结合，引入 lidar、risk features、静态障碍物和动态障碍物，用于研究导航避障与飞行稳定之间的平衡；Task4 使用解析深度图与 compact state 构建视觉窄门穿越任务，用于研究视觉输入、中心线跟踪、门框避障、姿态对齐和连续通过目标的策略学习问题。

框架强调工程一致性和可验证性。动作接口统一采用“策略输出 → 动作清洗 → 比例缩放 → 平滑滤波 → 电机倍率 → 推力/力矩 → 机体系 wrench”的处理链路，避免不同任务之间动作含义不一致造成训练和导出偏差。训练过程中会保存模型权重、skrl 模型状态、normalizer、训练元信息和 `policy_io.json`。其中 `policy_io.json` 用于记录 observation 维度、action 维度、动作语义、控制周期、normalizer 类型和导出信息，是后续 ONNX 导出、Torch/ONNX 输出对齐、sim2sim 回放和部署前一致性检查的重要依据。

框架同时区分两类测试：纯 Python 单元测试和 IsaacLab standalone 环境测试。纯单元测试用于检查动作语义、电机推力、policy metadata、ONNX 导出等不依赖仿真 GUI 的逻辑；环境测试用于检查 IsaacLab 仿真上下文、无人机资产、环境 reset、step 结构、终止事件、world 生成和任务观测是否正常。通过这种分层测试方式，可以在不启动完整训练的情况下尽早发现动作维度、观测维度、坐标系、normalizer、checkpoint 和环境接口问题。

本项目适合用于机器人强化学习工程学习、无人机控制任务建模、Isaac Lab 环境开发、多任务训练流程组织、策略导出和跨仿真验证流程设计。项目不包含真实无人机飞控固件，不提供真实飞行安全保证，也不建议将未经严格验证的策略直接部署到真实设备。若需要进一步开展真实平台部署，应在安全防护、飞控接口、动力学辨识、传感器标定、仿真到现实差距评估、低层控制器保护和紧急停机机制完备的前提下进行。

---

## 框架特点

- 基于 NVIDIA Isaac Lab 构建四旋翼无人机物理仿真任务。
- 采用 Crazyflie / Quadrotor 类无人机资产，使用 root wrench 方式施加推力和力矩。
- 支持四个递进式控制任务：悬停稳定、航点跟踪、动态障碍导航、视觉窄门穿越。
- 统一策略动作语义，避免不同任务之间 action scale、滤波和电机倍率解释不一致。
- 统一四旋翼推力和力矩计算接口，便于调试控制响应和 sim2sim 对齐。
- 统一姿态数学、坐标变换和 observation stacking 工具。
- 训练流程基于 `skrl` PPO，支持 smoke training、正式训练、checkpoint 保存和训练元信息记录。
- 评估流程支持 checkpoint 自动选择、rollout 记录、指标汇总和任务级推理测试。
- 导出流程支持 `policy_io.json`、ONNX 导出、输入输出检查和 Torch/ONNX 输出差异评估。
- sim2sim 模块提供轨迹格式、MuJoCo 回放接口和验证报告结构。
- Ubuntu 和 Windows 脚本保持任务命名对齐，便于跨平台测试、训练和评估。
- 测试体系区分纯单元测试和 IsaacLab standalone 环境测试，降低调试成本。
- 日志体系覆盖 reward components、events、telemetry、PPO 指标和 checkpoint metadata。
- 目录结构面向长期维护，便于后续扩展更多无人机任务、控制接口、观测形式和验证后端。

---

## 适用场景

本项目可以作为以下工作方向的基础工程：

1. **无人机强化学习控制任务开发**  
   用于构建悬停、导航、避障、穿门、轨迹跟踪等控制任务。

2. **Isaac Lab 环境开发学习**  
   用于理解资产加载、场景构建、环境 reset、step、reward、termination 和 vectorized simulation。

3. **多任务训练框架组织**  
   用于参考如何在一个仓库中维护多个递进任务，同时复用公共动作、配置、训练和评估逻辑。

4. **策略导出与部署前检查**  
   用于构建 `policy_io.json`、ONNX 导出、normalizer 检查和输入输出一致性验证流程。

5. **跨仿真验证流程设计**  
   用于开展 Isaac Lab 到 MuJoCo 等后端的 sim2sim 任务迁移、策略回放和动力学差异分析。

6. **机器人 RL 项目工程化展示**  
   用于展示从环境、训练、评估、导出、验证到文档说明的完整工程闭环。

---

## 项目结构

```text
quadrotor_isaaclab_rl/
├── assets/
│   ├── gifs/
│   ├── motions/
│   └── usd/
├── configs/
│   ├── local_paths.example.yaml
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

---

## 目录说明

| 目录 | 作用 |
|---|---|
| `assets/` | 存放展示素材、运动数据占位说明、USD 资产说明等资源文件。 |
| `configs/` | 存放任务配置和本地路径配置模板，便于将任务参数与代码逻辑分离。 |
| `docs/` | 存放设计文档、训练说明、排错说明、checkpoint 说明和任务设计细节。 |
| `scripts/ubuntu/` | Ubuntu 平台下的测试、训练、评估和可视化入口脚本。 |
| `scripts/windows/` | Windows 平台下的测试、训练、评估和可视化入口脚本。 |
| `src/quadrotor_rl/common/` | 通用模型、路径工具、评估辅助、running mean/std 和 skrl wrapper。 |
| `src/quadrotor_rl/core/` | 框架公共核心层，包括配置、姿态数学、动作语义、动力学映射、环境工具和场景工具。 |
| `src/quadrotor_rl/tasks/` | 四个任务的具体 MDP 实现，包括 config、scene、env、train 和 model test。 |
| `src/quadrotor_rl/training/` | 训练工具层，包括日志、PPO 参数、数值安全、runner 和 checkpoint 保存逻辑。 |
| `src/quadrotor_rl/evaluation/` | 评估工具层，包括 checkpoint 选择、评估指标、rollout 记录和评估 runner。 |
| `src/quadrotor_rl/export/` | 策略导出层，包括 policy IO、ONNX 导出、输入输出检查和 Torch/ONNX 对齐。 |
| `src/quadrotor_rl/sim2sim/` | 跨仿真验证层，包括轨迹格式、MuJoCo 模型接口、回放脚本和报告工具。 |
| `tests/core/` | 不依赖 IsaacLab 的纯 Python 单元测试。 |
| `tests/export/` | 策略元信息、checkpoint 和 ONNX 导出相关单元测试。 |
| `tests/taskX/` | 依赖 IsaacLab 的 standalone 环境测试和 world 测试。 |

---

## 任务总览

| 任务 | 名称 | 目标 | 环境特点 | 训练重点 |
|---|---|---|---|---|
| Task1 | 悬停稳定 | 在目标高度附近保持稳定飞行 | 单机体、固定目标高度、root wrench 控制 | 高度稳定、姿态稳定、动作平滑 |
| Task2 | 航点跟踪 | 连续跟踪随机三维目标点 | 目标点刷新、相对位置观测、速度引导 | 目标接近、速度控制、稳定移动 |
| Task3 | 动态障碍导航 | 在障碍物环境中到达目标点 | 解析障碍物世界、lidar、risk features | 避障、安全距离、目标推进 |
| Task4 | 视觉窄门穿越 | 根据深度图和状态信息穿越窄门 | 解析深度图、gate world、CNN policy | 中心线跟踪、门中心对齐、视觉安全 |

---

## 框架分层设计

### 公共核心层

公共核心层位于 `src/quadrotor_rl/core/`，主要负责无人机任务之间可以复用的底层逻辑。该层不直接决定某个任务的奖励和终止条件，而是提供各任务共享的配置结构、姿态计算、动作处理和动力学映射能力。

核心模块包括：

- `config/`：基础配置和任务配置 schema；
- `math/`：四元数、欧拉角、坐标系转换和姿态工具；
- `physics/`：动作语义、电机倍率、推力计算和 wrench 分配；
- `env/`：环境状态读取、reset 辅助、事件工具和 observation buffer；
- `scene/`：场景构建和资产加载相关工具。

这种分层可以减少多个任务之间的重复代码，使动作、姿态、推力和环境工具具有统一解释。对于强化学习任务而言，这一点非常重要，因为如果不同任务中的动作缩放、滤波或观测缓存含义不一致，后续 warm-start、导出、评估和 sim2sim 都可能出现隐性错误。

### 任务 MDP 层

任务 MDP 层位于 `src/quadrotor_rl/tasks/`。每个任务保留独立目录，包含该任务的配置、场景、环境、训练和模型测试入口。

每个任务主要负责：

- 定义任务目标；
- 构建任务场景；
- 生成任务特有 observation；
- 计算 reward components；
- 判断 termination 和 truncation；
- 提供训练入口；
- 提供模型推理测试入口。

Task1–Task4 的任务难度逐级提高，但底层无人机动作控制接口保持一致。这样既可以保证任务之间具有连续性，也方便从前序任务 checkpoint 迁移到后续任务。

### 训练工具层

训练工具层位于 `src/quadrotor_rl/training/`，用于封装与具体任务无关的训练辅助逻辑。

该层包含：

- PPO 配置构造；
- 训练日志格式化；
- 数值安全检查；
- NaN / Inf 清理；
- checkpoint 保存；
- normalizer 保存；
- policy metadata 保存；
- skrl runner 相关辅助函数。

训练工具层的目标是让每个任务的 `taskX_train.py` 更聚焦任务本身，而不是重复维护大量训练日志、checkpoint、resume 和配置转换代码。

### 评估工具层

评估工具层位于 `src/quadrotor_rl/evaluation/`，用于统一模型评估与 rollout 统计。

该层包含：

- checkpoint 路径解析；
- final checkpoint 自动选择；
- rollout 记录；
- 任务指标聚合；
- 成功率、碰撞率、动作幅值、姿态误差、目标误差等指标计算；
- eval report 结构。

评估层用于回答“策略是否稳定”“任务是否完成”“动作是否过大”“是否存在碰撞或偏离”“是否适合导出和进一步验证”等问题，而不是只依赖训练 reward 判断模型质量。

### 策略导出层

策略导出层位于 `src/quadrotor_rl/export/`，用于连接训练模型和部署前验证流程。

该层主要包含：

- `policy_io.json` 生成和检查；
- checkpoint 中 actor 权重解析；
- observation normalizer 处理；
- ONNX 导出；
- Torch 与 ONNX 输出对齐；
- 输入维度、输出维度和 action range 检查。

`policy_io.json` 是该层的核心文件。它用于明确记录策略输入输出协议，避免后续模型导出时出现“训练时 observation 是 normalized，但推理时输入 raw observation”“动作维度正确但动作缩放错误”“checkpoint 文件选错”等常见问题。

### 跨仿真验证层

跨仿真验证层位于 `src/quadrotor_rl/sim2sim/`，用于支持从 Isaac Lab 到其他仿真后端的策略验证。

该层包含：

- 统一轨迹数据格式；
- MuJoCo quadrotor 模型接口；
- ONNX policy replay；
- sim2sim 指标报告；
- rollout 对比结构。

当前 sim2sim 主要面向最小闭环验证和接口标准化，后续可以继续扩展动力学参数辨识、闭环 MuJoCo rollout、Isaac/MuJoCo trajectory comparison 和失败案例分析。

---

## 动作语义说明

无人机控制任务中，动作语义是否统一会直接影响训练稳定性、checkpoint 迁移、ONNX 导出和跨仿真验证。本框架采用统一动作处理链路：

```text
policy raw action
→ clamp / sanitize
→ scale
→ exponential moving average filter
→ motor multiplier
→ rotor thrust
→ body force and torque
→ IsaacLab root wrench
```

其中：

- `policy raw action` 是策略网络输出；
- `sanitize` 用于处理 NaN、Inf 和异常动作值；
- `scale` 控制策略动作对电机倍率的影响范围；
- `EMA filter` 用于抑制高频动作抖动；
- `motor multiplier` 表示相对悬停推力的电机倍率；
- `rotor thrust` 表示四个旋翼对应推力；
- `body wrench` 包含机体系总推力和 roll / pitch / yaw 力矩；
- `root wrench` 是最终施加到 IsaacLab 机体上的外力和外力矩。

统一动作语义可以保证 Task1、Task2、Task3 和 Task4 在底层控制上具有可比性，也方便将前序任务策略迁移到后续任务。

---

## 观测设计说明

四个任务的 observation 设计遵循“基础状态 + 任务目标 + 安全感知 + 历史动作”的原则。

基础状态通常包括：

- 相对高度；
- 机体线速度；
- 机体角速度；
- 姿态角；
- 重力方向投影；
- 上一步动作；
- 任务阶段或目标相关特征。

任务特有信息包括：

- Task1：目标高度和悬停误差；
- Task2：目标相对位置、目标距离和 lookahead 信息；
- Task3：lidar 距离、障碍物风险、目标方向和安全距离；
- Task4：深度图、门中心误差、中心线距离和 compact state。

历史观测或历史动作可以帮助策略从有限状态中估计运动趋势，改善速度、姿态和动作平滑控制。

---

## 奖励设计说明

奖励函数不是单纯追求单项指标最大化，而是需要在任务完成、飞行稳定、安全约束和动作质量之间建立平衡。

常见奖励项包括：

- 目标接近奖励；
- 高度保持奖励；
- 姿态稳定奖励；
- 速度方向奖励；
- 动作平滑奖励；
- 通过目标奖励；
- 避障安全奖励；
- 碰撞惩罚；
- 坠机惩罚；
- 偏离场地惩罚；
- 超时处理。

训练过程中应结合 reward components、events 和 telemetry 判断问题来源。例如，目标距离下降但 crash rate 上升，通常说明策略学会了快速接近目标但没有稳定飞行；动作幅值长期饱和，说明动作缩放、奖励权重或控制接口可能需要检查；success rate 不上升但 progress 为正，可能说明终止条件或目标到达判定过严。

---

## 终止条件说明

合理的 termination 和 truncation 对强化学习训练非常重要。本框架中的终止条件通常包含：

- 飞行高度过低；
- 姿态倾角过大；
- 角速度异常；
- 位置偏离场地范围；
- 与障碍物或门框碰撞；
- 漏门或严重偏离中心线；
- episode 达到最大长度；
- 达成任务成功条件。

终止条件不应只用于结束 episode，也应通过 events 写入日志，用于后续分析失败类型。对于无人机任务而言，常见失败类型包括坠机、翻转、动作饱和、目标震荡、绕圈、避障失败和视觉对齐失败。

---

## 环境要求

项目需要在可运行 Isaac Lab 的 Python 环境中使用。建议先完成 NVIDIA Isaac Sim / Isaac Lab 官方安装，并确认以下命令正常：

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import isaaclab; print('Isaac Lab import ok')"
```

常用 Python 依赖：

```bash
pip install skrl tensorboard tqdm numpy matplotlib
```

如需使用 ONNX 导出和输出对齐检查：

```bash
pip install onnx onnxruntime
```

不同 Isaac Sim、Isaac Lab、PyTorch、CUDA 和驱动版本之间存在兼容关系。实际环境应以官方文档和本地运行条件为准。README 不绑定具体工作站、显卡型号或个人路径。

---

## 快速开始

### 克隆仓库

```bash
git clone https://github.com/0324Lw/NVIDIA--Isaac-Lab-Quadrotor-control quadrotor_isaaclab_rl
cd quadrotor_isaaclab_rl
```

### 设置 Python 路径

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Ubuntu 和 Windows 脚本会自动识别项目根目录并设置 `PYTHONPATH`，通常可以直接使用脚本运行。

### 检查项目结构

```bash
bash scripts/ubuntu/checkProjectStructure.sh
```

### 检查无人机资产和控制接口

```bash
bash scripts/ubuntu/testQuadrotorAssetControl.sh
```

该测试用于确认无人机资产加载、质量估计、root wrench 控制和基础动力学响应是否正常。

---

## 测试方法

### 纯单元测试

`tests/core/` 和 `tests/export/` 属于纯 Python 单元测试，不需要启动完整 IsaacLab 仿真应用。

```bash
cd /path/to/quadrotor_isaaclab_rl

PYTHONPATH="$PWD/src" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/core/test_action_semantics.py \
  tests/core/test_legacy_action_equivalence.py \
  tests/export/test_policy_io.py \
  tests/export/test_export_onnx_checkpoint.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免系统中 ROS、launch testing 或其他 pytest 插件污染当前 Python 环境。

### IsaacLab 环境测试

Task 环境测试需要启动 IsaacLab 仿真上下文，应通过脚本运行：

```bash
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/testTask2Environment.sh
bash scripts/ubuntu/testTask3World.sh
bash scripts/ubuntu/testTask3Environment.sh
bash scripts/ubuntu/testTask4World.sh
bash scripts/ubuntu/testTask4Environment.sh
```

不要直接使用 pytest 收集 `tests/taskX/taskX_env_test.py`。这些文件属于 standalone IsaacLab 测试程序，脚本会按照正确方式启动和释放仿真资源。

---

## 训练方法

### Smoke 训练

Smoke 训练用于检查训练入口、环境 step、日志写入、checkpoint 保存和模型 forward 是否正常。该训练不用于评价最终策略质量。

```bash
bash scripts/ubuntu/trainTask1HoverSmoke.sh
bash scripts/ubuntu/trainTask2WaypointSmoke.sh
bash scripts/ubuntu/trainTask3ObstacleSmoke.sh
bash scripts/ubuntu/trainTask4GateSmoke.sh
```

可以通过环境变量调整并发数和训练步数：

```bash
NUM_ENVS=1024 TOTAL_ENV_STEPS=20000 bash scripts/ubuntu/trainTask2WaypointSmoke.sh
```

### 正式训练

```bash
bash scripts/ubuntu/trainTask1Hover.sh
bash scripts/ubuntu/trainTask2Waypoint.sh
bash scripts/ubuntu/trainTask3Obstacle.sh
bash scripts/ubuntu/trainTask4Gate.sh
```

推荐训练顺序：

```text
Task1 悬停稳定
→ Task2 航点跟踪
→ Task3 动态障碍导航
→ Task4 视觉窄门穿越
```

Task2、Task3 和 Task4 可以根据实验需要从前序任务 checkpoint 进行 warm-start，也可以从零开始训练。

---

## 评估方法

评估脚本会默认查找对应任务日志目录下最近的 `final_checkpoint`。也可以通过 `CHECKPOINT` 环境变量显式指定模型目录。

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

评估阶段建议重点关注：

- 平均 episode 长度；
- success rate；
- crash rate；
- goal distance；
- position error；
- action magnitude；
- action rate；
- motor saturation ratio；
- obstacle collision rate；
- gate pass count；
- depth safety；
- rollout 是否出现 NaN / Inf。

---

## 可视化方法

可视化脚本用于加载 checkpoint 并运行带渲染的策略推理。

```bash
CHECKPOINT=logs/task1/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask1Hover.sh
CHECKPOINT=logs/task2/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask2Waypoint.sh
CHECKPOINT=logs/task3/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask3Obstacle.sh
CHECKPOINT=logs/task4/<run_name>/final_checkpoint bash scripts/ubuntu/visual/visualizeTask4Gate.sh
```

可视化运行需要图形环境和 Isaac Sim 渲染支持。如果在无显示环境中运行，应优先使用 headless 环境测试、训练和评估脚本。

---

## Task1：悬停稳定

### 任务目标

Task1 是最基础的无人机控制任务，用于训练策略在固定目标高度附近保持稳定飞行。该任务主要验证无人机资产、root wrench 控制接口、推力响应、姿态稳定和动作平滑能力。

### 任务设计

Task1 使用单个无人机资产，目标点固定在预设高度附近。策略输出四个旋翼修正动作，环境将动作转换为电机倍率，再计算总推力和机体系力矩，最终通过 root wrench 施加到机体上。

观测通常包含：

- 当前高度误差；
- 机体线速度；
- 机体角速度；
- 重力方向投影；
- roll / pitch / yaw 姿态信息；
- 历史动作；
- episode 进度相关特征。

奖励主要关注：

- 高度误差降低；
- 姿态角保持稳定；
- 垂向速度不过大；
- 水平速度不过大；
- 动作幅值和动作变化率不过大；
- 避免坠机和翻转。

### 常用命令

```bash
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/trainTask1HoverSmoke.sh
bash scripts/ubuntu/trainTask1Hover.sh
bash scripts/ubuntu/evaluateTask1Hover.sh
```

---

## Task2：航点跟踪

### 任务目标

Task2 在悬停稳定基础上加入三维航点目标。无人机需要在保持飞行稳定的同时向目标点移动，并在到达当前目标后切换到下一个目标。

### 任务设计

Task2 的观测包含无人机自身状态和目标相对信息。目标点可以在空间范围内随机生成，策略需要根据目标方向和距离调整推力分布，使无人机逐步接近目标。

奖励主要关注：

- 目标距离下降；
- 朝目标方向的速度分量；
- 到达目标点；
- 高度保持；
- 姿态稳定；
- 动作平滑；
- 避免偏离场地范围。

Task2 是从基础悬停到导航控制的重要过渡任务，可以用于检查策略是否真正学会了稳定移动，而不仅仅是在固定点附近悬停。

### 常用命令

```bash
bash scripts/ubuntu/testTask2Environment.sh
bash scripts/ubuntu/trainTask2WaypointSmoke.sh
bash scripts/ubuntu/trainTask2Waypoint.sh
bash scripts/ubuntu/evaluateTask2Waypoint.sh
```

---

## Task3：动态障碍导航

### 任务目标

Task3 将目标导航与障碍物避让结合。无人机需要在包含静态障碍物和动态障碍物的场景中，根据目标点、lidar 信息和风险特征完成安全导航。

### 任务设计

Task3 使用“真实无人机物理控制 + 解析障碍物世界”的结构。障碍物、目标、lidar、风险特征和碰撞检测主要由 tensor 计算完成，避免在仿真场景中生成大量 USD prim，从而降低仿真开销，提高并发训练效率。

观测通常包含：

- 无人机基础状态；
- 目标相对位置；
- 目标距离；
- lidar rays；
- 最近障碍物距离；
- risk features；
- 历史动作。

奖励主要关注：

- 朝目标推进；
- 与障碍物保持安全距离；
- 避免碰撞；
- 维持稳定高度和姿态；
- 不发生过大动作；
- 不偏离有效飞行区域。

Task3 训练过程中需要同时观察目标距离和安全指标。如果目标距离下降但碰撞率上升，说明策略可能通过激进行为换取进度；如果安全距离稳定但目标推进很慢，说明避障惩罚可能过强或目标奖励不足。

### 常用命令

```bash
bash scripts/ubuntu/testTask3World.sh
bash scripts/ubuntu/testTask3Environment.sh
bash scripts/ubuntu/trainTask3ObstacleSmoke.sh
bash scripts/ubuntu/trainTask3Obstacle.sh
bash scripts/ubuntu/evaluateTask3Obstacle.sh
```

---

## Task4：视觉窄门穿越

### 任务目标

Task4 面向视觉输入下的连续穿门控制任务。无人机需要根据解析深度图和 compact state，沿中心线接近目标门，并避免与门框发生碰撞。

### 任务设计

Task4 使用 gate world 生成窄门布局、中心线和深度观测。策略输入由深度图和低维状态组成，适合采用 CNN encoder 处理视觉输入，再与 compact state 特征融合输出四旋翼动作。

观测包括：

- depth image；
- 当前目标门索引；
- 门中心相对位置；
- 中心线偏差；
- 姿态和速度；
- 历史动作或动作统计。

奖励主要关注：

- 接近目标门；
- 对准门中心；
- 沿中心线飞行；
- 保持深度安全；
- 成功通过门；
- 避免门框碰撞；
- 避免漏门和严重偏离。

Task4 的训练难度明显高于 Task1–Task3。Smoke 训练只用于验证视觉观测、CNN forward、环境 step 和 checkpoint 保存，不代表策略已经具备稳定穿门能力。

### 常用命令

```bash
bash scripts/ubuntu/testTask4World.sh
bash scripts/ubuntu/testTask4Environment.sh
bash scripts/ubuntu/trainTask4GateSmoke.sh
bash scripts/ubuntu/trainTask4Gate.sh
bash scripts/ubuntu/evaluateTask4Gate.sh
```

---

## 日志与模型保存

训练日志默认保存在：

```text
logs/task1/
logs/task2/
logs/task3/
logs/task4/
```

典型 checkpoint 目录结构：

```text
final_checkpoint/
├── quadrotor_taskX_model.pt
├── quadrotor_taskX_skrl_model.pt
├── _observation_preprocessor.pt
├── _state_preprocessor.pt
├── _value_preprocessor.pt
├── train_metadata.pt
└── policy_io.json
```

训练日志通常包含：

- reward components；
- success / crash / collision events；
- telemetry；
- PPO loss；
- KL divergence；
- entropy；
- learning rate；
- action statistics；
- checkpoint metadata。

使用 TensorBoard 查看训练曲线：

```bash
tensorboard --logdir logs
```

---

## Policy IO 文件说明

`policy_io.json` 用于描述策略输入输出协议，是策略导出和跨仿真验证的重要文件。

示例结构：

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

该文件用于确认：

- observation 维度是否正确；
- action 维度是否正确；
- action scale 是否一致；
- 控制周期是否一致；
- normalizer 是否存在；
- ONNX 推理输入是 raw observation 还是 normalized observation；
- checkpoint 是否与任务匹配。

---

## ONNX 导出与对齐检查

### 检查 Policy IO

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.export.check_policy_io \
  --policy-io logs/task1/<run_name>/final_checkpoint/policy_io.json
```

### 导出 ONNX

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.export.export_onnx \
  --checkpoint logs/task1/<run_name>/final_checkpoint \
  --output exports/task1/policy.onnx
```

### 比较 Torch 与 ONNX 输出

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.export.compare_torch_onnx \
  --checkpoint logs/task1/<run_name>/final_checkpoint \
  --onnx exports/task1/policy.onnx
```

导出前应确认：

- checkpoint 目录完整；
- `policy_io.json` 存在；
- normalizer 文件存在或 sidecar 中明确说明；
- observation sample 与策略输入维度一致；
- Torch 与 ONNX 输出误差在可接受范围内。

---

## Sim2Sim 验证

sim2sim 模块用于支持跨仿真策略验证。基本流程包括：

1. 在 Isaac Lab 中训练策略；
2. 保存 checkpoint 和 `policy_io.json`；
3. 导出 ONNX；
4. 检查 Torch 与 ONNX 输出一致性；
5. 在目标仿真后端加载无人机模型；
6. 使用同一动作语义执行 closed-loop 或 replay；
7. 统计轨迹误差、姿态误差、动作范围和失败事件。

运行示例：

```bash
PYTHONPATH="$PWD/src" python -m quadrotor_rl.sim2sim.mujoco_replay \
  --task task1 \
  --onnx exports/task1/policy.onnx
```

sim2sim 的重点不是直接证明策略可以真实部署，而是检查以下问题：

- observation 输入是否一致；
- action scale 是否一致；
- normalizer 是否一致；
- 控制周期是否一致；
- 推力模型是否一致；
- 坐标系是否一致；
- 策略输出是否稳定；
- 跨物理引擎后是否出现明显发散。

---

## Ubuntu 使用说明

Ubuntu 脚本位于：

```text
scripts/ubuntu/
```

常用命令：

```bash
bash scripts/ubuntu/checkProjectStructure.sh
bash scripts/ubuntu/testQuadrotorAssetControl.sh
bash scripts/ubuntu/testTask1Environment.sh
bash scripts/ubuntu/trainTask1HoverSmoke.sh
bash scripts/ubuntu/evaluateTask1Hover.sh
```

Ubuntu 脚本会自动识别项目根目录，并设置必要的 Python 路径和线程限制。若在无 GUI 环境中运行，应优先使用 headless 环境测试、训练和评估脚本。

---

## Windows 使用说明

Windows 脚本位于：

```text
scripts/windows/
```

常用命令示例：

```powershell
.\scripts\windows\checkProjectStructure.ps1
.\scripts\windows\testTask1Environment.ps1
.\scripts\windows\trainTask1HoverSmoke.ps1
.\scripts\windows\evaluateTask1Hover.ps1
```

Windows 运行时需要确保 Isaac Lab 对应 Python 环境、CUDA、PyTorch 和路径配置正确。不同安装方式可能需要调整本地 Python 解释器路径，但不应将个人路径写入仓库主配置。

---

## 资源调度建议

无人机任务的计算压力与任务复杂度、并发环境数、是否启用视觉观测、是否渲染 GUI 密切相关。

建议遵循以下原则：

- 先运行 asset test 和 environment test；
- 先进行小规模 smoke training；
- 确认无 NaN、无保存错误、无维度错误后再增加并发数；
- Task4 由于包含视觉观测，显存和计算压力更高；
- 可视化和训练不要同时开启过高并发；
- 训练脚本中限制常见 BLAS / OpenMP 线程数，避免 CPU 线程过度占用；
- 若出现显存不足，优先降低 `NUM_ENVS` 和视觉任务 batch 规模。

示例：

```bash
NUM_ENVS=128 TOTAL_ENV_STEPS=10000 bash scripts/ubuntu/trainTask4GateSmoke.sh
```

---

## 常见问题

### `ModuleNotFoundError: No module named quadrotor_rl`

确认在项目根目录运行，并设置：

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

或直接使用 `scripts/ubuntu/` 下的脚本。

### pytest 报 `No module named lark`

系统中可能存在 ROS 或其他 pytest 插件污染。运行纯单元测试时使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ...
```

### pytest 报 `fixture 'env' not found`

不要用 pytest 直接运行 `tests/taskX/taskX_env_test.py`。这些是 standalone IsaacLab 环境测试，应通过脚本运行：

```bash
bash scripts/ubuntu/testTask1Environment.sh
```

### checkpoint 保存时报 JSON 序列化错误

检查 `policy_io.json` 生成逻辑是否使用 JSON-safe 转换。Tensor、numpy array、Path、dataclass 等对象应在写入 JSON 前转换为基础类型。

### 训练 reward 上升但成功率很低

可能原因包括 success 判定过严、终止条件过早、reward 权重偏向局部目标、动作饱和或目标采样过难。建议同时查看 events、telemetry 和 reward components。

### Task3 避障策略容易停在原地

可能是避障惩罚过强或目标推进奖励不足。建议同时观察 `Goal_Dist`、`Progress`、`Min_Lidar` 和 `Obstacle_Collision_Rate`。

### Task4 视觉任务训练很慢

视觉观测任务本身更难，且 CNN policy 计算开销更高。建议先确认 world test 和 environment test 通过，再进行小规模 smoke training，最后逐步增加训练规模。

---

## 开发与维护建议

建议按照以下顺序修改框架：

1. 修改动作语义前，先运行 `tests/core`；
2. 修改 policy IO 或 ONNX 导出前，先运行 `tests/export`；
3. 修改任务环境前，先运行对应 `testTaskXEnvironment.sh`；
4. 修改 Task3 / Task4 world 前，先运行 world test；
5. 修改 reward 后，不直接看单次 reward，应结合 success、crash、collision 和 action 指标；
6. 修改 observation 后，同步检查 obs dim、policy IO 和 model input；
7. 修改 checkpoint 逻辑后，确认 `final_checkpoint` 和 `policy_io.json` 可正常生成；
8. 导出 ONNX 前，先确认 Torch checkpoint 可正常加载；
9. sim2sim 前，先检查 action scale、normalizer 和控制周期；
10. 每次大改前保存旧 checkpoint 和关键日志，便于对比。

---

## 当前限制

- 本项目主要面向仿真训练和工程验证，不提供真实无人机部署安全保证。
- Task3 和 Task4 使用解析 world，障碍物和 gate 不一定生成真实 USD prim。
- Task4 使用解析深度图，后续可以扩展为真实 Isaac camera rendering。
- sim2sim 模块提供验证接口和基本回放结构，完整动力学对齐仍需进一步扩展。
- 不同 Isaac Lab / Isaac Sim 版本之间可能存在 API 差异。
- 训练结果受随机种子、并发数、物理参数、reward 权重和训练步数影响。
- 真实飞行部署需要额外的飞控接口、安全保护、动力学辨识、传感器标定和应急机制。

---

## 许可证

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## 致谢

本项目基于以下开源工具和社区生态构建：

- NVIDIA Isaac Sim / Isaac Lab
- Crazyflie / Bitcraze ecosystem
- PyTorch
- skrl reinforcement learning library
- TensorBoard
- tqdm
- Open-source robotics and reinforcement learning communities
