
# PX4 四旋翼 Gym 环境 + Tianshou PPO 训练与测试

本项目提供一个基于 Gymnasium 的 PX4 四旋翼仿真环境（QuadcopterEnv），并给出使用 Tianshou 的 PPO 算法进行训练与评估的完整脚本。

- 环境：四旋翼动力学 + 轨迹生成 + PID 控制器
- 算法：PPO（优势函数、GAE、熵正则、梯度裁剪等）
- 特性：滑动窗口奖励、课程学习、日志与数据导出

---

## 目录结构

建议的项目目录（关键文件）：

```
project/
├─ envs/
│  └─ quadcopter_env.py           # 你的 QuadcopterEnv（Gym 环境）
├─ quadFiles/ ...                 # 四旋翼相关模块
├─ utils/ ...                     # 工具函数、风模型等
├─ train_px4_ppo.py               # 训练与测试脚本（PPO）
├─ README.md                      # 本文件
└─ requirements.txt               # 依赖
```

---

## 环境依赖

- Python 3.9+
- Gymnasium
- NumPy
- PyTorch (CUDA 可选)
- Tianshou >= 0.5.0
- Matplotlib（用于数据渲染）
- WandB（可选，用于云端日志）

### 安装

```bash
# 建议使用虚拟环境
conda create -n px4-ppo python=3.11 -y
conda activate px4-ppo

# 安装核心依赖
pip install -U pip
pip install gymnasium numpy torch tianshou matplotlib

# 可选：WandB
pip install wandb
```

> 如果你有 `requirements.txt`，直接 `pip install -r requirements.txt`。

---

## 环境说明（QuadcopterEnv）

- 动作空间：连续 Box，shape=(20, )，范围 \([0, 1]\)，映射到 PID 参数安全区间
- 观测空间：连续向量，shape=(24, )，包含参考与反馈的 [位置/速度/姿态/角速度]
- 奖励：密集奖励（每仿真子步）+ 滑动窗口分段奖励（近 3s）
- 终止条件：步数达上限或位置/姿态越界
- 渲染：
  - `mode='data_log'`：输出列式 CSV，包含整回合数据
  - `mode='human'`：静态图和动画（需要 GUI 环境）

---

## 训练脚本（train_px4_ppo.py）

此脚本基于 Tianshou 的 PPOPolicy，适配 QuadcopterEnv，支持：
- 向量化环境并行采样（SubprocVectorEnv）
- 课程学习（可选）
- 学习率线性衰减（可选）
- TensorBoard/WandB 日志
- 定期保存 checkpoint 与 Best Policy
- 评估与数据导出

### 运行方式

```bash
# 基础训练（推荐）
python train_px4_ppo.py --training-num 8 --test-num 4 --epoch 500

# 启用课程学习
python train_px4_ppo.py --use-curriculum --curriculum-stages 5 --training-num 8

# 使用 WandB 记录
python train_px4_ppo.py --logger wandb --wandb-project px4-pid-optimization

# 恢复训练（从 checkpoint 或 best policy）
python train_px4_ppo.py --resume-path log_px4/px4_quad/ppo/0/<timestamp>/best_policy.pth
```

---

## 常用参数

### 环境相关
- `--control-type`：控制模式，默认 `xyz_pos`，可选 `xy_vel_z_pos`、`xyz_vel`
- `--trajectory-type`：轨迹类型（0-13），默认 5
- `--yaw-type`：偏航控制类型（0-4），默认 3
- `--max-episode-steps`：每回合最大步数（RL 步），默认 2000
- `--rl-dt`：RL 决策步长（秒），默认 1.0（每个 step 执行多个仿真子步）
- `--sim-dt`：仿真积分步长（秒），默认 0.005
- `--segment-window`：滑动窗口奖励时长（秒），默认 3.0
- `--wind-type`：风扰类型，默认 `None`
- `--wind-strength`：风强度，默认 2.0

### 训练与 PPO
- `--training-num`：并行训练环境数，默认 8
- `--test-num`：测试环境数，默认 4
- `--buffer-size`：回放缓冲区大小（on-policy 也会用到），默认 20000
- `--hidden-sizes`：网络隐藏层维度，默认 `[256, 256]`
- `--lr`：学习率，默认 3e-4
- `--gamma`：折扣因子，默认 0.99
- `--epoch`：训练轮数，默认 500
- `--step-per-epoch`：每个 epoch 的环境步数，默认 100000
- `--step-per-collect`：每次收集的步数，默认 4096
- `--repeat-per-collect`：每收集一次的更新次数，默认 10
- `--batch-size`：批大小，默认 256
- `--eps-clip`：PPO 概率比裁剪阈值，默认 0.2
- `--vf-coef`：价值函数损失权重，默认 0.5
- `--ent-coef`：熵正则化，默认 0.01
- `--gae-lambda`：GAE 参数，默认 0.95
- `--value-clip`：是否对价值函数做裁剪，默认 1
- `--norm-adv`：优势归一化，默认 1
- `--recompute-adv`：重新计算优势（提高稳定性），默认 1
- `--lr-decay`：线性学习率衰减，默认 1
- `--max-grad-norm`：梯度裁剪，默认 0.5

### 日志与导出
- `--logdir`：日志根目录，默认 `log_px4`
- `--logger`：`tensorboard` 或 `wandb` 或 `both`（若脚本支持），默认 `tensorboard`
- `--render`：测试时渲染间隔（秒），默认 0（不渲染）
- `--save-interval`：每 N 个 epoch 保存一次 checkpoint，默认 10
- `--watch`：仅测试（不训练），默认 False
- `--save-video`：测试后保存数据日志（CSV），默认 False
- `--resume-path`：加载的模型路径
- `--resume-id`：日志 run id（与 logger 相关）

---

## 日志与模型文件

脚本将创建类似如下的目录结构：

```
log_px4/
└─ px4_quad/
   └─ ppo/
      └─ 0/
         └─ 250101-120000/          # 时间戳
            ├─ events.out.tfevents* # TensorBoard 日志
            ├─ best_policy.pth      # 表现最佳的策略
            ├─ checkpoint_epoch_XX.pth
            └─ final_policy.pth
```

- `best_policy.pth`：训练过程中自动保存的最佳策略（基于测试回报）
- `checkpoint_epoch_XX.pth`：按 `--save-interval` 定期保存
- `final_policy.pth`：训练结束后的最终策略

> 使用 TensorBoard 查看日志：
> ```bash
> tensorboard --logdir log_px4
> ```

---

## 仅测试与数据导出

```bash
# 仅测试，不训练
python train_px4_ppo.py --watch --resume-path <path-to-policy.pth> --test-num 10

# 测试并保存完整一集的飞行数据（CSV）
python train_px4_ppo.py --watch --save-video --resume-path <path-to-policy.pth>
```

保存的 CSV 位于当前运行的日志目录下的一个 `episode_<timestamp>` 子目录中，包含整回合时序数据，便于分析或绘图。

---

## 课程学习（可选）

启用课程学习后，训练会从简单任务逐步过渡到复杂任务。脚本中提供了一个简单的 `CurriculumManager` 示例，它会：
- 根据测试回报估算一个“成功率”
- 在达到阈值（`--stage-threshold`）时切换到下一个阶段
- 自动调用环境的 `set_course_stage(stage)` 更新任务配置

使用方式：

```bash
python train_px4_ppo.py --use-curriculum --curriculum-stages 5 --stage-threshold 0.75
```

> 你可以在 `CurriculumManager` 中自定义每个阶段使用的 `trajectory_type` 与 `control_type`。

---

## 常见问题（FAQ）

- **显存/内存不足**：适当降低 `--training-num`、`--step-per-collect`、`--batch-size`。
- **训练不收敛**：
  - 检查动作是否已在环境中映射到 PID 安全范围（脚本已设置 `action_scaling=True` 且 `unbounded=False`）。
  - 降低 `--ent-coef` 或增大 `--vf-coef`。
  - 减小 `--eps-clip`，或启用 `--value-clip 1`。
  - 减小风强度，简化轨迹，或启用课程学习。
- **训练太慢**：使用 SubprocVectorEnv（默认），增大 `--training-num`，或开启 CUDA。
- **日志看不到**：确认 `--logdir` 权限；TensorBoard 指向根目录 `log_px4`。

---

## 复现建议的默认参数

```bash
python train_px4_ppo.py \
  --training-num 8 --test-num 4 \
  --epoch 500 --step-per-epoch 100000 \
  --step-per-collect 4096 --repeat-per-collect 10 \
  --batch-size 256 --hidden-sizes 256 256 \
  --ent-coef 0.01 --vf-coef 0.5 --eps-clip 0.2 \
  --value-clip 1 --norm-adv 1 --recompute-adv 1 \
  --lr 3e-4 --lr-decay 1
```

---

## 许可证

- 环境代码遵循你给出的许可证声明（MIT）。
- 本训练脚本及 README 默认为 MIT 开源（可按需更改）。

---

## 致谢

- [Tianshou](https://github.com/thu-ml/tianshou)
- Gymnasium / PyTorch 社区贡献
- PX4 相关研究和开源项目
