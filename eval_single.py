#!/usr/bin/env python3
"""
✅ Quadcopter PPO 策略评估脚本（离线渲染完整版）
包含：
1. 主状态图表（位置、速度、姿态、误差）
2. PID 增益详细波形图（20 路）
3. PID 增益摘要图（热力图、柱状图、关键波形）
4. PID 增益统计分析图
"""
import time
import torch
import numpy as np
import random
import os
import json
from datetime import datetime
from torch import nn
from torch.distributions import Independent, Normal
from tianshou.policy import PPOPolicy
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from px4Env import QuadcopterEnv

# ============================================================
# 1️⃣ 加载 checkpoint
# ============================================================
ckpt_path = "./logs/log_train_results/1001/251115-195805/best_policy.pth"
print("=" * 80)
print(f"📂 加载模型: {ckpt_path}")
print("=" * 80)

ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt.get('args', None) if isinstance(ckpt, dict) else None
# args = ckpt["args"]
model_state = ckpt["model"] if 'model' in ckpt else ckpt['state_dict']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 如果 args 为 None，使用空字典作为默认值并从模型权重推断网络结构
if args is None:
    args = {}
    print("⚠️  检查点中未找到 args，尝试加载 config.json 或从模型权重推断配置")

    # 先尝试读取同目录下的 config.json
    config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                args.update(cfg)
                print(f"   ✅ 已加载配置: {config_path}")
        except Exception as cfg_err:
            print(f"   ⚠️  读取配置失败: {cfg_err}")
    else:
        print(f"   ⚠️  未找到配置文件: {config_path}")
    
    # 从 actor 网络权重推断隐藏层大小
    if 'hidden_sizes' not in args:
        if 'actor.preprocess.model.model.0.weight' in model_state:
            hidden_size = model_state['actor.preprocess.model.model.0.weight'].shape[0]
            args['hidden_sizes'] = [hidden_size, hidden_size]
            print(f"   ✅ 推断隐藏层大小: {args['hidden_sizes']}")
        else:
            args['hidden_sizes'] = [128, 128]  # 默认值
            print(f"   ⚠️  无法推断隐藏层大小，使用默认值: {args['hidden_sizes']}")

print(f"✅ 成功加载检查点")
print(f"   Device: {device}")
print(f"   训练轮数: {ckpt.get('epoch', 'Unknown')}")

# ============================================================
# 2️⃣ 设置随机种子
# ============================================================
seed = args.get("seed", 1001)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"🌱 随机种子: {seed}")

# ============================================================
# 3️⃣ 创建环境（eval_mode=True 启用 PID 历史记录）
# ============================================================
print("\n" + "=" * 80)
print("🚁 环境配置")
print("=" * 80)

env = QuadcopterEnv(
    control_type=args.get("control_type", "xyz_pos"),
    trajectory_type=args.get("trajectory_type", 1001),
    yaw_type=args.get("yaw_type", 3),
    dt=args.get("sim_dt", args.get("dt", 0.005)),  
    max_episode_steps=args.get("max_episode_steps", 6000),  
    wind_type=args.get("wind_type", "None"),
    wind_strength=args.get("wind_strength", 0.0),
    attitude_bound_start=args.get("attitude_bound_start", 50),
    attitude_bound_end=args.get("attitude_bound_end", 50),
    attitude_bound_decay_epochs=args.get("attitude_bound_decay_epochs", 1),
    eval_mode=True,  # ✅ 启用评估模式（记录 PID 历史）
)
obs, info = env.reset(seed=seed)

print(f"   控制类型: {env.control_type}")
print(f"   轨迹类型: {env.trajectory_type}")
print(f"   偏航类型: {env.yaw_type}")
print(f"   时间步长: {env.dt}s")
print(f"   最大步数: {env.max_episode_steps}")
print(f"   状态维度: {env.observation_space.shape}")
print(f"   动作维度: {env.action_space.shape}")

# ============================================================
# 4️⃣ 构建策略网络
# ============================================================
print("\n" + "=" * 80)
print("🧠 构建策略网络")
print("=" * 80)

state_shape = env.observation_space.shape
action_shape = env.action_space.shape
max_action = float(env.action_space.high[0])

# Actor 网络
net_a = Net(
    state_shape, 
    hidden_sizes=args.get("hidden_sizes", [128, 128]), 
    activation=nn.Tanh, 
    device=device
)
actor = ActorProb(
    net_a, 
    action_shape, 
    max_action=max_action, 
    unbounded=False, 
    device=device
).to(device)

# Critic 网络
net_c = Net(
    state_shape, 
    hidden_sizes=args.get("hidden_sizes", [128, 128]), 
    activation=nn.Tanh, 
    device=device
)
critic = Critic(net_c, device=device).to(device)

# 占位优化器（评估不需要训练）
optim = torch.optim.Adam([torch.zeros(1, device=device)], lr=1e-5)

def dist(loc_scale):
    """正态分布"""
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

# 构建 PPO 策略
policy = PPOPolicy(
    actor=actor,
    critic=critic,
    optim=optim,
    dist_fn=dist,
    discount_factor=args.get("gamma", 0.99),
    gae_lambda=args.get("gae_lambda", 0.95),
    max_grad_norm=args.get("max_grad_norm", 0.5),
    vf_coef=args.get("vf_coef", 0.5),
    ent_coef=args.get("ent_coef", 0.0),
    reward_normalization=args.get("rew_norm", False),
    action_scaling=True,
    action_bound_method=args.get("bound_action_method", "tanh"),
    action_space=env.action_space,
    eps_clip=args.get("eps_clip", 0.2),
    value_clip=args.get("value_clip", True),
    dual_clip=args.get("dual_clip", None),
    advantage_normalization=args.get("norm_adv", True),
    recompute_advantage=args.get("recompute_adv", False),
)

# 加载模型参数
policy.load_state_dict(model_state)
policy.eval()

print(f"✅ 策略网络已加载")
print(f"   隐藏层: {args.get('hidden_sizes', [128, 128])}")
print(f"   动作缩放: {policy.action_scaling}")
print(f"   边界方法: {policy.action_bound_method}")

# ============================================================
# 5️⃣ 运行评估（不调用 render）
# ============================================================
print("\n" + "=" * 80)
print("🎮 开始运行 Episode")
print("=" * 80)

done = False
total_reward = 0.0
step_count = 0
start_time = time.time()

while not done:
    # 策略推理
    with torch.no_grad():
        batch = Batch(
            obs=torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
            info={}
        )
        res = policy(batch, deterministic=True)  # 确定性动作
        action = res.act[0].cpu().numpy()

    # 执行动作（❌ 不调用 render）
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1
    
    # 进度显示（每 100 步）
    if step_count % 100 == 0:
        print(f"  Step {step_count}/{env.max_episode_steps} | "
              f"Reward: {total_reward:.2f} | "
              f"Pos: [{env.quad.pos[0]:.2f}, {env.quad.pos[1]:.2f}, {env.quad.pos[2]:.2f}]",
              end='\r')

episode_time = time.time() - start_time

print(f"\n\n{'=' * 80}")
print(f"🎯 Episode 完成")
print("=" * 80)
print(f"   总步数: {step_count}")
print(f"   总奖励: {total_reward:.2f}")
print(f"   运行时间: {episode_time:.2f}s")
print(f"   平均步频: {step_count / episode_time:.1f} steps/s")
print(f"   仿真时间: {env.time:.2f}s")

# 终止原因
if terminated:
    print(f"   终止原因: {info.get('termination_reason', 'Unknown')}")
else:
    print(f"   终止原因: 达到最大步数")

# ============================================================
# 6️⃣ 离线渲染并保存结果
# ============================================================
print("\n" + "=" * 80)
print("📊 开始生成图表")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"./logs/log_eval_results/{env.trajectory_type}/{timestamp}/"
os.makedirs(out_dir, exist_ok=True)

render_start = time.time()

# ==================== 1. 主图表 ====================
print("\n🎨 生成主图表...")
try:
    main_plot_path = os.path.join(out_dir, "episode_overview.svg")
    env.render(mode='human', save_path=main_plot_path)
    print(f"✅ 主图表已保存: {main_plot_path}")
except Exception as e:
    print(f"❌ 保存主图表失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 2. PID 增益详细波形图（20 路）====================
print("\n🎨 生成 PID 增益详细波形图...")
try:
    if hasattr(env, 'render_pid_gains'):
        pid_detail_path = os.path.join(out_dir, "pid_gains_detail.svg")
        env.render_pid_gains(save_path=pid_detail_path)
        print(f"✅ PID 详细波形图已保存: {pid_detail_path}")
    else:
        print("⚠️ 环境未实现 render_pid_gains 方法")
except Exception as e:
    print(f"⚠️ 保存 PID 详细波形图失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 3. PID 增益摘要图 ====================
print("\n🎨 生成 PID 增益摘要图...")
try:
    if hasattr(env, 'render_pid_summary'):
        pid_summary_path = os.path.join(out_dir, "pid_summary.svg")
        env.render_pid_summary(save_path=pid_summary_path)
        print(f"✅ PID 摘要图已保存: {pid_summary_path}")
    else:
        print("⚠️ 环境未实现 render_pid_summary 方法")
except Exception as e:
    print(f"⚠️ 保存 PID 摘要图失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 4. PID 增益统计分析图 ====================
print("\n🎨 生成 PID 增益统计分析图...")
try:
    if hasattr(env, 'render_pid_statistics'):
        pid_stats_path = os.path.join(out_dir, "pid_statistics.svg")
        env.render_pid_statistics(save_path=pid_stats_path)
        print(f"✅ PID 统计图已保存: {pid_stats_path}")
    else:
        print("⚠️ 环境未实现 render_pid_statistics 方法")
except Exception as e:
    print(f"⚠️ 保存 PID 统计图失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 5. 保存原始数据 ====================
print("\n💾 保存原始数据...")
try:
    data_path = os.path.join(out_dir, "episode_data.npz")
    end_idx = env.current_step
    
    save_data = {
        # 时间序列
        'time': env.episode_t_all[:end_idx],
        
        # 状态数据
        'position': env.episode_pos_all[:end_idx],
        'velocity': env.episode_vel_all[:end_idx],
        'attitude': env.episode_euler_all[:end_idx],
        'omega': env.episode_omega_all[:end_idx],
        
        # 参考轨迹
        'position_ref': env.episode_pos_des_all[:end_idx],
        'velocity_ref': env.episode_vel_des_all[:end_idx],
        'attitude_ref': env.episode_euler_des_all[:end_idx],
        'omega_ref': env.episode_omega_des_all[:end_idx],
        
        # Episode 信息
        'episode_reward': env.episode_reward,
        'total_steps': step_count,
        'episode_time': episode_time,
        'control_type': env.control_type,
        'trajectory_type': env.trajectory_type,
    }
    
    # 添加 PID 历史（如果有）
    if hasattr(env, '_pid_history') and len(env._pid_history) > 0:
        save_data['pid_history'] = np.array(env._pid_history)
        save_data['pid_time'] = np.array(env._pid_time_history)
        print(f"   PID 历史数据点数: {len(env._pid_history)}")
    
    np.savez_compressed(data_path, **save_data)
    print(f"✅ 原始数据已保存: {data_path}")
except Exception as e:
    print(f"⚠️ 保存原始数据失败: {e}")
    import traceback
    traceback.print_exc()

# ==================== 6. 保存评估摘要 ====================
print("\n📝 生成评估摘要...")
try:
    summary_path = os.path.join(out_dir, "summary.txt")
    end_idx = env.current_step
    
    # 计算误差统计
    pos_err = env.episode_pos_des_all[:end_idx] - env.episode_pos_all[:end_idx]
    vel_err = env.episode_vel_des_all[:end_idx] - env.episode_vel_all[:end_idx]
    att_err = np.rad2deg(env.episode_euler_des_all[:end_idx] - env.episode_euler_all[:end_idx])
    
    pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_err**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_err**2, axis=0))
    
    pos_max_err = np.max(np.abs(pos_err), axis=0)
    vel_max_err = np.max(np.abs(vel_err), axis=0)
    att_max_err = np.max(np.abs(att_err), axis=0)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # 基本信息
        f.write("1. Basic Information\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: {ckpt_path}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {device}\n\n")
        
        # 环境配置
        f.write("2. Environment Configuration\n")
        f.write("-" * 80 + "\n")
        f.write(f"Control Type: {env.control_type}\n")
        f.write(f"Trajectory Type: {env.trajectory_type}\n")
        f.write(f"Yaw Type: {env.yaw_type}\n")
        f.write(f"Time Step: {env.dt}s\n")
        f.write(f"Max Episode Steps: {env.max_episode_steps}\n")
        f.write(f"State Dimension: {env.observation_space.shape}\n")
        f.write(f"Action Dimension: {env.action_space.shape}\n\n")
        
        # Episode 结果
        f.write("3. Episode Results\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Steps: {step_count}\n")
        f.write(f"Total Reward: {total_reward:.4f}\n")
        f.write(f"Episode Time: {episode_time:.2f}s\n")
        f.write(f"Average Step Rate: {step_count / episode_time:.1f} steps/s\n")
        f.write(f"Simulation Time: {env.time:.2f}s\n")
        
        if terminated:
            f.write(f"Termination: {info.get('termination_reason', 'Unknown')}\n\n")
        else:
            f.write(f"Termination: Max steps reached\n\n")
        
        # 跟踪性能
        f.write("4. Tracking Performance\n")
        f.write("-" * 80 + "\n")
        
        f.write("\nPosition RMSE (m):\n")
        f.write(f"  X: {pos_rmse[0]:.6f}\n")
        f.write(f"  Y: {pos_rmse[1]:.6f}\n")
        f.write(f"  Z: {pos_rmse[2]:.6f}\n")
        f.write(f"  Total: {np.linalg.norm(pos_rmse):.6f}\n")
        
        f.write("\nPosition Max Error (m):\n")
        f.write(f"  X: {pos_max_err[0]:.6f}\n")
        f.write(f"  Y: {pos_max_err[1]:.6f}\n")
        f.write(f"  Z: {pos_max_err[2]:.6f}\n")
        
        f.write("\nVelocity RMSE (m/s):\n")
        f.write(f"  Vx: {vel_rmse[0]:.6f}\n")
        f.write(f"  Vy: {vel_rmse[1]:.6f}\n")
        f.write(f"  Vz: {vel_rmse[2]:.6f}\n")
        f.write(f"  Total: {np.linalg.norm(vel_rmse):.6f}\n")
        
        f.write("\nVelocity Max Error (m/s):\n")
        f.write(f"  Vx: {vel_max_err[0]:.6f}\n")
        f.write(f"  Vy: {vel_max_err[1]:.6f}\n")
        f.write(f"  Vz: {vel_max_err[2]:.6f}\n")
        
        f.write("\nAttitude RMSE (deg):\n")
        f.write(f"  Roll:  {att_rmse[0]:.6f}\n")
        f.write(f"  Pitch: {att_rmse[1]:.6f}\n")
        f.write(f"  Yaw:   {att_rmse[2]:.6f}\n")
        f.write(f"  Total: {np.linalg.norm(att_rmse):.6f}\n")
        
        f.write("\nAttitude Max Error (deg):\n")
        f.write(f"  Roll:  {att_max_err[0]:.6f}\n")
        f.write(f"  Pitch: {att_max_err[1]:.6f}\n")
        f.write(f"  Yaw:   {att_max_err[2]:.6f}\n")
        
        # PID 信息
        if hasattr(env, '_pid_history') and len(env._pid_history) > 0:
            f.write("\n5. PID Gains Information\n")
            f.write("-" * 80 + "\n")
            pid_array = np.array(env._pid_history)
            f.write(f"Total PID Updates: {len(env._pid_history)}\n")
            f.write(f"PID Parameters: {pid_array.shape[1]}\n\n")
            
            f.write("Initial PID Gains:\n")
            for i, val in enumerate(pid_array[0, :]):
                f.write(f"  Gain[{i:2d}]: {val:8.4f}\n")
            
            f.write("\nFinal PID Gains:\n")
            for i, val in enumerate(pid_array[-1, :]):
                f.write(f"  Gain[{i:2d}]: {val:8.4f}\n")
            
            f.write("\nPID Gains Statistics:\n")
            pid_mean = np.mean(pid_array, axis=0)
            pid_std = np.std(pid_array, axis=0)
            pid_min = np.min(pid_array, axis=0)
            pid_max = np.max(pid_array, axis=0)
            
            for i in range(pid_array.shape[1]):
                f.write(f"  Gain[{i:2d}]: mean={pid_mean[i]:8.4f}, "
                       f"std={pid_std[i]:8.4f}, "
                       f"min={pid_min[i]:8.4f}, "
                       f"max={pid_max[i]:8.4f}\n")
        
    print(f"✅ 评估摘要已保存: {summary_path}")
except Exception as e:
    print(f"⚠️ 保存摘要失败: {e}")
    import traceback
    traceback.print_exc()

render_time = time.time() - render_start

# ============================================================
# 7️⃣ 最终输出
# ============================================================
print("\n" + "=" * 80)
print("🎉 评估完成！")
print("=" * 80)
print(f"\n📊 图表生成耗时: {render_time:.2f}s")
print(f"📁 所有结果已保存到: {out_dir}\n")

print("生成的文件:")
print("=" * 80)
print(f"  1. episode_overview.svg       (主状态图表)")
print(f"  2. pid_gains_detail.svg       (PID 20路详细波形)")
print(f"  3. pid_summary.svg            (PID 摘要图)")
print(f"  4. pid_statistics.svg         (PID 统计分析)")
print(f"  5. episode_data.npz           (原始数据)")
print(f"  6. summary.txt                (评估摘要)")
print("=" * 80)

# 显示关键性能指标
print(f"\n关键性能指标:")
print("=" * 80)
print(f"  总奖励:     {total_reward:10.2f}")
print(f"  位置 RMSE:  {np.linalg.norm(pos_rmse):10.6f} m")
print(f"  速度 RMSE:  {np.linalg.norm(vel_rmse):10.6f} m/s")
print(f"  姿态 RMSE:  {np.linalg.norm(att_rmse):10.6f} deg")
print("=" * 80 + "\n")