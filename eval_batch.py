#!/usr/bin/env python3
"""
✅ Quadcopter PPO 策略批量评估脚本（完整离线版）
支持：
1. 主状态图表（位置、速度、姿态、误差）
2. PID 增益详细波形图（20 路）
3. PID 增益摘要图（热力图、柱状图、关键波形）
4. PID 增益统计分析图
5. 批量对比可视化
"""
import time
import torch
import numpy as np
import random
import os
import argparse
from datetime import datetime
from torch import nn
from torch.distributions import Independent, Normal
from tianshou.policy import PPOPolicy
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from px4Env import QuadcopterEnv

# ============================================================
# 命令行参数
# ============================================================
parser = argparse.ArgumentParser(description='Batch Evaluation for PPO Policy')
parser.add_argument('--ckpt', type=str, 
                    default="./logs/log_train_results/1012/251031-220741/best_policy.pth",
                    help='Checkpoint path')
parser.add_argument('--num-episodes', type=int, default=10,
                    help='Number of episodes to evaluate')
parser.add_argument('--seed', type=int, default=None,
                    help='Random seed (None to use checkpoint seed)')
parser.add_argument('--deterministic', action='store_true', default=True,
                    help='Use deterministic policy')
parser.add_argument('--save-all', action='store_true', default=False,
                    help='Save plots for all episodes')
parser.add_argument('--device', type=str, default='auto',
                    help='Device: cpu, cuda, or auto')
cmd_args = parser.parse_args()

# ============================================================
# 1️⃣ 加载 checkpoint
# ============================================================
print("=" * 80)
print("📂 加载模型检查点")
print("=" * 80)

ckpt_path = cmd_args.ckpt
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"❌ 找不到检查点: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt["args"]
model_state = ckpt["model"]

if cmd_args.device == 'auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(cmd_args.device)

print(f"✅ 成功加载: {ckpt_path}")
print(f"   Device: {device}")

# ============================================================
# 2️⃣ 设置随机种子
# ============================================================
seed = cmd_args.seed if cmd_args.seed is not None else args.get("seed", 1012)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print(f"🌱 随机种子: {seed}")

# ============================================================
# 3️⃣ 创建环境
# ============================================================
print("\n" + "=" * 80)
print("🚁 环境配置")
print("=" * 80)

env = QuadcopterEnv(
    control_type=args.get("control_type", "xyz_vel"),
    trajectory_type=args.get("trajectory_type", 1012),
    yaw_type=args.get("yaw_type", 3),
    dt=args.get("sim_dt", args.get("dt", 0.01)),
    max_episode_steps=args.get("max_episode_steps", 3000),
    wind_type=args.get("wind_type", "None"),
    wind_strength=args.get("wind_strength", 0.0),
    attitude_bound_start=args.get("attitude_bound_start", 50),
    attitude_bound_end=args.get("attitude_bound_end", 50),
    attitude_bound_decay_epochs=args.get("attitude_bound_decay_epochs", 1),
    eval_mode=True,
)

print(f"   控制类型: {env.control_type}")
print(f"   轨迹类型: {env.trajectory_type}")
print(f"   时间步长: {env.dt}s")
print(f"   最大步数: {env.max_episode_steps}")

# ============================================================
# 4️⃣ 构建策略网络
# ============================================================
print("\n" + "=" * 80)
print("🧠 构建策略网络")
print("=" * 80)

state_shape = env.observation_space.shape
action_shape = env.action_space.shape
max_action = float(env.action_space.high[0])

net_a = Net(state_shape, hidden_sizes=args["hidden_sizes"], activation=nn.Tanh, device=device)
actor = ActorProb(net_a, action_shape, max_action=max_action, unbounded=False, device=device).to(device)

net_c = Net(state_shape, hidden_sizes=args["hidden_sizes"], activation=nn.Tanh, device=device)
critic = Critic(net_c, device=device).to(device)

optim = torch.optim.Adam([torch.zeros(1, device=device)], lr=1e-5)

def dist(loc_scale):
    loc, scale = loc_scale
    return Independent(Normal(loc, scale), 1)

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

policy.load_state_dict(model_state)
policy.eval()
print(f"✅ 策略网络已加载")

# ============================================================
# 5️⃣ 批量评估函数
# ============================================================
def run_single_episode(env, policy, episode_id, seed_offset, device, deterministic=True):
    """运行单个 episode"""
    obs, info = env.reset(seed=int(seed + seed_offset))
    done = False
    total_reward = 0.0
    step_count = 0
    start_time = time.time()
    
    while not done:
        with torch.no_grad():
            batch = Batch(
                obs=torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                info={}
            )
            res = policy(batch, deterministic=deterministic)
            action = res.act[0].cpu().numpy()
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 500 == 0:
            print(f"    Episode {episode_id}: Step {step_count}/{env.max_episode_steps} | "
                  f"Reward: {total_reward:.2f}", end='\r')
    
    episode_time = time.time() - start_time
    
    # 计算误差统计
    end_idx = env.current_step
    pos_err = env.episode_pos_des_all[:end_idx] - env.episode_pos_all[:end_idx]
    vel_err = env.episode_vel_des_all[:end_idx] - env.episode_vel_all[:end_idx]
    att_err = np.rad2deg(env.episode_euler_des_all[:end_idx] - env.episode_euler_all[:end_idx])
    
    pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_err**2, axis=0))
    att_rmse = np.sqrt(np.mean(att_err**2, axis=0))
    
    result = {
        'episode_id': episode_id,
        'total_reward': total_reward,
        'step_count': step_count,
        'episode_time': episode_time,
        'pos_rmse': pos_rmse,
        'vel_rmse': vel_rmse,
        'att_rmse': att_rmse,
    }
    
    return result

# ============================================================
# 6️⃣ 批量评估主循环
# ============================================================
print("\n" + "=" * 80)
print(f"🎮 开始批量评估（共 {cmd_args.num_episodes} 个 episodes）")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"./logs/log_eval_results/{env.trajectory_type}/batch_{timestamp}/"
os.makedirs(out_dir, exist_ok=True)

all_results = []
eval_start_time = time.time()

for episode_id in range(cmd_args.num_episodes):
    print(f"\n{'─' * 80}")
    print(f"📍 Episode {episode_id + 1}/{cmd_args.num_episodes}")
    print(f"{'─' * 80}")
    
    result = run_single_episode(env, policy, episode_id, episode_id, device, cmd_args.deterministic)
    all_results.append(result)
    
    print(f"\n✅ Episode {episode_id + 1} 完成:")
    print(f"   总步数: {result['step_count']}")
    print(f"   总奖励: {result['total_reward']:.2f}")
    print(f"   位置 RMSE: {result['pos_rmse']}")
    
    # 保存图表
    if cmd_args.save_all:
        episode_dir = os.path.join(out_dir, f"episode_{episode_id:03d}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 主图表
        env.render(mode='human', save_path=os.path.join(episode_dir, "overview.png"))
        
        # PID 增益图（3 种）
        env.render_pid_gains(save_path=os.path.join(episode_dir, "pid_gains_detail.png"))
        env.render_pid_summary(save_path=os.path.join(episode_dir, "pid_summary.png"))
        env.render_pid_statistics(save_path=os.path.join(episode_dir, "pid_statistics.png"))

total_eval_time = time.time() - eval_start_time

# ============================================================
# 7️⃣ 统计分析
# ============================================================
print("\n" + "=" * 80)
print("📊 批量评估统计分析")
print("=" * 80)

rewards = [r['total_reward'] for r in all_results]
pos_rmses = np.array([r['pos_rmse'] for r in all_results])

best_idx = np.argmax(rewards)
worst_idx = np.argmin(rewards)
median_idx = np.argsort(rewards)[len(rewards) // 2]

print(f"\n📈 奖励统计:")
print(f"   平均值: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"   最大值: {np.max(rewards):.2f} (Episode {best_idx + 1})")
print(f"   最小值: {np.min(rewards):.2f} (Episode {worst_idx + 1})")

# ============================================================
# 8️⃣ 保存关键 episodes
# ============================================================
print("\n" + "=" * 80)
print("💾 保存关键 episode 图表")
print("=" * 80)

key_episodes = {
    'best': best_idx,
    'worst': worst_idx,
    'median': median_idx,
}

for key, idx in key_episodes.items():
    print(f"\n🎨 渲染 {key} episode (Episode {idx + 1})...")
    
    # 重新运行
    run_single_episode(env, policy, idx, int(idx), device, cmd_args.deterministic)
    
    # 保存所有图表
    key_dir = os.path.join(out_dir, f"{key}_episode_{idx:03d}")
    os.makedirs(key_dir, exist_ok=True)
    
    # 1. 主图表
    env.render(mode='human', save_path=os.path.join(key_dir, "overview.png"))
    print(f"   ✅ 主图表已保存")
    
    # 2. PID 增益详细波形（20 路）
    env.render_pid_gains(save_path=os.path.join(key_dir, "pid_gains_detail.png"))
    print(f"   ✅ PID 详细波形已保存")
    
    # 3. PID 增益摘要
    env.render_pid_summary(save_path=os.path.join(key_dir, "pid_summary.png"))
    print(f"   ✅ PID 摘要图已保存")
    
    # 4. PID 统计分析
    env.render_pid_statistics(save_path=os.path.join(key_dir, "pid_statistics.png"))
    print(f"   ✅ PID 统计图已保存")
    
    # 5. 原始数据
    end_idx = env.current_step
    np.savez_compressed(
        os.path.join(key_dir, "data.npz"),
        time=env.episode_t_all[:end_idx],
        position=env.episode_pos_all[:end_idx],
        pid_history=np.array(env._pid_history) if hasattr(env, '_pid_history') else None,
    )
    print(f"   ✅ 原始数据已保存")

# ============================================================
# 9️⃣ 生成批量对比图
# ============================================================
print("\n" + "=" * 80)
print("📊 生成批量对比图")
print("=" * 80)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Batch Evaluation Summary ({cmd_args.num_episodes} episodes)", fontsize=14, fontweight='bold')
    
    # 1. 奖励分布
    ax = axes[0, 0]
    ax.hist(rewards, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Total Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 奖励趋势
    ax = axes[0, 1]
    ax.plot(range(1, len(rewards) + 1), rewards, 'o-', markersize=6)
    ax.axhline(np.mean(rewards), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Trend')
    ax.grid(alpha=0.3)
    
    # 3. 位置 RMSE
    ax = axes[1, 0]
    x_pos = np.arange(3)
    mean_pos = np.mean(pos_rmses, axis=0)
    std_pos = np.std(pos_rmses, axis=0)
    ax.bar(x_pos, mean_pos, yerr=std_pos, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Position Tracking Error')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Evaluation Summary
    ==================
    Episodes: {cmd_args.num_episodes}
    Total Time: {total_eval_time:.2f}s
    
    Reward:
      Mean: {np.mean(rewards):.2f}
      Std:  {np.std(rewards):.2f}
      Max:  {np.max(rewards):.2f}
      Min:  {np.min(rewards):.2f}
    
    Position RMSE (m):
      X: {mean_pos[0]:.4f} ± {std_pos[0]:.4f}
      Y: {mean_pos[1]:.4f} ± {std_pos[1]:.4f}
      Z: {mean_pos[2]:.4f} ± {std_pos[2]:.4f}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "batch_summary.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 批量对比图已保存")
    
except Exception as e:
    print(f"⚠️ 生成对比图失败: {e}")

# ============================================================
# 🔟 最终总结
# ============================================================
print("\n" + "=" * 80)
print("🎉 批量评估完成！")
print("=" * 80)
print(f"📁 结果保存目录: {out_dir}")
print("\n生成的文件:")
print(f"  - batch_summary.png          (批量对比)")
print(f"  - best_episode_XXX/")
print(f"      - overview.png           (主图表)")
print(f"      - pid_gains_detail.png   (PID 20路波形)")
print(f"      - pid_summary.png        (PID 摘要)")
print(f"      - pid_statistics.png     (PID 统计)")
print(f"      - data.npz               (原始数据)")
print(f"  - worst_episode_XXX/         (同上)")
print(f"  - median_episode_XXX/        (同上)")
if cmd_args.save_all:
    print(f"  - episode_XXX/               (所有episodes)")
print("=" * 80 + "\n")