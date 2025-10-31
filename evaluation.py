#!/usr/bin/env python3
"""
✅ Quadcopter PPO 策略评估脚本（修复版）
"""
import time
import torch
import numpy as np
import random
import os
from datetime import datetime
from torch import nn
from torch.distributions import Independent, Normal
from tianshou.policy import PPOPolicy
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from px4Env import QuadcopterEnv

# ==== 1️⃣ 加载 checkpoint ====
ckpt_path = "./logs/log_train/1002/251030-215756/best_policy.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt["args"]
model_state = ckpt["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 2️⃣ 设置随机种子 ====
seed = args.get("seed", 1002)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ==== 3️⃣ 创建环境 ====
env = QuadcopterEnv(
    control_type=args.get("control_type", "xyz_vel"),
    trajectory_type=args.get("trajectory_type", 1002),
    yaw_type=args.get("yaw_type", 3),
    dt=args.get("sim_dt", args.get("dt", 0.005)),
    rl_dt=args.get("rl_dt", 0.02),
    max_episode_steps=args.get("max_episode_steps", 1500),
    wind_type=args.get("wind_type", "none"),
    wind_strength=args.get("wind_strength", 0.0),
    attitude_bound_start=args.get("attitude_bound_start", 50),
    attitude_bound_end=args.get("attitude_bound_end", 50),
    attitude_bound_decay_epochs=args.get("attitude_bound_decay_epochs", 1),
    eval_mode=True,
)
obs, info = env.reset(seed=seed)

# ==== 4️⃣ 构建策略网络 ====
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
max_action = float(env.action_space.high[0])

net_a = Net(state_shape, hidden_sizes=args["hidden_sizes"], activation=nn.Tanh, device=device)
actor = ActorProb(net_a, action_shape, max_action=max_action, unbounded=False, device=device).to(device)

net_c = Net(state_shape, hidden_sizes=args["hidden_sizes"], activation=nn.Tanh, device=device)
critic = Critic(net_c, device=device).to(device)

# 占位优化器（评估不需要训练）
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

print("=" * 60)
print("✅ 成功加载 PPO 策略")
print(f"Device: {device}")
print(f"Action scaling: {policy.action_scaling}")
print(f"Bound method: {policy.action_bound_method}")
print("=" * 60)

# ==== 5️⃣ 运行评估 ====
done = False
total_reward = 0.0
step_count = 0

while not done:
    step_start = time.time()
    
    with torch.no_grad():
        batch = Batch(
            obs=torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
            info={}
        )
        res = policy(batch, deterministic=True)
        action = res.act[0].cpu().numpy()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step_count += 1

    env.render()
    
    # 动态帧率控制
    elapsed = time.time() - step_start
    time.sleep(max(0, 0.1 - elapsed))

print(f"\n🎯 Episode finished:")
print(f"   Total steps: {step_count}")
print(f"   Total reward: {total_reward:.2f}")

# ==== 6️⃣ 保存结果 ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"./logs/log_eval_results/{env.trajectory_type}/{timestamp}/"
os.makedirs(out_dir, exist_ok=True)

# 保存渲染图
if hasattr(env, "_render_fig") and env._render_fig is not None:
    render_svg = os.path.join(out_dir, "render_full.svg")
    try:
        env._render_fig.savefig(render_svg, format="svg", bbox_inches="tight")
        print(f"🖼️ 渲染图已保存: {render_svg}")
    except Exception as e:
        print(f"⚠️ 保存渲染图失败: {e}")

# 保存增益图
if hasattr(env, "_gains_fig") and env._gains_fig is not None:
    gains_svg = os.path.join(out_dir, "gains_full.svg")
    try:
        env._gains_fig.savefig(gains_svg, format="svg", bbox_inches="tight")
        print(f"🖼️ 增益波形图已保存: {gains_svg}")
    except Exception as e:
        print(f"⚠️ 保存增益波形图失败: {e}")

# 保存完整分析
if hasattr(env, "plot_all_results"):
    try:
        env.plot_all_results(save_dir=out_dir, show=False)
        print(f"📊 完整分析图已保存到: {out_dir}")
    except Exception as e:
        print(f"⚠️ 保存分析图失败: {e}")

print(f"\n✅ 所有结果已保存到: {out_dir}")