#!/usr/bin/env python3
"""
✅ Quadcopter PPO 策略评估脚本（兼容性完整版）
- 兼容不同版本 QuadcopterEnv（自动处理 sim_dt / dt 参数）
- 确保动作范围在 [-1, 1]（调用 policy.map_action）
- 自动选择 GPU / CPU
"""

import time
import torch
import numpy as np
# 新增
import os
from torch import nn
from torch.distributions import Independent, Normal
from tianshou.policy import PPOPolicy
from tianshou.data import Batch
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from px4Env import QuadcopterEnv  # 请确保路径正确

# ==== 1️⃣ 加载 checkpoint ====
ckpt_path = "./logs/log_train/1001/251030-111456/best_policy.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")
args = ckpt["args"]
model_state = ckpt["model"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 2️⃣ 创建环境（自动兼容 sim_dt / dt） ====
env = QuadcopterEnv(
    control_type=args.get("control_type", "xyz_vel"),
    trajectory_type=args.get("trajectory_type", 1001),
    yaw_type=args.get("yaw_type", 3),
    dt=args.get("sim_dt", args.get("dt", 0.005)),    
    rl_dt=args.get("rl_dt", 0.05),
    max_episode_steps=args.get("max_episode_steps", 1500),
    wind_type=args.get("wind_type", "none"),
    wind_strength=args.get("wind_strength", 0.0),
    attitude_bound_start=args.get("attitude_bound_start", 50),
    attitude_bound_end=args.get("attitude_bound_end", 50),
    attitude_bound_decay_epochs=args.get("attitude_bound_decay_epochs", 1),
    eval_mode=True,
)
obs, info = env.reset(seed=args.get("seed", 1001))

# ==== 3️⃣ 构建与训练相同结构的策略网络 ====
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
max_action = float(env.action_space.high[0])

net_a = Net(state_shape, hidden_sizes=args["hidden_sizes"], activation=nn.Tanh, device=device)
actor = ActorProb(net_a, action_shape, max_action=max_action, unbounded=False, device=device).to(device)

net_c = Net(state_shape, hidden_sizes=args["hidden_sizes"], activation=nn.Tanh, device=device)
critic = Critic(net_c, device=device).to(device)

optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args["lr"], eps=1e-5)

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

print("##########################################################")
print("✅ 成功加载 PPO 策略参数")
print("Action scaling:", policy.action_scaling)
print("Bound method:", policy.action_bound_method)
print("Actor unbounded:", getattr(actor, "unbounded", None))
print("Max action:", getattr(actor, "max_action", None))
print("##########################################################")

# ==== 4️⃣ 运行单回合测试 ====
done = False
total_reward = 0.0

while not done:
    with torch.no_grad():
        batch = Batch(obs=torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0), info={})
        res = policy(batch, deterministic=True)
        action = res.act[0]
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        # ✅ 关键修复：调用 PPOPolicy 内部 map_action，确保动作范围一致
        if hasattr(policy, "map_action"):
            action = policy.map_action(action)

    # print(f"Applying action: {np.round(action, 4)}")
    # print(f"Action range -> min: {action.min():.3f}, max: {action.max():.3f}")

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

    env.render()  # 如环境支持图形，取消注释开启可视化
    time.sleep(0.03)

print(f"🎯 Total reward: {total_reward:.2f}")

# ==== 5️⃣ 保存渲染图与完整分析图（.svg） ====
# 输出目录+trajectory_type
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"./logs/log_eval_results/{env.trajectory_type}/{timestamp}/"
os.makedirs(out_dir, exist_ok=True)
# 保存 render 窗口（全程曲线）
if hasattr(env, "_render_fig"):
    render_svg = os.path.join(out_dir, "render_full.svg")
    try:
        env._render_fig.savefig(render_svg, format="svg", bbox_inches="tight")
        print(f"🖼️ 渲染图已保存: {render_svg}")
    except Exception as e:
        print(f"保存渲染图失败: {e}")

# 新增：保存 20 路增益 4x5 波形图
if hasattr(env, "_gains_fig"):
    gains_svg = os.path.join(out_dir, "gains_full.svg")
    try:
        env._gains_fig.savefig(gains_svg, format="svg", bbox_inches="tight")
        print(f"🖼️ 增益波形图已保存: {gains_svg}")
    except Exception as e:
        print(f"保存增益波形图失败: {e}")

# # 保存完整 episode 分析图（环境自带，自动 .svg）
# try:
#     env.plot_all_results(save_dir=out_dir, show=False)
#     print(f"📊 分析图已保存到目录: {out_dir}")
# except Exception as e:
#     print(f"保存分析图失败: {e}")
