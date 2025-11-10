#!/usr/bin/env python3
"""
PX4四旋翼PID参数优化PPO训练脚本
基于tianshou框架实现 - 重构版
"""

import argparse
import datetime
import os
import pprint
from typing import Tuple
import json

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic

from px4Env import QuadcopterEnv


# ==================== JSON编码器 ====================
class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


# ==================== 参数解析 ====================
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO for PX4 Quadcopter PID Optimization")
    
    # 环境参数
    parser.add_argument("--control-type", type=str, default="xyz_pos", 
                       choices=["xyz_pos", "xy_vel_z_pos", "xyz_vel"])
    parser.add_argument("--trajectory-type", type=int, default=206)
    parser.add_argument("--yaw-type", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--wind-type", type=str, default="None")
    parser.add_argument("--wind-strength", type=float, default=0.0)

    # 训练参数
    parser.add_argument("--seed", type=int, default=206)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-num", type=int, default=16)
    parser.add_argument("--test-num", type=int, default=8)
    
    # PPO超参数
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="tanh")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-decay", type=int, default=False,
                       help="Enable learning rate decay (linear to 0)")
    
    parser.add_argument("--max-grad-norm", type=float, default=0.95)
    parser.add_argument("--grad-norm-decay", action="store_true", default=False,
                       help="Enable linear decay of max_grad_norm from start to end over epochs")
    parser.add_argument("--max-grad-norm-start", type=float, default=2.0,
                       help="Starting value of max_grad_norm when --grad-norm-decay is enabled")
    parser.add_argument("--max-grad-norm-end", type=float, default=0.5,
                       help="Ending value of max_grad_norm when --grad-norm-decay is enabled")
    
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    
    # 姿态越界阈值课程学习参数
    parser.add_argument("--attitude-bound-start", type=float, default=90.0,
                       help="Initial attitude bound threshold (degrees)")
    parser.add_argument("--attitude-bound-end", type=float, default=90.0,
                       help="Final attitude bound threshold (degrees)")
    parser.add_argument("--attitude-bound-decay-epochs", type=int, default=1000,
                       help="Number of epochs to decay from start to end threshold")

    # 日志和保存
    parser.add_argument("--logdir", type=str, default="logs/log_train_results")
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None, 
                       help="Path to resume training from checkpoint")
    parser.add_argument("--resume-id", type=str, default=None,
                       help="Resume wandb run id")
    parser.add_argument("--logger", type=str, default="tensorboard", 
                       choices=["tensorboard", "wandb"])
    parser.add_argument("--wandb-project", type=str, default="px4-pid-optimization")
    parser.add_argument("--watch", default=False, action="store_true",
                       help="watch the play of pre-trained policy only")
    
    return parser.parse_args()


# ==================== 环境创建函数 ====================
def make_px4_env(
    control_type: str,
    trajectory_type: int,
    yaw_type: int,
    max_episode_steps: int,
    sim_dt: float,
    wind_type: str,
    wind_strength: float,
    seed: int,
    training_num: int,
    test_num: int,
    attitude_bound_start: float,
    attitude_bound_end: float,
    attitude_bound_decay_epochs: int,
) -> Tuple:
    """创建PX4四旋翼环境"""
    
    def make_env(seed_offset: int = 0):
        def _init():
            env = QuadcopterEnv(
                control_type=control_type,
                trajectory_type=trajectory_type,
                yaw_type=yaw_type,
                max_episode_steps=max_episode_steps,
                dt=sim_dt,
                wind_type=wind_type,
                wind_strength=wind_strength,
                attitude_bound_start=attitude_bound_start,
                attitude_bound_end=attitude_bound_end,
                attitude_bound_decay_epochs=attitude_bound_decay_epochs,
            )
            env.reset(seed=seed + seed_offset)
            return env
        return _init
    
    # 创建单个环境用于获取状态和动作空间
    env = make_env(0)()
    
    # 训练环境
    if training_num > 1:
        train_envs = SubprocVectorEnv(
            [make_env(i) for i in range(training_num)],
            share_memory=False
        )
    else:
        train_envs = DummyVectorEnv([make_env(0)])
    
    # 测试环境
    test_envs = SubprocVectorEnv(
        [make_env(training_num + i) for i in range(test_num)],
        share_memory=False
    )
    
    return env, train_envs, test_envs


# ==================== 主训练函数 ====================
def train_px4_ppo(args: argparse.Namespace = get_args()) -> None:
    """主训练函数"""
    
    # 创建环境
    env, train_envs, test_envs = make_px4_env(
        control_type=args.control_type,
        trajectory_type=args.trajectory_type,
        yaw_type=args.yaw_type,
        max_episode_steps=args.max_episode_steps,
        sim_dt=args.sim_dt,
        wind_type=args.wind_type,
        wind_strength=args.wind_strength,
        seed=args.seed,
        training_num=args.training_num,
        test_num=args.test_num,
        attitude_bound_start=args.attitude_bound_start,
        attitude_bound_end=args.attitude_bound_end,
        attitude_bound_decay_epochs=args.attitude_bound_decay_epochs,
    )
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.min_action = env.action_space.low[0]
    
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", args.min_action, args.max_action)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # ==================== 构建神经网络 ====================
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        unbounded=False,
        device=args.device,
    ).to(args.device)
    
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    
    actor_critic = ActorCritic(actor, critic)
    
    # ==================== 网络初始化 ====================
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    # do last policy layer scaling
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    # ==================== 优化器和学习率调度器 ====================
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr, eps=1e-5)
    
    lr_scheduler = None
    if args.lr_decay:
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
        
        lr_scheduler = CosineAnnealingLR(
            optim,
            T_max=int(max_update_num),
            eta_min=args.lr * 0.1  # 最小学习率为初始值的10%
        )
        
        print(f"✅ Learning rate decay enabled (cosine annealing):")
        print(f"   Initial LR: {args.lr:.2e}")
        print(f"   Minimum LR: {args.lr * 0.1:.2e}")
        print(f"   T_max: {int(max_update_num)} updates")
    
    # ==================== 分布函数 ====================
    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)
    
    # ==================== PPO策略 ====================
    policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=env.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    # ==================== 加载checkpoint ====================
    if args.resume_path:
        print(f"📂 Loading model from: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=args.device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        print(f"✅ Loaded agent from: {args.resume_path}")
    
    # ==================== 创建缓冲区 ====================
    buffer: VectorReplayBuffer | ReplayBuffer
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    
    # ==================== 创建收集器 ====================
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs)

    # ==================== 设置日志 ====================
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    
    # logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    else:
        logger_factory.logger_type = "tensorboard"
    
    logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )
    
    print(f"📝 Logging to: {log_path}")
    
    # 保存配置
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4, cls=NumpyEncoder)

    # ==================== 保存函数 ====================
    def save_best_fn(policy: BasePolicy) -> None:
        """保存最佳模型"""
        best_path = os.path.join(log_path, "best_policy.pth")
        torch.save({"model": policy.state_dict()}, best_path)
        print(f"💾 Best policy saved to {best_path}")
    
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> None:
        """保存checkpoint并更新环境的姿态阈值"""
        
        # 更新姿态阈值（课程学习）
        if hasattr(train_envs, 'env_method'):
            train_envs.env_method('update_attitude_bound', epoch)
        if hasattr(test_envs, 'env_method'):
            test_envs.env_method('update_attitude_bound', epoch)
        
        # 保存checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epoch:
            ckpt_path = os.path.join(log_path, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "model": policy.state_dict(),
                "epoch": epoch,
                "env_step": env_step,
                "gradient_step": gradient_step,
            }, ckpt_path)
            
            # 打印状态信息
            print(f"\n{'='*60}")
            print(f"💾 Checkpoint saved: epoch {epoch}/{args.epoch}")
            print(f"   📊 Environment steps: {env_step:,}")
            print(f"   📊 Gradient steps: {gradient_step:,}")
            
            if lr_scheduler is not None:
                current_lr = optim.param_groups[0]['lr']
                print(f"   📊 Current learning rate: {current_lr:.2e}")
            # 打印当前梯度裁剪阈值
            if hasattr(policy, 'max_grad_norm'):
                try:
                    print(f"   ✂️  Current max_grad_norm: {float(policy.max_grad_norm):.4f}")
                except Exception:
                    pass
            
            print(f"{'='*60}\n")

    # ==================== 打印训练配置 ====================
    print("\n" + "="*60)
    print("🚀 Training Configuration Summary")
    print("="*60)
    print(f"📦 Environment: {args.control_type} (trajectory {args.trajectory_type})")
    print(f"🎯 Total epochs: {args.epoch}")
    print(f"🔄 Steps per epoch: {args.step_per_epoch:,}")
    print(f"🎲 Training envs: {args.training_num}")
    print(f"🧪 Test envs: {args.test_num}")
    print(f"🧠 Network: {args.hidden_sizes}")
    print(f"📊 Learning rate: {args.lr:.2e}" + (" (with decay)" if args.lr_decay else ""))
    if args.grad_norm_decay:
        print(f"✂️  Gradient clipping (cosine anneal): {args.max_grad_norm_start:.4f} → {args.max_grad_norm_end:.4f}")
    else:
        print(f"✂️  Gradient clipping: {args.max_grad_norm:.4f}")
    print(f"🎓 Attitude bound: {args.attitude_bound_start}° → {args.attitude_bound_end}°")
    print("="*60 + "\n")

    # ==================== 训练或测试 ====================
    if not args.watch:
        # 训练
        print("🚀 Starting training...\n")
        # 可选：按 Epoch 线性衰减 max_grad_norm（例如 1.0 -> 0.5）
        def _maybe_decay_grad_norm(cur_epoch: int) -> None:
            if not args.grad_norm_decay:
                return
            # 余弦退火：从 start 到 end，ratio ∈ [0,1]
            ratio = min(max(cur_epoch / max(1, args.epoch), 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * ratio))
            cur = args.max_grad_norm_end + (args.max_grad_norm_start - args.max_grad_norm_end) * cosine
            # 动态更新策略中的裁剪阈值
            policy.max_grad_norm = float(cur)

        def train_fn(epoch: int, env_step: int) -> None:
            _maybe_decay_grad_norm(epoch)
            # 可选：周期性打印
            if args.grad_norm_decay and (epoch % 10 == 0):
                try:
                    print(f"[Epoch {epoch}] max_grad_norm -> {float(policy.max_grad_norm):.4f}")
                except Exception:
                    pass

        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            train_fn=train_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            logger=logger,
            test_in_train=False,
        ).run()
        
        pprint.pprint(result)
        
        # 保存最终模型
        final_path = os.path.join(log_path, "final_policy.pth")
        torch.save({
            "model": policy.state_dict(),
            "result": result,
        }, final_path)
        
        print(f"\n{'='*60}")
        print(f"✅ Training completed!")
        print(f"💾 Final model saved: {final_path}")
        print(f"📊 Best model: {os.path.join(log_path, 'best_policy.pth')}")
        print(f"{'='*60}\n")
    
    else:
        # 测试模式：观看训练好的策略表现
        print("👀 Watch mode: Testing pre-trained policy...\n")
        test_envs.seed(args.seed)
        test_collector.reset()
        collector_stats = test_collector.collect(n_episode=args.test_num)
        print(collector_stats)


if __name__ == "__main__":
    train_px4_ppo()