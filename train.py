#!/usr/bin/env python3
"""
PX4四旋翼PID参数优化PPO训练脚本
基于tianshou框架实现
学习率和梯度裁剪均基于epoch调度
"""

import argparse
import datetime
import os
import pprint
from typing import Tuple, Dict
import json

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ExponentialLR

from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic

from px4Env import QuadcopterEnv


# ==================== 梯度裁剪调度器 ====================
class GradNormScheduler:
    """梯度裁剪范数调度器（基于epoch）"""
    def __init__(self, initial_norm=1.0, final_norm=0.01, total_epochs=1000, decay_type='linear'):
        """
        Args:
            initial_norm: 初始梯度裁剪范数
            final_norm: 最终梯度裁剪范数
            total_epochs: 总训练轮数
            decay_type: 衰减类型 ('linear', 'cosine', 'exponential')
        """
        self.initial_norm = initial_norm
        self.final_norm = final_norm
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.decay_type = decay_type
        
    def get_norm(self, epoch=None):
        """
        获取指定epoch的梯度裁剪范数
        
        Args:
            epoch: 指定的epoch，如果为None则使用当前epoch
        """
        if epoch is None:
            epoch = self.current_epoch
            
        if epoch >= self.total_epochs:
            return self.final_norm
            
        progress = epoch / self.total_epochs
        
        if self.decay_type == 'linear':
            # 线性衰减
            norm = self.initial_norm - (self.initial_norm - self.final_norm) * progress
        elif self.decay_type == 'cosine':
            # 余弦衰减
            import math
            norm = self.final_norm + (self.initial_norm - self.final_norm) * \
                   0.5 * (1 + math.cos(math.pi * progress))
        elif self.decay_type == 'exponential':
            # 指数衰减
            import math
            norm = self.final_norm + (self.initial_norm - self.final_norm) * \
                   math.exp(-5 * progress)
        else:
            norm = self.initial_norm
            
        return max(norm, self.final_norm)
    
    def step(self):
        """更新epoch计数"""
        self.current_epoch += 1
        
    def state_dict(self):
        """保存状态"""
        return {
            'current_epoch': self.current_epoch,
            'initial_norm': self.initial_norm,
            'final_norm': self.final_norm,
            'total_epochs': self.total_epochs,
            'decay_type': self.decay_type
        }
    
    def load_state_dict(self, state_dict):
        """加载状态"""
        self.current_epoch = state_dict['current_epoch']
        self.initial_norm = state_dict.get('initial_norm', self.initial_norm)
        self.final_norm = state_dict.get('final_norm', self.final_norm)
        self.total_epochs = state_dict.get('total_epochs', self.total_epochs)
        self.decay_type = state_dict.get('decay_type', self.decay_type)


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
    parser.add_argument("--control-type", type=str, default="xyz_vel", 
                       choices=["xyz_pos", "xy_vel_z_pos", "xyz_vel"])
    parser.add_argument("--trajectory-type", type=int, default=1012)
    parser.add_argument("--yaw-type", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=6000)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--wind-type", type=str, default="None")
    parser.add_argument("--wind-strength", type=float, default=0.0)

    # 训练参数
    parser.add_argument("--seed", type=int, default=1012)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=30000)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--training-num", type=int, default=32)
    parser.add_argument("--test-num", type=int, default=16)
    
    # PPO超参数
    parser.add_argument("--rew-norm", type=int, default=True)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="tanh")
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    
    # 学习率衰减参数
    parser.add_argument("--lr-decay", action="store_true",
                       help="Enable learning rate decay based on epoch")
    parser.add_argument("--lr-decay-type", type=str, default="linear",
                       choices=["linear", "cosine", "exponential"],
                       help="Learning rate decay type")
    parser.add_argument("--min-lr-ratio", type=float, default=0.6,
                       help="Minimum learning rate ratio (0.0 = decay to 0, 0.1 = decay to 10%)")

    # 梯度裁剪衰减参数
    parser.add_argument("--grad-norm-decay", action="store_true",
                       help="Enable gradient norm decay based on epoch")
    parser.add_argument("--grad-norm-decay-type", type=str, default="linear",
                       choices=["linear", "cosine", "exponential"],
                       help="Gradient norm decay type")
    parser.add_argument("--max-grad-norm", type=float, default=0.9,
                       help="Initial gradient clipping norm")
    parser.add_argument("--final-grad-norm", type=float, default=0.1,
                       help="Final gradient clipping norm (no decay if equal to max-grad-norm)")
    
    # 姿态越界阈值课程学习参数
    parser.add_argument("--attitude-bound-start", type=float, default=90.0,
                       help="Initial attitude bound threshold (degrees)")
    parser.add_argument("--attitude-bound-end", type=float, default=80.0,
                       help="Final attitude bound threshold (degrees)")
    parser.add_argument("--attitude-bound-decay-epochs", type=int, default=1000,
                       help="Number of epochs to decay from start to end threshold")

    # 日志和保存
    parser.add_argument("--logdir", type=str, default="logs/log_train_results")
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None, 
                       help="Path to resume training from checkpoint")
    parser.add_argument("--logger", type=str, default="tensorboard", 
                       choices=["tensorboard", "wandb"])
    
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
    
    # 创建环境
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
    
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = float(env.action_space.high[0])
    args.min_action = float(env.action_space.low[0])
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 构建神经网络
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
    
    # 网络初始化
    torch.nn.init.constant_(actor.sigma_param, -0.8)
    
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    if hasattr(critic, 'last'):
        for m in [critic.last]:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.1 * m.weight.data)
    
    # 优化器
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr, eps=1e-5)
    
    # ==================== 调度器设置 ====================
    lr_scheduler = None
    grad_norm_scheduler = None
    
    # 学习率衰减（使用 PyTorch 自带调度器）
    if args.lr_decay:
        min_lr = args.lr * args.min_lr_ratio
        if args.lr_decay_type == "linear":
            # LinearLR: end_factor 为 lr 最终相对比例
            lr_scheduler = LinearLR(optim, start_factor=1.0, end_factor=max(args.min_lr_ratio, 0.0), total_iters=max(1, args.epoch))
        elif args.lr_decay_type == "cosine":
            # CosineAnnealingLR: eta_min 为最终 lr（绝对值）
            lr_scheduler = CosineAnnealingLR(optim, T_max=max(1, args.epoch), eta_min=min_lr)
        elif args.lr_decay_type == "exponential":
            # ExponentialLR 需要 gamma，计算使 lr_end = lr_start * gamma^epoch
            if args.epoch > 0 and args.min_lr_ratio > 0.0:
                gamma = args.min_lr_ratio ** (1.0 / args.epoch)
            else:
                # 若 min_lr_ratio==0 或 epoch==0，使用较小的 gamma，使 lr 快速减小
                gamma = 0.0
            lr_scheduler = ExponentialLR(optim, gamma=gamma)

        print(f"✅ Learning rate decay enabled ({args.lr_decay_type}):")
        print(f"   Initial LR: {args.lr:.2e}")
        print(f"   Final LR (target): {min_lr:.2e}")
        print(f"   Decay over: {args.epoch} epochs")
    # 梯度裁剪衰减
    if args.grad_norm_decay and args.max_grad_norm != args.final_grad_norm:
        grad_norm_scheduler = GradNormScheduler(
            initial_norm=args.max_grad_norm,
            final_norm=args.final_grad_norm,
            total_epochs=args.epoch,
            decay_type=args.grad_norm_decay_type
        )
        print(f"✅ Gradient norm decay enabled ({args.grad_norm_decay_type}):")
        print(f"   Initial norm: {args.max_grad_norm:.4f}")
        print(f"   Final norm: {args.final_grad_norm:.4f}")
        print(f"   Decay over: {args.epoch} epochs")
    elif args.grad_norm_decay:
        print(f"⚠️  Gradient norm decay disabled (max_grad_norm == final_grad_norm = {args.max_grad_norm})")
    else:
        print(f"📌 Fixed gradient clipping norm: {args.max_grad_norm:.4f}")

    # 分布函数
    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)
    
    # PPO策略
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

    # 加载checkpoint/最优模型继续训练
    start_epoch = 0
    if args.resume_path:
        print(f"📂 Loading model from: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=args.device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        
        # 如果是checkpoint，恢复优化器和epoch
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
            start_epoch = ckpt.get("epoch", 0)
            
            # 恢复学习率调度器
            if lr_scheduler and "lr_scheduler" in ckpt:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                print(f"   ✅ Restored LR scheduler state")
            
            # 恢复梯度裁剪调度器
            if grad_norm_scheduler and "grad_norm_scheduler" in ckpt:
                grad_norm_scheduler.load_state_dict(ckpt["grad_norm_scheduler"])
                print(f"   ✅ Restored gradient norm scheduler state")
            
            print(f"✅ Resumed from checkpoint at epoch {start_epoch}")
            print(f"   Current LR: {optim.param_groups[0]['lr']:.2e}")
            if grad_norm_scheduler:
                print(f"   Current grad norm: {grad_norm_scheduler.get_norm():.4f}")
        else:
            print(f"✅ Loaded best model for fine-tuning")
    
    # 创建缓冲区
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    
    # 创建收集器
    train_collector = Collector[CollectStats](policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector[CollectStats](policy, test_envs)

    # 设置日志
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    
    logger_factory = LoggerFactoryDefault()
    logger_factory.logger_type = args.logger
    exp_logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=None,
        config_dict=vars(args),
    )
    
    print(f"📝 Logging to: {log_path}")
    
    # 保存配置
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4, cls=NumpyEncoder)

    # 保存函数
    def save_best_fn(policy: BasePolicy) -> None:
        best_path = os.path.join(log_path, "best_policy.pth")
        torch.save({"model": policy.state_dict(), "args": vars(args)}, best_path)
        print(f"💾 Best policy saved")
    
    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> None:
        """在每个epoch结束时调用，更新所有调度器"""
        
        # ==================== 更新梯度裁剪范数 ====================
        if grad_norm_scheduler is not None:
            current_grad_norm = grad_norm_scheduler.get_norm(epoch)
            policy._max_grad_norm = current_grad_norm
            grad_norm_scheduler.step()  # epoch + 1
            
            # 记录到日志
            if hasattr(exp_logger, 'writer'):
                exp_logger.writer.add_scalar("train/grad_norm", current_grad_norm, epoch)
        
        # ==================== 记录学习率 ====================
        if lr_scheduler is not None:
            current_lr = optim.param_groups[0]['lr']
            if hasattr(exp_logger, 'writer'):
                exp_logger.writer.add_scalar("train/learning_rate", current_lr, epoch)
        
        # ==================== 更新姿态阈值 ====================
        if hasattr(train_envs, 'env_method'):
            train_envs.env_method('update_attitude_bound', epoch)
        if hasattr(test_envs, 'env_method'):
            test_envs.env_method('update_attitude_bound', epoch)
        
        # ==================== 保存checkpoint ====================
        if epoch % args.save_interval == 0 or epoch == args.epoch:
            ckpt_path = os.path.join(log_path, f"checkpoint_epoch{epoch}.pth")
            ckpt_data = {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "env_step": env_step,
                "gradient_step": gradient_step,
                "args": vars(args),
            }
            
            # 保存学习率调度器状态
            if lr_scheduler is not None:
                ckpt_data["lr_scheduler"] = lr_scheduler.state_dict()
            
            # 保存梯度裁剪调度器状态
            if grad_norm_scheduler is not None:
                ckpt_data["grad_norm_scheduler"] = grad_norm_scheduler.state_dict()
            
            torch.save(ckpt_data, ckpt_path)
            
            # 打印状态信息
            print(f"\n{'='*60}")
            print(f"💾 Checkpoint saved: epoch {epoch}/{args.epoch}")
            print(f"   📊 Environment steps: {env_step:,}")
            print(f"   📊 Gradient steps: {gradient_step:,}")
            
            if lr_scheduler is not None:
                current_lr = optim.param_groups[0]['lr']
                print(f"   📊 Current learning rate: {current_lr:.2e}")
            
            if grad_norm_scheduler is not None:
                print(f"   📊 Current grad norm: {policy._max_grad_norm:.4f}")
            
            print(f"{'='*60}\n")

    # 训练
    trainer = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        repeat_per_collect=args.repeat_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        step_per_collect=args.step_per_collect,
        save_best_fn=save_best_fn,
        save_checkpoint_fn=save_checkpoint_fn,
        logger=exp_logger,
        test_in_train=True,
        verbose=True,
    )
    
    # 打印训练配置摘要
    print("\n" + "="*60)
    print("🚀 Training Configuration Summary")
    print("="*60)
    print(f"📦 Environment: {args.control_type} (trajectory {args.trajectory_type})")
    print(f"🎯 Total epochs: {args.epoch}")
    print(f"🔄 Steps per epoch: {args.step_per_epoch:,}")
    print(f"🎲 Training envs: {args.training_num}")
    print(f"🧪 Test envs: {args.test_num}")
    print(f"🧠 Network: {args.hidden_sizes}")
    print(f"📊 Initial learning rate: {args.lr:.2e}")
    
    if args.lr_decay:
        min_lr = args.lr * args.min_lr_ratio
        print(f"   └─ Decays to: {min_lr:.2e} ({args.lr_decay_type})")
    
    print(f"✂️  Gradient clipping: {args.max_grad_norm:.4f}")
    
    if grad_norm_scheduler:
        print(f"   └─ Decays to: {args.final_grad_norm:.4f} ({args.grad_norm_decay_type})")
    
    print("="*60 + "\n")

    # 开始训练
    print("🚀 Starting training...\n")
    result = trainer.run()
    
    pprint.pprint(result)
    
    # 保存最终模型
    final_path = os.path.join(log_path, "final_policy.pth")
    torch.save({
        "model": policy.state_dict(),
        "args": vars(args),
        "result": result,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"💾 Final model saved: {final_path}")
    print(f"📊 Best model: {os.path.join(log_path, 'best_policy.pth')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train_px4_ppo()