#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PX4四旋翼环境PPO训练脚本
基于Tianshou框架训练PID参数优化任务

作者: 基于原始PPO示例修改
用途: 训练强化学习智能体学习最优PID控制器参数
"""

import argparse
import datetime
import os
import pprint
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal

from tianshou.algorithm import PPO  
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.trainer import OnPolicyTrainerParams  
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ContinuousActorProbabilistic, ContinuousCritic

# 导入我们的PX4环境
from px4Env import QuadcopterEnv


def get_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PX4四旋翼PPO训练")
    
    # 环境相关参数
    parser.add_argument("--task", type=str, default="PX4Quadcopter", help="任务名称")
    parser.add_argument("--control_type", type=str, default="xyz_pos", choices=["xyz_pos", "xy_vel_z_pos", "xyz_vel"], help="控制类型")
    parser.add_argument("--trajectory_type", type=int, default=5, help="轨迹类型 (0-13)")
    parser.add_argument("--yaw_type", type=int, default=3, help="偏航控制类型 (0-4)")
    parser.add_argument("--max_episode_steps", type=int, default=4000, help="每个回合的最大步数")
    parser.add_argument("--dt", type=float, default=0.005, help="控制时间步长")
    parser.add_argument("--rl_dt", type=float, default=0.1, help="RL决策时间步长")
    parser.add_argument("--segment_window", type=float, default=3.0, help="分段累计窗口长度（秒）")
    parser.add_argument("--target_pos", type=float, nargs=3, default=[2.0, 2.0, 1.0], help="目标位置 [x, y, z]")
    parser.add_argument("--wind_type", type=str, default="None", help="风模型类型")
    parser.add_argument("--wind_strength", type=float, default=2.0, help="风力强度")
    
    # 训练相关参数
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--buffer_size", type=int, default=4096, help="经验回放缓冲区大小")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[256, 256], help="网络隐藏层大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epoch", type=int, default=50, help="训练轮数")
    parser.add_argument("--epoch_num_steps", type=int, default=20000, help="每轮步数")
    parser.add_argument("--collection_step_num_env_steps", type=int, default=2048, help="每次收集的环境步数")
    parser.add_argument("--update_step_num_repetitions", type=int, default=10, help="每次更新的重复次数")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--num_train_envs", type=int, default=8, help="训练环境数量")
    parser.add_argument("--num_test_envs", type=int, default=4, help="测试环境数量")
    
    # PPO特定参数
    parser.add_argument("--return_scaling", type=bool, default=True, help="返回值缩放")
    parser.add_argument("--vf_coef", type=float, default=0.25, help="价值函数系数")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="熵系数")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE Lambda")
    parser.add_argument("--bound_action_method", type=str, default="clip", help="动作边界方法")
    parser.add_argument("--lr_decay", type=bool, default=True, help="学习率衰减")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="最大梯度范数")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO剪切参数")
    parser.add_argument("--dual_clip", type=float, default=None, help="双重剪切")
    parser.add_argument("--value_clip", type=bool, default=False, help="价值剪切")
    parser.add_argument("--advantage_normalization", type=bool, default=False, help="优势标准化")
    parser.add_argument("--recompute_adv", type=bool, default=True, help="重新计算优势")
    
    # 日志和保存
    parser.add_argument("--logdir", type=str, default="log_px4", help="日志目录")
    parser.add_argument("--render", type=float, default=0.0, help="渲染频率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--resume_path", type=str, default=None, help="恢复训练路径")
    parser.add_argument("--resume_id", type=str, default=None, help="恢复训练ID")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="日志记录器类型")
    parser.add_argument("--wandb_project", type=str, default="px4_ppo_drl", help="WandB项目名")
    parser.add_argument("--watch", default=False, action="store_true", help="仅观看预训练策略表现")

    return parser.parse_args()

def make_px4_env(args: argparse.Namespace, training: bool = True):
    """创建PX4环境函数"""
    def _make_env():
        env = QuadcopterEnv(
            control_type=args.control_type,
            trajectory_type=args.trajectory_type,
            yaw_type=args.yaw_type,
            max_episode_steps=args.max_episode_steps,
            dt=args.dt,
            target_pos=np.array(args.target_pos),
            wind_type=args.wind_type,
            wind_strength=args.wind_strength,
            render_mode=None  # 训练时不渲染以提高速度
        )
        return env
    
    return _make_env


def make_px4_envs(args: argparse.Namespace):
    """创建训练和测试环境"""
    # 创建单个环境以获取空间信息
    single_env = make_px4_env(args)()
    
    # 创建训练环境
    if args.num_train_envs > 1:
        train_envs = SubprocVectorEnv(
            [make_px4_env(args, training=True) for _ in range(args.num_train_envs)]
        )
    else:
        train_envs = DummyVectorEnv([make_px4_env(args, training=True)])
    
    # 创建测试环境
    if args.num_test_envs > 1:
        test_envs = SubprocVectorEnv(
            [make_px4_env(args, training=False) for _ in range(args.num_test_envs)]
        )
    else:
        test_envs = DummyVectorEnv([make_px4_env(args, training=False)])
    
    return single_env, train_envs, test_envs


def main(args: Optional[argparse.Namespace] = None) -> None:
    """主训练函数"""
    if args is None:
        args = get_args()
    
    print("=== PX4四旋翼PPO训练开始 ===")
    print(f"控制类型: {args.control_type}")
    print(f"轨迹类型: {args.trajectory_type}")
    print(f"目标位置: {args.target_pos}")
    print(f"训练环境数: {args.num_train_envs}")
    print(f"测试环境数: {args.num_test_envs}")
    print(f"设备: {args.device}")
    
    # 创建环境
    env, train_envs, test_envs = make_px4_envs(args)
    
    # 获取环境信息
    args.state_shape = env.observation_space.shape or (26,)  # 默认观测维度
    args.action_shape = env.action_space.shape or (20,)      # 默认动作维度
    args.max_action = 1.0  # PX4环境的动作空间是[0,1]
    
    print(f"观测空间形状: {args.state_shape}")
    print(f"动作空间形状: {args.action_shape}")
    print(f"动作范围: [0.0, 1.0]")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建神经网络
    # Actor网络
    net_a = Net(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    actor = ContinuousActorProbabilistic(
        preprocess_net=net_a,
        action_shape=args.action_shape,
        unbounded=False,
    ).to(args.device)
    
    # Critic网络
    net_c = Net(
        state_shape=args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
    )
    critic = ContinuousCritic(
        preprocess_net=net_c,
    ).to(args.device)
    
    # 组合Actor-Critic
    actor_critic = ActorCritic(actor, critic)
    
    # 网络初始化
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # 正交初始化
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    # 最后一层策略层缩放，这将使初始动作接近0均值和标准差
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    # 优化器
    optim = AdamOptimizerFactory(lr=args.lr)
    
    if args.lr_decay:
        optim.with_lr_scheduler_factory(
            LRSchedulerFactoryLinear(
                max_epochs=args.epoch,
                epoch_num_steps=args.epoch_num_steps,
                collection_step_num_env_steps=args.collection_step_num_env_steps,
            )
        )
    
    # 分布函数
    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)
    
    # 策略
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist,
        action_scaling=False,
        action_bound_method=None,
        action_space=env.action_space,
    )
    
    # PPO算法
    algorithm: PPO = PPO(
        policy=policy,
        critic=critic,
        optim=optim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        return_scaling=args.return_scaling,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.advantage_normalization,
        recompute_advantage=args.recompute_adv,
    )
    
    # 加载之前的策略（如果有）
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        algorithm.load_state_dict(ckpt["model"])
        print(f"从以下路径加载智能体: {args.resume_path}")
    
    # 收集器
    if args.num_train_envs > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    
    train_collector = Collector[CollectStats](
        algorithm, train_envs, buffer, exploration_noise=True
    )
    test_collector = Collector[CollectStats](algorithm, test_envs)
    
    # 日志配置
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    
    # 日志记录器
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
    
    def save_best_fn(policy: Algorithm) -> None:
        """保存最佳策略"""
        state = {"model": policy.state_dict()}
        torch.save(state, os.path.join(log_path, "policy.pth"))
        print(f"保存最佳策略到: {os.path.join(log_path, 'policy.pth')}")
    
    if not args.watch:
        # 开始训练
        print("\n=== 开始PPO训练 ===")
        print(f"日志保存到: {log_path}")
        
        result = algorithm.run_training(
            OnPolicyTrainerParams(
                train_collector=train_collector,
                test_collector=test_collector,
                max_epochs=args.epoch,
                epoch_num_steps=args.epoch_num_steps,
                update_step_num_repetitions=args.update_step_num_repetitions,
                test_step_num_episodes=args.num_test_envs,
                batch_size=args.batch_size,
                collection_step_num_env_steps=args.collection_step_num_env_steps,
                save_best_fn=save_best_fn,
                logger=logger,
                test_in_train=False,
            )
        )

        print("\n=== 训练完成 ===")
        pprint.pprint(result)
    
    # 观看表现
    print("\n=== 测试训练好的策略 ===")
    test_envs.seed(args.seed)
    test_collector.reset()
    collector_stats = test_collector.collect(
        n_episode=args.num_test_envs, 
        render=args.render
    )
    print("测试结果:")
    print(collector_stats)
    
    # 清理环境
    env.close()
    train_envs.close()
    test_envs.close()
    
    print("\n=== PX4四旋翼PPO训练结束 ===")


if __name__ == "__main__":
    main()