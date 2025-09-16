#!/usr/bin/env python3
"""
PX4四旋翼PID参数优化PPO训练脚本
基于tianshou框架实现
"""

import argparse
import datetime
import os
import pprint
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution, Independent, Normal
from torch.optim.lr_scheduler import LambdaLR

from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic

# 导入你的四旋翼环境
from px4Env import QuadcopterEnv

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO for PX4 Quadcopter PID Optimization")
    
    # 环境相关参数
    parser.add_argument("--control-type", type=str, default="xyz_pos",
                       choices=["xyz_pos", "xy_vel_z_pos", "xyz_vel"],
                       help="Control mode for the quadcopter")
    parser.add_argument("--trajectory-type", type=int, default=5,
                       help="Trajectory type (0-13)")
    parser.add_argument("--yaw-type", type=int, default=3,
                       help="Yaw control type (0-4)")
    parser.add_argument("--max-episode-steps", type=int, default=2000,
                       help="Maximum steps per episode")
    parser.add_argument("--rl-dt", type=float, default=1.0,
                       help="RL decision timestep (seconds)")
    parser.add_argument("--sim-dt", type=float, default=0.01,
                       help="Simulation timestep (seconds)")
    parser.add_argument("--segment-window", type=float, default=3.0,
                       help="Sliding window length for reward calculation (seconds)")
    parser.add_argument("--wind-type", type=str, default="None",
                       help="Wind disturbance type")
    parser.add_argument("--wind-strength", type=float, default=2.0,
                       help="Wind strength")
    
    # 训练参数
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=2048)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--step-per-epoch", type=int, default=20780)
    parser.add_argument("--step-per-collect", type=int, default=2048)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-num", type=int, default=4)
    parser.add_argument("--test-num", type=int, default=4)
    
    # PPO特定参数
    parser.add_argument("--rew-norm", type=int, default=1,
                       help="Reward normalization")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                       help="Value function coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda")
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=1)
    
    # 课程学习参数
    parser.add_argument("--use-curriculum", action="store_true", default=False,
                       help="Enable curriculum learning")
    parser.add_argument("--curriculum-stages", type=int, default=5,
                       help="Number of curriculum stages")
    parser.add_argument("--stage-threshold", type=float, default=0.75,
                       help="Success rate threshold to advance stage")
    
    # 日志和保存
    parser.add_argument("--logdir", type=str, default="log_px4")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--save-interval", type=int, default=10,
                       help="Save model every N epochs")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "both"],
    )
    parser.add_argument("--wandb-project", type=str, default="px4-pid-optimization")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="Watch the play of pre-trained policy only",
    )
    parser.add_argument(
        "--save-video",
        default=False,
        action="store_true",
        help="Save test episodes as video",
    )
    
    return parser.parse_args()


def make_px4_env(
    control_type: str,
    trajectory_type: int,
    yaw_type: int,
    max_episode_steps: int,
    rl_dt: float,
    sim_dt: float,
    segment_window: float,
    wind_type: str,
    wind_strength: float,
    seed: int,
    training_num: int,
    test_num: int,
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
                rl_dt=rl_dt,
                segment_window=segment_window,
                wind_type=wind_type,
                wind_strength=wind_strength,
                render_mode=None,
            )
            env.reset(seed=seed + seed_offset)
            return env
        return _init
    
    # 创建单个环境用于获取空间信息
    env = make_env()()
    
    # 创建向量化训练环境
    if training_num > 1:
        train_envs = SubprocVectorEnv(
            [make_env(i) for i in range(training_num)],
            share_memory=False
        )
    else:
        train_envs = DummyVectorEnv([make_env()])
    
    # 创建向量化测试环境
    test_envs = SubprocVectorEnv(
        [make_env(training_num + i) for i in range(test_num)],
        share_memory=False
    )
    
    return env, train_envs, test_envs


class CurriculumManager:
    """课程学习管理器"""
    
    def __init__(self, num_stages: int = 5, threshold: float = 0.75):
        self.num_stages = num_stages
        self.threshold = threshold
        self.current_stage = 0
        self.stage_performance = []
        
        # 定义每个阶段的配置
        self.stage_configs = [
            {"trajectory_type": 0, "control_type": "xyz_pos"},  # 悬停
            {"trajectory_type": 1, "control_type": "xyz_pos"},  # 简单圆形
            {"trajectory_type": 2, "control_type": "xyz_pos"},  # 8字形
            {"trajectory_type": 3, "control_type": "xyz_pos"},  # 螺旋
            {"trajectory_type": 5, "control_type": "xyz_pos"},  # 复杂轨迹
        ]
    
    def should_advance(self, success_rate: float) -> bool:
        """判断是否应该进入下一阶段"""
        self.stage_performance.append(success_rate)
        
        if success_rate >= self.threshold and self.current_stage < self.num_stages - 1:
            return True
        return False
    
    def advance_stage(self) -> dict:
        """进入下一阶段"""
        self.current_stage = min(self.current_stage + 1, self.num_stages - 1)
        return self.get_current_config()
    
    def get_current_config(self) -> dict:
        """获取当前阶段配置"""
        return self.stage_configs[self.current_stage]
    
    def reset(self):
        """重置课程进度"""
        self.current_stage = 0
        self.stage_performance = []


def train_px4_ppo(args: argparse.Namespace = get_args()) -> None:
    """主训练函数"""
    
    # 创建环境
    env, train_envs, test_envs = make_px4_env(
        control_type=args.control_type,
        trajectory_type=args.trajectory_type,
        yaw_type=args.yaw_type,
        max_episode_steps=args.max_episode_steps,
        rl_dt=args.rl_dt,
        sim_dt=args.sim_dt,
        segment_window=args.segment_window,
        wind_type=args.wind_type,
        wind_strength=args.wind_strength,
        seed=args.seed,
        training_num=args.training_num,
        test_num=args.test_num,
    )
    
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    args.min_action = env.action_space.low[0]
    
    logger.info(f"Observations shape: {args.state_shape}")
    logger.info(f"Actions shape: {args.action_shape}")
    logger.info(f"Action range: [{args.min_action}, {args.max_action}]")
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
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
        unbounded=False,  # PID参数有界
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
    
    # 初始化网络权重
    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    
    # Actor输出层特殊初始化
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    
    # 优化器
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr, eps=1e-5)
    
    # 学习率调度器
    lr_scheduler = None
    if args.lr_decay:
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
        lr_scheduler = LambdaLR(
            optim,
            lr_lambda=lambda epoch: 1 - epoch / max_update_num
        )
    
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
    
    # 加载之前的策略
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt["model"])
        logger.info(f"Loaded agent from: {args.resume_path}")
    
    # 创建缓冲区
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    
    # 创建收集器
    train_collector = Collector[CollectStats](policy,train_envs,buffer,exploration_noise=True)
    test_collector = Collector[CollectStats](policy,test_envs)
    
    # 课程学习管理器
    curriculum_manager = None
    if args.use_curriculum:
        curriculum_manager = CurriculumManager(
            num_stages=args.curriculum_stages,
            threshold=args.stage_threshold
        )
        logger.info("Curriculum learning enabled")
    
    # 设置日志
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ppo"
    log_name = os.path.join("px4_quad", args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    
    # 创建logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    elif args.logger == "tensorboard":
        logger_factory.logger_type = "tensorboard"
    
    exp_logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),
    )
    
    # 保存最佳策略函数
    def save_best_fn(policy: BasePolicy) -> None:
        state = {
            "model": policy.state_dict(),
            "args": vars(args),
            "curriculum_stage": curriculum_manager.current_stage if curriculum_manager else 0,
        }
        torch.save(state, os.path.join(log_path, "best_policy.pth"))
        logger.info(f"Best policy saved to {log_path}/best_policy.pth")
    
    # 保存检查点函数
    def save_checkpoint_fn(epoch: int, env_step: int) -> None:
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(log_path, f"checkpoint_epoch_{epoch}.pth")
            state = {
                "model": policy.state_dict(),
                "optim": optim.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "env_step": env_step,
                "curriculum_stage": curriculum_manager.current_stage if curriculum_manager else 0,
            }
            torch.save(state, ckpt_path)
            logger.info(f"Checkpoint saved to {ckpt_path}")
    
    # 自定义训练函数（用于课程学习）
    def train_fn(epoch: int, env_step: int) -> None:
        """在每个epoch开始时调用"""
        if curriculum_manager is not None and epoch > 0:
            # 获取最近的测试性能
            if hasattr(test_collector, 'last_rew'):
                avg_reward = test_collector.last_rew
                # 简单的成功判据：平均奖励超过某个阈值
                success_rate = min(1.0, max(0.0, (avg_reward + 500) / 1000))
                
                if curriculum_manager.should_advance(success_rate):
                    new_config = curriculum_manager.advance_stage()
                    logger.info(f"Advancing to curriculum stage {curriculum_manager.current_stage}")
                    logger.info(f"New config: {new_config}")
                    
                    # 更新环境配置
                    for env in train_envs:
                        if hasattr(env, 'set_course_stage'):
                            env.set_course_stage(curriculum_manager.current_stage)
    
    def test_fn(epoch: int, env_step: int) -> None:
        """在每个测试后调用"""
        if epoch % args.save_interval == 0:
            # 使用当前训练步数，而不是policy.gradient_step
            save_checkpoint_fn(epoch, env_step)
    
    if not args.watch:
        # 训练器
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
            logger=exp_logger,
            test_in_train=False,
            train_fn=train_fn if curriculum_manager else None,
            test_fn=test_fn,
        )
        
        result = trainer.run()
        pprint.pprint(result)
        
        # 保存最终模型
        final_path = os.path.join(log_path, "final_policy.pth")
        state = {
            "model": policy.state_dict(),
            "args": vars(args),
            "result": result,
        }
        torch.save(state, final_path)
        logger.info(f"Final policy saved to {final_path}")
    
    # 观看训练好的策略表现
    if args.watch:
        logger.info("Testing trained policy...")
        test_envs.seed(args.seed)
        test_collector.reset()
        
        # 收集测试数据
        collector_stats = test_collector.collect(
            n_episode=args.test_num,
            render=args.render
        )
        
        logger.info(f"Test results: {collector_stats}")
        
        # 如果需要保存视频或数据
        if args.save_video:
            # 创建一个带渲染的环境
            render_env = QuadcopterEnv(
                control_type=args.control_type,
                trajectory_type=args.trajectory_type,
                yaw_type=args.yaw_type,
                max_episode_steps=args.max_episode_steps,
                dt=args.sim_dt,
                rl_dt=args.rl_dt,
                segment_window=args.segment_window,
                render_mode='data_log',
            )
            
            # 运行一个episode并保存数据
            obs, info = render_env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            
            while not (done or truncated):
                action, _ = policy(obs[np.newaxis, :])
                action = action.squeeze(0)
                obs, reward, done, truncated, info = render_env.step(action)
                episode_reward += reward
            
            # 渲染数据到文件
            render_env.render(mode='data_log', save_root=log_path)
            logger.info(f"Episode data saved. Total reward: {episode_reward}")
            
            render_env.close()


if __name__ == "__main__":
    train_px4_ppo()