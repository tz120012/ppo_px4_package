#!/usr/bin/env python3
"""
PX4四旋翼PID参数优化PPO训练脚本
基于tianshou框架实现
"""

import argparse
import datetime
import os
import pprint
from typing import Tuple

import matplotlib.pyplot as plt
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

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO for PX4 Quadcopter PID Optimization")
    
    # 环境相关参数
    parser.add_argument("--control-type", type=str, default="xyz_pos", choices=["xyz_pos", "xy_vel_z_pos", "xyz_vel"], help="Control mode for the quadcopter")
    parser.add_argument("--trajectory-type", type=int, default=5, help="Trajectory type (0-13)")
    parser.add_argument("--yaw-type", type=int, default=3, help="Yaw control type (0-4)")
    parser.add_argument("--max-episode-steps", type=int, default=100, help="Maximum steps per episode")      ##
    parser.add_argument("--rl-dt", type=float, default=0.5, help="RL decision timestep (seconds)")          ##
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Simulation timestep (seconds)")        ##
    parser.add_argument("--wind-type", type=str, default="None", help="Wind disturbance type")                       
    parser.add_argument("--wind-strength", type=float, default=0.0, help="Wind strength")
    # parser.add_argument("--dense-reward-scale", type=float, default=0.1, help="Scale factor for dense reward terms in environment")
    parser.add_argument("--test-env-mode", action="store_true", default=False, help="Enable single-env mode and write per-step info to CSV within env.step")  # 测试环境默认为False，记录step的info数据

    # 训练参数
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=4096)                   ##
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])  ##
    parser.add_argument("--lr", type=float, default=3e-4)                           ##
    parser.add_argument("--gamma", type=float, default=0.99)                        ##
    parser.add_argument("--epoch", type=int, default=800)                           ##
    parser.add_argument("--step-per-epoch", type=int, default=2048)                ##
    parser.add_argument("--step-per-collect", type=int, default=512)              ## 4096
    parser.add_argument("--repeat-per-collect", type=int, default=10)               ## 每次收集后重复更新6次，10次就太稳了
    parser.add_argument("--batch-size", type=int, default=256)                      ##
    parser.add_argument("--training-num", type=int, default=16)                     ##
    parser.add_argument("--test-num", type=int, default=16)                         ##
    
    # PPO特定参数
    parser.add_argument("--rew-norm", type=int, default=1, help="Reward normalization")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")        ## 
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")             ##
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")                    ##
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.6)                                     ##
    parser.add_argument("--eps-clip", type=float, default=0.25)                                         ##
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=1)
    parser.add_argument("--norm-adv", type=int, default=1)
    parser.add_argument("--recompute-adv", type=int, default=1)
    
    # 课程学习参数
    parser.add_argument("--use-curriculum", action="store_true", default=True, help="Enable curriculum learning")
    parser.add_argument("--curriculum-stages", type=int, default=5, help="Number of curriculum stages")
   
    # 日志和保存
    parser.add_argument("--logdir", type=str, default="log_px4")
    parser.add_argument("--render", type=float, default=0.0)
    parser.add_argument("--save-interval", type=int, default=10, help="Save model every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "both"])
    parser.add_argument("--wandb-project", type=str, default="px4-pid-optimization")
    parser.add_argument("--watch", default=False, action="store_true", help="Watch the play of pre-trained policy only")
    
    return parser.parse_args()

def make_px4_env(
    control_type: str,
    trajectory_type: int,
    yaw_type: int,
    max_episode_steps: int,
    rl_dt: float,
    sim_dt: float,
    # dense_reward_scale: float,
    wind_type: str,
    wind_strength: float,
    seed: int,
    training_num: int,
    test_num: int,
    test_env_mode: bool,
    watch: bool,
) -> Tuple:
    """创建PX4四旋翼环境"""
    
    def make_env(seed_offset: int = 0, enable_step_csv: bool = False):
        def _init():
            env = QuadcopterEnv(
                control_type=control_type,
                trajectory_type=trajectory_type,
                yaw_type=yaw_type,
                max_episode_steps=600 if (watch) else max_episode_steps,
                dt=sim_dt,
                rl_dt=rl_dt,
                # dense_reward_scale=dense_reward_scale,
                wind_type=wind_type,
                wind_strength=wind_strength,
                render_mode="rgb_array" if (watch) else None,           #"human","rgb_array"
                test_env_mode=enable_step_csv,  # 仅在 test_env_mode 下启用写 CSV
            )
            env.reset(seed=seed + seed_offset)
            return env
        return _init
    
    if test_env_mode or watch:
        # 单环境训练/测试，便于在 env.step 内写入 CSV
        env = make_env(seed_offset=0, enable_step_csv=True)()
        train_envs = DummyVectorEnv([lambda: env])  # 使用已创建的同一个 env
        # 测试环境单独一个，以免在测试时复用训练 env 的内部 buffer
        test_envs = DummyVectorEnv([make_env(seed_offset=1, enable_step_csv=True)])
    else:
        # 创建单个环境用于获取空间信息
        env = make_env(seed_offset=0, enable_step_csv=False)()
        # 创建向量化训练环境
        if training_num > 1:
            train_envs = SubprocVectorEnv(
                [make_env(i, enable_step_csv=False) for i in range(training_num)],
                share_memory=False
            )
        else:
            train_envs = DummyVectorEnv([make_env(enable_step_csv=False)])
        # 创建向量化测试环境
        test_envs = SubprocVectorEnv(
            [make_env(training_num + i, enable_step_csv=False) for i in range(test_num)],
            share_memory=False
        )
    
    return env, train_envs, test_envs

class CurriculumManager:
    """课程学习管理器"""
    
    def __init__(self, num_stages: int = 5, start_stage: int = 0,) -> None:
        # 阶段数与索引
        self.num_stages = num_stages
        self.current_stage = int(start_stage)
        self.last_advance_epoch = -10**9  # 初始化为负值以允许立即升级
        self.cooldown_epochs = 1 # 冷却期，避免阶段转换过于频繁

        self.threshold_by_epochs = {
            0:150,      # 阶段0 → 阶段1
            1:300,      # 阶段1 → 阶段2
            2:450,      # 阶段2 → 阶段3
            3:600,      # 阶段3 → 阶段4
            4:750,      # 阶段4不再升级
        }

    def should_advance_by_epochs(self, epoch: int) -> bool:
        """基于训练步数判断是否升级"""
        # 冷却期
        if (epoch - self.last_advance_epoch) < self.cooldown_epochs:
            return False
        # 当前阶段阈值
        thr = self.threshold_by_epochs.get(self.current_stage, None)
        if thr is None:
            return False

        return epoch >= thr and (self.current_stage < self.num_stages - 1)

    def advance_stage(self, epoch: int) -> None:
        # 按照 0 -> 1 -> 2 -> 3 -> 4的顺序进行转换
        if self.current_stage == 0:
            self.current_stage = 1
        elif self.current_stage == 1:
            self.current_stage = 2
        elif self.current_stage == 2:
            self.current_stage = 3
        elif self.current_stage == 3:
            self.current_stage = 4
        # 当 stage 为 4 时，不再改变

        self.last_advance_epoch = int(epoch)

    def get_stage_index(self) -> int:
        return self.current_stage

    def reset(self):
        self.current_stage = 0
        self.last_advance_epoch = -10**9  # 允许立即升级


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
        # dense_reward_scale=args.dense_reward_scale,
        wind_type=args.wind_type,
        wind_strength=args.wind_strength,
        seed=args.seed,
        training_num=args.training_num,
        test_num=args.test_num,
        test_env_mode=args.test_env_mode,
        watch=args.watch,
    )
    
    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.shape
    args.max_action = env.action_space.high[0]
    args.min_action = env.action_space.low[0]
    
    print(f"Observations shape: {args.state_shape}")
    print(f"Actions shape: {args.action_shape}")
    print(f"Action range: [{args.min_action}, {args.max_action}]")
    print(f"Device: {args.device}")
    
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

    #############################################################################################
    # 加载之前的策略
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        print(f"Loaded agent from: {args.resume_path}")
    
    # 创建缓冲区
    if args.training_num > 1 and not args.test_env_mode:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    
    # 创建收集器
    train_collector = Collector[CollectStats](policy,train_envs,buffer,exploration_noise=True)
    test_collector = Collector[CollectStats](policy,test_envs)

    # 设置日志
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    log_name = os.path.join(str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)
    
    # 创建logger
    logger_factory = LoggerFactoryDefault()
    if args.logger == "wandb":
        logger_factory.logger_type = "wandb"
        logger_factory.wandb_project = args.wandb_project
    elif args.logger == "tensorboard":
        logger_factory.logger_type = "tensorboard"
    elif args.logger == "both":
        # LoggerFactoryDefault 当前仅支持单一类型，这里简单地默认 tensorboard。
        logger_factory.logger_type = "tensorboard"
    
    exp_logger = logger_factory.create_logger(
        log_dir=log_path,
        experiment_name=log_name,
        run_id=args.resume_id,
        config_dict=vars(args),     # Tensorboard/W&B 记录所有超参配置，便于复现
    )
    
    # 课程学习管理器
    curriculum_manager = None
    if args.use_curriculum:
        curriculum_manager = CurriculumManager(
            num_stages=args.curriculum_stages,
        )
        print("Curriculum learning enabled")
      
    # 自定义训练函数（用于课程学习）
    # -------------------------------------
    def train_fn(epoch: int, env_step: int) -> None:
        """在每个epoch开始时调用"""
        if curriculum_manager is not None and curriculum_manager.should_advance_by_epochs(epoch):
            curriculum_manager.advance_stage(epoch)
            new_stage = curriculum_manager.get_stage_index()
            print(f"[Curriculum] advance to stage {new_stage} \n")

            # 3) 更新训练环境配置
            try:
                train_collector.reset_env(gym_reset_kwargs={"options": {"course_stage": new_stage}})
            except Exception as ex:
                print(f"[Curriculum] Failed to update course stage on train envs: {ex}")

    # 保存最佳策略函数
    def save_best_fn(policy: BasePolicy) -> None:
        state = {
            "model": policy.state_dict(),
            "args": vars(args),
            "curriculum_stage": curriculum_manager.current_stage if curriculum_manager is not None else 0,
        }
        print(f"current curriculum stage: {curriculum_manager.current_stage if curriculum_manager else 0}")
        torch.save(state, os.path.join(log_path, "best_policy.pth"))

    if not args.watch:
        # 训练器
        trainer = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            repeat_per_collect=args.repeat_per_collect,
            episode_per_test=1 if args.test_env_mode else args.test_num,  # 单环境下每次仅测1个episode
            batch_size=args.batch_size,
            step_per_collect=args.step_per_collect,
            save_best_fn=save_best_fn,
            logger=exp_logger,
            test_in_train=True,
            train_fn=train_fn if curriculum_manager else None,
            verbose=True,
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
    
    else:
        print("仅观看模式：加载已训练模型，与环境交互 1 个回合，并绘图")
        if args.resume_path is None:
            raise ValueError("Please provide --resume-path to watch the pre-trained policy.")

        ckpt = torch.load(args.resume_path, map_location=args.device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        policy.load_state_dict(state_dict)
        policy.eval()

        # 重置收集器与环境
        test_collector.reset()
        result = test_collector.collect(n_episode=1, render=args.render)

        print(f"观看结果: {result}")

if __name__ == "__main__":
    train_px4_ppo()