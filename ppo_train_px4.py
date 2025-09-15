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
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from tianshou.algorithm import PPO  
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory, LRSchedulerFactoryLinear
from tianshou.data import Collector, CollectStats, ReplayBuffer, VectorReplayBuffer, Batch
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
    parser.add_argument("--max_episode_steps", type=int, default=2000, help="每个回合的最大步数")
    parser.add_argument("--dt", type=float, default=0.005, help="控制时间步长")
    parser.add_argument("--rl_dt", type=float, default=0.1, help="RL决策时间步长")
    parser.add_argument("--segment_window", type=float, default=3.0, help="分段累计窗口长度（秒）")
    parser.add_argument("--target_pos", type=float, nargs=3, default=[2.0, 2.0, 1.0], help="目标位置 [x, y, z]")
    parser.add_argument("--wind_type", type=str, default="None", help="风模型类型")
    parser.add_argument("--wind_strength", type=float, default=2.0, help="风力强度")
    parser.add_argument("--course_stage_init", type=int, default=0, help="初始课程学习等级")
    
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
    parser.add_argument("--render", type=float, default=0.0, help="渲染频率 (0.0表示不渲染)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    parser.add_argument("--resume_path", type=str, default=None, help="恢复训练路径")
    parser.add_argument("--resume_id", type=str, default=None, help="恢复训练ID")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="日志记录器类型")
    parser.add_argument("--wandb_project", type=str, default="px4_ppo_drl", help="WandB项目名")
    parser.add_argument("--watch", default=False, action="store_true", help="仅观看预训练策略表现")

    # 最佳策略模型可视化测试参数
    parser.add_argument("--viz_mode", type=str, default="human",
                        choices=["none", "data_log", "human"],
                        help="训练完成后可视化模式：none 不可视化；data_log 保存列式CSV；human 绘图+动画")
    parser.add_argument("--viz_seed", type=int, default=None,
                        help="可视化回合的随机种子（默认沿用 --seed）")


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
    single_env  = make_px4_env(args)()
    
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
    
    return single_env , train_envs, test_envs

# 新增函数：训练后单环境评估与一次性可视化
def eval_and_visualize_once(args, algorithm, policy, log_path ):
    # 1) 加载最佳checkpoint（若存在）
    best_ckpt = os.path.join(log_path, "policy.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=args.device, weights_only=False)
        # 依据你的保存格式选择：若保存了 policy/algorithm 的 state_dict
        if "model" in ckpt:
            algorithm.load_state_dict(ckpt["model"])
        elif "state_dict" in ckpt:
            algorithm.load_state_dict(ckpt["state_dict"])
        else:
            # 若直接保存了 policy 的
            policy.load_state_dict(ckpt)
        print(f"[Viz] Loaded best policy from {best_ckpt}")
    else:
        print(f"[Viz] Best checkpoint not found at {best_ckpt}, using algorithm in memory.")

    # 2) 评估模式 & 设备
    algorithm.eval()
    policy.eval()
    policy.to(args.device)

    # 3) 创建单环境并 reset
    vis_env = make_px4_env(args, training=False)()
    viz_seed = getattr(args, "viz_seed", None) or getattr(args, "seed", None)
    obs, _ = vis_env.reset(seed=viz_seed)

    done, truncated = False, False

    with torch.no_grad():
        while not (done or truncated):
            # A) 用 Batch 进行封装
            batch = Batch(obs=np.asarray(obs, dtype=np.float32)[None, ...], info=None)

            action_tensor = None

            # B-1) 优先走 policy 的直接调用（Tianshou 推荐）
            try:
                # policy(batch, state=None, explore=False) 版本差异较多，以下两种都尝试
                out = policy(batch)
                # 常见返回含 "act"、"logits" 字段，也可能直接是动作张量
                if isinstance(out, Batch) and hasattr(out, "act") and out.act is not None:
                    action_tensor = out.act
                elif torch.is_tensor(out):
                    action_tensor = out
            except Exception:
                pass

            # B-2) 尝试 policy.forward
            if action_tensor is None:
                try:
                    out = policy.forward(batch)
                    if isinstance(out, Batch) and hasattr(out, "act") and out.act is not None:
                        action_tensor = out.act
                    elif isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
                        action_tensor = out[0]
                except Exception:
                    pass

            # B-3) 兜底：从 actor 获取分布或原始输出
            if action_tensor is None:
                obs_tensor = torch.as_tensor(obs, device=args.device, dtype=torch.float32).unsqueeze(0)
                if hasattr(policy, "actor") and policy.actor is not None:
                    raw = None
                    # 优先 dist 接口
                    if hasattr(policy.actor, "dist"):
                        try:
                            dist = policy.actor.dist(obs_tensor)
                            action_tensor = dist.mean
                        except Exception:
                            pass
                    if action_tensor is None:
                        # 直接调用 actor
                        raw = policy.actor(obs_tensor)
                        # 直接返回分布
                        if isinstance(raw, torch.distributions.Distribution):
                            action_tensor = raw.mean
                        # 返回 (loc, scale)
                        elif isinstance(raw, (tuple, list)) and len(raw) == 2:
                            loc, scale = raw
                            if not torch.is_tensor(loc):
                                loc = torch.as_tensor(np.array(loc), device=args.device, dtype=torch.float32)
                            else:
                                loc = loc.to(device=args.device, dtype=torch.float32)
                            if not torch.is_tensor(scale):
                                scale = torch.as_tensor(np.array(scale), device=args.device, dtype=torch.float32)
                            else:
                                scale = scale.to(device=args.device, dtype=torch.float32)
                            dist = torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)
                            action_tensor = dist.mean
                        else:
                            raise RuntimeError(f"Unsupported actor output type: {type(raw)}")
                else:
                    raise RuntimeError("Policy has no actor; unable to derive action.")

            # C) 连续动作裁剪到 [0, 1]（你的动作空间定义）
            action_tensor = action_tensor.to(dtype=torch.float32, device=args.device)
            action_tensor = action_tensor.clamp(0.0, 1.0)

            # D) 转 numpy 并交给环境
            action_np = action_tensor.squeeze(0).detach().cpu().numpy()
            obs, reward, done, truncated, info = vis_env.step(action_np)

    # 4) 渲染输出
    try:
        if getattr(args, "viz_mode", "data_log") == "data_log":
            vis_env.render(mode="data_log")
            print("[Viz] Data log saved by env.render('data_log').")
        elif args.viz_mode == "human":
            vis_env.render(mode="human")
            print("[Viz] Plotted figures and animation via env.render('human').")
        else:
            print("[Viz] viz_mode=none, skip visualization.")
    except Exception as e:
        print(f"[Viz] Visualization error: {e}")

    vis_env.close()




# # ############# 加入课程学习方案 #############
# def curriculum_learning_scheme(course_stage:int, test_avg_reward:float) -> int:
#     """课程学习方案"""
#     last_stage = course_stage
#     # 当测试时的平均奖励超过某个阈值时，课程等级+1
#     if test_avg_reward < 100 :
#         if course_stage < 4:
#             # 初始阶段，使用简单的目标位置
#             course_stage += 1
#         else:
#             print(f"课程学习已完成所有阶段。")

#     if course_stage == last_stage + 1:
#         print(f"Curriculum stage upgrade : Current-Stage {course_stage}\n")
#     return course_stage

# def apply_course_stage_to_envs(train_envs, test_envs, stage: int):
#     # 要求你的 QuadcopterEnv 实现了 set_course_stage()
#     # 若未实现，请在环境中添加该方法，内部根据 stage 调整目标、风力、轨迹等
#     try:
#         # 对 SubprocVectorEnv/DummyVectorEnv 使用 .env_method
#         if hasattr(train_envs, "env_method"):
#             train_envs.env_method("set_course_stage", stage)
#             test_envs.env_method("set_course_stage", stage)
#         else:
#             # 退化：逐个访问
#             for e in train_envs.envs: 
#                 e.set_course_stage(stage)
#             for e in test_envs.envs:
#                 e.set_course_stage(stage)
#         print(f"[Curriculum] Applied course stage = {stage} to all envs.")
#     except AttributeError:
#         print("Warning: Env does not implement set_course_stage(stage). Please add it to QuadcopterEnv.")

def main(args: Optional[argparse.Namespace] = None) -> None:
    """主训练函数"""
    if args is None:
        args = get_args()
    course_stage = args.course_stage_init

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
        print("\n=== 开始PPO训练（带课程学习） ===")
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
                verbose = False,
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
        render=float(args.render)
    )
    print("测试结果:")
    print(collector_stats)

    # 4) 在关闭 env 前调用一次性可视化
    log_path = "/home/ai/ws_tianshou_drl/ppo_px4_package/log_px4/PX4Quadcopter/ppo/0/250915-105046"
    eval_and_visualize_once(args, algorithm, policy, log_path)
    
    # 清理环境
    env.close()
    train_envs.close()
    test_envs.close()
    
    print("\n=== PX4四旋翼PPO训练结束 ===")


if __name__ == "__main__":
    main()