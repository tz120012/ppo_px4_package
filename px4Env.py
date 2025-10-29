# -*- coding: utf-8 -*-
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils

import logging
# 初始化日志
logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("PX4Env")

def _quaternion_to_euler(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    phi = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    theta = np.arcsin(2.0 * (w * y - z * x))
    psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([psi, theta, phi])

class QuadcopterEnv(gym.Env):
    """
    PX4四旋翼仿真环境用于强化学习
    """
    def __init__(self,
                 control_type: str = "xyz_vel",
                 trajectory_type: int = 5,
                 yaw_type: int = 3,
                 averVel: float = 1,
                 max_episode_steps: int = 600,
                 rl_dt: float = 0.05,   # 20Hz决策步长：0.05s，20Hz
                 dt: float = 0.005,     # 控制步长：0.005s,200Hz
                 eval_mode: bool = False,  # 评估模式开关
                 target_pos: Optional[np.ndarray] = None,
                 wind_type: str = 'None',
                 wind_strength: float = 0.0,
                 #  姿态越界阈值课程学习参数
                 attitude_bound_start: float = 90.0,  # 初始阈值（度）
                 attitude_bound_end: float = 60.0,    # 最终阈值（度）
                 attitude_bound_decay_epochs: int = 1000,  # 衰减周期（epoch数）
                 compat_truncated_as_done: bool = True,
            ):

        super(QuadcopterEnv, self).__init__()
        
        # ==================== 基础环境参数 ====================
        self.control_type = control_type
        self.trajectory_type = trajectory_type
        self.yaw_type = yaw_type
        self.averVel = averVel
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.rl_dt = rl_dt
        self.eval_mode = eval_mode          
        self.target_pos = target_pos if target_pos is not None else np.array([2.0, 2.0, 1.0])
        self.wind_type = wind_type
        self.wind_strength = wind_strength
        
        self.ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]

        # ==================== 动作空间模式（直接式 vs 增量式） ====================
        self.enable_smooth_transition = True   # train时不用Smooth，测试时用Smooth过渡
        self.alpha = 0.8                        # EMA平滑系数（仅用于平滑过渡，不影响最终收敛）
        
        # 奖励缩放因子
        self.reward_scale = 0.01 #将奖励从[-10000,0]缩放到[-100,0]
        self.failure_penalty = -100000.0   

        #  子步改进相关参数
        self.enable_oscillation_penalty = True  # 是否启用震荡惩罚
        self.compat_truncated_as_done = bool(compat_truncated_as_done)   # 兼容旧训练循环：将截断同时视为 done（terminated）以触发reset
        
        #  根据控制模式自动调整位置和速度权重
        if self.control_type == "xyz_pos":
            # 位置控制模式：位置误差为主要优化目标
            pos_weight_xy = 3.0
            pos_weight_z = 2.0
            vel_weight_xy = 0.5  # 速度作为辅助指标（防止超调）
            vel_weight_z = 0.3
        elif self.control_type == "xy_vel_z_pos":
            # 混合模式：XY速度 + Z位置
            pos_weight_xy = 0.0
            pos_weight_z = 2.0
            vel_weight_xy = 3.0
            vel_weight_z = 1.5  # Z轴速度权重降低，因为主要控制Z位置
        elif self.control_type == "xyz_vel":
            # 速度控制模式：速度误差为主要优化目标
            pos_weight_xy = 0.0
            pos_weight_z = 0.0
            vel_weight_xy = 5.0
            vel_weight_z = 3.0
        else:
            # 默认配置（向后兼容）
            log.warning(f" Unknown control_type: {self.control_type}, using default weights")
            pos_weight_xy = 0.0
            pos_weight_z = 0.0
            vel_weight_xy = 5.0
            vel_weight_z = 3.0

        self.segment_weights = dict(
            # 跟踪误差项（速度 >> 位置）
            W_pos_xy=pos_weight_xy,         # 位置误差XY（速度模式不启用）
            W_pos_z=pos_weight_z,           # 位置误差Z（速度模式不启用）
            W_vel_xy=vel_weight_xy,         #  速度误差XY：0.15 → 3.0（提升20倍）
            W_vel_z=vel_weight_z,           #  速度误差Z：0.1 → 2.0（提升20倍）
            W_att=15.0,                      #  姿态误差：0.1 → 0.8（提升8倍）
            W_omega=1.0,                    # 角速度误差（保持）
            
            # 惩罚项
            W_u=0.0,        #0.0                    # 控制能量
            W_dv=0.0,        #0.05,                 # 速度变化率
            W_domega=0.0,    #0.02,                 # 角速度变化率
            W_du=0.0,        #1.0,                  # 电机指令平滑
            W_sat=0.0,       #0.5,                  # 饱和惩罚
            W_over=0.0,      #0.2,                  # 超调惩罚

            #  PID参数变化惩罚
            W_pid_change=0.0,   #1.0,   
        )
        #  速度跟踪成功阈值
        self.vel_tracking_excellent_threshold = 0.2   # m/s（精确跟踪）
        self.vel_tracking_good_threshold = 0.5        # m/s（良好跟踪）
        self.vel_tracking_poor_threshold = 1.0        # m/s（较差跟踪）

        #  分档奖励系数
        self.vel_reward_excellent_bonus = 50.0   # 精确跟踪的正奖励
        self.vel_reward_poor_penalty = 100.0     # 极差跟踪的额外惩罚


        # ==================== 限制与判据 ====================
        self.limits = dict(
            w_cmd_min=0.0,
            w_cmd_max=2e3,
            overshoot_z_tol=0.2,
            normalize_rate_by_dt=True,
            overshoot_pos_tol=1.0,
            overshoot_vel_tol=0.5,
            overshoot_att_tol=5.0,
            overshoot_ang_vel_tol=30.0,
        )

        # ==================== 回合跟踪 ====================
        self.current_step = 0
        self.current_substep = 0
        self.episode_reward = 0.0

        # ==================== 动作空间定义（直接式映射） ====================
        # 直接式：action ∈ [-1, 1]^20 直接映射到 PID参数 ∈ [K_min, K_max]^20
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # ====================  观测空间定义（去掉误差/参数变化率） ====================
        # 观测维度计算：
        # - 参考状态: pos(3) + vel(3) + eul(3) + omega(3) = 12
        # - 反馈状态: pos(3) + vel(3) + eul(3) + omega(3) = 12
        # - 误差信号: pos_err(3) + vel_err(3) + eul_err(3) + omega_err(3) = 12
        # - PID参数归一化: 20维（当前PID增益值，归一化到[-1,1]）
        # 总计: 12 + 12 + 12 + 20 = 56
        obs_dim = 56
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ==================== PID参数安全范围（物理边界） ====================
        self.pid_safe_min = np.array([
            0.0, 0.1,           # pos_P_gains: Pxy, Pz
            1.2, 2.0,           # vel_P_gains: Pxydot, Pzdot
            0.0, 0.2,           # vel_I_gains: Ixydot, Izdot
            0.1, 0.0,           # vel_D_gains: Dxydot, Dzdot
            0.0, 0.0, 0.0,      # att_P_gains: Pphi, Ptheta, Ppsi
            0.01, 0.01, 0.0,    # rate_P_gains: Pp, Pq, Pr
            0.0, 0.0, 0.0,      # rate_I_gains: Ip, Iq, Ir
            0.0, 0.0, 0.0       # rate_D_gains: Dp, Dq, Dr
        ], dtype=np.float32)
        
        self.pid_safe_max = np.array([
            2.0, 1.5,           # pos_P_gains

            15.0, 15.0,          # vel_P_gains
            60.0, 3.0,          # vel_I_gains
            2.0, 2.0,           # vel_D_gains

            12.0, 12.0, 5.0,    # att_P_gains

            0.5, 0.6, 0.6,      # rate_P_gains
            10.0, 10.0, 10.0,   # rate_I_gains
            0.01, 0.01, 0.01    # rate_D_gains
        ], dtype=np.float32)

        # ==================== 初始化仿真组件 ====================
        self._initialize_simulation()

        # ==================== 基线PID参数（初始化为中值，并应用到ctrl） ====================
        self.pid_K_prev_vec = np.array([
            1.0, 0.8,                   # pos_P_gains: Pxy, Pz
            12, 8.5,                   # vel_P_gains: Pxydot, Pzdot
            10.0, 1.6,                  # vel_I_gains: Ixydot, Izdot
            1.5, 1.0,                  # vel_D_gains: Dxydot, Dzdot
            8.0, 8.0, 2.75,             # att_P_gains: Pphi, Ptheta, Ppsi
            0.5, 0.5, 0.305,            # rate_P_gains: Pp, Pq, Pr
            3.0, 3.0, 5.0,              # rate_I_gains: Ip, Iq, Ir
            0.008, 0.008, 0.005         # rate_D_gains: Dp, Dq, Dr
        ],dtype=np.float32)
        self.init_actions = self.normalize_pid_parameters(self.pid_K_prev_vec)
        self.prev_action = self.init_actions
        self._apply_pid_parameters(self.init_actions)

        # 记录PID变化信息
        self.last_pid_delta_norm = 0.0
        self.last_pid_delta_max = 0.0
        
        # 步级日志与 segment 统计
        self.step_log_rows = []

        # 误差变化率相关缓存已移除
        self.prev_rmse_vel_xy = 0.0
        self.prev_rmse_vel_z = 0.0
  
        # ==================== 姿态越界阈值课程学习 ====================
        self.attitude_bound_start = attitude_bound_start  # 90度（初始）
        self.attitude_bound_end = attitude_bound_end      # 50度（最终）
        self.attitude_bound_decay_epochs = attitude_bound_decay_epochs  # 衰减周期
        
        # 当前阈值（弧度，初始化为起始值）
        self.attitude_bound_current = np.deg2rad(attitude_bound_start)
        
        # 记录当前epoch（由外部训练脚本更新）
        self.current_epoch = 0
        
        log.info(
            f"Attitude bound curriculum: "
            f"{attitude_bound_start}° → {attitude_bound_end}° "
            f"over {attitude_bound_decay_epochs} epochs"
        )

    def normalize_pid_parameters(self, pid_K_vec):
        # 计算范围
        range_vec = self.pid_safe_max - self.pid_safe_min
        # 归一化到 [-1, 1]
        normalized_pid = 2 * (pid_K_vec - self.pid_safe_min) / range_vec - 1
        return normalized_pid
    
    def update_attitude_bound(self, current_epoch: int) -> float:
        """
        根据当前epoch更新姿态越界阈值（线性衰减）
        """
        self.current_epoch = current_epoch
        
        # 线性衰减
        progress = min(current_epoch / self.attitude_bound_decay_epochs, 1.0)
        # # 余弦退火
        # progress = 0.5 * (1 - np.cos(np.pi * min(current_epoch / self.attitude_bound_decay_epochs, 1.0)))  
              
        # 从起始值线性插值到最终值
        current_deg = self.attitude_bound_start + progress * (
            self.attitude_bound_end - self.attitude_bound_start
        )
        
        # 转换为弧度并更新
        self.attitude_bound_current = np.deg2rad(current_deg)
        
        return current_deg
    # =========================
    # 仿真初始化与内部工具
    # =========================
    def _initialize_simulation(self):
        """初始化仿真组件"""
        self.time = 0.0
        self.quad = Quadcopter(self.time)
        self.traj_select = np.array([self.trajectory_type, self.yaw_type, self.averVel])  
        self.traj = Trajectory(self.quad, ctrlType=self.control_type, trajSelect=self.traj_select)
        self.ctrl = Control(self.quad, self.traj.yawType)
        self.wind = Wind(self.wind_type, self.wind_strength, 90, -15)

        sDes = self.traj.desiredState(0, self.dt, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.dt)

        # 计算每个 RL step 的子步数
        self.substeps_per_rl = max(1, int(round(self.rl_dt / self.dt)))
        T = self.max_episode_steps * self.substeps_per_rl

        self.episode_t_all = np.zeros((T,), dtype=float)
        self.episode_s_all = np.zeros((T, len(self.quad.state)))

        self.episode_pos_all = np.zeros((T, len(self.quad.pos)))
        self.episode_vel_all = np.zeros((T, len(self.quad.vel)))
        self.episode_euler_all = np.zeros((T, len(self.quad.euler)))
        self.episode_omega_all = np.zeros((T, len(self.quad.omega)))

        self.episode_pos_des_all = np.zeros((T, len(self.quad.pos)))
        self.episode_vel_des_all = np.zeros((T, len(self.quad.vel)))
        self.episode_euler_des_all = np.zeros((T, len(self.quad.euler)))
        self.episode_omega_des_all = np.zeros((T, len(self.quad.omega)))

    # =========================
    # Gym API - reset
    # =========================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._already_terminated = False  # 清除终止标记

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.current_substep = 0
        self.episode_reward = 0.0
        self.step_log_rows = []

        if options is not None and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'])


        # 重置仿真组件（会重新初始化控制器）
        self._initialize_simulation()
        log.info(f" reset() called: env_id={id(self)}, resetting to step 0, time={self.time:.3f}s")

        # ==================== 基线PID参数（初始化为中值，并应用到ctrl） ====================
        self.pid_K_prev_vec = (self.pid_safe_min + self.pid_safe_max) / 2.0
        self.init_actions = np.zeros(20)
        self._apply_pid_parameters(self.init_actions)

        if options and options.get('random_init', False):
            noise = np.random.normal(0, 0.1, 3)
            self.quad.pos += noise
            self.quad.state[0:3] = self.quad.pos

    #  重置历史状态缓存（去除误差变化率缓存）
        self.prev_action = self.init_actions
        self.prev_rmse_vel_xy = 0.0
        self.prev_rmse_vel_z = 0.0
        
    # 去除PID变化率历史缓存
        
        # 重置PID变化记录
        self.last_pid_delta_norm = 0.0
        self.last_pid_delta_max = 0.0

        self.step_action = np.zeros((20,), dtype=getattr(self.action_space, "dtype", np.float32))
        self.step_reward = 0.0

        obs = self._get_observation(self.ctrl.sDesCalc)
        # 评估模式：返回初始状态详细信息
        if self.eval_mode:
            info = {
                'time': float(self.time),
                'position': self.quad.pos.copy(),
                'velocity': self.quad.vel.copy(),
                'attitude': self.quad.euler.copy(),
                'angular_velocity': self.quad.omega.copy(),
                'position_des': self.ctrl.sDesCalc[0:3] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'velocity_des': self.ctrl.sDesCalc[3:6] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'attitude_des': _quaternion_to_euler(self.ctrl.sDesCalc[9:13]) if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'angular_velocity_des': self.ctrl.sDesCalc[13:16] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
            }
        else:
            # 训练模式：最小化
            info = {}
        return obs, info

    # =========================
    # 核心：直接式PID参数应用
    # ========================= 
    def _apply_pid_parameters(self, action: np.ndarray) -> dict:
        """
        直接式动作空间 + 可选平滑过渡
        """
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != 20:
            raise ValueError(f"Expected action shape (20,), got {action.shape}")
        
        K_min = self.pid_safe_min
        K_max = self.pid_safe_max
        
        # 验证边界形状
        assert K_min.shape == (20,) and K_max.shape == (20,), \
            f"K_min/K_max shape mismatch: {K_min.shape}, {K_max.shape}"
        
        range_vec = K_max - K_min
        
        # 改进：更明确的初始化检查（应该在 __init__ 中完成）
        if not hasattr(self, 'pid_K_prev_vec') or self.pid_K_prev_vec is None:
            log.warning(" pid_K_prev_vec not initialized in __init__! Using mid-range fallback...")
            self.pid_K_prev_vec = (K_min + K_max) / 2.0
        
        K_prev_vec = self.pid_K_prev_vec
        
        # 核心：直接映射 action ∈ [-1,1] → K_target ∈ [K_min, K_max]
        K_target = K_min + (action + 1.0) / 2.0 * range_vec
        
        # 可选：平滑过渡（EMA 低通滤波）
        if self.enable_smooth_transition:
            # alpha ∈ (0, 1]: alpha=1 无平滑，alpha→0 强平滑
            alpha = float(np.clip(self.alpha, 1e-6, 1.0))
            K_new_vec = (1.0 - alpha) * K_prev_vec + alpha * K_target
        else:
            K_new_vec = K_target
        
        # 最终边界保护（必须！平滑可能导致越界）
        K_new_vec = np.clip(K_new_vec, K_min, K_max)
        
        # 生产模式：仍需记录基本变化（如果奖励函数需要）
        pid_delta = K_new_vec - K_prev_vec
        self.last_pid_delta_norm = float(np.linalg.norm(pid_delta))
        self.last_pid_delta_max = float(np.max(np.abs(pid_delta)))
        self.last_pid_delta_rel_max = float(np.max(np.abs(pid_delta) / np.maximum(range_vec, 1e-6)))
        
        # 应用到控制器
        gains_from_action = self._unpack_vec_to_groups(K_new_vec)
        self.ctrl.update_gains(gains_from_action)
        
        # 更新历史状态（只保留向量格式）
        self.pid_K_prev_vec = K_new_vec        
        return gains_from_action

    # =========================
    # 辅助函数
    # =========================
    def _unpack_vec_to_groups(self, K_vec: np.ndarray) -> dict:
        """将20维向量解包为8组增益"""
        K_vec = np.asarray(K_vec, dtype=np.float32).reshape(20,)

        posP_xy, posP_z = K_vec[0], K_vec[1]
        velP_xy, velP_z = K_vec[2], K_vec[3]
        velI_xy, velI_z = K_vec[4], K_vec[5]
        velD_xy, velD_z = K_vec[6], K_vec[7]
        attP = K_vec[8:11]
        rP = K_vec[11:14]
        rI = K_vec[14:17]
        rD = K_vec[17:20]

        pos_P_gains = (float(posP_xy), float(posP_xy), float(posP_z))
        vel_P_gains = (float(velP_xy), float(velP_xy), float(velP_z))
        vel_I_gains = (float(velI_xy), float(velI_xy), float(velI_z))
        vel_D_gains = (float(velD_xy), float(velD_xy), float(velD_z))
        att_P_gains = tuple(attP.tolist())
        rate_P_gains = tuple(rP.tolist())
        rate_I_gains = tuple(rI.tolist())
        rate_D_gains = tuple(rD.tolist())

        return {
            "pos_P_gains": pos_P_gains,
            "vel_P_gains": vel_P_gains,
            "vel_I_gains": vel_I_gains,
            "vel_D_gains": vel_D_gains,
            "att_P_gains": att_P_gains,
            "rate_P_gains": rate_P_gains,
            "rate_I_gains": rate_I_gains,
            "rate_D_gains": rate_D_gains,
        }
    
    def _error_gated_action_adaptation(self, action: np.ndarray) -> np.ndarray:
        """
        分层+分通道误差门控机制（12 个独立通道）
        
        根据当前各通道误差大小，对 action 进行冻结/限幅/放开处理：
        - 小误差（< thresh1）：冻结该通道 action
        - 中误差（thresh1 ~ thresh2）：限幅调整
        - 大误差（> thresh2）：自由探索
        
        Args:
            action: 原始 action（20维）
        
        Returns:
            处理后的 action（20维）
        """
        # ============================================================
        # 提取当前观测中的误差信号（已归一化）
        # ============================================================
        obs_cur = self._get_observation(self.ctrl.sDesCalc)
        
        pos_err = obs_cur[24:27]    # [x, y, z]
        vel_err = obs_cur[27:30]    # [x, y, z]
        att_err = obs_cur[30:33]    # [roll, pitch, yaw]
        omega_err = obs_cur[33:36]  # [p, q, r]

        # ============================================================
        # 阈值配置（建议后续移到 __init__）
        # ============================================================
        thresh1 = 0.03      # 小误差冻结阈值
        thresh2 = 0.3      # 大误差放开阈值
        limit_delta = 0.2  # 限幅范围

        # ============================================================
        # 通用门控函数（闭包，可访问外部变量）
        # ============================================================
        def _gate_indices(err_val, idxs):
            """根据误差值对指定 action 索引进行门控"""
            if err_val < thresh1:
                # 冻结：保持上一步 action
                action[idxs] = self.prev_action[idxs]
            elif thresh1 <= err_val < thresh2:
                # 限幅：限制变化幅度
                delta = np.clip(action[idxs] - self.prev_action[idxs], 
                            -limit_delta, limit_delta)
                action[idxs] = self.prev_action[idxs] + delta
            else:
                # 放开：不干预（保持原 action）
                pass

        # ============================================================
        # 1. 位置环（2 通道：xy, z）["xyz_pos", "xy_vel_z_pos", "xyz_vel"] 
        # ============================================================
        if self.control_type == "xyz_vel":
            pos_err_xy = 0
            pos_err_z = 0
        elif self.control_type == "xyz_pos":
            pos_err_xy = max(abs(pos_err[0]), abs(pos_err[1]))
            pos_err_z = abs(pos_err[2])
        elif self.control_type == "xy_vel_z_pos":
            pos_err_xy = 0
            pos_err_z = abs(pos_err[2])
        
        _gate_indices(pos_err_xy, [0])  # pos_P_xy
        _gate_indices(pos_err_z, [1])   # pos_P_z

        # ============================================================
        # 2. 速度环（2 通道：xy, z）   
        # ============================================================
        vel_err_xy = max(abs(vel_err[0]), abs(vel_err[1]))
        vel_err_z = abs(vel_err[2])
        
        _gate_indices(vel_err_xy, [2, 4, 6])  # vel_P/I/D_xy
        _gate_indices(vel_err_z, [3, 5, 7])   # vel_P/I/D_z

        # ============================================================
        # 3. 姿态环（3 通道：roll, pitch, yaw）
        # ============================================================
        att_err_roll = abs(att_err[0])
        att_err_pitch = abs(att_err[1])
        att_err_yaw = abs(att_err[2])
        
        _gate_indices(att_err_roll, [8])    # att_P_roll
        _gate_indices(att_err_pitch, [9])   # att_P_pitch
        _gate_indices(att_err_yaw, [10])    # att_P_yaw

        # ============================================================
        # 4. 角速度环（3 通道：p, q, r）
        # ============================================================
        omega_err_roll = abs(omega_err[0])   # p (roll rate)
        omega_err_pitch = abs(omega_err[1])  # q (pitch rate)
        omega_err_yaw = abs(omega_err[2])    # r (yaw rate)
        
        _gate_indices(omega_err_roll, [11, 14, 17])    # rate_P/I/D_p
        _gate_indices(omega_err_pitch, [12, 15, 18])   # rate_P/I/D_q
        _gate_indices(omega_err_yaw, [13, 16, 19])     # rate_P/I/D_r

        # ============================================================
        # 返回处理后的 action
        # ============================================================
        return action

    def _log_reward_to_csv(self, reward_dict):
        """将奖励详情记录到CSV文件"""
        # 定义日志文件路径
        log_dir = "logs/reward_debug"
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用时间戳区分不同训练会话
        if not hasattr(self, 'csv_log_file'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_log_file = os.path.join(log_dir, f"reward_log_{timestamp}.csv")
            self.csv_initialized = False
        
        # 准备数据行
        row_data = {
            'step': getattr(self, 'step_count', 0),
            'episode': getattr(self, 'episode_count', 0),
            **reward_dict  # 展开所有奖励组件
        }
        
        # 写入CSV
        file_exists = os.path.isfile(self.csv_log_file)
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            # 首次写入添加表头
            if not file_exists or not self.csv_initialized:
                writer.writeheader()
                self.csv_initialized = True
            
            writer.writerow(row_data)

    # =========================
    # step() 函数（主循环）
    # =========================
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个环境步骤
        """
        # ============================================================
        # 阶段1: 误差门控动作调整
        # ============================================================
        action = self._error_gated_action_adaptation(action)

        # 保存历史动作（用于下一步比较）
        self.prev_action = np.asarray(action, dtype=np.float32).ravel().copy()

        # ============================================================
        # 阶段2: 应用 PID 参数
        # ============================================================
        self.step_action = self._apply_pid_parameters(action)
        
        substeps_count = self.substeps_per_rl
        
        step_samples: List[Dict[str, float]] = []

        # 如果是评估模式，收集子步详细数据
        substep_records = [] if self.eval_mode else None

        #  使用当前状态初始化 prev_cache
        prev_cache = {
            "vel": self.quad.vel.copy(),
            "omega": self.quad.omega.copy(),
            "w_cmd": self.ctrl.w_cmd.copy() if hasattr(self.ctrl, "w_cmd") and self.ctrl.w_cmd is not None else None
        }
        
        terminated = False
        truncated = False
        
        for substep_i in range(substeps_count):
            # 2a) 生成期望轨迹
            sDes = self.traj.desiredState(self.time, self.dt, self.quad)
            
            # 2b) 控制器计算电机指令
            self.ctrl.controller(self.traj, self.quad, sDes, self.dt)
            
            # 2c) 四旋翼动力学更新
            self.quad.update(self.time, self.dt, self.ctrl.w_cmd, self.wind)

            # 2d) 记录历史数据
            substep_data = self._record_episode_step()  # 现在可能返回数据
            if substep_data is not None:
                substep_records.append(substep_data)

            # 2e) 时间推进
            self.time += self.dt
            self.current_substep += 1

            # 2f) 提取误差
            sDesclc = self.ctrl.sDesCalc
            m = self._extract_metrics(sDesclc)

            # 2g) 收集样本（用于奖励计算）
            sample = self._collect_step_sample(m, prev_cache)
            step_samples.append(sample)

            # 2h) 边界检查（姿态超过边界立即终止）
            if self._is_out_of_bounds():
                # 标记最后一个样本为异常
                terminated = True

                log.warning(
                    f" step() called: Out of bounds at substep {substep_i+1}/{substeps_count} "
                    f"(step {self.current_step}, time {self.time:.3f}s, env_id={id(self)})"
                )

                # 填充缺失子步（保持样本数一致性，避免奖励计算异常）
                missing_substeps = substeps_count - len(step_samples)
                if missing_substeps > 0:
                    # 用最后一个有效样本填充（标记为padding）
                    last_sample = step_samples[-1].copy()
                    last_sample["is_padding"] = True
                    
                    for _ in range(missing_substeps):
                        step_samples.append(last_sample.copy())
                    
                    log.info(
                        f" Padded {missing_substeps} missing substeps to ensure consistent reward calculation"
                    )
                
                break

        # 越界后立即返回，不执行后续逻辑
        if terminated:
            self.current_step += 1  # 在返回前更新计数器
            self._already_terminated = True  # 标记已终止

            # 施加失败惩罚
            reward_before_scale = self.failure_penalty
            reward = reward_before_scale * self.reward_scale
            self.episode_reward += reward
            
            # 获取最终观测（用于value function估计）
            obs = self._get_observation(self.ctrl.sDesCalc)
            
            # 返回终止信号
            info = {'termination_reason': 'out_of_bounds', 'step': self.current_step, 'env_id': id(self)}

            log.warning(
                f"🚨 Episode terminated at step {self.current_step}, "
                f"penalty: {self.failure_penalty} (scaled: {reward:.2f})"
            )
            
            return obs, reward, terminated, False, info  # 立即返回

        # ============================================================
        # 阶段3: 奖励计算
        # ============================================================
        
        # 正常情况：计算完整奖励
        # 3.1 基础奖励（基于RMSE误差）
        segment_reward, reward_for_debug, metrics = self._compute_segment_reward_step(step_samples)
                    
        # 3.3 速度跟踪成功奖励（正向激励）
        vel_tracking_bonus = 0.0
        if metrics:
            rmse_vel_xy = metrics.get("rmse_vel_xy", 999.0)
            rmse_vel_z = metrics.get("rmse_vel_z", 999.0)
            
            # 阈值奖励
            if rmse_vel_xy < 0.5 and rmse_vel_z < 0.5:
                vel_tracking_bonus += 2.0
                if rmse_vel_xy < 0.2 and rmse_vel_z < 0.2:
                    vel_tracking_bonus += 5.0
            
            #  修复：改进奖励（误差下降），添加上限约束防止震荡
            vel_improvement_xy = self.prev_rmse_vel_xy - rmse_vel_xy
            vel_improvement_z = self.prev_rmse_vel_z - rmse_vel_z
            
            # 只奖励合理范围内的改进
            if vel_improvement_xy > 0 and vel_improvement_xy < 0.5:  # 单步改进上限0.5m/s
                vel_tracking_bonus += min(1.0 * vel_improvement_xy, 2.0)  # 总奖励上限2.0
            if vel_improvement_z > 0 and vel_improvement_z < 0.5:
                vel_tracking_bonus += min(1.0 * vel_improvement_z, 2.0)
            
            # 更新历史
            self.prev_rmse_vel_xy = rmse_vel_xy
            self.prev_rmse_vel_z = rmse_vel_z

        # 记录调试信息
        reward_for_debug['segment_reward'] = segment_reward
        reward_for_debug['vel_tracking_bonus'] = vel_tracking_bonus  

        # 3.4 合成总奖励（缩放前）
        reward_before_scale = segment_reward + vel_tracking_bonus
        reward_for_debug['reward_total_before_scale'] = reward_before_scale

        # 调试日志输出                       
        # log.warning(f"reward_pos_xy={reward_for_debug['reward_pos_xy']}\n")
        # log.warning(f"reward_pos_z={reward_for_debug['reward_pos_z']}\n")
        # log.warning(f"reward_vel_xy={reward_for_debug['reward_vel_xy']}\n")
        # log.warning(f"reward_vel_z={reward_for_debug['reward_vel_z']}\n")
        # log.warning(f"reward_att={reward_for_debug['reward_att']}\n")
        # log.warning(f"reward_omega={reward_for_debug['reward_omega']}\n")
        # log.warning(f"reward_u={reward_for_debug['reward_u']}\n")
        # log.warning(f"reward_dv={reward_for_debug['reward_dv']}\n")
        # log.warning(f"reward_domega={reward_for_debug['reward_domega']}\n")
        # log.warning(f"reward_du={reward_for_debug['reward_du']}\n")
        # log.warning(f"reward_sat={reward_for_debug['reward_sat']}\n")
        # log.warning(f"reward_over={reward_for_debug['reward_over']}\n")
        # log.warning(f"reward_oscillation_penalty={reward_for_debug['reward_oscillation_penalty']}\n")
        # log.warning(f"reward_vel_bonus={reward_for_debug['reward_vel_bonus']}\n")
        # log.warning(f"segment_reward={segment_reward}\n")
        # log.warning(f"vel_tracking_bonus={vel_tracking_bonus}\n")
        # log.warning(f"reward_before_scale={reward_before_scale}\n")

        # 记录到CSV文件
        if self.eval_mode:
            self._log_reward_to_csv(reward_for_debug)
        # ============================================================
        # 阶段4: 终止条件判断
        # ============================================================
        # 统一奖励缩放
        reward = reward_before_scale * self.reward_scale
        self.step_reward = reward
        self.episode_reward += reward
        self.current_step += 1

        if terminated or truncated:
            max_substeps_total = self.episode_t_all.shape[0]
            self._episode_len = min(self.current_substep, max_substeps_total)

        # 超时：提前返回，并可选将截断视为done（兼容旧训练循环仅检查done）
        if self.current_step >= self.max_episode_steps:
            truncated = True
            if getattr(self, "compat_truncated_as_done", False):
                terminated = True  # 兼容：同时置为True，触发外部reset
            self._already_terminated = True
            # 记录episode长度
            max_substeps_total = self.episode_t_all.shape[0]
            self._episode_len = min(self.current_substep, max_substeps_total)
            # 终止信息
            info = {
                'termination_reason': 'time_limit',
                'step': self.current_step,
                'TimeLimit.truncated': True,
                'env_id': id(self),
            }
            log.info(f"⏱️ Episode time limit reached at step {self.current_step} (env_id={id(self)})")
            obs = self._get_observation(self.ctrl.sDesCalc)
            return obs, reward, terminated, truncated, info
        
        # 正常路径：返回本步结果
        obs = self._get_observation(self.ctrl.sDesCalc)
        info = {}
        return obs, reward, False, False, info

    def _record_episode_step(self) -> Optional[Dict]:
        """
        记录当前子步的历史数据
        """
        idx = self.current_substep
        
        # 检查索引越界
        if idx >= self.episode_t_all.shape[0]:
            log.error(f" substep索引 {idx} 超出预分配数组大小")
            return None
        
        # 记录数据到内部数组（训练和评估都需要）
        self.episode_t_all[idx] = self.time
        self.episode_pos_all[idx] = self.quad.pos
        self.episode_vel_all[idx] = self.quad.vel
        self.episode_omega_all[idx] = self.quad.omega
        self.episode_euler_all[idx] = self.quad.euler
        
        # 记录期望状态
        if hasattr(self.ctrl, 'sDesCalc') and self.ctrl.sDesCalc is not None:
            sDesCalc = self.ctrl.sDesCalc
            self.episode_pos_des_all[idx] = sDesCalc[0:3]
            self.episode_vel_des_all[idx] = sDesCalc[3:6]
            self.episode_euler_des_all[idx] = _quaternion_to_euler(sDesCalc[9:13])
            self.episode_omega_des_all[idx] = sDesCalc[13:16]

        # 评估模式：返回当前步的数据副本
        if self.eval_mode:
            return {
                'time': float(self.time),
                # 实际状态
                'position': self.quad.pos.copy(),
                'velocity': self.quad.vel.copy(),
                'attitude': self.quad.euler.copy(),  # [roll, pitch, yaw]
                'omega': self.quad.omega.copy(),
                'quaternion': self.quad.quat.copy(),
                
                # 期望状态
                'position_des': sDesCalc[0:3] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'velocity_des': sDesCalc[3:6] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'attitude_des': _quaternion_to_euler(sDesCalc[9:13]) if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'omega_des': sDesCalc[13:16] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
            }
        else:
            # 训练模式：不返回（节省开销）
            return None

    # =========================
    # 奖励设计
    # =========================
    def _extract_metrics(self, sDesCalc: np.ndarray) -> Dict[str, Any]:
        """从 sDesCalc 及当前状态提取误差"""
        pos_ref = sDesCalc[0:3]
        vel_ref = sDesCalc[3:6]
        eul_ref = _quaternion_to_euler(sDesCalc[9:13])
        omega_ref = sDesCalc[13:16]

        pos = self.quad.pos
        vel = self.quad.vel
        euler = self.quad.euler
        omega = self.quad.omega

        def angle_wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        pos_err_vec = pos_ref - pos
        pos_err_x = pos_err_vec[0]
        pos_err_y = pos_err_vec[1]
        pos_err_z = pos_err_vec[2]
        pos_err_xy = float(np.linalg.norm(pos_err_vec[:2]))

        vel_err_vec = vel_ref - vel
        vel_err_x = vel_err_vec[0]
        vel_err_y = vel_err_vec[1]
        vel_err_z = vel_err_vec[2]
        vel_err_xy = float(np.linalg.norm(vel_err_vec[:2]))

        att_err_vec = eul_ref - euler
        att_err_vec = np.array([angle_wrap(att_err_vec[0]),
                                angle_wrap(att_err_vec[1]),
                                angle_wrap(att_err_vec[2])], dtype=float)
        att_err_roll = float(att_err_vec[0])
        att_err_pitch = float(att_err_vec[1])
        att_err_yaw = float(att_err_vec[2])
        att_norm = float(np.linalg.norm(att_err_vec))

        omega_err_vec = omega_ref - omega
        omega_err_p = float(omega_err_vec[0])
        omega_err_q = float(omega_err_vec[1])
        omega_err_r = float(omega_err_vec[2])
        omega_norm = float(np.linalg.norm(omega_err_vec))

        return {
            "pos_err_x": pos_err_x,
            "pos_err_y": pos_err_y,
            "pos_err_z": pos_err_z,
            "pos_err_xy": pos_err_xy,

            "vel_err_x": vel_err_x,
            "vel_err_y": vel_err_y,
            "vel_err_z": vel_err_z,
            "vel_err_xy": vel_err_xy,

            "att_err_roll": abs(att_err_roll),
            "att_err_pitch": abs(att_err_pitch),
            "att_err_yaw": att_err_yaw,
            "att_norm": att_norm,

            "omega_err_p": omega_err_p,
            "omega_err_q": omega_err_q,
            "omega_err_r": omega_err_r,
            "omega_norm": omega_norm,

        }

    def _collect_step_sample(self, m: Dict[str, Any], prev_cache: Dict[str, Any]) -> Dict[str, float]:
        """
         在本步子步中收集一次原子指标
            1. 计算即时奖励（用于子步级别分析）
            2. 标记异常状态（越界、高误差等）
            3. 记录时间戳和子步索引
        """
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            control_effort = float(np.sum(np.square(self.ctrl.w_cmd)))
        else:
            control_effort = 0.0

        dv = 0.0        # 速度变化率
        domega = 0.0    # 角速度变化率
        du = 0.0        # 控制输入变化率

        if prev_cache.get("vel") is not None:
            dv = float(np.linalg.norm(self.quad.vel - prev_cache["vel"]))
        
        if prev_cache.get("omega") is not None:
            domega = float(np.linalg.norm(self.quad.omega - prev_cache["omega"]))
        
        if hasattr(self.ctrl, "w_cmd"):
            if prev_cache.get("w_cmd") is not None and self.ctrl.w_cmd is not None:
                du = float(np.linalg.norm(self.ctrl.w_cmd - prev_cache["w_cmd"]))

        if self.limits.get("normalize_rate_by_dt", True) and self.dt > 0:
            inv_dt = 1.0 / self.dt
            dv *= inv_dt
            domega *= inv_dt
            du *= inv_dt

        prev_cache["vel"] = np.copy(self.quad.vel)
        prev_cache["omega"] = np.copy(self.quad.omega)
        if hasattr(self.ctrl, "w_cmd") and self.ctrl.w_cmd is not None:
            prev_cache["w_cmd"] = np.copy(self.ctrl.w_cmd)

        saturation_hit = 0
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            wmin, wmax = self.limits["w_cmd_min"], self.limits["w_cmd_max"]
            if np.any(self.ctrl.w_cmd < wmin) or np.any(self.ctrl.w_cmd > wmax):
                saturation_hit = 1

        # 超调检测，应包括三轴速度，三轴姿态，三轴角速度等
        overshoot_hit = 0
        overshoot_hit = self._detect_overshoot(m)

        entry = {
            # 原有指标
            "pos_x2": m["pos_err_x"]**2,
            "pos_y2": m["pos_err_y"]**2,
            "pos_z2": m["pos_err_z"]**2,
            "pos_xy2": m["pos_err_xy"]**2,

            "vel_x2": m["vel_err_x"]**2,
            "vel_y2": m["vel_err_y"]**2,
            "vel_z2": m["vel_err_z"]**2,
            "vel_xy2": m["vel_err_xy"]**2,

            "att2": m["att_norm"]**2,
            "omega2": m["omega_norm"]**2,

            "u2": control_effort,
            "dv2": dv**2,
            "domega2": domega**2,
            "du2": du**2,

            "sat_hit": saturation_hit,
            "over_hit": overshoot_hit,
        }
        return entry

    def _detect_overshoot(self, error):
        # 初始化超调检测结果
        overshoot_results = {
            "vel": [0, 0, 0],     # 三轴速度超调结果
            "att": [0, 0, 0],     # 三轴姿态超调结果
            "ang_vel": [0, 0, 0]  # 三轴角速度超调结果
        }
        
        # 定义对应的容忍度 
        vel_tol = self.limits["overshoot_vel_tol"]
        att_tol = self.limits["overshoot_att_tol"]
        ang_vel_tol = self.limits["overshoot_ang_vel_tol"]

        # 把需要用到的误差根据通道分类
        pos_err = [error["pos_err_x"], error["pos_err_y"], error["pos_err_z"]]
        vel_err = [error["vel_err_x"], error["vel_err_y"], error["vel_err_z"]]
        att_err = [error["att_err_roll"], error["att_err_pitch"], error["att_err_yaw"]]
        ang_vel_err = [error["omega_err_p"], error["omega_err_q"], error["omega_err_r"]]

        # 记录是否发生超调
        overthrow_detected = False

        # 检测超调：三轴速度
        for i in range(3):  # 0: X, 1: Y, 2: Z
            current_vel_err = vel_err[i]  # 从分类后的 vel_err 提取误差
            current_vz = float(self.quad.vel[i])
            if current_vel_err * np.sign(current_vz) < 0 and abs(current_vel_err) > vel_tol:
                overshoot_results['vel'][i] = 1
                overthrow_detected = True  # 检测到超调

        # 检测超调：三轴姿态
        for i in range(3):  # 0: Roll, 1: Pitch, 2: Yaw
            current_att_err = att_err[i]  # 从分类后的 att_err 提取误差
            current_ang_att = float(self.quad.euler[i]) #self.quad.euler : yaw,pitch,roll
            if current_att_err * np.sign(current_ang_att) < 0 and abs(current_att_err) > att_tol:
                overshoot_results['att'][i] = 1
                overthrow_detected = True  # 检测到超调

        # 检测超调：三轴角速度
        for i in range(3):  # 0: Roll rate, 1: Pitch rate, 2: Yaw rate
            current_ang_vel_err = ang_vel_err[i]  # 从分类后的 ang_vel_err 提取误差
            current_vz = float(self.quad.omega[i])
            if current_ang_vel_err * np.sign(current_vz) < 0 and abs(current_ang_vel_err) > ang_vel_tol:
                overshoot_results['ang_vel'][i] = 1
                overthrow_detected = True  # 检测到超调

        # # 仅在发生超调时记录
        # if overthrow_detected:
        #     # 这里可以添加记录超调的逻辑，具体实现可以根据需要进行修改
        #     log.warning("超调发生！")  # 示例操作，您可以替换为日志记录或其他动作

        # 返回结果可以选择仅返回是否发生超调或详细结果
        return 1 if overthrow_detected else 0

    def _compute_segment_reward_step(self, samples: List[Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
        """
        计算当前段的奖励及指标（基于子步样本列表）
        """
        valid_samples = samples

        keys = ["pos_xy2", "pos_z2", 
                "vel_xy2", "vel_z2",
                "att2", 
                "omega2",
                "u2", "dv2", "domega2", "du2",
                "sat_hit", "over_hit"]
        S = {k: np.array([e[k] for e in valid_samples], dtype=float) for k in keys}


        # 标准RMSE
        rmse_pos_xy = float(np.sqrt(np.mean(S["pos_xy2"])))
        rmse_pos_z = float(np.sqrt(np.mean(S["pos_z2"])))
        rmse_vel_xy = float(np.sqrt(np.mean(S["vel_xy2"])))
        rmse_vel_z = float(np.sqrt(np.mean(S["vel_z2"])))
        rmse_att = float(np.sqrt(np.mean(S["att2"])))
        rms_omega = float(np.sqrt(np.mean(S["omega2"])))

        mean_control_effort = float(np.mean(S["u2"]))
        rms_dv = float(np.sqrt(np.mean(S["dv2"])))
        rms_domega = float(np.sqrt(np.mean(S["domega2"])))
        rms_du = float(np.sqrt(np.mean(S["du2"])))

        saturation_hits = float(np.sum(S["sat_hit"]))
        saturation_hits_norm = saturation_hits / 100.0

        overshoot_hits = float(np.sum(S["over_hit"]))
        overshoot_hits_norm = overshoot_hits / 100.0

        #  震荡模式检测（改进算法）
        oscillation_penalty = 0.0
        if self.enable_oscillation_penalty and len(valid_samples) >= 3:
            vel_err_xy_list = [s.get("vel_err_xy", 0.0) for s in valid_samples]
            vel_err_z_list = [s.get("vel_err_z", 0.0) for s in valid_samples]
            
            # 检测符号变化次数（震荡指标）
            vel_xy_sign_changes = sum(
                1 for i in range(1, len(vel_err_xy_list))
                if vel_err_xy_list[i] * vel_err_xy_list[i-1] < 0
            )
            vel_z_sign_changes = sum(
                1 for i in range(1, len(vel_err_z_list))
                if vel_err_z_list[i] * vel_err_z_list[i-1] < 0
            )
            
            # 检测标准差（震荡幅度）
            vel_err_xy_std = float(np.std(vel_err_xy_list))
            vel_err_z_std = float(np.std(vel_err_z_list))
            
            # 综合评估震荡程度
            oscillation_ratio = (vel_xy_sign_changes + vel_z_sign_changes) / (2 * len(valid_samples))
            oscillation_amplitude = (vel_err_xy_std + vel_err_z_std) / 2.0
            
            # 如果震荡频繁（超过50%子步发生符号变化）且幅度较大，施加惩罚
            if oscillation_ratio > 0.5 and oscillation_amplitude > 0.3:
                oscillation_penalty = 0.3 * oscillation_ratio + 0.2 * min(oscillation_amplitude, 2.0)

        #  改进：速度跟踪分档奖励
        vel_error_total = rmse_vel_xy + rmse_vel_z  # 总误差
        
        if vel_error_total < self.vel_tracking_excellent_threshold:  # <1m/s
            # 精确跟踪：大额正奖励
            vel_bonus = self.vel_reward_excellent_bonus * (1.0 - vel_error_total / self.vel_tracking_excellent_threshold)
        elif vel_error_total > self.vel_tracking_poor_threshold:    # >3m/s
            # 极差跟踪：额外惩罚
            vel_bonus = -self.vel_reward_poor_penalty * (vel_error_total / self.vel_tracking_poor_threshold - 1.0)
        else:
            # 中间状态：无额外奖励
            vel_bonus = 0.0

        #  奖励函数（优化后的权重）
        sw = self.segment_weights

        reward_pos_xy = sw["W_pos_xy"] * rmse_pos_xy
        reward_pos_z = sw["W_pos_z"] * rmse_pos_z
        reward_vel_xy = sw["W_vel_xy"] * rmse_vel_xy
        reward_vel_z = sw["W_vel_z"] * rmse_vel_z
        reward_att = sw["W_att"] * rmse_att              
        reward_omega = sw["W_omega"] * rms_omega
        reward_u = sw["W_u"] * mean_control_effort
        reward_dv = sw["W_dv"] * rms_dv
        reward_domega = sw["W_domega"] * rms_domega
        reward_du = sw["W_du"] * rms_du
        reward_sat = sw["W_sat"] * saturation_hits_norm
        reward_over = sw["W_over"] * overshoot_hits_norm
        reward_oscillation_penalty = oscillation_penalty
        reward_vel_bonus = vel_bonus

        segment_reward = - (reward_pos_xy +
                            reward_pos_z +
                            reward_vel_xy +
                            reward_vel_z +
                            reward_att +
                            reward_omega +
                            reward_u +
                            reward_dv +
                            reward_domega +
                            reward_du +
                            reward_sat +
                            reward_over +
                            reward_oscillation_penalty) + reward_vel_bonus

        reward_for_debug = {
            "reward_pos_xy": reward_pos_xy,
            "reward_pos_z": reward_pos_z,
            "reward_vel_xy": reward_vel_xy,
            "reward_vel_z": reward_vel_z,
            "reward_att": reward_att,
            "reward_omega": reward_omega,
            "reward_u": reward_u,
            "reward_dv": reward_dv,
            "reward_domega": reward_domega,
            "reward_du": reward_du,
            "reward_sat": reward_sat,
            "reward_over": reward_over,
            "reward_oscillation_penalty": reward_oscillation_penalty,
            "reward_vel_bonus": reward_vel_bonus,
        }

        metrics = {
            'segment_duration': float(len(valid_samples) * self.dt),
            'rmse_pos_xy': rmse_pos_xy,
            'rmse_pos_z': rmse_pos_z,
            'rmse_vel_xy': rmse_vel_xy,
            'rmse_vel_z': rmse_vel_z,
            'rmse_att': rmse_att,
            'rms_omega': rms_omega,

            'mean_control_effort': mean_control_effort,
            'rms_dv': rms_dv,
            'rms_domega': rms_domega,
            'rms_du': rms_du,

            'saturation_hits': saturation_hits,
            'overshoot_hits': overshoot_hits,
            'segment_reward': float(segment_reward),
            'samples': int(len(valid_samples)),
            
            #  诊断信息
            'total_samples': int(len(samples)),
            'oscillation_penalty': float(oscillation_penalty),
        }

        return float(segment_reward), reward_for_debug, metrics

    # =========================
    #  观测函数
    # =========================
    def _get_observation(self, sDesCalc: np.ndarray) -> np.ndarray:
        """
        构建观测向量（56维）
        """
        # 参考状态
        pos_ref = sDesCalc[0:3]
        vel_ref = sDesCalc[3:6]
        eul_ref = _quaternion_to_euler(sDesCalc[9:13])
        omega_ref = sDesCalc[13:16]

        # 反馈状态
        pos_fb = self.quad.pos
        vel_fb = self.quad.vel
        eul_fb = self.quad.euler
        omega_fb = self.quad.omega

        # 误差信号（核心特征）
        pos_err = pos_ref - pos_fb
        vel_err = vel_ref - vel_fb
        eul_err = eul_ref - eul_fb
        eul_err = np.arctan2(np.sin(eul_err), np.cos(eul_err))  # 将欧拉角误差包裹到 [-pi, pi]
        omega_err = omega_ref - omega_fb
        log.warning(f"Obs Extraction: pos_err={pos_err}, vel_err={vel_err}, eul_err={eul_err}, omega_err={omega_err}")
        
        # 误差归一化
        pos_err_norm = np.clip(pos_err / 10.0, -1, 1)      # 位置误差 /10m
        vel_err_norm = np.clip(vel_err / 15.0, -1, 1)      # 速度误差 /15m/s
        eul_err_norm = np.clip(eul_err / np.pi, -1, 1)     # 角度误差 /π
        omega_err_norm = np.clip(omega_err / 2*np.pi, -1, 1)  # 角速度误差 /2π
        
        # 去除误差变化率相关特征与缓存        
        if hasattr(self, 'pid_K_prev_vec') and self.pid_K_prev_vec is not None:
            pid_normalized = 2.0 * (self.pid_K_prev_vec - self.pid_safe_min) / (self.pid_safe_max - self.pid_safe_min + 1e-8) - 1.0
            pid_normalized = np.clip(pid_normalized, -1.0, 1.0)
        else:
            pid_normalized = np.zeros(20, dtype=np.float32)
                
        # 参考状态和反馈状态归一化
        pos_ref_norm = np.clip(pos_ref / 10.0, -1, 1)
        vel_ref_norm = np.clip(vel_ref / 15.0, -1, 1)
        eul_ref_norm = np.clip(eul_ref / np.pi, -1, 1)
        omega_ref_norm = np.clip(omega_ref / 2*np.pi, -1, 1)
        
        pos_fb_norm = np.clip(pos_fb / 10.0, -1, 1)
        vel_fb_norm = np.clip(vel_fb / 15.0, -1, 1)
        eul_fb_norm = np.clip(eul_fb / np.pi, -1, 1)
        omega_fb_norm = np.clip(omega_fb / 2*np.pi, -1, 1)
        
        # 拼接观测向量（56维，全部已归一化）
        obs = np.concatenate([
            # 参考状态 (12)
            pos_ref_norm, vel_ref_norm, eul_ref_norm, omega_ref_norm,
            # 反馈状态 (12)
            pos_fb_norm, vel_fb_norm, eul_fb_norm, omega_fb_norm,
            # 误差信号 (12)
            pos_err_norm, vel_err_norm, eul_err_norm, omega_err_norm,
            # PID参数归一化 (20)
            pid_normalized
        ])  

        return obs.astype(np.float32)
    
    # =========================
    # 终止与信息
    # =========================
    def _is_out_of_bounds(self) -> bool:
        """ 改进：使用动态姿态阈值[90~60]，根据epoch线性调整"""
        if (np.abs(self.quad.euler[0]) > self.attitude_bound_current or 
            np.abs(self.quad.euler[1]) > self.attitude_bound_current):
            return True
        return False

    def render(self, mode='human'):
        """渲染环境（在线可视化，显示全程）
        - 上排：位置、速度、姿态（度）连续曲线；虚线为参考
        - 下排：角速度连续曲线；当前 PID 20 维增益柱状图；PID 全程历史热力图
        - 独立窗口：4x5 子图绘制 20 个 PID 增益的全程波形
        """
        if mode not in ('human', None):
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            log.warning(f"Matplotlib 不可用，跳过渲染: {e}")
            return

        # 懒加载初始化（主渲染窗口）
        if not hasattr(self, "_render_initialized") or not getattr(self, "_render_initialized", False):
            plt.ion()
            self._render_fig = plt.figure(figsize=(16, 9))
            self._render_fig.suptitle(f"PX4 Env Render - ctrl: {self.control_type}, traj: {self.trajectory_type}", fontsize=12)

            gs = self._render_fig.add_gridspec(2, 3, height_ratios=[2.0, 1.4], wspace=0.25, hspace=0.35)
            self._ax_pos = self._render_fig.add_subplot(gs[0, 0])
            self._ax_vel = self._render_fig.add_subplot(gs[0, 1])
            self._ax_att = self._render_fig.add_subplot(gs[0, 2])
            self._ax_omega = self._render_fig.add_subplot(gs[1, 0])
            self._ax_act_bar = self._render_fig.add_subplot(gs[1, 1])
            self._ax_act_hist = self._render_fig.add_subplot(gs[1, 2])

            # 位置曲线（实线：实际，虚线：参考）
            (self._ln_px,) = self._ax_pos.plot([], [], 'b-', lw=1.5, label='X')
            (self._ln_py,) = self._ax_pos.plot([], [], 'g-', lw=1.5, label='Y')
            (self._ln_pz,) = self._ax_pos.plot([], [], 'r-', lw=1.5, label='Z')
            (self._ln_px_ref,) = self._ax_pos.plot([], [], 'b--', lw=1.0, alpha=0.9, label='X_ref')
            (self._ln_py_ref,) = self._ax_pos.plot([], [], 'g--', lw=1.0, alpha=0.9, label='Y_ref')
            (self._ln_pz_ref,) = self._ax_pos.plot([], [], 'r--', lw=1.0, alpha=0.9, label='Z_ref')
            self._ax_pos.set_title("Position (m)")
            self._ax_pos.set_xlabel("t (s)")
            self._ax_pos.legend(loc='upper right', fontsize=8)
            self._ax_pos.grid(alpha=0.3)

            # 速度曲线
            (self._ln_vx,) = self._ax_vel.plot([], [], 'b-', lw=1.5, label='Vx')
            (self._ln_vy,) = self._ax_vel.plot([], [], 'g-', lw=1.5, label='Vy')
            (self._ln_vz,) = self._ax_vel.plot([], [], 'r-', lw=1.5, label='Vz')
            (self._ln_vx_ref,) = self._ax_vel.plot([], [], 'b--', lw=1.0, alpha=0.9, label='Vx_ref')
            (self._ln_vy_ref,) = self._ax_vel.plot([], [], 'g--', lw=1.0, alpha=0.9, label='Vy_ref')
            (self._ln_vz_ref,) = self._ax_vel.plot([], [], 'r--', lw=1.0, alpha=0.9, label='Vz_ref')
            self._ax_vel.set_title("Velocity (m/s)")
            self._ax_vel.set_xlabel("t (s)")
            self._ax_vel.legend(loc='upper right', fontsize=8)
            self._ax_vel.grid(alpha=0.3)

            # 姿态曲线（度）
            (self._ln_roll,) = self._ax_att.plot([], [], 'b-', lw=1.5, label='Roll')
            (self._ln_pitch,) = self._ax_att.plot([], [], 'g-', lw=1.5, label='Pitch')
            (self._ln_yaw,) = self._ax_att.plot([], [], 'r-', lw=1.5, label='Yaw')
            (self._ln_roll_ref,) = self._ax_att.plot([], [], 'b--', lw=1.0, alpha=0.9, label='Roll_ref')
            (self._ln_pitch_ref,) = self._ax_att.plot([], [], 'g--', lw=1.0, alpha=0.9, label='Pitch_ref')
            (self._ln_yaw_ref,) = self._ax_att.plot([], [], 'r--', lw=1.0, alpha=0.9, label='Yaw_ref')
            self._ax_att.set_title("Attitude (deg)")
            self._ax_att.set_xlabel("t (s)")
            self._ax_att.legend(loc='upper right', fontsize=8)
            self._ax_att.grid(alpha=0.3)

            # 角速度曲线
            (self._ln_p,) = self._ax_omega.plot([], [], 'b-', lw=1.5, label='p')
            (self._ln_q,) = self._ax_omega.plot([], [], 'g-', lw=1.5, label='q')
            (self._ln_r,) = self._ax_omega.plot([], [], 'r-', lw=1.5, label='r')
            (self._ln_p_ref,) = self._ax_omega.plot([], [], 'b--', lw=1.0, alpha=0.9, label='p_ref')
            (self._ln_q_ref,) = self._ax_omega.plot([], [], 'g--', lw=1.0, alpha=0.9, label='q_ref')
            (self._ln_r_ref,) = self._ax_omega.plot([], [], 'r--', lw=1.0, alpha=0.9, label='r_ref')
            self._ax_omega.set_title("Angular Velocity (rad/s)")
            self._ax_omega.set_xlabel("t (s)")
            self._ax_omega.legend(loc='upper right', fontsize=8)
            self._ax_omega.grid(alpha=0.3)

            # PID 当前值柱状图
            self._ax_act_bar.set_title("PID Gains (current, 20 dims)")
            self._ax_act_bar.set_xlabel("index (0-19)")
            y_min = float(np.min(self.pid_safe_min)) if hasattr(self, "pid_safe_min") else 0.0
            y_max = float(np.max(self.pid_safe_max)) if hasattr(self, "pid_safe_max") else 1.0
            pad = 0.1 * max(1e-6, (y_max - y_min))
            self._ax_act_bar.set_ylim(y_min - pad, y_max + pad)
            self._ax_act_bar.set_xticks(range(0, 20, 2))
            self._act_bars = self._ax_act_bar.bar(range(20), np.zeros(20), color="#4393c3")

            # PID 全程历史热力图
            self._hist_pid = []
            self._hist_pid_t = []  # 对应时间戳（秒）
            self._ax_act_hist.set_title("PID Gains History (full)")
            self._ax_act_hist.set_xlabel("dim (0-19)")
            self._ax_act_hist.set_ylabel("step")
            vmin = float(np.min(self.pid_safe_min)) if hasattr(self, "pid_safe_min") else 0.0
            vmax = float(np.max(self.pid_safe_max)) if hasattr(self, "pid_safe_max") else 1.0
            init_img = np.zeros((1, 20), dtype=float)
            self._im_action = self._ax_act_hist.imshow(init_img, aspect='auto', origin='lower',
                                                       vmin=vmin, vmax=vmax, cmap='viridis')
            self._cbar = self._render_fig.colorbar(self._im_action, ax=self._ax_act_hist, fraction=0.046, pad=0.04)
            self._cbar.set_label("Gain value")

            self._render_initialized = True

        # 基于 episode_* 子步缓冲绘制连续曲线（全程，不做滑窗）
        end_idx = int(self.current_substep)
        if end_idx <= 1:
            return
        start_idx = 0

        t = self.episode_t_all[start_idx:end_idx]
        pos = self.episode_pos_all[start_idx:end_idx]
        vel = self.episode_vel_all[start_idx:end_idx]
        eul = np.rad2deg(self.episode_euler_all[start_idx:end_idx])
        omg = self.episode_omega_all[start_idx:end_idx]

        pos_ref = self.episode_pos_des_all[start_idx:end_idx]
        vel_ref = self.episode_vel_des_all[start_idx:end_idx]
        eul_ref = np.rad2deg(self.episode_euler_des_all[start_idx:end_idx])
        omg_ref = self.episode_omega_des_all[start_idx:end_idx]

        # 更新位置
        self._ln_px.set_data(t, pos[:, 0]); self._ln_py.set_data(t, pos[:, 1]); self._ln_pz.set_data(t, pos[:, 2])
        self._ln_px_ref.set_data(t, pos_ref[:, 0]); self._ln_py_ref.set_data(t, pos_ref[:, 1]); self._ln_pz_ref.set_data(t, pos_ref[:, 2])
        self._ax_pos.relim(); self._ax_pos.autoscale_view()

        # 更新速度
        self._ln_vx.set_data(t, vel[:, 0]); self._ln_vy.set_data(t, vel[:, 1]); self._ln_vz.set_data(t, vel[:, 2])
        self._ln_vx_ref.set_data(t, vel_ref[:, 0]); self._ln_vy_ref.set_data(t, vel_ref[:, 1]); self._ln_vz_ref.set_data(t, vel_ref[:, 2])
        self._ax_vel.relim(); self._ax_vel.autoscale_view()

        # 更新姿态（度）
        self._ln_roll.set_data(t, eul[:, 0]); self._ln_pitch.set_data(t, eul[:, 1]); self._ln_yaw.set_data(t, eul[:, 2])
        self._ln_roll_ref.set_data(t, eul_ref[:, 0]); self._ln_pitch_ref.set_data(t, eul_ref[:, 1]); self._ln_yaw_ref.set_data(t, eul_ref[:, 2])
        self._ax_att.relim(); self._ax_att.autoscale_view()

        # 更新角速度
        self._ln_p.set_data(t, omg[:, 0]); self._ln_q.set_data(t, omg[:, 1]); self._ln_r.set_data(t, omg[:, 2])
        self._ln_p_ref.set_data(t, omg_ref[:, 0]); self._ln_q_ref.set_data(t, omg_ref[:, 1]); self._ln_r_ref.set_data(t, omg_ref[:, 2])
        self._ax_omega.relim(); self._ax_omega.autoscale_view()

        # PID 当前值与历史
        pid_vec = getattr(self, "pid_K_prev_vec", None)
        if pid_vec is None or np.size(pid_vec) != 20:
            if hasattr(self, "pid_safe_min") and hasattr(self, "pid_safe_max"):
                pid_vec = (self.pid_safe_min + self.pid_safe_max) / 2.0
            else:
                pid_vec = np.zeros(20, dtype=float)
        pid_vec = np.asarray(pid_vec, dtype=float).ravel()

        # 柱状图（当前 PID）
        for i, rect in enumerate(self._act_bars):
            rect.set_height(float(pid_vec[i]))

        # 全程历史热力图（每次 render 追加一行，不截断）
        self._hist_pid.append(pid_vec)
        self._hist_pid_t.append(float(self.time))
        pid_hist_arr = np.asarray(self._hist_pid)  # [steps, 20]
        if pid_hist_arr.ndim == 2 and pid_hist_arr.shape[0] > 0:
            self._im_action.set_data(pid_hist_arr)
            vmin = float(np.min(self.pid_safe_min)) if hasattr(self, "pid_safe_min") else float(pid_hist_arr.min())
            vmax = float(np.max(self.pid_safe_max)) if hasattr(self, "pid_safe_max") else float(pid_hist_arr.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            self._im_action.set_clim(vmin, vmax)
            self._ax_act_hist.set_ylim(0, max(1, pid_hist_arr.shape[0]-1))
            self._ax_act_hist.set_xlim(0, 19)

        # 懒加载初始化（20 路增益 4x5 波形窗口）
        if not hasattr(self, "_gains_initialized") or not getattr(self, "_gains_initialized", False):
            self._gains_fig, axes = plt.subplots(4, 5, figsize=(14, 10), sharex=True)
            self._gains_fig.suptitle("PID Gains Waveforms (full)", fontsize=12)
            self._gain_axes = axes.flatten().tolist()
            self._gain_lines = []

            # 每个维度的名字（与 20 维顺序一致）
            self._gain_names = [
                "posP_xy", "posP_z",
                "velP_xy", "velP_z",
                "velI_xy", "velI_z",
                "velD_xy", "velD_z",
                "attP_phi", "attP_theta", "attP_psi",
                "rateP_p", "rateP_q", "rateP_r",
                "rateI_p", "rateI_q", "rateI_r",
                "rateD_p", "rateD_q", "rateD_r",
            ]

            # 逐轴初始化
            for i, ax in enumerate(self._gain_axes):
                (ln,) = ax.plot([], [], 'C0-', lw=1.2)
                self._gain_lines.append(ln)
                title = self._gain_names[i] if i < len(self._gain_names) else f"K[{i}]"
                ax.set_title(title, fontsize=9)
                # 维度对应的安全边界
                if hasattr(self, "pid_safe_min") and hasattr(self, "pid_safe_max") and len(self.pid_safe_min) == 20:
                    ymin = float(self.pid_safe_min[i]); ymax = float(self.pid_safe_max[i])
                    if ymax <= ymin:
                        ymax = ymin + 1.0
                    pad = 0.08 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
                ax.grid(alpha=0.3)
                if i >= 15:
                    ax.set_xlabel("t (s)")
                # 更紧凑布局，隐藏 y 标签刻度密度
                # ax.set_ylabel("K")
            plt.tight_layout()
            self._gains_initialized = True

        # 更新 20 路增益波形
        if hasattr(self, "_gains_initialized") and self._gains_initialized:
            t_gain = np.asarray(self._hist_pid_t, dtype=float) if len(self._hist_pid_t) > 0 else np.arange(pid_hist_arr.shape[0])
            for i, ln in enumerate(self._gain_lines):
                ln.set_data(t_gain, pid_hist_arr[:, i])
                ax = self._gain_axes[i]
                # x 轴自适应
                if t_gain.size > 1:
                    ax.set_xlim(t_gain[0], t_gain[-1])

        # 刷新两个窗口
        for fig in [getattr(self, "_render_fig", None), getattr(self, "_gains_fig", None)]:
            if fig is None:
                continue
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except Exception:
                pass
        try:
            plt.pause(0.001)
        except Exception:
            pass