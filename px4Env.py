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
import config

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
    return np.array([phi, theta, psi])  #[R,P,Y]

class QuadcopterEnv(gym.Env):
    """
    PX4四旋翼仿真环境用于强化学习（统一时间步长版本）
    """
    def __init__(self,
                 control_type: str = "xyz_vel",
                 trajectory_type: int = 5,
                 yaw_type: int = 3,
                 averVel: float = 1,
                 max_episode_steps: int = 6000,  
                 dt: float = 0.005,      
                 eval_mode: bool = False,
                 wind_type: str = 'None',
                 wind_strength: float = 0.0,
                 attitude_bound_start: float = 90.0,        #  姿态越界阈值课程学习参数
                 attitude_bound_end: float = 80.0,
                 attitude_bound_decay_epochs: int = 1000,
              compat_truncated_as_done: bool = True,
              error_gating_enabled: Optional[bool] = None,
            ):

        super(QuadcopterEnv, self).__init__()
        
        # ==================== 基础环境参数 ====================
        self.control_type = control_type
        self.trajectory_type = trajectory_type
        self.yaw_type = yaw_type
        self.averVel = averVel
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.eval_mode = eval_mode          
        self.wind_type = wind_type
        self.wind_strength = wind_strength
        self.terminated_count = 0
        
        self.ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]

        # ==================== 动作空间模式 ====================
        self.enable_smooth_transition = True
        self.alpha = 0.3    # 平滑过渡系数（仅在enable_smooth_transition=True时生效）
        
        # 奖励缩放因子
        self.reward_scale = 1.0
        self.failure_penalty = -15000.0   

        #  子步改进相关参数
        self.compat_truncated_as_done = bool(compat_truncated_as_done)

        # 降低速度/姿态权重（单步误差波动更大）
        self.segment_weights = dict(
            # 跟踪误差项
            W_pos_xy=0.1,
            W_pos_z=0.1,
            W_vel_xy=0.3,   
            W_vel_z=0.3,    
            W_att=0.2,                       
            W_omega=0.1,                      
            
            # 惩罚项
            W_u=0.0,
            W_dv=0.0,
            W_domega=0.0,
            W_du=0.0,
            W_sat=0.0,
            W_over=1.0,
        )

        # ==================== 限制与判据 ====================
        self.limits = dict(
            w_cmd_min=0.0,
            w_cmd_max=2e3,
            overshoot_z_tol=0.0,
            normalize_rate_by_dt=True,
            overshoot_pos_tol=0.0,
            overshoot_vel_tol=0.0,
            overshoot_att_tol=0.0,
            overshoot_ang_vel_tol=0.0,
        )

        # ==================== 回合跟踪 ====================
        self.current_step = 0
        self.episode_reward = 0.0

        # ==================== 动作空间定义 ====================
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # ====================  观测空间定义 ====================
        obs_dim = 56
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ==================== 速度分段PID参数安全范围 ====================
        # 定义三个速度域，每个域有不同的PID参数安全范围
        self.velocity_based_pid_domains = {
            # 低速域 (0-5 m/s): 悬停、慢速巡航，允许较大增益范围
            'low': {
                'vel_range': (0.0, 5.0),
                'pid_safe_min': np.array([
                    0.5, 0.4,      # posP_xy, posP_z
                    10.0, 8,      # velP_xy, velP_z
                    18.0, 1.2,      # velI_xy, velI_z
                    0.3, 0.6,      # velD_xy, velD_z
                    5.0, 5.0, 2.0, # attP (roll, pitch, yaw)
                    0.3, 0.3, 0.2, # rateP
                    1.5, 1.5, 4.0, # rateI
                    0.005, 0.005, 0.003  # rateD
                ], dtype=np.float32),
                'pid_safe_max': np.array([
                    1.5, 1.2,      # posP_xy, posP_z
                    14.0, 9.0,    # velP_xy, velP_z
                    22.0, 1.8,     # velI_xy, velI_z (允许较大积分)
                    0.8, 1.2,      # velD_xy, velD_z
                    10.0, 10.0, 4.0, # attP
                    0.8, 0.8, 0.5, # rateP
                    3.5, 3.5, 6.0, # rateI
                    0.01, 0.01, 0.008  # rateD
                ], dtype=np.float32)
            },
            
            # 中速域 (5-10 m/s): 正常巡航，平衡响应与稳定性
            'medium': {
                'vel_range': (5.0, 10.0),
                'pid_safe_min': np.array([
                    0.5, 0.4,
                    8.0, 6.5,      # 提高速度环P增益下限
                    18, 1.4,      # 提高I增益下限
                    0.4, 0.9,      # 提高D增益下限（抑制振荡）
                    6, 6, 2.5, # 提高姿态环P增益下限
                    0.4, 0.4, 0.25,
                    2.0, 2.0, 4.0,
                    0.006, 0.006, 0.003
                ], dtype=np.float32),
                'pid_safe_max': np.array([
                    1.5, 1.2,      # 降低位置环上限
                    12.0, 10.0,    # 适度降低速度环上限
                    22.0, 1.8,     # 降低I增益上限（防止积分饱和）
                    0.6, 1.1,      # 降低D增益上限
                    10.0, 10.0, 4.0, # attP
                    0.8, 0.8, 0.5, # rateP
                    3.5, 3.5, 6.0, # rateI
                    0.01, 0.01, 0.008  # rateD
                ], dtype=np.float32)
            },
            
            # 高速域 (10-15 m/s): 高速飞行，强调稳定性和安全性
            'high': {
                'vel_range': (10.0, 15.0),
                'pid_safe_min': np.array([
                    0.5, 0.4,     # 降低位置环下限
                    7, 8,      # 进一步提高速度环下限
                    18, 1.5,      # 提高I增益下限
                    0.45, 0.9,      # 更高的D增益下限（强化阻尼）
                    6.5, 6.5, 2.5, # 提高姿态环P增益下限
                    0.5, 0.5, 0.25,
                    2.0, 2.0, 4.0,
                    0.006, 0.006, 0.003
                ], dtype=np.float32),
                'pid_safe_max': np.array([
                    1.5, 1.2,      # 进一步降低位置环上限
                    9, 9,    # 降低速度环上限（防止过冲）
                    21, 1.7,     # 大幅降低I增益（高速时积分需谨慎）
                    0.55, 1.1,      # 降低D增益上限
                    10.0, 10.0, 4.0, # attP
                    0.8, 0.8, 0.5, # rateP
                    3.5, 3.5, 6.0, # rateI
                    0.01, 0.01, 0.008  # rateD
                ], dtype=np.float32)
            }
        }
        
        # 当前使用的参数域（初始化为低速域）
        self.current_pid_domain = 'low'
        self.pid_safe_min = self.velocity_based_pid_domains['low']['pid_safe_min'].copy()
        self.pid_safe_max = self.velocity_based_pid_domains['low']['pid_safe_max'].copy()
        
        # 域切换平滑过渡参数
        self.domain_transition_steps = 10  # 切换域时的平滑步数
        self._domain_transition_counter = 0
        self._domain_transition_target_min = None
        self._domain_transition_target_max = None

        # ==================== 初始化仿真组件 ====================
        self._initialize_simulation()

        self.pid_K_prev_vec=(self.pid_safe_min + self.pid_safe_max)/2.0  # 使用中间值作为初始值

        self.init_actions = self.normalize_pid_parameters(self.pid_K_prev_vec)
        self.prev_action = self.init_actions
        self._apply_pid_parameters(self.init_actions)

        # ==================== 误差门控  ====================
        # 控制优先级
        if error_gating_enabled is not None:
            self.error_gating_enabled = bool(error_gating_enabled)
        else:
            try:
                self.error_gating_enabled = bool(config.ERROR_GATING_ENABLED)
            except Exception:
                self.error_gating_enabled = True

        # 门控统计信息（在 reset() 中会被重置为 0）
        self.gating_stats = {
            'frozen_count': 0,
            'limited_count': 0,
            'free_count': 0,
            'total_count': 0,
        }

        # 记录PID变化信息
        self.last_pid_delta_norm = 0.0
        self.last_pid_delta_max = 0.0
        
        # 步级日志
        self.step_log_rows = []
  
        # ==================== 姿态越界阈值课程学习 ====================
        self.attitude_bound_start = attitude_bound_start
        self.attitude_bound_end = attitude_bound_end
        self.attitude_bound_decay_epochs = attitude_bound_decay_epochs
        self.attitude_bound_current = np.deg2rad(attitude_bound_start)
        self.current_epoch = 0
        
        log.info(
            f"Attitude bound curriculum: "
            f"{attitude_bound_start}° → {attitude_bound_end}° "
            f"over {attitude_bound_decay_epochs} epochs"
        )

    def normalize_pid_parameters(self, pid_K_vec):
        """将PID参数向量归一化到[-1, 1]范围"""
        range_vec = self.pid_safe_max - self.pid_safe_min
        normalized_pid = 2 * (pid_K_vec - self.pid_safe_min) / range_vec - 1
        return normalized_pid
    
    def update_attitude_bound(self, current_epoch: int) -> float:
        """根据当前epoch更新姿态越界阈值（线性衰减）"""
        self.current_epoch = current_epoch
        progress = min(current_epoch / self.attitude_bound_decay_epochs, 1.0)
        current_deg = self.attitude_bound_start + progress * (
            self.attitude_bound_end - self.attitude_bound_start
        )
        self.attitude_bound_current = np.deg2rad(current_deg)
        return current_deg

    # =========================
    # 仿真初始化与内部工具
    # =========================
    def _initialize_simulation(self):
        """
        初始化仿真组件（仅涉及对象创建和初始状态）+

        该方法负责：
        - 创建 Quadcopter, Trajectory, Control, Wind 实例
        - 使用初始时间、时间步长生成初始参考状态并让控制器计算一次命令
        - 为训练/评估预分配用于记录整个 episode 的数组

        注意：此函数不推进仿真时间，仅做初始化与预分配，便于后续快速 step
        """
        # 当前仿真时间
        self.time = 0.0
        self.quad = Quadcopter(self.time)
        self.traj_select = np.array([self.trajectory_type, self.yaw_type, self.averVel])  
        self.traj = Trajectory(self.quad, ctrlType=self.control_type, trajSelect=self.traj_select)
        self.ctrl = Control(self.quad, self.traj.yawType)
        self.wind = Wind(self.wind_type, self.wind_strength, 90, -15)

        sDes = self.traj.desiredState(0, self.dt, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.dt)

        # 直接使用单步
        T = self.max_episode_steps      #6000步 = 30s

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
    #  reset
    # =========================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态并返回初始观测（符合 Gym API）

        该函数要点：
        - 可选使用传入 seed 保证可重复性
        - 重置时间步计数、奖励累计、历史缓存
        - 重新初始化仿真组件并初始化 PID 域为低速
        - 在 eval 模式下，返回额外的调试信息
        """
        super().reset(seed=seed)

        self._already_terminated = False

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.step_log_rows = []

        if hasattr(self, "_pid_history"):
            self._pid_history = []
            self._pid_time_history = []

        # 重置仿真组件
        self._initialize_simulation()
        log.info(f" reset() called: env_id={id(self)}, resetting to step 0, time={self.time:.3f}s")

        # 🆕 重置PID域到低速域
        self.current_pid_domain = 'low'
        self.pid_safe_min = self.velocity_based_pid_domains['low']['pid_safe_min'].copy()
        self.pid_safe_max = self.velocity_based_pid_domains['low']['pid_safe_max'].copy()
        self._domain_transition_counter = 0
        self._domain_transition_target_min = None
        self._domain_transition_target_max = None

        # 基线PID参数
        self.pid_K_prev_vec = (self.pid_safe_min + self.pid_safe_max) / 2.0
        self.init_actions = np.zeros(20)
        self._apply_pid_parameters(self.init_actions)

        if options and options.get('random_init', False):
            noise = np.random.normal(0, 0.1, 3)
            self.quad.pos += noise
            self.quad.state[0:3] = self.quad.pos

        # 重置历史状态缓存
        self.prev_action = self.init_actions
        self.last_pid_delta_norm = 0.0
        self.last_pid_delta_max = 0.0

        # 重置门控统计（每个 episode 重置）
        if hasattr(self, 'gating_stats'):
            self.gating_stats = {k: 0 for k in self.gating_stats.keys()}

        self.step_action = np.zeros((20,), dtype=getattr(self.action_space, "dtype", np.float32))
        self.step_reward = 0.0

        obs = self._get_observation(self.ctrl.sDesCalc)
        
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
            info = {}
        return obs, info

    # =========================
    #  直接式PID参数应用
    # ========================= 
    def _apply_pid_parameters(self, action: np.ndarray) -> dict:
        """
        将动作向量（归一化到 [-1,1]）映射回当前 pid_safe_min / pid_safe_max 定义的真实 PID 参数，
        并将结果应用到内部控制器（self.ctrl.update_gains）。

        细节：
        - action 形状必须为 (20,)
        - action -> K_target 使用线性映射：K_min + (action+1)/2 * (K_max-K_min)
        - 支持平滑过渡（self.enable_smooth_transition），通过与上一步 pid_K_prev_vec 做加权平均
        - 在最终写回控制器前会对参数做 clip，确保在安全范围内

        返回值：用于 ctrl.update_gains 的字典（每个回路的增益元组）
        """
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != 20:
            raise ValueError(f"Expected action shape (20,), got {action.shape}")
        
        K_min = self.pid_safe_min
        K_max = self.pid_safe_max
        
        assert K_min.shape == (20,) and K_max.shape == (20,), \
            f"K_min/K_max shape mismatch: {K_min.shape}, {K_max.shape}"
        
        range_vec = K_max - K_min
        
        if not hasattr(self, 'pid_K_prev_vec') or self.pid_K_prev_vec is None:
            log.warning(" pid_K_prev_vec not initialized in __init__! Using mid-range fallback...")
            self.pid_K_prev_vec = (K_min + K_max) / 2.0
        
        K_prev_vec = self.pid_K_prev_vec
        K_target = K_min + (action + 1.0) / 2.0 * range_vec
        
        if self.enable_smooth_transition:
            alpha = float(np.clip(self.alpha, 1e-6, 1.0))
            K_new_vec = alpha * K_prev_vec + (1.0 - alpha) * K_target
        else:
            K_new_vec = K_target
        
        K_new_vec = np.clip(K_new_vec, K_min, K_max)
        
        pid_delta = K_new_vec - K_prev_vec
        self.last_pid_delta_norm = float(np.linalg.norm(pid_delta))
        self.last_pid_delta_max = float(np.max(np.abs(pid_delta)))
        self.last_pid_delta_rel_max = float(np.max(np.abs(pid_delta) / np.maximum(range_vec, 1e-6)))
        
        gains_from_action = self._unpack_vec_to_groups(K_new_vec)
        self.ctrl.update_gains(gains_from_action)
        
        self.pid_K_prev_vec = K_new_vec        
        return gains_from_action

    # =========================
    # 辅助函数
    # =========================
    def _update_pid_domain_based_on_velocity(self):
        """
        根据当前参考速度更新PID参数域
        
        Returns:
            tuple: (速度大小, 当前域名称)
        """
        # 计算参考速度的模
        if hasattr(self.ctrl, 'sDesCalc') and self.ctrl.sDesCalc is not None:
            vel_ref = self.ctrl.sDesCalc[3:6]
            vel_mag = float(np.linalg.norm(vel_ref))
        else:
            vel_mag = 0.0
        
        # 确定应该使用的域
        target_domain = None
        for domain_name, domain_config in self.velocity_based_pid_domains.items():
            vel_min, vel_max = domain_config['vel_range']
            if vel_min <= vel_mag < vel_max:
                target_domain = domain_name
                break
        
        # 如果速度超过所有域的范围，使用最高域
        if target_domain is None:
            if vel_mag >= 15.0:
                target_domain = 'high'
            else:
                target_domain = 'low'
        
        # -----------------------------
        # 检查是否需要切换域
        # 说明：域切换会启动平滑过渡流程，将 pid_safe_min/pid_safe_max 线性插值到目标域
        # 这样可避免参数突变导致飞行不稳定
        # -----------------------------
        if target_domain != self.current_pid_domain:
            log.info(
                f"🔄 PID域切换: {self.current_pid_domain} → {target_domain} "
                f"(参考速度: {vel_mag:.2f} m/s)"
            )
            
            # 启动平滑过渡
            self._domain_transition_counter = self.domain_transition_steps
            self._domain_transition_target_min = self.velocity_based_pid_domains[target_domain]['pid_safe_min'].copy()
            self._domain_transition_target_max = self.velocity_based_pid_domains[target_domain]['pid_safe_max'].copy()
            self.current_pid_domain = target_domain
        
        # 执行平滑过渡
        if self._domain_transition_counter > 0:
            alpha = 1.0 - (self._domain_transition_counter / self.domain_transition_steps)
            
            # 线性插值
            current_min = self.velocity_based_pid_domains[self.current_pid_domain]['pid_safe_min']
            current_max = self.velocity_based_pid_domains[self.current_pid_domain]['pid_safe_max']
            
            self.pid_safe_min = (1 - alpha) * self.pid_safe_min + alpha * self._domain_transition_target_min
            self.pid_safe_max = (1 - alpha) * self.pid_safe_max + alpha * self._domain_transition_target_max
            
            self._domain_transition_counter -= 1
            
            if self._domain_transition_counter == 0:
                log.info(f"✅ PID域切换完成: {self.current_pid_domain}")
        
        return vel_mag, target_domain

    def _metrics_to_error_vector(self, m: Dict[str, Any]) -> np.ndarray:
        """
        将 _extract_metrics 返回的字典转换为长度为12的误差向量，顺序为：
        [pos_err(x,y,z), vel_err(x,y,z), att_err(roll,pitch,yaw), rate_err(p,q,r)]
        """
        pos = np.array([m.get('pos_err_x', 0.0), m.get('pos_err_y', 0.0), m.get('pos_err_z', 0.0)], dtype=np.float32)
        vel = np.array([m.get('vel_err_x', 0.0), m.get('vel_err_y', 0.0), m.get('vel_err_z', 0.0)], dtype=np.float32)

        # _extract_metrics 中命名略显不规则，按 (roll, pitch, yaw) 的顺序组织
        att = np.array([m.get('att_err_roll', 0.0), m.get('att_err_pitch', 0.0), m.get('att_err_yaw', 0.0)], dtype=np.float32)
        rate = np.array([m.get('omega_err_p', 0.0), m.get('omega_err_q', 0.0), m.get('omega_err_r', 0.0)], dtype=np.float32)

        return np.concatenate([pos, vel, att, rate]).astype(np.float32)

    def _error_gated_action_adaptation(self, action: np.ndarray, errors: np.ndarray) -> np.ndarray:
        """
        基于误差的门控动作自适应（参考 run_single_with_PPOAgent.py 的实现）
        """
        if not getattr(self, 'error_gating_enabled', False):
            return action

        # 根据当前域调整阈值
        if self.current_pid_domain == 'low':
            thresh1 = 0.1
            thresh2 = 0.3
        elif self.current_pid_domain == 'medium':
            thresh1 = 0.15
            thresh2 = 0.5
        else:
            thresh1 = 0.2
            thresh2 = 0.7

        pos_err = float(np.linalg.norm(errors[0:3]))
        vel_err = float(np.linalg.norm(errors[3:6]))
        att_err = float(np.linalg.norm(errors[6:9]))
        rate_err = float(np.linalg.norm(errors[9:12]))

        # 映射为参数组（基于 px4Env 中 20 维向量的组织）
        param_groups = {
            'pos_P': (0, 2, pos_err, 1.0),
            'vel_P': (2, 4, vel_err, 1.0),
            'vel_I': (4, 6, vel_err, 0.5),
            'vel_D': (6, 8, vel_err, 0.8),
            'att_P': (8, 11, att_err, 1.0),
            'rate_P': (11, 14, rate_err, 1.0),
            'rate_I': (14, 17, rate_err, 0.5),
            'rate_D': (17, 20, rate_err, 0.8),
        }

        gated_action = np.copy(action)

        for _, (start, end, error_norm, thresh_scale) in param_groups.items():
            t1 = thresh1 * thresh_scale
            t2 = thresh2 * thresh_scale

            if error_norm < t1:
                # 冻结
                gated_action[start:end] = self.prev_action[start:end]
                self.gating_stats['frozen_count'] += (end - start)

            elif error_norm < t2:
                # 限制调整幅度
                delta = action[start:end] - self.prev_action[start:end]
                error_ratio = (error_norm - t1) / max((t2 - t1), 1e-8)
                limit_delta = 0.1 + 0.4 * error_ratio
                delta = np.clip(delta, -limit_delta, limit_delta)
                gated_action[start:end] = self.prev_action[start:end] + delta
                self.gating_stats['limited_count'] += (end - start)

            else:
                # 自由调整
                self.gating_stats['free_count'] += (end - start)

        self.gating_stats['total_count'] += 20
        # 保存为 prev_action 供下一步使用
        self.prev_action = np.copy(gated_action)
        return gated_action

    def apply_pid_parameters_with_gating(self, action: np.ndarray, errors: np.ndarray) -> dict:
        """
        对动作进行误差门控后应用 PID 参数（和 _apply_pid_parameters 行为一致）
        """
        gated_action = self._error_gated_action_adaptation(np.asarray(action, dtype=np.float32).ravel(), errors)
        return self._apply_pid_parameters(gated_action)
    
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
    
    def _log_reward_to_csv(self, reward_dict):
        """将奖励详情记录到CSV文件"""
        log_dir = "logs/reward_debug"
        os.makedirs(log_dir, exist_ok=True)
        
        if not hasattr(self, 'csv_log_file'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_log_file = os.path.join(log_dir, f"reward_log_{timestamp}.csv")
            self.csv_initialized = False
        
        row_data = {
            'step': getattr(self, 'current_step', 0),
            **reward_dict
        }
        
        file_exists = os.path.isfile(self.csv_log_file)
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            if not file_exists or not self.csv_initialized:
                writer.writeheader()
                self.csv_initialized = True
            
            writer.writerow(row_data)
    
    # =========================
    # step() 函数（主循环）- 🔄 统一时间步长版本
    # =========================
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个环境步骤（统一时间步长后无需子步循环）
        """
        # 阶段1：根据参考速度更新PID安全域，并提前准备门控误差
        vel_mag, current_domain = self._update_pid_domain_based_on_velocity()
        errors_vec = self._prepare_errors_for_gating()
        self.step_action = self.apply_pid_parameters_with_gating(action, errors_vec)

        # 阶段2：推进一次物理仿真，并记录用于奖励的单步样本
        prev_cache = self._capture_prev_cache()
        _, sample = self._run_single_physics_step(prev_cache)

        # 阶段3：基于单步样本计算奖励并完成缩放
        reward, segment_reward, reward_for_debug, _ = self._compute_scaled_reward_from_sample(sample)

        # 阶段4：统一处理终止/截断判据
        terminated, truncated, reward, term_info = self._check_and_handle_termination(
            reward, vel_mag, current_domain
        )

        if terminated or truncated:
            obs = self._get_observation(self.ctrl.sDesCalc)
            info = term_info or {}
            if 'velocity_magnitude' not in info:
                info['velocity_magnitude'] = float(vel_mag)
            if 'pid_domain' not in info:
                info['pid_domain'] = str(current_domain)
            return obs, reward, terminated, truncated, info

        # 记录历史数据到数组
        self._record_episode_step()

        # 记录调试信息
        if self.eval_mode:
            if not hasattr(self, "_pid_history"):
                self._pid_history = []
                self._pid_time_history = []
            
            pid_vec = getattr(self, "pid_K_prev_vec", None)
            if pid_vec is not None and np.size(pid_vec) == 20:
                self._pid_history.append(np.asarray(pid_vec, dtype=float).copy())
                self._pid_time_history.append(float(self.time))

            reward_for_debug['segment_reward'] = segment_reward
            reward_for_debug['reward_total_before_scale'] = segment_reward
            self._log_reward_to_csv(reward_for_debug)

        # 正常路径：返回本步结果
        obs = self._get_observation(self.ctrl.sDesCalc)
        info = {
            'velocity_magnitude': float(vel_mag),
            'pid_domain': str(current_domain),
            'domain_transition_progress': float(1.0 - self._domain_transition_counter / self.domain_transition_steps) if self._domain_transition_counter > 0 else 1.0
        }
        return obs, reward, False, False, info

    def _prepare_errors_for_gating(self) -> np.ndarray:
        """基于当前参考状态提取误差向量，为误差门控提供输入。"""
        try:
            m_pre = self._extract_metrics(self.ctrl.sDesCalc)
            return self._metrics_to_error_vector(m_pre)
        except Exception:
            return np.zeros((12,), dtype=np.float32)

    def _capture_prev_cache(self) -> Dict[str, Optional[np.ndarray]]:
        """缓存当前速度/角速度/电机命令，用于计算dv、dω、du等变化率指标。"""
        cache = {
            "vel": self.quad.vel.copy(),
            "omega": self.quad.omega.copy(),
            "w_cmd": None,
        }
        if hasattr(self.ctrl, "w_cmd") and self.ctrl.w_cmd is not None:
            cache["w_cmd"] = self.ctrl.w_cmd.copy()
        return cache

    def _run_single_physics_step(self, prev_cache: Dict[str, Optional[np.ndarray]]):
        """推进一次仿真并返回 (误差指标, 奖励样本)。"""
        sDes = self.traj.desiredState(self.time, self.dt, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.dt)
        self.quad.update(self.time, self.dt, self.ctrl.w_cmd, self.wind)

        self.time += self.dt
        self.current_step += 1

        sDesCalc = self.ctrl.sDesCalc
        metrics = self._extract_metrics(sDesCalc)
        sample = self._collect_step_sample(metrics, prev_cache)
        return metrics, sample

    def _compute_scaled_reward_from_sample(self, sample: Dict[str, float]):
        """对单步样本调用奖励函数，并完成reward_scale缩放。"""
        segment_reward, reward_for_debug, metrics = self._compute_reward_step([sample])
        reward = segment_reward * self.reward_scale
        self.step_reward = reward
        self.episode_reward += reward
        return reward, segment_reward, reward_for_debug, metrics

    def _check_and_handle_termination(self, reward: float, vel_mag: float, current_domain: str):
        """统一处理越界和时间上限逻辑，返回(terminated, truncated, reward, info)。"""
        info: Dict[str, Any] = {}

        if self._is_out_of_bounds():
            self._already_terminated = True
            self.terminated_count += 1

            penalty = self.failure_penalty * self.reward_scale
            self.episode_reward += penalty

            info = {
                'termination_reason': 'out_of_bounds',
                'step': self.current_step,
                'env_id': id(self),
                'velocity_magnitude': float(vel_mag),
                'pid_domain': str(current_domain),
            }

            log.warning(
                f"🚨 Episode terminated at step {self.current_step}, "
                f"penalty: {self.failure_penalty} (scaled: {penalty:.2f})"
            )

            return True, False, penalty, info

        terminated = False
        truncated = False

        if self.current_step >= self.max_episode_steps:
            truncated = True
            if getattr(self, "compat_truncated_as_done", False):
                terminated = True
            self._already_terminated = True

            info = {
                'termination_reason': 'time_limit',
                'step': self.current_step,
                'TimeLimit.truncated': True,
                'env_id': id(self),
                'velocity_magnitude': float(vel_mag),
                'pid_domain': str(current_domain),
            }

            log.info(f"⏱️ Episode time limit reached at step {self.current_step}")
            return terminated, truncated, reward, info

        return False, False, reward, info

    # ============================================================
    # 记录历史数据
    # ============================================================
    def _record_episode_step(self) -> Optional[Dict]:
        """
        记录当前步的历史数据（🔄 统一时间步长后直接使用 current_step）
        """
        idx = self.current_step  # 🔄 修改：直接使用 current_step
        
        # 检查索引越界
        if idx >= self.episode_t_all.shape[0]:
            log.error(f"❌ step索引 {idx} 超出预分配数组大小 {self.episode_t_all.shape[0]}")
            return None
        
        # 记录数据
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

        # 评估模式：返回数据
        if self.eval_mode:
            return {
                'time': float(self.time),
                'position': self.quad.pos.copy(),
                'velocity': self.quad.vel.copy(),
                'attitude': self.quad.euler.copy(),
                'omega': self.quad.omega.copy(),
                'quaternion': self.quad.quat.copy(),
                'position_des': sDesCalc[0:3] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'velocity_des': sDesCalc[3:6] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'attitude_des': _quaternion_to_euler(sDesCalc[9:13]) if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
                'omega_des': sDesCalc[13:16] if hasattr(self.ctrl, 'sDesCalc') else np.zeros(3),
            }
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
        
        att_err_yaw = float(att_err_vec[0])
        att_err_pitch = float(att_err_vec[1])
        att_err_roll = float(att_err_vec[2])
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

            "att_err_yaw": att_err_yaw,
            "att_err_roll": abs(att_err_roll),
            "att_err_pitch": abs(att_err_pitch),
            "att_norm": att_norm,

            "omega_err_p": omega_err_p,
            "omega_err_q": omega_err_q,
            "omega_err_r": omega_err_r,
            "omega_norm": omega_norm,
        }

    def _detect_overshoot(self, error):
        """
        检测超调（overshoot）事件。

        输入：error 字典（由 _extract_metrics 提供），包含速度误差、姿态误差和角速度误差。
        逻辑：如果误差与当前量的符号相反且超过对应容忍阈值，则认为发生超调。

        返回：超调命中数（如果检测到任何超调），否则返回 0。
        """
        overshoot_results = {
            "vel": [0, 0, 0],
            "att": [0, 0, 0],
            "ang_vel": [0, 0, 0]
        }
        
        vel_err = [error["vel_err_x"], error["vel_err_y"], error["vel_err_z"]]
        att_err = [error["att_err_roll"], error["att_err_pitch"], error["att_err_yaw"]]
        ang_vel_err = [error["omega_err_p"], error["omega_err_q"], error["omega_err_r"]]

        overthrow_detected = False

        # 检测超调：三轴速度
        for i in range(3):
            current_vel = float(self.quad.vel[i])
            if vel_err[i] * current_vel < 0 and abs(vel_err[i]) > self.limits["overshoot_vel_tol"]:
                overshoot_results['vel'][i] = vel_err[i]
                overthrow_detected = True

        # 检测超调：三轴姿态
        for i in range(3):
            current_att = float(self.quad.euler[i])
            if att_err[i] * current_att < 0 and abs(att_err[i]) > self.limits["overshoot_att_tol"]:
                overshoot_results['att'][i] = att_err[i]
                overthrow_detected = False

        # 检测超调：三轴角速度
        for i in range(3):
            current_ang_vel = float(self.quad.omega[i])
            if ang_vel_err[i] * current_ang_vel < 0 and abs(ang_vel_err[i]) > self.limits["overshoot_ang_vel_tol"]:
                overshoot_results['ang_vel'][i] = ang_vel_err[i]
                overthrow_detected = False
        
        if isinstance(overshoot_results, dict):
            overshoot_hits_sum = sum(overshoot_results.get('vel', []))
        else:
            overshoot_hits_sum = 0.0

        return overshoot_hits_sum if overthrow_detected else 0

    def _collect_step_sample(self, m: Dict[str, Any], prev_cache: Dict[str, Any]) -> Dict[str, float]:
        """
        在本步中收集原子指标
        """
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            control_effort = float(np.sum(np.square(self.ctrl.w_cmd)))
        else:
            control_effort = 0.0

        dv = 0.0
        domega = 0.0
        du = 0.0

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

        overshoot_hit = self._detect_overshoot(m)

        entry = {
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

    def _compute_reward_step(self, samples: List[Dict[str, float]]) -> Tuple[float, Dict[str, float], Dict]:
        """
        计算当前步的奖励并返回 (segment_reward, reward_for_debug, metrics)
        """

        assert len(samples) == 1, f"Expected 1 sample after time unification, got {len(samples)}"

        sample = samples[0]

        metrics_map = self._reward_metric_values(sample)
        components = self._build_reward_components(metrics_map)

        total_penalty = sum(comp["penalty"] for comp in components)
        segment_reward = -float(total_penalty)

        reward_for_debug = {
            comp["debug_key"]: comp["penalty"] for comp in components
        }

        metrics = {
            'segment_duration': float(self.dt),
            'segment_reward': segment_reward,
            'samples': 1,
            'rmse_pos_xy': metrics_map['rmse_pos_xy'],
            'rmse_pos_z': metrics_map['rmse_pos_z'],
            'rmse_vel_xy': metrics_map['rmse_vel_xy'],
            'rmse_vel_z': metrics_map['rmse_vel_z'],
            'rmse_att': metrics_map['rmse_att'],
            'rms_omega': metrics_map['rms_omega'],
            'mean_control_effort': metrics_map['mean_control_effort'],
            'rms_dv': metrics_map['rms_dv'],
            'rms_domega': metrics_map['rms_domega'],
            'rms_du': metrics_map['rms_du'],
            'saturation_hits': metrics_map['saturation_hits'],
            'overshoot_hits': metrics_map['overshoot_hits'],
            'oscillation_penalty': metrics_map['oscillation_penalty'],
        }

        return segment_reward, reward_for_debug, metrics

    def _reward_metric_values(self, sample: Dict[str, float]) -> Dict[str, float]:
        """将原始 sample 还原为便于阅读的 RMS/能量/命中指标。"""
        return {
            'rmse_pos_xy': float(np.sqrt(sample["pos_xy2"])),
            'rmse_pos_z': float(np.sqrt(sample["pos_z2"])),
            'rmse_vel_xy': float(np.sqrt(sample["vel_xy2"])),
            'rmse_vel_z': float(np.sqrt(sample["vel_z2"])),
            'rmse_att': float(np.sqrt(sample["att2"])),
            'rms_omega': float(np.sqrt(sample["omega2"])),
            'mean_control_effort': float(sample["u2"]),
            'rms_dv': float(np.sqrt(sample["dv2"])),
            'rms_domega': float(np.sqrt(sample["domega2"])),
            'rms_du': float(np.sqrt(sample["du2"])),
            'saturation_hits': float(sample["sat_hit"]),
            'overshoot_hits': float(sample["over_hit"]),
            'oscillation_penalty': 0.0,
        }

    def _reward_component_specs(self) -> List[Dict[str, str]]:
        """定义奖励组成所需的元数据（映射到 segment_weights 与 metrics）。"""
        return [
            {"debug_key": "reward_pos_xy", "metric_key": "rmse_pos_xy", "weight_key": "W_pos_xy", "description": "XY 位置误差"},
            {"debug_key": "reward_pos_z", "metric_key": "rmse_pos_z", "weight_key": "W_pos_z", "description": "Z 轴位置误差"},
            {"debug_key": "reward_vel_xy", "metric_key": "rmse_vel_xy", "weight_key": "W_vel_xy", "description": "XY 速度误差"},
            {"debug_key": "reward_vel_z", "metric_key": "rmse_vel_z", "weight_key": "W_vel_z", "description": "Z 轴速度误差"},
            {"debug_key": "reward_att", "metric_key": "rmse_att", "weight_key": "W_att", "description": "姿态误差"},
            {"debug_key": "reward_omega", "metric_key": "rms_omega", "weight_key": "W_omega", "description": "角速度误差"},
            {"debug_key": "reward_u", "metric_key": "mean_control_effort", "weight_key": "W_u", "description": "控制能量"},
            {"debug_key": "reward_dv", "metric_key": "rms_dv", "weight_key": "W_dv", "description": "速度变化率"},
            {"debug_key": "reward_domega", "metric_key": "rms_domega", "weight_key": "W_domega", "description": "角速度变化率"},
            {"debug_key": "reward_du", "metric_key": "rms_du", "weight_key": "W_du", "description": "电机命令变化率"},
            {"debug_key": "reward_sat", "metric_key": "saturation_hits", "weight_key": "W_sat", "description": "推力饱和命中"},
            {"debug_key": "reward_over", "metric_key": "overshoot_hits", "weight_key": "W_over", "description": "超调命中"},
            {"debug_key": "reward_oscillation_penalty", "metric_key": "oscillation_penalty", "weight_override": 1.0, "description": "振荡惩罚"},
        ]

    def _build_reward_components(self, metrics_map: Dict[str, float]) -> List[Dict[str, float]]:
        """依据 specs 将权重、指标和惩罚整理成易读表格。"""
        components: List[Dict[str, float]] = []
        for spec in self._reward_component_specs():
            metric_value = metrics_map.get(spec["metric_key"], 0.0)
            weight = spec.get("weight_override", self.segment_weights.get(spec.get("weight_key", ""), 0.0))
            penalty = float(weight * metric_value)
            components.append({
                **spec,
                "metric_value": metric_value,
                "weight": weight,
                "penalty": penalty,
            })
        return components

    # =========================
    #  构造观测向量
    # =========================
    def _get_observation(self, sDesCalc: np.ndarray) -> np.ndarray:
        """
        构建并返回 56 维观测向量，包含：
        [参考状态(12), 反馈状态(12), 误差信号(12), PID 参数归一化(20)]

        说明：为稳定训练，参考/反馈/误差会做简单的剪切归一化，便于网络学习。
        """
        # 参考状态
        pos_ref, vel_ref, eul_ref, omega_ref = self._reference_state_blocks(sDesCalc)
        pos_fb, vel_fb, eul_fb, omega_fb = self._feedback_state_blocks()

        pos_err, vel_err, eul_err, omega_err = self._error_state_blocks(
            pos_ref, vel_ref, eul_ref, omega_ref,
            pos_fb, vel_fb, eul_fb, omega_fb,
        )

        pos_ref_norm, vel_ref_norm, eul_ref_norm, omega_ref_norm = self._normalize_reference_blocks(
            pos_ref, vel_ref, eul_ref, omega_ref
        )
        pos_fb_norm, vel_fb_norm, eul_fb_norm, omega_fb_norm = self._normalize_feedback_blocks(
            pos_fb, vel_fb, eul_fb, omega_fb
        )
        pos_err_norm, vel_err_norm, eul_err_norm, omega_err_norm = self._normalize_error_blocks(
            pos_err, vel_err, eul_err, omega_err
        )

        pid_normalized = self._normalized_pid_vector()

        obs = np.concatenate([
            pos_ref_norm, vel_ref_norm, eul_ref_norm, omega_ref_norm,
            pos_fb_norm, vel_fb_norm, eul_fb_norm, omega_fb_norm,
            pos_err_norm, vel_err_norm, eul_err_norm, omega_err_norm,
            pid_normalized,
        ])

        return obs.astype(np.float32)

    def _reference_state_blocks(self, sDesCalc: np.ndarray):
        """提取轨迹给定的参考位置/速度/姿态/角速度。"""
        pos_ref = sDesCalc[0:3]
        vel_ref = sDesCalc[3:6]
        eul_ref = _quaternion_to_euler(sDesCalc[9:13])
        omega_ref = sDesCalc[13:16]
        return pos_ref, vel_ref, eul_ref, omega_ref

    def _feedback_state_blocks(self):
        """读取四旋翼当前的反馈状态块。"""
        return self.quad.pos, self.quad.vel, self.quad.euler, self.quad.omega

    def _error_state_blocks(self, pos_ref, vel_ref, eul_ref, omega_ref,
                            pos_fb, vel_fb, eul_fb, omega_fb):
        """计算参考与反馈之间的误差（姿态误差做±π映射）。"""
        pos_err = pos_ref - pos_fb
        vel_err = vel_ref - vel_fb
        eul_err = eul_ref - eul_fb
        eul_err = np.arctan2(np.sin(eul_err), np.cos(eul_err))
        omega_err = omega_ref - omega_fb
        return pos_err, vel_err, eul_err, omega_err

    def _normalize_reference_blocks(self, pos_ref, vel_ref, eul_ref, omega_ref):
        """对参考状态按经验范围做剪切归一化。"""
        pos_ref_norm = np.clip(pos_ref / 10.0, -1, 1)
        vel_ref_norm = np.clip(vel_ref / 15.0, -1, 1)
        eul_ref_norm = np.clip(eul_ref / np.pi, -1, 1)
        omega_ref_norm = np.clip(omega_ref / (2*np.pi), -1, 1)
        return pos_ref_norm, vel_ref_norm, eul_ref_norm, omega_ref_norm

    def _normalize_feedback_blocks(self, pos_fb, vel_fb, eul_fb, omega_fb):
        """对反馈状态按同样尺度处理，避免量纲差异。"""
        pos_fb_norm = np.clip(pos_fb / 10.0, -1, 1)
        vel_fb_norm = np.clip(vel_fb / 15.0, -1, 1)
        eul_fb_norm = np.clip(eul_fb / np.pi, -1, 1)
        omega_fb_norm = np.clip(omega_fb / (2*np.pi), -1, 1)
        return pos_fb_norm, vel_fb_norm, eul_fb_norm, omega_fb_norm

    def _normalize_error_blocks(self, pos_err, vel_err, eul_err, omega_err):
        """对误差信号做幅值压缩，确保网络输入稳定。"""
        pos_err_norm = np.clip(pos_err / 10.0, -1, 1)
        vel_err_norm = np.clip(vel_err / 15.0, -1, 1)
        eul_err_norm = np.clip(eul_err / np.pi, -1, 1)
        omega_err_norm = np.clip(omega_err / (2*np.pi), -1, 1)
        return pos_err_norm, vel_err_norm, eul_err_norm, omega_err_norm

    def _normalized_pid_vector(self):
        """返回最近一次写入控制器的PID参数（归一化后）。"""
        if hasattr(self, 'pid_K_prev_vec') and self.pid_K_prev_vec is not None:
            pid_normalized = 2.0 * (self.pid_K_prev_vec - self.pid_safe_min) / (
                self.pid_safe_max - self.pid_safe_min + 1e-8
            ) - 1.0
            pid_normalized = np.clip(pid_normalized, -1.0, 1.0)
        else:
            pid_normalized = np.zeros(20, dtype=np.float32)
        return pid_normalized
    
    # =========================
    # 终止与信息
    # =========================
    def _is_out_of_bounds(self) -> bool:
        """使用动态姿态阈值"""
        if (np.abs(self.quad.euler[0]) > self.attitude_bound_current or 
            np.abs(self.quad.euler[1]) > self.attitude_bound_current):
            return True
        return False

    # =========================
    # 渲染
    # =========================
    def render(self, mode='human', save_path=None):
        """
        离线渲染环境（episode 结束后绘制完整图表）
        
        Args:
            mode: 渲染模式，'human' 或 'rgb_array'
            save_path: 图片保存路径，如 'results/episode_001.png'
        """
        if mode not in ('human', 'rgb_array', None):
            return

        try:
            import matplotlib
            if save_path:
                matplotlib.use('Agg')  # 无GUI后端，用于保存图片
            import matplotlib.pyplot as plt
        except Exception as e:
            log.warning(f"Matplotlib 不可用，跳过渲染: {e}")
            return

        # 检查是否有数据
        end_idx = int(self.current_step)
        if end_idx <= 1:
            log.warning("渲染失败：没有足够的数据（current_step <= 1）")
            return
        
        start_idx = 0

        # 提取数据
        t = self.episode_t_all[start_idx:end_idx]
        pos = self.episode_pos_all[start_idx:end_idx]
        vel = self.episode_vel_all[start_idx:end_idx]
        eul = np.rad2deg(self.episode_euler_all[start_idx:end_idx])
        omg = self.episode_omega_all[start_idx:end_idx]

        pos_ref = self.episode_pos_des_all[start_idx:end_idx]
        vel_ref = self.episode_vel_des_all[start_idx:end_idx]
        eul_ref = np.rad2deg(self.episode_euler_des_all[start_idx:end_idx])
        omg_ref = self.episode_omega_des_all[start_idx:end_idx]

        # PID 历史数据（如果有）
        pid_history = getattr(self, "_pid_history", [])
        pid_time = getattr(self, "_pid_time_history", [])

        # ==================== 创建图表 ====================
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"Episode Render - ctrl: {self.control_type}, traj: {self.trajectory_type}\n"
            f"Steps: {end_idx}, Time: {t[-1]:.2f}s, Final Reward: {self.episode_reward:.2f}",
            fontsize=14, fontweight='bold'
        )

        # 布局：3行4列
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.30)

        # ==================== 第一行：位置、速度、姿态、角速度 ====================
        ax_pos = fig.add_subplot(gs[0, 0])
        ax_vel = fig.add_subplot(gs[0, 1])
        ax_att = fig.add_subplot(gs[0, 2])
        ax_omega = fig.add_subplot(gs[0, 3])

        # 位置
        ax_pos.plot(t, pos[:, 0], 'b-', lw=1.5, label='X')
        ax_pos.plot(t, pos[:, 1], 'g-', lw=1.5, label='Y')
        ax_pos.plot(t, pos[:, 2], 'r-', lw=1.5, label='Z')
        ax_pos.plot(t, pos_ref[:, 0], 'b--', lw=1.0, alpha=0.7, label='X_ref')
        ax_pos.plot(t, pos_ref[:, 1], 'g--', lw=1.0, alpha=0.7, label='Y_ref')
        ax_pos.plot(t, pos_ref[:, 2], 'r--', lw=1.0, alpha=0.7, label='Z_ref')
        ax_pos.set_title("Position (m)", fontweight='bold')
        ax_pos.set_xlabel("Time (s)")
        ax_pos.set_ylabel("Position (m)")
        ax_pos.legend(loc='best', fontsize=8, ncol=2)
        ax_pos.grid(alpha=0.3)

        # 速度
        ax_vel.plot(t, vel[:, 0], 'b-', lw=1.5, label='Vx')
        ax_vel.plot(t, vel[:, 1], 'g-', lw=1.5, label='Vy')
        ax_vel.plot(t, vel[:, 2], 'r-', lw=1.5, label='Vz')
        ax_vel.plot(t, vel_ref[:, 0], 'b--', lw=1.0, alpha=0.7, label='Vx_ref')
        ax_vel.plot(t, vel_ref[:, 1], 'g--', lw=1.0, alpha=0.7, label='Vy_ref')
        ax_vel.plot(t, vel_ref[:, 2], 'r--', lw=1.0, alpha=0.7, label='Vz_ref')
        ax_vel.set_title("Velocity (m/s)", fontweight='bold')
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.legend(loc='best', fontsize=8, ncol=2)
        ax_vel.grid(alpha=0.3)

        # 姿态
        ax_att.plot(t, eul[:, 0], 'b-', lw=1.5, label='Roll')
        ax_att.plot(t, eul[:, 1], 'g-', lw=1.5, label='Pitch')
        ax_att.plot(t, eul[:, 2], 'r-', lw=1.5, label='Yaw')
        ax_att.plot(t, eul_ref[:, 0], 'b--', lw=1.0, alpha=0.7, label='Roll_ref')
        ax_att.plot(t, eul_ref[:, 1], 'g--', lw=1.0, alpha=0.7, label='Pitch_ref')
        ax_att.plot(t, eul_ref[:, 2], 'r--', lw=1.0, alpha=0.7, label='Yaw_ref')
        ax_att.set_title("Attitude (deg)", fontweight='bold')
        ax_att.set_xlabel("Time (s)")
        ax_att.set_ylabel("Angle (deg)")
        ax_att.legend(loc='best', fontsize=8, ncol=2)
        ax_att.grid(alpha=0.3)

        # 角速度
        ax_omega.plot(t, omg[:, 0], 'b-', lw=1.5, label='p')
        ax_omega.plot(t, omg[:, 1], 'g-', lw=1.5, label='q')
        ax_omega.plot(t, omg[:, 2], 'r-', lw=1.5, label='r')
        ax_omega.plot(t, omg_ref[:, 0], 'b--', lw=1.0, alpha=0.7, label='p_ref')
        ax_omega.plot(t, omg_ref[:, 1], 'g--', lw=1.0, alpha=0.7, label='q_ref')
        ax_omega.plot(t, omg_ref[:, 2], 'r--', lw=1.0, alpha=0.7, label='r_ref')
        ax_omega.set_title("Angular Velocity (rad/s)", fontweight='bold')
        ax_omega.set_xlabel("Time (s)")
        ax_omega.set_ylabel("Ang. Vel. (rad/s)")
        ax_omega.legend(loc='best', fontsize=8, ncol=2)
        ax_omega.grid(alpha=0.3)

        # ==================== 第二行：误差分析 ====================
        ax_pos_err = fig.add_subplot(gs[1, 0])
        ax_vel_err = fig.add_subplot(gs[1, 1])
        ax_att_err = fig.add_subplot(gs[1, 2])
        ax_omega_err = fig.add_subplot(gs[1, 3])

        # 位置误差
        pos_err = pos_ref - pos
        ax_pos_err.plot(t, pos_err[:, 0], 'b-', lw=1.5, label='err_X')
        ax_pos_err.plot(t, pos_err[:, 1], 'g-', lw=1.5, label='err_Y')
        ax_pos_err.plot(t, pos_err[:, 2], 'r-', lw=1.5, label='err_Z')
        ax_pos_err.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_pos_err.set_title("Position Error (m)", fontweight='bold')
        ax_pos_err.set_xlabel("Time (s)")
        ax_pos_err.set_ylabel("Error (m)")
        ax_pos_err.legend(loc='best', fontsize=8)
        ax_pos_err.grid(alpha=0.3)

        # 速度误差
        vel_err = vel_ref - vel
        ax_vel_err.plot(t, vel_err[:, 0], 'b-', lw=1.5, label='err_Vx')
        ax_vel_err.plot(t, vel_err[:, 1], 'g-', lw=1.5, label='err_Vy')
        ax_vel_err.plot(t, vel_err[:, 2], 'r-', lw=1.5, label='err_Vz')
        ax_vel_err.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_vel_err.set_title("Velocity Error (m/s)", fontweight='bold')
        ax_vel_err.set_xlabel("Time (s)")
        ax_vel_err.set_ylabel("Error (m/s)")
        ax_vel_err.legend(loc='best', fontsize=8)
        ax_vel_err.grid(alpha=0.3)

        # 姿态误差
        att_err = eul_ref - eul
        ax_att_err.plot(t, att_err[:, 0], 'b-', lw=1.5, label='err_Roll')
        ax_att_err.plot(t, att_err[:, 1], 'g-', lw=1.5, label='err_Pitch')
        ax_att_err.plot(t, att_err[:, 2], 'r-', lw=1.5, label='err_Yaw')
        ax_att_err.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_att_err.set_title("Attitude Error (deg)", fontweight='bold')
        ax_att_err.set_xlabel("Time (s)")
        ax_att_err.set_ylabel("Error (deg)")
        ax_att_err.legend(loc='best', fontsize=8)
        ax_att_err.grid(alpha=0.3)

        # 角速度误差
        omega_err = omg_ref - omg
        ax_omega_err.plot(t, omega_err[:, 0], 'b-', lw=1.5, label='err_p')
        ax_omega_err.plot(t, omega_err[:, 1], 'g-', lw=1.5, label='err_q')
        ax_omega_err.plot(t, omega_err[:, 2], 'r-', lw=1.5, label='err_r')
        ax_omega_err.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_omega_err.set_title("Angular Velocity Error (rad/s)", fontweight='bold')
        ax_omega_err.set_xlabel("Time (s)")
        ax_omega_err.set_ylabel("Error (rad/s)")
        ax_omega_err.legend(loc='best', fontsize=8)
        ax_omega_err.grid(alpha=0.3)

        # ==================== 第三行：PID 增益分析 ====================
        if len(pid_history) > 0 and len(pid_time) > 0:
            pid_array = np.array(pid_history)  # shape: (T, 20)
            pid_t = np.array(pid_time)

            # 3D轨迹
            ax_3d = fig.add_subplot(gs[2, 0], projection='3d')
            ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', lw=1.5, label='Actual')
            ax_3d.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'r--', lw=1.0, alpha=0.7, label='Reference')
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title("3D Trajectory", fontweight='bold')
            ax_3d.legend()
            ax_3d.grid(alpha=0.3)

            # PID 当前值柱状图
            ax_pid_bar = fig.add_subplot(gs[2, 1])
            pid_final = pid_array[-1, :]  # 最后一步的 PID 值
            colors = plt.cm.viridis(np.linspace(0, 1, 20))
            ax_pid_bar.bar(range(20), pid_final, color=colors, alpha=0.8)
            ax_pid_bar.set_title("Final PID Gains", fontweight='bold')
            ax_pid_bar.set_xlabel("Gain Index")
            ax_pid_bar.set_ylabel("Gain Value")
            ax_pid_bar.set_xticks(range(0, 20, 2))
            ax_pid_bar.grid(axis='y', alpha=0.3)

            # PID 历史热力图
            ax_pid_heat = fig.add_subplot(gs[2, 2])
            im = ax_pid_heat.imshow(
                pid_array.T,  # shape: (20, T)
                aspect='auto',
                origin='lower',
                cmap='viridis',
                interpolation='nearest',
                extent=[pid_t[0], pid_t[-1], 0, 20]
            )
            ax_pid_heat.set_title("PID Gains Heatmap", fontweight='bold')
            ax_pid_heat.set_xlabel("Time (s)")
            ax_pid_heat.set_ylabel("Gain Index")
            ax_pid_heat.set_yticks(range(0, 20, 2))
            cbar = fig.colorbar(im, ax=ax_pid_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Gain Value")

            # 关键 PID 增益波形（选择 4 个代表性增益）
            ax_pid_wave = fig.add_subplot(gs[2, 3])
            key_indices = [0, 2, 8, 11]  # posP_xy, velP_xy, attP_phi, rateP_p
            key_names = ["posP_xy", "velP_xy", "attP_phi", "rateP_p"]
            for idx, name in zip(key_indices, key_names):
                ax_pid_wave.plot(pid_t, pid_array[:, idx], lw=1.5, label=name)
            ax_pid_wave.set_title("Key PID Gains Over Time", fontweight='bold')
            ax_pid_wave.set_xlabel("Time (s)")
            ax_pid_wave.set_ylabel("Gain Value")
            ax_pid_wave.legend(loc='best', fontsize=8)
            ax_pid_wave.grid(alpha=0.3)
        else:
            # 如果没有 PID 数据，显示 3D 轨迹和统计信息
            ax_3d = fig.add_subplot(gs[2, :2], projection='3d')
            ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', lw=2, label='Actual')
            ax_3d.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], 'r--', lw=1.5, alpha=0.7, label='Reference')
            ax_3d.set_xlabel('X (m)')
            ax_3d.set_ylabel('Y (m)')
            ax_3d.set_zlabel('Z (m)')
            ax_3d.set_title("3D Trajectory", fontweight='bold')
            ax_3d.legend()
            ax_3d.grid(alpha=0.3)

            # 统计信息
            ax_stats = fig.add_subplot(gs[2, 2:])
            ax_stats.axis('off')
            
            # 计算 RMSE
            pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))
            vel_rmse = np.sqrt(np.mean(vel_err**2, axis=0))
            att_rmse = np.sqrt(np.mean(att_err**2, axis=0))
            
            stats_text = f"""
            Episode Statistics:
            ==================
            Total Steps: {end_idx}
            Episode Time: {t[-1]:.2f} s
            Episode Reward: {self.episode_reward:.2f}
            
            Position RMSE (m):
            X: {pos_rmse[0]:.4f}
            Y: {pos_rmse[1]:.4f}
            Z: {pos_rmse[2]:.4f}
            
            Velocity RMSE (m/s):
            Vx: {vel_rmse[0]:.4f}
            Vy: {vel_rmse[1]:.4f}
            Vz: {vel_rmse[2]:.4f}
            
            Attitude RMSE (deg):
            Roll:  {att_rmse[0]:.4f}
            Pitch: {att_rmse[1]:.4f}
            Yaw:   {att_rmse[2]:.4f}
            """
            ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                        verticalalignment='center', transform=ax_stats.transAxes)

        # ==================== 保存或显示 ====================
        if save_path:
            # 确保目录存在
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            log.info(f"✅ 图表已保存到: {save_path}")
            plt.close(fig)
        else:
            # 交互式显示
            plt.show()
        
        if mode == 'rgb_array':
            # 返回图像数组（用于视频制作）
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image

    def render_pid_gains(self, save_path=None):
        """
        离线绘制 PID 增益波形图（20 路参数）
        
        Args:
            save_path: 图片保存路径，如 'results/pid_gains.png'
                    None 则交互式显示
        """
        try:
            import matplotlib
            if save_path:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"⚠️ Matplotlib 不可用: {e}")
            return
        
        # 检查数据
        pid_history = getattr(self, "_pid_history", [])
        pid_time = getattr(self, "_pid_time_history", [])
        
        if len(pid_history) == 0:
            print("⚠️ 没有 PID 历史数据，跳过绘制")
            return
        
        pid_array = np.array(pid_history)  # shape: (T, 20)
        pid_t = np.array(pid_time)
        
        # PID 参数名称
        gain_names = [
            "posP_xy", "posP_z",
            "velP_xy", "velP_z",
            "velI_xy", "velI_z",
            "velD_xy", "velD_z",
            "attP_phi", "attP_theta", "attP_psi",
            "rateP_p", "rateP_q", "rateP_r",
            "rateI_p", "rateI_q", "rateI_r",
            "rateD_p", "rateD_q", "rateD_r",
        ]
        
        # ==================== 创建图表 ====================
        fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=True)
        fig.suptitle(
            f"PID Gains Evolution - {self.control_type} | Traj: {self.trajectory_type}\n"
            f"Total Steps: {len(pid_t)}, Time: {pid_t[-1]:.2f}s",
            fontsize=14, fontweight='bold'
        )
        
        # 绘制 20 路 PID 波形
        for i, (ax, name) in enumerate(zip(axes.flatten(), gain_names)):
            ax.plot(pid_t, pid_array[:, i], 'C0-', lw=1.5, alpha=0.8)
            ax.set_title(name, fontsize=10, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # 设置 Y 轴范围（如果有安全边界）
            if hasattr(self, "pid_safe_min") and hasattr(self, "pid_safe_max"):
                ymin = float(self.pid_safe_min[i])
                ymax = float(self.pid_safe_max[i])
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)
                
                # 绘制安全边界
                ax.axhline(ymin, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax.axhline(ymax, color='red', linestyle='--', linewidth=1, alpha=0.5)
            
            # 最后一行添加 X 轴标签
            if i >= 15:
                ax.set_xlabel("Time (s)", fontsize=9)
            
            # 第一列添加 Y 轴标签
            if i % 5 == 0:
                ax.set_ylabel("Gain Value", fontsize=9)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ PID 增益图已保存: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def render_pid_summary(self, save_path=None):
        """
        离线绘制 PID 增益摘要图（包含热力图、柱状图、关键波形）
        
        Args:
            save_path: 图片保存路径，如 'results/pid_summary.png'
        """
        try:
            import matplotlib
            if save_path:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"⚠️ Matplotlib 不可用: {e}")
            return
        
        # 检查数据
        pid_history = getattr(self, "_pid_history", [])
        pid_time = getattr(self, "_pid_time_history", [])
        
        if len(pid_history) == 0:
            print("⚠️ 没有 PID 历史数据，跳过绘制")
            return
        
        pid_array = np.array(pid_history)  # shape: (T, 20)
        pid_t = np.array(pid_time)
        
        gain_names = [
            "posP_xy", "posP_z", "velP_xy", "velP_z",
            "velI_xy", "velI_z", "velD_xy", "velD_z",
            "attP_phi", "attP_theta", "attP_psi",
            "rateP_p", "rateP_q", "rateP_r",
            "rateI_p", "rateI_q", "rateI_r",
            "rateD_p", "rateD_q", "rateD_r",
        ]
        
        # ==================== 创建图表 ====================
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(
            f"PID Gains Summary - {self.control_type} | Traj: {self.trajectory_type}",
            fontsize=14, fontweight='bold'
        )
        
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # ==================== 1. 热力图 ====================
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(
            pid_array.T,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest',
            extent=[pid_t[0], pid_t[-1], 0, 20]
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Gain Index")
        ax1.set_title("PID Gains Heatmap", fontweight='bold')
        ax1.set_yticks(range(0, 20, 2))
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Gain Value")
        
        # ==================== 2. 最终值柱状图 ====================
        ax2 = fig.add_subplot(gs[0, 1])
        pid_final = pid_array[-1, :]
        colors = plt.cm.viridis(np.linspace(0, 1, 20))
        ax2.bar(range(20), pid_final, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel("Gain Index")
        ax2.set_ylabel("Gain Value")
        ax2.set_title("Final PID Gains", fontweight='bold')
        ax2.set_xticks(range(0, 20, 2))
        ax2.grid(axis='y', alpha=0.3)
        
        # ==================== 3. 初始值 vs 最终值对比 ====================
        ax3 = fig.add_subplot(gs[0, 2])
        x_pos = np.arange(20)
        width = 0.35
        ax3.bar(x_pos - width/2, pid_array[0, :], width, label='Initial', alpha=0.7, color='steelblue')
        ax3.bar(x_pos + width/2, pid_array[-1, :], width, label='Final', alpha=0.7, color='orange')
        ax3.set_xlabel("Gain Index")
        ax3.set_ylabel("Gain Value")
        ax3.set_title("Initial vs Final PID Gains", fontweight='bold')
        ax3.set_xticks(range(0, 20, 2))
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # ==================== 4. 位置控制增益波形 ====================
        ax4 = fig.add_subplot(gs[1, 0])
        pos_indices = [0, 1, 2, 3]  # posP_xy, posP_z, velP_xy, velP_z
        pos_names = ["posP_xy", "posP_z", "velP_xy", "velP_z"]
        for idx, name in zip(pos_indices, pos_names):
            ax4.plot(pid_t, pid_array[:, idx], lw=1.5, label=name)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Gain Value")
        ax4.set_title("Position Control Gains", fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(alpha=0.3)
        
        # ==================== 5. 姿态控制增益波形 ====================
        ax5 = fig.add_subplot(gs[1, 1])
        att_indices = [8, 9, 10, 11]  # attP_phi, attP_theta, attP_psi, rateP_p
        att_names = ["attP_phi", "attP_theta", "attP_psi", "rateP_p"]
        for idx, name in zip(att_indices, att_names):
            ax5.plot(pid_t, pid_array[:, idx], lw=1.5, label=name)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Gain Value")
        ax5.set_title("Attitude Control Gains", fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(alpha=0.3)
        
        # ==================== 6. 增益变化率 ====================
        ax6 = fig.add_subplot(gs[1, 2])
        pid_diff = np.diff(pid_array, axis=0)  # 一阶差分
        pid_change = np.abs(pid_diff).sum(axis=1)  # 总变化量
        ax6.plot(pid_t[1:], pid_change, 'C0-', lw=1.5)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Total Change Rate")
        ax6.set_title("PID Gains Change Rate", fontweight='bold')
        ax6.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ PID 摘要图已保存: {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def render_pid_statistics(self, save_path=None):
        """
        离线绘制 PID 增益统计分析图
        
        Args:
            save_path: 图片保存路径
        """
        try:
            import matplotlib
            if save_path:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"⚠️ Matplotlib 不可用: {e}")
            return
        
        pid_history = getattr(self, "_pid_history", [])
        
        if len(pid_history) == 0:
            print("⚠️ 没有 PID 历史数据")
            return
        
        pid_array = np.array(pid_history)
        
        # 计算统计量
        pid_mean = np.mean(pid_array, axis=0)
        pid_std = np.std(pid_array, axis=0)
        pid_min = np.min(pid_array, axis=0)
        pid_max = np.max(pid_array, axis=0)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("PID Gains Statistical Analysis", fontsize=14, fontweight='bold')
        
        # 1. 均值 ± 标准差
        ax = axes[0, 0]
        x_pos = np.arange(20)
        ax.bar(x_pos, pid_mean, yerr=pid_std, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel("Gain Index")
        ax.set_ylabel("Mean ± Std")
        ax.set_title("Mean and Standard Deviation")
        ax.set_xticks(range(0, 20, 2))
        ax.grid(axis='y', alpha=0.3)
        
        # 2. 最小值 vs 最大值
        ax = axes[0, 1]
        width = 0.35
        ax.bar(x_pos - width/2, pid_min, width, label='Min', alpha=0.7, color='red')
        ax.bar(x_pos + width/2, pid_max, width, label='Max', alpha=0.7, color='green')
        ax.set_xlabel("Gain Index")
        ax.set_ylabel("Value")
        ax.set_title("Min vs Max")
        ax.set_xticks(range(0, 20, 2))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 3. 变化范围（Max - Min）
        ax = axes[1, 0]
        pid_range = pid_max - pid_min
        colors = ['red' if r > 0.5 else 'green' for r in pid_range]
        ax.bar(x_pos, pid_range, alpha=0.7, color=colors, edgecolor='black')
        ax.set_xlabel("Gain Index")
        ax.set_ylabel("Range (Max - Min)")
        ax.set_title("Variation Range")
        ax.set_xticks(range(0, 20, 2))
        ax.grid(axis='y', alpha=0.3)
        
        # 4. 变异系数（Coefficient of Variation）
        ax = axes[1, 1]
        pid_cv = pid_std / (pid_mean + 1e-8)  # 避免除零
        colors = ['red' if cv > 0.3 else 'green' for cv in pid_cv]
        ax.bar(x_pos, pid_cv, alpha=0.7, color=colors, edgecolor='black')
        ax.set_xlabel("Gain Index")
        ax.set_ylabel("CV (Std / Mean)")
        ax.set_title("Coefficient of Variation")
        ax.set_xticks(range(0, 20, 2))
        ax.axhline(0.3, color='orange', linestyle='--', linewidth=2, label='Threshold=0.3')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ PID 统计图已保存: {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def render_domain_switches(self, save_path=None):
        """
        绘制PID域切换历史
        
        Args:
            save_path: 图片保存路径
        """
        try:
            import matplotlib
            if save_path:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"⚠️ Matplotlib 不可用: {e}")
            return
        
        # 检查数据
        if not hasattr(self, 'episode_vel_des_all') or len(self.episode_vel_des_all) == 0:
            print("⚠️ 没有速度历史数据")
            return
        
        end_idx = int(self.current_step)
        if end_idx <= 1:
            print("⚠️ 没有足够的数据")
            return
        
        t = self.episode_t_all[:end_idx]
        vel_des = self.episode_vel_des_all[:end_idx]
        vel_mag = np.linalg.norm(vel_des, axis=1)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        fig.suptitle("PID Domain Switching History", fontsize=14, fontweight='bold')
        
        # 上图：速度大小
        ax1.plot(t, vel_mag, 'b-', lw=2, label='Reference Velocity')
        ax1.axhline(5.0, color='green', linestyle='--', lw=1.5, label='Low→Medium (5 m/s)')
        ax1.axhline(10.0, color='orange', linestyle='--', lw=1.5, label='Medium→High (10 m/s)')
        ax1.fill_between(t, 0, 5, alpha=0.2, color='green', label='Low Domain')
        ax1.fill_between(t, 5, 10, alpha=0.2, color='yellow', label='Medium Domain')
        ax1.fill_between(t, 10, 15, alpha=0.2, color='red', label='High Domain')
        ax1.set_ylabel("Velocity (m/s)", fontsize=12)
        ax1.set_title("Reference Velocity Magnitude", fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(0, max(15, vel_mag.max() * 1.1))
        
        # 下图：域标识
        domain_history = []
        for v in vel_mag:
            if v < 5.0:
                domain_history.append(0)  # low
            elif v < 10.0:
                domain_history.append(1)  # medium
            else:
                domain_history.append(2)  # high
        
        domain_colors = {0: 'green', 1: 'yellow', 2: 'red'}
        domain_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        
        for i in range(len(t)-1):
            ax2.fill_between(
                [t[i], t[i+1]], 
                0, 1, 
                color=domain_colors[domain_history[i]], 
                alpha=0.6
            )
        
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("PID Domain", fontsize=12)
        ax2.set_yticks([0.25, 0.5, 0.75])
        ax2.set_yticklabels(['Low', 'Medium', 'High'])
        ax2.set_title("Active PID Domain", fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(alpha=0.3, axis='x')
        
        # 统计域切换次数
        switches = 0
        for i in range(1, len(domain_history)):
            if domain_history[i] != domain_history[i-1]:
                switches += 1
        
        # 添加统计信息
        stats_text = f"Domain Switches: {switches}\n"
        stats_text += f"Low: {domain_history.count(0)/len(domain_history)*100:.1f}%\n"
        stats_text += f"Medium: {domain_history.count(1)/len(domain_history)*100:.1f}%\n"
        stats_text += f"High: {domain_history.count(2)/len(domain_history)*100:.1f}%"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 域切换历史图已保存: {save_path}")
            plt.close(fig)
        else:
            plt.show()
        
        # 打印详细统计
        print("\n" + "="*60)
        print("PID Domain Switching Statistics")
        print("="*60)
        print(f"Total domain switches: {switches}")
        print(f"Low domain time: {domain_history.count(0)/len(domain_history)*100:.2f}%")
        print(f"Medium domain time: {domain_history.count(1)/len(domain_history)*100:.2f}%")
        print(f"High domain time: {domain_history.count(2)/len(domain_history)*100:.2f}%")
        print(f"Average velocity: {vel_mag.mean():.2f} m/s")
        print(f"Max velocity: {vel_mag.max():.2f} m/s")
        print("="*60)