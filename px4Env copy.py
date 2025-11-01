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
                 #  姿态越界阈值课程学习参数
                 attitude_bound_start: float = 90.0,
                 attitude_bound_end: float = 60.0,
                 attitude_bound_decay_epochs: int = 1000,
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
        self.eval_mode = eval_mode          
        self.wind_type = wind_type
        self.wind_strength = wind_strength
        self.terminated_count = 0
        
        self.ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]

        # ==================== 动作空间模式（直接式 vs 增量式） ====================
        self.enable_smooth_transition = False
        self.alpha = 0.1
        
        # 奖励缩放因子
        self.reward_scale = 0.1
        self.failure_penalty = -10000.0   

        #  子步改进相关参数
        self.compat_truncated_as_done = bool(compat_truncated_as_done)
        
        #  根据控制模式自动调整位置和速度权重
        if self.control_type == "xyz_pos":
            pos_weight_xy = 3.0
            pos_weight_z = 2.0
            vel_weight_xy = 0.5
            vel_weight_z = 0.3
        elif self.control_type == "xy_vel_z_pos":
            pos_weight_xy = 0.0
            pos_weight_z = 2.0
            vel_weight_xy = 3.0
            vel_weight_z = 1.5
        elif self.control_type == "xyz_vel":
            pos_weight_xy = 0.0
            pos_weight_z = 0.0
            vel_weight_xy = 5.0
            vel_weight_z = 3.0
        else:
            log.warning(f" Unknown control_type: {self.control_type}, using default weights")
            pos_weight_xy = 0.0
            pos_weight_z = 0.0
            vel_weight_xy = 5.0
            vel_weight_z = 3.0

        # 🔄 修改：降低速度/姿态权重（单步误差波动更大）
        self.segment_weights = dict(
            # 跟踪误差项
            W_pos_xy=pos_weight_xy,
            W_pos_z=pos_weight_z,
            W_vel_xy=vel_weight_xy * 0.7,    # 🔄 降低到原来的 70%
            W_vel_z=vel_weight_z * 0.7,      # 🔄 降低到原来的 70%
            W_att=50.0,                       # 🔄 从 75.0 降低到 50.0
            W_omega=8.0,                      # 🔄 从 10.0 降低到 8.0
            
            # 惩罚项
            W_u=0.0,
            W_dv=0.0,
            W_domega=0.0,
            W_du=0.0,
            W_sat=0.0,
            W_over=2.0,
        )

        # ==================== 限制与判据 ====================
        self.limits = dict(
            w_cmd_min=0.0,
            w_cmd_max=2e3,
            overshoot_z_tol=0.2,
            normalize_rate_by_dt=True,
            overshoot_pos_tol=0.2,
            overshoot_vel_tol=0.2,
            overshoot_att_tol=0.2,
            overshoot_ang_vel_tol=0.1,
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

        # ==================== PID参数安全范围 ====================
        self.pid_safe_min = np.array([
            0.0, 0.1,
            1.2, 2.0,
            0.0, 0.2,
            0.1, 0.0,
            0.0, 0.0, 0.0,
            0.01, 0.01, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        self.pid_safe_max = np.array([
            2.0, 1.5,
            15.0, 15.0,
            60.0, 3.0,
            2.0, 2.0,
            12.0, 12.0, 5.0,
            0.5, 0.6, 0.6,
            10.0, 10.0, 10.0,
            0.01, 0.01, 0.01
        ], dtype=np.float32)

        # ==================== 初始化仿真组件 ====================
        self._initialize_simulation()

        # ==================== 基线PID参数 ====================
        # self.pid_K_prev_vec = np.array([
        #     1.0, 0.8,
        #     6.5, 8.5,
        #     10.0, 1.6,
        #     0.01, 1.0,
        #     8.0, 8.0, 2.75,
        #     0.5, 0.5, 0.305,
        #     3.0, 3.0, 5.0,
        #     0.008, 0.008, 0.005
        # ],dtype=np.float32)

        self.pid_K_prev_vec=(self.pid_safe_min + self.pid_safe_max)/2.0  # 使用中间值作为初始值

        self.init_actions = self.normalize_pid_parameters(self.pid_K_prev_vec)
        self.prev_action = self.init_actions
        self._apply_pid_parameters(self.init_actions)

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
        """初始化仿真组件"""
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
    # Gym API - reset
    # =========================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self._already_terminated = False

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.episode_reward = 0.0
        self.step_log_rows = []

        # 重置仿真组件
        self._initialize_simulation()
        log.info(f" reset() called: env_id={id(self)}, resetting to step 0, time={self.time:.3f}s")

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
    # 核心：直接式PID参数应用
    # ========================= 
    def _apply_pid_parameters(self, action: np.ndarray) -> dict:
        """直接式动作空间 + 可选平滑过渡"""
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
        """分层+分通道误差门控机制（12 个独立通道）"""
        obs_cur = self._get_observation(self.ctrl.sDesCalc)
        
        pos_err = obs_cur[24:27]
        vel_err = obs_cur[27:30]
        att_err = obs_cur[30:33]
        omega_err = obs_cur[33:36]

        thresh1 = 0.03
        thresh2 = 0.3
        limit_delta = 1.0

        def _gate_indices(err_val, idxs):
            if err_val < thresh1:
                action[idxs] = self.prev_action[idxs]
            elif thresh1 <= err_val < thresh2:
                delta = np.clip(action[idxs] - self.prev_action[idxs], 
                            -limit_delta, limit_delta)
                action[idxs] = self.prev_action[idxs] + delta
            else:
                pass

        # 位置环
        if self.control_type == "xyz_vel":
            pos_err_xy = 0
            pos_err_z = 0
        elif self.control_type == "xyz_pos":
            pos_err_xy = max(abs(pos_err[0]), abs(pos_err[1]))
            pos_err_z = abs(pos_err[2])
        elif self.control_type == "xy_vel_z_pos":
            pos_err_xy = 0
            pos_err_z = abs(pos_err[2])
        
        _gate_indices(pos_err_xy, [0])
        _gate_indices(pos_err_z, [1])

        # 速度环
        vel_err_xy = max(abs(vel_err[0]), abs(vel_err[1]))
        vel_err_z = abs(vel_err[2])
        
        _gate_indices(vel_err_xy, [2, 4, 6])
        _gate_indices(vel_err_z, [3, 5, 7])

        # 姿态环
        att_err_roll = abs(att_err[0])
        att_err_pitch = abs(att_err[1])
        att_err_yaw = abs(att_err[2])
        
        _gate_indices(att_err_roll, [8])
        _gate_indices(att_err_pitch, [9])
        _gate_indices(att_err_yaw, [10])

        # 角速度环
        omega_err_roll = abs(omega_err[0])
        omega_err_pitch = abs(omega_err[1])
        omega_err_yaw = abs(omega_err[2])
        
        _gate_indices(omega_err_roll, [11, 14, 17])
        _gate_indices(omega_err_pitch, [12, 15, 18])
        _gate_indices(omega_err_yaw, [13, 16, 19])

        return action

    def _log_reward_to_csv(self, reward_dict):
        """将奖励详情记录到CSV文件"""
        log_dir = "logs/reward_debug"
        os.makedirs(log_dir, exist_ok=True)
        
        if not hasattr(self, 'csv_log_file'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_log_file = os.path.join(log_dir, f"reward_log_{timestamp}.csv")
            self.csv_initialized = False
        
        row_data = {
            'step': getattr(self, 'step_count', 0),
            'episode': getattr(self, 'episode_count', 0),
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
        # ============================================================
        # 阶段1: 误差门控动作调整
        # ============================================================
        action = self._error_gated_action_adaptation(action)
        self.prev_action = np.asarray(action, dtype=np.float32).ravel().copy()

        # ============================================================
        # 阶段2: 应用 PID 参数
        # ============================================================
        self.step_action = self._apply_pid_parameters(action)

        # ============================================================
        # 阶段3: 单步仿真更新（🔄 替代原子步循环）
        # ============================================================
        # 初始化 prev_cache（用于计算变化率）
        prev_cache = {
            "vel": self.quad.vel.copy(),
            "omega": self.quad.omega.copy(),
            "w_cmd": self.ctrl.w_cmd.copy() if hasattr(self.ctrl, "w_cmd") and self.ctrl.w_cmd is not None else None
        }

        sDes = self.traj.desiredState(self.time, self.dt, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.dt)
        self.quad.update(self.time, self.dt, self.ctrl.w_cmd, self.wind)

        # 3e) 时间推进
        self.time += self.dt
        self.current_step += 1

        # 3f) 提取误差
        sDesclc = self.ctrl.sDesCalc
        m = self._extract_metrics(sDesclc)

        # 3g) 收集样本（单个样本）
        sample = self._collect_step_sample(m, prev_cache)

        # ============================================================
        # 阶段4：奖励计算
        # ============================================================
        segment_reward, reward_for_debug, metrics = self._compute_segment_reward_step([sample])

        # 统一奖励缩放
        reward = segment_reward * self.reward_scale
        self.step_reward = reward
        self.episode_reward += reward

        # 3h) 边界检查（立即终止）
        terminated = False
        if self._is_out_of_bounds():
            terminated = True
            self._already_terminated = True
            self.terminated_count += 1
            
            # 施加失败惩罚
            reward = self.failure_penalty * self.reward_scale
            self.episode_reward += reward
            
            obs = self._get_observation(self.ctrl.sDesCalc)
            info = {
                'termination_reason': 'out_of_bounds',
                'step': self.current_step,
                'env_id': id(self)
            }
            
            log.warning(
                f"🚨 Episode terminated at step {self.current_step}, "
                f"penalty: {self.failure_penalty} (scaled: {reward:.2f})"
            )
            
            return obs, reward, terminated, False, info

        # 终止条件判断
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
            }
            log.info(f"⏱️ Episode time limit reached at step {self.current_step}")
            obs = self._get_observation(self.ctrl.sDesCalc)
            return obs, reward, terminated, truncated, info

        # 记录调试信息
        if self.eval_mode:
            reward_for_debug['segment_reward'] = segment_reward
            reward_for_debug['reward_total_before_scale'] = segment_reward
            self._log_reward_to_csv(reward_for_debug)

        # 正常路径：返回本步结果
        obs = self._get_observation(self.ctrl.sDesCalc)
        info = {}
        return obs, reward, False, False, info

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
        """检测超调（三轴速度、姿态、角速度）"""
        overshoot_results = {
            "vel": [0, 0, 0],
            "att": [0, 0, 0],
            "ang_vel": [0, 0, 0]
        }
        
        vel_tol = self.limits["overshoot_vel_tol"]
        att_tol = self.limits["overshoot_att_tol"]
        ang_vel_tol = self.limits["overshoot_ang_vel_tol"]

        vel_err = [error["vel_err_x"], error["vel_err_y"], error["vel_err_z"]]
        att_err = [error["att_err_roll"], error["att_err_pitch"], error["att_err_yaw"]]
        ang_vel_err = [error["omega_err_p"], error["omega_err_q"], error["omega_err_r"]]

        overthrow_detected = False

        # 检测超调：三轴速度
        for i in range(3):
            current_vel = float(self.quad.vel[i])
            if vel_err[i] * current_vel < 0 and abs(vel_err[i]) > vel_tol:
                overshoot_results['vel'][i] = vel_err[i]
                overthrow_detected = True

        # 检测超调：三轴姿态
        for i in range(3):
            current_att = float(self.quad.euler[i])
            if att_err[i] * current_att < 0 and abs(att_err[i]) > att_tol:
                overshoot_results['att'][i] = att_err[i]
                overthrow_detected = True

        # 检测超调：三轴角速度
        for i in range(3):
            current_ang_vel = float(self.quad.omega[i])
            if ang_vel_err[i] * current_ang_vel < 0 and abs(ang_vel_err[i]) > ang_vel_tol:
                overshoot_results['ang_vel'][i] = ang_vel_err[i]
                overthrow_detected = True
        
        if isinstance(overshoot_results, dict):
            overshoot_hits_sum = sum(
                sum(1.0 for v in overshoot_results.get(key, []) if v != 0)
                for key in ['vel', 'att', 'ang_vel']
            )
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

    def _compute_segment_reward_step(self, samples: List[Dict[str, float]]) -> Tuple[float, Dict[str, float], Dict]:
        """
        计算当前步的奖励（🔄 统一时间步长后样本列表长度恒为1）
        """
        # 🔄 修改：断言样本数为 1
        assert len(samples) == 1, f"Expected 1 sample after time unification, got {len(samples)}"
        
        sample = samples[0]
        
        # 🔄 修改：直接使用误差平方根（单样本无需聚合）
        rmse_pos_xy = float(np.sqrt(sample["pos_xy2"]))
        rmse_pos_z = float(np.sqrt(sample["pos_z2"]))
        rmse_vel_xy = float(np.sqrt(sample["vel_xy2"]))
        rmse_vel_z = float(np.sqrt(sample["vel_z2"]))
        rmse_att = float(np.sqrt(sample["att2"]))
        rms_omega = float(np.sqrt(sample["omega2"]))

        mean_control_effort = float(sample["u2"])
        rms_dv = float(np.sqrt(sample["dv2"]))
        rms_domega = float(np.sqrt(sample["domega2"]))
        rms_du = float(np.sqrt(sample["du2"]))

        saturation_hits_norm = float(sample["sat_hit"])  # 🔄 单步无需归一化
        overshoot_hits_norm = float(sample["over_hit"])  # 🔄 单步无需归一化

        # 🔄 删除震荡检测（单步无法检测震荡）
        oscillation_penalty = 0.0

        # 奖励计算
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
                            oscillation_penalty)

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
            "reward_oscillation_penalty": oscillation_penalty,
        }

        metrics = {
            'segment_duration': float(self.dt),  # 🔄 单步时长
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

            'saturation_hits': saturation_hits_norm,
            'overshoot_hits': overshoot_hits_norm,
            'segment_reward': float(segment_reward),
            'samples': 1,  # 🔄 固定为 1
            
            'oscillation_penalty': oscillation_penalty,
        }

        return float(segment_reward), reward_for_debug, metrics

    # =========================
    #  构造观测向量
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

        # 误差信号
        pos_err = pos_ref - pos_fb
        vel_err = vel_ref - vel_fb
        eul_err = eul_ref - eul_fb
        eul_err = np.arctan2(np.sin(eul_err), np.cos(eul_err))
        omega_err = omega_ref - omega_fb
        
        # 误差归一化
        pos_err_norm = np.clip(pos_err / 10.0, -1, 1)
        vel_err_norm = np.clip(vel_err / 15.0, -1, 1)
        eul_err_norm = np.clip(eul_err / np.pi, -1, 1)
        omega_err_norm = np.clip(omega_err / (2*np.pi), -1, 1)
        
        if hasattr(self, 'pid_K_prev_vec') and self.pid_K_prev_vec is not None:
            pid_normalized = 2.0 * (self.pid_K_prev_vec - self.pid_safe_min) / (self.pid_safe_max - self.pid_safe_min + 1e-8) - 1.0
            pid_normalized = np.clip(pid_normalized, -1.0, 1.0)
        else:
            pid_normalized = np.zeros(20, dtype=np.float32)
                
        # 参考状态和反馈状态归一化
        pos_ref_norm = np.clip(pos_ref / 10.0, -1, 1)
        vel_ref_norm = np.clip(vel_ref / 15.0, -1, 1)
        eul_ref_norm = np.clip(eul_ref / np.pi, -1, 1)
        omega_ref_norm = np.clip(omega_ref / (2*np.pi), -1, 1)
        
        pos_fb_norm = np.clip(pos_fb / 10.0, -1, 1)
        vel_fb_norm = np.clip(vel_fb / 15.0, -1, 1)
        eul_fb_norm = np.clip(eul_fb / np.pi, -1, 1)
        omega_fb_norm = np.clip(omega_fb / (2*np.pi), -1, 1)
        
        # 拼接观测向量（56维）
        obs = np.concatenate([
            pos_ref_norm, vel_ref_norm, eul_ref_norm, omega_ref_norm,
            pos_fb_norm, vel_fb_norm, eul_fb_norm, omega_fb_norm,
            pos_err_norm, vel_err_norm, eul_err_norm, omega_err_norm,
            pid_normalized
        ])

        return obs.astype(np.float32)
    
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
    def render(self, mode='human'):
        """渲染环境（在线可视化）"""
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

            # 位置曲线
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
            self._hist_pid_t = []
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

        # 🔄 修改：使用 current_step 替代 current_substep
        end_idx = int(self.current_step)
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

        # 柱状图
        for i, rect in enumerate(self._act_bars):
            rect.set_height(float(pid_vec[i]))

        # 全程历史热力图
        self._hist_pid.append(pid_vec)
        self._hist_pid_t.append(float(self.time))
        pid_hist_arr = np.asarray(self._hist_pid)
        if pid_hist_arr.ndim == 2 and pid_hist_arr.shape[0] > 0:
            self._im_action.set_data(pid_hist_arr)
            vmin = float(np.min(self.pid_safe_min)) if hasattr(self, "pid_safe_min") else float(pid_hist_arr.min())
            vmax = float(np.max(self.pid_safe_max)) if hasattr(self, "pid_safe_max") else float(pid_hist_arr.max())
            if vmax <= vmin:
                vmax = vmin + 1.0
            self._im_action.set_clim(vmin, vmax)
            self._ax_act_hist.set_ylim(0, max(1, pid_hist_arr.shape[0]-1))
            self._ax_act_hist.set_xlim(0, 19)

        # 懒加载初始化（20 路增益波形窗口）
        if not hasattr(self, "_gains_initialized") or not getattr(self, "_gains_initialized", False):
            self._gains_fig, axes = plt.subplots(4, 5, figsize=(14, 10), sharex=True)
            self._gains_fig.suptitle("PID Gains Waveforms (full)", fontsize=12)
            self._gain_axes = axes.flatten().tolist()
            self._gain_lines = []

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

            for i, ax in enumerate(self._gain_axes):
                (ln,) = ax.plot([], [], 'C0-', lw=1.2)
                self._gain_lines.append(ln)
                title = self._gain_names[i] if i < len(self._gain_names) else f"K[{i}]"
                ax.set_title(title, fontsize=9)
                if hasattr(self, "pid_safe_min") and hasattr(self, "pid_safe_max") and len(self.pid_safe_min) == 20:
                    ymin = float(self.pid_safe_min[i]); ymax = float(self.pid_safe_max[i])
                    if ymax <= ymin:
                        ymax = ymin + 1.0
                    pad = 0.08 * (ymax - ymin)
                    ax.set_ylim(ymin - pad, ymax + pad)
                ax.grid(alpha=0.3)
                if i >= 15:
                    ax.set_xlabel("t (s)")
            plt.tight_layout()
            self._gains_initialized = True

        # 更新 20 路增益波形
        if hasattr(self, "_gains_initialized") and self._gains_initialized:
            t_gain = np.asarray(self._hist_pid_t, dtype=float) if len(self._hist_pid_t) > 0 else np.arange(pid_hist_arr.shape[0])
            for i, ln in enumerate(self._gain_lines):
                ln.set_data(t_gain, pid_hist_arr[:, i])
                ax = self._gain_axes[i]
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
            plt.pause(0.005)
        except Exception:
            pass