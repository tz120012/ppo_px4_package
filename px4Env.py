
# -*- coding: utf-8 -*-
"""
作者: John Bass, 修改为Gym环境接口
邮箱: john.bobzwik@gmail.com
许可证: MIT
请随意使用和修改，但请保留上述信息。谢谢！
"""

import numpy as np
from numpy import sin, cos, tan, pi, sign, sqrt
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Union, Any

from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils
import config

import logging
# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("PX4Env")


class QuadcopterEnv(gym.Env):
    """
    PX4四旋翼仿真环境用于强化学习

    一个Gym兼容的环境，用于训练四旋翼控制任务的RL智能体。
    基于PX4控制算法和动力学。
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 control_type: str = "xyz_pos",
                 trajectory_type: int = 5,
                 yaw_type: int = 3,
                 max_episode_steps: int = 4000,     # 每个回合的最大步数（以RL步计）
                 dt: float = 0.005,        # 仿真积分时间步长（动力学/控制）
                 rl_dt: float = 0.1,       # RL决策时间步长（每次step保持action的时长）
                 segment_window: float = 3.0,  # 分段累计窗口长度（秒）
                 target_pos: Optional[np.ndarray] = None,
                 wind_type: str = 'None',
                 wind_strength: float = 2.0,
                 render_mode: Optional[str] = None):
        """
        初始化四旋翼环境

        参数:
            control_type: 控制类型 ("xyz_pos", "xy_vel_z_pos", "xyz_vel")
            trajectory_type: 轨迹选择 (0-13)
            yaw_type: 偏航控制类型 (0-4)
            max_episode_steps: 每个回合的最大步数（以RL步计）
            rl_dt: RL 决策步长（秒），一个step保持action的时长
            segment_window: 分段累计奖励窗口时长（秒），默认 5s
            dt: 仿真时间步长（秒）
            target_pos: 目标位置 [x, y, z]
            wind_type: 风模型类型
            wind_strength: 风力强度
            render_mode: 渲染模式 ('human' 或 'rgb_array')
        """
        super(QuadcopterEnv, self).__init__()

        # 环境参数
        self.control_type = control_type
        self.trajectory_type = trajectory_type
        self.yaw_type = yaw_type
        self.max_episode_steps = max_episode_steps
        self.rl_dt = rl_dt
        self.segment_window = segment_window
        self.dt = dt
        self.target_pos = target_pos if target_pos is not None else np.array([2.0, 2.0, 1.0])
        self.wind_type = wind_type
        self.wind_strength = wind_strength
        self.render_mode = render_mode

        # 回合跟踪（以RL步为单位）
        self.current_step = 0
        self.episode_reward = 0.0

        # 控制和轨迹选项
        self.ctrl_options = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]

        # 初始化仿真组件
        self._initialize_simulation()

        # PID参数安全范围（从归一化动作映射）
        self.pid_safe_min = np.array([
            0.0, 0.1,   # pos_P_gains: Pxy, Pz (注意我们会复制到X/Y)
            1.2, 2.0,   # vel_P_gains: Pxydot, Pzdot
            0.0, 0.2,   # vel_I_gains: Ixydot, Izdot
            0.1, 0.0,   # vel_D_gains: Dxydot, Dzdot
            0.0, 0.0, 0.0,  # att_P_gains: Pphi, Ptheta, Ppsi
            0.01, 0.01, 0.0,  # rate_P_gains: Pp, Pq, Pr
            0.0, 0.0, 0.0,  # rate_I_gains: Ip, Iq, Ir
            0.0, 0.0, 0.0   # rate_D_gains: Dp, Dq, Dr
        ])

        self.pid_safe_max = np.array([
            2.0, 1.5,     # pos_P_gains
            5.0, 15.0,   # vel_P_gains
            60.0, 3.0,   # vel_I_gains
            2.0, 2.0,     # vel_D_gains
            12.0, 12.0, 5.0,  # att_P_gains
            0.5, 0.6, 0.6,  # rate_P_gains
            10.0, 10.0, 10.0,    # rate_I_gains
            0.01, 0.01, 0.01     # rate_D_gains
        ])

        # 定义动作空间（所有PID参数值）
        # 参数顺序: [0-7]位置P/速度PID, [8-10]姿态P, [11-19]角速率PID
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(20,), dtype=np.float32
        )

        # 定义观测空间
        # 观测: [位置参考(3), 速度参考(3), 欧拉姿态参考(3), 角速度参考(3),
        #        位置反馈(3), 速度反馈(3), 欧拉姿态反馈(3), 角速度反馈(3)]
        obs_dim = 24
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 渲染
        self.fig = None
        self.ax = None
        self._ax_initialized = False

        # 分段统计累积器初始化
        self._reset_segment_accumulators()

    # =========================
    # 仿真初始化与内部工具
    # =========================
    def _initialize_simulation(self):
        """初始化仿真组件"""
        self.time = 0.0
        self.quad = Quadcopter(self.time)
        traj_select = np.array([self.trajectory_type, self.yaw_type, 1])  # 使用平均速度
        self.traj = Trajectory(self.quad, ctrlType=self.control_type, trajSelect=traj_select)
        self.ctrl = Control(self.quad, self.traj.yawType)
        self.wind = Wind(self.wind_type, self.wind_strength, 90, -15)

        sDes = self.traj.desiredState(0, self.dt, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.dt)

        self.pos_history = []
        self.target_history = []

    def _reset_segment_accumulators(self):
        self.segment_time = 0.0
        self.seg_samples = 0
        # 分轴误差
        self.seg_sum_pos_err_xy2 = 0.0
        self.seg_sum_pos_err_z2  = 0.0
        self.seg_sum_vel_err_xy2 = 0.0
        self.seg_sum_vel_err_z2  = 0.0
        # 姿态/角速度
        self.seg_sum_att_roll2 = 0.0
        self.seg_sum_att_pitch2 = 0.0
        self.seg_sum_omega2 = 0.0
        # 能量/平滑性
        self.seg_sum_control_effort = 0.0
        self.seg_sum_dv2 = 0.0
        self.seg_sum_domega2 = 0.0
        self.seg_sum_du2 = 0.0
        # 约束计数
        self.seg_saturation_hits = 0
        self.seg_overshoot_hits = 0
        # 上一时刻缓存（用于差分）
        self._seg_prev_vel = None
        self._seg_prev_omega = None
        self._seg_prev_w_cmd = None
        # 最近一次指标
        self.last_segment_metrics = None

    # =========================
    # Gym API
    # =========================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.episode_reward = 0.0

        if options is not None and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'])

        self._initialize_simulation()
        self._reset_segment_accumulators()

        if options and options.get('random_init', False):
            noise = np.random.normal(0, 0.1, 3)
            self.quad.pos += noise
            self.quad.state[0:3] = self.quad.pos

        self.pos_history = [self.quad.pos.copy()]
        self.target_history = [self.target_pos.copy()]

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一个环境步骤：保持该action在rl_dt时长内，包含多个仿真子步"""
        log.info(f"Step {self.current_step}: action={action}")
        action = np.clip(action, 0.0, 1.0)          # Action参数剪切处理
        self._apply_pid_parameters(action)          # 将参数写入PID控制器

        # 需要运行的仿真子步数量
        substeps = max(1, int(round(self.rl_dt / self.dt)))     # 计算每个Step 需要多少个仿真子步
        step_reward_dense_sum = 0.0                             # 聚合该RL step内的即时奖励（dense）

        # 在每个仿真步（步长：0.005s，200Hz）
        for _ in range(substeps):
            # 目标/控制/动力学推进
            sDes = self.traj.desiredState(self.time, self.dt, self.quad)        # 期望轨迹
            self.ctrl.controller(self.traj, self.quad, sDes, self.dt)           # 控制器更新
            self.quad.update(self.time, self.dt, self.ctrl.w_cmd, self.wind)    # 动力学更新

            # 时间推进与采样
            self.time += self.dt

            # 分段统计累积（每个仿真子步采样一次）
            self._accumulate_segment_metrics(sDes)

            # 即时奖励（dense，细粒度，引导稳定）
            step_reward_dense_sum += self._dense_reward_instant()

        # RL步计数
        self.current_step += 1

        # 分段奖励：当累计时长>=segment_window时触发一次
        segment_reward = 0.0
        if self.segment_time >= self.segment_window:
            segment_reward, metrics = self._compute_segment_reward_and_reset()
            self.last_segment_metrics = metrics

        # step奖励 = 平均即时dense + 分段奖励（若触发）
        reward = (step_reward_dense_sum / substeps) + segment_reward
        self.episode_reward += reward

        # 终止/截断判定
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps

        # 历史
        self.pos_history.append(self.quad.pos.copy())
        self.target_history.append(self.target_pos.copy())

        obs = self._get_observation()
        info = self._get_info()
        # 在info中包含本次分段奖励与指标（如果触发）
        if self.last_segment_metrics is not None and 'last_segment' not in info:
            info['last_segment'] = self.last_segment_metrics

        return obs, reward, terminated, truncated, info

    # =========================
    # 参数应用与仿真推进
    # =========================
    def _apply_pid_parameters(self, action: np.ndarray):
        """将归一化的动作转换为PID参数并应用到控制系统"""
        pos_P_gains_from_action = self.pid_safe_min[0:2] + (self.pid_safe_max[0:2] - self.pid_safe_min[0:2]) * action[0:2]
        vel_P_gains_from_action = self.pid_safe_min[2:4] + (self.pid_safe_max[2:4] - self.pid_safe_min[2:4]) * action[2:4]
        vel_I_gains_from_action = self.pid_safe_min[4:6] + (self.pid_safe_max[4:6] - self.pid_safe_min[4:6]) * action[4:6]
        vel_D_gains_from_action = self.pid_safe_min[6:8] + (self.pid_safe_max[6:8] - self.pid_safe_min[6:8]) * action[6:8]
        att_P_gains_from_action = self.pid_safe_min[8:11] + (self.pid_safe_max[8:11] - self.pid_safe_min[8:11]) * action[8:11]
        rate_P_gains_from_action = self.pid_safe_min[11:14] + (self.pid_safe_max[11:14] - self.pid_safe_min[11:14]) * action[11:14]
        rate_I_gains_from_action = self.pid_safe_min[14:17] + (self.pid_safe_max[14:17] - self.pid_safe_min[14:17]) * action[14:17]
        rate_D_gains_from_action = self.pid_safe_min[17:20] + (self.pid_safe_max[17:20] - self.pid_safe_min[17:20]) * action[17:20]

        # 将XY共用参数复制展开为三轴
        pos_P_gains = (pos_P_gains_from_action[0], pos_P_gains_from_action[0], pos_P_gains_from_action[1])
        vel_P_gains = (vel_P_gains_from_action[0], vel_P_gains_from_action[0], vel_P_gains_from_action[1])
        vel_I_gains = (vel_I_gains_from_action[0], vel_I_gains_from_action[0], vel_I_gains_from_action[1])
        vel_D_gains = (vel_D_gains_from_action[0], vel_D_gains_from_action[0], vel_D_gains_from_action[1])
        att_P_gains = att_P_gains_from_action
        rate_P_gains = rate_P_gains_from_action
        rate_I_gains = rate_I_gains_from_action
        rate_D_gains = rate_D_gains_from_action

        gains_from_action = {
            'pos_P_gains': pos_P_gains,
            'vel_P_gains': vel_P_gains,
            'vel_I_gains': vel_I_gains,
            'vel_D_gains': vel_D_gains,
            'att_P_gains': att_P_gains,
            'rate_P_gains': rate_P_gains,
            'rate_I_gains': rate_I_gains,
            'rate_D_gains': rate_D_gains,
        }
        self.ctrl.update_gains(gains_from_action)

    # =========================
    # 观测
    # =========================
    def _get_observation(self) -> np.ndarray:
        sDes = self.traj.desiredState(self.time, self.dt, self.quad)
        pos_ref = sDes[0:3]
        vel_ref = sDes[3:6]
        eul_ref = sDes[12:15]
        omega_ref = sDes[15:18]

        pos_fb = self.quad.pos
        vel_fb = self.quad.vel
        eul_fb = self._quaternion_to_euler(self.quad.quat)
        omega_fb = self.quad.omega

        obs = np.concatenate([
            pos_ref, vel_ref, eul_ref, omega_ref,
            pos_fb, vel_fb, eul_fb, omega_fb
        ])
        return obs.astype(np.float32)

    def _euler_to_quaternion(self, euler):
        phi, theta, psi = euler[0], euler[1], euler[2]
        cos_phi_2 = np.cos(phi / 2.0)
        sin_phi_2 = np.sin(phi / 2.0)
        cos_theta_2 = np.cos(theta / 2.0)
        sin_theta_2 = np.sin(theta / 2.0)
        cos_psi_2 = np.cos(psi / 2.0)
        sin_psi_2 = np.sin(psi / 2.0)

        w = cos_phi_2 * cos_theta_2 * cos_psi_2 + sin_phi_2 * sin_theta_2 * sin_psi_2
        x = sin_phi_2 * cos_theta_2 * cos_psi_2 - cos_phi_2 * sin_theta_2 * sin_psi_2
        y = cos_phi_2 * sin_theta_2 * cos_psi_2 + sin_phi_2 * cos_theta_2 * sin_psi_2
        z = cos_phi_2 * cos_theta_2 * sin_psi_2 - sin_phi_2 * sin_theta_2 * cos_psi_2
        return np.array([w, x, y, z])

    def _quaternion_to_euler(self, quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        phi = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        theta = np.arcsin(2.0 * (w * y - z * x))
        psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return np.array([phi, theta, psi])

    # =========================
    # 奖励设计
    # =========================
    def _dense_reward_instant(self) -> float:
        """
        即时（密集）奖励：每个仿真子步计算。
        - 分轴位置误差（XY、Z分权）
        - 速度跟踪误差（相对 v_ref）
        - 姿态误差（phi、theta 分开）
        - 角速度范数
        - 轻微的平滑性（动作/状态变化率）可选
        """
        # 从轨迹取参考
        sDes = self.traj.desiredState(self.time, self.dt, self.quad)
        pos_ref = sDes[0:3]
        vel_ref = sDes[3:6]
        # eul_ref = sDes[12:15]  # 若需要姿态参考，这里可用

        # 分轴位置误差
        pos_err_vec = self.quad.pos - pos_ref
        ex, ey, ez = pos_err_vec[0], pos_err_vec[1], pos_err_vec[2]
        pos_err_xy = np.sqrt(ex**2 + ey**2)
        pos_err_z = np.abs(ez)

        # 速度跟踪误差（相对参考）
        vel_err_vec = self.quad.vel - vel_ref
        vel_err_xy = np.linalg.norm(vel_err_vec[:2])
        vel_err_z = np.abs(vel_err_vec[2])

        # 姿态与角速度（分轴/总体）
        phi, theta, psi = self.quad.euler  # 偏航误差可在轨迹需要时加入
        att_err_roll = np.abs(phi)
        att_err_pitch = np.abs(theta)
        omega = self.quad.omega
        omega_norm = np.linalg.norm(omega)

        # 平滑性（局部项，可选）：状态变化率与控制变化率
        # 需要缓存上一时刻值，可在 env.__init__ 中初始化 self._prev_vel, self._prev_omega, self._prev_w_cmd
        dv = 0.0
        domega = 0.0
        du = 0.0
        if hasattr(self, "_prev_vel") and self._prev_vel is not None:
            dv = float(np.linalg.norm(self.quad.vel - self._prev_vel))
        if hasattr(self, "_prev_omega") and self._prev_omega is not None:
            domega = float(np.linalg.norm(self.quad.omega - self._prev_omega))
        if hasattr(self.ctrl, "w_cmd"):
            if not hasattr(self, "_prev_w_cmd") or self._prev_w_cmd is None:
                self._prev_w_cmd = np.copy(self.ctrl.w_cmd)
            du = float(np.linalg.norm(self.ctrl.w_cmd - self._prev_w_cmd))
            self._prev_w_cmd = np.copy(self.ctrl.w_cmd)
        self._prev_vel = np.copy(self.quad.vel)
        self._prev_omega = np.copy(self.quad.omega)

        # 权重（保守型）
        w_pos_xy = 0.8
        w_pos_z = 1.2
        w_vel_xy = 0.10
        w_vel_z = 0.12
        w_att_roll = 0.15
        w_att_pitch = 0.15
        w_omega = 0.05
        # 平滑性局部权重（非常小，避免盖过主项）
        w_dv = 0.01
        w_domega = 0.005
        w_du = 1e-6

        dense = -(
            w_pos_xy * pos_err_xy +
            w_pos_z * pos_err_z +
            w_vel_xy * vel_err_xy +
            w_vel_z * vel_err_z +
            w_att_roll * att_err_roll +
            w_att_pitch * att_err_pitch +
            w_omega * omega_norm +
            w_dv * dv + w_domega * domega + w_du * du
        )
        return float(dense)

    def _accumulate_segment_metrics(self, sDes):
        """在每个仿真子步累积 segment 窗口统计量（分轴 + 总体 + 平滑性）"""
        pos_ref = sDes[0:3]
        vel_ref = sDes[3:6]

        # 分轴位置误差
        pos_err = self.quad.pos - pos_ref
        ex, ey, ez = pos_err[0], pos_err[1], pos_err[2]
        pos_err_xy = np.sqrt(ex**2 + ey**2)
        pos_err_z = np.abs(ez)

        # 速度跟踪误差
        vel_err = self.quad.vel - vel_ref
        vel_err_xy = np.linalg.norm(vel_err[:2])
        vel_err_z = np.abs(vel_err[2])

        # 姿态与角速度分轴
        phi, theta, psi = self.quad.euler
        att_err_roll = np.abs(phi)
        att_err_pitch = np.abs(theta)
        omega = self.quad.omega
        omega_norm = np.linalg.norm(omega)

        # 控制能量
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            control_effort = float(np.sum(np.square(self.ctrl.w_cmd)))
        else:
            control_effort = 0.0

        # 平滑性（时间变化率）——对 segment 用平方累积，更稳
        dv = 0.0
        domega = 0.0
        du = 0.0
        if hasattr(self, "_seg_prev_vel") and self._seg_prev_vel is not None:
            dv = float(np.linalg.norm(self.quad.vel - self._seg_prev_vel))
        if hasattr(self, "_seg_prev_omega") and self._seg_prev_omega is not None:
            domega = float(np.linalg.norm(self.quad.omega - self._seg_prev_omega))
        if hasattr(self.ctrl, "w_cmd"):
            if not hasattr(self, "_seg_prev_w_cmd") or self._seg_prev_w_cmd is None:
                self._seg_prev_w_cmd = np.copy(self.ctrl.w_cmd)
            du = float(np.linalg.norm(self.ctrl.w_cmd - self._seg_prev_w_cmd))
            self._seg_prev_w_cmd = np.copy(self.ctrl.w_cmd)
        self._seg_prev_vel = np.copy(self.quad.vel)
        self._seg_prev_omega = np.copy(self.quad.omega)

        # 饱和与简单超调判据
        saturation_hit = 0
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            if np.any(self.ctrl.w_cmd < 0.0) or np.any(self.ctrl.w_cmd > 2e4):
                saturation_hit = 1

        overshoot_hit = 1 if (self.quad.pos[2] - self.target_pos[2]) * np.sign(self.quad.vel[2]) > 0 and abs(self.quad.pos[2] - self.target_pos[2]) > 0.2 else 0

        # 分段累积（平方和/总和）
        # 位置与速度分轴
        self.seg_sum_pos_err_xy2 = getattr(self, "seg_sum_pos_err_xy2", 0.0) + pos_err_xy**2
        self.seg_sum_pos_err_z2  = getattr(self, "seg_sum_pos_err_z2", 0.0)  + pos_err_z**2
        self.seg_sum_vel_err_xy2 = getattr(self, "seg_sum_vel_err_xy2", 0.0) + vel_err_xy**2
        self.seg_sum_vel_err_z2  = getattr(self, "seg_sum_vel_err_z2", 0.0)  + vel_err_z**2

        # 姿态与角速度
        self.seg_sum_att_roll2   = getattr(self, "seg_sum_att_roll2", 0.0)   + att_err_roll**2
        self.seg_sum_att_pitch2  = getattr(self, "seg_sum_att_pitch2", 0.0)  + att_err_pitch**2
        self.seg_sum_omega2      = getattr(self, "seg_sum_omega2", 0.0)      + omega_norm**2

        # 能量与平滑性
        self.seg_sum_control_effort = getattr(self, "seg_sum_control_effort", 0.0) + control_effort
        self.seg_sum_dv2        = getattr(self, "seg_sum_dv2", 0.0) + dv**2
        self.seg_sum_domega2    = getattr(self, "seg_sum_domega2", 0.0) + domega**2
        self.seg_sum_du2        = getattr(self, "seg_sum_du2", 0.0) + du**2

        # 计数项
        self.seg_saturation_hits = getattr(self, "seg_saturation_hits", 0) + saturation_hit
        self.seg_overshoot_hits  = getattr(self, "seg_overshoot_hits", 0)  + overshoot_hit
        self.seg_samples += 1
        self.segment_time += self.dt

    def _compute_segment_reward_and_reset(self) -> Tuple[float, Dict[str, float]]:
        """计算窗口奖励（含分轴与总体指标），然后重置累积器"""
        if self.seg_samples <= 0:
            return 0.0, {}

        # 分轴RMSE
        rmse_pos_xy = np.sqrt(self.seg_sum_pos_err_xy2 / self.seg_samples)
        rmse_pos_z  = np.sqrt(self.seg_sum_pos_err_z2  / self.seg_samples)
        rmse_vel_xy = np.sqrt(self.seg_sum_vel_err_xy2 / self.seg_samples)
        rmse_vel_z  = np.sqrt(self.seg_sum_vel_err_z2  / self.seg_samples)
        rmse_att_roll  = np.sqrt(self.seg_sum_att_roll2  / self.seg_samples)
        rmse_att_pitch = np.sqrt(self.seg_sum_att_pitch2 / self.seg_samples)
        rms_omega   = np.sqrt(self.seg_sum_omega2 / self.seg_samples)

        mean_control_effort = self.seg_sum_control_effort / self.seg_samples
        rms_dv     = np.sqrt(self.seg_sum_dv2 / self.seg_samples)
        rms_domega = np.sqrt(self.seg_sum_domega2 / self.seg_samples)
        rms_du     = np.sqrt(self.seg_sum_du2 / self.seg_samples)

        # 总体指标（可选，用于日志或辅助）
        rmse_pos_total = np.sqrt((self.seg_sum_pos_err_xy2 + self.seg_sum_pos_err_z2) / self.seg_samples)
        rmse_vel_total = np.sqrt((self.seg_sum_vel_err_xy2 + self.seg_sum_vel_err_z2) / self.seg_samples)

        # 权重（保守型，轨迹任务）
        W_pos_xy = 4.0
        W_pos_z  = 6.0
        W_vel_xy = 1.0
        W_vel_z  = 1.2
        W_att_roll  = 1.2
        W_att_pitch = 1.2
        W_omega  = 0.6
        W_u      = 2e-7         # 能量尺度大，系数很小
        W_dv     = 0.05         # 短期平滑性
        W_domega = 0.02
        W_du     = 2e-6
        W_sat    = 3.0
        W_over   = 1.5

        segment_reward = -(
            W_pos_xy * rmse_pos_xy +
            W_pos_z  * rmse_pos_z  +
            W_vel_xy * rmse_vel_xy +
            W_vel_z  * rmse_vel_z  +
            W_att_roll  * rmse_att_roll +
            W_att_pitch * rmse_att_pitch +
            W_omega  * rms_omega +
            W_u      * mean_control_effort +
            W_dv     * rms_dv +
            W_domega * rms_domega +
            W_du     * rms_du +
            W_sat    * self.seg_saturation_hits +
            W_over   * self.seg_overshoot_hits
        )

        metrics = {
            'segment_duration': self.segment_time,
            # 分轴指标
            'rmse_pos_xy': float(rmse_pos_xy),
            'rmse_pos_z': float(rmse_pos_z),
            'rmse_vel_xy': float(rmse_vel_xy),
            'rmse_vel_z': float(rmse_vel_z),
            'rmse_att_roll': float(rmse_att_roll),
            'rmse_att_pitch': float(rmse_att_pitch),
            'rms_omega': float(rms_omega),
            # 总体指标（便于汇总/可视化）
            'rmse_pos_total': float(rmse_pos_total),
            'rmse_vel_total': float(rmse_vel_total),
            # 平滑与能量
            'mean_control_effort': float(mean_control_effort),
            'rms_dv': float(rms_dv),
            'rms_domega': float(rms_domega),
            'rms_du': float(rms_du),
            # 约束计数
            'saturation_hits': int(self.seg_saturation_hits),
            'overshoot_hits': int(self.seg_overshoot_hits),
            # 奖励
            'segment_reward': float(segment_reward),
            'samples': int(self.seg_samples),
        }

        overflow = self.segment_time - self.segment_window
        self._reset_segment_accumulators()
        self.segment_time = max(0.0, overflow)

        return float(segment_reward), metrics

    # =========================
    # 终止与信息
    # =========================
    def _is_terminated(self) -> bool:
        if self.quad.pos[2] > 10.0 or self.quad.pos[2] < -10.0:
            return True
        if np.abs(self.quad.pos[0]) > 20.0 or np.abs(self.quad.pos[1]) > 20.0:
            return True
        if np.abs(self.quad.euler[0]) > np.pi/2 or np.abs(self.quad.euler[1]) > np.pi/2:
            return True

        pos_error = np.linalg.norm(self.quad.pos - self.target_pos)
        vel_magnitude = np.linalg.norm(self.quad.vel)
        if pos_error < 0.1 and vel_magnitude < 0.5:
            return True
        return False

    def _get_info(self) -> Dict:
        pos_error = np.linalg.norm(self.quad.pos - self.target_pos)
        info = {
            'position': self.quad.pos.copy(),
            'velocity': self.quad.vel.copy(),
            'attitude': self.quad.euler.copy(),
            'target_position': self.target_pos.copy(),
            'position_error': pos_error,
            'episode_reward': self.episode_reward,
            'time': self.time,
            'segment_time_accum': self.segment_time,
        }
        if self.last_segment_metrics is not None:
            info['last_segment'] = self.last_segment_metrics
        return info

    # =========================
    # 渲染与关闭
    # =========================
    def render(self, mode: str = 'human'):
        try:
            if mode == 'human':
                if self.fig is None:
                    self.fig = plt.figure(figsize=(10, 8))
                    self.ax = self.fig.add_subplot(111, projection='3d')
                    plt.ion()

                if self.ax is not None:
                    self.ax.clear()
                    pos = self.quad.pos
                    self.ax.scatter(pos[0], pos[1], pos[2], color='red', s=100, label='Quadcopter')  # type: ignore

                    target = self.target_pos
                    self.ax.scatter(target[0], target[1], target[2],
                                    color='green', s=100, marker='*', label='Target')  # type: ignore

                    if len(self.pos_history) > 1:
                        history = np.array(self.pos_history)
                        self.ax.plot(history[:, 0], history[:, 1], history[:, 2],
                                     'b-', alpha=0.6, label='Trajectory')

                    self.ax.set_xlabel('X (m)')
                    self.ax.set_ylabel('Y (m)')
                    self.ax.set_zlabel('Z (m)')  # type: ignore
                    self.ax.legend()

                    all_pos = np.array(self.pos_history + [self.target_pos])
                    margin = 2.0
                    x_min, x_max = float(all_pos[:, 0].min() - margin), float(all_pos[:, 0].max() + margin)
                    y_min, y_max = float(all_pos[:, 1].min() - margin), float(all_pos[:, 1].max() + margin)
                    z_min, z_max = float(all_pos[:, 2].min() - margin), float(all_pos[:, 2].max() + margin)

                    self.ax.set_xlim(x_min, x_max)
                    self.ax.set_ylim(y_min, y_max)
                    self.ax.set_zlim(z_min, z_max)  # type: ignore

                    title_seg = ""
                    if self.last_segment_metrics is not None:
                        sr = self.last_segment_metrics.get('segment_reward', 0.0)
                        rmse_pos = self.last_segment_metrics.get('rmse_pos', 0.0)
                        title_seg = f" | segR: {sr:.2f} RMSEpos:{rmse_pos:.2f}"

                    plt.title(f'Quadcopter - RL Step: {self.current_step}, EpR: {self.episode_reward:.2f}{title_seg}')
                    plt.pause(0.01)

            elif mode == 'rgb_array':
                return np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Render error: {e}")
            pass

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    # 创建四旋翼环境
    env = QuadcopterEnv(
        control_type="xyz_pos",
        trajectory_type=5,
        max_episode_steps=2000,  # 以RL步计
        dt=0.005,                # 仿真步长
        rl_dt=0.1,               # RL决策步长：每步保持0.1s
        segment_window=5.0,      # 分段统计窗口5s
        target_pos=np.array([2.0, 2.0, 1.0]),
        render_mode='human'
    )

    print("=== 四旋翼环境信息 ===")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"最大步数(RL): {env.max_episode_steps}")
    print(f"仿真步长 dt: {env.dt}s, RL决策步长 rl_dt: {env.rl_dt}s, Segment: {env.segment_window}s")
    print(f"目标位置: {env.target_pos}")

    obs, info = env.reset()
    print(f"\n初始观测维度: {obs.shape}")
    print(f"初始位置: {info['position']}")

    print("\n开始仿真...")
    total_reward = 0.0

    for step in range(env.max_episode_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 1 == 0:       # 每个RL步都输出
            pos_error = info['position_error']
            print(f"RL步: {step:4d}, pos_err: {pos_error:.3f}m, 累计奖励: {total_reward:.2f}, 当前位置: [{info['position'][0]:.2f}, {info['position'][1]:.2f}, {info['position'][2]:.2f}]")
            env.render()

        if terminated:
            print(f"\n在第{step}个RL步终止!")
            if info['position_error'] < 0.1:
                print("成功到达目标位置!")
            else:
                print("因其他条件终止（坠毁或极端状态）")
            break

        if truncated:
            print(f"\n在第{step}个RL步达到最大步数限制")
            break

    print("\n=== 仿真完成 ===")
    print(f"总RL步数: {step + 1}")
    print(f"累计奖励: {total_reward:.2f}")
    print(f"平均奖励/步: {total_reward/(step + 1):.4f}")
    print(f"最终位置: {info['position']}")
    print(f"目标位置: {env.target_pos}")
    print(f"最终位置误差: {info['position_error']:.3f}m")
    print(f"最终速度: {info['velocity']}")
    print(f"最终姿态(欧拉角): {info['attitude']}")
    env.render()
    env.close()
    print("\n仿真环境已关闭")



# 调参建议：

# 1.固定 dt=0.005, rl_dt=0.1, segment_window=2.0–3.0s，先看 dense 奖励是否稳定上升；
# 2.打开 segment 奖励，观察 rmse_pos_xy/z 与 rms_du 是否下降；
# 3.如果轨迹跟不上，提高 W_pos_xy/z 和 W_vel_xy/z；
# 4.如果抖动/能量大，提高 W_u, W_du, W_domega；
# 5.若电机常饱和，提高 W_sat，同时在控制器输出加硬/软限幅。