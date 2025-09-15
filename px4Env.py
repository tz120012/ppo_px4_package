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
import csv
import os
from collections import deque

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
    # metadata = {"render_modes": ["human", "rgb_array", "data_log"]}

    def __init__(self,
                 control_type: str = "xyz_pos",
                 trajectory_type: int = 5,
                 yaw_type: int = 3,
                 max_episode_steps: int = 2000,     # 每个回合的最大步数（以RL步计）约20s
                 dt: float = 0.005,         # 仿真积分时间步长（动力学/控制）
                 rl_dt: float = 1,          # RL决策时间步长（每次step保持action的时长）
                 segment_window: float = 3.0,  # 滑动窗口长度（秒）
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
            segment_window: 滑动窗口奖励时长（秒），默认 3s
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

        # 奖励权重（即时）
        self.reward_weights = dict(
            w_pos_xy=0.8,
            w_pos_z=1.2,
            w_vel_xy=0.10,
            w_vel_z=0.12,
            w_att=0.15,
            w_omega=0.05,
        )

        # Segment 权重（窗口奖励）
        self.segment_weights = dict(
            W_pos_xy=4.0,
            W_pos_z=6.0,
            W_vel_xy=1.0,
            W_vel_z=1.2,
            W_att_roll=1.2,
            W_att_pitch=1.2,
            W_omega=0.6,

            # 暂时去掉平滑惩罚（可按需开启）
            W_u=0.0,
            W_dv=0.0,
            W_domega=0.0,
            W_du=0.0,

            W_sat=3.0,
            W_over=1.5
        )

        # 限制与判据
        self.limits = dict(
            w_cmd_min=0.0,
            w_cmd_max=2e4,
            overshoot_z_tol=0.2,
            normalize_rate_by_dt=True  # 平滑性指标按dt归一
        )

        # 成功/失败的附加奖励
        self.success_bonus = 500.0      # 任务完成时一次性正奖励
        self.failure_penalty = -1500.0  # 非正常截断的惩罚

        # 回合跟踪（以RL步为单位）
        self.current_step = 0
        self.current_substep = 0   # 以仿真子步为单位
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

        # 分段统计缓冲初始化（滑动窗口）
        self._reset_segment_window()

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

        # 计算每个 RL step 的子步数，并按“子步总数”预分配
        self.substeps_per_rl = max(1, int(round(self.rl_dt / self.dt)))
        T = self.max_episode_steps * self.substeps_per_rl

        self.episode_t_all          = np.zeros((T,), dtype=float)
        self.episode_s_all          = np.zeros((T, len(self.quad.state)))
        self.episode_pos_all        = np.zeros((T, len(self.quad.pos)))
        self.episode_vel_all        = np.zeros((T, len(self.quad.vel)))
        self.episode_quat_all       = np.zeros((T, len(self.quad.quat)))
        self.episode_omega_all      = np.zeros((T, len(self.quad.omega)))
        self.episode_euler_all      = np.zeros((T, len(self.quad.euler)))
        self.episode_sDes_traj_all  = np.zeros((T, len(self.traj.sDes)))
        self.episode_sDes_calc_all  = np.zeros((T, len(self.ctrl.sDesCalc)))
        self.episode_w_cmd_all      = np.zeros((T, len(self.ctrl.w_cmd)))
        self.episode_wMotor_all     = np.zeros((T, len(self.quad.wMotor)))
        self.episode_thr_all        = np.zeros((T, len(self.quad.thr)))
        self.episode_tor_all        = np.zeros((T, len(self.quad.tor)))

        # 课程学习阶段
        self.stage = 0          # 课程学习的阶段

        # 子步越界保护标记
        self._force_truncate = False

    def _reset_segment_window(self):
        """初始化滑动窗口缓冲与差分缓存"""
        self.window_max_samples = max(1, int(round(self.segment_window / self.dt)))
        self.buf = deque(maxlen=self.window_max_samples)

        # 差分缓存
        self._seg_prev_vel = None
        self._seg_prev_omega = None
        self._seg_prev_w_cmd = None

        # 最近一次窗口指标
        self.last_segment_metrics = None

    # 定义记录substep的历史数据的函数
    def record_episode_step(self):
        i = self.current_substep
        if i < 0 or i >= self.episode_t_all.shape[0]:
            # 超界则不写，并打标记
            self._force_truncate = True
            return

        self.episode_t_all[self.current_substep]           = self.time
        self.episode_s_all[self.current_substep,:]         = self.quad.state
        self.episode_pos_all[self.current_substep,:]       = self.quad.pos
        self.episode_vel_all[self.current_substep,:]       = self.quad.vel
        self.episode_quat_all[self.current_substep,:]      = self.quad.quat
        self.episode_omega_all[self.current_substep,:]     = self.quad.omega
        self.episode_euler_all[self.current_substep,:]     = self.quad.euler
        self.episode_sDes_traj_all[self.current_substep,:] = self.traj.sDes
        self.episode_sDes_calc_all[self.current_substep,:] = self.ctrl.sDesCalc
        self.episode_w_cmd_all[self.current_substep,:]     = self.ctrl.w_cmd
        self.episode_wMotor_all[self.current_substep,:]    = self.quad.wMotor
        self.episode_thr_all[self.current_substep,:]       = self.quad.thr
        self.episode_tor_all[self.current_substep,:]       = self.quad.tor

    # =========================
    # Gym API
    # =========================
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.current_substep = 0
        self.episode_reward = 0.0

        if options is not None and 'target_pos' in options:
            self.target_pos = np.array(options['target_pos'])

        self._initialize_simulation()
        self._reset_segment_window()

        if options and options.get('random_init', False):
            noise = np.random.normal(0, 0.1, 3)
            self.quad.pos += noise
            self.quad.state[0:3] = self.quad.pos

        self.step_action = None
        self.step_reward = None

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一个环境步骤：保持该action在rl_dt时长内，包含多个仿真子步"""
        self.step_action = action
        action = np.clip(action, 0.0, 1.0)          # Action参数剪切处理
        self._apply_pid_parameters(action)          # 将参数写入PID控制器

        # 需要运行的仿真子步数量
        substeps = max(1, int(round(self.rl_dt / self.dt)))     # 计算每个Step 需要多少个仿真子步
        step_reward_dense_sum = 0.0                             # 聚合该RL step内的即时奖励（dense）

        terminated = False
        truncated = False
        success_just_now = False

        # 在每个仿真步（步长：0.005s，200Hz）
        for _ in range(substeps):
            # 目标/控制/动力学推进
            sDes = self.traj.desiredState(self.time, self.dt, self.quad)        # 期望轨迹
            self.ctrl.controller(self.traj, self.quad, sDes, self.dt)           # 控制器更新
            self.quad.update(self.time, self.dt, self.ctrl.w_cmd, self.wind)    # 动力学更新

            # 记录substep历史数据
            self.record_episode_step()

            # 时间推进与采样
            self.time += self.dt
            self.current_substep += 1

            # 计算即时状态误差
            m = self._extract_metrics(sDes)

            # 即时奖励（dense，细粒度，引导稳定）
            step_reward_dense_sum += self._dense_reward_instant(m)

            # 滑动窗口采样（每个仿真子步采样一次）
            self._push_window_sample(m)

            # 过程内快速越界检查（如希望提前退出，可在此判定）
            if self._is_out_of_bounds():
                truncated = True
                break

        # RL步计数
        self.current_step += 1

        # 滑动窗口分段奖励：每个 step 计算一次（覆盖最近 ~3 秒）
        segment_reward, metrics = self._compute_segment_reward_from_window()
        self.last_segment_metrics = metrics if metrics else self.last_segment_metrics

        # step奖励 = 平均即时dense + 窗口分段奖励
        reward = (step_reward_dense_sum / max(1, min(substeps, self.current_substep))) + segment_reward

        # 终止/截断判定（step 粒度收尾）
        # 1) 达到最大步数 -> terminated
        if self.current_step >= self.max_episode_steps or self._force_truncate:
            terminated = True

        # 2) 非法姿态/位置越界 -> truncated
        if not terminated and self._is_out_of_bounds():
            truncated = True

        # 成功奖励或失败惩罚
        if truncated and not terminated:
            reward += self.failure_penalty

        self.step_reward = reward
        self.episode_reward += reward

        # 结束时记录可用 episode 长度
        if terminated or truncated:
            max_substeps_total = self.episode_t_all.shape[0]
            self._episode_len = min(self.current_substep, max_substeps_total)

        obs = self._get_observation()
        info = self._get_info()
        # 在info中包含本次分段奖励与指标
        if metrics is not None:
            info['last_segment'] = metrics

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
        eul_fb = self.quad.euler
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

    # 统一的指标/误差提取（供奖励与窗口累计复用）
    def _extract_metrics(self, sDes: np.ndarray) -> Dict[str, Any]:
        """
        从 sDes 及当前状态提取误差指标。
        返回:
            - pos_err_xy, pos_err_z
            - vel_err_xy, vel_err_z
            - att_norm, att_err_roll, att_err_pitch, att_err_yaw
            - omega_norm, omega_err_p/q/r
            - pos_err_vec, vel_err_vec
        """
        # 参考状态
        pos_ref = sDes[0:3]
        vel_ref = sDes[3:6]
        eul_ref = sDes[12:15]
        omega_ref = sDes[15:18]

        # 当前状态
        pos = self.quad.pos
        vel = self.quad.vel
        euler = self.quad.euler
        omega = self.quad.omega

        # wrap 欧拉角误差到 (-pi, pi]
        def angle_wrap(a):
            return (a + np.pi) % (2 * np.pi) - np.pi

        pos_err_vec = pos_ref - pos
        pos_err_xy = float(np.linalg.norm(pos_err_vec[:2]))
        pos_err_z = float(np.abs(pos_err_vec[2]))

        vel_err_vec = vel_ref - vel
        vel_err_xy = float(np.linalg.norm(vel_err_vec[:2]))
        vel_err_z = float(np.abs(vel_err_vec[2]))

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
            "pos_err_xy": pos_err_xy,
            "pos_err_z": pos_err_z,
            "vel_err_xy": vel_err_xy,
            "vel_err_z": vel_err_z,
            "att_norm": att_norm,
            "att_err_roll": abs(att_err_roll),
            "att_err_pitch": abs(att_err_pitch),
            "att_err_yaw": att_err_yaw,
            "omega_norm": omega_norm,
            "omega_err_p": omega_err_p,
            "omega_err_q": omega_err_q,
            "omega_err_r": omega_err_r,
            "pos_err_vec": pos_err_vec,
            "vel_err_vec": vel_err_vec,
        }

    def _dense_reward_instant(self, m) -> float:
        """
        即时（密集）奖励：每个仿真子步计算。
        - 分轴位置误差（XY、Z分权）
        - 速度跟踪误差（相对 v_ref）
        - 姿态误差（phi、theta 分开）
        - 角速度范数
        """
        rw = self.reward_weights
        dense = -(
            rw["w_pos_xy"] * m["pos_err_xy"] +
            rw["w_pos_z"]  * m["pos_err_z"]  +
            rw["w_vel_xy"] * m["vel_err_xy"] +
            rw["w_vel_z"]  * m["vel_err_z"]  +
            rw["w_att"]    * m["att_norm"]   +
            rw["w_omega"]  * m["omega_norm"]
        )
        return float(dense)

    def _push_window_sample(self, m: Dict[str, Any]):
        """将当前子步的原子指标压入滑动窗口缓冲"""
        # 控制能量
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            control_effort = float(np.sum(np.square(self.ctrl.w_cmd)))
        else:
            control_effort = 0.0

        # 平滑性（时间变化率）
        dv = 0.0
        domega = 0.0
        du = 0.0
        if self._seg_prev_vel is not None:
            dv = float(np.linalg.norm(self.quad.vel - self._seg_prev_vel))
        if self._seg_prev_omega is not None:
            domega = float(np.linalg.norm(self.quad.omega - self._seg_prev_omega))
        if hasattr(self.ctrl, "w_cmd"):
            if self._seg_prev_w_cmd is None and self.ctrl.w_cmd is not None:
                self._seg_prev_w_cmd = np.copy(self.ctrl.w_cmd)
            if self.ctrl.w_cmd is not None and self._seg_prev_w_cmd is not None:
                du = float(np.linalg.norm(self.ctrl.w_cmd - self._seg_prev_w_cmd))
                self._seg_prev_w_cmd = np.copy(self.ctrl.w_cmd)

        if self.limits.get("normalize_rate_by_dt", True) and self.dt > 0:
            inv_dt = 1.0 / self.dt
            dv *= inv_dt
            domega *= inv_dt
            du *= inv_dt

        self._seg_prev_vel = np.copy(self.quad.vel)
        self._seg_prev_omega = np.copy(self.quad.omega)

        # 饱和与简单超调判据
        saturation_hit = 0
        if hasattr(self.ctrl, 'w_cmd') and self.ctrl.w_cmd is not None:
            wmin, wmax = self.limits["w_cmd_min"], self.limits["w_cmd_max"]
            if np.any(self.ctrl.w_cmd < wmin) or np.any(self.ctrl.w_cmd > wmax):
                saturation_hit = 1

        z_err = float(self.quad.pos[2] - self.target_pos[2])
        vz = float(self.quad.vel[2])
        overshoot_tol = self.limits["overshoot_z_tol"]
        overshoot_hit = 1 if (z_err * np.sign(vz) > 0 and abs(z_err) > overshoot_tol) else 0

        entry = {
            "pos_xy2": m["pos_err_xy"]**2,
            "pos_z2": m["pos_err_z"]**2,
            "vel_xy2": m["vel_err_xy"]**2,
            "vel_z2": m["vel_err_z"]**2,
            "att_roll2": m["att_err_roll"]**2,
            "att_pitch2": m["att_err_pitch"]**2,
            "omega2": m["omega_norm"]**2,
            "u2": control_effort,           # 与原实现保持一致：w_cmd 的平方和
            "dv2": dv**2,
            "domega2": domega**2,
            "du2": du**2,
            "sat_hit": saturation_hit,
            "over_hit": overshoot_hit,
            "t": float(self.time),          #  记录采样时间戳
        }
        self.buf.append(entry)

    def _compute_segment_reward_from_window(self) -> Tuple[float, Dict[str, float]]:
        """基于滑动窗口缓冲计算分段奖励及指标"""
        if len(self.buf) == 0:
            return 0.0, {}

        now = self.time

        # 分段常数权重
        def weight_for(dt: float) -> float:
            # dt >= 0
            if dt <= 1.0:
                return 1.0
            elif dt <= 2.0:
                return 0.5
            elif dt <= 3.0:
                return 0.25
            else:
                return 0.0  # 理论上不会出现，因为窗口最大 3s

        # 带权聚合
        s = dict(pos_xy2=0.0, pos_z2=0.0, vel_xy2=0.0, vel_z2=0.0,
                att_roll2=0.0, att_pitch2=0.0, omega2=0.0,
                u2=0.0, dv2=0.0, domega2=0.0, du2=0.0,
                sat_hits=0.0, over_hits=0.0)
        W = 0.0

        for e in self.buf:
            w = float(weight_for(now - e["t"]))
            if w <= 0.0:
                continue
            W += w
            s["pos_xy2"]  += w * e["pos_xy2"]
            s["pos_z2"]   += w * e["pos_z2"]
            s["vel_xy2"]  += w * e["vel_xy2"]
            s["vel_z2"]   += w * e["vel_z2"]
            s["att_roll2"]  += w * e["att_roll2"]
            s["att_pitch2"] += w * e["att_pitch2"]
            s["omega2"]   += w * e["omega2"]
            s["u2"]       += w * e["u2"]
            s["dv2"]      += w * e["dv2"]
            s["domega2"]  += w * e["domega2"]
            s["du2"]      += w * e["du2"]
            s["sat_hits"] += w * e["sat_hit"]
            s["over_hits"] += w * e["over_hit"]

        if W <= 1e-9:
            return 0.0, {}

        # 用总权重 W 做加权均值
        rmse_pos_xy   = (s["pos_xy2"]  / W) ** 0.5
        rmse_pos_z    = (s["pos_z2"]   / W) ** 0.5
        rmse_vel_xy   = (s["vel_xy2"]  / W) ** 0.5
        rmse_vel_z    = (s["vel_z2"]   / W) ** 0.5
        rmse_att_roll = (s["att_roll2"]/ W) ** 0.5
        rmse_att_pitch= (s["att_pitch2"]/W) ** 0.5
        rms_omega     = (s["omega2"]   / W) ** 0.5

        mean_control_effort = s["u2"] / W
        rms_dv      = (s["dv2"]      / W) ** 0.5
        rms_domega  = (s["domega2"]  / W) ** 0.5
        rms_du      = (s["du2"]      / W) ** 0.5

        rmse_pos_total = ((s["pos_xy2"] + s["pos_z2"]) / W) ** 0.5
        rmse_vel_total = ((s["vel_xy2"] + s["vel_z2"]) / W) ** 0.5

        sw = self.segment_weights
        segment_reward = -(
            sw["W_pos_xy"] * rmse_pos_xy +
            sw["W_pos_z"]  * rmse_pos_z  +
            sw["W_vel_xy"] * rmse_vel_xy +
            sw["W_vel_z"]  * rmse_vel_z  +
            sw["W_att_roll"]  * rmse_att_roll +
            sw["W_att_pitch"] * rmse_att_pitch +
            sw["W_omega"]  * rms_omega +
            sw["W_u"]      * mean_control_effort +
            sw["W_dv"]     * rms_dv +
            sw["W_domega"] * rms_domega +
            sw["W_du"]     * rms_du +
            sw["W_sat"]    * s["sat_hits"] +     # 注意：这里是加权计数
            sw["W_over"]   * s["over_hits"]      # 同上
        )

        metrics = {
            'segment_duration': float(min(3.0, len(self.buf) * self.dt)),
            'rmse_pos_xy': float(rmse_pos_xy),
            'rmse_pos_z': float(rmse_pos_z),
            'rmse_vel_xy': float(rmse_vel_xy),
            'rmse_vel_z': float(rmse_vel_z),
            'rmse_att_roll': float(rmse_att_roll),
            'rmse_att_pitch': float(rmse_att_pitch),
            'rms_omega': float(rms_omega),
            'rmse_pos_total': float(rmse_pos_total),
            'rmse_vel_total': float(rmse_vel_total),
            'mean_control_effort': float(mean_control_effort),
            'rms_dv': float(rms_dv),
            'rms_domega': float(rms_domega),
            'rms_du': float(rms_du),
            'saturation_hits': float(s["sat_hits"]),  # 现在是加权“命中强度”
            'overshoot_hits': float(s["over_hits"]),
            'segment_reward': float(segment_reward),
            'samples': int(len(self.buf)),
            'weight_sum': float(W),
        }

        return float(segment_reward), metrics
    # =========================
    # 终止与信息
    # =========================
    def _is_out_of_bounds(self) -> bool:
        """位置/姿态越界判定"""
        if self.quad.pos[2] > 10.0 or self.quad.pos[2] < -10.0:
            return True
        if np.abs(self.quad.pos[0]) > 20.0 or np.abs(self.quad.pos[1]) > 20.0:
            return True
        if np.abs(self.quad.euler[0]) > np.pi/2 or np.abs(self.quad.euler[1]) > np.pi/2:
            return True
        return False

    def _get_info(self) -> Dict:
        pos_error = float(np.linalg.norm(self.quad.pos - self.target_pos))
        info = {
            'position': self.quad.pos.copy(),
            'velocity': self.quad.vel.copy(),
            'attitude': self.quad.euler.copy(),
            'angularvelocity': self.quad.omega.copy(),
            'pos_ref': self.traj.desPos.copy(),
            'vel_ref': self.traj.desVel.copy(),
            'att_ref': self.traj.desEul.copy(),
            'angularvel_ref': self.traj.desPQR.copy(),
            'episode_reward': self.episode_reward,
            'time': self.time,
            'segment_time_accum': float(len(self.buf) * self.dt),
            'position_error': pos_error,
        }
        if self.last_segment_metrics is not None:
            info['last_segment'] = self.last_segment_metrics
        return info

    # =========================
    # 课程学习相关
    # =========================
    def set_course_stage(self, stage:int):
        """根据课程阶段设置新的控制类型和轨迹类型"""
        if stage == 0:
            self.control_type = "xyz_pos"
            self.trajectory_type = 0
        elif stage == 1:
            self.control_type = "xyz_pos"
            self.trajectory_type = 1
        elif stage == 2:
            self.control_type = "xyz_pos"
            self.trajectory_type = 2
        elif stage == 3:
            self.control_type = "xyz_pos"
            self.trajectory_type = 3
        elif stage == 4:
            self.control_type = "xyz_pos"
            self.trajectory_type = 4
        else:
            raise ValueError(f"Invalid course stage: {stage}")

        self.stage = int(stage)

        logging.info(f"Course stage set to {self.stage}: control_type={self.control_type}, trajectory_type={self.trajectory_type}")

        # 应用到轨迹/控制器（不重置时间）
        traj_select = np.array([self.trajectory_type, self.yaw_type, 1])
        self.traj = Trajectory(self.quad, ctrlType=self.control_type, trajSelect=traj_select)
        self.ctrl = Control(self.quad, self.traj.yawType)
        sDes = self.traj.desiredState(self.time, self.dt, self.quad)
        self.ctrl.controller(self.traj, self.quad, sDes, self.dt)

        # 重置滑动窗口，避免阶段切换时历史统计干扰
        self._reset_segment_window()

    # =========================
    # 渲染与关闭
    # =========================
    def render(self, mode: str = 'data_log', save_root: str = "log_data", csv_filename: str = "episode_log.csv"):
        """
        mode:
            - 'human': 绘图 + 动画（plt.show）
            - 'data_log': 保存为“列式 CSV”，第一行为列名；第一列为 t_all
        save_root: 保存目录
        csv_filename: CSV 文件名
        """
        import os
        import csv
        import numpy as np
        from datetime import datetime
        import matplotlib.pyplot as plt

        try:
            if mode == 'human':
                # 绘制静态图
                utils.makeFigures(
                    self.quad.params, self.episode_t_all, self.episode_pos_all, self.episode_vel_all,
                    self.episode_quat_all, self.episode_omega_all, self.episode_euler_all,
                    self.episode_w_cmd_all, self.episode_wMotor_all, self.episode_thr_all,
                    self.episode_tor_all, self.episode_sDes_traj_all, self.episode_sDes_calc_all
                )

                # 动画（注意 ifsave 变量）
                ifsave = False
                ani = utils.sameAxisAnimation(
                    self.episode_t_all, self.traj.wps, self.episode_pos_all,
                    self.episode_quat_all, self.episode_sDes_traj_all, self.dt,
                    self.quad.params, self.traj.xyzType, self.traj.yawType, ifsave
                )
                plt.show()

            elif mode == 'data_log':
                os.makedirs(save_root, exist_ok=True)
                # 建立一个带时间戳的子目录，避免覆盖
                from datetime import datetime
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                ep_dir = os.path.join(save_root, f"episode_{stamp}")
                os.makedirs(ep_dir, exist_ok=True)
                csv_path = os.path.join(ep_dir, csv_filename)

                # 1) 准备时间列
                def to_np(x):
                    return x if isinstance(x, np.ndarray) else np.array(x)

                t_all = to_np(self.episode_t_all).reshape(-1)  # shape (T,)
                T = t_all.shape[0]

                # 2) 需要写入的数组列表（名称: 数组）
                arrays = {
                    "t_all": t_all,
                    "s_all": to_np(self.episode_s_all),
                    "pos_all": to_np(self.episode_pos_all),
                    "vel_all": to_np(self.episode_vel_all),
                    "quat_all": to_np(self.episode_quat_all),
                    "omega_all": to_np(self.episode_omega_all),
                    "euler_all": to_np(self.episode_euler_all),
                    "sDes_traj_all": to_np(self.episode_sDes_traj_all),
                    "sDes_calc_all": to_np(self.episode_sDes_calc_all),
                    "w_cmd_all": to_np(self.episode_w_cmd_all),
                    "wMotor_all": to_np(self.episode_wMotor_all),
                    "thr_all": to_np(self.episode_thr_all),
                    "tor_all": to_np(self.episode_tor_all),
                }

                # 3) 规范化每个数组到 (T, D) 形状，并生成列名
                columns = []          # 列名列表
                matrix_parts = []     # 对应的二维数据块 (T, D)

                for name, arr in arrays.items():
                    # 确保是 numpy
                    arr = to_np(arr)

                    # 自动对齐到时间长度 T
                    if name == "t_all":
                        arr_T = arr.reshape(T, 1)
                        columns.append("t_all")
                        matrix_parts.append(arr_T)
                        continue

                    if arr.ndim == 0:
                        arr = np.full((T, 1), float(arr))
                    elif arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    else:
                        first_dim = arr.shape[0]
                        rest = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
                        arr = arr.reshape(first_dim, rest)

                    N = arr.shape[0]
                    D = arr.shape[1]

                    if N < T:
                        pad = np.full((T - N, D), np.nan, dtype=float)
                        arr = np.vstack([arr, pad])
                    elif N > T:
                        arr = arr[:T, :]

                    if D == 1:
                        columns.append(name)
                    else:
                        columns.extend([f"{name}_{i}" for i in range(D)])

                    matrix_parts.append(arr)

                full_matrix = np.hstack(matrix_parts)  # shape (T, total_D)

                with open(csv_path, mode="w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    for i in range(T):
                        row = []
                        for j in range(full_matrix.shape[1]):
                            val = full_matrix[i, j]
                            row.append("" if (isinstance(val, float) and np.isnan(val)) else val)
                        writer.writerow(row)

                print(f"[render:data_log] Saved columnar CSV to: {csv_path}")

            else:
                print(f"[render] Unsupported mode: {mode}")

        except Exception as e:
            print(f"Render error: {e}")
            pass

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None